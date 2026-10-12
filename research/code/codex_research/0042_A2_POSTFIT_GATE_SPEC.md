# 0042 A2 规格:官方 OPSD 底座上的 post-fit exact commit gate(2026-08-30)

**状态:实现规格。未写代码,未提交任务。等 baseline 确认完成(0041 §⑤ 口径)后再动手。**

## ① 为什么 A2 不等 Gate A

0040 §④ 的判据表里,**VFR ≥ 0.90 只杀 A1,不杀 A2**。
A1 管的是"目标构造阶段是否做了有害权衡";A2 管的是"**拟合之后**是否还成立" ——
这是两个独立缺口,Phase-1 的实测把它们分得很清楚:

| Phase-1 实测 | 含义 |
|---|---|
| QP 一阶投影 8/8 探针中**改变方向 0/8**,是恒等映射 | ⇒ A2 **不能**做一阶投影 |
| 线性模型说约束满足,精确重算却显示违反;一阶信息约 **10× 过于乐观** | ⇒ 必须**精确**重算,不能用代理 |
| **同状态精确重算 + 回溯 8/8 成功**(α=0.25~0.5) | ⇒ A2 的形式由实测**反向确定** |
| 实际 AdamW 更新与累积原始梯度 cos 仅 **0.339** | ⇒ 参数级实现与目标级意图差距大,必须在参数更新**之后**验收 |

**A2 不是又一个待调超参,它的形式是被数据逼出来的。**

## ② 挂载点(pristine 行号,commit 9c4c48c)

```
L1349-1354  collated_samples["opa_z_q" / "opa_y_plus" / "opa_y_minus" / "opa_sig_q"]
            ↑ 同状态重算所需的一切**已经缓存**,不必新增采样
L1576       scaled_loss = loss / opa_accum   ... backward
L1589       if current_accumulated_steps % opa_accum == 0:
L1594           scaler.step(optimizer); scaler.update()      ← A2 包住这一步
L1598           optimizer.zero_grad()
L980 注释    "1 optimizer step == 1 epoch (gradient_step_per_epoch=1)"
```

**每 epoch 只有 1 次优化器步** ⇒ A2 的验收开销是**每 epoch 1 次评估**,不是每个 micro-batch。

## ③ 算法

```text
每次优化器步(每 epoch 一次):
  1. 取参考分数 R_prev  —— 用**当前** adapter 在缓存 state 上的三路 reward。
     (可复用上一次 commit 后的验收结果 ⇒ 摊销后每 epoch 仅 1 次评估)
  2. 快照 θ 与 optimizer state          # Phase-1 的 0025 已验证过该机制
  3. 执行真实 AdamW step -> θ_step,记 Δ = θ_step − θ_prev
  4. 同状态精确重算:
         y_new = z_q − σ_q · v_θ(z_q, σ_q)        # 用 default adapter,非 behavior
         R_new = 三路 scorer(decode(y_new))
  5. 验收(向量,不做标量加权):
         对每个 k:  mean_k(R_new) ≥ mean_k(R_prev) − τ_k
     全部满足 -> commit,把 R_new 存为下一轮的 R_prev
     否则     -> 回溯
  6. 回溯:θ ← θ_prev + α·Δ,α ∈ {0.5, 0.25, 0.125},逐个重试第 4~5 步
     全部失败 -> **no-op**(θ ← θ_prev),记 no-op 计数
```

### 两个必须写死的实现决定

1. **回溯缩放的是参数增量 Δ,不是重跑 Adam。**
   `θ ← θ_prev + α·Δ` 是精确且便宜的;重跑 Adam 需要重新 backward。
   ⚠️ 这里有一个 Phase-1 踩过的坑的**镜像**:当时的教训是
   "回溯必须对 `α·Δ_adam` 重解 QP,而不是把整个解按比例缩放"。
   A2 **没有 QP**,所以缩放 Δ 是正确的;但**验收必须对缩放后的 θ 重新精确重算**,
   不能假设"α 小一半,违反量就小一半"—— 那正是一阶思维,已被 8/8 否定。
2. **optimizer state 的处理:保留全步的 Adam 状态(m, v),只回退参数。**
   理由:m/v 反映的是**实际看到的梯度**,与是否 commit 无关;
   若连 m/v 一起回退,下一步会重复同一方向的动量累积。
   **⚠️ 这是一个未经实测的选择**,须在首次运行时同时记录两种做法的 no-op 率作对照,
   不能默认它对。

## ④ 参数与预注册判据

| 项 | 取值 | 依据 |
|---|---|---|
| τ_k(每轴容差) | 各轴重测 σ 的 **1×**(0034 §⑦ 实测:pick 0.00075 / clip 0.00108 / hps 0.00043) | 容差取重测噪声量级,不是拍脑袋 |
| α 序列 | 0.5, 0.25, 0.125 | Phase-1 实测 8/8 落在 0.25~0.5 |
| 失败上限 | 3 次回溯后 no-op | 同上 |

**先于数据写定的 kill 条件:**
- **no-op 率 ≥ 70%** ⇒ A2 实质上等于"不训练",判死。
  不得通过放宽 τ_k 让它复活 —— 放宽容差就是取消约束。
- **no-op 率 ≤ 5% 且各轴末值与 baseline 无显著差异** ⇒ A2 是空操作,同样判死。
- 有意义的区间是中间:**commit 率高但确实拦下了一部分更新**,且末值优于 baseline。

## ⑤ 开销

每 epoch:1 次 forward(缓存 state)+ 1 次 decode + 3 次 scorer,无 backward;
回溯每次重试再来一遍。按 Phase-1 的 α 分布估计,平均 **1.3~1.7 次/epoch**。
相对每 epoch 1152 样本的 rollout + 训练,**开销可忽略**。

## ⑥ 与 baseline 的可比性

A2 臂与 baseline 臂必须满足 0041 §⑤ 的冻结口径,**且同一固定种子**(0041 §⑧)。
`opsd_manifest_diff.py --allow` 只允许声明与 A2 相关的字段;
任何其他差异出现即判该次 A/B 无效。
