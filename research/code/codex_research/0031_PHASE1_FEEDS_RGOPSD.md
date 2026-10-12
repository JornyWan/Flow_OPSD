# 0031 Phase-1 结果对 RG-OPSD(0027)的输入:**AP 候选不加入 portfolio**

## codex 0027 §5 Phase 0 的原话

> 先让 `1036869` 给出真实 active rate/certificate;
> **该结果只决定是否把 `AP` 加入 portfolio**,不决定 RG-OPSD 是否继续。

Phase-1 已完成(job 1036899,8/8 probe,见 `0025` 终局节)。答案是:

## ⛔ `AP`(optimizer-delta projected candidate)**不加入**

| 证据 | 值 |
|---|---|
| QP 真正改动更新方向的次数 | **0 / 8** |
| 一阶预测 / 精确重算 的比值 | ocr 中位数 **0.117**、geneval **0.051** |
| exact certificate 通过 | 8/8,但**全部靠回溯缩步长**(α=0.25~0.5),不是靠投影 |

每个 probe 的机制都相同:线性模型下 `C_m + a_mᵀΔ_adam < 0`(判定"满足"),
精确重算却 `C_m(θ+Δ_adam) > 0`(实际违反)。
**一阶信息系统性乐观约 10 倍,投影因此是恒等映射。**

`AP` 若加入 portfolio,在这些 window 上会与 `AD` **逐位相同** ——
它不会增加 selector 的决策空间,只会增加一次无意义的候选生成开销。

⇒ **portfolio 冻结为 codex 的最小三臂 `A0 / AD / AG`。**

## 同时,Phase-1 反过来**支持** RG-OPSD 的核心论点

RG-OPSD 的主张是"把选择从 target/trajectory space **后移到真实 optimizer update 之后**"。
Phase-1 恰好是这个主张的一个受控证据:

- **更新前 + 一阶**(QP projection):0/8 有效 ——
  它只能看到一阶信息,而一阶在这里解释不到真实变化的 1/8;
- **更新后 + 精确**(exact certificate + 回溯):8/8 有效。

**同一份数据里,"更新后按精确评估决策"胜过"更新前按一阶预测决策"。**
这正是 RG-OPSD 要推广的机制 —— 从"选步长"推广到"选候选"。

## 可直接复用的代码

`code/0026_phase1_probe.patch` 已实现并验证了 Phase 1R 需要的大部分机制:

- 参数 + optimizer state 的快照/还原(`_p1_restore` + `deepcopy(state_dict)`);
- 从同一 `(θ, AdamW state)` 出发走**真实** AdamW 更新并取回 `Δ`;
- 在**同一缓存窗口**上精确重算约束(eval 趟);
- eval 趟也 backward 以释放计算图(否则 H200 140GiB 会被耗尽,已踩过)。

Phase 1R 相对它的增量是:**第二个候选(guarded loss)** 与 **选择规则**,
以及 reward 侧的 realized gain 度量。
codex §2.5 要求用真实 `J_pick`(需按相同 suffix seed 生成/解码),
若首版改用训练侧的 direct objective 作代理,**必须显式标注为偏离**,不得默认等同。

## 未启动

本条只做衔接与冻结 portfolio,**未提交 Phase 1R 任务** ——
当前队列里还有 12 个基线评测在跑(mopdc/crkl5 的六点 + StudentSI),
它们是用户明确要求的"公平真 baseline",优先级高于新方法的探针。
