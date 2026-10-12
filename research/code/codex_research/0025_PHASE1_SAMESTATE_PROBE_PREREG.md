# 0025 Phase-1 预注册:same-state / optimizer-aware / finite-step certificate probe

**写在实现之前**(codex 0023 §8 要求)。本文件冻结 probe 口径、样本量与 go/no-go,
实现与运行必须照此执行;若实现中发现口径需改,**新开一份预注册**并说明原因,不改本文件的判据。

## 0. 为什么要重做这个闸(我上一次关闭 QP 的推理哪里错了)

我在 `0016` 用 0019/0020 探针得到"一阶模型只解释实际约束变化的 8~10%",
据此按预注册判据**关闭了 QP 路线**。codex 0023 §3.3 指出这个推理**越界**了,我核实后接受:

那次比较 `<a, u>` 与"下一 epoch 的 violation 变化"时,**同时**变了四样东西:
1. prompt / timestep / 噪声 / on-policy latent(重采样);
2. 当前 student state 与 frozen-reference threshold;
3. 梯度估计的 batch(`a` 主要来自**第一个** microbatch,而 `u` 是 **12 个** microbatch 累积后的更新);
4. AdamW 的参数更新映射。

所以"符号约 50%、幅度只解释 8~10%"**同样可以**由重采样与状态漂移解释,
**不能**归因于 Taylor 近似本身失效。
它证明的是"**当前这个 raw-gradient probe 不可用**",不是"optimizer-aware projection 不可用"。

`cos(实际 AdamW 更新, −累积梯度) = 0.339` 这个数**仍然成立且有用** ——
它说明在**原始梯度空间**做投影是错的,同时指出了正确改法:
在**真实 optimizer proposal 的参数增量空间**投影,并做 exact same-state certificate。

## 1. probe 的精确口径(每次 probe 的六个步骤)

在一个**冻结的 accumulation window** 上做,全程不重采样:

1. **缓存**整个 window 的 prompt、timestep、噪声、latent、frozen-reference 输出。
   window 大小 = 训练配置的 `nbpe`(=12),即 72 个样本 —— **与训练一致**。
2. 在旧参数 θ 上计算每个受保护能力的 `C_m(θ)` 与约束梯度 `a_m`。
   **`a_m` 必须在同一个完整 window 上累积**,不能只取第一个 microbatch(这是上次的错)。
3. **复制参数与 optimizer state**,只用 reward objective 走一次真实的 AdamW 更新,
   得到 `Δ_adam = θ_candidate − θ`。**不是 −g,也不是 −ηg。**
4. 在**同一批缓存状态**上精确重算 `C_after_direct`,与一阶预测 `a_mᵀΔ_adam` 比较。
   —— 这一步才是对 Taylor 近似的**干净**检验:状态没变、batch 没变、用的是真实更新。
5. 解带 repair RHS 的 update-space 投影(求解器 `code/0025_qp_repair_solver.py`,
   已过 300 组合成对拍含 100 个 repair 算例):

       min_Δ ½‖Δ − Δ_adam‖²   s.t.  C_m(θ) + a_mᵀΔ ≤ 0

   得 `Δ_proj`,在同一缓存状态上**精确重算** `C_after_proj`(不是用一阶预测)。
6. 若 full step 不可行,按 `{1, 1/2, 1/4, 1/8}` backtrack,记录:
   - 每个约束是否真的满足(exact certificate,非预测);
   - `reward_retention = <Δ, Δ_adam>/‖Δ_adam‖²`;
   - direct loss 的**真实**变化;
   - 用了几次 backtrack。

## 2. 样本量与"active probe"的定义

- **probe 数 = 24**(在 codex 给的 16~32 区间内),覆盖训练早/中/晚三段各 8 个,
  取自已完成的 `crd`(w=0)轨迹的 checkpoint,**不新起训练**。
- **active probe** := 该 probe 的 direct proposal 使**至少一个**约束在精确重算下被违反
  (`C_after_direct > 0`)。非 active 的 probe 不计入通过率分母。

## 3. 先于数据冻结的 go/no-go

| 观测 | 决定 |
|---|---|
| **active probe 占比 < 20%** | direct proposal 基本不破坏约束 ⇒ **投影没有要解决的问题,关闭 QP 路线** |
| **≥90% 的 active probe 在 ≤3 次 backtrack 内通过 exact certificate,且 reward_retention 中位数 ≥ 0.5** | **进入 Phase-2 短训练**(20~30 步三臂) |
| 其余情形 | **永久关闭 QP**,把 fixed hinge 作为最终结果,不再追加几何探针 |

**附带记录(不参与判定,但必须报告)**:第 4 步得到的
"一阶预测 vs 精确重算"的符号一致率与幅度比 —— 这是对 Taylor 近似的干净检验,
无论 go/no-go 结论如何都应写进论文的方法讨论。

## 4. 这个 probe 不回答什么

- 不回答"投影能否在完整训练中胜过 fixed hinge" —— 那是 Phase-2。
- 不回答 `G_fit` 的归因(`0021` 的开放问题) —— 那需要有效批量剂量-反应。
- 不产出任何可写进主表的模型分数。

## 5. 实现约束

- **不改任何守卫**;新代码走新 patch 号,不覆盖 0019/0020。
- 复制 optimizer state 必须用 `copy.deepcopy(optimizer.state_dict())` 并在 probe 后还原,
  probe **不得污染**真实训练轨迹(本 probe 在已完成的 checkpoint 上离线跑,不在训练中跑)。
- 每个 probe 的缓存状态要落盘或固定随机种子,使结果**可复现**。

---

## 附录 A(实现期修正):backtrack 规则有数学缺陷,已改;**go/no-go 判据不变**

移植进 trainer 之前先跑了机制合成验证(`code/0025_probe_machinery_test.py`),
发现 §1 第 6 步"按 `{1,1/2,1/4,1/8}` backtrack"在 **repair 情形下注定失败**:

QP 解满足 `C_m + a_mᵀΔ_proj = 0`。整体缩放 α 后

    C_m + a_mᵀ(α·Δ_proj) = C_m + α·(−C_m) = C_m·(1 − α)

当 `C_m > 0`(当前已违反、需要修复)时,**任何 α < 1 都严格 > 0**。
即"先解 QP 再整体缩小步长"会把 repair 分量一并削弱,线性模型下**不可能**通过 certificate。
合成验证实测:α=1 可行,α=0.5/0.25/0.125 的约束值分别回到 0.050/0.075/0.088(全部违反)。

**修正**:backtrack 只缩放 **reward 分量**,repair 分量保持全量 ——
即对 `α·Δ_adam` **重新求解** QP:

    min_Δ ½‖Δ − α·Δ_adam‖²   s.t.  C_m + a_mᵀΔ ≤ 0

实测该规则在 α = 1 / 0.5 / 0.25 / 0.125 下约束值恒为 0(全部可行)。

**判据不变**:仍是"≥90% 的 active probe 在 ≤3 次 backtrack 内通过 exact certificate,
且 reward_retention 中位数 ≥ 0.5"。改的只是 backtrack 的**执行方式**,不是通过门槛。
`reward_retention` 仍按 `<Δ, Δ_adam>/‖Δ_adam‖²` 计(注意分母始终是**全量** `Δ_adam`,
这样 α<1 会如实体现为 retention 下降,不会因为换了基准而虚高)。

## 附录 B:合成验证顺带给出的一个旁证(不构成结论)

同一缓存状态下,一阶预测与精确重算的比值为 **0.868 / 0.904**(两个约束),
即 Taylor 近似解释了约 87~90% 的真实变化。
而我此前那个**重采样**版本只解释 8~10%。

这在**机制层面**印证了 codex §3.3:那 8~10% 主要来自重采样与状态漂移,不是 Taylor 失效。
**但这是合成 MLP,不能外推到扩散模型** —— 真实数值必须由 Phase-1 §1 第 4 步给出。
在此之前不得引用 0.87/0.90 这两个数。

---

## 附录 C(实现期修正之二):`C_m` 的定义必须与被测方法一致,否则 certificate 无意义

合成验证发现,即使 backtrack 按附录 A 修正,**`C_before = 0` 时仍然 4 次全败**。
查真因不是实现问题,是**判据本身不可满足**:

当参考点就是当前参数、`C_before = 0` 时,一阶条件 `a_mᵀΔ = 0` 恰好是**切方向**,
而二阶曲率使任何非零步都让 `C_after > 0`。
**严格 `C ≤ 0` 的 exact certificate 对任何方法都是 100% 失败的**,与方法好坏无关。

真实的 CR 方法本来就带容差带:`relu(cur_kl − (1+rel_tol)·ref_kl − abs_tol)`,
`rel_tol = 0.02` —— 实测 `cr_violation_ocr` 的首值是 **−9.59e−7**(负 = 有余量)。
probe 必须用**同一个 C 定义**,否则测的不是这个方法。

合成验证(warm=1,同一模型/同一 batch,只改 `rel_tol`):

| rel_tol | C_before | direct 后 active? | certificate |
|---|---|---|---|
| 0.0 | 0 | 是 | **全败**(对任何方法都不可能) |
| 0.002 | −1.9e−3 | 是 | 全败 |
| **0.02**(方法实际值) | −1.9e−2 | 是 | **通过 α=0.5,retention=0.500** |
| 0.05 | −4.8e−2 | **否** | 通过但无事可做 |

**两端都会让 probe 失去信息量**:tol 太小 ⇒ 谁都过不了;tol 太大 ⇒ direct 根本不违反,
落进 go/no-go 的第一支("投影没有要解决的问题")。

**冻结**:Phase-1 probe 的 `C_m` 使用 **`rel_tol = 0.02`、`abs_tol = 1e-8`**,
与 `codex_cr_prod` 逐位一致。这不是为了让 probe 好过 —— 是为了让它测的确实是被比较的那个方法。
**go/no-go 判据仍不变。**

**必须一并报告的诊断**(否则无法区分"方法不行"与"这一步根本做不到"):
1. `C_before` 的量级与"单步可达变化量"`|a_mᵀΔ_adam|` 的比值 ——
   比值 ≫1 说明累积违反已超出单步可修复范围,失败不能归因于投影方法;
2. `active probe` 占比(第一支 go/no-go 直接用它);
3. 每次 backtrack 的 α 与对应的 exact `C_after`。

**合成验证的限度**:上表来自 4 层 MLP,数值不可外推到扩散模型。
它证明的是**probe 设计的可用区间存在且窄**,真实数值必须由 Phase-1 给出。

---

## 附录 D(实现期修正之三、之四):两个实测崩溃及真因

移植进 trainer 后连崩两次,都不是"调参问题",记录真因与修法:

### D1. `RuntimeError: unscale_() has already been called`(job 1036848)

eval 趟跳过了 `optimizer.step()`,却仍走 `accelerator.clip_grad_norm_` ——
后者内部调 `scaler.unscale_(opt)`,而 `unscale_` 在两次 `scaler.update()` 之间只能调一次;
没有 step 就没有 update,于是下一趟再 unscale 直接抛异常。

**修法**:非 step 趟**完全不碰 scaler**,直接读 `.grad` 再手动除以 `scaler.get_scale()`。
附带好处:`a_m` 因此取的是**未裁剪**的真实梯度(裁剪会改变约束梯度的方向与幅度),
而 `Δ_adam` 仍走含裁剪的真实路径,与训练一致 —— 两者各自取对了。

### D2. `torch.OutOfMemoryError`,H200 139.8GiB 被耗尽(job 1036866)

我让 eval 趟"只前向、不 backward"。但**前向已经建了图**,不 backward 图就不释放,
12 microbatch × 10 timestep 的计算图全部堆积。

**修法**:eval 趟**也调 backward**(loss 已乘 0,得到零梯度但会逐 microbatch 释放图),
只跳过 clip 与 step。代价是多一次零值反传,可接受。

**教训**:"跳过 backward 省时间"在 PyTorch 里是错的直觉 ——
backward 同时承担**释放计算图**的职责。要真正省,必须在前向就 `torch.no_grad()`,
而这个循环的前向深埋在 `compute_student_step` 里,改动面远大于多跑一次零反传。

---

## 首份真实 probe(epoch 0,job 1036869)—— **n=1,不是结论**

```
C_before        ocr -1.204e-06   geneval -1.135e-06     <- 小负数,有容差余量 ✓
C_after_direct  ocr +2.457e-06   geneval -1.212e-07     <- ocr 实际违反 => active=true ✓
taylor ratio    ocr 0.129        geneval 0.091          <- 一阶只解释 9~13%
Cbefore/step    ocr 2.54         geneval 12.28
cert@1    pass=false  retention=1.00
cert@0.5  pass=false  retention=0.50
cert@0.25 pass=true   retention=0.25
```

### 最要紧的一条:QP 在每个 alpha 上都**什么也没做**

`cert@1` 的 `C_after` 与 `C_after_direct` **逐位相同** ⇒ mu = 0,没有发生投影。
原因可直接算出来:线性模型下

    C_ocr + a_ocr^T Delta_adam = -1.204e-06 + 4.733e-07 = -7.3e-07 < 0

**约束在一阶意义下是"满足"的**,QP 因此无事可做。
而精确重算给出 `C_after = +2.457e-06 > 0` —— **实际违反了**。
一阶只捕捉 13%,线性模型系统性地过于乐观。

α=0.25 之所以通过 certificate,是因为步长缩小后真实变化也变小,
**不是因为投影修正了方向**(此时 mu 仍为 0)。

### 若这一现象在更多 probe 上重复,含义是

**价值在 exact-certificate + backtracking line search,不在 QP projection。**
一个基于一阶梯度的投影**看不见**这种违反,因此无从修正;
要么给约束加保守安全裕度、要么用二阶信息、要么就是直接做精确线搜索(α=0.25 实际做的事)。

这会是一个干净的结论:**对 QP 是否定的,对"精确证书 + 回溯"是肯定的**。

### 与我此前被推翻的 0016 结论的关系(必须说清)

codex 0023 §3.3 认为我原来那个"一阶只解释 8~10%"主要来自重采样与状态漂移。
**本次是干净的 same-state 测量**(同一缓存窗口、同一 batch、真实 AdamW 更新、
`a_m` 整窗累积),却得到 **9~13%** —— 与原数字量级相同。

若后续 probe 稳定复现,说明**原结论的数值是对的**,
但我当时的论证不足以支撑它(codex 的批评仍然成立:我用一个混淆的测量下了正确的结论)。
**在 n 足够之前不下这个判断。** 按预注册需 active probe 达标才谈 go/no-go。

### 首轮的一个"bug 恰好做对了事、但结果丢了"的记录(job 1036869)

作业 exit=0、33 分钟正常结束,但**只留下 1 份报告**,而我要 8 份。查真因:

epoch 循环是 `while True:`,`epoch += 1` 是**循环体最后一条语句**(在 save_freq 块之后),
而我的 `continue` 把它跳过了 ⇒ `epoch` 恒为 0。

**这对 probe 恰恰是期望行为**:8 个 probe 都落在同一个 checkpoint 的 θ 上,
只有采样窗口不同 —— 正是预注册要的"同一 checkpoint 上 8 个 probe"。
但报告文件名按 `epoch` 命名 ⇒ **8 份互相覆盖,只剩最后一份**。

**修法**:文件名改按 probe 序号 `phase1_probe_{idx:02d}.json`,序号在写文件前自增。
同时把 `global_step` 写进报告 —— `[resume]` 日志被 paddleocr/mmdet 吞掉,
无法从日志确认 resume 是否生效,只能靠这个字段核实。

**教训**:"作业 exit=0 且有产出"仍然不等于"拿到了要的数据"。
这次是**产出被自己覆盖**,只有对着"我要 8 份、实际 1 份"这个数量差才发现。

---

# ✅ Phase-1 判定(8/8 probe,job 1036899)—— **永久关闭 QP;价值在 exact certificate + backtracking**

8 个 probe 全部落在 `crd` 的 checkpoint-91 上(`global_step` 8~64,同 θ、不同采样窗口)。

| 闸门 | 结果 |
|---|---|
| ① active 占比 | **8/8 = 100%** ⇒ 不落"投影没有要解决的问题"支 |
| ② 一阶 vs 精确(同缓存状态) | ocr ratio 中位数 **+0.117**(范围 0.020~0.288)、geneval **+0.051**(−0.031~0.142);符号 8/8、7/8 一致 |
| ③ exact certificate | **8/8 通过**,α ∈ {0.25, 0.5}(即 1~2 次 backtrack) |
| ④ **QP 改动更新方向的次数** | **0/8** |

## 判定:按预注册,QP **永久关闭**

预注册的进入 Phase-2 条件是"≥90% 通过 **且** `reward_retention` 中位数 ≥ 0.5"。
实测通过率 100% ✓,但 retention = α,中位数 **0.375 < 0.5** ✗
⇒ 落入"其余情形 ⇒ 永久关闭 QP,保留 fixed hinge 作为结果"。

## 但更重要的是**为什么** —— 这一条比判定本身有价值

**QP 在 8 个 probe 里 μ 恒为 0。** 每次原因相同:

    线性模型:  C_m + a_m^T Delta_adam < 0   =>  "约束满足",投影无事可做
    精确重算:  C_m(theta + Delta_adam) > 0  =>  实际违反

一阶模型只解释真实变化的 **5~12%**,因此**系统性地过于乐观约 10 倍**。
任何**基于一阶梯度的可行性判据**都会漏掉这种违反 —— 不是 QP 的求解器不好,
是**它赖以工作的一阶信息在这个问题上不够用**。

certificate 之所以 8/8 通过,靠的是 **backtracking 把步长缩到 1/4~1/2**,
不是投影修正了方向。代价是丢掉约 60% 的 reward 更新(retention 中位数 0.375)。

⇒ **可写进论文的机制结论**:
在这个设置下,保证能力约束的有效手段是
**"精确 same-state 证书 + 回溯线搜索"**,而不是一阶投影;
后者在此问题上是**恒等映射**。

## 与 0016 旧结论、以及 codex 复核的关系(必须说清)

我在 `0016` 用一个**混淆的**探针得到"一阶只解释 8~10%",据此关闭了 QP。
codex 0023 §3.3 指出那个测量同时改变了重采样/状态/batch/优化器映射,
**推理越界**,并要求重做干净的 same-state 测量。**这个批评是对的,我接受并重做了。**

现在干净测量的结果是 **5~12%** —— 与原数字量级相同。

所以准确的表述是:
**我原来的数字是对的,但当时的论证不足以支撑它。**
codex 的方法学批评成立,而它的结论(QP 不应据此关闭)在干净数据下被推翻 ——
QP 确实应该关闭,只是需要这份数据才能这么说。

**两件事都要记:** 我曾用错误的理由得到正确的数字;
也曾因为判据写得不严谨(附录 A/C)而差点用不可满足的 certificate 去否定任何方法。

## 后续

- **不进入 Phase-2 的 projection 臂**;
- fixed hinge (w=10) 作为方法侧的最终结果,基线链继续补齐
  (MOPD-continue 已跑完 200 epoch,朴素 KL 在跑);
- `code/0017_qp_solver.py` 与 `code/0025_qp_repair_solver.py` 作为**已验证但未采用**的组件保留,
  论文中若讨论"为什么不用投影",可直接引用本节的 0/8 与 5~12%。
