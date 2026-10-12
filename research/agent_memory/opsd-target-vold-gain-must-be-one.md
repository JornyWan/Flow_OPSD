---
name: opsd-target-vold-gain-must-be-one
description: OPSD 变体登记前先算残差对学生输出的反馈系数 κ(old≈θ):canonical κ=0、MSP 额外态 κ<0 稳定、κ>0 必发散(LT-OPSD 实例);κ≤0 必要但不充分
metadata:
  type: project
---

设计 OPSD 目标变体时,先写出分支残差 R,令 old=θ(Polyak 平均的 old 只滞后学生约一步),
看 R 对学生输出 v 的依赖系数 κ。小步长下学生速度 `Δv ∝ +R`,所以:

| 构造 | κ | 结果 |
|---|---|---|
| canonical(query 态) | 0 | 只沿 reward 方向 δ 漂移(中性)|
| MSP 额外态 k(目标 ȳ⁺ 来自 query 态) | −σ_k | 负反馈,收向固定目标,**任何 β 都稳定** |
| LT-OPSD(PCFM 式全局插值,§4.832) | +βσ_s(1−σ_q) | **正反馈**:query 态 x̂0 范数 237→~2450,连被训练的 pickscore 都掉到底模之下(t=−74)|

**Why:** LT 的失败 smoke 五条全过、单步数字看不出来,~5 个 epoch 才显形,发现时已烧掉一条满预算臂;
两行代数就能提前排掉。最初我用"v_old 系数 g 必须为 1"表述(离散不动点图景),它对 LT 结论相同,
但会把 β=0.1 的 MSP 误判成振荡发散(g=1−1/β=−9)—— 那是错的,所以改用 κ。

**How to apply:**
- 新目标构造登记前写出 R(v) 并算 κ;κ>0 不跑。
- **κ≤0 只是必要条件**:`rbltpcfm`/`rbltbridge` κ<0 也发散了(原因未定位)。
- 训练中盯范数类诊断(query 态 ‖y0‖ 可由信任域位移反推:‖d‖≈0.709·ρ·‖y0‖),比 reward_avg 更早暴露失控。
- 相关:[[opsd-single-reward-confounded-with-beta]](β 的作用不能靠代数恒等式下结论,只能靠实测)
