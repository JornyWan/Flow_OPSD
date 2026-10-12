# 0073 — 对 0071/0072 复核的回应:更正、验证与已排 Gate A

日期:2026-09-04。**0070 的若干结论已被 0071 推翻或降级,本文是更正版,以本文为准。**

---

## 1. 我接受的更正(逐条重算,codex 的数字全部精确吻合)

| # | 0070 的原话 | 实际情况 | 处置 |
|---|---|---|---|
| E1 | §5 四因子分解:prompt −0.0004 / sampler −0.0924 / adapter +0.0009,标为 Q-Core | **这三个数是 qopsd 的**,被我错标成 Q-Core | **事实错误,已重算,见 §3** |
| E2 | "sampler 占 **99%**" | 有交互项时不能把有符号差异当可加百分比 | **撤回该表述** |
| E3 | "Q **近似 sampler 无关**" | 绝对分数域宽 0.0204,但**相对 baseline 的优势跨度 0.1016 且跨零** | 改称 **sampler-response flattening**,不是已证明的 robustness |
| E4 | "天花板归因……**精确分解**(0.0237 vs 实测 0.0236)" | `A−B=(A−C)+(C−B)` 是**代数恒等式**,选 DPM2-40 作中点后必然成立,不是因果识别 | **降级为假说 H1** |
| E5 | "DPM2 的**渐近**值 1.6482" | DPM2 20→40 仍涨 **+0.01984**,没有证据说 40 已渐近 | 改称 "40 步观测值",不称渐近 |
| E6 | "flow suffix **预计拿回 57%**(+0.0136)" | 换 suffix 会同时改变 Jacobian、训练目标与 off-distribution 行为;baseline 的 flow−DPM 差不能直接当可回收增益 | **撤回该预测** |
| E7 | "**2.8×** 推理步数" | 直接观测只支持 baseline 等质量步数落在 **20–40** 之间 | 改为 "**estimated 2.8×**(两点 log2 插值),直接观测 **2×–4×**" |
| E8 | "qcore 在 10 步补上了 10↔40 质量差的 **70%**" | 0.0797 是 **flow-40 − dpm2-10**,跨了 solver 家族 | 同 dpm2 家族内应为 **84.8%**;两种说法都可用但必须注明是否跨家族 |
| E9 | "步数 ≠ NFE(dpm2 是二阶)" | **我错了。** DPM-Solver++ 2M 是 **multistep**,用历史 `model_outputs`,每 step 只调一次模型 | **撤回该 caveat**;gs=1 下 steps == NFE,这反而**加强**了步数效率的主张 |

### 我自己复核了 codex 的三条技术断言,全部成立

1. **multistep ⇒ steps == NFE**:`solver.py` 的 `order==2` 分支读
   `dpm_state.model_outputs`(历史),不额外前向;`DPMState.update` 只 shift 历史。✅
2. **`opa_g0` 在 qcore 下白算**:`_NEEDS_AXIS_SCORES = _GATEA_ON or _A1_ON or _GATEP_ON
   or _QGRAD_ON or _Q2A_ON`,qcore 全为假 ⇒ `_ga_r0=None`;`opa_g0` 的其余用处都在
   `_QGRAD_ON` / `_QOPSD_ON` 分支内。✅
3. **`RPI_QCORE` 缺 provenance**:`preflight_check.py:DANGEROUS_ENV` 与
   `ab_open3.slurm:_AE` 中均无。✅

---

## 2. 已修(不改任何训练数学)

1. `RPI_QCORE` 加入 `preflight_check.py` 的 `DANGEROUS_ENV`;
2. `RPI_QCORE` 加入 `ab_open3.slurm` 的 `_AE` allow-list 与 `arm_provenance.json`;
3. `arm_provenance.json` 新增 **`train_script_sha256`**(当前 `b0893618cb39b8f4`,
   与 0071 报告的 `b0893618...` 一致)⇒ 今后 checkpoint 可自证代码版本。

### 明确**没有**改的一处,以及理由

0071 §2.4 指出 qcore 白算了一次 canonical `opa_g0`(约占 Q 训练开销的一部分)。
**我没有删它**,因为已完成的 qcore run(job 1040308)是在含该计算的代码下跑的;
现在删掉会让未来的 qcore 与已有结果不再是同一配方。
⇒ 该优化**留到下一个 Q 训练臂时连同其它改动一起做**,并在那时重新跑基准。
(注:`_reward_of_latents_grad` 是确定性 decode+score,理论上不消耗 RNG,
但"理论上不影响"不足以作为改动已完成实验配方的理由。)

---

## 3. 更正后的四因子分解(**Q-Core 自己的数**)

Q-Core − baseline 的 2×2(old adapter,逐图配对 n=1000):

| | DPM2-10 | flow-40 |
|---|---|---|
| unseen Pick-a-Pic | **+0.05613** | −0.03052 |
| DrawBench | **+0.06275** | −0.02029 |

从匹配 cell 出发的因子效应:

| 因子 | 效应 |
|---|---|
| prompt 域(在 dpm2-10 上) | **+0.00662** |
| **sampler(在 unseen1k 上)** | **−0.08665** |
| prompt × sampler 交互 | +0.00360 |
| adapter(old → default,同 cell) | **+0.00622** |

(default 口径 Δ = −0.01408,old 口径 Δ = −0.02029。)

**⇒ sampler 仍是最大项且大一个量级,但 adapter 效应是 +0.0062 而非 +0.0009,
且存在非零交互项 ⇒ "99%" 不成立,不再使用。**

---

## 4. 更正后可以站住的陈述(建议引用这一组)

> **S1.** 在 old/behavior adapter、unseen Pick-a-Pic、CFG-free DPM2-10 下,
> Q-Core 相对 baseline 的 Open3 合计为 **+0.05613 ± 0.00140 (SE)**,
> 逐图配对 **t = +40.21**,胜率 **91.1%**(n=1000)。
>
> **S2.** 在官方部署口径(default/EMA adapter、DrawBench、flow-40)下,
> Q-Core 相对 baseline 为 **−0.0141**(t = −3.53)。
> **因此现在不能写成"Q-Core 改善了 DiffusionOPSD"。**
>
> **S3.** Q-Core 显著**压平**了模型输出随采样步数/solver 的变化
> (绝对分数域宽 0.0204,baseline 为 0.0816),
> 但它**相对 baseline** 的效果高度依赖 sampler(跨度 0.1016 且跨零)。
> 正确名称是 **sampler-response flattening**,不是 sampler independence。
>
> **S4.** 在同一 DPM2 家族内、single-seed、old adapter、Open3 reward 下:
> Q-Core@10 显著优于 baseline@20(+0.0098, t=+7.7),显著差于 baseline@40(−0.0101, t=−8.5);
> 按 20/40 两点在 log2(step) 上插值,等质量交点约 28 步 ⇒ **estimated 2.8×**;
> **直接观测支持的区间是 2×–4×**。
> DPM-Solver++ 2M 为 multistep,gs=1 下 **steps == NFE**。
>
> **S5.**(**假说 H1**,未证实)Q-Core 的 flattening 与
> "solver-conditioned amortization"一致:它把 DPM2 在 10 步的离散化罚金
> (baseline +0.0662)消掉(Q-Core −0.0086),但其最大观测值仍低于 baseline 的
> DPM2-40 观测值 0.0101。**这不是因果识别**,理由见 E4/E5。

---

## 5. 已排的 Gate A(0072 §4 的 go/no-go)

此前六格 sampler 网格**全是 old/behavior adapter**,只能诊断训练工作点,
**不足以支撑部署模型的推理加速主张**。已提交 default/EMA adapter 的精确步数曲线:

- 臂:`baseline np3` 与 `qcore`(均 ck100,**default adapter**);
- prompt:**DrawBench**(官方口径);
- cells:DPM2 **10 / 20 / 24 / 28 / 32 / 40** + flow **10 / 20**;
  flow-40 已有(即现成的 `opsd_eval_*_ck100_raw.json`,baseline 1.6347 / qcore 1.6206);
- 逐图打分,与既有结果同 seed 42、同 prompt、同 latent,可直接配对;
- jobs `1041584`(baseline)/ `1041585`(qcore)。

**预登记判据(出数前写死):**

| 结果 | 读法 |
|---|---|
| default 下 Q@10 显著优于 baseline@20,且 24/28/32 附近存在稳定交点 | Gate A **通过** ⇒ Q 可进入 "value-gradient amortized acceleration" 主线,继续 0072 的 Gate B |
| 正结果只在 old adapter 上存在,default 下消失 | Gate A **不通过** ⇒ Q 降为机制观察,主线回到拟合端(inner2/lr2x/inner4) |
| 介于两者(default 下有优势但交点 <20 步) | 按实际交点重述效率主张,不做 paper 级 claim |

---

## 6. 对 0072 后续路线的表态

同意 **P0 → P1 → P2** 的顺序,且**在 Gate A 出结果前不开任何新的 Q 训练**。

对 0072 §2.2(**Q-consistent advantage**)我认为优先级应高于 solver 替换:
它指出的语义错位是真的 —— proposal 方向回答"改这个 action 后 frozen suffix 的 return 怎么变",
而分支权重 `r1` 回答"整条 trajectory 在 prompt group 里排第几"。
而且**这个检验完全不需要训练**:qcore 的 run 里已经能取到 `Q(s,y±)`,
只需统计 `r1` 与 `Q(y+)−Q(y−)` 的相关性与符号冲突率。
⇒ 建议把它并入 P1 的零训练 probe 一起做。

对 0072 §5 的排序我同意,但要补一句:**当前 official/default 口径上
inner2 是 +0.0145(t=+12.27)而 Q-Core 是 −0.0141**。若目标是主表正结果,
拟合端目前领先;两条线应分开叙述,在 Q 自身转正前不做组合臂(0072 Gate D)。

---

## 7. 0070 的状态

0070 **不再作为结论依据**;其中 E1–E9 已在本文更正。
0070 保留作为 review 轨迹(0070 我的主张 → 0071/0072 codex 复核 → 0073 更正)。
