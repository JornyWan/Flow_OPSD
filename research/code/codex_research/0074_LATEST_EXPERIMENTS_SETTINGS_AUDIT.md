# 0074 — Claude 最新实验、结果与 setting 的清账审计

日期：2026-09-10。审计截止到 `EXPERIMENT_LOG.md` §4.507（2026-09-10 13:46）以及同一时刻的输出目录和队列状态。

本轮只做只读审计和记录，**没有提交、取消或修改任何训练/评测任务**。

---

## 0. 先给结论

目前不是“完全没有正结果”，而是有一个很强但不 novel 的正结果，被大量机制探针、组合实验和不断修订的解释淹没了。

1. **最可靠的正结果是学习率调大。** 官方 DiffusionOPSD Open3 配方的学习率是 `3e-4`；`lr4x=1.2e-3` 和 `lr8x=2.4e-3` 在两个 seed 上都显著改善官方评测目标。当前最高绝对分是 `lr8x` 的 **1.6952**，对应同硬件 baseline 1.6378，差 **+0.05740**。
2. **这首先说明官方/当前 baseline 欠调，不是新的 OPSD 方法。** `lr8x` 只改一项 config，没有新的 target、objective 或 optimization principle。它应成为以后所有新方法必须击败的 **tuned baseline**，不能包装成现有方法贡献。
3. **Q-core 目前不应作为 paper 主线。** 它确实压平了 DPM2 的 step-response，但官方 `flow-40` 上比 baseline 低 **−0.0141**；`Q+lr2x` 也只回到 baseline（−0.00160），远低于单独 `lr2x`（+0.02817）。更关键的是，同为 **10 NFE** 时，原 baseline 的 `flow-10=1.6350` 已高于 `Q-core+DPM2-10=1.6263`。因此当前 Q 没有形成质量–推理成本 Pareto 改善。
4. **inner/gacc 是优化强度旋钮，但都不如直接调 LR。** 它们在默认 LR 下有效；一旦基础 LR 已提高，边际收益单调下降并转负。`lr4x+inner2`、`lr4x+gacc2` 都明显差于 `lr4x` 单臂，`lr8x+inner2` 直接失稳。
5. **query sigma、rho、单步 κ/probe 均没有导出一个能 work 的新方法。** probe 排序甚至与真实训练结果不一致；这些结果适合写成负面机制诊断，不应继续无边界扩展 arm。
6. **最新 §4.507 的“只差训练硬件”结论不满足严格单变量条件。** 被比较的两条 `lr8x` run 的 `train_script_sha256` 分别是 `2218e07425c9b2e0` 和 `d150b446857543fc`。在找到旧源码快照并证明差异全部位于关闭的 probe 分支之前，只能称为“与硬件偏移一致”，不能称为严格的硬件因果实验。

一句话总结：

> **现在得到的是一个强的 tuned-DiffusionOPSD baseline 和若干优化动力学观察，还没有得到自己的新 OPSD 方法。继续堆小改动不会自动变成 paper。**

---

## 1. 为什么现在看起来很 messy

从 9 月 4 日以后，目录新增/更新了：

- 57 份 `run_config.json`；
- 188 份 `matched_pi_*_default_drawbench_flow40.json`；
- 276 份顶层 JSON 评测产物；
- 211 份 Slurm 输出；
- 整个 `EXPERIMENT_LOG.md` 已有 **508 个 `4.x` 小节**，最新到 §4.507。

这些实验混合了五种不同目的：

1. baseline 超参搜索（LR）；
2. 额外优化步/批量实验（inner、gacc）；
3. 方法修改（Q-core）；
4. 零训练机制探针（Q-sigma、κ-fit、梯度相消）；
5. 基础设施校准（训练 GPU、评测 GPU、Slurm/QOS、watcher）。

它们不能放在同一张“method 排名”里讲。尤其是 probe 的局部量不等于训练后的 held-out 指标，硬件校准也不等于方法贡献。

---

## 2. 所有主结果共用的标准 setting

| 项目 | 当前主表设置 |
|---|---|
| 起点 | `stabilityai/stable-diffusion-3.5-medium` |
| 训练代码 | official DiffusionOPSD fork，`config/public.py:sd35_open3`，LoRA |
| 训练 rollout | DPM-Solver++ 2M，10 steps，`guidance_scale=1.0`，noise level 0.7，512×512 |
| 数据规模 | Pick-a-Pic 全量配置；每外层 epoch 48 prompt groups × 24 images = 1152 images |
| 训练目标 | `PickScore/26 + CLIPScore + HPSv2.1` |
| 默认 optimizer setting | constant LR `3e-4`，`num_inner_epochs=1`，gradient accumulation 48，AdamW，max grad norm 1.0 |
| 主评测 | DrawBench 1000 prompts，flow sampler，40 steps，CFG=1，default/EMA adapter，seed 42，raw PickScore |
| 主表统计 | 同 prompt/seed 的逐图配对；项目自定 MDE≈0.0033 |
| 非训练轴 | Aesthetic、ImageReward，只作额外探针 |
| 主要训练 seeds | 20260830（s1）、20260905（s2） |
| 配对 baseline | H200+s1→`basenp3`；A100+s1→`baseAA`；A100+s2→`bases2`；H100→`baseH100` |

三条主 baseline 的 A40 主评测绝对分为：

| baseline | 训练硬件 / seed | 目标合成分 |
|---|---|---:|
| `basenp3` | H200 / s1 | 1.63471 |
| `baseAA` | A100 / s1 | 1.6378 |
| `bases2` | A100 / s2 | 1.63360 |

后续所有 Δ 都必须相对同 seed、同训练硬件的 baseline；评测还必须在同 GPU 型号上生成，才能保留逐图配对。

---

## 3. 干净的实验台账

### 3.1 学习率系列：唯一清楚、稳健、实用的正结果

除 LR 外设置不变，100 个外层 epoch，`CK_MULT=1`。

| arm | LR | seed / 训练 HW | ep100 绝对分 | 配对 Δ | 结果 |
|---|---:|---|---:|---:|---|
| `lrhalf` | 1.5e-4（0.5×） | s1 / A100 | 1.6069 | **−0.03094** | 明显更差 |
| `lr2x` | 6e-4（2×） | s1 / H200 | 1.6629 | **+0.02817** | 稳定正结果 |
| `lr2x_s2` | 6e-4 | s2 / A100 | 1.6653 | **+0.03171** | seed 复制 |
| `lr4x` | 1.2e-3（4×） | s1 / H200 | 1.6828 | **+0.04807** | 强正结果 |
| `lr4x_s2` | 1.2e-3 | s2 / A100 | 1.6893 | **+0.05573** | seed 复制 |
| `lr6x` | 1.8e-3（6×） | — | — | — | **实际尚未启动**；§4.494 已更正早先“已启动”的误记 |
| `lr8x` | 2.4e-3（8×） | s1 / A100 | **1.6952** | **+0.05740** | 当前最高绝对分 |
| `lr8x_s2` | 2.4e-3 | s2 / A100 | 1.6921 | **+0.05851** | seed 复制 |
| `lr8x_u200` @ep100 | 2.4e-3 | s1 / H200 | ≈1.69085 | **+0.05614** | 与硬件偏移模型一致；源码 hash 混杂，见 §5.2 |
| `lr10x` | 3.0e-3（10×） | s1 / A100 | 无 ep100 | ep30 +0.04934；ep40 +0.05053；ep50 +0.02584 | ep55 左右发散；在发散前已显著差于 8× |
| `lr12x` | 3.6e-3（12×） | s1 | 无 | — | ep21 左右发散 |
| `lr16x` | 4.8e-3（16×） | s1 | 无 | — | ep5 左右发散 |
| `lr12xeps` | 3.6e-3，Adam ε=6.5e-7 | s1 / A100 | 1.68868（A40） | **+0.05085** | 稳住 12×，但仍比 8× 低 −0.00655（t=−4.96） |

当前 LR 结论：

- 在已测网格里，最佳点是 **8×**；稳定性边界位于 8× 与 10× 之间。
- 4×/8× 的正效应跨两个 seed 复制，是目前最可信的结果。
- `lr8x` 达到同硬件 baseline@ep100 的分数约需 46.5 epoch，因此日志称 **2.15× training acceleration**。这个值来自开发集曲线内插；“零额外单步成本”可信，精确的 2.15× 仍不是跨任务定律。
- `lr12xeps` 说明“把更高 LR 稳住”本身并不会超过 8×；至少 Adam ε 这条稳定化办法已失败。

### 3.2 inner / gradient-accumulation 系列：默认 LR 下有效，但不是最佳路线

| arm | 只改什么 | 每外层 optimizer steps | 等效 ck | ep100 Δ | 解释 |
|---|---|---:|---:|---:|---|
| `inner2` | `num_inner_epochs=2`，同一构造目标完整吃两遍 | 2 | 200 | +0.01138（s1），+0.01501（s2） | 有效但边际小；每 epoch 约 +9% 成本 |
| `inner4` | `num_inner_epochs=4` | 4 | 400 | +0.03586 | 默认 LR 下有效；成本约 +30%（估计） |
| `gacc2` | accum 48→24，把一次大更新拆成 2 次 | 2 | 200 | +0.00720（s1），+0.01319（s2，硬件不配对） | 几乎不增单 epoch 成本；收益小 |
| `gacc4` | accum 48→12，拆成 4 次 | 4 | 400 | +0.02940 | 默认 LR 下有效；仍低于 lr4x/lr8x |

这组实验最稳定的结构性结果不是“多拟合更好”，而是：

> **额外拟合步的边际价值随基础 LR 升高而下降。**

`inner2` 在 ep100 相对同 LR 单臂的贡献为：

| 基础 LR | 0.5× | 1× | 2× | 4× |
|---|---:|---:|---:|---:|
| `inner2` 边际贡献 | +0.02158 | +0.01138 | +0.00250 | **−0.01592** |

该顺序在多个 epoch、两个 seed 和 `gacc` 旋钮上方向一致。这是一个可信的优化动力学观察，但还不是方法。

### 3.3 组合系列：没有超过 tuned LR；大部分是负交互

| arm | setting | ep100 Δ vs baseline | 相对关键单臂 | 判读 |
|---|---|---:|---:|---|
| `lr2xinner2` | LR=6e-4，inner=2，ck200 | +0.03067 | 仅略高于 lr2x；训练 HW 不同 | 基本冗余 |
| `lr4xinner2` | LR=1.2e-3，inner=2，ck200 | +0.03215；s2 +0.03600 | 比 lr4x 低 −0.01592；s2 低约 −0.0197 | 明确负交互，跨 seed |
| `lr4xgacc2` | LR=1.2e-3，accum=24，ck200 | +0.03053；s2 +0.03246 | 比 lr4x 低 −0.01754；s2 低约 −0.0233 | 明确负交互，跨 seed |
| `inner2gacc2` | inner=2，accum=24，ck400 | +0.03874；s2 +0.04189 | 高于两个小单臂之和，但仍低于 lr4x/lr8x | 低 LR 下存在正交互，但非最佳配方 |
| `inner2lrhalf` | LR=1.5e-4，inner=2，ck200 | −0.00935 | 比 lrhalf 好 +0.02158，仍低于 baseline | 支持“低 LR 时 extra fit 更有用” |
| `lr8xinner2` | LR=2.4e-3，inner=2 | 无 | ep3 即触发失稳标准 | 不可用 |

即使采用 Claude 的墙钟换算，排序仍是 `lr8x≈2.15×`、`lr4x≈1.92×`、`inner2gacc2≈1.54×`，组合路线没有击败直接调 LR。

### 3.4 Q-core：有机制现象，没有可用的主线改善

Q-core 的真实改动是：保留官方两步 OPSD target construction 和 advantage 语义，只把 clean reward oracle 换成 frozen suffix action-value oracle；`qcorelr` 再额外把 LR 改为 6e-4。

default/EMA + DrawBench 的 step curve：

| sampler / steps | baseline | Q-core | Q−base |
|---|---:|---:|---:|
| DPM2-10 | 1.5629 | **1.6263** | +0.0634 |
| DPM2-20 | 1.6052 | 1.6245 | +0.0193 |
| DPM2-28 | 1.6160 | 1.6197 | +0.0037 |
| DPM2-32 | 1.6191 | 1.6187 | −0.0004 |
| DPM2-40 | 1.6231 | 1.6187 | −0.0044 |
| flow-10 | **1.6350** | 1.6074 | −0.0276 |
| flow-20 | **1.6369** | 1.6187 | −0.0182 |
| flow-40（官方） | **1.6347** | 1.6206 | **−0.0141** |

可以成立的观察：

- baseline 的 sampler/step 分数域宽约 0.074，Q-core 约 0.019；**Q 改变了曲线形状，而 LR/inner 主要做竖直平移**。
- Q-core DPM2-10 比 baseline DPM2-40 高 +0.0032，t≈2.2。这只能说“与近似打平相容”；没有预先定义 non-inferiority margin 时，`p>0.05` 或低于 MDE 不能证明等价。
- 聚合打平依赖轴间补偿：Q-core 的 PickScore 更差，而 CLIP/HPS/Aesthetic/ImageReward 更高，并非各轴等质量。

不能成立的实用 claim：

- **不是实际 4× Pareto acceleration。** 同样 10 次模型调用时，baseline `flow-10=1.6350` 已比 Q-core `DPM2-10=1.6263` 高 **0.0087**；baseline 用当前可用的更好 solver 即支配 Q。
- Q-core 的最佳观测分 1.6263 也低于 baseline flow-10/20/40。
- 官方 flow-40 上 Q-core 为 −0.0141；`qcorelr` 为 −0.00160，而单独 `lr2x` 为 +0.02817。也就是说，Q 与更强 LR 存在明显负交互。

因此 Q 目前最多是“reward finetuning 如何制造 step sensitivity、suffix-Q 如何阻止这种 sensitivity”的机制附录，不具备 paper 主方法资格。

### 3.5 query sigma、rho 与 Q probe：方向已基本关闭

#### Q-sigma probe

`qsig` 只运行 probe，不训练正式 arm；覆盖 DPM10 schedule 的 kq=1…9，对应 sigma 约：

`0.9601, 0.9133, 0.8577, 0.7904, 0.7073, 0.6022, 0.4649, 0.2780, 0.0089`。

probe 找到 `sigma≈0.35–0.65` 的 clean-direction suffix gain 负区间，并观察到非退化点上 Q direction 通常优于 clean direction。这个局部测量跨 seed/硬件可以复现，但**不能预测完整训练**。

#### 正式 qstrain 训练

这些 arm 不开 Q-core，只改变官方 OPSD 的 `opa_query_sigma`：

| arm | sigma | ep100 Δ | 结果 |
|---|---:|---:|---|
| baseline | 0.278049 | 0 | 默认值 |
| `qstrain9` | 0.008929 | **−0.11829** | 灾难性退化 |
| `qstrain7` | 0.464876 | −0.00134 | 与 baseline 打平 |
| `qstrain5` | 0.707278 | **−0.04706** | 明显退化 |

probe 排序与真实训练排序的 Spearman ρ=−0.40（n=4）。没有一个 sigma 超过默认值，因此这条方向应关闭。

#### rho

`rho2x` 把 `opa_rho` 从 0.1 改到 0.2。A100 评测 Δ=−0.00116（t=−1.06），最新 A40 总表 Δ=−0.00259；均在接近零的量级。HPS 上升、Pick/Aesthetic 下降后在合成目标中抵消。它说明旋钮生效但没有净收益，不应称为方法正结果。

### 3.6 κ-fit / 梯度探针：主要结果是“局部 probe 不能解释最终收益”

这些实验没有产生新的训练方法，得到的是若干排除性结果：

- 单步目标实现率很低，输出位移大多与逐样本构造目标不对齐；
- κ 在较大步长区间饱和甚至下降，不能解释为什么更大 LR 的长期 held-out 表现更好；
- 在 baseline@100、lr4x@100、baseline@200 等起点上，对齐度没有随性能明显改善；
- 尝试从现有 30 个训练字段找高 LR 失稳的廉价预警，13 个 arm 上最佳候选 Pearson r=−0.021，失败；
- 早先从小子集提出的“族峰位”“上升时间”“probe 排序”等规律，后续加入更多点后多次被撤回。

所以 κ/probe 当前只能用来排除解释，不能用来选 arm，更不能再据此批量派训练。

### 3.7 评测与硬件校准

已确认：

- 同 GPU 型号、同 checkpoint、同 seed/protocol 的生成是逐图 deterministic；
- 换 GPU 型号时均值偏差通常小于 MDE，但逐图样本会全部改变，因此跨型号不能保留 paired difference；
- 训练硬件会带来约 0.003–0.004 的系统差，已经和 MDE 同量级，所以 baseline 必须按训练硬件配对；
- §4.506 仍有两行训练硬件不配对：`lr2xinner2`（H100 arm/H200 base，估计影响小）和 `gacc2_s2`（H200 arm/A100 base，可能低估约 1 MDE）。

最新 §4.507 的 H200 `lr8x_u200@ep100` 给出 Δ=+0.05614，确实与先前硬件偏移预测 +0.05740 接近。但它与旧 A100 `lr8x` 的训练脚本 hash 不同，因此只能作为支持性证据，不能写“唯一变量是硬件”。

---

## 4. 哪些结论真正站得住

### 高可信

1. **当前官方 DiffusionOPSD 配方在这个 model/data/reward/protocol 下明显欠调 LR。** 4×/8× 相对各自同硬件 baseline 的增益在两个 seed 上重复出现。
2. **已测 LR 网格的最佳区间靠近 8×，10× 已过最优并最终失稳。** 10× 在明显崩溃前的 ep30/40 就已落后 8×。
3. **额外 inner/gacc 的边际收益随基础 LR 增加而下降，并在 2×–4× 之间过零。** 该方向跨 seed、不同拟合旋钮和多个 checkpoint 基本一致。
4. **把当前有效旋钮机械相加不是方法。** lr4x+inner/gacc 都被 lr4x 单臂支配；lr8x+inner 不稳定。
5. **Q-core 改变 sampler-response 形状，但没有改善官方 flow protocol，也没有击败同 NFE 的 baseline flow sampler。**
6. **query sigma probe 不预测正式训练；实际 sigma 调参没有优于默认值。**

### 中等可信、必须带限制

1. `lr8x` 的“2.15× training acceleration”：在当前 DrawBench 开发协议内合理，但依赖曲线内插、已被反复用于选择的评测集，并未跨任务/模型验证。
2. 训练硬件偏移约 0.003–0.004：多条数据相容，但最新独立验证仍有脚本版本差异。
3. `inner2gacc2` 的低 LR 超可加性：跨两个 seed，但它仍被 lr4x/lr8x 支配，且不能与高 LR 安全组合。

### 目前不能写进 paper 主张

1. “Q-core 带来 4× 实际推理加速”；
2. “Q 对 sampler robust/independent”；
3. “κ、gain_clean 或某个训练 metric 能预测最终 arm”；
4. “inner family 与 LR family 有稳定的二元峰位签名”；
5. “当前已经有一个新的 OPSD algorithm”；
6. “最新 H200/A100 lr8x 比较严格识别了纯硬件因果效应”。

Claude 自己在 §4.498/§4.507 的记账也支持这种保守判断：真正关于系统行为的 9 个明确先验中 **0/9 完全兑现**；后来唯一完全兑现的一项是把刚测出的硬件校准值外推到新 arm，而不是新方法预测。

---

## 5. 当前实验设计/记录的关键问题

### 5.1 DrawBench 已经是开发集，不再是未见 test set

9 月 4 日以后，仅 `DrawBench + flow40 + default` 的 matched JSON 就有 **188 份**。同一 1000 prompts 被用于：

- 选 LR；
- 选 checkpoint；
- 判断组合；
- 修改机制故事；
- 决定是否继续新 arm。

因此它已经是 validation/development set。表里的 paired t 值只描述某一个固定比较的逐图误差，**没有校正大量 arm/checkpoint 选择带来的 selection bias**。最终论文必须使用一套从未用于选方法/超参的新 prompt test set；不能继续把 DrawBench 称为真正 held-out test。

### 5.2 provenance 仍不足以复现实验

我直接检查了产物：

- `run_config.json` 有 sampler、reward、OPSD setting，但**没有 train 配置**，因此看不到有效 LR、inner epochs、gradient accumulation；
- `git_commit` 是 `unknown`；该 fork 当前也不是一个可用的 Git repository；
- `arm_provenance.json` 记录 arm 名、环境开关和截断的脚本 hash，但**不记录 `EXTRA_CFG`**，所以 LR/inner/gacc 的真实数值仍需回查 Slurm case 和 stdout；
- 现在同时存在 `ab_open3.slurm`、`ab_open3_v2.slurm`、`ab_open3_v3.slurm`，内容已分叉；
- 旧 `lr8x_s1` hash=`2218e...`，新 `lr8x_s2/u200` hash=`d150b...`，却被叙述成同配方比较；没有保存与每个 hash 对应的完整源码 snapshot。

这意味着当前结果大多可以理解，但还不能由单个 run 目录自包含地复现。

### 5.3 baseline 不是一个数，而是不断变化的一组锚点

训练硬件和 seed 确实影响分数，因此 Claude 后期为 H200/A100/H100、seed1/seed2 建了多条 baseline。这是必要修正，但也造成早期很多表后来重算。以后每行结果必须明确绑定：

`init + data + seed + training GPU + script hash + effective config + checkpoint + adapter + sampler/NFE + eval GPU`。

少一项就不能叫严格 paired comparison。

### 5.4 大多数“新方法”只在同一训练 reward 上验证

主目标同时用于训练和主排名，只是 prompts 换成 DrawBench。Aesthetic/ImageReward 没进训练目标，算部分独立检查，但还不足以排除 reward overfitting。第二个 seed 也不能替代第二个 benchmark、第二个 reward set、第二个 backbone 或 human preference evaluation。

### 5.5 实验生成速度高于假设淘汰速度

日志中多次出现：

- 小子集规律扩到全集后失效；
- 判据没有覆盖实际结果；
- 数据出来后才发现比较轴/硬件/续跑关系没对齐；
- watcher 重复、锁静默 exit 0、错误 partition/QOS、发散监控被删等操作问题。

Claude 后期主动更正了这些问题，这一点是好的；但正确的反应不是继续加 arm，而是停止扩张并重建一个更小的 confirmatory protocol。

---

## 6. 我建议现在怎么收束

### 6.1 立即改变项目状态定义

把当前所有结果标记为 **exploratory/dev phase**。保留正在跑的 `lr8x_u200` 完成 ep200，它只回答长期稳定性；在完成清账前，不再启动 `lr6x`、新 Q sigma、更多组合或新 κ probe。

这不是说 lr6x 永远不跑。它是 4×–8× 之间唯一合理的 LR 补点，但只服务于选择 tuned baseline，不会产生方法 novelty。

### 6.2 冻结两个 baseline

以后论文表必须同时报告：

1. **Official baseline**：官方 DiffusionOPSD，LR=3e-4；
2. **Tuned baseline**：当前先用 LR=2.4e-3（8×），并保留 LR=1.2e-3（4×）作为更保守的稳定性点。

任何新方法必须与 tuned baseline 比，而不是只相对明显欠调的 3e-4 baseline 报正结果。

### 6.3 Q 降为机制支线

当前 Q 需要满足的最低复活条件不是再调 sigma，而是：

- 在相同 sampler、相同 NFE 下超过 tuned baseline；
- 或在 fresh test set 上画出严格的 quality–NFE Pareto frontier，至少有一个 Q 点不被 baseline 的 flow sampler 支配；
- 再谈 Q-consistent advantage / direct suffix value-gradient 等真正改变算法核心的设计。

在此之前，不应以 Q 的 DPM2 10-vs-40 比较作为 paper title claim。

### 6.4 如果要做 paper，当前有两条诚实路线

**路线 A：分析/benchmark paper。** 主发现是 DiffusionOPSD 的官方优化配方欠调，以及 LR 与 extra-fit 之间存在稳定负交互。要达到论文强度，必须在至少第二个 backbone、第二个 reward/data setting 和新 test set 上复现。现在只有一个 SD3.5-M/Open3 setting，不够。

**路线 B：真正的方法 paper。** 需要一个能在 tuned LR baseline 之上工作的算法，例如自动选择安全有效步长/更新预算、并跨 setting 泛化。当前 `lr12xeps` 和所有廉价预警 proxy 都失败，所以这个算法还不存在；不能从现有 LR sweep 直接命名包装出来。

### 6.5 下一轮 confirmatory matrix 应极小

在 provenance 修好、fresh test 冻结以后，只做：

- official LR1×、LR4×、LR8×；
- 相同代码 snapshot、相同训练硬件；
- 至少 3 个真正独立 seed；
- 新的未使用 prompt test set；
- 训练目标三轴 + independent metrics；
- quality、NFE、训练采样量和 wall-clock 同时报告。

这组实验回答“强 baseline 是否真的成立”。在它之前继续发明组合，会让方法比较建立在移动的 baseline 上。

---

## 7. 当前运行状态

审计时队列中与本项目相关的训练只有：

- job `1046750`，`lr8x_u200`，H200，ai-gpu17，RUNNING；
- 最新 metrics 已到 epoch 185，checkpoint 已到 180；
- ep100 的 A40 评测已经完成：Δ=+0.05614；
- `lr6x` 没有 output directory，也不在 `squeue`，所以并未真正启动。

本轮 Codex 没有对这些任务做任何操作。

---

## 8. 最终判断

如果现在必须选择一句对项目最准确的话，应当是：

> **我们发现 DiffusionOPSD 的训练优化远未调好：单纯把 LR 调到 4×–8× 就能稳定超过官方配方，而且会使 inner/gacc 等“方法改动”的表面收益大幅缩小甚至反转；当前 Q 方法虽产生有趣的 sampler-response flattening，但在相同 NFE 的强 sampler baseline 与 tuned training baseline 面前被支配。**

这是一个清楚、有价值、也比较不舒服的结论：它推翻了不少此前的 idea story，但能让下一步终于建立在正确 baseline 上。
