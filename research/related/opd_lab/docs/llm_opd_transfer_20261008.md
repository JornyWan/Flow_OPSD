# LLM OPD 到扩散模型的迁移判断

2026-10-08。研究讨论稿；没有实现新算法或提交 GPU。推荐先验收完整 OPD 学习链，再研究从学生实际状态出发的短轨迹闭环蒸馏。后者是待证伪假说，不是已经确定的论文贡献。近期论文的作者实验不能替代本地复现；下列未特别说明的 2026 工作按预印本证据处理。

**最重要的区分：LLM 中报告有效、数学上可迁移、适合当前失败模式、具有新颖性，是四个不同条件。**

## 候选文献与迁移边界

| 工作与一手来源 | 机制与证据范围 | 对本项目的判断 |
|---|---|---|
| [EOPD](https://arxiv.org/html/2603.07079v1)，[ICML 2026 PMLR](https://proceedings.mlr.press/v306/jin26e.html) | 在高教师熵位置增加 forward KL，兼顾 reverse KL 与多样性；有数学任务多尺度实验 | 发表证据较强，但固定同协方差高斯下两个 KL 相同；不能直接搬熵加权 |
| [Trajectory-Refined Distillation](https://arxiv.org/html/2606.08432v1)，[代码](https://github.com/louieworth/trd) | 教师改写学生轨迹，再在修正后的前缀监督；数学与代码实验 | 最值得借鉴连续修正的学习对象；不是把文字前缀问题原样套到可编辑的噪声 latent |
| [Reward-Aligned OPD](https://arxiv.org/html/2608.27960v1) | 依据轨迹蒸馏回报与任务结果的符号关系筛选 | 不是反事实终点验证，也不是精确参数梯度对齐；直接迁移容易回到已有稀疏筛选困境 |
| [LGR](https://arxiv.org/html/2605.30833v1) | 用候选下一状态的教师分布峰值构造 lookahead 信号，关注长前缀监督退化 | 固定高斯的峰值由协方差决定，不会自动度量教师修正质量；论文自身短序列无优势，大权重可产生重复 |
| [Calibrating Teacher–Student Discrepancy](https://arxiv.org/html/2609.21619v1) | 特权正负条件干预估计教师自身偏移，校准应学习的差异 | “差得多”不等于“值得学”有启发；修改图像 prompt 同时修改任务，不能直接当教师不确定度 |
| [Trajectory Learnability / LWD](https://arxiv.org/html/2609.18321v1) | 辅助训练后用缓存轨迹的似然变化构造有界权重 | 可学习性不等于终端效用，另有辅助训练成本；不优先再加 shadow update |
| [Sequential Beats Joint](https://arxiv.org/html/2609.04108v1) | 比较先 OPD 后 RLVR 与联合优化等方案 | 实用备选是 OPD 后接联合任务 NFT/FlowGRPO；迁移假说清晰，但配方本身的新颖性较弱 |
| [Open-MOPD](https://arxiv.org/html/2608.19098v1) | 多领域能力失衡可来自 token 数量、收敛预算和奖励滞后，未必主要来自教师冲突 | 应审计有效梯度份额；固定长度 diffusion 不直接具有文本长度失衡，不能照抄其因果解释 |
| [SDPO](https://arxiv.org/html/2601.20802v1)，[代码](https://github.com/lasgroup/SDPO) | 利用反馈上下文构造自教师的密集监督 | SD3.5M 没有现成的语言式反思能力；加入 VLM/编辑器会改变教师、输入信息与成本 |
| [REVO](https://arxiv.org/html/2609.37500v1) | 用重要性修正和方差信息复用 rollout | 确定性 ODE 没有可直接复用的 token likelihood ratio；先解决学习收益，再考虑采样效率 |

TRD 不能被解读为无代价的在线增强：其附录主要训练流程使用离线采集与 refinement；1.7B OPD 示例总耗时由 4h40 增至 8h10。文本变短的成本收益也不直接适用于固定步数 diffusion。其部分代码任务 pass@16 仍低于底座，不能称普遍提高能力边界。

其他已阅读的边界材料：[Thinking Machines OPD](https://thinkingmachines.ai/blog/on-policy-distillation/)、[官方 recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)、[Does OPD Really Distill](https://arxiv.org/html/2608.31046v1)、[ThinkOPD](https://arxiv.org/html/2609.37044v1)。低概率 token 的负优势、自反思模式或语言推理轨迹都不能默认对应当前的确定性速度回归。未独立复现这些工作的作者结果。

## 数学上能搬什么

在共享正定协方差的转移模型中，设学生与教师分别为 N(mu_S, Sigma_t)、N(mu_T, Sigma_t)：

\[
D_{KL}(S\|T)=D_{KL}(T\|S)
=\tfrac12(\mu_S-\mu_T)^\top\Sigma_t^{-1}(\mu_S-\mu_T).
\]

教师熵也只由 Sigma_t 决定。因此交换 KL 方向或使用这个熵，无法得到语言模型中基于词分布形状的同一机制。ODE 的零方差极限是退化分布，不能把有限速度 MSE 宣称为普通非退化概率密度的精确 KL。这是针对当前参数化的推导，不是否认其他 diffusion 分布或可学习协方差中的 KL 差异。[DiffusionOPD 原文](https://arxiv.org/html/2605.15055v1)的转移均值匹配也应在其假设下理解。

一个 Euler 段的映射是 F_theta(z)=z+delta_sigma*v_theta(z)。连续两段的参数梯度包含后续状态对前一步输出的导数。原生 OPD 在 detach 的访问状态上逐状态匹配教师；闭环短段损失则对学生自己产生的中间状态传播梯度。区别是学习信号中的轨迹依赖，而非简单把同一局部 MSE 多算几次。

若每个相关状态上教师向量场都能被准确拟合，常规稳定性条件下 OPD 本来也可以接近教师轨迹。短段方法只可能改善有限容量、有限覆盖和有限优化量时的取舍，不能宣称理论上必然更好。

## 本地证据对方向的约束

- Phase68 严格接受仅 51/1024，开发 joint 为验证 8/128、随机 9/128；不能代表完整 OPD 已被公平超越。
- Phase70 验证臂自身 51 状态：behavior joint 1、教师介入 33、最终学生局部 1。说明这些单步目标有可用收益，但尚未兑现；它不支持“单步目标全都无效”。随机臂状态不同，不能用教师 33 对 19 宣称匹配选择优势。
- Phase54 更多拟合使训练修复恢复，但开发仍输随机；提高训练拟合不能代替部署泛化。
- Phase71 恢复大量监督后，base/all/veto/random joint 为 9/5/3/4；广覆盖或负例屏蔽本身也不是已验证的解法。
- Phase73 的同 64 训练 prompt 前向比较：S0/GenEval 局部/完整 joint 为 10/11/22，OCR 局部/完整为 7/7。完整教师不同 CFG、完整采样能力与局部介入都不等于学生提升。OCR 局部预声明机会门槛未过。

以上以 [CURRENT](../experiments/CURRENT.md) 和 [Phase73 解释](../experiments/local_expert_v73_followup/SUMMARY.md)为项目来源。不同 phase 的起点和 cohort 不可直接合并。现有证据不能区分优化量、实现偏离、任务冲突、局部目标参数化和后缀漂移的全部贡献。

## 推荐检验的短轨迹候选

从当前 behavior 实际访问的状态 z_a 开始，冻结起点。教师与学生分别执行相同预声明短噪声区间 [a,b]。第一轮只考虑一个固定两步段，不做长度网格：

\[
z_b^T=\Phi_T^{a:b}(\operatorname{sg}z_a),\quad
z_b^\theta=\Phi_\theta^{a:b}(\operatorname{sg}z_a),\quad
L_{block}=\|z_b^\theta-\operatorname{sg}z_b^T\|^2/d.
\]

sg 表示停止梯度，d 为 latent 元素数。学生内部两步不能 detach；教师和起点停止梯度。该候选是 on-policy 起点加教师短段目标，教师后续状态不能冒称学生 on-policy 状态。主实验应先使用可审计的相同 Euler 日程；多步 solver 需包含历史状态，不能只复制 latent。

先检验不加筛选的 block 学习是否有价值。若成立，再从相同起点比较教师短段与 behavior 短段，并用**同一冻结 behavior 后缀**续完，保持随机噪声耦合，以自动联合约束结果构造非负 detached 权重 w。不可微 OCR 可以用于权重，不需对 OCR 反传；unknown 不是失败真值。组合候选为 L_OPD + lambda*E[w*L_block]，保留原生覆盖，不将大量未验证状态全部丢掉。

这不是最终冻结损失：lambda、归一化、质量保护规则、实际梯度尺度与预算尚待协议。增加辅助损失会改变梯度大小，保持名义 LR 不足以排除优化尺度因素。两步反传会增加显存与计算；更长段未必更容易学。按噪声区间对齐仅是跨步数设计的起点，不能自动保证跨 sampler 泛化。

## 新颖性风险

[RATS](https://arxiv.org/html/2604.14910v1) 已经结合多 horizon 轨迹匹配、相对奖励门控和奖励优化，使用多步 EMA 教师与少步学生。[DanceOPD](https://arxiv.org/html/2606.27377v1) 已包含低噪声单查询和能力路由。因此“多步蒸馏加筛选”或“少查询”本身不够新。

候选需要证明的实质是：外部专家从学生真实中间状态执行短段，在共同学生后缀下验证这个局部动作段，学生再沿自己的状态链学会兑现修正。它与从共同初始噪声出发的全轨迹奖励门控有不同的局部归因条件，但这些差异是否足够构成贡献，仍需直接对照和更完整相关工作审查。不能把不可微奖励、外部教师或联合约束的替换单独当成创新证明。

[Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation](https://arxiv.org/html/2607.24731v1) 的分支失配分析也值得实现审计，但其核心特权负条件设置不是当前共同空负提示的自动等价物。GenEval CFG1 与学生 CFG4.5 本身不证明发生该文所述根因。

## 最小判别实验与停止条件

1. 先按 [现有下一步](../experiments/local_expert_v73_followup/NEXT.md)验收官方固定 commit 的 loss、梯度、Adam、EMA、完整部署权重，并建立足量的同起点任务适配 OPD 对照。不能继续用 64 prompt/16 更新代替强 baseline。
2. 用新训练诊断 cohort、固定起点/噪声/区间比较单步与两步教师介入的联合修复及损伤。不是挑最有利时间点。若短段没有额外有效机会，就不推进这条分支；Phase73 门槛保持原判。
3. 使用同一短段目标库，比较逐状态教师监督与学生闭环段端点监督；控制状态曝光、教师调用、优化步数，并另做计算匹配控制。该阶段先不加入新选择器，诊断本地拟合、共同后缀结果与完整学生开发结果。
4. 只有闭环学习成立，才测试终点验证权重对同量同分层随机/不筛选权重的增益；然后做完整在线 OPD 比较。随机配额相同不代表后续在线状态相同。
5. 全部报 joint/OCR/结构/CER/unknown 和 PickScore、CLIP、美学、HPSv2.1、ImageReward。完整学生是主终点；教师输出、训练修复和局部后缀只是机制证据。
6. 若收益只在训练状态、加算力 OPD 可追平、五质量明显受损或完整学生无增益，停止叠加机制并重新评估。跨 seed、跨 step/sampler 和独立 cohort 是后续资格条件；confirmation512/final768 在方案冻结前仍封存。

工程备选是强 OPD 初始化后做联合任务 NFT/FlowGRPO，对比 OPD 等算力续训、RL 单独训练和联合训练。它可利用离散终端约束，但仍需匹配预算验证；不预设比短轨迹候选成功率更高，也不将其单纯组合包装为充分论文创新。

**讨论结论：保留外部专家 OPD 主线，优先修复学习兑现链；短轨迹闭环监督是条件性第一研究候选。暂不采用熵/KL 改名、再次扩大硬筛选、更多 fit 或自蒸馏作为默认答案。**
