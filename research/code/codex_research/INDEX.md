# Ordered index

| ID | Date | Entry | Status |
|---|---|---|---|
| 0001 | 2026-08-28 | Project audit and recommended next direction | complete |
| 0002 | 2026-08-28 | Constrained-retention OPD smoke, job `1035889` | invalid: init-order bug |
| 0003 | 2026-08-28 | Corrected constrained-retention smoke, job `1035891` | completed |
| 0004 | 2026-08-28 | Constrained-retention smoke result and next gate | valid control-flow result |
| 0005 | 2026-08-28 | Validate direct engine before constraint integration | active decision gate |
| 0006 | 2026-08-29 | Most promising paper method: GuardedOPSD | recommendation; no job submitted |
| 0007 | 2026-08-29 | Claude audit of 0002–0004 | 4 checks pass; **found 10x calibration asymmetry** |
| 0008 | 2026-08-29 | KL-proxy premise check (0006 §7) | necessary condition passes (r≈−0.70); **per-teacher KL not logged → must add first** |
| 0009 | 2026-08-29 | Per-teacher KL logging + branch-step counters | implemented; **unreachable on RTC path (continue at L2535)** |
| 0010 | 2026-08-29 | **固定权重 hinge penalty** 配对实验(w=0/1/3/10) | **作为 fixed-penalty baseline 判负**(n=10:OCR/CLIP 显著更好,Pick/Aes/HPS 显著更差);**⚠️ 非 0006 主方法** —— 三 slot 误用同一 PickScore prompt 池,且实现是 scalarized hinge 而非 QP projection(见 0014) |
| 0011 | 2026-08-29 | Per-teacher capability budget | **INVALID — 前提错误**;`K_max=1` 意味着 0002 本来就是逐 teacher,不存在聚合求和;gather 越界致 device-side assert,两作业 exit=1。测试全绿但测的是错误的心智模型 |
| 0012 | 2026-08-29 | Forward-mode JVP isolation test for RTC-GN | completed; unrelated to GuardedOPSD |
| 0013 | 2026-08-29 | PEFT/partial-parameter forward-mode isolation test | completed; unrelated to GuardedOPSD |
| 0014 | 2026-08-29 | Codex review of Claude GuardedOPSD implementation/results | fixed-penalty result valid but not the 0006 projection; next gate defined |
| 0015A | 2026-08-29 | Baseline and comparison protocol | DiffusionOPD Student = anchor/init; own OPSD-direct = primary paired control; own MOPD-continue = paper baseline |
| 0015B | 2026-08-29 | **正式** CR 配对实验(真实 per-slot 数据集)+ 逐 slot 诊断 | w=0/w=3/w=10 训练与六点评测均完成 |
| 0016 | 2026-08-29 | Gradient-geometry probe + 一阶有效性闸 | 原结论:一阶模型只解释实际约束变化的 8~10% ⇒ 不实现 QP;**该推论范围已被 0023 修正** |
| 0017 | 2026-08-29 | 二维 active-set QP 求解器 | 已过 200 组合成对拍,但按闸门结论**未接入训练**,仅作已验证组件保留 |
| 0018 | 2026-08-29 | CR 正式对照预注册 + 逐次统计量记录 | 三臂完整 6 点定稿;w=10 在当前 trajectory Pareto 上优于 w=3;checkpoint 不作独立 seed |
| 0019 | 2026-08-29 | 一阶有效性探针(patch) | 在真实 optimizer step 处捕获参数增量 u,记 <a_m,u> 与 cos(u,−g) |
| 0020 | 2026-08-29 | AdamW 旋转分解(patch) | 拆开批噪声/retention/预条件三种解释;乘积一致性 0.339×0.153≈0.051 ✓ |
| 0021 | 2026-08-30 | G_fit 损失归因(开放问题) | 混淆已排除(w=1 与 w=0 探针 n=40 匹配),低余弦确为批噪声;剂量-反应设计已写就**未启动** |
| 0022 | 2026-08-30 | **相对官方基线的全景定位** | 原结论:无方法全面超过 Student;缺外部对照臂;**CR 的积极结论已被 0023 重新界定** |
| 0023 | 2026-08-30 | Codex 复核正式 CR 结果、QP 判断与下一步 | fixed hinge 有窄而可信的正结果;checkpoint 不是独立 n;QP 只允许一次 same-state optimizer-aware gate;下轮补 own-run MOPD/KL 基线 |
| 0024A | 2026-08-30 | Self-OPD (arXiv:2608.26872) 新论文复核 | 最新最近邻 baseline;已覆盖 teacher-free branch self-distillation/reward-level fusion;未覆盖 continual capability budgets;公开 mixed release 尚不完整 |
| 0024B | 2026-08-30 | 朴素 KL 外部对照臂 `crkl5` | `abs_tol=-1.0` 使 hinge 恒激活 ⇒ 梯度等价朴素 KL;梯度匹配 w=5;job `1036782` 运行中 |
| 0025 | 2026-08-30 | **Phase-1 same-state probe 预注册 + repair-QP 求解器** | 接受 codex 推翻;新求解器带 repair RHS(300/300 对拍);24 个 probe;go/no-go 已冻结 |
| 0026 | 2026-08-30 | Phase-1 probe 实现与重试 | `1036848` GradScaler 失败、`1036866` graph 累积 OOM;两处真因已修;`1036869` 等待资源 |
| 0027 | 2026-08-30 | **RG-OPSD：post-fit optimizer-candidate selection** | 具体方法、one-step cached certificate、最小三臂与 go/no-go 已写;本条未提交新任务 |
| 0028a `_METHOD_GENEALOGY_AND_STARTING_POINTS` | 2026-08-30 | 方法谱系、论文 baseline 与初始化协议澄清 | DiffusionOPSD/Self-OPD 原论文均主要从 SD3.5-M base 开始;本项目为 continual comparison 改用同一 released DiffusionOPD Student 起点 |
| 0028b `_SAMEIMAGE_EVAL_PREREG` | 2026-08-30 | same-image 评测 | 在同一批 task images 上同时算任务指标与 Pick/HPS;缺 StudentSI 对照,暂不解读 |
| 0029 | 2026-08-30 | 基线对照表脚本 | 五臂 × 六轴,数据未齐时主动警告 |
| 0030 | 2026-08-30 | **五臂基线轨迹** | 论文方法基线跑通;HPS 损失归因于 reward 优化本身;hinge 优于朴素 KL(后者 pick 为负) |
| 0031 | 2026-08-30 | Phase-1 对 RG-OPSD 的输入 | **AP 候选不加入**(QP 0/8 恒等);portfolio 冻结为 A0/AD/AG;Phase-1 反向支持 RG-OPSD 核心论点 |
| 0032 | 2026-08-30 | 两篇论文的复现状态 | DiffusionOPD 8/8 列;DiffusionOPSD 仅 1/5 列(其余缺 reward 专训臂);Aes 12.08 红旗撤销 |
| 0033 | 2026-08-30 | **OPSD Open3 对比的结构** | 论文两套口径(reward-specific 主表 vs Open3 对比);~~Open3 无 published 数值~~ **已更正(§⑭):数值在正文 experiments.tex L46 —— Open3(300 upd) Pick 25.51/CLIP 0.333/HPS 0.389**;DiffusionOPD 是 Open3 对比里的 baseline |
| 0034 | 2026-08-30 | ~~能力代价~~ → **CFG headroom 迁移**(原主张已推翻) | gs=4.5 下 OCR 似乎下降,但 **gs=1.0(训练匹配口径)下三轴全升且单调增强**(ck100 OCR t=+10.97);CFG 增益 base +0.4436 → ck20 +0.2111 → ck100 +0.1203 ⇒ OPSD 把能力从"靠 CFG"迁移为"内建";**方法学推论:高 CFG 评测会系统性低估 CFG-free 训练的模型** |
| 0035 | 2026-08-30 | continuation 是否有效的最终区分 | MOPD-continue≈不动;direct reward continuation 强但偏科;hinge 有窄正结果但未解决 held-out HPS/CLIP;当前无全轴赢家 |
| 0036 | 2026-08-30 | **RPI-OPSD：从 diffusion/flow 与 LLM OPSD 得出的统一方向** | 对普通 continuation 信心低;保留一次 selective reliable Pareto-improvement gate;target-level hard veto + positive-only distillation + general clean-view replay + post-fit exact commit |
| 0037 | 2026-08-30 | **baseline 忠实度:问题确认与两条路线** | `crd` 是我们重写的 RTC-raw,非官方 OPSD(margin/缩放参数化/opa_mb 三处偏离);Student 锚点与"支配朴素 KL"不受影响,"把损害压到 −2.0%"受影响;建议路线 B(校准目标函数)而非 A(移植 guard 进官方仓库——官方无 teacher 会使 hinge 退化成阈值 KL) |
| 0040 | 2026-08-30 | **Gate A 预注册:底座决定 + 向量可行性** | 底座定为**官方 DiffusionOPSD**(唯一有确认复现);主线改 from-base;预注册 VFR 判据 **≥0.90 则杀掉 A1**,禁止放宽容差复活;A2(post-fit gate)形式由 Phase-1 实测反向确定 |
| 0041 | 2026-08-30 | **实验口径章程 + 参考仓库审计** | 发现参考仓库被我们改过(+201 行);拉原版 `9c4c48c` 逐 hunk 比对 ⇒ **默认值下行为完全一致**(已验证);落地三项机制:原版固化只读、`opsd_manifest_diff.py` 未声明差异即报错、`OPSD_WF_MODE` 启动侧中和;种子政策分离(复现跑不固定/A-B 跑配对固定) |
| 0042 | 2026-08-30 | **A2 规格:post-fit exact commit gate** | 挂在 pristine L1589 优化器步外;每 epoch 仅 1 次验收(1 step/epoch);同状态精确重算 + 回溯 α∈{0.5,0.25,0.125};形式由 Phase-1 反向确定(一阶投影 0/8 无效、精确重算+回溯 8/8 有效);预注册 kill:no-op 率 ≥70% 或 ≤5% 均判死,不得放宽 τ 复活 |
| 0043 | 2026-08-30 | **Gate A 结果** | **VFR=0.9935**(2289/2304)⇒ A1 现形态在 objective 内无立足点(初步判定,非判死);三轴 P(Δ<0)=0.001~0.005、轴间相关全正(hps-pick +0.715)⇒ **Open3 的三 reward 在目标构造层面不冲突**;成因对照:轴选错=强支持、容差=排除、时点=待中期探针、rank 采样=部分开放 |
| 0044 | 2026-08-30 | **复现状态 → 判为可行** | pickscore +1.1% ✅;**clipscore ck50 = 0.3105 vs 论文 0.3122,在其 ±0.0050 误差棒内** ✅;ck100 −6.3% 但仍在上升;论文正文明写专家预算就是 100 updates ⇒ 步数非原因;**判定:趋势全符、早期点精确吻合 ⇒ 复现可行**,后半程增益偏低记为已知偏差不阻塞 |
| 0045 | 2026-08-30 | **Open3 baseline 表征** | 一个 checkpoint 在 Pick/CLIP/HPS **三列同时超论文 ReFL 与 NFT**(后者每列都是专训);与论文 OPSD 差 −0.7%/−6.1%/−4.8%,**CLIP 的 −6.1% 与独立 clipscore 臂的 −6.3% 同源** ⇒ 指向跨臂共同实现原因(world_size 3 vs 8);Aes 下滑减速(−13.2 → −2.31 ×2σ)且始终高于 base;ck100 仍在上升,A/B 有动态范围 |
| 0046 | 2026-08-30 | **独立证实实现缺口** | 用 Gate A 探针的构造增益 vs 同一次训练的实现增益:实现率 clip 1.04% / hps 3.22% / pick 2.04%,**夹住论文报的 1.31%** ⇒ 97~99% 构造增益在拟合阶段丢失;澄清 RTC 的功过:**问题陈述正确、GN 修法证伪**;A2 的逻辑链每环均有实测支撑 |
| 0048 | 2026-08-31 | **N1 结果:判负** | 最低保留率 **87.9% → 75.6%**(极差 9.3→19.5pp);pick +8.40×2σ 但 clip −7.81、hps −9.57;静态 diagonal-response 权重有害；**0050 更正：+0.715 是 reward-gain correlation，不是 gradient Gram，`G w` 机制尚未实测** |
| 0038 | 2026-08-30 | **官方代码底座与 baseline 决定** | 主 fork Self-OPD（原生 K-branch/warm LoRA）；DiffusionOPSD 保持独立 baseline 并复用 reward/eval；主表同起点比较 Frozen Student、DiffusionOPSD-cont、Self-OPD-cont、RPI-OPSD；Self-OPD 当前仅内置 OCR reward，mixed 需明确标注我们的 adapters |
| 0039 | 2026-08-30 | **方向更正：from-base 应为主线** | continuation 是历史路径依赖且 strong Student 局部 Pareto 空间可能过窄；主表改为 SD3.5-base 上 DiffusionOPSD-Open3 vs Self-OPD-mixed vs Pareto Self-OPD/RPI；Student continuation 降为方法成功后的 stress test |
| 0050 | 2026-08-31 | **clean DiffusionOPSD 审计 + idea map** | baseline harness 可信；A1 机制判空、N1 判负、N2 有行程但解释需收窄；更正 gain-correlation≠gradient Gram；首推 **LSR-OPSD：state-adaptive local-specialist target portfolio**，先做零训练 Gate P |
| 0051 | 2026-08-31 | Gate P / LSR 零训练预注册 | 已升级为 rollout-endpoint v2；held；top-k 结果使先验下调 |
| 0052 | 2026-08-31 | A1 held-out 结果 | 五轴全打平；veto 仅触发 1.39%，封存 |
| 0053 | 2026-08-31 | clean construction 与 rollout gap | cert rate 随训练下降；提出无干预 cert probe |
| 0054 | 2026-08-31 | cert probe 预注册 | 只测不门控；区分内在退化与门控伪像 |
| 0055 | 2026-08-31 | cert probe 结果 | H1 成立；local→suffix retention 63.4%→43.0%；高 reward/up-direction 特异退化 |
| 0056 | 2026-08-31 | N2 held-out 结果 | 五轴全打平；每步生效仍无收益，冻结 normalizer 假设判负 |
| 0057 | 2026-08-31 | A3 official `opa_cert=1` 预注册 | same-seed Open3 A/B 已提交；旧 Pick 结果提示批量缩减/过优化风险 |
| 0058 | 2026-08-31 | **第一性原理重置 + Q-OPSD** | 现有臂都保留 local-clean proxy；首推对 exact suffix action value 求梯度的 Q-OPSD；先做 Q1/Q2 零训练 gate；本轮无新 job |
| 0059 | 2026-08-31 | Gate Q1 预注册 | 固定半径比较 clean/Q/random suffix directions；阈值先于结果冻结 |
| 0060 | 2026-09-01 | 递归诊断 | 汇总 local target、suffix dynamics、参数拟合和 prompt noise 四层证据 |
| 0061 | 2026-09-01 | Gate Q1/Q1b 结果 | 固定半径 Q1 尾部未过；逐轴 backtracking 的 Q1b 四条判据全过 |
| 0062 | 2026-09-01 | Gate Q2 预注册 | 检查 Q target 到共享 LoRA 参数更新的相容性 |
| 0063 | 2026-09-01 | Gate Q2a 结果 | 单独显著性比较的原判词后被撤回；paired Q-vs-canonical coherence 差异不显著 |
| 0064 | 2026-09-01 | Gate B noise-scale 预注册 | 用 prompt-disjoint half-batch 估计 critical batch 与 realization rate |
| 0065 | 2026-09-01 | realization-rate 预注册 | 冻结实现率判据与可证伪范围 |
| 0066 | 2026-09-01 | Gate B 结果 | B_noise 中位约 740、双种子一致；固定 4x batch 缺少探针支持 |
| 0067 | 2026-09-01 | reachability time-series 预注册 | 测 reward direction 的参数可达性是否随训练收缩 |
| 0068 | 2026-09-01 | reachability 结果 | 未发现随训练收缩；训练期 reward-specific reachability 缺口未获支持 |
| 0069 | 2026-09-02 | **Q-OPSD 实现与结果审计** | DrawBench 正式负结果成立；发现 scalar/per-axis 偏差与镜像 loss 使 advantage 精确消失；先做 matched 2x2 eval，再决定 Q-OPSD-Core |
| 0070 | 2026-09-04 | Claude 的 Q vs baseline 完整汇总 | Q-Core matched DPM2-10 强正、official flow-40 负；提出 sampler flattening、2.8× 与 solver-ceiling 假说，结论待 0071 审计 |
| 0071 | 2026-09-04 | **Claude 最新 Q-Core 代码与结果复核** | 核心两步 suffix-Q 实现成立；matched 正结果可靠；更正 sampler-independent/99%/70%/ceiling/57% 回收等过强论断；给出 default 曲线与 state×suffix probe 闸门 |
| 0072 | 2026-09-04 | **Q 主线资格与增强路线** | step curve 是 go/no-go 而非既定主线；依次解决 solver/value 不一致、Q-consistent advantage、低 realization；提出 direct suffix value-gradient distillation |
| 0073 | 2026-09-04 | **Claude 对 0071/0072 的更正回应** | 接受九项更正并补 default/EMA step curve；后续“4×”表述仍需同 NFE flow baseline 与 equivalence 标准约束 |
| 0074 | 2026-09-10 | **Claude 最新全部实验、结果与 setting 清账审计** | 强正结果是 LR4×/8× tuned baseline；Q 被同 NFE flow baseline 支配；组合/探针未形成新方法；发现 DrawBench 选择污染与脚本 hash/provenance 缺口；本轮无 job |

## Current invariant

No job is considered a test of constrained retention until its log confirms all of:

- initialization from the released DiffusionOPD Student LoRA;
- PickScore reward loss is applied only to the PickScore task slot;
- OCR and GenEval slots use their own routed teachers;
- retention compares current-student KL with frozen-initial-student KL on the same
  current on-policy state;
- the one-sided gate, direct loss, retention loss, active rate, and both gradient norms
  are finite and nonzero where expected;
- optimizer/global-step/checkpoint progression is observed.
