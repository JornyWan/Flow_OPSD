# research/ —— DiffusionOPSD / MSP 研究全档案(给接手的 agent)

> 快照时间:**2026-10-11 23:45 EDT**。仓库根目录仍是给合作者的**干净版**(官方 DiffusionOPSD `9c4c48c` + 只有 MSP,
> 见根 `README.md`);本目录是到快照为止的**全部** idea、代码、实验记录、结果与结论。旧的干净版另有 tag `v0.1-msp-release`。
> 记录语言以中文为主;节号 `§4.xxx` 一律指 `docs/EXPERIMENT_LOG.md` 里的节。

## 0. 先读什么(按顺序)

1. **本文件**:目标、协议、每个 idea 的结论、当前状态、代码怎么用。
2. **`docs/IDEAS_TRACKER.md`**:idea 与实验总表(§0 = 快照时在跑/排队的东西;§1–§4 = 各 idea 状态与判定;节号可跳进日志)。
3. **`agent_memory/`**:前一个 agent 的长期记忆(`MEMORY.md` 是索引)—— 大多是**踩过的坑与方法论规矩**,每条有 Why / How to apply。强烈建议全读。
4. **`docs/EXPERIMENT_LOG.md`**(2.4 MB,~46k 行,唯一主记录):按需用节号 grep。重要的综述节:
   - §4.864 DiffusionOPSD 让学生"持续又快速"提升的本质
   - §4.873 不必胜过 OPSD 时,哪些 idea 相对底模有稳定收益
   - §4.878 MSP 掉点的诚实盘点(已确立 / 被否 / 最佳假设)
   - §4.892 等效提前步数 Δ(c) 分析;§4.893 净位移标定;§4.895–4.897 最终交付与 split 筛查
5. `docs/PAPER_PLAN.md` 与 `paper/msp_draft/`:MSP 论文计划与 LaTeX 草稿(含 PDF)。

## 1. 项目目标(用户定,2026-10-11)

- **首要目标**:在 **40 步采样(flow-40)下、每个 checkpoint 都**胜过 canonical DiffusionOPSD;少步(flow-4)只作副读数。
- **最终交付**:
  1. **multi-reward(open3)300 步** 胜 canonical —— H200×4、np4,对现成 `o3canon300`(s0830)/ `canon300s2`(s0905)。
  2. **single-reward 100 步** 胜 canonical —— np4,对现成 `rs{reward}canon`(s0830)。
- **canonical 已复现完,不再跑 canonical**:只和现成 canonical 比,要求**同硬件型号、同 np、同 seed**。
- 用户明确否定"调 lr 类"臂(lr×k 是步长收益,不是方法收益):要试**没跑过但可能 work** 的东西。

## 2. 复现结论(canonical 与论文对齐)

DrawBench 1000 prompts、flow-40、gs 1.0、512²、eval seed 42:

| | Pick | CLIP | HPSv2.1 | ImgR |
|---|---|---|---|---|
| 论文 open3 300 updates | 25.51 | 0.333 | 0.389 | 1.51 |
| 我们 `o3canon300` ck300(s0830,4 卡)| 25.46 | 0.332 | 0.388 | 1.51 |
| 我们 `canon300s2` ck300(s0905,4 卡)| 25.51 | 0.332 | 0.390 | 1.52 |
| 论文单 reward 专家 100 updates | 24.94 | 0.340 | 0.390 | 1.76 |
| 我们单 reward canonical ck100(各自训练轴,4 卡)| 24.96 | **0.320** | 0.392 | 1.756 |

- open3 300 步全部吻合;单 reward 四列三列吻合,**CLIPScore 专家低 5.8%**(已知缺口,ck50 时与论文一致,后半段涨得慢;机制未查明)。
- **卡数**:论文 8 卡;我们最终口径 4 卡、筛查 3 卡。`config/public.py` 固定**每次更新 48 prompt × 24 图 = 1152 样本**
  (8 卡 bs9×gacc16、4 卡 bs6×gacc48、3 卡 bs8×gacc48)⇒ 有效批、lr、更新次数与论文相同,卡数只改微批切分。
  7 个 canonical(3 种 GPU × np3/np4 × 2 seed)ck100 的 PickScore 极差 0.02。**但跨硬件/跨 np 仍有 ≤0.7×MDE 的偏差 ⇒ 臂间对比必须同硬件同 np。**
- 配置:`sd35_open3`(PickScore+CLIP+HPSv2,**β=0.1**,论文 300 updates);单 reward `sd35_{pickscore,hpsv2,clipscore,imagereward}`(**β=1.0**,100 updates)。

## 3. 评测与判定协议(所有结论都基于它)

- **配对逐图评测**:同一 1000 个 prompt、同 eval seed,两臂逐图配对差。两个集:`drawbench`、`unseen1k`;两种步数:flow-40、flow-4。
  评测统一在 **A40** 上跑(评测硬件不同会差 ~0.001)。产物 `matched_pi_<TAG>_<CK>_default_<set>_flow<N>.json`(逐图 5 个 reward)。
- **open3 复合分** = PickScore/26 + CLIPScore + HPSv2.1;**MDE = 0.0033**(28 对 canonical-vs-canonical 标定,中位 0.39×MDE、最大 1.23×MDE)。
- **显著**:单 seed **≥2.5×MDE 且 |t|>2**;**<1×MDE 为无效应**。格判定:P(两集都显著正)/ P−(一正无负)/ 0 / N(任一显著负)。
- **single-reward(R 轴,§4.709)**:判在训练所用 reward 的配对 t 上:R1 t>7.5 且胜率>55% / R2 |t|<5.5 / R3 t<−7.5 / R4 其余。
- **held-out 轴 = ImageReward**(与 open3 增益 r=+0.894、12/12 同号);aesthetic **不是** held-out 轴(r=−0.395)。
- **配对三轴必须一致**:训练硬件型号 / 评测硬件 / 外层 epoch(ck 编号);seed 也要同。逐图模式由 seed 主导,跨臂比逐图必须跨 seed。
- **规矩**:判定规则**数据前写定**(`code/codex_research/judge_*.py` 的 docstring 写明写定时间);预登记的格必须穷尽(加"皆非"格、
  以 0 为界先切 |Δ|≤MDE 的无效应格);**首轮为负先归因**(实现 / 口径 / 机制)再判负;先打印原始逐点序列再看汇总量。

## 4. Idea 与结论总览

### 4.1 主方法 MSP(multi-state propagation,`multikF4`,K={3,6},σ 0.858/0.602)
canonical OPSD 在单个 query 态上做蒸馏回归;MSP 把同一个 x0 空间目标 ȳ± 复用到同一 rollout 的两个高噪态(无额外 reward/decode)。
- ✅ ck100:flow-40 **+3.3~5.6×MDE**、flow-4 **+8~11×MDE**,两 seed 与跨运行复现(§4.632–4.656、§4.736)。
- ✅ 族形状:K={3,6} 内部最优;{5,6,7}、高噪、随机抽 3 等变体与 MSP 无差或更差(§4.696/4.834/4.846/4.877)。
- 单 reward(R 轴):imagereward R1;pickscore **R3 负**;clipscore / hpsv2 R2 无效(§4.719–4.779)。⚠ 单 reward 用 β=1.0,而 β×MSP 在 β=1.0 下转强负(§4.767)⇒ 单 reward 为零不能读成"MSP 需要多 reward"。
- **300 步问题**:seed1 ck200 后崩(ck300 −9.3×MDE f40);seed2 不崩,但领先从 ck100 +2.8 缩到 ck300 +1.1。
- **§4.892 Δ(c) 分析**:40 步上 MSP ≈ canonical 沿同一轨迹**固定提前 ~10–40 步**(不是速率增益),canonical 斜率衰减 ⇒ 换算成 MDE 越来越小;
  4 步上提前量随训练增长(+78 → +269 步)= 真速率增益。⇒ 40 步每 ck 都显著需要**速率**增益。

### 4.2 MSP 后期掉点的机制(§4.874–4.885)
- Adam 有效步长突发只出现在 MSP 系;额外态梯度 = 好/坏样本两半拔河的**平衡残差**(cos≈−0.97)。
- **MSP 特有失效 = 两半"抵消失效"**(cos −0.98 → ≈0),正常训练里自然发生,损害随失效持续时间增大(§4.882)。振荡本身 canonical 也有,不是原因。
- 额外态梯度进同一个 Adam 的 v̂ ⇒ 主项有效步长早期 ≈0.46×、突发后长期被压 —— 这是 split 的动机。
- 已否的修复:退火关 MSP(A4)、checkpoint soup、步长上限、额外态归一化 detach、D/C 拆分(只留 reward 方向发散 / 只留一致性 −10×MDE)、
  周期 EMA 重启(防崩但增益小、对 canonical 有害)。⚠ 续训 = EMA 重启,对 MSP 产生 +11~19×MDE 伪效应 ⇒ **MSP 系 run 不能断点续训**。

### 4.3 当前候选(快照时)
- 🟢 **split = 额外态独立 Adam(`multikF4sa`,`RPI_MK_SPLIT=λ`,λ=0.5)**:主优化器只吃主项梯度(= canonical 的 Adam 动力学),额外态走另一套 m/v。
  - λ=0.5 的净 LoRA 位移 = canonical(1.00–1.01×)⇒ **不是步长收益**;λ=1.0 为 1.22–1.34×,其 40 步领先与 canonical lr×2 打平 ⇒ 维持 λ=0.5(§4.891/4.893)。
  - ck50 两 seed:40 步对 canonical P/P、对 MSP P−/P−。
  - 稳定性代理(`judge_sa_late.py` 第三段):H100 s0830 在 ep100–148 内 r>0.75 最长连续 **4 步**(未门控 MSP 在 ep120–146 为 18 步)—— 单 seed、只是代理。
  - **ck100 s0830(H100 np3)**:对 canonical **+5.85 / +4.67(t≈16)P**,对 MSP +0.99 / +2.03(0,无负);held-out IR 四格正(§4.897)。
  - s0905(A100 np3)ck100 drawbench 已出:对 canonical f40 **+5.87(t+15.0)**、对 MSP +1.97(t+6.3);unseen1k 评测中。
  - 待:s0905 ck100 两集齐 ⇒ 按 `judge_sa888.py` 晋级 ⇒ H200×4 跑 300 步(`session_scripts/submit_h2sa300.sh`)。
  - 单 reward split(一个 seed,np4,对现成 `rs*canon`):pickscore / hpsv2 在 H100 NVL×4(preempt,被抢占即从零重投 —— 不可续训);clipscore / imagereward 需 H200×4。判定器 `judge_rs_sa.py`。
- ✅ **门控 `multikF4gate`(`RPI_MK_GATE=0.75`)**:失衡(r=od_gn_x/od_gn_q>0.75 的连续步数)18 → 0;ck100 无代价;ck150 40 步 +5.5/+3.6、少步 −4.6/−5.2(取舍)。单 seed,300 步 H200 排队中(`judge_g300.py`)。

### 4.4 已关闭方向(有判定;详见 `docs/IDEAS_TRACKER.md` §3–§4)
| 方向 | 结论 |
|---|---|
| TPD(倾斜后验蒸馏)| ❌ 估计量噪声主导(两组独立续采夹角 f≈0.002;噪声 ‖·‖ 25–33 ≫ 信号)⇒ 机制性负(§4.889)|
| `vadv`(已验证增益加权)| ❌ N;归因 = w₋ 饱和、w₊ 衰减 ⇒ 实为给高优势样本降权(§4.890)|
| RB-LT / LT-OPSD(残差桥 / 直接迁移)| ❌ 全部发散:反馈系数 κ>0(§4.804–4.840)|
| L1 / suffix-aware 局部目标(`qlook1*`、`canonsfx`)| ❌ 先涨后跌;L1 只是相对 canonical 更好且状态依赖翻转(§4.846–4.863)|
| Q-core / Q-oracle | ❌ 机制性负(§4.599/4.609)|
| C/M 去噪前后缀交换(P15)| ➖ 4 步 S2、40 步 S0 ⇒ 不进第二步(§4.881)|
| 策略空间搜索 + on-policy 蒸馏(PSD)| ⏸ PS1 不进 PS2(§4.871)|
| 搜索预算分配 / latent repair(agentic)| ❌ 天花板不成立(§4.583–4.591)|
| 逐 reward 梯度冲突 | C2 无冲突(§4.604)|
| 早期 CA-OPD(conflict-averse 多教师,DiffusionOPD 底座)| 见 `related/CA-OPD/` 与日志 §1–§4.1xx;N2 判负、Gate Q2a 不通过(目标侧优化到头)|
| lr / inner / gacc 旋钮 | 是步长/拟合端收益,不算方法;拟合端收益持续流失、步长收益回落后守住(agent_memory/opsd-fitting-vs-stepsize-decay.md)|

### 4.5 理论 / 机制性结论(可直接复用)
- **反馈系数 κ**(agent_memory/opsd-target-vold-gain-must-be-one.md):残差对学生输出的依赖 —— canonical κ=0;MSP 额外态 κ<0 稳定;**κ>0 必发散**(LT 系实例)。κ≤0 必要不充分。
- **MSP 额外态梯度是平衡残差**:好/坏两半 cos≈−0.97 相互抵消;删掉一半(ADVPOS)⇒ ×3.3 同向大梯度。改额外态加权先看 `RPI_OPT_DIAG=2` 分解。
- **40 步收益 = 固定提前量,4 步收益 = 速率增益**(§4.892);新候选要报 Δ(c)。
- **步长 vs 方法**:净 LoRA 位移比 ∈[0.95,1.10] 才算步长匹配(§4.893);λ、lr、inner 都会改位移。
- **β 混淆**:单 reward 官方 β=1.0、open3 β=0.1。
- **一步镜像退化 bug**:一步镜像使 pos_loss≡neg_loss、advantage 的 r1 贡献为零 —— 查**差值**不查量级。
- **100 updates 未收敛**(ck50→ck100 仍 +5×MDE)⇒ 所有 ck100 判定只是"任意早停点比较"。
- **训练指标不是 held-out 代理**(臂间比较可反号;同臂内部变化追踪很好);**权重范数不是分数代理**。
- **噪声地板**:canonical 两两差中位 0.39×MDE、最大 1.23×MDE ⇒ 单 seed 阈 2.5×MDE。
- **探针步长要按 Adam 标定**(步长≈lr·√N,与 ‖g‖ 无关);**autocast 缓存吃掉原地改权重**(扰动探针用 `cache_enabled=False`)。

## 5. 代码

### 5.1 研究 fork = 根目录 + 3 个文件的覆盖层
```bash
# 在仓库根目录执行:得到与研究所用 DiffusionOPSD-rpi 逐字节相同的代码(已用 diff -rq 验证,只差 README/.gitignore)
cp -r research/code/fork_overlay/scripts research/code/fork_overlay/config .
```
- `scripts/train_opsd_ri_sd3.py`(3704 行,所有 idea 的开关;**全部默认关闭 = canonical**)、`config/public.py`、`config/mixed.py`。
- ⚠ 开关名与根目录干净版不同:fork 用 `RPI_SEED`、`RPI_MULTIK_STATES`;干净版用 `OPSD_SEED`、`MSP_STATES`。
- 包导入:研究时用 `PYTHONPATH=<shim>:<repo>`,其中 `<shim>/diffusionopsd -> <repo>/src`(软链);按上游 README `pip install -e .` 也可。

主要开关(由启动脚本按 `ARM` 设置,见 `code/slurm/ab_open3_v6.slurm` 的 `case "$ARM"`):
| ARM | 环境变量 | 含义 |
|---|---|---|
| `baseline` | `RPI_A1=0` | canonical OPSD |
| `multikF4` | `RPI_MULTIK_STATES=3,6 RPI_MULTIK_WEIGHT=uniform` | MSP 主臂 |
| `multikF4sa` | + `RPI_MK_SPLIT=0.5` | **split**:额外态独立 Adam,λ=MK_SPLIT |
| `multikF4gate` | + `RPI_MK_GATE=0.75` | 门控(两半失衡时关额外态)|
| `multikF4ema` | + `RPI_EMA_RESTART=50` | 周期 EMA 重启 |
| — | `RPI_OPT_DIAG=1/2` | 优化器诊断(`od_gn_x`、`od_gn_q`、`od_cos_xx` 等写进 metrics.jsonl)|
其余开关(`RPI_TPD*`、`RPI_VADV`、`RPI_QCORE*`、`RPI_RBLT`、`RPI_LTOPSD`、`RPI_SELECT_*`、`RPI_KFIT*`、`RPI_A1*`、`RPI_N2*` …)对应 §4.4 已关闭方向。

### 5.2 启动 / 评测 / 判定
- **训练启动**:`code/slurm/ab_open3_v6.slurm`,接口 `ARM SEED UPDATES OUTPUT_DIR AB_CONFIG(默认 config/public.py:sd35_open3) MK_SPLIT OPT_DIAG EXPECT_NPROC`。
  只接受 nproc ∈ {3,4,6,8};有显存守卫(<80GB 卡不行:48GB 卡加载 reward 模型即 OOM);按 OUTPUT_DIR 加锁、ckpt-final 存在即退出;被 requeue 时自动续训(**MSP 系续训会污染**)。
  训练前跑 `code/analysis/preflight_check.py`(仓库里不能留 .bak 等临时文件,否则 fork-clean 检查失败 exit 42)。
- **评测**:`code/codex_research/watch_matched_ckpt.sh`(等 ckpt 出现)→ `code/slurm/opsd_eval_matched.slurm` → 根目录 `scripts/cross_eval.py`(A40)。
  watcher 用 `code/codex_research/list_watchers.sh` 管理(按 cmdline token 位置匹配,避免 pkill 误杀自己)。
- **判定**:`code/codex_research/judge_*.py`(每个对应一个预登记判定);共用 `code/session_scripts/judge.py`(`paired()`、`load()`、MDE)。
  ⚠ 这些脚本里写死了旧集群路径(`/midtier/...`,包括指向 judge.py 的 scratchpad 路径)—— 新环境需改 `SP=` / `OPD_ROOT=`。
- 其它:`code/probes/`(p1–p15 机制探针)、`code/analysis/`(早期分析脚本)、`code/session_scripts/`(前一个 agent 的全部临时脚本,含 `preempt_guard.sh`、`submit_h2sa300.sh`、`xshift.py`(Δ(c))、`disp.py`(净位移))。

## 6. 结果数据
- `results/eval_summary.csv`:全部 1195 个评测 json 的均值与 SE(5 个 reward),含 `tag → run_dir` 映射(如 `sah3s30 → sa_h3_s30`)。
- `results/eval_per_image_json.tar.xz`(44 MB,解压 ~170 MB):**逐图**评测 json(配对检验需要它);`tar -xJf` 后放到 `outputs/` 即可被 judge 脚本读取。
- `results/train_metrics_and_configs.tar.xz`:每个训练 run 的 `metrics.jsonl`(reward 曲线、优化器诊断)与 `run_config.json`。
- `results/codex_runs_summaries.tar.xz`:早期 codex 线(0002–0027)的 run 摘要;`results/watch_logs/`:部分 watcher 日志。
- **不在仓库里**:LoRA checkpoints(数百 GB)、reward 权重、生成图片、wandb。

## 7. 目录
```
research/
  README.md               本文件
  docs/                   EXPERIMENT_LOG.md(主记录)、IDEAS_TRACKER.md、PAPER_PLAN.md、早期 PLAN/SUMMARY/OVERNIGHT/RESULTS/REPRODUCTION_VERDICT
  agent_memory/           前一个 agent 的长期记忆(经验教训 + 项目事实),MEMORY.md 为索引
  code/fork_overlay/      研究 fork 的 3 个文件(见 §5.1)
  code/slurm/             训练/评测/探针的 SLURM 脚本(ab_open3_v6.slurm 为现行启动器)
  code/codex_research/    编号设计文档 00XX_*.md(codex 早期线)、judge_*.py、watcher、check_*.py
  code/probes/            机制探针 p1–p15
  code/analysis/          早期分析脚本、preflight_check.py
  code/session_scripts/   前一个 agent 的工作脚本(judge.py 在这里)
  results/                见 §6
  paper/msp_draft/        MSP 论文 LaTeX 草稿 + PDF;paper/figs/ 图与作图脚本
  related/CA-OPD/         早期 CA-OPD(conflict-averse 多教师 OPD,DiffusionOPD 底座):README、日志、论文、eval、代码
  related/opd_lab/        另一条由 codex 维护的 OPD Lab 线(原生 DiffusionOPD 上的 local-expert / 两步学习系列,Phase 1–81)的代码与实验摘要
```

## 8. 快照时的待办(接手者可从这里继续)
1. 读 s0905 A100 的 split ck100(`judge_sa888.py`)⇒ 晋级则上 H200×4 300 步,两 seed,对 `o3canon300` / `canon300s2`(`judge_sa_late.py` 末段)。
2. split ck150–300 早读(H100 s0830 那条 UPDATES=300,对 H200np4 canonical,阈 3.5×MDE)。
3. 单 reward split 四个 reward(`judge_rs_sa.py`;主判定 ck100 flow-40 两集都 R1)。
4. 门控 300 步(`judge_g300.py`)。
5. 若 split 300 步仍出现"领先被吃掉",下一步看 Δ(c) 与稳定性 Lr(`judge_sa_late.py` 第三段)。
