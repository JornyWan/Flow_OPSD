#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# A/B 主实验:官方 Open3 目标 vs 我们的改动。100 optimizer updates(用户 2026-08-30 定)。
# **两臂共用同一 fork、同一 config、同一数据、同一固定种子**,只差 RPI_A1 这一个 flag。
# 口径见 codex_research/0041_CLEAN_EXPERIMENT_CHARTER.md §⑤;提交前须过 opsd_manifest_diff.py。
#
# **不是 smoke**:完整 pickapic 25415 条、1152 samples/epoch、β=0.1、gs=1.0、512²,
# 只是把 optimizer updates 定在 100(与三个 reward-specific 专家同预算)。
set -uo pipefail
OPA_CERT_ARG=""
EXTRA_CFG=""      # 逐臂追加的 --config.* 覆盖(与 OPA_CERT_ARG 分开,便于共存)
# checkpoint 编号 = 外层 epoch × num_inner_epochs(global_step 在内层循环自增,L2275)。
# 完成检查必须乘上这个系数,否则 inner 系列臂会在跑到 1/N 处被误判『已完成』而退出。
CK_MULT=1
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$OPD_ROOT/CA-OPD/code/DiffusionOPSD-rpi
SHIM=$OPD_ROOT/CA-OPD/code/shim_rpi
PY=$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python
ARM=${ARM:?必须指定 ARM=baseline|a1|n1|n2|gatep|certp|a3|q1|q1b|q2a|bnoise|bnoisep|rtcreach|qopsd|b4x|a1v2|qbt|inner2|inner4|lr2x|lr4x|qcore|qcorelr|kfit|lrhalf|lr16x|lr12x|lr10x|lr12xeps|qsig|qstrain|opacfg|lr4xinner2|lr4xgacc2|lr2xinner2|inner2gacc2|lr8xinner2|inner2lrhalf|lr6x|multik4|multik4s|multik2|multikF4|vadv|multiklr|selhi|selhitr|sello|hybrid|hybgrad|hybridw|hybridtr|baseA4|multikALL|multikHI|multik4H|multikS3|multikSY|base4|base6|base6mk|qs7mk|multikF4cfg|poseuler|poseulerrot|rblt|rbltd0|rbltpcfm|rbltbridge|rbltrand|ltopsd|ltopsd0|qseuler|multikF4pos|qlook1|qlook1fz|canonsfx|qlook1sfx|qlook1fzsfx|multikF4ema|canonema|mkF4h|mkdirF4|mkconsF4|tpdF4|multikF4cap|multikF4wfd|multikF4gate|hybridtr36|hybridw36}
# N1(尺度均衡加权,见 0047)走 mixed.py —— 已逐字段核对与 public.py:sd35_open3
# 在单位权重下等价(72 字段仅 3 个不同,且都不影响计算)。
AB_CONFIG=${AB_CONFIG:-config/public.py:sd35_open3}
MIXW=${MIXW:-}
SEED=${SEED:?必须指定 SEED(两臂必须相同)}
UPDATES=${UPDATES:-100}
OUTPUT_DIR=${OUTPUT_DIR:-$OPD_ROOT/outputs/ab_open3_${ARM}_s${SEED}_u${UPDATES}}
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env" 2>/dev/null
cd "$R" || exit 1
NPROC_REQ=${NPROC:-}          # 调用方期望的卡数(下一行会用实测值覆盖 NPROC)
NPROC=$(nvidia-smi -L | wc -l)
# ⚠️ 显存守卫:论文报 50.0GB 峰值(8 卡,per-device bs=9);我们 3 卡 per-device bs=8,量级相同。
# A40/L40(48GB)不够,A100(80GB)/H100(80GB)/H200(141GB)够。比 --exclude 名单可靠。
# 必须用 `-i 0` 取单卡 —— `| head -1` 在 set -o pipefail 下会 SIGPIPE 退出 13(踩过)。
GPU_MEM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0)
echo "[ab] GPU0 显存=${GPU_MEM_MIB}MiB $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
# MIN_GPU_MIB:48GB 卡 + SD3_TRANSFORMER_GRADCKPT=1 时显式放宽(§4.888);默认 60000 不变。
# ⚠️ smoke 里 `bash` 调本脚本时,下面的 requeue 打的是 smoke 作业本身(§4.888:smk_l40 被重排 40 次、空耗 5h)。
if [ "${GPU_MEM_MIB:-0}" -lt "${MIN_GPU_MIB:-60000}" ]; then
  echo "[guard] 显存 ${GPU_MEM_MIB}MiB < ${MIN_GPU_MIB:-60000} —— OPSD 训练会 OOM。requeue 换节点。"
  _RS=${SLURM_RESTART_COUNT:-0}
  if [ "$_RS" -ge 40 ]; then echo "[guard] 已重启 $_RS 次 -> 放弃"; exit 41; fi
  sleep 300; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
echo "[diag] nvidia-smi 可见=$NPROC | AllocTRES: $(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -oE 'AllocTRES=[^ ]*' | head -1)"
# ⚠️ A/B 两臂的 world_size **必须相同**,否则 opsd_manifest_diff.py 会判不可比。
# SLURM 的 gres 降级不可预测(申请 3 得 2、申请 4 得 3 都实测过),所以这里不接受
# "任何合法值",而是要求**恰好等于** EXPECT_NPROC。不符即 requeue。
# ⚠️ 2026-09-01 踩坑:L36 的 `NPROC=$(nvidia-smi -L | wc -l)` 会**覆盖** --export 传入的 NPROC,
# 而本行默认 3。qopsd 臂申请 4 卡被 SLURM 降级到 3,守卫比的是 3==3 于是放行,
# 结果臂以 nproc=3 跑,与 nproc=4 的 baseline/a1/n1/n2 口径不一致。
# ⇒ **提交 A/B 臂时必须显式 `--export=...,EXPECT_NPROC=<期望卡数>`**;
#    若未显式给,而调用方给了 NPROC,就以 NPROC 为期望值(下面这行),避免同一个坑再犯。
# EXPECT_NPROC 可以给**一个集合**(空格分隔,如 "3 4")。
# 起因:本集群会把 gres:4 的请求只授予 3 张(2026-09-04 实测多次),守卫拦下后 requeue,
# 一整天没跑成任何臂。而我们**同时有 np3 与 np4 两条 baseline 且都有逐图数据**,
# 拿到 3 或 4 都能做同 world_size 的对比 —— 所以正确做法不是死等,是接受任一并记录。
# 仍然拒绝集合外的值(例如 2),因为那既不满足 48x24 分组也没有对应 baseline。
EXPECT_NPROC=${EXPECT_NPROC:-${NPROC_REQ:-3}}
_NP_OK=0
for _e in $EXPECT_NPROC; do [ "$NPROC" = "$_e" ] && _NP_OK=1; done
if [ "$_NP_OK" != "1" ]; then
  _RS=${SLURM_RESTART_COUNT:-0}
    # 2026-09-06 重修:代价来自 `sleep 300` **持有那几张卡**空等,不是 requeue 次数本身。
    # 改 sleep 20 后单轮占卡约 20 秒,上限交给 REQUEUE_CAP(默认 6;过夜无人值守投 40)。
    # 背景:2026-09-03 inner4 循环 24 轮才被人工发现(申请 h200:3、SLURM 只给 2)。
  if [ "$_RS" -ge "${REQUEUE_CAP:-6}" ]; then
    echo "[guard] nproc=$NPROC 不在 EXPECT_NPROC={$EXPECT_NPROC} 内,已重启 $_RS 次 -> 放弃(释放显卡)"
    echo "[guard] ⇒ 集群大概率凑不出 $EXPECT_NPROC 张连续卡。重投时加 --exclusive 要整节点。"
    exit 1
  fi
  echo "[guard] nproc=$NPROC 不在 EXPECT_NPROC={$EXPECT_NPROC} 内(两臂须同 world_size) -> requeue(第 $_RS 次)"
  sleep 20; scontrol requeue "$SLURM_JOB_ID"; exit 0
fi
case "$NPROC" in
  3|4|6|8) ;;
  *) _RS=${SLURM_RESTART_COUNT:-0}
     if [ "$_RS" -ge 6 ]; then echo "[guard] nproc=$NPROC 已重启 $_RS 次 -> 放弃(释放显卡)"; exit 1; fi
     echo "[guard] nproc=$NPROC 不满足 48x24 分组 -> requeue(第 $_RS 次)"
     sleep 900; scontrol requeue "$SLURM_JOB_ID"; exit 0;;
esac
# ⚠️ world_size 进 manifest。两臂 world_size 必须相同,否则 opsd_manifest_diff.py 会报未声明差异。
export PYTHONPATH="$SHIM:$R"
export PUBLIC_N_GPUS="$NPROC" PUBLIC_LAUNCH_WORLD_SIZE="$NPROC" PUBLIC_POLICY_WORLD_SIZE="$NPROC"
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0
export OPSD_WF_MODE=adaptive          # 会改目标函数且不进 manifest,显式中和(0041 §③)
export RPI_GATEA_PROBE=0              # A/B 不开探针
export RPI_N2=0                       # 默认关闭;仅 ARM=n2 下面会置 1
export RPI_GATEP_PROBE=0              # 默认关闭;仅 ARM=gatep 下面会置 1
export RPI_CERT_PROBE=0               # 默认关闭;仅 ARM=certp 下面会置 1
export RPI_QGRAD_PROBE=0              # 默认关闭;仅 ARM=q1/q1b 下面会置 1
export RPI_Q1B_PROBE=0                # 默认关闭;仅 ARM=q1b 下面会置 1
export RPI_Q2A_PROBE=0                # 默认关闭;仅 ARM=q2a 下面会置 1
export RPI_BNOISE_PROBE=0             # 默认关闭;仅 ARM=bnoise 下面会置 1
export RPI_BNOISE_SPLIT=random        # 半批切分:random(默认)| prompt(0064 §⑧)
export RPI_RTCREACH_PROBE=0           # 默认关闭;仅 ARM=rtcreach 下面会置 1
export RPI_QOPSD=0                    # 默认关闭;仅 ARM=qopsd 下面会置 1(**训练干预,不是探针**)
export RPI_QCORE_SUFFIX=dpm2          # Q 后缀 solver:dpm2(官方)| flow(部署家族,§4.517)
export RPI_QCORE_FROZENREF=0          # 1 ⇒ Q oracle 去噪器 = 冻结底模(§4.847)
export RPI_QCORE_LOOKAHEAD=0          # 1 ⇒ Q oracle 只走一步取下一态 x̂0(L1,§4.844)
export RPI_QCORE_FLOW_STEPS=0         # >0 时把 flow 后缀剩余段均匀细分成该步数(§4.517-P 探针用)
unset PUBLIC_NUM_GROUPS               # 默认用官方的 48;仅 ARM=b4x 下面会置 192(4× batch)
unset RPI_MULTIK_SUFFIX               # 低噪 suffix 锚(§4.855 (a));仅 canonsfx/qlook1sfx 下面会置值
export RPI_EMA_RESTART=0              # 周期性 EMA 重启(§4.860);仅 multikF4ema/canonema 下面会置 50
export RPI_MULTIK_WFDETACH=0          # §4.876 额外态归一化 detach;仅 multikF4wfd 下面会置 1
export RPI_MK_GATE=0                  # §4.883 额外态门控;仅 multikF4gate 下面会置 ${MK_GATE:-0.75}
export RPI_MK_SPLIT=0                 # §4.888 额外态独立 Adam;仅 multikF4sa 下面会置 ${MK_SPLIT:-0.5}
export RPI_STEP_CAP=0                 # §4.875 每步位移上限;仅 multikF4cap 下面会置 ${STEP_CAP:-1.5}
export RPI_OPT_DIAG=${OPT_DIAG:-0}     # §4.875 优化器诊断(纯日志);由提交时的 OPT_DIAG=1 打开,任何臂都可加
export RPI_MULTIK_TARGET=prop         # MSP 额外态目标分解(§4.861);仅 mkdirF4/mkconsF4 下面会改
export RPI_TPD=0                      # TPD(§4.868);仅 tpdF4 下面会置 1
unset RPI_TPD_M                       # TPD 续采条数;仅 tpdF4 下面会置
export RPI_TPD_XDIAG=0                # §4.889 TPD 归因探针(两组独立续采的夹角,纯日志);仅 tpdF4 下面可置 1
export RPI_A1_MODE=abs                # A1 判据:abs(原版,实测只否决 1.39%)| rel(A1-v2)
export RPI_QOPSD_DIR=q                # Q-OPSD 方向:q(d_Q)| canon(仅回溯消融,ARM=qbt)
export WANDB_MODE=${WANDB_MODE:-offline}
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NPROC ))
export MASTER_PORT=$(( 29000 + SLURM_JOB_ID % 20000 ))
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
# ⚠️ 种子只能用 env 传:`--config.seed=N` 会 FATAL —— `config.seed` 默认 None,
# 而 ml_collections config_flags 不支持覆盖 NoneType 字段(2026-08-31 实测 job 1037437)。
export RPI_SEED="$SEED"
case "$ARM" in
  baseline) export RPI_A1=0 ;;
  baseA4)   export RPI_A1=0 ;;    # canonical,但 A100/nproc=4 —— 给 hybrid 做精确对照(§4.608)
  # §4.561:纯重复 run —— 与 baseline 完全同配置同 seed,唯一目的是测 run-to-run 噪声(§4.549)。
  baserep)  export RPI_A1=0 ;;        # 官方 Open3 目标,原样
  a1)       export RPI_A1=1 ;;        # + 向量可行性 veto
  n1)       export RPI_A1=0 ;;        # 尺度均衡加权:不动 veto,只改权重
  n2)       export RPI_A1=0; export RPI_N2=1 ;;   # 冻结 |r| 归一化分母(步长类干预)
  gatep)    export RPI_A1=0; export RPI_GATEP_PROBE=1 ;;  # LSR 方向组合的零训练 oracle 探针
  certp)    export RPI_A1=0; export RPI_CERT_PROBE=1 ;;   # rollout 可达率探针(只测不门控)
  a3)       export RPI_A1=0; OPA_CERT_ARG="--config.opsd.opa_cert=1" ;;  # 打开官方 rollout 认证门控
  q1)       export RPI_A1=0; export RPI_QGRAD_PROBE=1 ;;   # Gate Q1:suffix action-value 梯度探针
  # qsig:§4.315/§4.317 的 query-sigma 扫描探针。与 q1 相同的探针开关,但把 opa_query_sigma
  # 做成必填参数。不复用 EXTRA_CFG 直传,是因为 L22 会无条件清空外部传入的 EXTRA_CFG。
  qsig)     export RPI_A1=0; export RPI_QGRAD_PROBE=1; \
            EXTRA_CFG="--config.opsd.opa_query_sigma=${QSIG_SIGMA:?qsig 臂必须显式指定 QSIG_SIGMA}" ;;
  # qstrain:与 qsig 同一个旋钮,但**不开探针** —— 用于真正训练一条改了 query sigma 的臂。
  # qsig 带 RPI_QGRAD_PROBE=1 会多做后缀前向,不能用来跑 100 epoch 的正式臂。
  qstrain)  export RPI_A1=0; \
            EXTRA_CFG="--config.opsd.opa_query_sigma=${QSIG_SIGMA:?qstrain 臂必须显式指定 QSIG_SIGMA}" ;;
  # opacfg:通用的 config.opsd.* 覆盖臂。把整串覆盖放进 OPA_CFG,避免每加一个旋钮就新增一个臂。
  # 例:OPA_CFG="--config.opsd.opa_rho=0.2"。无探针,可跑 100 epoch 的正式臂。
  opacfg)   export RPI_A1=0; EXTRA_CFG="${OPA_CFG:?opacfg 臂必须显式指定 OPA_CFG}" ;;
  q1b)      export RPI_A1=0; export RPI_QGRAD_PROBE=1; export RPI_Q1B_PROBE=1 ;;  # + 多步长回溯
  q2a)      export RPI_A1=0; export RPI_Q2A_PROBE=1 ;;    # Gate Q2a:逐样本 LoRA 梯度相容性
  bnoise)   export RPI_A1=0; export RPI_BNOISE_PROBE=1 ;; # Gate B:半批梯度噪声尺度/临界批量(0064)
  bnoisep)  export RPI_A1=0; export RPI_BNOISE_PROBE=1; export RPI_BNOISE_SPLIT=prompt ;; # + 按 prompt 切半批(0064 §⑧)
  rtcreach) export RPI_A1=0; export RPI_RTCREACH_PROBE=1 ;;  # RTC 可达性 R(g)/E[R(u)](0067 §⑦)
  qopsd)    export RPI_A1=0; export RPI_QOPSD=1 ;;           # Q-OPSD:d_Q 方向 + Q1b 在线回溯(训练干预)
  # ── §4.517/§4.519:后缀 solver 对齐 ───────────────────────────────────────
  # 探针(短跑,读 outputs/<tag>/gateQ1_probe.jsonl 的 cos_clean_q):
  q1flow)   export RPI_A1=0; export RPI_QGRAD_PROBE=1; export RPI_QCORE_SUFFIX=flow ;;
  q1flow40) export RPI_A1=0; export RPI_QGRAD_PROBE=1; export RPI_QCORE_SUFFIX=flow; \
            export RPI_QCORE_FLOW_STEPS=40 ;;   # ⚠️ 可微后缀 40 次前向,显存重,须先试一个 chunk
  # 训练臂(成本 = baseline:后缀仍只 1 次前向):
  qcoresolv) export RPI_A1=0; export RPI_QCORE=1; export RPI_QCORE_SUFFIX=flow ;;
  # qlook1(§4.844):canonical,唯一改动 = target 方向的 oracle 换成"一步前瞻"R(D(z_s − s·v_old(z_s,s)))。
  #   均匀 5 步细网格的首步(h≈0.0556,对齐 P10 在 σ≈0.26 处的 0.0546);对 v_old 的输入梯度保留。
  qlook1)   export RPI_A1=0; export RPI_QCORE=1; export RPI_QCORE_SUFFIX=flow; export RPI_QCORE_FLOW_STEPS=5; export RPI_QCORE_LOOKAHEAD=1 ;;
  # qlook1fz(§4.847):qlook1 唯一改动 = 前瞻去噪器换成冻结底模(不随训练漂移)。
  qlook1fz) export RPI_A1=0; export RPI_QCORE=1; export RPI_QCORE_SUFFIX=flow; export RPI_QCORE_FLOW_STEPS=5; export RPI_QCORE_LOOKAHEAD=1; export RPI_QCORE_FROZENREF=1 ;;
  # §4.855 (a) 2×2 的两个新格:低噪 suffix 锚,σ=0.2224 = L1 前瞻的 σ_s(=σ_q·4/5);canonsfx 是"只加锚"的对照。
  canonsfx)  export RPI_A1=0; export RPI_MULTIK_SUFFIX=0.2224 ;;
  qlook1sfx) export RPI_A1=0; export RPI_QCORE=1; export RPI_QCORE_SUFFIX=flow; export RPI_QCORE_FLOW_STEPS=5; export RPI_QCORE_LOOKAHEAD=1; export RPI_MULTIK_SUFFIX=0.2224 ;;
  # §4.859:冻结前瞻 × 低噪锚 的第四格(qlook1 / qlook1fz / qlook1sfx 已有)
  qlook1fzsfx) export RPI_A1=0; export RPI_QCORE=1; export RPI_QCORE_SUFFIX=flow; export RPI_QCORE_FLOW_STEPS=5; export RPI_QCORE_LOOKAHEAD=1; export RPI_QCORE_FROZENREF=1; export RPI_MULTIK_SUFFIX=0.2224 ;;
  # §4.521:trust-region 半径减半(探针实测 α=0.5/0.25 并列最优,生产 α=1.0 过冲 5.69×MDE)。
  # 对照是既有 qcore(ρ=0.10),单因素。❌ 不要改 opa_n_ascent=1:那等于 α=1.0,实测最差。
  qcorerho) export RPI_A1=0; export RPI_QCORE=1; EXTRA_CFG="--config.opsd.opa_rho=0.05" ;;
  # 通用 Q-core + config 覆盖臂(例:OPA_CFG="--config.opsd.opa_query_sigma=0.4649")
  qcorecfg) export RPI_A1=0; export RPI_QCORE=1; EXTRA_CFG="${OPA_CFG:?qcorecfg 臂必须显式指定 OPA_CFG}" ;;
  # 提案 A(§4.587/§4.588):把同一个 y± 传播到 rollout 的额外状态。
  # 状态集由探针的 SNR 定标(§4.588):{5,6,7}+kq,transformer 前向次数与 inner4 相同 ⇒ 等成本对照。
  multik4)  export RPI_A1=0; export RPI_MULTIK_STATES="5,6,7"; export RPI_MULTIK_WEIGHT=uniform ;;
  multik4s) export RPI_A1=0; export RPI_MULTIK_STATES="5,6,7"; export RPI_MULTIK_WEIGHT=snr ;;
  multik2)  export RPI_A1=0; export RPI_MULTIK_STATES="7";     export RPI_MULTIK_WEIGHT=uniform ;;
  # multikF4(§4.597 的设计原则):监督集覆盖**部署采样网格**。flow-4 的网格 σ={1.000,0.858,0.602,0.009},
  # 其内部两点在 10 步训练网格上是 k=3(σ=0.858)与 k=6(σ=0.602)。canonical 的 {0.278} 与之完全不相交。
  multikF4) export RPI_A1=0; export RPI_MULTIK_STATES="3,6";   export RPI_MULTIK_WEIGHT=uniform ;;
  # §4.860:MSP 40 步稳健性 —— 周期性 EMA 重启(P=50)与它的 canonical 对照
  multikF4ema) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_EMA_RESTART=${EMA_P:-50} ;;
  canonema)    export RPI_A1=0; export RPI_EMA_RESTART=${EMA_P:-50} ;;
  # §4.861:MSP 额外态 = (D) reward 方向 + (C) 跨噪声一致性 的 2×2(canonical = 两者都无,已有 baseH100)
  mkF4h)    export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform ;;   # = multikF4,换个目录名跑在 H100 上
  mkdirF4)  export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_TARGET=dir ;;
  mkconsF4) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_TARGET=cons ;;
  # §4.868:TPD —— 与 mkF4h(MSP)唯一的差别:额外态 k=3,6 的目标由 query 的 y± 换成 y0,k ± Δ_k(M 条续采的倾斜修正)
  tpdF4)    export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_TPD=1; export RPI_TPD_M=${TPD_M:-8}; export RPI_TPD_XDIAG=${TPD_XDIAG:-0} ;;
  # multikF4pos(§4.839):multikF4 唯一改动 = 额外态 r1 下限 0.5,去掉坏样本那一半的正反馈项。
  # §4.883:MSP + 额外态门控(τ=MK_GATE,默认 0.75;需同时给 OPT_DIAG≥1)
  multikF4gate) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MK_GATE=${MK_GATE:-0.75} ;;
  # §4.888:MSP + 额外态独立 Adam(λ=MK_SPLIT,默认 0.5);需提交时 OPT_DIAG=1
  multikF4sa) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MK_SPLIT=${MK_SPLIT:-0.5} ;;
  # §4.876:MSP + 额外态归一化 detach(与官方 adaptive weight_factor / canonical 主项同式)
  multikF4wfd) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_WFDETACH=1 ;;
  # §4.875 修复臂 B:MSP + 每步参数位移上限(c=STEP_CAP,默认 1.5;前 10 步只记不截)
  multikF4cap) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_STEP_CAP=${STEP_CAP:-1.5} ;;
  multikF4pos) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_ADVPOS=1 ;;
  # multikF4cfg:multikF4(K={3,6})+ 任意 config 覆盖的组合臂(§4.719 的 β×MSP 交互判定格)。
  # ── RB-LT 三臂(§PCFM 迁移)。都要 Euler rollout:只有 solver=flow + deterministic=True 时
  #    flow_grpo_step 化简为 z+Δσ·v,RB-LT 的 v_tar=v_old−d 才精确(dpm2 不成立)。
  #    都要 positive-only(opa_dual_neg=0):改了输入路径后镜像负支不再有 OPSD 的等价关系。
  #    评测用的也是 flow 采样器 ⇒ 训练 rollout 与评测同族。
  poseuler) export RPI_A1=0; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow" ;;
  # poseulerrot:**单变量对照** —— canonical 目标构造 + 与 rblt 相同的轮转 σ_k。
  #   rblt − poseulerrot 只差"目标构造";rblt − poseuler 还差"训练状态分布"。
  poseulerrot) export RPI_A1=0; export RPI_ROTQ=1; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow" ;;
  rblt)     export RPI_A1=0; export RPI_RBLT=1; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow" ;;
  # 目标构造的另两种(用户表格第 3/4 行)。都与 rblt 同状态、同 d、同 ρ ⇒ 只差"目标怎么构"。
  rbltpcfm)   export RPI_A1=0; export RPI_RBLT=1; export RPI_LTVAR=pcfm;   EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow" ;;
  rbltbridge) export RPI_A1=0; export RPI_RBLT=1; export RPI_LTVAR=bridge; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow" ;;
  rbltd0)   export RPI_A1=0; export RPI_RBLT=1; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow --config.opsd.opa_dir_mode=noop" ;;
  # rbltrand:§4.817 的归因对照 —— RB-LT 机制完全不变,只把 d 换成同信任域的随机方向
  # (dir_mode=rand ⇒ 无 reward 信息)。分开"reward 内容有害"与"RB-LT 机制本身有害"。
  # ⚠️ rho=0.07088 不是笔误:rand 的两步共用**同一个** fixed_dir(L575 在循环外抽),
  #    所以 ‖d‖ = budget = rho·‖x0‖ **精确**;而 grad 每步重算梯度、两步近乎正交
  #    (实测 cos≈0.005)⇒ ‖d‖ 只有 0.709·budget。rho=0.10 时 rand 的 ‖d‖=23.858
  #    对 rbltg 的 16.911 高 41.1%(§4.819)。配对要配**进 loss 的那个量 ‖d‖**,
  #    不是产生它的旋钮 rho ⇒ rho := 0.10×16.911/23.858 = 0.07088。
  #    rho 在 RB-LT 路径上只经 budget→‖d‖ 这一条通道(消费端的 y_plus 已被 rblt_y_tar 换掉、
  #    neg 支关闭),所以改 rho 不带别的副作用。
  rbltrand) export RPI_A1=0; export RPI_RBLT=1; EXTRA_CFG="--config.opsd.opa_dual_neg=0 --config.sample.solver=flow --config.opsd.opa_dir_mode=rand --config.opsd.opa_rho=0.07088" ;;
  # ── LT-OPSD(§4.830,用户原方案的**直接迁移版**)。与上面 rblt 那条路线互斥。
  #    三条臂都用 Euler rollout(solver=flow)、**保留双支**(不设 opa_dual_neg=0)、query σ=0.46488。
  #    σ 必须显式写:默认 0.278 在 10 步网格上 ⇒ σ_s=sig[9]=0.009 ⇒ b_s≈y0、z_s±≈y±
  #    ⇒ 构造退化成 ~97% canonical(θ 依赖项只差 0.269/0.278 的尺度)—— §4.830 §二。
  ltopsd)   export RPI_A1=0; export RPI_LTOPSD=1; EXTRA_CFG="--config.sample.solver=flow --config.opsd.opa_query_sigma=0.46488" ;;
  # 插值偏移的零控制:dir_mode=noop ⇒ y±≡y0 ⇒ z_s^±≡b_s,只剩"全局插值线 vs 原 rollout 路径"那一项。
  ltopsd0)  export RPI_A1=0; export RPI_LTOPSD=1; EXTRA_CFG="--config.sample.solver=flow --config.opsd.opa_query_sigma=0.46488 --config.opsd.opa_dir_mode=noop" ;;
  # 匹配对照:canonical OPSD,**同 query σ**、同 reward、同 solver。
  # 现有 poseuler 的 query σ 是默认 0.278,不能拿来比(§4.648 已证 query σ 本身会动分数)。
  qseuler)  export RPI_A1=0; EXTRA_CFG="--config.sample.solver=flow --config.opsd.opa_query_sigma=0.46488" ;;
  multikF4cfg) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; EXTRA_CFG="${OPA_CFG:?multikF4cfg 臂必须显式指定 OPA_CFG}" ;;
  # 提案 B:verified-advantage weighting。必须开 opa_cert=1(R_plus/R_minus 由它算),
  # 再用 RPI_VADV 把二值换成连续权重。c=0.02 ≈ §4.375 的 gain 量级(±0.046 的一半)。
  vadv)     export RPI_A1=0; export RPI_VADV="${VADV:-0.02}"; EXTRA_CFG="--config.opsd.opa_cert=1" ;;
  # 提案 H(§4.602):高噪区用**选择**构造目标。probe_select 实测交叉点在 σ≈0.465,
  # 高 σ 侧选择增益是梯度的 6×(σ=0.707)/39×(σ=0.913) ⇒ 主臂放在 σ=0.707,
  # 与 qstrain5(clean 梯度 @ 同一 σ,−14.7×MDE)构成**只换 oracle** 的干净 A/B。
  selhi)    export RPI_A1=0; export RPI_SELECT_M="${SELM:-4}"; \
            EXTRA_CFG="--config.opsd.opa_query_sigma=0.707278" ;;
  selhitr)  export RPI_A1=0; export RPI_SELECT_M="${SELM:-4}"; export RPI_SELECT_TR=1; \
            EXTRA_CFG="--config.opsd.opa_query_sigma=0.707278" ;;
  # 低噪对照:同一实现放在 canonical 的 σ=0.278。探针预测这里**梯度赢 5 倍**
  # ⇒ 它应当**输**给 baseline。这是本方法的"反向门":若它反而赢了,说明增益与"选择"无关。
  sello)    export RPI_A1=0; export RPI_SELECT_M="${SELM:-4}" ;;
  # ★ 混合臂(§4.604):以 probe_select 实测的交叉点 σ≈0.465 为门 ——
  #   σ<门 的状态(kq=8, σ=0.278)用**梯度**目标;σ≥门 的状态(k=5 σ=0.707, k=2 σ=0.913)用**选择**目标。
  #   这是把 §4.602 的交叉点当**设计规则**用,而不是只换单个查询状态。
  hybrid)   export RPI_A1=0; export RPI_MULTIK_STATES="2,5"; export RPI_MULTIK_WEIGHT=uniform; \
            export RPI_SELECT_M="${SELM:-4}"; export RPI_HYBRID_SIGMA="${HYB:-0.465}" ;;
  # 单调极值:同一个 kq=8 目标铺到 k=1..7(σ=0.960→0.465)。无额外 reward 调用(§4.611)。
  # 形状对照:与 multik4 同为 3 个状态,但位置纯高噪(σ 0.913/0.858/0.790)。
  # 与 multik4("5,6,7")配对即可把"哪些状态"与"多少个状态"分开(multikF4 只有 2 个,混淆了两者)。
  # multik4 的独立重跑(H200/np4),自己的 OUTPUT_DIR ⇒ 与 A100 那份互不干扰。
  # 既是族内精确对照(同 H200/np4,对照 baseline),也是一次独立 run 的复现(§4.614)。
  # ★ 论文的决定性对照(审稿人必问):"你要 4 步模型,为什么不直接用 4 步 rollout 训 OPSD?"
  #   canonical 目标构造不变,只把 rollout 从 10 步降到 4/6 步 ⇒ 训练与测试步数匹配。
  #   注意 kq 会随 σ 表变化(4 步表里最接近 0.278 的是 ~0.35)—— 这是该对照**固有**的,不是缺陷。
  # ⚠️ 4 步 σ 表 = [1.000, 0.858, 0.602, 0.00893, 0];|0.602−0.278|=0.324 > |0.00893−0.278|=0.269
  #    ⇒ 默认 opa_query_sigma=0.278 会把查询态选到**几乎全干净**的 0.00893(目标退化,实测 reward 从
  #    ep15 的 1.2342 跌到 ep29 的 0.8845)。必须显式钉到倒数第二个非退化态 σ=0.60215。
  #    (canonical 的 0.278 正是 10 步表的倒数第二个非退化态,所以这是同一设计规则的正确类比。)
  # ★ 组合臂(§4.646):6 步 rollout + 多态传播。
  #   6 步 σ 表 = [1.000, 0.923, 0.819, 0.668, 0.432, 0.00893, 0],kq=4(σ=0.432,非退化,已核)。
  #   额外态取 index 2(σ=0.819)与 3(σ=0.668)—— 10 步表里 "3,6"(σ=.858/.602)的最近类比。
  # ★★ 决定性组合臂(§4.648):查询 σ 上移到 0.465 + 多态传播,rollout 仍 10 步。
  #    §4.647 的 Q1 表明"上移查询 σ"单独就给出 +12.71/+13.18×MDE 的少步增益且免费;
  #    本方法单独给出跨全预算的 +5.62/+3.34(满步)与 +10.70/+8.20(4 步)。
  #    两者机制不同(前者满步中性、后者满步正向)⇒ 若可叠加,组合才是论文的方法。
  #    额外态仍取 "3,6"(σ=.858/.602),两者都在新查询态 σ=0.465 之上。
  qs7mk)    export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; \
            EXTRA_CFG="--config.opsd.opa_query_sigma=0.46488" ;;
  base6mk)  export RPI_A1=0; export RPI_MULTIK_STATES="2,3"; export RPI_MULTIK_WEIGHT=uniform; \
            EXTRA_CFG="--config.sample.num_steps=6" ;;
  base4)    export RPI_A1=0; EXTRA_CFG="--config.sample.num_steps=4 --config.opsd.opa_query_sigma=0.60215" ;;
  # 6 步 σ 表 = [1.000, 0.923, 0.819, 0.668, 0.432, 0.00893, 0];argmin 选中 index 4 = 0.43223,
  # **非退化,无需干预**(已核)。
  base6)    export RPI_A1=0; EXTRA_CFG="--config.sample.num_steps=6" ;;
  multik4H) export RPI_A1=0; export RPI_MULTIK_STATES="5,6,7"; export RPI_MULTIK_WEIGHT=uniform ;;
  # ★ 方法改进候选:存 1..7 全部状态,但每步**随机取 3 个**反传
  #   ⇒ multikALL 的 σ 覆盖 + multik4 的代价。与 multik4("5,6,7")同成本,可直接对比。
  multikS3) export RPI_A1=0; export RPI_MULTIK_STATES="1,2,3,4,5,6,7"; \
            export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_SAMPLE=3 ;;
  # 合成态版:状态集与 multik4 相同,但 z_k 由该样本自己的 y0 重加噪得到(非 rollout 态)。
  #   同效 ⇒ 方法不需要 rollout 状态;失效 ⇒ on-policy 轨迹态是本质。
  multikSY) export RPI_A1=0; export RPI_MULTIK_STATES="5,6,7"; \
            export RPI_MULTIK_WEIGHT=uniform; export RPI_MULTIK_SYNTH=1 ;;
  multikHI) export RPI_A1=0; export RPI_MULTIK_STATES="2,3,4"; export RPI_MULTIK_WEIGHT=uniform ;;
  multikALL) export RPI_A1=0; export RPI_MULTIK_STATES="1,2,3,4,5,6,7"; export RPI_MULTIK_WEIGHT=uniform ;;
  hybridw)  export RPI_A1=0; export RPI_MULTIK_STATES="2,5"; export RPI_MULTIK_WEIGHT=uniform; \
            export RPI_SELECT_M="${SELM:-4}"; export RPI_HYBRID_SIGMA="${HYB:-0.465}"; \
            export RPI_SELECT_TAU="${TAU:-1.0}" ;;
  hybridtr) export RPI_A1=0; export RPI_MULTIK_STATES="2,5"; export RPI_MULTIK_WEIGHT=uniform; \
            export RPI_SELECT_M="${SELM:-4}"; export RPI_HYBRID_SIGMA="${HYB:-0.465}"; \
            export RPI_SELECT_TR=1 ;;
  # §4.884 补:H100 np3 版 —— 额外态取 MSP 的 K={3,6}(σ .858/.602,都 ≥ 门 0.465 ⇒ 两态都用选择式目标),
  #   与 mkF4hr2(MSP,同 H100 np3)只差"额外态目标怎么造":传播的 query 目标 vs 选择式目标。
  hybridtr36) export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; \
            export RPI_SELECT_M="${SELM:-4}"; export RPI_HYBRID_SIGMA="${HYB:-0.465}"; export RPI_SELECT_TR=1 ;;
  hybridw36)  export RPI_A1=0; export RPI_MULTIK_STATES="3,6"; export RPI_MULTIK_WEIGHT=uniform; \
            export RPI_SELECT_M="${SELM:-4}"; export RPI_HYBRID_SIGMA="${HYB:-0.465}"; export RPI_SELECT_TAU="${TAU:-1.0}" ;;
  # 对照:同样的状态集,但**全部用梯度**(门设成 0 关掉选择)⇒ 隔离"选择"这一项
  hybgrad)  export RPI_A1=0; export RPI_MULTIK_STATES="2,5"; export RPI_MULTIK_WEIGHT=uniform ;;
  # 组合臂:multik4 叠在最强的步长臂上,测目标端增益是否与优化器端增益可加
  multiklr) export RPI_A1=0; export RPI_MULTIK_STATES="5,6,7"; export RPI_MULTIK_WEIGHT=uniform; \
            EXTRA_CFG="--config.train.learning_rate=2.4e-3" ;;   # = lr8x 的 lr,基准 3e-4 × 8
  b4x)      export RPI_A1=0; export PUBLIC_NUM_GROUPS=192 ;; # 4× batch:192 prompt × 24 图 = B_eff 4608
  a1v2)     export RPI_A1=1; export RPI_A1_MODE=rel ;;       # A1-v2:相对改善 veto(原版空转,§4.21 预计否决约 23%)
  qbt)      export RPI_A1=0; export RPI_QOPSD=1; export RPI_QOPSD_DIR=canon ;; # 消融:canonical 方向 + 同样的在线回溯
  # inner2:每个**已构造好**的目标多吃一次梯度步。目标构造在 L1573(内层循环 L1883 之外),
  # 故外层 epoch 内目标固定 —— 这是对固定目标的回归,不是 PPO,无需重要性比修正。
  # 动机:实测实现率仅 1~3%(G_realized/G_construct),亏损在拟合端而非构造端。
  inner2)   export RPI_A1=0; EXTRA_CFG="--config.train.num_inner_epochs=2"; CK_MULT=2 ;;
  # lr4xinner2:组合臂。检验 §4.298 的"一维线"在**幅度**上是否可加/可乘 ——
  # §4.298 说的是臂效应**方向**共线,没说两个干预组合后不会超出这条线。
  # lr=1.2e-3(4×)+ num_inner_epochs=2,CK_MULT=2 ⇒ 100 外层 = checkpoint-200。
  lr4xinner2) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.2e-3 --config.train.num_inner_epochs=2"; CK_MULT=2 ;;
  # lr4xgacc2:第二条组合臂。与 lr4xinner2 同为"lr4x + 一个拟合端旋钮",但换成 gacc。
  # 作用:若组合增益只在 inner 上出现 ⇒ 效应定位在内循环;两条都出现 ⇒ 是一般性的组合效应。
  lr4xgacc2) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.2e-3 --config.train.gradient_accumulation_steps=24"; CK_MULT=2 ;;
  # lr2xinner2:组合臂的**低剂量点**。与 lr4xinner2 同旋钮、不同 lr,构成组合效应的剂量-反应。
  # §4.385 显示 2× 的逐图方向与 4×/8× 显著不平行,低剂量端正是方向旋转最快处。
  lr2xinner2) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=6e-4 --config.train.num_inner_epochs=2"; CK_MULT=2 ;;
  # inner2gacc2:组合设计里唯一没填的格 —— **两个拟合端旋钮互相组合,不含 lr**。
  # 问:§4.405 的"收益次可加"是 lr 特有的,还是拟合端旋钮之间也如此?
  # 每外层 epoch 的优化步数 = (48/24) × 2 = 4 ⇒ CK_MULT=4。
  inner2gacc2) export RPI_A1=0; EXTRA_CFG="--config.train.num_inner_epochs=2 --config.train.gradient_accumulation_steps=24"; CK_MULT=4 ;;
  # lr8xinner2:组合剂量-反应的**高剂量点**。与 lr2xinner2/lr4xinner2 构成三点。
  # 问:组合惩罚(贡献 = 组合 − lr 单臂)是否随 lr 剂量单调增大?
  lr8xinner2) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=2.4e-3 --config.train.num_inner_epochs=2"; CK_MULT=2 ;;
  # inner2lrhalf:**位移对照臂**。inner2 让每个外层 epoch 多走一步位移;
  # 把 lr 减半后,每外层 epoch 的总位移回到基线水平。
  # 若此时仍显著优于"等效 lr"的预测 ⇒ inner 的收益不是来自位移,而是来自更准的拟合。
  inner2lrhalf) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.5e-4 --config.train.num_inner_epochs=2"; CK_MULT=2 ;;
  # inner4:同一旋钮的剂量点。inner2 在同采样预算下 +0.0145(t=+3.56),
  # 但那条臂跑在 A100 上而 basenp3 在 H200 上 —— inner4 投 H200 兼作同硬件对照。
  # ⚠️ global_step 在**内层**循环自增(L2275),故 ck 编号是外层 epoch 的 N 倍:
  #    inner4 的 ck400 才等于 baseline 的 ck100(同采样预算)。
  inner4)   export RPI_A1=0; EXTRA_CFG="--config.train.num_inner_epochs=4"; CK_MULT=4 ;;
  # §4.544:拟合族 2→4 仍有 +7.42×MDE 未饱和,测 4→8 能否追上 lr8x(+17.39×MDE)。
  inner8)   export RPI_A1=0; EXTRA_CFG="--config.train.num_inner_epochs=8"; CK_MULT=8 ;;
  # gacc4:与 inner4 **严格配对**的对照 —— 同采样成本、同 4 次优化器更新/外层 epoch,
  # 唯一差别是每次更新平均多少张图。inner4 = 4 遍全批(每次 1152 张);
  # gacc4 = 1 遍拆 4 次(每次 288 张)。accum 48→12。
  # 动机:§4.171 测出对齐效率仅 0.89%,而 §4.71 测出逐样本梯度 ρ≈0(无系统冲突,
  # 纯维度稀释)⇒ 若 κ ∝ 1/√B,减小每次更新的 B 应直接提高实现率。
  gacc4)    export RPI_A1=0; EXTRA_CFG="--config.train.gradient_accumulation_steps=12"; CK_MULT=4 ;;
  # gacc2:与 inner2 严格配对(同 2 次更新/epoch),accum 48→24 ⇒ 每次更新平均 576 张图
  gacc2)    export RPI_A1=0; EXTRA_CFG="--config.train.gradient_accumulation_steps=24"; CK_MULT=2 ;;
  # lr2x:与 inner2 同一个问题的**另一条路径**。inner2 靠"多走一遍"把构造目标拟合得更狠;
  # lr2x 靠"每步走更远",学习率 3e-4 → 6e-4(无 LR scheduler,常数)。
  # 判别:若两者收益相近 ⇒ 机制是"拟合强度不足";若 lr2x 无效 ⇒ 收益特指"多遍"
  # (每遍会重新 shuffle 样本序与 timestep 子集),而非单纯步长。lr2x **零额外算力**。
  lr2x)     export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=6e-4" ;;
  # lr4x:lr2x 的剂量点。lr2x(3e-4→6e-4)给了 +0.0282(t=+6.93)且零额外算力,
  # 4× 是继续涨、饱和、还是发散?常数 lr、无 scheduler,故这是干净的单因子。
  lr4x)     export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.2e-3" ;;
  # lr8x:剂量-反应的第四点。lr4x 实测 2.06× 且远未饱和(§4.228),测最优点在哪。
  # 风险:8× lr 可能发散;lr4x 已确认不发散(§4.203),此臂需盯 total_loss。
  lr6x)        export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.8e-3"; CK_MULT=1 ;;
  lr8x)     export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=2.4e-3" ;;
  # lrhalf:剂量-反应的**低端点**(1.5e-4 = 0.5× base)。§4.241 测出单步实现率 κ 的
  # 峰值恰落在 base lr(3e-4)处,而 held-out 一路涨到 4×。若 lrhalf 显著差于 base,
  # 则确认 base 位于 held-out 曲线的**上升段**,单步最优点远低于 held-out 最优点。
  lrhalf)   export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=1.5e-4" ;;
  # lr16x:剂量-反应的第六点(4.8e-3 = 16× base)。ck100 处的曲线为
  #   0.5× 0.66 | 1× 1.00 | 2× 1.54 | 4× 2.00 | 8× 2.39,**在 8× 仍未见平台**。
  #   两种外推给 16× 的倍率 2.72×(增量递减)与 2.86×(log 线性)。
  #   风险:16× 可能发散;lr8x 的失稳旗标未触发(最长连降 3 < 4),波动 ∝ lr^0.25。
  lr16x)    export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=4.8e-3" ;;
  # lr12x:发散边界的二分点(3.6e-3 = 12× base)。8× 有效(+0.0583,倍率 2.39×),
  #   16× 发散(§4.285:连续 4 个 epoch 低于起点,ep5 比 baseline 低 0.584)。
  #   12× 把边界区间 (8×, 16×) 缩到一半。
  lr12x)    export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=3.6e-3" ;;
  # lr10x:§4.299 头条("最优 lr 被失稳切断,不是曲线转平")的直接检验点。
  # 已知 8× 稳定(+0.0583)、12×/16× 发散 ⇒ 边界在 (8×,12×)。10× 三种结局都有信息量。
  lr10x)    export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=3e-3" ;;
  # lr12xeps:§4.301 指出高 lr 失稳来自**步长**而非梯度尖峰,且 AdamW 的 eps 是
  #   唯一零代码改动的步长限制旋钮。实测逐坐标 |g_i| ≈ 1.29e-6,而默认 eps=1e-8
  #   小 129 倍 ⇒ 完全不起作用。取 eps=6.5e-7 使阻尼因子 f≈0.667,
  #   把 lr12x 的有效步长 15.60 压到 **10.4 = lr8x 的步长**。
  #   判别:同样有效步长下,12×+选择性阻尼 是否优于 8×+无阻尼。
  lr12xeps) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=3.6e-3 --config.train.adam_epsilon=6.5e-7" ;;
  # §4.541:10× + 同一个 eps。测峰值是否在 8× 以上(lr8x +17.4 vs lr12xeps +15.4×MDE)。
  lr10xeps) export RPI_A1=0; EXTRA_CFG="--config.train.learning_rate=3e-3 --config.train.adam_epsilon=6.5e-7" ;;
  # qcore:Q-OPSD-Core(0069 §10)。保留官方两步构造 + advantage 语义,
  # 只把 oracle 从 ∇R(D(y)) 换成 ∇R(D(F_old(s,y)))。无回溯。
  qcore)    export RPI_A1=0; export RPI_QCORE=1 ;;
  # qcorelr:2×2 的第四格 {baseline,qcore} × {3e-4,6e-4}。前三格已有
  #   baseline@3e-4=1.6347  baseline@6e-4=1.6629(lr2x)  qcore@3e-4=1.6206
  # 预登记假说:Q 的构造目标考虑整条后缀、离当前策略更远 ⇒ **更难拟合**
  #   ⇒ Q 应比 baseline 更受益于大 LR(**正交互**)。若交互为正且 qcore@6e-4>1.6629,
  #   则 Q 在官方口径转正,且 4× 步数效率仍在 ⇒ 质量与推理成本同时改善。
  qcorelr)  export RPI_A1=0; export RPI_QCORE=1; EXTRA_CFG="--config.train.learning_rate=6e-4" ;;
  # kfit:κ_fit 探针臂。从 baseline ck100 续跑 2 个 epoch,只为在**同一批 state、同一构造目标**上
  # 扫步长 η∈{0,3e-4,6e-4,1.2e-3},测目标位移被实现的比例与正交漂移。测完复原参数。
  # η=0 那一格是**实现门**:κ 必须恰为 0,否则说明复原/测量有 bug。
  kfit)     export RPI_A1=0; export RPI_KFIT=1 ;;
  *) echo "[guard] 未知 ARM=$ARM"; exit 1 ;;
esac
[ -n "$MIXW" ] && export PUBLIC_MIXED_REWARDS="$MIXW"
# ── 训练前一致性自检(用户 2026-08-30 要求:每次 run 前都查) ──────────────
# 查数据 md5/行数、仓库与 pristine 的差异、危险 env、config 解析与 samples/epoch。
# 不通过直接退出 —— 宁可不跑,也不要跑出一个口径不明的结果。
_AE=(--allow-env RPI_A1 --allow-env RPI_N2 --allow-env RPI_GATEP_PROBE --allow-env RPI_CERT_PROBE --allow-env RPI_QGRAD_PROBE --allow-env RPI_Q1B_PROBE --allow-env RPI_Q2A_PROBE --allow-env RPI_BNOISE_PROBE --allow-env RPI_BNOISE_SPLIT --allow-env RPI_RTCREACH_PROBE --allow-env RPI_QOPSD --allow-env RPI_QCORE --allow-env RPI_KFIT --allow-env RPI_KFIT_STEPS --allow-env RPI_KFIT_PS --allow-env RPI_KFIT_PS_N --allow-env RPI_KFIT_PS_STEPS --allow-env RPI_KFIT_NMB --allow-env PUBLIC_NUM_GROUPS --allow-env RPI_A1_MODE --allow-env RPI_QOPSD_DIR --allow-env RPI_QCORE_SUFFIX --allow-env RPI_QCORE_FLOW_STEPS --allow-env RPI_GATEA_PROBE --allow-env RPI_MULTIK_STATES --allow-env RPI_MULTIK_WEIGHT --allow-env RPI_VADV --allow-env RPI_SELECT_M --allow-env RPI_SELECT_TR --allow-env RPI_HYBRID_SIGMA --allow-env RPI_SELECT_TAU --allow-env RPI_MULTIK_SAMPLE --allow-env RPI_MULTIK_SYNTH --allow-env RPI_RBLT --allow-env RPI_ROTQ --allow-env RPI_LTVAR --allow-env RPI_LTOPSD --allow-env RPI_MULTIK_ADVPOS --allow-env RPI_QCORE_LOOKAHEAD --allow-env RPI_QCORE_FROZENREF --allow-env RPI_MULTIK_SUFFIX --allow-env RPI_EMA_RESTART --allow-env RPI_OPT_DIAG --allow-env RPI_STEP_CAP --allow-env RPI_MK_GATE --allow-env RPI_MK_SPLIT --allow-env RPI_MULTIK_WFDETACH --allow-env RPI_MULTIK_TARGET --allow-env RPI_TPD --allow-env RPI_TPD_M --allow-env RPI_TPD_XDIAG)
"$PY" "$OPD_ROOT/preflight_check.py" --repo "$R" --config sd35_open3 \
      --nproc "$NPROC" --seed "$SEED" --expect-updates "$UPDATES" \
      --expect-fork-clean "${_AE[@]}" || { echo "[ab] ❌ preflight 未通过,放弃"; exit 42; }

# ── 多分区投递的"先跑到的赢"互斥 ────────────────────────────────────────
# 用户 2026-08-31:"你也可以多处提交,哪边触发了先 run 了就 run 哪边"。
# 同一条臂会被投到 sablab-gpu / minilab-gpu / scu-gpu 三处,必须保证**只有一个副本真的训**,
# 否则两个副本会写同一个 save_dir,互相覆盖 checkpoint。
# 锁放在所有守卫**之后** —— 因 nproc/显存不合格而 requeue 的副本不该占着锁。
# 锁必须按 **OUTPUT_DIR** 命名,不能按 ARM_SEED_UPDATES ——
# qsig 的各 sigma 格共用 ARM/SEED/UPDATES 但 OUTPUT_DIR 不同,
# 旧命名会让第二格抢不到锁而 **静默 exit 0**(白丢一次 GPU 分配)。
# OUTPUT_DIR 正是锁要保护的东西(防两个副本写同一个 save_dir),按它命名语义也更正确。
LOCK="/midtier/minilab/scratch/chw4026/tmp/ablock_$(basename "$OUTPUT_DIR")"
# 1) 已经跑完就别重跑
_CK_FINAL=$((UPDATES * CK_MULT))
if [ -f "$OUTPUT_DIR/checkpoints/checkpoint-${_CK_FINAL}/lora/adapter_config.json" ]; then
  echo "[lock] $OUTPUT_DIR 已有 checkpoint-$_CK_FINAL(=${UPDATES}外层×${CK_MULT}),该臂已完成,本副本退出"; exit 0
fi
# 2) mkdir 是原子操作,用它抢锁
if mkdir "$LOCK" 2>/dev/null; then
  echo "$SLURM_JOB_ID" > "$LOCK/jobid"; echo "[lock] 本副本($SLURM_JOB_ID)取得 $ARM 的锁"
else
  _OWNER=$(cat "$LOCK/jobid" 2>/dev/null)
  if [ -n "$_OWNER" ] && [ -n "$(squeue -j "$_OWNER" -h 2>/dev/null)" ]; then
    echo "[lock] $ARM 已由 job $_OWNER 在跑,本副本($SLURM_JOB_ID)退出"; exit 0
  fi
  echo "[lock] 锁的前任 ${_OWNER:-?} 已不在,本副本($SLURM_JOB_ID)接管"
  echo "$SLURM_JOB_ID" > "$LOCK/jobid"
fi

# 干预旁证:RPI_* 是 env 型干预,在 run_config.json 里**不留任何痕迹**,
# manifest 比对因此分辨不出 baseline 与 a1/n2。把 flag 落到 run 目录里,
# 与数据同处一地。**不写进 run_config.json** —— 那会让已完成臂缺字段、破坏比对。
mkdir -p "$OUTPUT_DIR" 2>/dev/null || true
cat > "$OUTPUT_DIR/arm_provenance.json" <<EOF || true
{"arm":"$ARM","RPI_A1":"$RPI_A1","RPI_N2":"$RPI_N2","RPI_GATEP_PROBE":"$RPI_GATEP_PROBE","RPI_CERT_PROBE":"$RPI_CERT_PROBE","RPI_QGRAD_PROBE":"$RPI_QGRAD_PROBE","RPI_Q1B_PROBE":"$RPI_Q1B_PROBE","RPI_Q2A_PROBE":"$RPI_Q2A_PROBE","RPI_BNOISE_PROBE":"$RPI_BNOISE_PROBE","RPI_BNOISE_SPLIT":"$RPI_BNOISE_SPLIT","RPI_RTCREACH_PROBE":"$RPI_RTCREACH_PROBE","RPI_QOPSD":"$RPI_QOPSD","RPI_QCORE":"${RPI_QCORE:-0}","RPI_QCORE_SUFFIX":"${RPI_QCORE_SUFFIX:-dpm2}","RPI_QCORE_FLOW_STEPS":"${RPI_QCORE_FLOW_STEPS:-0}","RPI_QCORE_LOOKAHEAD":"${RPI_QCORE_LOOKAHEAD:-0}","RPI_QCORE_FROZENREF":"${RPI_QCORE_FROZENREF:-0}","RPI_GATEA_PROBE":"${RPI_GATEA_PROBE:-0}","RPI_HYBRID_SIGMA":"${RPI_HYBRID_SIGMA:-0}","RPI_SELECT_M":"${RPI_SELECT_M:-0}","RPI_SELECT_TR":"${RPI_SELECT_TR:-0}","RPI_SELECT_TAU":"${RPI_SELECT_TAU:-0}","RPI_MULTIK_SAMPLE":"${RPI_MULTIK_SAMPLE:-0}","RPI_MULTIK_SYNTH":"${RPI_MULTIK_SYNTH:-0}","RPI_VADV":"${RPI_VADV:-0}","RPI_MULTIK_STATES":"${RPI_MULTIK_STATES:-}","RPI_RBLT":"${RPI_RBLT:-0}","RPI_ROTQ":"${RPI_ROTQ:-0}","RPI_LTVAR":"${RPI_LTVAR:-}","RPI_LTOPSD":"${RPI_LTOPSD:-0}","RPI_MULTIK_ADVPOS":"${RPI_MULTIK_ADVPOS:-0}","RPI_MULTIK_SUFFIX":"${RPI_MULTIK_SUFFIX:-}","RPI_EMA_RESTART":"${RPI_EMA_RESTART:-0}","RPI_OPT_DIAG":"${RPI_OPT_DIAG:-0}","RPI_STEP_CAP":"${RPI_STEP_CAP:-0}","RPI_MK_GATE":"${RPI_MK_GATE:-0}","RPI_MK_SPLIT":"${RPI_MK_SPLIT:-0}","SD3_TRANSFORMER_GRADCKPT":"${SD3_TRANSFORMER_GRADCKPT:-0}","RPI_MULTIK_WFDETACH":"${RPI_MULTIK_WFDETACH:-0}","RPI_MULTIK_TARGET":"${RPI_MULTIK_TARGET:-prop}","RPI_TPD":"${RPI_TPD:-0}","RPI_TPD_M":"${RPI_TPD_M:-}","RPI_TPD_XDIAG":"${RPI_TPD_XDIAG:-0}","RPI_MULTIK_WEIGHT":"${RPI_MULTIK_WEIGHT:-uniform}","RPI_KFIT":"${RPI_KFIT:-0}","RPI_KFIT_STEPS":"${RPI_KFIT_STEPS:-}","RPI_KFIT_PS":"${RPI_KFIT_PS:-0}","RPI_KFIT_PS_N":"${RPI_KFIT_PS_N:-}","RPI_KFIT_PS_STEPS":"${RPI_KFIT_PS_STEPS:-}","RPI_KFIT_NMB":"${RPI_KFIT_NMB:-}","PUBLIC_NUM_GROUPS":"${PUBLIC_NUM_GROUPS:-48}","RPI_A1_MODE":"$RPI_A1_MODE","RPI_QOPSD_DIR":"$RPI_QOPSD_DIR","RPI_SEED":"$SEED","opa_resume_dir":"${OPA_RESUME_DIR:-}","extra_cfg":"${EXTRA_CFG:-}","opa_cfg":"${OPA_CFG:-}","opa_cert_arg":"${OPA_CERT_ARG:-}",
 "config":"$AB_CONFIG","mixw":"${MIXW:-}","updates":"$UPDATES","nproc":"$NPROC",
 "train_script_sha256":"$(sha256sum "$R/scripts/train_opsd_ri_sd3.py" | cut -c1-16)",
 "host":"$(hostname)","job":"$SLURM_JOB_ID","gpu":"$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)"}
EOF
echo "[ab] arm=$ARM RPI_A1=$RPI_A1 RPI_N2=$RPI_N2 RPI_GATEP=$RPI_GATEP_PROBE RPI_CERTP=$RPI_CERT_PROBE RPI_Q1=$RPI_QGRAD_PROBE RPI_Q1B=$RPI_Q1B_PROBE RPI_BN=$RPI_BNOISE_PROBE RPI_BNSPLIT=$RPI_BNOISE_SPLIT RPI_RTCR=$RPI_RTCREACH_PROBE RPI_QOPSD=$RPI_QOPSD RPI_QCORE=${RPI_QCORE:-0} QSUF=${RPI_QCORE_SUFFIX:-dpm2} QFLOWN=${RPI_QCORE_FLOW_STEPS:-0} QLOOK1=${RPI_QCORE_LOOKAHEAD:-0} QFROZEN=${RPI_QCORE_FROZENREF:-0} GATEA=${RPI_GATEA_PROBE:-0} HYB=${RPI_HYBRID_SIGMA:-0} SELM=${RPI_SELECT_M:-0}/${RPI_SELECT_TR:-0} TAU=${RPI_SELECT_TAU:-0} VADV=${RPI_VADV:-0} RBLT=${RPI_RBLT:-0} ROTQ=${RPI_ROTQ:-0} LTVAR=${RPI_LTVAR:-none} LTOPSD=${RPI_LTOPSD:-0} MKADVPOS=${RPI_MULTIK_ADVPOS:-0} MKSFX=${RPI_MULTIK_SUFFIX:-none} EMARS=${RPI_EMA_RESTART:-0} OPTDIAG=${RPI_OPT_DIAG:-0} STEPCAP=${RPI_STEP_CAP:-0} MKGATE=${RPI_MK_GATE:-0} MKSPLIT=${RPI_MK_SPLIT:-0} GRADCKPT=${SD3_TRANSFORMER_GRADCKPT:-0} WFDET=${RPI_MULTIK_WFDETACH:-0} MKTGT=${RPI_MULTIK_TARGET:-prop} TPD=${RPI_TPD:-0}/${RPI_TPD_M:-none}/x${RPI_TPD_XDIAG:-0} MULTIK=${RPI_MULTIK_STATES:-none}/${RPI_MULTIK_WEIGHT:-uniform}/samp${RPI_MULTIK_SAMPLE:-0}/syn${RPI_MULTIK_SYNTH:-0} KFIT=${RPI_KFIT:-0} NGROUPS=${PUBLIC_NUM_GROUPS:-48} A1MODE=$RPI_A1_MODE QDIR=$RPI_QOPSD_DIR certarg=${OPA_CERT_ARG:-none} extracfg=${EXTRA_CFG:-none} resumedir=${OPA_RESUME_DIR:-none} ckmult=$CK_MULT seed=$SEED updates=$UPDATES nproc=$NPROC config=$AB_CONFIG mixw=${MIXW:-<默认>} host=$(hostname) job=$SLURM_JOB_ID $(date)"
echo "[ab] fork_from=$(cat $R/FORKED_FROM_COMMIT.txt 2>/dev/null)"
# ── 断点续跑(让 preempt_gpu 可用)────────────────────────────────────────
# `config.resume_from` 是字符串(非 None),可直接用 --config 覆盖;
# `resolve_resume_checkpoint` 会自动取 run 目录下**最新**的 checkpoint。
# save_freq=10 ⇒ 被抢占最多损失 10 步(约 38 分钟)。
# ⚠️ **RNG 不会被恢复**(trainer 调 restore_ema_and_rng 时未传 restore_rng=True),
#    所以"被抢占过的臂"与"一路跑完的臂"在续跑段的随机性不同,**部分破坏配对**。
#    因此把重启次数写进产物,分析时必须核对两臂是否都为 0。
RESUME_ARG=()
if ls "$OUTPUT_DIR/checkpoints"/checkpoint-* >/dev/null 2>&1; then
  RESUME_ARG=(--config.resume_from="$OUTPUT_DIR")
  echo "[ab] 检测到已有 checkpoint,断点续跑: $OUTPUT_DIR"
elif [ -n "${OPA_RESUME_DIR:-}" ]; then
  # 跨 run 续训(退火实验):从**另一条 run** 的指定 checkpoint 起跑。
  # `resolve_resume_checkpoint` 名字匹配 checkpoint-NNN 时直接用该目录,
  # 否则取父目录下**最新**的 —— 所以这里必须给到 checkpoint-NNN 这一层。
  # ⚠️ 本分支只在本 run 尚无自己的 checkpoint 时生效:上面的自恢复分支优先,
  #    否则 requeue 会把进度退回源 checkpoint。
  [ -d "$OPA_RESUME_DIR" ] || { echo "[ab] ❌ OPA_RESUME_DIR 不存在: $OPA_RESUME_DIR"; exit 43; }
  [ -f "$OPA_RESUME_DIR/trainer_state.json" ] || { echo "[ab] ❌ OPA_RESUME_DIR 无 trainer_state.json(resume_position 会失败): $OPA_RESUME_DIR"; exit 43; }
  case "$(basename "$OPA_RESUME_DIR")" in checkpoint-*) :;; *) echo "[ab] ❌ OPA_RESUME_DIR 必须指到 checkpoint-NNN 这一层,否则会取到最新 ckpt: $OPA_RESUME_DIR"; exit 43;; esac
  RESUME_ARG=(--config.resume_from="$OPA_RESUME_DIR")
  echo "[ab] 跨 run 续训: $OPA_RESUME_DIR (起点 $(cat "$OPA_RESUME_DIR/trainer_state.json"))"
fi
mkdir -p "$OUTPUT_DIR"
echo "{\"slurm_restart_count\": ${SLURM_RESTART_COUNT:-0}, \"job\": \"$SLURM_JOB_ID\", \"partition\": \"${SLURM_JOB_PARTITION:-?}\", \"resumed\": ${#RESUME_ARG[@]}}" \
  >> "$OUTPUT_DIR/run_attempts.jsonl"

"$PY" scripts/check_reward_setup.py --reward open3 --backbone sd35 || exit 1
"$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
     scripts/train_opsd_ri_sd3.py \
     --config "$AB_CONFIG" ${OPA_CERT_ARG:+$OPA_CERT_ARG} \
     --config.num_epochs="$UPDATES" ${EXTRA_CFG:+$EXTRA_CFG} \
     --config.save_dir="$OUTPUT_DIR" "${RESUME_ARG[@]}"
_RC=$?
echo "[ab] arm=$ARM exit=$_RC $(date)"
python3 -c "
import json;d=json.load(open('$OUTPUT_DIR/run_config.json'))
print('[ab] seed_policy =',d.get('seed_policy'),'| world_size =',d.get('world_size'))
" 2>/dev/null
exit $_RC
