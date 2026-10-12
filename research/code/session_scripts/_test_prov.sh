set -uo pipefail
OUTPUT_DIR="$1"; ARM="$2"
# 模拟 v6 顶层恒 export 的变量
RPI_A1=0; RPI_N2=0; RPI_GATEP_PROBE=0; RPI_CERT_PROBE=0; RPI_QGRAD_PROBE=0
RPI_Q1B_PROBE=0; RPI_Q2A_PROBE=0; RPI_BNOISE_PROBE=0; RPI_BNOISE_SPLIT=0
RPI_RTCREACH_PROBE=0; RPI_QOPSD=0; RPI_A1_MODE=abs; RPI_QOPSD_DIR=""
RPI_GATEA_PROBE=0; RPI_QCORE_SUFFIX=dpm2; RPI_QCORE_FLOW_STEPS=0
SEED=20260830; AB_CONFIG=sd35_open3; UPDATES=100; NPROC=3
R="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPSD-rpi"
SLURM_JOB_ID=999999
# kfit 臂才 export 的:非 kfit 臂故意**不定义**,考验 ${X:-} 兜底
if [ "$ARM" = kfit ]; then export RPI_KFIT=1 RPI_KFIT_STEPS=8 RPI_KFIT_PS=1 RPI_KFIT_PS_N=4 RPI_KFIT_PS_STEPS=2 RPI_KFIT_NMB=3; fi
if [ "$ARM" = qcoresolv ]; then export RPI_QCORE=1 RPI_QCORE_SUFFIX=flow; fi
cat > "$OUTPUT_DIR/arm_provenance.json" <<EOF || true
{"arm":"$ARM","RPI_A1":"$RPI_A1","RPI_N2":"$RPI_N2","RPI_GATEP_PROBE":"$RPI_GATEP_PROBE","RPI_CERT_PROBE":"$RPI_CERT_PROBE","RPI_QGRAD_PROBE":"$RPI_QGRAD_PROBE","RPI_Q1B_PROBE":"$RPI_Q1B_PROBE","RPI_Q2A_PROBE":"$RPI_Q2A_PROBE","RPI_BNOISE_PROBE":"$RPI_BNOISE_PROBE","RPI_BNOISE_SPLIT":"$RPI_BNOISE_SPLIT","RPI_RTCREACH_PROBE":"$RPI_RTCREACH_PROBE","RPI_QOPSD":"$RPI_QOPSD","RPI_QCORE":"${RPI_QCORE:-0}","RPI_QCORE_SUFFIX":"${RPI_QCORE_SUFFIX:-dpm2}","RPI_QCORE_FLOW_STEPS":"${RPI_QCORE_FLOW_STEPS:-0}","RPI_GATEA_PROBE":"${RPI_GATEA_PROBE:-0}","RPI_KFIT":"${RPI_KFIT:-0}","RPI_KFIT_STEPS":"${RPI_KFIT_STEPS:-}","RPI_KFIT_PS":"${RPI_KFIT_PS:-0}","RPI_KFIT_PS_N":"${RPI_KFIT_PS_N:-}","RPI_KFIT_PS_STEPS":"${RPI_KFIT_PS_STEPS:-}","RPI_KFIT_NMB":"${RPI_KFIT_NMB:-}","PUBLIC_NUM_GROUPS":"${PUBLIC_NUM_GROUPS:-48}","RPI_A1_MODE":"$RPI_A1_MODE","RPI_QOPSD_DIR":"$RPI_QOPSD_DIR","RPI_SEED":"$SEED",
 "config":"$AB_CONFIG","mixw":"${MIXW:-}","updates":"$UPDATES","nproc":"$NPROC",
 "train_script_sha256":"$(sha256sum "$R/scripts/train_opsd_ri_sd3.py" | cut -c1-16)",
 "host":"$(hostname)","job":"$SLURM_JOB_ID","gpu":"$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)"}
EOF
