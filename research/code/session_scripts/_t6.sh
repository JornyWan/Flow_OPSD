set -uo pipefail
OUTPUT_DIR="$1"; ARM="$2"
RPI_A1=0; RPI_N2=0; RPI_GATEP_PROBE=0; RPI_CERT_PROBE=0; RPI_QGRAD_PROBE=0
RPI_Q1B_PROBE=0; RPI_Q2A_PROBE=0; RPI_BNOISE_PROBE=0; RPI_BNOISE_SPLIT=0
RPI_RTCREACH_PROBE=0; RPI_QOPSD=0; RPI_A1_MODE=abs; RPI_QOPSD_DIR=""
RPI_GATEA_PROBE=0; RPI_QCORE_SUFFIX=dpm2; RPI_QCORE_FLOW_STEPS=0
SEED=20260830; AB_CONFIG=sd35_open3; UPDATES=100; NPROC=3; SLURM_JOB_ID=999
R="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/CA-OPD/code/DiffusionOPSD-rpi"
if [ "$ARM" = qcorecfg ]; then export RPI_QCORE=1; EXTRA_CFG="--config.opsd.opa_query_sigma=0.464876"; OPA_CFG="$EXTRA_CFG"; fi
cat > "$OUTPUT_DIR/arm_provenance.json" <<EOF || true
{"arm":"$ARM","RPI_A1":"$RPI_A1","RPI_N2":"$RPI_N2","RPI_GATEP_PROBE":"$RPI_GATEP_PROBE","RPI_CERT_PROBE":"$RPI_CERT_PROBE","RPI_QGRAD_PROBE":"$RPI_QGRAD_PROBE","RPI_Q1B_PROBE":"$RPI_Q1B_PROBE","RPI_Q2A_PROBE":"$RPI_Q2A_PROBE","RPI_BNOISE_PROBE":"$RPI_BNOISE_PROBE","RPI_BNOISE_SPLIT":"$RPI_BNOISE_SPLIT","RPI_RTCREACH_PROBE":"$RPI_RTCREACH_PROBE","RPI_QOPSD":"$RPI_QOPSD","RPI_QCORE":"${RPI_QCORE:-0}","RPI_QCORE_SUFFIX":"${RPI_QCORE_SUFFIX:-dpm2}","RPI_QCORE_FLOW_STEPS":"${RPI_QCORE_FLOW_STEPS:-0}","RPI_GATEA_PROBE":"${RPI_GATEA_PROBE:-0}","RPI_KFIT":"${RPI_KFIT:-0}","RPI_KFIT_STEPS":"${RPI_KFIT_STEPS:-}","RPI_KFIT_PS":"${RPI_KFIT_PS:-0}","RPI_KFIT_PS_N":"${RPI_KFIT_PS_N:-}","RPI_KFIT_PS_STEPS":"${RPI_KFIT_PS_STEPS:-}","RPI_KFIT_NMB":"${RPI_KFIT_NMB:-}","PUBLIC_NUM_GROUPS":"${PUBLIC_NUM_GROUPS:-48}","RPI_A1_MODE":"$RPI_A1_MODE","RPI_QOPSD_DIR":"$RPI_QOPSD_DIR","RPI_SEED":"$SEED","extra_cfg":"${EXTRA_CFG:-}","opa_cfg":"${OPA_CFG:-}","opa_cert_arg":"${OPA_CERT_ARG:-}",
 "config":"$AB_CONFIG","mixw":"${MIXW:-}","updates":"$UPDATES","nproc":"$NPROC",
 "train_script_sha256":"$(sha256sum "$R/scripts/train_opsd_ri_sd3.py" | cut -c1-16)",
 "host":"$(hostname)","job":"$SLURM_JOB_ID","gpu":"$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null)"}
EOF
