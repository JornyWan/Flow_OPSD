set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
P=$B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt
L=$B/outputs/ab_open3_baseAA100_s20260830_u100/checkpoints/checkpoint-100/lora
export HF_HUB_OFFLINE=1 TORCHINDUCTOR_COMPILE_THREADS=1
export PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
echo "===== P4 smoke ====="
$PY $B/probes/p4_zo_feasibility.py --repo $R --lora $L --prompts $P \
  --out /tmp/p4_smoke.json --n-prompts 2 --chunk 2 --max-pairs 2 2>&1 | tail -12
echo "rc=$?"
echo "===== P1 smoke ====="
$PY $B/probes/p1_reward_reachability.py --repo $R --lora $L --prompts $P \
  --out /tmp/p1_smoke.json --n-prompts 2 --chunk 2 --seeds 2 2>&1 | tail -12
echo "rc=$?"
