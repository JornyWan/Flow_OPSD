set -uo pipefail
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$B/CA-OPD/code/DiffusionOPSD-rpi
PY=$B/../conda_envs/envs/diffusionopd/bin/python
export HF_HUB_OFFLINE=1 PYTHONPATH=$B/CA-OPD/code/shim_rpi:$R/src:$R
$PY $B/probes/p4_zo_feasibility.py --repo $R \
  --lora $B/outputs/ab_open3_baseAA100_s20260830_u100/checkpoints/checkpoint-100/lora \
  --prompts $B/CA-OPD/reference_papers/DiffusionOPSD-main/data/pickapic/test.txt \
  --out /tmp/p4_smoke.json --n-prompts 2 --chunk 2 --max-pairs 4
echo "rc=$?"
