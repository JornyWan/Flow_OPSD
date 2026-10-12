#!/bin/bash
#SBATCH --account=sablab
#SBATCH --partition=preempt_gpu
#SBATCH --qos=low
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --mem=120G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$O/CA-OPD/code/DiffusionOPSD-rpi
PY=$O/../conda_envs/envs/diffusionopd/bin/python
SHIM=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$O/CA-OPD/code/DiffusionOPD-main/reward_ckpts
cd "$R" || exit 1
# unselected 面板:评测文件**原始顺序**的前 8 条,CFG=1.5 —— 与 fig6 只差"选不选 prompt"。
for pair in "canonUnsel_cfg15_n4:$O/outputs/ab_open3_baseline_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "mspUnsel_cfg15_n4:$O/outputs/ab_open3_multikF4_s20260830_u100/checkpoints/checkpoint-100/lora"; do
  tag=${pair%%:*}; ck=${pair#*:}
  out="$O/paper_figs/qualcfg/${tag}"
  [ -d "$out" ] && { echo "[q] $tag 已存在,跳过"; continue; }
  echo "[q] ===== $tag $(date) ====="
  "$PY" scripts/cross_eval.py --ckpt "$ck" \
    --config sd35_open3 --config_file config/public.py \
    --prompts "$R/data/pickapic/test.txt" --prompt_set_name drawbench \
    --protocol_name "qualunsel_cfg15" --num_steps 4 --sampler flow \
    --guidance_scale 1.5 --seed 42 --batch_size 4 --pickscore_scale raw --n_prompts 8 \
    --images_dir "$out" \
    --out "$O/outputs/qualunsel_${tag}.json" --rewards hpsv2,clipscore,pickscore 2>&1 | grep -avE "[0-9]+%\|"
  echo "[q] $tag 产出 $(ls "$out"/*.png 2>/dev/null | wc -l) 张"
done
