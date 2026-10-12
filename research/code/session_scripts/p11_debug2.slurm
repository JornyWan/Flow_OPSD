#!/bin/bash
#SBATCH --account=sablab --partition=preempt_gpu --qos=low
#SBATCH --nodes=1 --cpus-per-task=8 --mem=96G --gres=gpu:a40:1 --time=00:30:00
#SBATCH --exclude=ai-gpu01,ai-gpu03
#SBATCH --job-name=p11dbg --output=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/p11_debug2_%j.out
OPD_ROOT=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related; R=$OPD_ROOT/CA-OPD/reference_papers/DiffusionOPSD-main
cd $R; export PYTHONPATH="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/opsd_shim:$R" PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4 ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
[ -f "$HOME/.secrets/hf_jorny.env" ] && . "$HOME/.secrets/hf_jorny.env"
$OPD_ROOT/../conda_envs/envs/diffusionopd/bin/python /midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/p11_debug2.py $R $OPD_ROOT/outputs/ab_open3_baseline_s20260830_u300/checkpoints/checkpoint-300/lora $OPD_ROOT/outputs/p11/dirs_pilot.pt 2>&1 | grep -v "Warn\|warn\|XFORMERS\|PyTorch 2\|Python 3\|Please\|Memory-eff"
