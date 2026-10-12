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
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
SHIM=$SP/opsd_shim          # ← 与 qual3.slurm 一致:PYTHONPATH 要的是 opsd_shim,不是 scratchpad 根
export PYTHONPATH="$SHIM:$R"
export PUBLIC_POLICY_WORLD_SIZE=4 PUBLIC_N_GPUS=4 PUBLIC_LAUNCH_WORLD_SIZE=4
export ZIMAGE_HEAVY_BRIDGE=0 ZIMAGE_HEAVY_DIFF_BRIDGE=0 WANDB_MODE=offline
export REWARD_CKPT_PATH=$O/CA-OPD/code/DiffusionOPD-main/reward_ckpts
PF=$O/paper_figs/qual3/prompts_gain.txt
cd "$R" || exit 1
RC=0
# tag:config:guidance:nfe:batch —— base model(无 LoRA),只改推理协议
for spec in "canon_cfg15_n4:config/public.py:1.5:4:4:$O/outputs/ab_open3_baseline_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "msp_cfg15_n4:config/public.py:1.5:4:4:$O/outputs/ab_open3_multikF4_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "canon_cfg20_n4:config/public.py:2.0:4:4:$O/outputs/ab_open3_baseline_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "msp_cfg20_n4:config/public.py:2.0:4:4:$O/outputs/ab_open3_multikF4_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "canon_cfg30_n4:config/public.py:3.0:4:4:$O/outputs/ab_open3_baseline_s20260830_u100/checkpoints/checkpoint-100/lora" \
            "msp_cfg30_n4:config/public.py:3.0:4:4:$O/outputs/ab_open3_multikF4_s20260830_u100/checkpoints/checkpoint-100/lora"; do IFS=: read -r tag cfgf gs nfe bs ck <<< "$spec"
  out="$O/paper_figs/qualcfg/${tag}"
  [ -d "$out" ] && { echo "[cfg] $tag 已存在,跳过"; continue; }
  echo "[cfg] ===== $tag  cfg=$cfgf gs=$gs nfe=$nfe $(date) ====="
  "$PY" scripts/cross_eval.py --ckpt "$ck" \
    --config sd35_open3 --config_file "$cfgf" \
    --prompts "$PF" --prompt_set_name drawbench \
    --protocol_name "diag_${tag}" --num_steps "$nfe" --sampler flow \
    --guidance_scale "$gs" --seed 42 --batch_size "$bs" --pickscore_scale raw \
    --images_dir "$out" --out "$O/outputs/qualcfg_${tag}.json" \
    --rewards hpsv2,clipscore,pickscore 2>&1 | grep -avE "[0-9]+%\|"
  rc=${PIPESTATUS[0]}; n=$(ls "$out"/*.png 2>/dev/null | wc -l)
  echo "[cfg] $tag rc=$rc -> $n 张"
  { [ "$rc" -ne 0 ] || [ "$n" -eq 0 ]; } && { echo "[cfg] ❌ $tag 失败"; RC=1; }
done
echo "[cfg] exit=$RC $(date)"; exit $RC
