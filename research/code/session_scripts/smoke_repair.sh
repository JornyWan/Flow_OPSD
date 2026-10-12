set -uo pipefail
G=/midtier/minilab/scratch/chw4026/projects/side_project/generative
A=$G/agentic
PY=$G/conda_envs/envs/diffusionopd/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/midtier/minilab/scratch/chw4026/.cache/huggingface
$PY $A/scripts/run_repair.py \
  --metadata $G/project3/tts/psp/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl \
  --prompt-ids $A/outputs/smoke_ids.json \
  --out /midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/repair_smoke2.csv --img-out /midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/repair_smoke2_imgs \
  --geneval-dir $G/project3/tts/psp/geneval \
  --model-config $G/opd_related/src/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
  --model-path $G/project3/tts/psp/geneval/objdet \
  --base-seeds 1
echo "rc=$?"
cat /midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/repair_smoke2.csv 2>/dev/null | head -5
