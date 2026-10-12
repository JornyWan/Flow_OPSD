#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
cd "$B" || exit 1
export SCRIPT=slurm/ab_open3_v6.slurm
export REQ=4 ACCEPT="3 4"
# inner4 是 A100 训练 ⇒ 等成本对照必须同硬件,故也投 A100(§4.568 三条轴)
exec bash "$SP/wait_and_launch_v4.sh" multik4 20260830 100 \
     "$B/outputs/ab_open3_multik4_s20260830_u100" a100 3 sablab-gpu sablab \
     "ai-gpu06 ai-gpu07 ai-gpu08 ai-gpu09"
