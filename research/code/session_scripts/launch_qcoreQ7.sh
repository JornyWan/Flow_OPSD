#!/bin/bash
set -u
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
cd "$B" || exit 1
export SCRIPT=slurm/ab_open3_v6.slurm                      # §4.575:v6 才把 QCORE/SUFFIX/extra_cfg 写进 provenance
export OPA_CFG="--config.opsd.opa_query_sigma=0.464876"    # kq=7 的节点值(§4.317 的调度表)
export REQ=4 ACCEPT="3 4"                                  # 铁律:申请整节点卡数,EXPECT_NPROC="3 4"
exec bash "$SP/wait_and_launch_v4.sh" qcorecfg 20260830 100 \
     "$B/outputs/ab_open3_qcoreQ7_s20260830_u100" h200 3 sablab-gpu sablab "ai-gpu17 ai-gpu18"
