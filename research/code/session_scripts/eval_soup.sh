#!/bin/bash
# checkpoint soup 的评测:平均 ck280/290/300 的 LoRA 权重,不训练。
# 动机:A0 门实测"从 EMA 权重续训"值 +6.4~+9.1 MDE(§4.789)⇒ 平均本身很值钱。
#       若 soup 能救回 msp300@ck300 的衰退,§4.762 的"训练长度边界"就部分是**单点评测的伪影**。
# ⚠️ LoRA 的 mean(B)·mean(A) ≠ mean(B·A),差额是二阶项;相邻 ck 位移仅 ~1.4% ⇒ 二阶 ~2e-4 可忽略。
#    (只对**相邻** checkpoint 成立,远点不可这么做。)
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
SP=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad
sub(){ local tag=$1 dir=$2
  ARM="pi_${tag}" ADAPTER=default CKDIR="$dir" \
  CELLS="drawbench:flow:40 drawbench:flow:4 unseen1k:flow:40 unseen1k:flow:4" \
  REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
  sbatch --parsable --job-name="mev_${tag}" \
    --account=sablab --partition=preempt_gpu --qos=low \
    --gres=gpu:a40:1 --exclude=ai-gpu01,ai-gpu03 \
    "$O/slurm/opsd_eval_matched.slurm"; }
echo "  mspsoup3_300   job=$(sub mspsoup3_300   "$SP/soup_msp300_280-300")"
echo "  canonsoup3_300 job=$(sub canonsoup3_300 "$SP/soup_canon300_280-300")"
