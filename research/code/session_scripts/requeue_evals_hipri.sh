#!/bin/bash
# 把我的评测从 preempt_gpu/qos=low(prio 50,已饿 40+ 分钟)搬到 sablab-gpu(prio 1000)。
# 安全论证:sablab-high 只抢占 QOS 名为 "low" 的作业;
#   gpu02 = 4×sablab-high、gpu04 = 3×sablab-high + 1×sablab-low ⇒ 都不可被 sablab-high 抢占;
#   4 个 low 作业(lus4017)全在 gpu05,而 gpu05 不在 sablab-gpu 分区 ⇒ 够不着。
#   仍显式 --exclude ai-gpu05 作双保险。
set -u
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$O" || exit 1
EX="ai-gpu01,ai-gpu03,ai-gpu05"
echo "== 取消旧的 low 版本 =="
scancel 1047171 1047286 1047287 1047298 1047299 1047300 1047301 1047302 1047318 1047319 2>&1 | sed 's/^/  /'
sleep 3
K=checkpoints/checkpoint-100
sub(){ # jobname ARM adapter ckdir cells
  local jn=$1 arm=$2 ad=$3 ck=$4 cells=$5
  local J
  J=$(ARM="$arm" ADAPTER="$ad" CKDIR="$ck" CELLS="$cells" \
      REWARDS=hpsv2,clipscore,aesthetic,pickscore,imagereward \
      sbatch --parsable --job-name="$jn" --account=sablab --partition=sablab-gpu \
      --gres=gpu:a40:1 --exclude="$EX" "$O/slurm/opsd_eval_matched.slurm" 2>&1)
  echo "  $jn -> $J"
}
echo "== 重投(sablab-gpu, prio 1000) =="
sub lownfe_basemodel dflt_basemodel none "" "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub lownfe_basenp3 dflt_basenp3 default "$O/outputs/ab_open3_baseline_np3_s20260830_u100/$K" "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub lownfe_qcore   dflt_qcore   default "$O/outputs/ab_open3_qcore_s20260830_u100/$K"        "drawbench:flow:2 drawbench:flow:4 drawbench:flow:6"
sub lownfe_lr2x    dflt_lr2x    default "$O/outputs/ab_open3_lr2x_s20260830_u100/$K"         "drawbench:flow:4"
sub lownfe_lr8x    dflt_lr8x    default "$O/outputs/ab_open3_lr8x_s20260830_u100/$K"         "drawbench:flow:4 drawbench:flow:10 drawbench:flow:20"
sub qflow10_qopsd  dflt_qopsd   default "$O/outputs/ab_open3_qopsd_s20260830_u100/$K"        "drawbench:flow:10"
sub qflow10_qbt    dflt_qbt     default "$O/outputs/ab_open3_qbt_s20260830_u100/$K"          "drawbench:flow:10"
sub mev_pi_qopsd_100 pi_qopsd_100 default "$O/outputs/ab_open3_qopsd_s20260830_u100/$K"      "drawbench:flow:40"
sub mev_pi_qbt_100   pi_qbt_100   default "$O/outputs/ab_open3_qbt_s20260830_u100/$K"        "drawbench:flow:40"
sub mev_pi_lr8x200_150 pi_lr8x200_150 default "$O/outputs/ab_open3_lr8x_s20260830_u200/checkpoints/checkpoint-150" "drawbench:flow:40"
