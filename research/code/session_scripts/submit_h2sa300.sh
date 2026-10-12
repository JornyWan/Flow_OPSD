#!/bin/bash
# §4.894/§4.895/§4.896 晋级后执行:H200×4(--exclusive)顺序 = 单 reward split(clipscore → imagereward,各 100 步,~9 h)→ split300 s0830 → split300 s0905 → 门控两臂。
# 仅在 judge_sa888 = 晋级 时运行。
set -e
cd /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
R=$PWD
sub(){ # $1=job-name $2=seed $3=outdir $4=dependency(可空) $5=config $6=updates $7=time
  local dep=""; [ -n "$4" ] && dep="--dependency=$4"
  ARM=multikF4sa MK_SPLIT=0.5 SEED=$2 UPDATES=$6 OPT_DIAG=1 EXPECT_NPROC=4 AB_CONFIG=config/public.py:$5 OUTPUT_DIR=$3 \
  sbatch --parsable --exclusive --account=minilab --partition=minilab-gpu --qos=high --gres=gpu:h200:4 --cpus-per-task=32 --mem=300G \
    --time=$7 --requeue --exclude=ai-gpu01,ai-gpu03 --job-name=$1 $dep \
    --output=$R/outputs/slurm_%x_%j.out --export=ALL slurm/ab_open3_v6.slurm; }
J1=$(sub rs_clipscore_sa_h200 20260830 $R/outputs/ab_clipscore_multikF4sa_s20260830_u100 "" sd35_clipscore 100 12:00:00)
J2=$(sub rs_imagereward_sa_h200 20260830 $R/outputs/ab_imagereward_multikF4sa_s20260830_u100 "afterany:$J1" sd35_imagereward 100 12:00:00)
J3=$(sub h2_sa300_s30 20260830 $R/outputs/h2_sa300_s30 "afterany:$J2" sd35_open3 300 2-00:00:00)
J4=$(sub h2_sa300_s05 20260905 $R/outputs/h2_sa300_s05 "afterany:$J3" sd35_open3 300 2-00:00:00)
echo "H200 chain: $J1 $J2 $J3 $J4"
scontrol update JobId=1077774 Dependency=afterany:$J4
echo "gate300_s30 1077774 -> afterany:$J4(1077775 仍 afterany 1077774)"
EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=4320 nohup bash $R/codex_research/watch_matched_ckpt.sh h2sa30 $R/outputs/h2_sa300_s30 100 150 200 250 300 > $R/outputs/watch_h2sa30.log 2>&1 &
EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=4320 nohup bash $R/codex_research/watch_matched_ckpt.sh rsclipscoresa $R/outputs/ab_clipscore_multikF4sa_s20260830_u100 50 100 > $R/outputs/watch_rsclipscoresa.log 2>&1 &
EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=5760 nohup bash $R/codex_research/watch_matched_ckpt.sh rsimagerewardsa $R/outputs/ab_imagereward_multikF4sa_s20260830_u100 50 100 > $R/outputs/watch_rsimagerewardsa.log 2>&1 &
EVAL_PART=preempt_gpu EVAL_QOS=low WAIT_MIN=8640 nohup bash $R/codex_research/watch_matched_ckpt.sh h2sa05 $R/outputs/h2_sa300_s05 100 150 200 250 300 > $R/outputs/watch_h2sa05.log 2>&1 &
echo "watchers started"
