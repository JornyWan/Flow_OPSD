#!/bin/bash
# sbatch 的 "I/O error writing script/environment to file" 期间的绕行路径。
# 已验证:同一时刻 srun 分配成功、sbatch 失败 ⇒ 故障只在 batch 脚本落盘。
# 用法: srun_arm.sh ARM SEED UPDATES OUTDIR NODE GRES PART ACCT NP
set -u
export ARM=$1 SEED=$2 UPDATES=$3 OUTPUT_DIR=$4
NODE=$5; GRES=$6; PART=$7; ACCT=$8; export EXPECT_NPROC=$9
B=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
cd "$B" || exit 1
exec srun --job-name="$ARM" --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 \
     --mem=300G --time=24:00:00 --gres="$GRES" --nodelist="$NODE" \
     --partition="$PART" --account="$ACCT" \
     bash slurm/ab_open3.slurm
