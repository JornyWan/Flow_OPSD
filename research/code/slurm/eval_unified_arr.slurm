#!/bin/bash
#SBATCH --job-name=evaluni
#SBATCH --array=0-7%2
#SBATCH --account=sablab
#SBATCH --partition=sablab-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/slurm_%x_%j.out
#
# UnifiedReward 评测:它不是本地模型调用,而是通过 http://127.0.0.1:17140/v1 访问一个 sglang 服务
# (flow_grpo/unifiedreward_scorer.py 里 AsyncOpenAI 的 base_url 写死了这个地址)。
# 编排: 最后一张卡起 sglang 7B 服务 -> 等就绪 -> 其余卡跑评测 -> 收尾杀服务。
set -uo pipefail
source /midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/env.sh
cd "$OPD_REPO"
NALL=$(nvidia-smi -L | wc -l)
case "$NALL" in 2|4|8) ;; *) echo "[guard] nproc=$NALL 不合适 -> requeue"; sleep 60; scontrol requeue "$SLURM_JOB_ID"; exit 0;; esac
SGPU=$((NALL-1))            # 最后一张卡给 sglang
NPROC=$SGPU                 # 其余跑评测
SGENV=/midtier/minilab/scratch/chw4026/projects/side_project/generative/conda_envs/envs/sglang
export OMP_NUM_THREADS=$(( 32 / NPROC ))
export EVAL_REWARDS=${EVAL_REWARDS:-pickscore,clipscore,aesthetic,hpsv2,imagereward,unifiedreward}

echo "[uni] host=$(hostname) 总卡=$NALL  评测用 0-$((SGPU-1))  sglang 用 $SGPU  $(date)"
# --- 起 sglang 服务 (独占最后一张卡) ---
# flashinfer 会在运行时 JIT 编译 attention kernel,链接时按 $CUDA_HOME/lib64 找 -lcudart/-lcuda。
# env.sh 把 CUDA_HOME 指向 diffusionopd(其 conda cuda-toolkit 只有 lib/,无 lib64) -> ld 找不到库。
# 已建 lib64->lib 软链;这里再显式补 LIBRARY_PATH/LD_LIBRARY_PATH 双保险。
CUDA_VISIBLE_DEVICES=$SGPU HF_HUB_OFFLINE=1 \
  LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib/stubs:${LIBRARY_PATH:-}" \
  LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}" \
  "$SGENV/bin/python" -m sglang.launch_server \
  --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 \
  --chat-template chatml-llava --mem-fraction-static 0.85 --host 127.0.0.1 \
  > "$OPD_ROOT/outputs/sglang_${SLURM_JOB_ID}.log" 2>&1 &
SG_PID=$!
trap 'echo "[uni] 收尾:杀 sglang $SG_PID"; kill $SG_PID 2>/dev/null' EXIT

echo "[uni] 等 sglang 就绪(最多 15 分钟)..."
for i in $(seq 1 90); do
  if curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer flowgrpo" \
       http://127.0.0.1:17140/v1/models 2>/dev/null | grep -q 200; then
    echo "[uni] sglang 就绪 (等了 $((i*10))s)"; break
  fi
  kill -0 $SG_PID 2>/dev/null || { echo "[uni] ❌ sglang 进程已退出,看 sglang_${SLURM_JOB_ID}.log"; exit 1; }
  sleep 10
done
curl -s -H "Authorization: Bearer flowgrpo" http://127.0.0.1:17140/v1/models | head -c 200; echo

# --- 跑评测 ---
LINE=$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" ${UNI_TARGETS:?需要 UNI_TARGETS})
NAME=$(echo "$LINE" | cut -f1); CKPT=$(echo "$LINE" | cut -f2)
GS=$(echo "$LINE" | cut -f3); NS=$(echo "$LINE" | cut -f4)
CKARG=(); [ "$CKPT" != "base" ] && CKARG=(--checkpoint_path "$CKPT")
OUT="$OPD_ROOT/outputs/eval/${NAME}_drawbench"; mkdir -p "$OUT"
echo "[uni] ===== $NAME / drawbench  n=$NS gs=$GS ====="
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((SGPU-1))) \
python -m torch.distributed.run --nproc_per_node=$NPROC --master_port=29540 scripts/evaluation.py \
  "${CKARG[@]}" --model_type sd3 --dataset drawbench --guidance_scale "$GS" \
  --max_samples "$NS" --mixed_precision fp16 --output_dir "$OUT" 2>&1 | grep -avE "[0-9]+%\|"
echo "[uni] --- 结果 ---"; cat "$OUT/average_scores.json" 2>/dev/null; echo
echo "UNI_EVAL_DONE"
