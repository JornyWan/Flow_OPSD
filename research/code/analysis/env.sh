# DiffusionOPD / CA-OPD —— 本集群统一环境段。用法: source <此文件>
GEN=/midtier/minilab/scratch/chw4026/projects/side_project/generative
export OPD_ROOT=$GEN/opd_related
export OPD_ENV=$GEN/conda_envs/envs/diffusionopd
export OPD_REPO=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main

# --- env 激活: 直接上 PATH,不用 conda activate (抗集群维护抖动) ---
export PATH="$OPD_ENV/bin:$PATH"
export CONDA_PREFIX="$OPD_ENV"
export CUDA_HOME="$OPD_ENV"

# --- 模型路径 (config/opd.py & baselines.py 读这些) ---
export OPD_MODELS_DIR=$OPD_ROOT/models
export SD35_PATH=$OPD_MODELS_DIR/stable-diffusion-3.5-medium
export AES_TEACHER_PATH=$OPD_MODELS_DIR/DiffusionOPD/AesTeacher/lora
export OCR_TEACHER_PATH=$OPD_MODELS_DIR/DiffusionOPD/OCRTeacher/lora
export GENEVAL_TEACHER_PATH=$OPD_MODELS_DIR/DiffusionOPD/GenEvalTeacher/lora
export PADDLEOCR_PATH=$OPD_REPO/reward_ckpts/paddleocr

# --- 缓存全部指到 scratch (本集群约定) ---
export XDG_CACHE_HOME=/midtier/minilab/scratch/chw4026/.cache
export HF_HOME=/midtier/minilab/scratch/chw4026/.cache/huggingface
export TORCH_HOME=/midtier/minilab/scratch/chw4026/.cache/torch
export TMPDIR=/midtier/minilab/scratch/chw4026/tmp

# --- 计算节点离线 ---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# wandb: 2026-08-28 实测计算节点(ai-cpu01 / ai-gpu16)**有外网**(api.wandb.ai TCP 443 可达, 0.07s),
# 故由 offline 切为 online。若某节点无网,wandb 会自行退回 offline 并在 run 目录留下可 sync 的数据,
# 不会中断训练(WANDB_MODE 可用 WANDB_FORCE_OFFLINE=1 强制回退)。
export WANDB_MODE=${WANDB_FORCE_OFFLINE:+offline}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=jorny-cornell-university
export WANDB_PROJECT=${WANDB_PROJECT:-diffusionopsd}
export WANDB_INIT_TIMEOUT=120   # 连接慢时不要卡死训练
export WANDB_START_METHOD=thread
export WANDB_DIR=/midtier/minilab/scratch/chw4026/.cache/wandb

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$OPD_REPO:${PYTHONPATH:-}"   # evaluation.py 按路径跑,需要能 import flow_grpo

# --- 训练步数上限 (1 epoch = 1 optimizer step) ---
# 放在 env.sh 而非 slurm 脚本里,因为 SLURM 在**提交时**就固化了批处理脚本,
# 改 slurm 脚本影响不到已排队的作业;env.sh 是运行时 source 的,改它立刻对全队列生效。
export OPD_NUM_EPOCHS=${OPD_NUM_EPOCHS:-800}
export OPD_EVAL_FREQ=${OPD_EVAL_FREQ:-150}
