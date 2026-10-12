#!/bin/bash
#SBATCH --job-name=netprobe
#SBATCH --time=00:03:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad/netprobe_%j.out
echo "host=$(hostname)"
timeout 10 curl -sS -o /dev/null -w "api.wandb.ai HTTP=%{http_code} time=%{time_total}s\n" https://api.wandb.ai/ 2>&1 || echo "curl 失败(无外网或被墙)"
timeout 10 python -c "import socket; socket.create_connection(('api.wandb.ai',443),timeout=8); print('TCP 443 可达')" 2>&1 | tail -1
