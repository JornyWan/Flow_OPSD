cd "$(dirname "$0")"
while IFS= read -r q; do echo "=== $q"; python hf.py "$q" 20 260 2025-06; sleep 8; done << 'QS'
DiffusionNFT with reward gradient differentiable reward combination
early denoising steps layout reinforcement learning late steps direct reward fine-tuning
timestep-specific LoRA experts reinforcement learning diffusion alignment
noise-level dependent choice of gradient estimator diffusion reward fine-tuning
group relative advantage forward noising final images high noise only
first-order reward gradient versus policy gradient sample efficiency diffusion comparison
layout composition high noise timesteps reward alignment text-to-image coarse-to-fine RL
combining direct reward fine-tuning with reinforcement learning diffusion unified
on-policy self-distillation diffusion reward gradient target
separate LoRA for high-noise and low-noise timesteps preference optimization
QS
