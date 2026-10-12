---
name: opd-caopd-cluster-setup
description: "How the DiffusionOPD/CA-OPD env is built and run on the minilab SLURM cluster — env path, GPU topology choice, and the build gotchas"
metadata:
  type: reference
---

Env: `conda_envs/envs/diffusionopd` (single env = the original server's `venv_mopd`, i.e. base + all reward deps). Rebuildable from `opd_related/build/0{1..4}_*.sh`. Common env段 in `opd_related/env.sh` — source it, it sets `OPD_MODELS_DIR`/`SD35_PATH`/teacher paths/`PADDLEOCR_PATH`/offline flags/caches.

Pinned stack (matches the original): py3.10.16 · torch 2.6.0+cu126 (**cxx11abi=True** → flash-attn `cu12torch2.6cxx11abiTRUE` wheel) · diffusers 0.33.1 · transformers 4.40.0 · accelerate 1.4.0 · peft 0.10.0 · deepspeed 0.16.4 · numpy 1.26.4 · mmcv-full 1.7.2 (+CUDA ops) · mmdet 2.28.2 · paddlepaddle-gpu 2.6.2 · paddleocr 2.9.1 · hpsv2x 1.2.0 · open_clip 3.3.0 · image-reward 1.5.

**GPU topology — user wants 4 GPUs max** (2026-08-23: "这个服务器没有之前那个服务器的gpu多，所以我们最好只用4张卡"). Upstream configs assume 8 procs (`bs3 × 8 × 3 batches = 72 samples/epoch`). `config/opd.py:_scale_to_nproc` (added here, also in `baselines.py`) reads **`OPD_NPROC`** and doubles `num_batches_per_epoch` 3→6 when it is 4, giving `4 × 3 × 6 = 72` — **gradient-exact**, not an approximation: per-rank `total_batch_size = bs × nbpe` reshapes to micro-batches of size `bs`(=3) either way; `gas = nbpe × T` divides each backward, DDP averages ranks, so both equal the mean over the same 72 samples' micro-losses; each teacher still sees 24 prompts/epoch; `DistributedKRepeatSampler`'s `total_samples % k` assert is safe (k=1). `OPD_NPROC=8` leaves behavior identical to upstream. Bonus: 4 GPUs/arm means **two arms run in parallel** on the 8 H200s.
Scripts: `slurm/opd_arm.slurm` (1-node 4-GPU, the one to use), `opd_eval.slurm`, `eval_repro.slurm`, `resume_test.slurm`. `opd_mopd8.slurm` is 2-node and its mutex cancels by job-name — only for racing one config across A100/H200.
Multi-node NCCL was verified working over **InfiniBand** if 8 GPUs are ever needed again.

**Build gotchas beyond the original server's list:**
- conda `cuda-toolkit` puts headers in `$ENV/targets/x86_64-linux/include/`, NOT `$ENV/include` → mmcv's pure-C++ files (`info.cpp`) fail with `cuda_runtime_api.h: No such file`. Fix: `export CPATH=$ENV/targets/x86_64-linux/include`. (`libcudart.so` IS in `$ENV/lib`, so linking is fine.)
- `MAX_JOBS=48` makes the mmcv nvcc build fail from resource contention; **16 is stable**.
- `timm` ends at 0.6.13 (pinned by image-reward) while open-clip-torch wants ≥1.0.17 — cosmetic, ViT-L-14/H-14 still build fine.
- xformers prints a cu124-vs-cu126 warning — harmless, the code never enables xformers.
- Compute nodes are OFFLINE → `build/04_warm_and_check.sh` must pre-warm PaddleOCR en det/rec/cls and open_clip ViT-L-14 on the login node.
- Must `cd` to the repo root before launching (configs use `os.getcwd()` for dataset paths); `evaluation.py` needs `PYTHONPATH=$OPD_REPO`.
- **Resume was added here** (upstream saves but never loads). `save_ckpt` also writes `train_state.pt` (~300MB) per checkpoint: RAW (non-EMA) weights + AdamW state + EMA shadow + epoch/global_step + CA-OPD `teacher_kl_ema`/`teacher_kl_floor`; written via `os.replace` (atomic) and pruned to the newest only. Load via **`OPD_RESUME=auto`** (scans `save_dir`) or an explicit checkpoint dir; unset/`none` = upstream behavior. **The `lora/` dir holds EMA weights (for eval) — resuming from it would re-stack EMA every restart, hence the separate raw-weight state.** Verified end-to-end by `slurm/resume_test.slurm`.
- ⚠️ **glob trap**: `config.save_dir` contains `[...]` from `run_name`, which glob reads as a character class → any `glob.glob` over it silently returns nothing. Must use `glob.escape(save_dir)`. This bit the first resume implementation (auto-resume found nothing AND stale train_state files never got pruned).

Related: [[opd-caopd-project]], [[cluster-usage-conventions]]
