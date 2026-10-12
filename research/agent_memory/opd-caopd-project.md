---
name: opd-caopd-project
description: "CA-OPD project — DiffusionOPD (on-policy distillation for diffusion) reproduction + conflict-averse extension, migrated from the colligo/Pluto server to the minilab SLURM cluster on 2026-08-23"
metadata:
  type: project
---

**CA-OPD** = Conflict-Averse / Value-Gradient Multi-Teacher **On-Policy Distillation**. Built on **DiffusionOPD** (ali-vilab, SIGGRAPH Asia 2026, arXiv 2605.15055) — NOT the 3D "openable part detection" OPD. SD3.5-Medium@512 + LoRA r32/α64; 3 task teachers (Aes/OCR/GenEval) round-robin distilled into one student via closed-form per-step KL.

Originally run on the **colligo/Pluto** server (ephemeral pods, `/mnt/localssd`, S3 bundles, uv venv, 8×H200 single node). **Migrated 2026-08-23** to this cluster; everything lives in `projects/side_project/generative/opd_related/` (see its `README_MIGRATION.md`). Full archive mirrored from the PRIVATE HF repo `Jorny/CA-OPD` (94.17 GB / 8597 files, verified complete) — needs a Jorny-account HF token; the repo also carries `conversation/opd_project_notes.md`, the detailed handoff log worth re-reading before any new work.

**Migration validated 2026-08-23**: re-ran all archived evals on the rebuilt env — **92/96 comparisons within sampling noise**; overall z mean −0.15, sd 0.90 (≈ standard normal) ⇒ no systematic bias. Official Student LoRA (external fixed artifact) matched on all 6 axes within **0.22%**. Full writeup: `opd_related/REPRODUCTION_VERDICT.md`; regenerate with `slurm/eval_all.slurm` + `compare_eval.py`. Judge by "within sampling noise", never bitwise — different GPUs' fp16 kernels diverge over 40 sampling steps into different images.

**Status when migrated:**
- **Reproduction is faithful and validated**: mopd ckpt-721 → GenEval 0.958 / OCR 0.939 / PickScore 23.98 / Clip 0.298 / HPS 0.342 / Aes 6.172, matching the paper's 6/8 columns ≤1%. Official Student release cross-check agrees (triple-validated).
- **CA-OPD honest verdict**: converges FASTER (hits the lagging aesthetics axis ~180 steps earlier) but plateaus at a slightly LOWER ceiling than uniform mopd. Not a SOTA win — frame as convergence efficiency + controllability.
- ~12 explored arms: caopd / regret / conflict / tau-sweep / **poe (λ sweep)** / cagrad+pcgrad / robust (neutral) / snr (hurts) / anneal (hurts most) / vteacher+vvalue (virtual teachers, no RL expert — clearly worse) / 3 multi-task RL baselines.
- **POE is a clean MONOTONIC trade-off, not undertraining** (assembled 2026-08-23; nobody had tabulated it). As λ rises 1.0→1.5→2.0→3.0: Aesthetic climbs 6.172→6.256→6.364→6.406 and HPS 0.342→0.345→0.347→0.344, while **GenEval collapses 0.958→0.901→0.766→0.501**. Amplifying the per-teacher residual helps soft continuous rewards and destroys the hard compositional constraint. The interesting operating point is **λ=1.5** (beats mopd@721 on Aes+HPS at only 271 steps, −0.057 GenEval); **λ=1.25 (`mopd_poe_l125`) was defined but NEVER RUN** — the obvious gap. λ=2.0/3.0 are already past the cliff, so training them longer is not informative.
- **Virtual teachers are a ZERO-gain negative, stronger than the notes said**: `vt_*` pickscore 0.858-0.861 vs **`base_gs4.5` 0.863** — the reward-gradient virtual teacher does not even beat the base model with CFG; `vvalue_*` (0.843-0.849) is below it. Real Aes-teacher distillation (`sopd_pickscore`) reaches 0.924. The RL expert is doing essential work.
- **`mtnft690` is a training COLLAPSE, not a converged baseline** (geneval 0.054 / ocr 0.010, vs `mtnft570` 0.639/0.854). Report MT-NFT at 570; using 690 compares against someone's diverged run.
- **The "high conflict" arms have no discriminating power yet**: `conflict*` and `mopdconflict*` both degrade similarly (OCR 0.80-0.87, Aes ~5.5) and stopped at ~421, far from converged.
- **`regret` is the only arm not clearly worse on any axis** (regret_721: GenEval 0.963 / OCR 0.938 / Aes 6.157 vs mopd 0.958/0.939/6.172) — but all differences are within noise, so no win can be claimed.

**Hard-won lesson recorded in the project**: an earlier "we beat the paper 0.985 vs 0.929" claim was a **naive/circular min-max normalization artifact** (floor = own zero-shot, ceiling = best-of-own-runs, 6 metrics vs the paper's 8). User's decision: **report per-axis only, never an aggregate Average**. Do not reintroduce an aggregate score.

Related: [[opd-caopd-cluster-setup]], [[cluster-usage-conventions]]
