# CA-OPD paper — ground-truth context (for all agents)

## What this project is
We reproduced **DiffusionOPD** (arXiv-2605.15055v1, SIGGRAPH Asia 2026; papers at `/home/colligo/opd_related/papers/arXiv-2605.15055v1/`)
end-to-end on the colligo/Pluto cluster, then proposed and tested **our own improvement: CA-OPD**.
Repo: `/home/colligo/opd_related/DiffusionOPD-main/`. Base model SD3.5-Medium @512, LoRA r32/α64, 40-step ODE eval.

## OPD in one paragraph (base theory to reuse)
On-policy distillation: the student rolls out its OWN denoising trajectory; at each reverse step the student & teacher
one-step Gaussian transition kernels share the SAME covariance (depends only on scheduler + noise level a) and are
evaluated at the SAME student state x_{t_j}, so they differ ONLY in mean. Hence the per-step reverse-KL is closed-form
= ||μ_S − μ_T||²/(2σ_j²) ∝ ||v_θ − v_φ||² — dense, low-variance (pathwise, no REINFORCE) per-step velocity matching.
Multi-teacher: round-robin over M task teachers, losses summed uniformly (grad-accum = M). Key eqs: method.tex
eq:opd-diffusion (SDE KL), eq:opd-ode (ODE L2, DEFAULT), §Discussion (closed-form KL == PPO in expectation, lower variance).

## OUR CONTRIBUTION — CA-OPD (Conflict-Averse / Pareto multi-teacher OPD)
Problem: uniform per-task loss avg = LINEAR scalarization → student converges to the per-step teacher-MEAN velocity
v_S→(1/M)Σ v_{φ_m}, which only reaches the CONVEX HULL of the per-task Pareto front and is Pareto-DOMINATED when
teacher objectives conflict (paper's own data: DiffusionOPD ≤ each teacher on that teacher's axis; multi-task lags on aesthetics).
Fix: adaptive **Chebyshev (min-max) reweighting** over teachers — EMA of each teacher's per-step KL → softmax weights
up-weight the currently WORST-matched teacher → provably reaches the full (incl. non-convex) Pareto set; τ→∞ recovers mopd.
Why rigorous & novel: OPD's shared-covariance/shared-state construction makes the M per-task gradients live in the SAME
isotropic metric with a SHARED Jacobian J_θ=∇_θ v_θ → their conflict angles are geometrically meaningful, so multi-objective
gradient surgery (CAGrad/MGDA) / Chebyshev scalarization is well-posed HERE in a way it isn't for generic MTL.
Impl: `config/opd.py:mopd_caopd` (train.ca_opd/ca_opd_tau=0.5/ca_opd_ema=0.9); trainer edits in `scripts/train_sd3_opd.py`
(tag samples w/ teacher_idx; teacher_kl_ema; Chebyshev-weighted distill_loss). Baseline mopd unchanged (getattr defaults).

Theory check that ruled out a class of ideas: for EQUAL-covariance Gaussians forward-KL = reverse-KL = the same L2, so
"symmetrizing the KL" is vacuous in OPD. (Use this as a rigor signal.)

## Other ideas tested (for the ablation section)
- Idea2 `mopd_robust`: Huber/trust-region on velocity residual (robust to teacher reward-hacking). Result: NEUTRAL (~mopd).
- Idea3 `mopd_snr`: Min-SNR-γ per-step weighting. Result: HURTS badly (down-weighting high-noise steps kills GenEval/global structure).
- Idea4 `mopd_anneal`: SDE→ODE noise annealing. Result: HURTS most (early SDE phase degrades student; confirms paper's ODE>SDE).
These are honest negatives WITH mechanistic explanations — good ablation content showing CA-OPD's win is non-trivial.

## ⚠️ AUTHORITATIVE HONEST VERDICT — read data/RESULTS_at721.md
The @541 snapshot below was optimistic. At convergence (@721/@781) CA-OPD does NOT beat uniform-mopd on the aggregate
(CA-OPD ~0.977 vs mopd 0.985; ~1%, near noise). CA-OPD **converges FASTER** (reaches the lagging aesthetics axis ~180 steps
earlier) but plateaus at a slightly LOWER ceiling. Robust ≈ mopd (best OCR at convergence). SNR/anneal clearly hurt.
PAPER FRAMING (honest, do NOT overclaim a SOTA win): contribution = rigorous multi-objective theory + convergence-EFFICIENCY /
per-axis controllability (τ trades speed vs ceiling) + honest 4-idea ablation + faithful reproduction. Theory motivation was
CORRECTED by the independent verifier: it is REWARD-space frontier selection, NOT non-convex loss-hull (see sections/theory_review.md).

## RESULTS SO FAR
### Faithful reproduction (all match paper Table 1; n=256/512, gs=4.5 student, geneval/ocr rule-based, drawbench model-based)
- SD3.5-M w/o CFG: 0.32/0.11/20.6/0.238/0.201/5.18 ; +CFG: 0.59/0.56/22.4/0.286/0.283/5.42
- GenEval Teacher: GenEval 0.97, OCR 0.42, Pick 22.0 ; OCR Teacher: OCR 0.94, Pick 22.3 ; Aes Teacher: Pick 24.0/Clip .296/HPS .347/Aes 6.22 (≈exact)
- DiffusionOPD Ours (=our mopd ckpt-721): GenEval 0.958/OCR 0.939/Pick 23.98/Clip 0.298/HPS 0.342/Aes 6.172 ; official-Student release cross-check ≈ identical → triple-validated.
### CA-OPD vs uniform-mopd — SAME-STEP @541 (n=256, normalized score = paper-anchored min-max avg over 6 metrics)
| model@541 | GenEval | OCR | Pick | Clip | HPS | Aes | score | Δ |
| mopd | 0.947 | 0.939 | 23.92 | 0.297 | 0.340 | 6.120 | 0.9666 | — |
| CA-OPD | 0.955 | 0.927 | 23.95 | 0.297 | 0.341 | 6.172 | 0.9766 | +0.010 ✅ |
| robust | 0.944 | 0.927 | 23.94 | 0.298 | 0.338 | 6.108 | 0.9622 | −0.004 |
| SNR | 0.777 | 0.883 | 23.44 | 0.295 | 0.313 | 5.756 | 0.800 | −0.166 |
| anneal | 0.860 | 0.855 | 22.92 | 0.298 | 0.295 | 5.650 | 0.759 | −0.207 |
CA-OPD wins driven by AESTHETICS (its target axis): reaches mopd's OWN best-@721 aesthetic (6.172) already at ckpt-541 (~180 steps earlier).
uniform-mopd best over full trajectory = ckpt-721, score 0.985.
**@721 definitive eval RUNNING** (job 97f2db24: caopd721/robust721/snr721/anneal721/caopd781 vs mopd@721) — will finalize.

## Infra / where things live
- S3 root: `s3://dit-scale-up/cwan/opd_reproduction/`
  - training runs + checkpoints: `runs/{mopd,mopd_caopd,mopd_robust,mopd_snr,mopd_anneal,mt_nft,mt_grpoguard}/logs/<run_name>/checkpoints/checkpoint-N/lora/`
  - eval results: `eval/<name>_<dataset>/average_scores.json` (+ images/ sample jpgs)  — names: curve_<step> (=mopd trajectory), caopd<step>, robust<step>, snr<step>, anneal<step>, base_gs*, {Aes,OCR,GenEval}Teacher, Student, mopd_ckpt721
  - dashboard: `viz/index.html` → https://dit-scale-up.s3.us-west-2.amazonaws.com/cwan/opd_reproduction/viz/index.html
- training logs (per-epoch reward proxy for convergence curves): `/logging-fs/logs/pluto/jobs/cwan/pluto-prod-cwan-<uniqueName>/run-*/pod-0*/main_scripts/1/current`
  grep `monitor_` / `proxy` / `Epoch N` for the reward-vs-step curves (this is how the paper's Fig main_curve is made — auto-logged to wandb offline during training).
- comparison engine: `/mnt/localssd/opd/compare_ideas.py` (scans eval/, builds per-model trajectory, picks best ckpt, normalized-avg score).
- eval how-to: `s3://…/pod_main_eval.sh` (RUNS mode: newline "name|gs|s3src"; DATASETS, MAXSAMPLES, EVAL_REWARDS). Student gs=4.5 all datasets.
- MULTI-TASK BASELINES still training: mt_nft (job 2f936f20), mt_grpoguard (job 8d85289a); need eval when converged for the full table. Cascade NFT NOT yet run.

## Paper goal
Preliminary COMPLETE paper for CA-OPD: title + story + rigorous theory (multi-agent cross-checked) + full experiment table +
convergence figure (score/reward vs training steps, continuous lines, ours vs baselines) + qualitative comparison figure (cherry-picked) + ablation.
Reuse DiffusionOPD's base theory/framing where appropriate (cite). Paper dir: `/home/colligo/opd_related/CA-OPD-paper/`.
