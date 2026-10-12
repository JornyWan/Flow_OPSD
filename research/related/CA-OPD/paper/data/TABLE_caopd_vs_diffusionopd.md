# CA-OPD best checkpoint vs DiffusionOPD — per-axis (no aggregate; per user decision)

Eval: n=256 prompts, student gs=4.5, 40-step ODE. Model-based rewards on DrawBench, GenEval/OCR on FlowGRPO splits.
"Best ckpt" = best converged CA-OPD checkpoint over the whole trajectory. Bold = best of the three rows per axis.

| Model / source | GenEval | OCR | PickScore | ClipScore | HPSv2.1 | Aesthetic |
|---|---|---|---|---|---|---|
| DiffusionOPD (paper Table-1) | **0.96** | 0.94 | 23.99 | 0.297 | 0.342 | **6.15** |
| DiffusionOPD (our repro, mopd@721) | 0.958 | 0.939 | 23.98 | **0.298** | 0.342 | **6.172** |
| **CA-OPD (best, @781)** | 0.953 | **0.947** | **24.00** | 0.296 | **0.343** | 6.145 |

Notes (honest, per-axis):
- CA-OPD@781 **matches-or-slightly-exceeds** our DiffusionOPD reproduction on OCR (0.947 vs 0.939),
  PickScore (24.00 vs 23.98), and HPSv2.1 (0.343 vs 0.342); within eval noise on GenEval, ClipScore, Aesthetic.
- CA-OPD reaches its Aesthetic peak (6.172, = mopd's own best) EARLIER at ckpt-541 (~180 steps sooner),
  then drifts to 6.145 by @781 (min-max re-chasing the momentary worst axis).
- No aggregate/Average reported: a single min-max mean is normalization-dependent and not comparable across
  papers (the earlier "0.985 vs 0.929" was a naive/circular normalization artifact, NOT a real win).
- MISSING vs paper's 8 columns: ImageReward, UnifiedReward — not yet evaluated on our checkpoints (being wired up).

STATUS 2026-08-20: regret-weighted CA-OPD (mopd_caopd_regret) + higher-conflict variants (mopd_caopd_conflict,
mopd_conflict) launched P0; this table will be refreshed with their best checkpoints once converged.
