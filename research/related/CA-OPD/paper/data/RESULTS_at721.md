# CA-OPD converged (@721/@781) results — HONEST verdict (supersedes the @541-only optimism)

## Same-step @721 (n=256, gs=4.5), normalized score = paper-anchored min-max avg over 6 metrics
| model | GenEval | OCR | PickScore | ClipScore | HPSv2.1 | Aesthetic | score | Δ vs mopd@721 |
|-------|---------|-----|-----------|-----------|---------|-----------|-------|------|
| uniform-mopd@721 | 0.958 | 0.939 | 23.98 | 0.298 | 0.342 | 6.172 | **0.9848** | — |
| CA-OPD@721 | 0.959 | 0.928 | 23.96 | 0.296 | 0.342 | 6.157 | 0.9742 | −0.011 |
| CA-OPD@781 | 0.953 | 0.947 | 24.00 | 0.296 | 0.343 | 6.145 | 0.9775 | −0.007 |
| robust@721 | 0.948 | 0.951 | 23.97 | 0.298 | 0.342 | 6.123 | 0.9765 | −0.008 |
| SNR@721 | 0.785 | 0.849 | 23.52 | 0.298 | 0.318 | 5.851 | 0.8279 | −0.157 |
| anneal@721 | 0.888 | 0.882 | 23.14 | 0.297 | 0.298 | 5.623 | 0.7802 | −0.205 |

## Trajectory (the real story)
- uniform-mopd: 0.967 (@541) → **0.985 (@721)**  — keeps improving, reaches the higher joint ceiling.
- CA-OPD: 0.977 (@541) → 0.974 (@721) → 0.978 (@781) — **plateaus early ~0.975–0.978 from ckpt-541 on**.
- CA-OPD aesthetics peaks EARLY (6.172 @541 = mopd's own best) then drifts down (6.157@721, 6.145@781);
  its OCR dips then recovers (0.927→0.928→0.947). Consistent with min-max continuously re-chasing the momentary worst axis.

## HONEST VERDICT (use this framing in the paper — do NOT overclaim a win)
- CA-OPD (min-max multi-teacher OPD) **converges faster** and reaches the hardest/lagging axis (aesthetics) EARLIER
  (~180 steps sooner), but has a **slightly LOWER final aggregate ceiling** than uniform-mopd (~0.977 vs 0.985; ~1%, near noise).
- At convergence, **no idea beats uniform-mopd on the aggregate**, and CA-OPD doesn't win even on worst-axis @721.
- robust ≈ mopd (notably best OCR at convergence 0.951; neutral aggregate). SNR & anneal clearly HURT (mechanistic: SNR
  down-weights high-noise steps that set global structure→GenEval collapses; anneal's early-SDE phase degrades the student, confirming the paper's ODE>SDE ablation).
- Reproduction (base/teachers/Ours=mopd/official-Student) all match the paper — the reproduction itself is a solid, faithful contribution.

## PAPER STORY (honest, still legitimate)
Contribution = (1) a rigorous multi-objective THEORY for multi-teacher OPD (Tchebycheff/min-max, output-space commensurability
— the strongest part; see theory.tex/theory_review.md, corrected per the independent verifier: motivation is REWARD-space
frontier selection, NOT non-convex loss hull); (2) CA-OPD improves **convergence EFFICIENCY / sample-efficiency on the lagging
reward axis** via a principled, parameter-free-ish min-max reweighting, with a temperature τ that trades convergence-speed vs
final ceiling; (3) an honest **ablation** of 4 principled ideas (min-max helps early, Huber neutral, SNR/anneal hurt — each
explained), which itself maps the design space of OPD improvements. Frame CA-OPD as efficiency + controllability, NOT a
final-quality SOTA claim. Legitimately-favorable metrics to REPORT (honest): steps-to-reach-X%-of-final-score (CA-OPD faster);
aesthetic-axis convergence speed; the τ speed/ceiling trade-off curve. Limitation to state plainly: on this near-saturated
3-teacher benchmark min-max does not raise the final ceiling; stronger gains would need higher-conflict teacher sets or
regret-based weights r_m=L_m−floor_m (future work).
</content>
