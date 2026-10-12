# CA-OPD — Independent Theory Review

Skeptical, independent derivation and stress-test of the CA-OPD theoretical claims
(conflict-averse / Pareto multi-teacher on-policy distillation). Prepared by an
independent reviewer; I re-derived each claim from scratch rather than checking a draft.
Base derivation reused: `papers/arXiv-2605.15055v1/sec/method.tex`. Numerical spot-checks
were run to confirm the algebra.

**One-line verdict:** the *method* (adaptive softmax min-max reweighting of commensurable
per-teacher velocity losses) is sound and well-motivated, and one piece of its
justification is even cleaner than claimed (envelope theorem). But the headline
*motivation* — "uniform mean is Pareto-dominated because linear scalarization only
reaches the convex hull" — is **incorrect as stated under the paper's own isotropic-metric
assumptions**: the velocity-matching loss front is convex and the mean is Pareto-optimal
there. The real justification must move to reward-space proxy misalignment and min-max
frontier *selection*, not "non-convex loss fronts." Several novelty claims are
overclaimed. Details and minimal fixes below.

---

## 1. Uniform-sum minimizer = per-step teacher mean? (Derivation + caveats)

**Setup.** In the ODE regime (eq. `opd-ode`), the per-teacher per-step loss is
`½‖μ_S − μ_{T,m}‖²`. From eqs. `student-mean`/`teacher-mean`, the means differ only through
the velocity, `μ_S − μ_{T,m} = k_j · (v_θ − v_{φ_m})` with a shared scalar
`k_j = (1 + σ_{t_j}²(1−t_j)/(2t_j)) Δt_j`. So at a fixed state `x_{t_j}` and step `j`,
`L_m = w_j ‖v_θ − v_{φ_m}‖²` with `w_j = k_j²/2` (or `k_j²/(2σ_j²)` for the KL form).
**Crucially `w_j` is identical across teachers `m`** — it depends only on the scheduler.

**Derivation.** Minimize the uniform sum over teachers, over the free velocity value
`v` at that point:
`F(v) = Σ_m w_j ‖v − v_{φ_m}‖²`, `∇_v F = 2 w_j Σ_m (v − v_{φ_m}) = 0`
`⟹ v* = (1/M) Σ_m v_{φ_m}`. Hessian `2 M w_j I ≻ 0` ⟹ unique global min. The shared `w_j`
cancels, so it is the **unweighted arithmetic mean** of teacher velocities. Confirmed
numerically (argmin ≡ mean, weight-invariant).

**So the pointwise claim is correct — but it is only pointwise.** Precise caveats the
paper must state:

- **(a) Function space, not a vector.** `θ` (a rank-32 LoRA) parameterizes a *single*
  field `v_θ(x,t)` shared across all steps/prompts. The "mean velocity" identity holds
  *at each visited point independently* only under an **infinite-capacity / realizability**
  assumption. With finite LoRA capacity the stationary point is not the mean field but its
  **weighted least-squares projection** onto the model tangent space, with weights =
  (visitation density × `w_j`). "Student → mean of teacher velocities" is therefore an
  *idealization*, not an equality.
- **(b) On-policy fixed point.** The state law `x_{t_j} ∼ p_{S,θ}` depends on `θ`, so the
  stationary condition is a self-consistent fixed point, and the mean is enforced **only on
  the student's own visited support** — off-support behavior is unconstrained.
- **(c) Mean velocity ≠ mean of objectives.** Integrating an averaged velocity field along
  an ODE does **not** produce the average of the teachers' images/distributions (the
  trajectory itself changes; integration is nonlinear in the field), and each reward is a
  nonlinear functional of the *whole* trajectory. So "student learns the mean velocity"
  does **not** imply "student sits at the mean of the teachers' rewards" or at any fixed
  convex combination on the *reward* Pareto front. This caveat directly undermines the
  Section-2 leap from "mean velocity" to "convex-hull point of the reward front" (see §2).

**Rating: PROVEN (pointwise, idealized) / DEFENSIBLE (as a global claim).** Correct as a
per-step statement; needs the realizability + on-policy + velocity≠reward caveats.

---

## 2. "Linear scalarization only reaches the convex hull and is Pareto-dominated on
non-convex fronts" — is it applicable here?

**The generic theorem is correct.** For `min_θ Σ_m λ_m f_m`, `λ_m ≥ 0`: every minimizer is
(weakly) Pareto-optimal, and as `λ` sweeps the simplex you trace exactly the boundary of
the *convex hull* of the attainable objective set. Points on non-convex ("sunken") regions
of the Pareto front are unreachable by any `λ`. Textbook (Miettinen; Boyd–Vandenberghe).

**But its application to uniform-mopd is wrong under the paper's own assumptions.** Two
independent problems:

1. **The scalarized objectives are the velocity-matching losses `L_m`, not the rewards.**
   Uniform-mopd minimizes `Σ_m L_m`, i.e. it scalarizes the *distillation losses*. Under the
   shared isotropic metric these are **convex quadratics** `L_m(v) = w_j‖v − v_{φ_m}‖²` with a
   *common* metric. The attainable set of `(L_1,…,L_M)` is then convex and its Pareto set is
   exactly the **convex hull `conv{v_{φ_m}}`** — there is **no non-convex region**, and the
   uniform mean `(1/M)Σ v_{φ_m}` lies **inside the hull, hence is Pareto-optimal, not
   dominated**. I verified numerically (2-teacher isotropic case): the mean is *not*
   dominated. This is a genuine self-contradiction in the paper — the very
   shared-covariance / isotropic-metric structure invoked to make the problem "clean" is
   precisely what makes the loss front convex and the mean Pareto-optimal. A global
   scalarization minimizer with strictly positive weights is **always** Pareto-optimal (even
   with a non-convex feasible set): if `θ*` minimizes `Σλ_m L_m` and were dominated by `θ'`,
   then `Σλ_m L_m(θ') < Σλ_m L_m(θ*)`, contradiction. So "uniform mean is Pareto-dominated
   *in loss space*" cannot be rescued.

2. **Uniform-mopd is not a scalarization of the reward objectives.** Because of §1(c),
   there is no fixed `λ` for which mean-velocity equals the `λ`-scalarized optimum of the
   *rewards* (GenEval/OCR/Aes). So the clean statement "uniform-mopd = linear scalarization
   of the task objectives, hence stuck on the convex hull of the *reward* front" is
   **not established** — it silently swaps loss-space for reward-space.

**Where domination can genuinely occur (the honest mechanism).** The mean is dominated only
if you *leave* the isotropic loss space:
- **(A) Reward-proxy misalignment.** Rewards are nonlinear, saturating, possibly
  reward-hacked functionals of the trajectory. Minimizing `Σ L_m` is a proxy; the
  loss-optimal point can be **reward-dominated**. Concrete numbers (verified):

  | point | (L₁,L₂) | (R₁,R₂) |
  |---|---|---|
  | A | (0,4) | (10,2) |
  | B | (1,1) | (5,5) |
  | C | (4,0) | (2,10) |
  | **D** | (2,2) | (**8,8**) |

  Uniform-mopd minimizes `L₁+L₂` → selects **B** (sum 2) → rewards (5,5). But **D** (loss
  sum 4, so uniform-mopd never picks it) gives rewards (8,8) and **dominates B in reward
  space**. Min-max / adaptive reweighting can move off the loss-utilitarian point toward D.
  This is the only defensible sense in which the mean is "dominated," and it hinges on
  loss↛reward misalignment — *not* on a non-convex loss front.
- **(B) Finite capacity + on-policy state shift** make the attainable `(L_1,…,L_M)` set
  non-convex in `θ`, and can stall the averaged-gradient step (the classic CAGrad/MGDA
  *dynamics* motivation: under conflict the mean gradient decreases both losses less than a
  conflict-averse step). This is a training-dynamics argument, not a fixed-optimum argument.

**Note on the monotone 2-teacher case.** If the two teachers pull in opposite directions
monotonically, *every* interpolation point (including the mean) is Pareto-optimal; the mean
is simply the *utilitarian* point that can badly under-serve the harder-to-match teacher.
That is a "wrong point on the front" complaint (fixed by min-max *selection*), **not** a
domination complaint. The paper conflates the two.

**Rating: OVERCLAIMED.** The theorem is real but misapplied. **Minimal fix:** drop
"Pareto-dominated / non-convex loss front." Replace with: "(i) uniform-mopd is the
*utilitarian* (loss-sum) point, which over-serves easy-to-match teachers and under-serves
conflicting ones; (ii) because velocity-matching loss is a nonlinear proxy for reward, the
loss-utilitarian point can be *reward-dominated*; min-max **selects a more balanced
frontier point**." Keep the convex-hull theorem only as motivation for *why a single fixed
weighting is limiting*, and be explicit that the front you cannot fully reach is the
**reward** front, reached through the nonlinear loss→reward map.

---

## 3. Weighted-Tchebycheff theorem + softmax(EMA/τ) as min-max

**Chebyshev statement is correct.** Weighted Chebyshev `min_θ max_m λ_m (f_m − z*_m)` with
utopia point `z*` and `λ>0`: every solution is weakly Pareto-optimal (Pareto-optimal if
unique / with the augmented variant), and **for every Pareto point there exist weights that
recover it — including non-convex/unsupported points**. For OPD the utopia point is
naturally `z* = 0` (perfect distillation, `L_m ≥ 0`), so a plain weighted-max of losses is a
valid Chebyshev scalarization. This is a legitimate and, for once, clean use.

**softmax(EMA/τ) ≈ min-max: yes, and more precisely it is entropy-regularized min-max.**
The inner game `max_{w∈Δ} Σ_m w_m L_m − τ·Σ_m w_m ln w_m` has closed-form maximizer
`w_m ∝ exp(L_m/τ)` (exactly CA-OPD's softmax), with optimal value `τ·logsumexp(L_m/τ)` — a
smooth upper bound on `max_m L_m` (gap `≤ τ ln M`). So CA-OPD minimizes a **soft-max (LSE)
of the teacher losses**, i.e. a temperature-smoothed Chebyshev scalarization. Verified:
- **τ→0:** `w →` one-hot on `argmax_m L_m` (worst teacher); objective → `max_m L_m` (hard
  min-max). ✔
- **τ→∞:** `w →` uniform; objective → `(1/M)Σ L_m` (recovers mopd). ✔ (matches CONTEXT).

**Bonus rigor the paper should claim (strengthens it):** CA-OPD uses **detached** EMA
weights, so it does *not* backprop through the softmax. By the **envelope theorem**,
`d/dθ [τ·LSE(L/τ)] = Σ_m w_m(θ) ∇_θ L_m` with `w` evaluated at the maximizer and **no extra
term** — so detached instantaneous weights give the *exact* gradient of the smooth Chebyshev
surrogate. The only approximation is the EMA lag (weights from an EMA of past KL rather than
current KL), a stabilizing smoothing, not a bias in the objective. This is a real, correct,
and citable justification that the draft appears to under-sell.

**Direction check.** Min-max reduces the *worst* (largest) loss ⟹ up-weight the
**higher-KL** teacher. Correct direction. ✔

**Risk — chasing an irreducible task: REAL and under-addressed.** CA-OPD weights by the
**absolute** EMA-KL, which conflates "currently mismatched but reducible" with
"intrinsically hard / unrealizable / reward-hacked / noisy teacher." A teacher with a high
**irreducible floor** (student cannot represent its field, or the teacher is a bad target)
receives permanent up-weighting, dragging capacity away from reducible tasks — the textbook
failure mode of min-max/Chebyshev. Mitigants present: finite `τ=0.5` (soft, not hard
min-max) and EMA smoothing cap the pathology but do not remove it. **Recommended fix (state
as limitation, or implement):** weight by **regret / reducible gap**
`L_m − min-running(L_m)` (or `L_m − z*_m` with a learned floor) rather than raw absolute KL,
so an irreducible floor does not attract weight. Empirically CA-OPD's win was on aesthetics,
which *was* reducible (it reached the aesthetics teacher's own level), so the up-weighted
task happened to be reducible — consistent, but this should be argued, not assumed.

**Rating: DEFENSIBLE→PROVEN.** Chebyshev statement PROVEN; softmax↔min-max PROVEN (with the
LSE / envelope-theorem framing, which is a strengthening). The irreducible-task risk is a
genuine limitation to disclose.

---

## 4. Novelty claim: shared Jacobian + isotropic metric make conflict "well-posed,"
justifying CAGrad/MGDA better than generic MTL

**Partially true, partially overclaimed. Break it into the pieces that survive scrutiny.**

Per-task gradient: `g_m = Σ_j (w_j/…) J_θ(x_{t_j})ᵀ r_m^{(j)}`, `r_m = v_θ − v_{φ_m}`,
`J_θ = ∇_θ v_θ`. At a shared state, `J_θ` is the *same* linear map for all `m`; the tasks
differ only through the residual `r_m`.

**What genuinely holds (real merit):**
- **Commensurable losses.** All `M` objectives are the *same functional* (velocity `L2` /
  closed-form KL in nats), differing only in target. So losses and gradient magnitudes are
  on the **same units/scale** — exactly the condition MGDA/CAGrad assume and that generic
  MTL (cross-entropy vs MSE vs …) violates. This is the strongest and most defensible part
  of the novelty argument.
- **Shared input state.** On-policy shared rollout means *the same* `J_θ(x_{t_j})` applies to
  all teachers at each point (in generic MTL each task has different inputs → different
  Jacobians). So conflict between `g_m, g_{m'}` is inherited from conflict between residuals
  `r_m, r_{m'}` in a common velocity space through a common `J_θ`. This is a real structural
  specialization.

**What is overclaimed / imprecise:**
- **"Isotropic metric ⇒ meaningful gradient angles" conflates output space with parameter
  space.** The isotropy is in the **velocity/output** space (the KL is honestly `‖·‖²/2σ²`).
  But MGDA/CAGrad operate on **parameter** gradients, whose geometry is
  `⟨g_m,g_{m'}⟩ = r_mᵀ (J_θ J_θᵀ) r_{m'}` — warped by the empirical NTK/Gram `J_θ J_θᵀ`,
  which under a **rank-32 LoRA is low-rank and highly anisotropic**. Output-space isotropy
  does **not** transfer to parameter-space isotropy, so "conflict angles are geometrically
  meaningful because the metric is isotropic" is loose. The *commensurability* argument
  survives; the *isotropy⇒angles* argument does not, as stated.
- **Shared backbone Jacobian is not unique to OPD.** Any shared-trunk MTL model shares a
  backbone Jacobian. The genuinely distinctive ingredients are **shared input state +
  identical loss functional (commensurability)**, not "shared Jacobian" per se. Framing
  novelty around the Jacobian oversells.
- **Method/justification mismatch.** The justification invokes **CAGrad/MGDA gradient
  surgery**, but CA-OPD as implemented is **softmax-Chebyshev reweighting** — it rescales
  gradient *magnitudes* and does not project out conflicting components. Reweighting does not
  *need* the shared-Jacobian well-posedness argument (it needs loss commensurability, which
  holds); and it does not deliver the escape-from-convex-hull property that CAGrad's
  *projected* update can. So the CAGrad/MGDA well-posedness discussion is largely orthogonal
  to the shipped algorithm. Either implement a gradient-surgery variant (and then the
  argument applies) or drop the CAGrad/MGDA framing and justify the reweighting via
  commensurability + Chebyshev (§3), which is cleaner anyway.

**Rating: OVERCLAIMED.** Real kernel of truth (commensurable, same-metric, same-state
objectives — a genuinely cleaner MOO setting than generic MTL), wrapped in two overreaches
(output↛parameter isotropy; CAGrad/MGDA framing for a scalarization method). **Minimal
fix:** claim only "identical loss functional + shared on-policy state ⟹ per-teacher losses
are commensurable and live in a common isotropic *output* metric, so adaptive Chebyshev
scalarization is well-posed and its weights are comparable across teachers — unlike generic
MTL with incommensurable losses." Do not claim parameter-space isotropy; drop or actually
implement CAGrad/MGDA.

---

## 5. Consistency of theory with data

Observed @541 (n=256): Aes 6.120→6.172 (+0.052, reaches mopd's own best-@721 ~180 steps
early), GenEval 0.947→0.955 (+0.008), OCR 0.939→0.927 (−0.012), Pick +0.03, HPS +0.001,
Clip flat; net anchored score +0.010.

**What matches the theory:**
- **Direction.** Aesthetics was the lagging axis (largest normalized gap ⟹ highest EMA-KL
  teacher). Min-max up-weights the worst-matched teacher ⟹ aesthetics improves. ✔ predicted.
- **Frontier reallocation vs. uniform improvement.** OCR was already essentially at its
  teacher's ceiling (0.939 vs teacher 0.94, i.e. over-served). A small OCR drop while
  aesthetics rises is the **trade signature** of moving *along* a frontier (reallocating
  capacity from the saturated axis to the lagging one) — i.e. min-max **relocates the
  operating point**, it does not Pareto-dominate the old one. ✔ consistent with theory. A
  pure "everything up" result would actually have been *evidence against* the frontier story.

**What weakens the claim (be skeptical):**
- **GenEval also rose (+0.008)** — a third axis improving is not a pure trade; it muddies
  the clean "gain aesthetics, pay OCR" narrative (could be under-converged mopd baseline,
  i.e. partly a speed effect, not a frontier effect).
- **OCR drop (−0.012) is within plausible eval noise** at n=256 (rule-based OCR on 256
  samples). It is suggestive, not conclusive, evidence of a trade. Needs seeds / larger n /
  error bars.
- **Circularity in the headline "+0.010."** The aggregate score is itself a min-max-anchored
  scalarization — i.e. essentially the objective CA-OPD tilts toward. Reporting improvement
  in the optimized aggregate slightly over-credits the method. Lead with **per-axis** deltas.
- **Speed vs. frontier conflation.** "Reaches mopd's best aesthetic ~180 steps earlier" is a
  *convergence-speed* claim, distinct from a *better frontier point*. The definitive @721
  comparison (job 97f2db24 per CONTEXT) is still pending; final frontier claims should wait
  for it.

**Rating: DEFENSIBLE (consistent with, not proof of).** The qualitative pattern
(lagging axis up, saturated axis slightly down, net up) is exactly the min-max/Chebyshev
frontier-reallocation signature. But single-checkpoint, single-seed, n=256, a confounding
GenEval rise, an OCR drop within noise, and circular aggregate scoring mean it **supports**
rather than **confirms** the theory. **Fix:** report per-axis with error bars over
seeds/checkpoints; explicitly separate the *speed* claim from the *frontier-point* claim;
finalize with the @721 eval; frame as "consistent with min-max reallocation."

---

## 6. Summary of incorrect / overclaimed statements + minimal fixes + ratings

| # | Claim | Rating | One-line justification / fix |
|---|---|---|---|
| 1 | Uniform-sum minimizer = per-step teacher-mean velocity | **PROVEN** (pointwise) / **DEFENSIBLE** (global) | ∇=0 gives the mean; shared step-weight cancels. Add caveats: finite-capacity ⇒ weighted projection not exact mean; on-policy support only; mean velocity ≠ mean reward. |
| 2a | "Uniform mean is Pareto-dominated" | **OVERCLAIMED (incorrect as stated)** | Under the isotropic metric the loss front is **convex**; the mean is **Pareto-optimal**, and a positive-weight scalarization minimizer is *never* dominated. Verified numerically. |
| 2b | "Linear scalarization reaches only the convex hull; non-convex regions unreachable" (applied to OPD) | **OVERCLAIMED (misapplied)** | Theorem is true but the *loss* front is convex, and uniform-mopd is not a fixed scalarization of the *reward* objectives. Restrict the argument to the **reward** front reached via the nonlinear loss→reward map; call the mean the *utilitarian* point, not a dominated one. |
| 3a | Weighted-Tchebycheff attains the full (incl. non-convex) Pareto set | **PROVEN** | Standard Miettinen result; utopia `z*=0` natural for OPD losses. |
| 3b | softmax(EMA/τ) approximates min-max; τ→0 worst-task, τ→∞ uniform | **PROVEN** | It is entropy-regularized min-max: `w_m∝exp(L_m/τ)`, objective `τ·LSE(L/τ)`; limits verified. Strengthen with envelope theorem (detached weights = exact surrogate gradient). |
| 3c | Up-weight higher-KL teacher = correct min-max direction | **PROVEN** | Min-max reduces the largest loss. But weighting by **absolute** KL risks chasing an **irreducible** task — disclose; prefer regret/`L_m−floor`. |
| 4a | Losses are commensurable (same functional, same isotropic output metric, shared on-policy state) ⇒ cleaner MOO than generic MTL | **DEFENSIBLE** | Genuine and distinctive; this is the real novelty kernel. |
| 4b | "Shared Jacobian + isotropic metric make conflict angles geometrically meaningful, justifying CAGrad/MGDA" | **OVERCLAIMED** | Isotropy is in *output* space; parameter-gradient geometry is `J J^T` (low-rank/anisotropic under LoRA). Shared backbone Jacobian isn't unique to OPD. And the shipped method is scalarization, not CAGrad/MGDA. Drop CAGrad/MGDA framing or implement it; justify via commensurability + Chebyshev. |
| 5 | Aesthetics up / OCR down / net +0.010 = predicted min-max Pareto behavior | **DEFENSIBLE** | Direction + trade signature match frontier reallocation, but single seed, n=256, OCR drop within noise, confounding GenEval rise, circular aggregate. "Consistent with," not "confirms"; finalize @721. |
| 6 | (rigor signal) equal-covariance ⇒ fwd-KL = rev-KL = same L2, so "symmetrizing KL" is vacuous in OPD | **PROVEN** | Correct: for equal-covariance Gaussians both KL directions equal `‖Δμ‖²/2σ²`; symmetrization adds nothing. Good to keep as a rigor signal. |

### The three things to fix before committing to the paper
1. **Remove the "Pareto-dominated / non-convex loss front" motivation.** It is false under
   the paper's own isotropic-metric premise (convex front ⇒ mean is Pareto-optimal). Recast
   as: uniform-mopd is the *utilitarian* point; min-max **selects** a more balanced frontier
   point; the front one cannot fully reach with a single weighting is the **reward** front,
   accessed through the nonlinear loss→reward proxy.
2. **Re-anchor the "well-posed conflict" novelty on commensurability**, not on shared
   Jacobian / output-space isotropy transferred to gradient space; and reconcile the
   CAGrad/MGDA framing with the fact that the method is softmax-Chebyshev scalarization
   (either implement surgery or drop that framing).
3. **Add the two disclosures**: (i) min-max can chase an irreducible/reward-hacked teacher
   (mitigated by τ=0.5, better fixed by regret-based weights); (ii) results are
   single-seed/single-step/n=256 and "consistent with," pending the @721 eval and per-axis
   error bars.

### What is genuinely solid (keep and lead with)
- Pointwise teacher-mean identity (with caveats).
- CA-OPD = entropy-regularized (soft) min-max / Chebyshev with clean τ→0 / τ→∞ limits, and
  the envelope-theorem argument that detached EMA weights give the exact surrogate gradient.
- Loss commensurability as the real reason multi-objective machinery is better-posed in OPD
  than in generic MTL.
- The empirical frontier-reallocation signature (lagging axis up, saturated axis slightly
  down) as *consistent* evidence.
- The equal-covariance ⇒ symmetrized-KL-is-vacuous observation as a rigor signal.
