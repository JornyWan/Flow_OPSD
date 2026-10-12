# Phase79: full two-step learning versus qualified native OPD

Frozen before GPU. Phase78 COMPLETED0:0 in510L40S seconds:8 true Adam updates,
GPU one-step/native and full/recomputed two-step gradients exact, EMA/export exact,
37.079GiB peak,44.085seconds per epoch. This qualifies implementation, not efficacy.

Primary question: does the two-step learned student improve complete deployed
joint success over Phase77 native OPD(finalEMA49/128)? Same pure SD3.5M fresh
LoRA seed42,512 train prompts,batch3,256 updates,actualLR3e-4,original Adam/clip/
AMP/native repeated EMA. No continuation from Phase77/78. Same first256 actual
prompt schedule and Adam/LR log required, verified before evaluation. Online
latents differ once weights diverge; no claim of paired training states.

Use exactly the Phase78 qualified code: from each detached behavior state j0..8,
teacher and student each take two own-state Flow transitions; target detached,
student internal state differentiable; endpoint delta normalized by
 dt_first/(dt_first+dt_second), preserving one-step time weight for constant
velocity residuals. First forward checkpointed/recomputed, no native layer-level
checkpointing. Original loss reduction reused. No verifier,hard filter,extra
teacher,LR/CFG sweep or new time sampling. This tests a learning component of the
proposed method, not yet terminal verification or a proven novel contribution.

Matching limitations explicitly retained:19 teacher queries/trajectory vs10 native;
162 logical student forward examples+81 recomputations/update vs81 native;
second transition includes j9 untrained as a native single-step anchor. Equal
256 updates/2304 trajectory draws/20736 anchors does NOT equal compute/information.
Report measured GPU cost and require future same-information graph-detach and
compute-matched native controls before causal/efficiency claims.

Final256 EMA is primary, RAW secondary, save final only. Complete new S0/EMA/RAW
128 existing exploratory dev each, identical fixed seeds,Flow40/CFG4.5/text128/
512px/FP16 with explicitFP32CFG, one physical student adapter. Regenerate S0 to
verify exact128 image/task identity and quality<=1e-4 versus Phase77. Do not
reinterpret regenerated S0 as new independent evidence. Same frozen evaluator:
all joint/OCR/structure/CER/A/Bunknown and five quality metrics, paired20k
family-stratified prompt intervals.2 historical task references exact,8 quality
reference images/40 values tolerance1e-4.384 new endpoints/1920 main quality.

Authoritative primary comparison in comparison/result.json is blockEMA-nativeEMA;
evaluation/result.json contains within-run S0 comparisons only as diagnostics.
Also report RAW-nativeRAW and newS0/newEMA/newRAW. Never choose RAW/bestcheckpoint,
drop unfavorable family/metric, or evaluate confirmation512/final768. One training
seed and repeated128dev => exploratory intervals, not independent confirmation.

Prospective decision: primaryjoint interval wholly above0 => positive exploratory
signal; crossing0 => unresolved, wholly below0 => adverse under this fixed budget.
Always disclose effect size/gains/losses/quality/CER and costs; no claim of Pareto
improvement if any quality drops. Per user's2026-10-11 steering, a modest CLIP
mean decline alone is not an automatic stop. Do not choose a numeric tolerance
posthoc or change Phase77's historical all-means rule. If joint improves but
quality trades off, report that tradeoff and verify independently. If no learning
advantage, analyze target fitting versus whole deployment before adding mechanisms.
Next phase requires separate evidence/protocol; no automatic follow-on submission.

One new L40S job <=16200seconds(4h30), no requeue/retry/timer. Measured training
projection11286sec+prior evaluation3194sec leaves1720sec startup/safety, not a
completion guarantee. Stop on technical failure,OOM,AMP skip,source/cohort/
weight/reference parity error or preemption. Preserve incomplete evidence, no
automatic replay/GPU repair. CPU report fixes in independent records. Freeze
sources,assets,prior reference artifacts; backbone metadata inventory limitation
retained. Probe minilab/sablab/scu/preempt, sablab cross-project allocated+pending<=6,
compatible expectedstart<=24h, ordinary preferred for similar starts; do not alter
other jobs. Agent image reads0, no new agents/timers.
