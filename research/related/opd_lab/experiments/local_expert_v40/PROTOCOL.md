# Phase40: whole-cohort standard metric characterization

Freeze ALL 600 Phase33 images (60 prompts × 2 seeds × 5 learned checkpoints),
and ALL 600 Phase34 images (60 prompts × 2 seeds × 5 intervention modes).
No re-generation, training, threshold tuning, outcome-selected subset, checkpoint choice,
agent image viewing, remote vision API, or alteration of old archives.

Use pinned upstream OPD PickScore /26, CLIPScore, Aesthetic, HPSv2.1, and
ImageReward-v1.0 in FP32 with native preprocessing. Also report PickScore ×26
as the historical OPSD raw scale. These are separate correlated proxies, never
sum incompatible scales or describe them as independent human preferences.
ImageReward uses its library paired scalar API, verified against the upstream
cross-prompt matrix diagonal on two different prompts before scoring. Preserve
the native 35-token ImageReward /77-token CLIP/PickScore limits and audit target
word retention; do not rewrite prompts for a better outcome. Other scorer math unchanged.

Reuse existing full-image OCR and native-view GenEval strict cached outputs only
where available, distinctly labelled as our joint-prompt diagnostics. Phase33
has cached OCR but no complete automatic object evaluation; report unavailable,
not inferred from historical assistant observations. The custom two-view consensus,
OCR whitespace/exact-match logic, and visibility proxy are not standard benchmark
metrics. GenEval on these custom prompts is not the original GenEval benchmark score.

Report all methods and families. Phase33 primary descriptive contrasts: verified+
retention minus S1, and verified+retention minus random+retention. Secondary:
verified+retention minus verified, random+retention minus random. Phase34 contrasts:
early half/base, early full/base, late half/early full, both full/early full,
both full/base. Keep paired noise seeds; bootstrap prompt clusters within family
20,000 times, seed40040. Report 95% descriptive intervals, no multiplicity-adjusted
success claim or independence assumption across metrics. Intervals exclude
training-repeat uncertainty. Full 1200 images required for completion; errors
are recorded, never dropped or imputed. All are reused development data.

S1 is a previously trained common starting model, not raw backbone or native OPD.
Actions34 teacher suffix interventions are not trained students. Historical
DrawBench40-step scores cannot be directly compared to these custom Flow10 images.
The original full native benchmark comparison requires new shared prompts/seeds/
sampling/resolution and matched baseline checkpoints; this phase does not claim it.

Resource cap: one L40, 30 minutes, no automatic requeue. Save all 6000 scalar
predictions with image/prompt/model/code fingerprints and per-metric time.
Chat observations are distinct, unblinded, case-level qualitative human evidence;
do not infer mode-specific or binary labels. Preserve old Phase33–39 records.
