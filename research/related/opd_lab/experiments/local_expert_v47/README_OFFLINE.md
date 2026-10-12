# Phase47 offline report

Open report.html without a server: no images, network requests or external assets.
One independently refreshed online round, restoring each arm's Phase46 model,
Adam and scaler; 48 new training states per arm. Seven before/after models on
32 previously ungenerated prompts. The primary is verified AFTER minus its own
BEFORE, not an inherited cross-arm advantage. This reduced pilot does not prove
multi-seed stability, native OPD superiority, refresh-specific causality or
sampler robustness. Human ratings are auxiliary and were not used in this run.

CPU preflight found only 27 remaining family/template strata. Its original
preparation source and revision record are preserved in preflight/. Eight fresh
prompts per family were selected with maximal available template coverage before
any GPU execution; no reserved human prompts were used.

512 new images: 288 for collection tasks and 224 for full deployment evaluation.
All 224 deployment endpoints receive all five quality metrics. Resource limits
and original 1256/Phase46 448 image allowances remain separately recorded.
Large checkpoints, images and trajectories stay on the server, bound by hashes
in result.json, trajectory_hashes.json and the attestation. No agent image reads.

The original job timed out after completing training, generation and task scores.
The quality-only continuation is an explicit protocol exception, prospectively
recorded before effect analysis in QUALITY_CONTINUATION.md. It preserves all
690 cached scores and completes 430 missing cells without training or generation.
Both jobs, the 10-second scheduler overrun, and the denied (unapplied) time-limit
request are retained. The total remains charged to the original shared GPU pool.
