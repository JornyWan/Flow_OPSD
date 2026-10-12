# Phase61: common-start native OPD engineering acceptance

Frozen before submission; one L40, at most 1800 allocated GPU seconds, no retry.
This implements the daytime priority from the Phase60 handoff, not another CFG
learning pool. No new timer and no use of expired overnight reservations.

Question: can pinned native SOPD-GenEval continue the same published Student,
produce changed raw weights, save all checkpoint payloads, and reload an actual
single-adapter full trajectory? Measure runtime before freezing a formal budget.

Source: official OPD commit 0c85bf0193a6da24bb1f72152b612d353a154999,
fresh isolated snapshot, existing audited managed lifecycle hooks only. Same
prepared 512 training / 128 development text view (development only named test.txt).
Fresh native Adam; published Student, GenEval teacher CFG1, student CFG4.5, 512px,
Flow10, native text length128, lr3e-4. Two epochs, batch1, three trajectories per
epoch, nine supervised timesteps per trajectory, accumulation3*9: expected two
optimizer updates. This is a small hardware recipe, not a formal OPD reproduction
or a matched comparison against the older 64-state method. No candidate selection.

Checkpoint fixed at final update2, raw and EMA both checked. Native EMA interval8
is not reached; unchanged EMA is expected, unchanged raw is an engineering failure.
Require: two progress events at updates1/2; finite, shape/key-matched adapters;
changed raw; EMA identical to initial; Adam steps2 and scaler/EMA/RNG saved.
Deployment: first TRAIN prompt, seed61061, same Flow10/CFG4.5/text128; four latent
trajectories (initial/raw/EMA/raw replay), no decoder output or evaluation images.
Remove the behavior adapter: only default physically present, all ten steps use
student, no teacher/anchor fallback. Require exact exported raw state reload,
exact raw replay and initial/EMA replay, finite changed raw endpoint. Record load
and trajectory seconds separately; these are engineering timings on one GPU,
not a formal latency benchmark and not all-in training efficiency.

Primary endpoint is pass/fail of the checks above. Joint and all five quality
scores are explicitly NOT MEASURED; no efficacy claim or favorable-checkpoint
selection. Confirmation/final remain sealed; no human or agent image reads.
Initial adapter/teacher/data/source hashes are pinned. Backbone and other large
assets retain native inventory validation; full content fingerprints remain a
requirement for the later formal efficacy comparison.

Failure: stop, preserve evidence, no GPU retry. CPU analysis repairs live in a
separate directory. Success: use measured throughput to propose a common-data,
common-start normal-OPD vs verified vs count-matched-random experiment, with
all128 existing dev prompts, joint and all five quality metrics, total collection+
verification+training+evaluation cost. Do not submit that larger study from this
pilot. Full-trajectory learning for our method still needs implementation and
acceptance; a native deployment check does not establish our method's deployment.

Scheduling: immediately probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu; L40
only, no preemption recovery, conservative sablab GPU cap6 includes pending and
other projects. If estimated start exceeds one hour, retain prepared state.
