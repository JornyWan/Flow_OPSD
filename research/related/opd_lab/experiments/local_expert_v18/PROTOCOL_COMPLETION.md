## Evaluation completion after attempt003 time limit

All three arms completed128 updates and saved final weights plus optimizer/scaler/
RNG state. The source job1050296 reached its45minute time limit during random-arm
evaluation, after158 complete endpoint records. Preserve that run, its stale lock,
all images and TIMEOUT accounting. Do not relabel the source job as completed.

New run le18_completion_001 performs zero optimizer updates and zero new teacher
queries. Verify source manifest/inputs,384 update records,135 audit records, all
three initial/final models and final AdamW steps. Copy all158 complete recorded
endpoints without selection. Preserve all35 evaluation pairs and the same final
models. Incomplete/unlogged output, if any, remains in the source as an orphan;
only complete records are eligible for reuse. No image or score inspection informs
the missing-output set or replay selection.

Before producing any missing endpoint, require exact replay of the original24
reference images, and the first already-recorded endpoint for each model/path
combination (7 outputs). Then generate the27 missing outputs, giving185 scientific
endpoints total. Replay images are overhead and are not new statistical samples.
Expected measured completion calls: student400,behavior129,CFG examples1058,
decode/verifier58,backward0,teacher0. Cap one L40 allocation at15minutes.

Source cumulative runtime counters and per-arm timers were lost with the process.
Do not fabricate them: retain a clearly labelled logical count of unique scientific
work derived from the frozen loops and complete records, report measured completion
counters separately, and use Slurm allocations including timeout for total cost.
Per-arm elapsed_seconds is null. Deterministic fitting mode is recovered from the
attested source control flow: evaluation started after its checked training context
exited. No new training occurred in this completion job.

Scientific conditions, labels,32 fresh seeds, primary metric, decision thresholds,
and the200-view frozen review stay unchanged. Prepare review only after all185
endpoints are assembled and provenance validated. This is completion of the same
scientific comparison, not a new training replicate.
