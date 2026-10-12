# Phase46 offline numerical report

Open report.html directly. No images, external assets or network requests.
Three frozen-pool fits from the published Student: all targets, family-count
matched random targets, terminally accepted targets. The 64-case fresh FULL
student rollout is the deployment endpoint. The 64-case local intervention
cohort is a reused-data mechanism diagnostic. No native online OPD claim.

Failure 1062322 (123 seconds, zero updates/images) and its original sources
remain in failed_attempt_001/; ENGINEERING_REVISION.md declares the device-only
repair before training and generation. Job 1062324 completed training/generation
in its 1441-second allocation, then failed at a historical smoke logging key.
SCORING_RESUME.md documents the metadata-only fix before new endpoint scoring.
All images/checkpoints are frozen and reused by a separate scoring-only job;
attempt_002_scoring_failure/ retains its original code and logs. No regeneration.
All three jobs count toward time. Original 1256-image budget remains unchanged;
this learning phase has a separate 448-image allowance. Human labels were not
used for fitting, sampling, checkpoint selection or the primary decision.

Weights, PNGs and trajectory tensors remain on the server. result.json binds
checkpoint hashes; evaluation_trajectory_hashes.json binds every generated
trajectory; original training/mechanism sources are bound in learning_plan.json.
This ZIP is a numerical report and source archive, not a standalone environment.
Direct agent image reads: 0. Completed phases and annotations are unchanged.
