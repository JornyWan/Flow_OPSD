# New bounded paper-oriented session

The user explicitly requests completing the planned round and continuing to
more complete publication-oriented experiments. Keep the old shared 10800-second
pool untouched (10668 used, 132 remaining). This new session has at most 10800
allocated single-GPU seconds across all jobs, including failures and loading.
At most two GPUs concurrently for genuinely independent stages. No automatic
time extensions, image expansion or retries. Record every job and outcome.

First: Phase48 LR-only development ablation, 1200-second cap, 96 new images.
Second: irrespective of LR outcome, complete the missing learned-action local
intervention for both LR settings, frozen prefix/suffix, all32 development prompts;
teacher, high-LR student, low-LR student, 96 images, separately capped1200 seconds.
Do not choose a winner after viewing the development outcomes for that mechanism
test. This stage is diagnostic, not new generalization evidence.

In parallel, audit native baseline identities and standard evaluation datasets,
define and freeze paper claim/evidence tables, split/exposure controls, precision
planning and baseline budgets. Do not label pilot controls as native algorithms,
exposed prompts as fresh tests, or incomplete evaluation as paper-ready results.
Reserve subsequent GPU jobs individually only after protocols and preflight.
