# 2026-09-29 implementation review

Authorization: user's daytime request to continue and check implementation.
This is independent of the expired overnight budget. CPU inspection and tests;
zero image/screenshot/remote vision reads, no confirmation/final access.
Completed experiments and their source snapshots remain immutable.

First check: exporting a supplied EMA state in SD3Backend must preserve the
student, behavior, anchor, expert, active adapter, gradient ownership and mode,
both when saving succeeds and when saving raises. Use actual PEFT CPU adapters.
Preserve before/after sources and regression outputs. This path is not used by
Phase59/60 raw-only export, so it cannot explain those learning results.

Then audit active loss/selection/rollout and prepare the smallest justified next
step. Distinguish code defects, correctly implemented but limited designs, and
untested proposals. Any GPU pilot needs a separate prospective scope and budget.
