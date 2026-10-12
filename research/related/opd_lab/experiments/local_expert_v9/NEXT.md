# Next bounded training question: can the fresh repair survive learning?

Planning after phases8/9, not an executed/frozen GPU recipe. No phase10 job is
running. Keep the research objective: external experts propose corrections,
terminal verification accepts useful corrections, one ordinary student learns
them. The probes do not justify reopening a Q critic or making an inference mask.

The next most informative comparison is **fresh single-state fitting versus
complete verified-trajectory fitting**, with the same initial joint128 weights,
same five prompts/seeds, same one accepted teacher intervention and same loss/
optimizer/precision. The fresh apple/bowl GenEval1 branch is the positive repair;
use current-student baseline trajectories for the other four preservation controls.
This is deliberately a one-repair learning diagnostic, not a method comparison
or an unbiased estimate of verifier accuracy.

Data already exist in `runs/le9_requery_001/branches-19.pt` (apple), plus
branches09/11/12/17 for controls. Use record/seed/query index, every state/action
and their hashes. Apple branch `geneval@1` contains current-policy prefix,
the freshly verified expert action, and current-policy continuation evaluated
on the CORRECTED branch states. Other cases use `base`. Never substitute baseline
suffix actions evaluated on the wrong trajectory.

- Sparse arm: five query states per update; apple expert action, other four
  current-student actions as preservation targets.
- Dense arm: all ten state/action pairs of those five branch trajectories per
  update. The nine extra apple actions are not nine new expert queries: they
  specify the verified prefix/continuation. Preserve that provenance in the data.
- Both use original deterministic transition loss and LR3e-5/AdamW. Choose and
  explicitly label the common optimizer policy before execution: a weight warm
  start with reset Adam is not full optimizer-state resume. Never mix the two.
- A bounded initial budget can be dense32 updates ×50 states versus sparse320
  updates ×5 states:1600 backward queries each. Also save sparse32 to separate
  equal-update from equal-compute comparisons. Count all forward/audit/initialization
  costs, and report actual GPU-hours; matching backwards is not exact FLOPs.

Track query fitting, unchanged-prefix state displacement, complete-trajectory
prediction residuals, preservation on all four controls, and ordinary full
generation. Judge the fixed terminal checkpoints with a frozen rubric. No
per-image checkpoint selection and no automatic extra LR/budget if fitting fails.

If sparse learns the action but changes its approach state again while dense
preserves the new repair, that supports learning a coherent verified trajectory.
If dense underfits at this budget, do not attribute failure to the idea before
separating optimization from preservation pressure. If both learn locally but
lose whole-generation repair, investigate interpolation/state coverage further.

Only after this controlled one-repair loop works should the online algorithm
be expanded: refresh behavior and expert proposals on actual student states,
verify terminal gain against that same frozen current-policy continuation, then
fit with adequate trajectory coverage. The existing on-policy refresh is reused,
not claimed as a contribution. Compare best fixed expert/timing/strength and
density-matched screening on all training queries before held-out claims.
