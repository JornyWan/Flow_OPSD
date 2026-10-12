# Post-hoc CPU diagnosis, not a new confirmation

Phase47 failed its frozen primary gate. This diagnostic does not change that
decision. Read saved numerical records and tensors only; zero GPU, image reads,
generation, training, model selection or changes to completed archives.

For all three arms, summarize selected/unselected index1 action drift and target
fit; all eight logged gradient norms; exact Adam first-moment decomposition into
beta1^8 times the previous moment plus new accumulated contributions; and every
saved before/after deployment trajectory's per-step state/action differences.
Report all arms/steps and no causality p-values. Moments are not parameter-update
attribution. Action differences after step0 combine changed weights and states;
step0 alone has exactly paired states. Target MSE is not terminal correctness.
No counterfactual generation, so this cannot establish momentum, sparsity or
off-window updates as the cause. A later controlled intervention is required.
