## Numerical protocol amendment for attempt003 (before evaluation)

This amendment supersedes only the historical final-weight equality requirement
and fitting execution mode. Attempt001 remains failed under its frozen protocol:
24 references matched exactly, but the single arm after128 updates did not match
the phase15 model. Attempt002 failed before fitting on H100. Neither generated
fresh evaluation images. Their inputs and outputs are retained without edits.

The bounded L40S/PyTorch2.6.0+cu126 diagnostic le18_repeatability_001 uses the
unchanged phase15 fitting function, identical original12 queries, identical
targets in all three named slots, and identical weight/optimizer/scaler/RNG resets.
Default execution produces three different models after two updates (maximum
absolute pairwise difference against first repeat up to7.6021e-5). Deterministic
execution produces three identical models. The new schedule-driven fitter,
including its15-state audits, also matches the deterministic historical fitter
exactly after two updates. All15 initial saved actions match in both modes.
Diagnostic arm names are repetitions, not scientific method comparisons.

Attempt003 sets CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization and
torch.use_deterministic_algorithms(True,warn_only=False) only around fitting all
three scientific arms. Reference generation and evaluation keep the previous
execution mode. Restrict to the qualified GPU model and installed PyTorch version;
any unsupported nondeterministic operation fails. All24 exact reference gates,
frozen source queries, loss, initialization, schedules, optimizer hyperparameters,
128 updates per arm, evaluation seeds,185 outputs, rubric and decision criteria
remain unchanged. No extra main-study training or evaluation is introduced.

The two-update GPU qualification is attested before training. This is evidence
of old/new fitter equivalence and repeatability at two updates, not proof of
128-update reproduction of the historical nondeterministic model. The old model
hash is retained and its equality to the new single arm is reported descriptively;
historical equality is no longer the validity gate for this amended attempt.
The three scientific arms share the new numerical mode. Compare them within this
attempt; do not represent changes relative to phase15 as method-only changes.

Report the14 diagnostic optimizer updates/168 backward queries and all failed GPU
allocations separately from the main study's384 updates/4608 backward queries.
No generated-image result informed this amendment. Freeze every new observation
before unblinding identities or computing the scientific comparisons.
