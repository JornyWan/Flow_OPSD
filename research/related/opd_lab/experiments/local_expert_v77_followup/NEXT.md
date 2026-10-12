# After Phase77: distinguish learning coverage, task tradeoff and short-block fitting

Phase77 is the native dense task-adaptation comparator: fresh pure SD3.5M,
256 actual updates and final EMA primary. It does not test our innovation.
The launch includes complete384 paired new development images and all metrics.
No independent confirmation or publication claim follows from this128 reused dev.

Interpret primary EMA-S0 jointly with OCR, structure, CER, unknown and all five
quality metrics; RAW remains a declared diagnosis even if it happens to score
better. Complete bookkeeping or lower regression loss is not endpoint success.

- EMA joint improves with quality preserved: dense native OPD is a useful matched
  comparator. Test a fixed two-step closed-loop objective against the same native
  one-step objective, with same initial tensors, training prompts/seed policy,
  optimizer/actual LR, update count and final export/evaluation rules.
- Structure improves while OCR/joint declines: external GenEval expert may teach
  a task tradeoff. Do not assume more fitting fixes it. Quantify paired gained/lost
  constraints and trained-student quality before choosing the next experiment.
- Little functional change despite nonzero updates: inspect recorded timesteps,
  gradients/AMP evidence, loss dynamics, EMA versus RAW and prompt coverage. A
  finite256-update run alone does not establish convergence or underfitting.
- Execution/scorer/source failure: stop, retain evidence, diagnose independently;
  no silent GPU retry or promoting partial favorable outcomes.

Next method's distinguishing question: can a student learn a short sequence of
teacher corrections more faithfully when the second prediction sees its own first
updated state and gradients traverse that state? First isolate this mechanism
without changing verifier/teacher routing simultaneously. The CPU block core is
implemented; actual native trainer integration and GPU validation remain undone.
Do not use the old local rewrite with changed arithmetic as an unlabelled control.
Use the pinned native trainer as the integration point, and show a one-step
special-case regression against its existing objective. Freeze the two-step
teacher-target definition, per-block weighting/time coverage and whether any
extra teacher query is used. Match teacher-information exposure or explicitly
report that difference. At equal256 optimizer updates, extra forwards/backwards
need separate compute accounting and a subsequent compute-matched baseline.

Only after a matched learning signal, add terminal verification as one further
variable: dense base supervision retained, short-block intervention evaluated with
the same frozen student suffix/noise, clearly handle unknown as abstention, and
compare informative weighting with a matched random rule. This is the intended
terminally verified external-expert learning method; candidate interventions alone
are not the final claim. Future protocols and budgets remain separate and unsubmitted.
