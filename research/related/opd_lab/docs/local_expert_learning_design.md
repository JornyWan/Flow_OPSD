# Terminally verified external-expert learning

The question is whether a globally stronger teacher is locally helpful at the student's actual state. Teacher disagreement alone is not evidence of a beneficial correction. Compare final outcomes after a controlled intervention, then distill only corrections whose observed outcome repairs a constraint without a known loss elsewhere.

## Separation of claims

There are three distinct requirements:

1. **Expert opportunity:** a short external-expert intervention can improve the endpoint under a fixed continuation.
2. **Reliable selection:** an automatic verifier identifies useful interventions using only information available at selection time.
3. **Learning and deployment:** fitting the selected actions produces a student whose own deployed trajectory improves.

Phase27 primarily checked selection on new words/noise within familiar templates. Phase28 checks the third requirement with controlled learning. Passing the first two does not imply the third. A terminally good target can be difficult to fit, depend on a state the learned model no longer visits, or induce unwanted changes at other times.

## Controlled target construction

For prompt/noise pair i, obtain the frozen student's state after three actions. Compare its ordinary completion against an OCR expert's two sequential actions followed by frozen-student completion. The second expert query is on the state produced by its first action. Every continuation query is recomputed at the actual state.

The same terminal verifier decides whether to retain both expert actions. The action loss for query j is one half of the mean squared Euler-transition difference:

`0.5 * mean((delta_sigma_j * (v_student_j - stop_gradient(v_expert_j))) ** 2)`.

Both masked arms divide summed active losses by all240 possible training query slots. A rejected target gives no gradient; it is not silently replaced with a preservation target. This keeps target acceptance as the controlled difference. The all-target arm has more backward queries; only verified and random arms are matched in supervised-query count.

## Why the fixed-prefix diagnostic matters

Training on two local times updates shared model parameters that are also used at the other eight times during full generation. It can therefore change the prefix that reaches the supervised region and the suffix that converts a correction into an image. Lower target MSE alone cannot distinguish this effect from improved generation.

The Phase29 failure branch holds the old prefix and continuation policy fixed while inserting two actual learned actions. Compare each student's local endpoint with its own saved full endpoint, initial G, and the fixed OCR reference. A local-only gain is evidence that the intervention support matters. Failure even locally instead motivates checking target fitting, the second-action state distribution, and preservation of constraints. These are diagnostic patterns, not a causal separation of prefix and suffix effects.

The local branch could eventually correspond to a deployment policy that activates learned weights only within a time window. At the tested10-step grid that policy needs no online OCR expert, but it retains the initial adapter outside the window. It must be named separately from a single all-time student. Generalization to other grids would require a continuous-time definition and explicit testing; the present study provides no sampler robustness evidence.

## How to interpret small visual studies

Keep complete-scene correctness separate from per-clause improvements, harms and unknowns. A model can win on complete-scene count while breaking individual constraints elsewhere. Keep paired seeds together and bootstrap prompts within task family. Preserve all cases, not only attractive wins.

Method-hidden implementing-assistant observations are useful provisional evidence. They do not replace independent human validation, especially for ambiguous lettering, object identity and partial visibility. Reserved final prompts remain ungenerated while development choices are being made. A predeclared engineering threshold selects the next experiment; it is not a substitute for uncertainty intervals or a final held-out evaluation.
