# Evaluation resource policy

User instruction, 2026-09-22: repeated Codex usage limits make bulk agent image reading unacceptable.

- Default to **zero direct image reads** by the coding agent for experiment evaluation. Use pinned local evaluators, cached predictions, numeric summaries and logs. Do not send image batches to another agent or a remote vision API as a workaround.
- Preserve and summarize existing assistant observations separately from automated predictions. Never fill unread images with inferred assistant labels or describe assistant observations as independent human ground truth.
- Only inspect a small number of images when explicitly requested by the user, or when a concrete implementation defect cannot be diagnosed from saved evidence. State the reason and cap the latter at four images per task. Do not automatically open report screenshots or plots.
- Generate HTML comparison galleries for the user when useful; validate bindings, image hashes, loading and layout through browser code without returning screenshots to the agent.
- Cache automated evaluation by image content, prompt/constraints, evaluator code/config/model fingerprints. Report unknowns, errors and measured disagreement. Never tune an evaluator on the evaluation cohort and report it as independent validation.
- Keep raw per-image evidence in files. Read compact aggregate summaries and targeted error excerpts into the agent context; avoid dumping large JSON/JSONL files or repeated full logs.
- Phase34 assistant inspection stopped after cases 1–70 (350 images), by user instruction. Do not resume it. The original 600-image manual-review plan is superseded by `experiments/local_expert_v34/EVALUATION_AMENDMENT.md`.
