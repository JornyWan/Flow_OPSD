# CA-OPD — Conflict-Averse / Value-Gradient Multi-Teacher On-Policy Distillation
Full archive of the open-source DiffusionOPD reproduction + CA-OPD research.
- `code/DiffusionOPD-main/` — training/eval code (incl. CA-OPD, regret, conflict, tau-sweep, MVP-1 reward-grad + MVP-2 value-critic virtual teachers)
- `paper/` — CA-OPD paper (LaTeX source, figures, data/tables, compiled PDF)
- `reference_papers/` — arXiv sources of DiffusionOPD / Flow-OPD / DiffusionNFT / flow-grpo
- `results/eval/` — per-checkpoint per-axis eval JSONs (GenEval/OCR/PickScore/ClipScore/HPS/Aesthetic)
- `html/` — results dashboard
- `checkpoints/<run>/` — LoRA checkpoints for every run (every 30 steps)
