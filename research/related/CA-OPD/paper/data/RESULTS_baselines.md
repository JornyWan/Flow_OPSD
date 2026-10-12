# Multi-task RL baseline eval (n=256, gs=4.5) — for the comparison table
| method | GenEval | OCR | Pick | Clip | HPS | Aes | score |
| Multi-Task NFT @570 (still converging) | 0.639 | 0.854 | 21.34 | 0.280 | 0.240 | 4.907 | 0.440 |
| Multi-Task GRPO-Guard @720 | 0.718 | 0.826 | 22.93 | 0.291 | 0.305 | 5.603 | 0.706 |
| Multi-Task GRPO-Guard @780 | 0.720 | 0.826 | 22.92 | 0.289 | 0.304 | 5.648 | 0.707 |
| uniform DiffusionOPD (Ours) @721 | 0.958 | 0.939 | 23.98 | 0.298 | 0.342 | 6.172 | 0.985 |
| CA-OPD @781 | 0.953 | 0.947 | 24.00 | 0.296 | 0.343 | 6.145 | 0.978 |
OPD distillation >> direct multi-task RL (paper converged Avg: MT-NFT 0.715, MT-GRPO 0.763, DiffusionOPD 0.929).
Our MT-NFT/MT-GRPO are at current (non-converged) ckpts -> conservative lower bounds.

## UPDATE (re-eval at higher ckpts)
- Multi-Task NFT @690: COLLAPSED (GenEval 0.054/OCR 0.010/Pick 18.3/Aes 4.19 → score 0.000). Reward-hacking divergence after ~570. Best = pre-collapse @570 (0.440).
- Multi-Task GRPO-Guard @960: 0.757/0.849/22.94/0.291/0.308/5.637 → score 0.728 (up from @780 0.707; stable, still slowly improving).
