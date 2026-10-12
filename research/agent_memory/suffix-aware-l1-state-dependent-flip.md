---
name: suffix-aware-l1-state-dependent-flip
description: qlook1 (suffix-aware L1 target) one-step A→B→C→D forks — L1 update RELATIVELY better than canonical at ck300/50/60/70, flips worse in (70,80], in the post-query low-noise block; absolute gains mostly set by shared momentum
metadata:
  type: project
---

One-update forks (same ckpt / Adam state / seed, rollouts bit-identical, only the target differs), evaluated by
switching the updated weights in at subsets of flow-40 steps (§4.850–§4.854):
- canonical ck300 and qlook1 ck50/60/70: L1 update better than canonical *relative* (ΔD +0.2…+0.7×MDE), mostly via the
  post-query block (C−B = steps 36–39, order-dependent increment, not the σ≈0.22 call alone). But ABSOLUTE gains are
  dominated by the shared Adam momentum (G_M from −1.08 to +0.95) and L1's own batch gradient is ≈0 there — the relative
  win comes from canonical's batch gradient hurting held-out in qlook1's states (§4.857).
- qlook1 ck80: L1 worse (ΔD −0.55/−0.57; C−B −0.74, t=−17); query step neutral. cos(δ_canon, δ_L1) +0.45 → +0.40 → +0.30 → −0.13;
  at ck80 L1's gradient aligns with momentum (+0.15) while canonical's brakes it (−0.29) — so 'equal-norm' isn't equal there.
- Two fork seeds share the parent checkpoint: repeats, not independent training replicates.

**Why:** GPT's "shared parameters change other steps" mechanism is real but *state-dependent*: it only appears after L1
has moved the model (self-reinforcing drift on the steps its lookahead passes through), not as a one-step property.
**How to apply:** report G_L, G_C, G_M and ΔG together — never only the relative ΔG. Suffix protection must act on the
suffix response *after* the update (a plain keep-loss has ≈0 gradient since old≈θ). The §4.855 anchor (canonsfx/qlook1sfx) is
query-target PROPAGATION to the lookahead state (MSP gap term), not suffix-own OPSD nor behavior preservation — its failure
would not refute those. A frozen lookahead model (qlook1fz) alone does not protect the student's suffix steps.
One-step fork diagnostics must be done at several checkpoints — the plateau/early checkpoint gave the opposite sign.
Related: [[opsd-target-vold-gain-must-be-one]], [[dont-kill-idea-on-first-negative]].
