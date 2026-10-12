---
name: autocast-cache-stale-weights
description: "Inside one autocast context, in-place edits to requires_grad weights (e.g. PEFT active adapter) are invisible — fp16 cast is cached; use cache_enabled=False or a fresh context per weight change"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-04T22:00:45.560Z
---

torch autocast caches the fp16 cast of **leaf tensors with requires_grad=True** until the outermost autocast context exits;
an in-place `copy_` / `load_state_dict` does not invalidate it. PEFT's `set_adapter` makes the active adapter requires_grad,
so swapping LoRA weights in place (set_peft_model_state_dict) inside one long `with autocast():` silently reuses the
first weights. Hit 2026-10-04 in P11 (§4.850): ± parameter perturbations gave bit-identical latents (‖z₊−z₋‖=0) while
fp32 weights differed; a per-forward-autocast debug looked fine, which hid it.

**Why:** the first pilot's numbers were entirely the +r perturbation for every cell; any probe that perturbs/swaps weights
in place (finite differences, interpolation, adapter surgery) is exposed.

**How to apply:** for weight-perturbation probes use `autocast(..., cache_enabled=False)` (casts are deterministic, values
unchanged) or open a fresh autocast context after each weight change; always add a guard that the ± (or before/after)
outputs actually differ. Training scripts are safe only if no autocast block spans an optimizer step / adapter copy
(checked for train_opsd_ri_sd3.py: all 10 blocks ≤44 lines, none contains a weight update). Related: [[reuse-the-correct-impl-dont-rewrite]].
