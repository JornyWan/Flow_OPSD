# Phase42: published Student / OCRTeacher implementation smoke

Claim: C1 implementation prerequisite, not efficacy. Four fixed smoke prompts,
one per count/color/relation/multi family; separate from dev and confirmation.
Each gets noop and a single assigned window, covering starts {1,3} and lengths
{1,2}. No selection or best-route claim from four images/pairs. No training.

Published checkpoint revision and LFS content hashes must match local files;
original configs must match remote Git blobs. Runtime PEFT config compatibility
deletions explicit, all retained fields unchanged. Loaded student and expert
tensors must equal published tensors; student/behavior must match and all
adapters must be unchanged at completion. Backbone large files use the existing
framework's size/mtime fingerprint, not claimed as full content SHA; adapters
are fully hashed. Source/config/constraint manifests frozen before submission.

SD3.5-M, Flow Euler10 shift3, 512, CFG4.5 for both adapters, FP16 autocast with
FP32 transformer weights/latents/VAE; deterministic algorithms and CUBLAS
:4096:8. CPU FP32 per-image noise. Each correction uses the exact base prefix;
all suffix steps recomputed with the same frozen behavior. Full independent
latent rollout must match every saved branch state/action exactly, including
second-step teacher queries at the intervened state. Targets detached. Explicitly
unsupported multistep-solver branching stays disabled. This is a local mechanism
engine, not a full native OPD recipe reproduction.

Save only eight decoded PNGs; independent replays do not decode. Expected cost
128 transformer invocations / 256 CFG examples, 8 decodes, 8 text encodings
including empty contexts. Record startup, generation, scoring, peak allocated
memory, and scheduler sigmas. No optimizer creation, no agent image inspection.

Score all eight saved PNGs. A uses unchanged EvidenceVerifier clause evidence;
semantic_accept checks the declared semantic clauses, unknown abstains and
runtime errors fail this smoke. Visibility proxy retained separately, no added
full-visibility requirement in these prompts. B uses native original-image
GenEval strict plus whole-image OCR exact with native normalization; native OCR
substring reward and CER separately. A/B share models; neither is human truth.
Keep raw detections, OCR evidence, color logits/masks. No scorer retuning.

Resource: one L40, <=20 allocated minutes, no automatic requeue. All attempts
count toward the already specified 3 GPU-hour smoke+development+confirmation
candidate budget. Never infer efficacy from smoke scores. On pass, prepare the
48-prompt dev and 192-prompt confirmation manifests and review measured runtime
against the shared budget before full generation. On failure diagnose the
specific contract; retain failed inputs and count its cost.
