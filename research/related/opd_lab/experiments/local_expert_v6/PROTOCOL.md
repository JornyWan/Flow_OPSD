# Phase6: independent-model audit, before training-reward promotion

Freeze before Qwen outcomes. This phase adds an audit-only Qwen2.5-VL-7B reader
and scene judge, never a training reward. No optimizer update or validation/test
use. Old scores, reviewed images and labels remain immutable.

## Fixed cohorts

- Development: all44 phase4 noop/selected-target images and original frozen labels.
- Previously exposed calibration: all16 phase5 images, separately reported.
- New calibration:16 TRAIN prompts, four per family, excluding phase4 accepted
  IDs, phase3 review IDs and phase5 calibration IDs. Rank remaining IDs by
  SHA256(['phase6_calibration',id]); use stable_seed(271828,'phase6_calibration',id).
  Generate from the same original phase3 frozen behavior: SD3.5-M, fp16,512,
  flow10 and existing CFG. These prompts participated in earlier training; fresh
  noise and exclusion serve measurement calibration, not model generalization.

Freeze all16 new image observations before inspecting their v1/Qwen scores.
Same clause/visibility rubric as phase5, allowing true/false/null and literal
transcription. Implementing-assistant labels are provisional, not independent
human truth. Once the new images are viewed, never tune a rule on them and call
the same set an untouched check.

## Fixed model and calls

Qwen2.5-VL-7B-Instruct snapshot cc594898137f460bfe9f0759e9844b3ce807cfb5.
Hash all model/config/tokenizer/processor files and validate index/shard coverage.
Use the separate existing mi_reasoning interpreter, PyTorch2.9.1+cu128 and
Transformers4.57.6; do not upgrade diffusionopd. Record complete package versions
and key model/processor source hashes. One image/call, bfloat16, SDPA, eval mode,
greedy decoding (do_sample=false), seed0, maximum384 generated tokens per call.
Processor min_pixels3136/max_pixels262144, local-files-only, no remote code.

Two separate requests per image, without conversation history:

1. Image-only observation: all literal text lines, uncertainty and a short scene
   description. NO original prompt, target word, teacher, acceptance, score or
   other reader's output. Full-image text exact-match is computed afterwards with
   original v1 normalization. Also report whitespace-only compact matching as a
   sensitivity check, not as automatically preferred metric. Any uncertain text
   becomes an abstention rather than an automatic failure.
2. Scene grading: same image and explicit GenEval clause metadata, but NO expected
   text, original prose prompt, candidate identity, score or stage1 response.
   Return every requested clause and full_visibility as true/false/null, with
   evidence. Exact counts and required identity/colors/relations apply; missing,
   cropped or substantially occluded objects fail full visibility. Claimed scene
   satisfaction can still be biased by the requested metadata; report this limit.

The exact prompts, parser and decoding config are pinned in the plan. Strict JSON
schema validation allows one optional enclosing JSON fence, rejects duplicate keys,
unexpected/missing clauses and coercions (e.g. string 'true'). Preserve raw output,
token counts and parsing errors; invalid output is unknown, never a negative label.
No response repair or outcome-dependent retry. Infrastructure fixes use new attempts.

## Staging, analysis and budget

Smoke: first two development images in fixed plan order, four calls. Require model
loading and structurally valid output before the main sweep. Do not tune prompts
using the smoke's semantic correctness. Main sweep processes the remaining74
images, merging the successful smoke rows for full76-image coverage (152 calls).
The new SD3 calibration generation can run independently of VLM smoke.

Report development/exposed/new cohorts separately: valid outputs, abstentions,
coverage, TP/FP/TN/FN per OCR/count/color/relation/visibility, precision and recall
on reference-known cases. Unknown references never become false. Also give a recall
that counts abstention on known-positive cases as a miss, so abstaining does not
hide poor usable coverage. Compare original v1 on the same reference-known rows.
Count target preferences/independent scene checks descriptively; no new efficacy
claim, no target reselection or fitting based solely on this small audit.

One L40 per job: calibration generation<=10min, smoke<=10min, main<=30min. Log
actual GPU allocation, CPU hashes/reports, failed attempts and per-call token counts.
Finish all cohorts, integrity checks and records before deciding reward promotion.
