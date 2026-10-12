# Phase14: frozen paired repair/damage development audit

This protocol is fixed before the first phase14 model request. This is development
on previously exposed phase12 images and provisional implementing-assistant labels,
not independent human truth, new-seed confirmation, a trained-policy result or a
sampler-robustness experiment. Old reports, labels and verifier definitions stay frozen.

## Question and population

Can direct paired evidence identify useful external-expert local actions without
imagining repairs or sacrificing already-correct requirements? Use every phase12
train query:64 states,192 original512px endpoint images,128 noop/expert pairs. These
come from the fixed joint128 student,flow10,index3,alpha1 and the same OCR/GenEval
expert pool. No new diffusion call, teacher action or optimizer update is authorized
by this particular protocol. There are no validation/test images in its inference list.

For each of128 real pairs, evaluate both A/B orders; add64 noop/noop controls. Freeze
UIDs, all image hashes, source reports, environment, code, requirements and hash-based
request order in paired_plan.json before submission. Model messages contain no expert
identity, original score, selected flag, training id, review label or filenames.
Original/candidate identity remains in a separate analysis key. This is presentation
blinding only: researchers already saw the development pool.

## Fixed interface and inference

Reuse the phase6 fully hashed Qwen2.5-VL-7B-Instruct snapshot and independent
mi_reasoning interpreter/environment. Check all14 model assets, all5 weight shards,
source hashes and exact environment again on GPU; do not upgrade any environment.
BF16,SDPA,greedy,seed14014,batch1,min_pixels3136,max_pixels262144 per image. Preserve
actual image grids, processor configuration, raw answers and token counts.

First make192 independent literal-observation calls, with the unchanged target-free
phase6 observation request (no expected word, scene prompt or constraints). Freeze
all observations before the pair stage. Then320 two-image calls:256 swapped real
comparisons+64 self controls. Each gets its two observations and only the explicitly
encoded requirements. Compile relations as subject/direction/reference with exact
counts. Do not add sign color/location or other unlabelled prompt details.

Each pair must give per-clause A/B nullable booleans, repair/damage/same/unknown and
evidence in both images; full_visibility is a separate field. Nullable evidence is
allowed for unknown claims. A determinate claim without concrete nonempty evidence,
inconsistent change, malformed or duplicate clause becomes unknown for that clause;
valid other clauses survive. A bad individual boolean becomes unknown at that side.
Invalid top-level JSON, duplicate JSON keys or token truncation makes the whole answer
unknown. No regex value recovery, resubmission, formatting repair or adaptive prompting.
All errors count in denominators. Evidence is recorded, not assumed truthful.

Primary selection `paired_evidence_dev1` first maps both orders to noop/candidate;
a per-side label survives only if both orders agree. For text, it must also agree
with target-free literal exact match (NFKC,casefold,collapsed whitespace; preserve
extra letters, punctuation, all lines). Uncertain literal transcription abstains.
This is a joint interface/observation/consistency change, not a pure causal estimate
of two-image input alone.

An eligible candidate must repair at least one known false encoded clause to true,
have all its own labels known, and preserve each baseline true or possibly-true
clause/visibility by a candidate true. Unknown-to-true never earns repair credit;
baseline false may remain false. Visibility gain alone is not an encoded repair.
Rank eligible candidates by repair count, smaller saved action distance, then the
fixed OCR-before-GenEval tie order; otherwise noop. The baseline label may depend on
pair context and this limitation is explicitly measured through swapped responses.

Predeclared mechanism diagnostics use the same ranking: forward-only pair evidence,
reverse-only evidence and pair-order consensus without literal guard. These cannot
replace a failed primary result. Reproduce v1, discrete-event, phase13v2 and explicitly
posthoc dominance statistics for comparison, without changing their definitions.

## Budget and success criteria

OneL40,4CPU,96GB,45min maximum allocation (0.75GPUh). Exactly512 calls when complete;
observation cap256 new tokens, pair cap640, total output cap253952 tokens. Per-call
input limit12288 (aggregate upper bound6291456); exceeding it fails the run. No
validation generation is automatically triggered by partial/provisional outputs.
Failures retain logs and cost; recovery must get a separately named attempt/protocol
if the fixed worker or request design changes. Report Slurm allocation including
imports, hashing, model loading and all requests, separately from loop elapsed time.

Report all128 real pairs, all192 observations, and all64 self controls. Metrics include
per-field unknowns, parsing/truncation, candidate repair/harm confusion, candidate and
selected safe precision/recall, known harm, selected unknowns, complete-scene outcomes,
swapped agreement and self-image false change. For selected-state rates use2000 paired
whole-prompt bootstrap resamples,seed0,all64 prompts; pair ratios also cluster within
these64 prompts. Labels and the original strict safe-repair definition are unchanged.

Development readiness requires ALL:

- At least4 of the6 known safe states selected safely; conservative precision>=80%.
- Zero selected known harm and at most10% selected unknown under frozen review.
- Zero raw self-control repair OR damage assertions after schema validation, and
  >=80% self-control fields with both endpoint labels determinate.
- Among all128 real pairs, >=80% fields have all four swapped labels determinate;
  >=90% of those fields agree on the canonical noop/candidate pair.
- Complete planned coverage and valid input/code/environment/model attestation.

Same-image unknowns do not pass as known agreement. Report additional full-denominator
order agreement, state choice agreement and raw self-control acceptance. A passed
development gate permits preparing a separate prospective128-new-state protocol; it
does not establish independent validity. A failed gate stops automatic expansion and
training. Diagnose whether misses come from expert headroom, literal observations,
paired reasoning or consistency guards; narrow the scientifically testable task or
obtain independent annotation before spending on more fitting.
