# Phase76: full native OPD loop, fresh base, original per-card batch

Frozen before GPU; this is a capacity/cadence pilot, not an efficacy experiment.
One new L40S job <=1800 requested seconds; no retry, requeue, timer or reuse of
expired/failed budgets. No additional GPU launch from this script.

Question: does the pinned official SOPD-GenEval main run batch3 on L40S, execute
its true 27-backward/81-state update, preserve first native repeated EMA update,
save intact RAW/EMA/Adam/scaler/RNG, and what measured cost supports sufficient
training? No local loss replacement, precision-fix candidate or new method.

Pristine official commit0c85bf0193a6da24bb1f72152b612d353a154999. Existing managed
lifecycle/asset bindings plus read-only audit calls. Removing audit AST nodes
must exactly recover the managed original. The calls add timing/IO overhead,
which is included in measured epoch time and must not be called pure throughput.

Start pure SD3.5M with fresh rank32/alpha64 gaussian LoRA, seed42, initial LoRA B
all-zero checked. No published OPD Student continuation. One rank, 8 outer
epochs/8 actual Adam updates; sample/train batch3,3 batches/epoch,9 trajectories
and81 supervised states/epoch. Totals72 draws,648 states,216 backward calls.
512 TRAIN text prompts, original sampler/DataLoader num_workers1. True consumed
prompt IDs, pre-sampling CPU/CUDA RNG and actual initial latent noise saved.
Native same_latent=false has a shared RNG stream, NOT per-image integer seeds.

LR3e-4 and native AdamW/clip/FP16/TF32/config preserved; studentCFG4.5,
GenEvalCFG1, Flow10, text128,512px. No gradient-checkpointing change. Native
VAE decode/training previews remain in sampling; expected72 training decodes,
no agent reads. Empty monitoring-only proxy rewards avoid development generation;
no task scoring or five-quality evaluation in this bounded pilot. Original
teacher targets/loss/backward/Adam/EMA statements unmodified. Global batch9 is
one-card task adaptation, not original eight-card batch72/paper reproduction.

Checks: all expected calls/states, finite losses, true Adam1..8 without AMP skip,
actual LR3e-4 throughout; native EMA eligible calls exactly(epoch6,batch2,step7),
(epoch7,batch0,step7),(epoch7,batch1,step7). CPU replay/checkpoint EMA max error
<=1e-6. RAW and EMA both change relative to initial, original export/checkpoint
payloads present. No deployment quality inference. Previous save/replay tests
remain separate evidence; this pilot does not claim fresh trained-image gains.

Stop on OOM, skipped optimizer update, technical error or inconsistent evidence;
preserve failure and do not automatically lower batch/turn on checkpointing or
retry. If successful, use epoch timings/peak VRAM to freeze a fresh 256-update
matched OPD baseline and complete joint/OCR/structure/CER/unknown plus PickScore,
CLIP,aesthetic,HPSv2.1,ImageReward evaluation. 256 is a coverage-based candidate,
not a proof of convergence. Budget/evaluation and actual training launch require
separate protocol; no checkpoint cherry-picking. Candidate fixed two-step closed
loop remains CPU-core only and not mixed into this native baseline.

Probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu before each submission; sablab
allocated+pending across projects<=6, ordinary queue preferred at similar starts.
Expected start<=24h. Preemption means stop without retry. No other job changes.
Model large-file metadata inventory limitation disclosed; source, teacher and
data fingerprints retained. Confirmation512/final768 remain unopened.
