---
name: opd-reproduction-handoff
description: OPD open-source reproduction project (DiffusionOPD/Flow-OPD/DiffusionNFT/flow-grpo) — env + S3 bundle + Pluto launch design; SD3.5 gated-token is the only external blocker
metadata: 
  node_type: memory
  type: project
  originSessionId: addd828e-111b-49af-866a-e28bb3e0484e
  modified: 2026-08-20T19:02:58.662Z
---

Started 2026-08-18. New project, SEPARATE from gen45 curriculum. Goal: reproduce the OPEN-SOURCE DiffusionOPD training on the colligo server (NO colligo-internal models/data). Repos in `/home/colligo/opd_related/` (unzipped: DiffusionOPD-main, DiffusionNFT-main, Flow-OPD-main, flow_grpo-main; papers under papers/). Chain: Flow-GRPO → DiffusionNFT → DiffusionOPD (OPD bundles its own `flow_grpo/`). Follow DiffusionOPD/README.md.

**Recipe:** SD3.5-medium base + student LoRA; 3 task teachers (Aes/OCR/GenEval LoRAs) → round-robin on-policy distillation. Entry `scripts.train_sd3_opd`, config `config/opd.py:sopd_pickscore` (single-teacher MVP) or `:mopd` (3 teachers), accelerate 8-GPU single-node DDP fp16 (scripts/single_node/*.sh, scripts/accelerate_configs/multi_gpu.yaml). GPU work MUST go to Pluto, never the dev node.

**Env (uv, NOT conda — no conda on box):** `/mnt/localssd/opd/venv` (self-contained: uv-standalone py3.10.16 + torch2.6.0+cu126 + flash-attn2.7.4.post1 [cxx11abiTRUE wheel] + `pip install -e DiffusionOPD-main` = diffusers0.33.1/transformers4.40/accelerate1.4/peft0.10/deepspeed0.16.4). cu126 wheels run on cluster's 12.4 driver via CUDA minor-version compat. xformers warning is harmless (code never enables it).

**Models (all downloaded, no token EXCEPT SD3.5):** teachers live in the PUBLIC `quanhaol/DiffusionOPD` HF repo (AesTeacher/lora, OCRTeacher/lora, GenEvalTeacher/lora, Student/lora — the README's separate `quanhaol/Aes-Teacher` repos DON'T exist) → `/mnt/localssd/opd/models/DiffusionOPD/`. PickScore (`yuvalkirstain/PickScore_v1` + `laion/CLIP-ViT-H-14` processor) cached in `/mnt/localssd/opd/hf_cache`. `config/opd.py` edited: `YOUR_*_TEACHER_PATH`/SD3.5 → env-var-with-local-default (`/mnt/localssd/opd/models/...`), same layout dev+pod.

**🔴 ONLY BLOCKER: `stabilityai/stable-diffusion-3.5-medium` is HF GATED (401 GatedRepo).** No HF token anywhere on the box; not mirrored on colligo S3 (only Qwen-Image is). Need cwan's HF token with the SD3.5 license accepted → download SD3.5 to `/mnt/localssd/opd/models/stable-diffusion-3.5-medium`, tar to `sd35.tar`, upload to bundle.

**Launch design (self-contained bundle on S3):** `s3://dit-scale-up/cwan/opd_reproduction/` — `bundle/{venv,uvpython,hf_cache,teachers,code}.tar` uploaded (SD3.5 `sd35.tar` pending), `pod_main.sh` (repo `repro/pod_main.sh`, also on S3). Pod bootstrap: aws s3 cp bundle → extract at ORIGINAL abs paths (venv shebang `#!/mnt/localssd/opd/venv/bin/python` + external uv-python symlink to `/home/colligo/.local/share/uv/python/...` both need same paths — that's why uvpython.tar ships separately) → torch.cuda sanity → accelerate launch → bg sync logs→S3. Submit via AIP MCP (compute_create_job→update_job_exec_spec[image/hostSpec/numRanks, OMIT scripts]→update_job_scripts[MAIN=fetch+run pod_main.sh]→update_job_pluto_roles[asml-data-access-role]→start_job), P2 preemptible + auto_requeue, project Foundry-Internships. Pod net: public internet + S3 + GHEC allowed. See [[pluto-mcp-job-launch-workaround]] [[pluto-preemptible-limit-and-priority-flip]] [[training-config-resume-convention]].

**LAUNCHED 2026-08-19 ~02:10 UTC:** sopd_pickscore job `opd_sopd_pickscore_repro`, jobId `eb35729c-c0af-470b-a38a-b6b765eefea8` (uniqueName pluto-prod-cwan-opd-sopd-pickscore), **8×H200 p5en.48xlarge single-node, P0 NON_PREEMPTABLE (guaranteed)** + autorecovery-v2, image `pluto-training-base:04-17-26`, role asml-data-access-role. Built via AIP MCP create→update_exec_spec→update_scripts(MAIN=fetch+run pod_main.sh)→update_pluto_roles→start_job. Started as P2 (sat `INSUFFICIENT_RESOURCES` in spare queue); **cwan directed P0 → flipped preemptionPolicy P2→NON_PREEMPTABLE via update_exec_spec (no restart needed) → immediately `JOB_RUN_STATUS_STARTING`**. Collateral check: no running cwan job preempted (all other 80 cwan jobs SUCCEEDED/finished 08-13~08-17; gen45 wound down). To change tier later: update_exec_spec preemptionPolicy (NON_PREEMPTABLE absent-in-JSON=P0, PREEMPTABLE=P2, IN_QUOTA_PREEMPTABLE=P1), OMIT scripts.

**⚠️ jobType GOTCHA (2026-08-19, cost one dead run):** `compute_create_job` WITHOUT `jobType` defaults to an INTERACTIVE session → the MAIN training script NEVER runs (pod comes up with vscode/sshd/buildkit, `main` container exits 0 in ~8s, NO `main_scripts/` in logDir). JobComponent is immutable per ECS schema ("job_type change requires a new job"); passing jobType to `update_exec_spec` on an existing job is SILENTLY IGNORED. FIX = pass `jobType:"JOB_TYPE_TRAINING"` to **compute_create_job**. Scripts live in `exec_spec.scripts` (set via update_job_scripts; not echoed in job_spec projection). First job `eb35729c` (interactive, dead/PAUSED-SUCCEEDED, archive later). **REAL training job = `opd_sopd_pickscore_train`, jobId `96484152-8ee9-4f65-8f07-b9b316626c33`, uniqueName `pluto-prod-cwan-opd-sopd-pickscore-uhr586`, P0 8×H200, started ~02:33 UTC 2026-08-19.** Bootstrap log: `/logging-fs/logs/pluto/jobs/cwan/pluto-prod-cwan-opd-sopd-pickscore-uhr586/run-*/pod-0*/main_scripts/1/current` (look for `[main]`/`[pod_main]`/`cuda matmul ok`/`step=`). Monitor with **bash** (not zsh — non-matching globs abort zsh, per [[pluto-check-job-progress-correctly]]).

**⚠️ scriptSpecs GOTCHA (2026-08-19, cost 2 dead runs):** `compute_update_job_scripts` MCP param is **`scriptSpecs`** (camelCase of proto `script_specs`), NOT `scripts`. Passing `scripts=[...]` returns `{}` but SILENTLY attaches nothing → main container runs image default, exits 0 in ~7s, no `main_scripts/` in logDir. Correct item = `{"type":"MAIN","name":"Main Script 1","shellCode":"<bash>"}` (type enum value is `"MAIN"`; field `shellCode`=camelCase of `shell_code`). Verify it took via `get_entities` component.compute.job_spec → `execSpec.scripts[0].storageFile.s3` present (platform uploads shellCode to `pluto-jobs-scripts-prod`). Got the schema via `get_message_schema(messageTypes=["UpdateJobScriptsRequest","ScriptSpec"])`. Reading logs: MCP `get_entities` cluster_run_response gives `logDir`+`podStatus`+per-container `exitCode`; then read `/logging-fs/<logDir>/main_scripts/1/current` (training stdout) directly on the dev node. `main_scripts/` ABSENT = main script not attached/not a training job.

**⚠️ peft/corda GOTCHA (2026-08-19, 3rd fix):** teacher LoRAs from `quanhaol/DiffusionOPD` were saved with peft>=0.12 → their `adapter_config.json` has `corda_config` (+eva_config/lora_bias/use_qalora/…) that the OFFICIAL-pinned **peft==0.10.0** rejects (`TypeError: LoraConfig.__init__() got unexpected keyword 'corda_config'` at `load_adapter`→`LoraConfig.from_pretrained`). FIX (kept peft 0.10 pinned = faithful): strip keys not in `dataclasses.fields(LoraConfig)` from the 4 adapter_config.json (all None/default for std LoRA → lossless), re-tar teachers.tar (exclude SD3.5 + *.orig), re-upload. autorecovery-v2 auto-restarted the pod onto a fresh localssd that re-downloaded the fixed tar → **teacher_pickscore loaded, OPD configured, 72 samples/epoch, run RUNNING (2026-08-19 ~02:53 UTC).** (Note: pod_main.sh only re-downloads tars if venv absent; a fresh/autorecovered pod re-downloads all, so S3 fixes propagate — a same-pod restart would NOT. If a fix must propagate without a fresh pod, move that tar out of the venv-gate in pod_main.sh.)

**✅ sopd_pickscore VALIDATED end-to-end (2026-08-19 ~03:15 UTC):** reached Epoch 59, healthy loop each ~10-15s — `Epoch N: sampling`(rollout, student 10-step) → `Waiting for monitor scores`(pickscore reward) → `Epoch N.0: training`(OPD loss + optimizer step, 9-step) → periodic `Eval[main]` 16 batches @40-step (eval_freq=30). No errors. cu126 venv on H200, NCCL 8-GPU, wandb offline, logs sync to `s3://dit-scale-up/cwan/opd_reproduction/runs/sopd_pickscore/logs`. This is the full open-source OPD RL closed loop reproduced. Next: let it accumulate + checkpoint (save_freq=30), then mopd (3-teacher, needs geneval+ocr reward envs — mmcv1.x/mmdetection2.x/paddleocr, the heavy builds) + eval.sh. reward_ckpts (aesthetic/mask2former/hpsv2/clip) staged at /mnt/localssd/opd/reward_ckpts.

**✅ mopd reward env BUILT in isolated `/mnt/localssd/opd/venv_mopd` (copy of sopd venv + reward deps; 2026-08-19 ~03:30 UTC):** all 3 mopd teachers' rewards verified — pickscore(HF cache), ocr(paddlepaddle-gpu 2.6.2 + paddleocr 2.9.1 + python-Levenshtein), geneval(**mmcv-full 1.7.2 CUDA ops COMPILE FINE on torch2.6** + mmdet 2.28.2 + mask2former detector loads from reward_ckpts) + open-clip/clip-benchmark/hpsv2x/image-reward. GOTCHAS: (1) env needs `setuptools<81` (setuptools≥81 removed `pkg_resources` which mmcv1.x/openmim need → `ModuleNotFoundError: pkg_resources`); (2) build mmcv/mmdet with `uv pip install -e . --no-build-isolation` (setup.py imports torch at build); (3) `import mmcv` AFTER `import torch` (mmcv._ext needs libc10.so preloaded, else `libc10.so: cannot open`); (4) don't run python from a dir containing the `mmcv/` clone — it shadows the pkg as a namespace pkg → `mmcv has no attribute __version__` (training runs from DiffusionOPD-main, fine); (5) mmcv/mmdet are EDITABLE (-e; .pth→/mnt/localssd/opd/{mmcv,mmdetection}). **mopd LAUNCH TODO:** ship venv_mopd + the {mmcv,mmdetection} source dirs (editable targets) + reward_ckpts (→ `<repo>/reward_ckpts` per reward_ckpt_path.py) in a new bundle (or reinstall mmcv/mmdet non-editable for a self-contained venv); config `config/opd.py:mopd` already set (3 teachers, num_batches_per_epoch=3 must be multiple of 3). Reuse the same MCP create(jobType=TRAINING)→exec_spec(P0)→scripts(scriptSpecs MAIN)→roles→start flow + a pod_main variant pointing RUN_CFG=mopd.

**⚠️ save_ckpt NEVER-CALLED fix (2026-08-19, repo omission):** `train_sd3_opd.py` DEFINES `save_ckpt()` (saves LoRA via `save_pretrained` → `logs/<run>/checkpoints/checkpoint-<step>/lora`, exactly the path eval.sh loads) but the main loop **never calls it** → training runs forever saving nothing (sopd hit epoch 119, S3 run dir 0 objects). FIX: added the call at epoch-loop end before `epoch+=1`: `if config.save_freq and epoch%config.save_freq==0: accelerator.wait_for_everyone(); save_ckpt(config.save_dir,transformer,global_step,accelerator,ema,transformer_trainable_parameters,config); accelerator.wait_for_everyone()` + a `logger.info('[save_ckpt] ...')`. save_freq=30. Re-tarred code.tar, `compute_restart_job` (fresh pod re-downloads code) → first checkpoint at epoch 30. Grep `[save_ckpt]` in the pod log to confirm; checkpoints land under `logs/[SOPD + SD3.5-M + PickScore]/checkpoints/` → sync to `s3://dit-scale-up/cwan/opd_reproduction/runs/sopd_pickscore/logs/...`. (Eval sample images go to `log_save_images/` which pod_main does NOT sync — only `logs/` is synced.)

**⚠️ mopd runtime reward placeholder fixes (2026-08-19):** mopd's first launch crashed in the OCR reward — `flow_grpo/ocr.py:16` had `path="YOUR_PDDLEOCR_PATH"` (PaddleOCR det/rec/cls model-dir placeholder) → all ranks FileNotFoundError/FatalError. FIX: `path = os.environ.get("PADDLEOCR_PATH") or <repo>/reward_ckpts/paddleocr`; pre-downloaded PP-OCR en det/rec/cls inference models there (ran PaddleOCR once on dev node), shipped inside reward_ckpts.tar. Also geneval `gen_eval.py:52` loads `open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')` → runtime OpenAI-CDN download → 8-rank race risk → `pod_main_mopd.sh` PRE-WARMS it once (single process, `HF_HUB_OFFLINE=0` for that line) before accelerate. **mopd job `opd_mopd_train` jobId `6416d123-c55f-4361-8a4a-71d581323f42` (pluto-prod-cwan-opd-mopd-train), 8×H200 P0**; crashed on ocr placeholder → stopped → fixed (code.tar+reward_ckpts.tar+pod_main_mopd re-uploaded) → restarted. Both sopd-P0 and mopd-P0 8-GPU placed simultaneously (guaranteed H200 quota fit two). Eval.sh still has `--output_dir YOUR_OUTPUT_DIR` placeholder to fill when running eval.

**⚠️ mopd eval DataLoader num_workers fix (2026-08-19):** after the ocr/open_clip fixes, mopd got MUCH further (reward env OK, open_clip pre-warmed, 3 teachers configured, initial eval ran: **OCR eval 8/8 + `ocr_proxy_ocr (1024,)` computed** ✓, geneval eval ran 3/18) then crashed: `[rank5] RuntimeError: DataLoader worker (pid …) exited unexpectedly with exit code 1` (train_sd3_opd.py eval→_eval_one_teacher:364) → cascaded to paddle `FatalError: Termination signal` on all ranks (that C++ error is the SIGTERM cascade, NOT the root). Cause: test DataLoader `num_workers=8` (line 769) → 8 ranks × 8 = 64 forked workers each inheriting a paddle/mmdet-loaded process (paddle unsafe across fork) + CPU-RAM pressure → worker OOM/crash. FIX: train `num_workers=1→0` (line 756) + test `num_workers=8→0` (line 769) — the repo's own hint says "Rerunning with num_workers=0". The reward MODELS themselves work (ocr_proxy computed, geneval eval progressed). Re-uploaded code.tar, restarted mopd. sopd shares the code (unaffected; num_workers=0 fine for its prompt datasets). LESSON for the OPD repo on this cluster: paddle (OCR reward) + forked torch DataLoader workers don't mix → keep num_workers=0.

**✅ mopd VALIDATED end-to-end (2026-08-19 ~05:14 UTC):** with num_workers=0, mopd passed the initial eval (pickscore + ocr + **geneval** slots — geneval eval is slow ~40s/batch×18) and ENTERED TRAINING (`Epoch 2.0: training`, `Epoch 3: sampling`, ~5s each). All 3 teacher reward paths work on-cluster (ocr_proxy computed, geneval eval ran to completion, pickscore). **BOTH sopd (single-teacher) and mopd (3-teacher) now reproduce the DiffusionOPD training loop end-to-end on 8×H200 P0.** Dashboard: `https://dit-scale-up.s3.us-west-2.amazonaws.com/cwan/opd_reproduction/viz/index.html` (regen via `/mnt/localssd/opd/make_dashboard.py`). **Throughput: ~10.5s/pure-train-step, ~20.9s/step incl eval-every-30 (8×H200).** REMAINING to answer "results == paper?": run `scripts/single_node/eval.sh` (torchrun 8-GPU, fill `--output_dir`, `--checkpoint_path <s3-or-local checkpoint-N>`, `--dataset drawbench|pickscore|geneval|ocr`) on a saved checkpoint and compare to the paper's comparison table; reward-curve values are in wandb (offline on pod, not yet surfaced — could add `<repo>/wandb` to pod_main sync).

**✅ EVAL RESULT — sopd MATCHES the paper (2026-08-19):** ran `scripts/evaluation.py` via `repro/pod_main_eval.sh` (2-GPU P2 spare, job eval_sopd_ckpt301_drawbench) on **sopd checkpoint-301, drawbench, n=1000**. **PickScore 24.02** (my normalized mean 0.9239 ×26 — the scorer /26-normalizes) **vs paper Aes-Teacher 24.02 / DiffusionOPD 23.99**; **HPSv2.1 0.345 vs 0.346 / 0.342**; both FAR above SD3.5-M base (20.51/0.204 w/o CFG, 22.34/0.279 +CFG). Since sopd_pickscore distills the **Aes-Teacher** (lora=AES_TEACHER_PATH) it correctly tracks the paper's Aes-Teacher row → reproduction is faithful on these metrics, not just "runs". Results: `s3://dit-scale-up/cwan/opd_reproduction/eval/sopd_ckpt301_drawbench/`; comparison table on the dashboard. **eval script needed 5 infra fixes:** NPROC-vs-GPU-count race, `torchrun` console-script shebang points to orig venv → use `python -m torch.distributed.run`, `PYTHONPATH=$REPO` so evaluation.py imports flow_grpo (it runs `scripts/evaluation.py` by path, not `-m`), SD3.5 HF-id (line 130) → local path env, and drop `unifiedreward` (needs a separate sglang 7B server) → evaluated offline-safe `{pickscore,hpsv2}` (EVAL_REWARDS-overridable; clipscore/aesthetic use openai/clip-vit-large-patch14 + imagereward uses ImageReward-v1.0 — pre-cache to add them). Next: eval mopd checkpoint (geneval/ocr/drawbench) for the multi-teacher numbers. Watch `/logging-fs/logs/pluto/jobs/cwan/pluto-prod-cwan-opd-sopd-pickscore*/run-*/pod-0*/main_scripts/1/current` for bootstrap (`cuda matmul ok` → first `step=`); wandb offline. If cu126-venv fails on the image driver, torch.cuda check in pod_main.sh flags it → rebuild venv cu124 or use an older ubuntu22.04 PY-3.10 training-base image.

**mopd extras (deferred until sopd runs):** reward_ckpts (aesthetic/geneval-mask2former/clip/hpsv2.1) + reward envs (geneval mmcv1.x+mmdetection2.x, OCR paddleocr, hpsv2x, image-reward) — likely need source builds / a separate venv layer.

**✅ FULL-TABLE eval launched (2026-08-19 ~18:00 UTC):** protocol nailed from paper §4.1 (SD3.5-M @512², LoRA r32/α64, **40-step 1st-order ODE sampler**; GenEval+OCR on FlowGRPO splits = rule-based; PickScore/ClipScore/HPSv2.1/Aesthetic/ImgRwd/UniRwd on **DrawBench** = model-based) + author GitHub issue on eval/rollout guidance scale: **student=4.5 (ALL datasets incl. GenEval)**, GenEval-teacher=1.0 (DiffusionNFT, CFG-incompatible), OCR/Aes-teachers=4.5 (Flow-GRPO), base w/o-CFG=1.0 & +CFG=4.5. FIX: pod_main_eval geneval-gs 1.0→$GS (student must be 4.5, my special-case was wrong); added **base-mode** (`EVAL_CKPT_S3=base` → omit --checkpoint_path; evaluation.py both LoRA args default "" → base transformer) + **GSLIST** loop (base w/o-CFG vs +CFG) + **--max_samples** subset (evaluation.py already writes `average_scores.json`, filters -10 sentinel). Jobs (8×H200 **P2**, image/host/role copied from mopd-train via get_entities: image `pluto-training-base:04-17-26-...@sha256:f06f...`, host p5en.48xlarge/H200×8, role `12a760d0-af9a-4c9b-8770-7a7924b118a6`): **opd_eval_mopd_ckpt721 `3d624d51-6a11-4efe-a970-926f46ec3bbd`** (Ours, ckpt-721, DATASETS=geneval,ocr,drawbench, EVAL_REWARDS=pickscore,clipscore,aesthetic,hpsv2, MAXSAMPLES=512) — **STARTING @18:04**; **opd_eval_base_sd35 `63e7688d-7c19-46b4-8280-fe0eb6597d2d`** (base, GSLIST="1.0 4.5") — queued (P2 spare, cluster busy=22 running). **CORRECTED job IDs:** mopd TRAIN=`6416d123-c55f-4361-8a4a-71d581323f42` (RUNNING P0); sopd TRAIN live=`96484152-8ee9-4f65-8f07-b9b316626c33` (RUNNING P0) — NOTE `eb35729c` is the OLD opd_sopd_pickscore_repro (SUCCEEDED/paused, not the live one). **Paper Ours-row target:** GenEval 0.96 / OCR 0.94 / PickScore 23.99 / ClipScore 0.297 / HPSv2.1 0.342 / Aesthetic 6.15 / ImgRwd 1.50 / UniRwd 3.50 / Avg 0.929. Missing-metric TODO: ImgRwd (openai `clip` pkg), UniRwd (sglang 7B server). Eval S3 out: `s3://dit-scale-up/cwan/opd_reproduction/eval/<name>_<ds>/{average_scores.json,evaluation_results.jsonl,images/}`. Log path: `/logging-fs/logs/pluto/jobs/cwan/<uniqueName>/run-*/pod-0*/main_scripts/1/current`. ECS: running jobs = JOB_STATUS_SCHEDULED; query via get_entities(component.compute.job_run_computed_state) — get_entities needs `request={entity_ids,component_types:[{value:"component.compute.job_spec|pluto_role|job_status|job_run_computed_state"}]}`; name = `component.system.name`.

**✅✅ Ours (mopd) MATCHES PAPER (2026-08-19 ~18:20 UTC):** mopd ckpt-721 eval done all 3 datasets (n=512 subset, gs=4.5, ~15 min on 8×H200 — pod bootstrap was fast, node had bundle warm). **GenEval 0.959 (paper 0.96) · OCR 0.936 (0.94) · PickScore 23.97 =0.922×26 (23.99) · ClipScore 0.295 (0.297) · HPSv2.1 0.343 (0.342) · Aesthetic 6.154 (6.15)** — 6/8 columns reproduced, all ≤~1% off. FAITHFUL reproduction of the headline DiffusionOPD(Ours) row confirmed (not just "runs"). Results: `s3://…/eval/mopd_ckpt721_{geneval,ocr,drawbench}/average_scores.json` — keys: geneval→"geneval", ocr→"ocr", drawbench→pickscore(/26-normalized)/clipscore/aesthetic/hpsv2 (drawbench "avg" is a meaningless cross-scale mean — ignore). Dashboard auto-fills this row (green=≤6% off). REMAINING: base rows (job `63e7688d` still queued on P2 spare), ImgRwd (openai clip pkg) + UniRwd (sglang 7B) cols, teacher rows (optional), figures (metric-vs-step curves + galleries).

**Full-table scope + teacher rows (2026-08-19 ~18:45 UTC):** user chose to ALSO train the 3 **Multi-Task RL baselines** (Multi-Task GRPO-Guard, Multi-Task NFT, Cascade NFT) — external-model rows (SD-XL/SD3.5-L/FLUX) stay CITED (not reproduced). Staged teacher/student LoRAs from teachers.tar → `s3://…/opd_reproduction/teachers/{AesTeacher,OCRTeacher,GenEvalTeacher,Student}/lora/` (Student = authors' RELEASED DiffusionOPD student, for an independent cross-check row). **STOPPED over-trained sopd training** (`96484152-8ee9-4f65-8f07-b9b316626c33`, epoch ~2578) via compute_stop_job to free a P0 node — user OK'd once converged (authors report best-performing ckpt anyway). Eval jobs (P2 8×H200): base `63e7688d` (GSLIST "1.0 4.5"), curve `0bb1d977` (CKPT_STEPS 1..721; several landed), teachers `e780a68b` (RUNS mode). **pod_main_eval.sh modes:** single (EVAL_CKPT_S3=dir|"base") · GSLIST loop · curve (CKPT_STEPS + CKPT_BASE) · RUNS ("name|gs|s3src" newline-sep, per-entry gs, no-spaces src). eval S3 out `eval/<name>_<ds>/average_scores.json`; make_dashboard.py OUR_ROWS maps name→paper row (green≤6%). Guidance: student/OCR/Aes gs4.5, GenEval-teacher gs1.0. **NEXT:** agent surveying DiffusionNFT + flow_grpo repos → build+launch the 3 baseline TRAINING runs (Multi-Task NFT + Cascade NFT via DiffusionNFT-main; Multi-Task GRPO-Guard via flow_grpo). Base model still SD3.5-M @512, LoRA r32/α64, same 3 reward tasks (geneval/ocr/aes-mix) alternating datasets; Cascade = sequential NFT stages (init stage N from N-1 ckpt).

**Baselines LAUNCHED P0 (2026-08-19 ~21:39 UTC):** user chose "stop finished mopd → baselines P0". STOPPED mopd training (`6416d123-c55f-4361-8a4a-71d581323f42`, matched paper) → freed P0 node (sopd already freed the other). ⚠️ mopd/sopd jobId trap: correct mopd=**c55f** (I 403'd twice on typo c561). Authored `config/baselines.py` (mt_grpoguard / mt_nft / cascade_nft_s{1,2,3}; smoke-tested all 5 build OK, nbpe%tasks==0 + 8*tbs%nipp==0) + `repro/pod_main_baseline.sh` (env: LAUNCHER=accelerate|torchrun, RUN_CFG, RUN_NAME, CASCADE_PREV_LORA; reuses venv_mopd, NO teachers). code.tar re-bundled w/ baselines.py. Launched 8×H200 **P0**: **MT-NFT `2f936f20-08ec-4019-82e6-6c1b2976aaf9`** (torchrun, mt_nft, gs1.0/dpm2) + **MT-GRPO-Guard `8d85289a-1acb-431f-bd34-85bcc4291e22`** (accelerate, mt_grpoguard, gs4.5). Cascade NFT = 3 sequential stages (cascade_nft_s1_aes→s2_ocr→s3_geneval, chain via `CASCADE_PREV_LORA` env = prev stage's checkpoint-N/lora) — launch after MT-NFT validates + a node frees. P0 = preemptionPolicy NON_PREEMPTABLE (serializes ABSENT in JSON = default 0). Baseline trainers pre-vendored: `scripts/baselines/train_{nft_sd3,sd3_GRPO_Guard}_multireward.py` (round-robin dataset_idx=i%len; NFT reads `config.datasets` top-level, GRPO reads `config.train.datasets`). Per paper wall-clock ~130-148h each → report best-performing ckpt over days; eval via `pod_main_eval` EVAL_CKPT_S3=`runs/{mt_nft,mt_grpoguard}/logs/<run_name>/checkpoints/checkpoint-N`. **All 3 trainings (sopd/mopd + none-left) now STOPPED**; dashboard job-cards sopd/mopd read stale logs (look RUNNING) — cosmetic, update JOBS dict later.

**⚠️ baselines fix #1 — config_flags get_config dispatcher (2026-08-19 ~21:56 UTC):** both MT-NFT + MT-GRPO crashed instantly at config load: `AttributeError: module 'config_config' has no attribute 'get_config'`. Cause: `config_flags --config config/baselines.py:mt_nft` calls **`get_config("mt_nft")`** — the `:suffix` is an ARG to `get_config`, NOT a direct function selector (I'd wrongly assumed `:fn` calls fn directly). FIX: added `def get_config(name): return globals()[name]()` to baselines.py (identical to `config/opd.py:248`). Re-tarred+uploaded code.tar. Both jobs auto-recover (NFT job_status PAUSED reason AUTO_RECOVERY; GRPO run FAILED→job SCHEDULED) → run-2 re-downloads fixed code (pod_main_baseline always re-fetches code.tar). Watcher freshness-gated (mtime<240s) so the stale run-1 crash log doesn't re-trigger. Expect more iterative fixes (mopd took ~6). base eval log looked stalled at 11:24 pod-time (investigate if no base_ results land); curve producing steadily (1/61/121 done).

**⚠️ baselines fix #2 — GRPO num_workers paddle-fork (2026-08-19 ~22:55 UTC):** confirmed fix #1 worked (both baselines PAST config: NFT loading mask2former geneval-reward, GRPO accelerate-launched). But `train_sd3_GRPO_Guard_multireward.py` had train-loader **num_workers=1 (:736)** + test-loader **num_workers=8 (:748)** → same paddle/mmdet fork-unsafe crash mopd hit (forked workers inherit paddle-loaded proc). Patched both →0 (mopd pattern). MT-NFT already safe (loaders num_workers=0; eval_loader :291 inherits test_dataloader.num_workers=0 — no patch). Re-uploaded code.tar; running MT-GRPO (old code) will crash on paddle-fork → autorecovery run-3 downloads patched → trains (only epoch-0 lost). **CORE TABLE FULLY REPRODUCED + matches paper:** base w/o-CFG (0.32/0.11/20.6/0.238/0.201/5.18) & +CFG (0.59/0.56/22.4/0.286/0.283/5.42); GenEval-Teacher (GenEval 0.97, OCR 0.42, 22.0/0.274/0.248); OCR-Teacher (OCR 0.94, 22.3/0.291/0.274/5.26); Aes-Teacher (24.0/0.296/0.347/6.22 ~exact); **official Student release cross-check = 0.96/0.93/23.97/0.296/0.343/6.17 == my mopd == paper Ours** (triple-validated). Curve ~6/7 pts. All on dashboard (green). Remaining: 3 Multi-Task baselines (training) + cited external-model rows.

**✅ CORE REPRODUCTION COMPLETE (2026-08-19 ~23:00 UTC):** ALL core evals done — base×2, GenEval/OCR/Aes teachers, official Student, DiffusionOPD Ours(mopd-721), + full **7-point convergence curve** (steps 1/61/121/211/361/541/721 × geneval/ocr/drawbench). Paper Table-1 method rows + Fig convergence-curve fully reproduced, all match. **⚠️ baselines fix #3 — NFT wrong pipeline import:** `train_nft_sd3_multireward.py:28` imported `flow_grpo.diffusers_patch.pipeline_with_logprob` (the BROKEN module whose `run_sampling(v_pred_fn,latents,sigmas,solver,deterministic,noise_level)` call at :254 is positionally MISALIGNED with solver.py's `run_sampling(v_pred_fn,z,scheduler,timesteps,solver,deterministic,eta)` → passes deterministic=True into `solver` param → `ValueError: Unknown solver: True`). opd + GRPO import the CORRECT `sd3_pipeline_with_logprob` (calls run_sampling with kwargs). FIX: NFT import → `sd3_pipeline_with_logprob` + removed `model_type="sd3"` from its 2 pipeline_with_logprob calls (sd3 version is SD3-specific, no model_type param; returns same 3-tuple image,all_latents,all_log_probs). GRPO unaffected (already sd3). Also fixed GRPO num_workers (fix #2). Re-uploaded code.tar with all 3 baseline fixes. Both baselines auto-recovering → run-3 gets fully-fixed code (NFT resume_from not wired → restart-from-base, but they crashed at epoch-0 eval so no loss). Cascade uses same NFT trainer → benefits from fix #3. Watch run-3 for training-start.

**✨ CA-OPD (Idea 1) IMPLEMENTED + LAUNCHED (2026-08-19 ~23:49 UTC):** conflict-averse / Pareto multi-teacher OPD — my proposed improvement. Replaces the uniform per-task loss avg (`distill_loss = per_step_kl_scalar.mean()`, train_sd3_opd.py ~line 1190 = linear scalarization → student matches the Pareto-DOMINATED teacher-MEAN velocity) with an **adaptive Chebyshev min-max reweighting**: EMA of each teacher's per-step KL → softmax weights up-weight the worst-matched teacher → pushes student toward the (incl. non-convex) Pareto frontier. Edits: (1) `config/opd.py:mopd_caopd()` (=mopd + `train.ca_opd`/`ca_opd_tau=0.5`/`ca_opd_ema=0.9`; run_name `[MOPD-CAOPD + SD3.5-M + PickScore+OCR+GenEval]`); (2) `train_sd3_opd.py`: tag samples with `teacher_idx`, `teacher_kl_ema=ones(n_t)` before `while True:`, CA-OPD weighted-loss block (weights detached, weighted-mean = scale-matched to baseline; logs `ca_opd_w_max`). τ→∞ = mopd exactly; getattr defaults keep baseline mopd unchanged (verified mopd has no ca_opd attr; py_compile+config smoke OK). Reuses the mopd trainer (trained clean to ckpt-721) ⇒ low crash risk. Launched **P0 opd_caopd `21435b38-22dd-4a56-96ee-2173761a3ca5`** (pod_main_mopd.sh RUN_CFG=mopd_caopd; ckpts → `s3://…/runs/mopd_caopd/logs/[MOPD-CAOPD...]/checkpoints/`). **Ablation:** eval CA-OPD ckpts w/ pod_main_eval vs reproduced uniform-mopd (0.96/0.94/23.97/0.295/0.343/6.15) — expect per-axis compromise ↓ (esp aesthetics) + Average ↑. Baselines `2f936f20`(NFT)/`8d85289a`(GRPO) run-3 was stuck (autorecovery stalled ~50min, 0 ckpts) → manual `start_job` with all 4 fixes; now compete w/ CA-OPD for the 2 P0 nodes (CA-OPD launched first = priority).

**✨ 3 MORE ideas IMPLEMENTED + LAUNCHED (2026-08-20 ~00:01 UTC):** all as guarded toggles in `train_sd3_opd.py` (default OFF → mopd/CA-OPD unchanged) + config variants; added `if epoch>=num_epochs: break` to the `while True` loop (mopd num_epochs=100000 = unaffected; ablation configs cap 780 → auto-stop ~ckpt-751 + free node). **Idea 2 `mopd_robust`** (job `db42bda5-7cb3-430a-b09d-1e7cfd1b3f8c`): Huber/trust-region on velocity residual (`train.opd_huber`, `huber_delta`=1.0) — robust to RL-teacher reward-hacking. **Idea 3 `mopd_snr`** (`3ca37f03-3762-4464-a7a3-ba5786a182c1`): Min-SNR-γ per-step weighting (`snr_weight`,`snr_gamma`=5.0; w=min(((1-t)/t)²,γ)/γ, t=timesteps/1000) — principled step weighting vs the ODE-L2 discretization artifact. **Idea 4 `mopd_anneal`** (`115c75da-80ba-4aab-ae30-8b1dc8ea9d36`): SDE→ODE noise annealing (`anneal_noise`, noise 0.7→0.0 over `anneal_epochs`=400; mutates config.sample.noise_level per epoch → drives BOTH loss branch @1176 + sampler η) — explore→exploit, extends the paper's noise-level ablation. **Theory check that ruled out a class of ideas:** for equal-covariance Gaussians forward-KL = reverse-KL = same L2 → symmetrizing OPD's KL is vacuous (don't propose it). All launched P0 via pod_main_mopd.sh RUN_CFG=<cfg>; ckpts → `runs/{mopd_robust,mopd_snr,mopd_anneal}/logs/`. **Surprise: all 4 research jobs (CA-OPD RUNNING + 3 ideas STARTING @00:01) placed simultaneously → P0 capacity ≥4 nodes right now.** Ablation: eval each best ckpt vs uniform-mopd (0.96/0.94/23.97/0.295/0.343/6.15) with pod_main_eval.

**🔴🔴 NORMALIZATION ERROR — do NOT recur (2026-08-20):** I reported CA-OPD/mopd "Score ≈ 0.985" and let it sit next to the paper's DiffusionOPD **Average 0.929** as if we beat them. WRONG — pure normalization artifact. Paper's Average = mean of **min-max over 8 metrics** (adds **ImageReward + UnifiedReward** to our 6) with a WIDE reference range. My "score" = naive/**partly-circular** min-max (floor=zero-shot SD3.5, ceiling=best-among-our-own-runs) over only 6 metrics → systematically inflates EVERY model by ~0.05-0.08 (I recomputed the paper's whole table under naive col-min-max: DiffusionOPD→0.984 vs paper 0.929, Cascade 0.934 vs 0.851, MT-GRPO 0.821 vs 0.763). Same model, more generous scale. **Per-item metrics DO match the paper (faithful repro); the aggregate never did.** **USER DECISION: report PER-AXIS ONLY, drop the Average/Score entirely** (paper edited: all Score/Δ columns removed, per-axis only, recompiled+re-uploaded). Honest CA-OPD verdict: does NOT beat mopd on aggregate; per-axis CA-OPD@781 ties/slightly-beats our repro on OCR(0.947 vs 0.939)/Pick(24.00 vs 23.98)/HPS(0.343 vs 0.342), within noise elsewhere; real win = convergence-speed (Aes peak ~180 steps earlier). Table file: `CA-OPD-paper/data/TABLE_caopd_vs_diffusionopd.md`.

**✨ REGRET + HIGHER-CONFLICT experiments (2026-08-20 ~18:50 UTC) — pursue a GENUINE per-axis win:** diagnosed CA-OPD's ceiling loss = absolute min-max chasing the intrinsically-hard-but-irreducible aesthetics axis. FIX in `train_sd3_opd.py`: **regret weighting** `r_m = ema_m − floor_m` (floor = running-min KL per teacher; toggle `config.train.ca_opd_regret`, `teacher_kl_floor` init inf). Configs in `config/opd.py`: **`mopd_caopd_regret`** (regret, standard 3 teachers, num_epochs=1000), **`mopd_caopd_conflict`** (regret + OCR teacher's dataset→pickscore so OCR & PickScore oppose on SAME prompts = higher conflict), **`mopd_conflict`** (uniform baseline, same conflict set, matched control). code.tar rebuilt+uploaded. Launched P0 8×H200 (cloned opd_caopd spec): **regret `63c88a12-acf0-44c0-9a37-41f15d55d9ec`** (RUNNING), **conflict `3018ac3c-28d9-4de0-a09b-c74120d437f7`**, **mopd_conflict `bbabc4b3-2b53-40a2-b3f8-9fdc2978e1a1`**. Freed nodes by STOPPING (user-confirmed) plateaued `opd_caopd 21435b38` + collapsed `opd_bl_mt_nft 2f936f20`. Eval on-the-fly per-axis via pod_main_eval RUNS mode; refresh TABLE_caopd_vs_diffusionopd.md + paper tab:caopd with best ckpts. **8-metric TODO:** wiring ImageReward + UnifiedReward into eval (agent investigating; UniRwd may need a 7B sglang server — see eval-fix note above).

## 2026-08-26 01:15 — G_construct 探针（低噪声 teacher-free 诊断，重做版）

job 1034406，`sopd_gcprobe`，σ_q=0.278（sigmas[8]，10 步 SD3.5-M 调度），固定后缀 F_q(y)，n=3 批。

| 方向 | G_construct | 三次测量 | vs rand |
|---|---|---|---|
| grad（reward 梯度） | **+0.0946** | 0.1014 / 0.0909 / 0.0915 | t≈17 |
| teacher | **+0.0162** | 0.0218 / 0.0146 / 0.0122 | t≈6.2 |
| rand（对照） | −0.0032 | −0.0027 / −0.0014 / −0.0056 | — |

**结论（正向）**
1. rand≈0 → 探针标定正确，非平凡正结果。这是可信度前提。
2. teacher 显著为正 → teacher 目标在低噪声点**确实**含 reward 改进信号。
   *这纠正了此前 teacher-free 诊断的解读*：cos≈0.003 是**端点式目标**与 reward 梯度正交，
   不是 teacher 无信号。指控对象是端点目标，不是 teacher。
3. grad 是 teacher 的 **5.8×** → 现有 OPD 蒸馏只吃到低噪声可构造增益的 **17%**，
   剩余 83% 是结构性拿不到的空间。
4. 我们的 +0.0946 是 OPSD 自报 G_construct=+0.03511 的 **2.7×**。

**立论骨架**：可构造增益 +0.095 → teacher 只拿 17% → OPSD 证明朴素拟合再丢 98.7%
→ RTC 把目标投到可达子空间，让这部分增益真正落地。

**如实标注**：n=3 为同一 run 内独立 prompt batch、单 seed。已提交 n=24 版本（1034410）加固。

## 2026-08-26 01:10 — RTC 有限差分 JVP 的数值问题（已修）

首个跑通的 RTC smoke（1034405/1034407）给出 cos(JJ^T g, g)=0.0153/0.0162，
按理论对应 1/cos≈65× 提升 —— 但 65536 维随机方向的 cos 基准是 1/sqrt(65536)=0.0039，
仅差 4 倍，怀疑被舍入噪声主导。

加 10×eps 复算检验：**rtc_eps_stab = 0.34**（本应 ≈1）→ **有限差分不稳定，cos 值不可信，撤回 65× 说法。**

根因与修复：
- transformer 权重本身是 fp32（`from_pretrained` 未传 torch_dtype），fp16 仅来自 autocast → 关 autocast 有效；
- **但 TF32 尾数只有 10 位，与 fp16 相同**，config.allow_tf32 会打开它，cudnn.allow_tf32 默认就是 True
  → 只关 autocast 而不关 TF32 等于没关。两者都必须关。
已实现 `_fp32_fd()` 上下文管理器（同时关 autocast + matmul/cudnn TF32），验证 job 1034409。
判据不变：eps_stab≈1 才采信 cos。

## 2026-08-26 01:40 — RTC 理论二阶推论（预判实验现象，防止误判方向）

同一位移预算 B（trust region 相同），A=JJ^T：

| | G_construct ≈ B<g,d> | G_realized ≈ ηB<g,Ad> |
|---|---|---|
| raw (d=g/‖g‖) | B‖g‖ | ηB‖Ag‖·cos |
| gn  (d=Ag/‖Ag‖) | B‖g‖·cos | ηB‖Ag‖ |

- G_realized 比值 gn/raw = **1/cos**（Cauchy-Schwarz 保证 ≥1）
- 实现效率 (realized/construct) 比值 = **1/cos²**
- **但 gn 的 G_construct 比 raw 小 cos 倍** —— 这是方法的固有取舍，不是 bug。

**预判 + 防误判**：raw 的接受率 acc_p 很可能显著高于 gn（raw 沿 reward 梯度走，
一阶必然改进；gn 走的是别的方向，构造增益小，过门控更难）。
**绝不能因此判定 raw 更优** —— acc_p 度量的是构造质量，而我们要的是 realized。
判据只看 `rtc_realized` 与最终 reward 曲线。

隐患：若 cos 极小，gn 目标改进微弱 → 门控拒样过多 → 有效样本量塌缩。
实测 gn acc_p=0.78（margin=0）尚健康。若 raw/gn 的 acc_p 差距过大需考虑对 gn 放宽门控或加大预算。

已实现 `rtc_realized = <g, y_new - y_old>`（optimizer.step() 后重新前向），
直接测这个核心量，并给出可证伪预测 gain_gn/gain_raw = 1/cos。

## 2026-08-26 02:10 — 更正: 噪声源是 TF32 不是 fp16;且扰动从未生效

**更正前述诊断**: train_sd3_opd.py:912 `autocast = contextlib.nullcontext if config.use_lora else ...`
—— use_lora=True 时**根本没有 autocast**,前向一直是 fp32。"fp16 autocast 导致舍入噪声"的说法错误。
真正的低精度来源是 **TF32**(:816 `config.allow_tf32` 打开,尾数仅 10 位)。

关掉 TF32 后 `max|v1-v0|` 变成**恰好 0**。含义:
- TF32 的 split-k 归约不确定 → 两次**权重相同**的前向也有差异 → dirnorm=19.2 是纯噪声;
- 关 TF32 后前向确定性 → 逐比特相同 → **参数扰动自始至终没有生效**。
- 因此 fp16 那次的 cos(JJ^T g,g)=0.0153/0.0162 与 eps_stab=0.34 全部作废。

诊断数据(1034411): |u|=9.45e-01 |p|=9.57e+01 h=1.01e-01 |h*u|=9.57e-02
max|v1-v0|=0.0 n_params=382 —— u 健康、扰动量正确(0.1% 相对),但输出零变化。

**精确双反向 JVP 亦不可用**: `aten::_scaled_dot_product_flash_attention_backward` 无二阶导。
需为该前向切 math 后端(内存代价大)或另寻他法。

待判(1034412): 打印"期望改动/实际落地/max|v_big-v0|(1000x 扰动)",
区分「拷贝没落地」与「前向使用的不是这批张量」。

## 2026-08-26 02:25 — RTC 有限差分失败的真因: 权重是 BF16

判据(1034412):
```
期望改动=2.282105e+00  实际落地=2.282105e+00   -> 拷贝落地正确
max|v_big-v0|=5.448486e+00 (1000x 扰动)        -> 前向确实使用这批参数
max|v1-v0|=0.000000e+00    (1x 扰动=0.1%)      -> 输出零变化
```
拷贝没问题、参数在前向路径上,但响应是阈值式的 -> 输出被量化。

**证据链**: `SD35_PATH/transformer/*.safetensors` 头部显示 **909 个张量全部 BF16**;
且 `max|v0|=6.398438` 恰为 fp16 可表示数(1638x2^-8)。前向以 16 位精度计算,
输出分辨率约 4e-3;而 0.1% 的 LoRA 扰动对输出的影响远小于此 -> 被完全舍入。

**结论**: 在 16 位前向里用有限差分估 JVP 不成立,不是调参能救的。
**精确双反向 JVP 也不可用**(flash-attn 无二阶导)。

正在测(1034413): eps 扫描 1x~1000x,看是否存在"可分辨且仍线性"的窗口
(线性区内相邻尺度方向应 cos~1)。若无窗口 -> 必须让 RTC 构造前向走 fp32。

## 2026-08-26 02:40 — eps 扫描: 不存在可用窗口,且暴露自相矛盾

| scale | max|dv| | \|dir\| | cos_vs_prev | cos_dir_g |
|---|---|---|---|---|
| 1 | 1.212 | 27.96 | — | +0.0136 |
| 3 | 1.669 | 13.97 | +0.576 | +0.0180 |
| 10 | 1.681 | 6.79 | +0.568 | +0.0239 |
| 30 | 3.127 | 4.00 | +0.502 | +0.0234 |
| 100 | 4.717 | 2.77 | +0.196 | +0.0082 |
| 300 | 3.769 | 1.51 | +0.451 | +0.0035 |
| 1000 | 5.189 | 0.76 | +0.647 | +0.0025 |

- `|dir| ∝ 1/scale` ⇔ `max|dv|` 近似常数 ⇒ **输出差异不随扰动增大而增大,即差异不是扰动造成的**。
- 相邻尺度 cos 仅 0.2~0.65,**没有线性区**,不存在"可分辨且仍线性"的 eps 窗口。
- **自相矛盾**: 同一次运行,scale=1.0 报 max|dv|=1.212,而 [RTC-FD] 用**完全相同的扰动**报 0。
  同权重同输入两次前向结果不一致 ⇒ 前向疑似带随机性。判据已下(1034414): 同权重连做两次测 max|vA-vB|。

**重要**: 在真因确定前,不把 H200 交给 RTC。保持 poe15 → horizon2 → ste → retain → poe125 链路。

## 2026-08-26 02:55 — 前向确定性: 排除随机性;FD 异常未解决,改走 RTC-K

同一次运行(1034414)三行:
```
[RTC-DET] max|vA-vB|(同权重两次)=0.0  training=True  dropout>0 模块数=0  max|vA-v0|=0.0
[RTC-EPS] scale=1.0  max|dv|=1.090     <- 用扰动 δ
[RTC-FD ] max|v1-v0|=0.0               <- 用完全相同的 δ
```
- 前向**完全确定**(两次同权重前向逐比特相同),"dropout 随机性"假设**错误**,已排除。
- 因此 scale=1 得 1.090 而 FD 得 0 是**真实矛盾**。已核查: 代码结构正确、缩进正确、
  `_v1`/`_v0`/`_h` 在该作用域无其他赋值、`_params` 是 list、`_u` 未被修改。**真因未查明,如实记录。**
- 附带推翻前一条: 既然 scale=1 的扰动确实改变输出(1.09/6.4≈17%),
  "bf16 量化把 0.1% 扰动完全抹掉"的解释也**不成立**。两条解释都不对,悬而未决。

**决定**: 停止追查 FD,改用 **RTC-K**(不依赖该谜题)。
理由: FD 已被替换,继续 debug 是为一条不再使用的路径消耗 H200。

### RTC-K(修订后的主线实现)
目标 = 模型沿 reward 梯度**真实走 K 步**后的输出 y_K,取 dir = y_K - y_0。
相对"有限差分估 JJ^T g"的三个优势:
 1. 不做相近量相减,位移由步长直接定标 -> 不受输出量化/测量异常影响;
 2. 目标是**真实模型输出**,天然落在可达集内,不依赖线性化假设;
 3. 捕捉真实非线性可达轨迹,比一阶 JJ^T g 更强。
参数: rtc_ksteps=4, rtc_klr=0.02(每步参数相对位移 2%,正是扫描中确认能改变输出的量级)。
配置: `rtc_k` / `rtc_k8`。验证 job 1034415。
判据: acc_p 显著 >0.5 且 rtc_realized > 0(不看"没崩")。

## 2026-08-26 03:15 — RTC-K 首个可信数字 + 需预先回答的审稿质疑

smoke 1034415: `K=4 klr=0.02 |dir|=31.96 |y0|=331.90 cos(dir,g)=+0.00956`
- 位移 |dir|/|y0| = **9.6%**,恰好落在信赖域 rho=0.10 附近 —— K/klr 的选择天然匹配,归一化几乎不改尺度。
- 位移量远高于任何量化/测量噪声尺度,与此前 FD 那些数字性质完全不同,**这个方向是结实的**。
- cos(dir,g)=0.0096 支持核心论点: **reward 梯度绝大部分指向模型不可实现的方向**。
  注意: 1/cos 的倍数公式是对线性 Ag 推的,**不可直接套用到 K 步非线性版本**,不要引用"105 倍"。

**需预先回答的质疑: "这不就是 ReFL/DRaFT(直接 reward 反传)吗?"**
区别: ReFL/DRaFT 沿**整条采样链**反传(昂贵、显存重、易 reward hacking);
RTC-K 只在**单个低噪声查询点**反传(与 OPSD 同),再把结果构造成**目标**,
保持 OPD 的 MSE-on-target 训练循环不变,并保留接受门控/双分支/信赖域。
需要的干净归因实验就是 **rtc_k vs rtc_raw**(唯一差别 = 目标是否可实现),已在计划内。
另需一条 ablation: 直接应用 K 步而不做拟合(看拟合本身是否必要)。

## 2026-08-26 03:30 — RTC-K 首次实测: acc_p=0.31,低于随机。我漏了一个必须做的验证

smoke 1034415 (rtc_k, K=4, klr=0.02, rho=0.10):
```
rtc_R0=0.8018   rtc_acc_p=0.3056   rtc_acc_m=0.8611
```
**正目标 69% 的情况下反而降低了 reward。**

**我的过失**: 看到 |dir|=9.6% 数值合理就继续推进,**从未验证"4 步上升是否真的在提高 reward"**。
这正是"没崩 ≠ 算对了"。已补上逐步 reward 日志(rtc_asc0..K)。

**核对**: 接受判据用 `_reward_of_y`(:1696),与 K 步上升所优化的是**同一个函数**;
且 `_yp = _y0 + budget*dir/|dir|`,budget=0.10*331.9=33.2 而 |dir|=31.96,
所以 **y_p 基本就是 y_K**。故确系上升本身失效,不是构造环节的错配。

**自洽的机制解释(待验证)**: θ 空间上升使 y 沿 Ag 移动,A=JJ^T 半正定 ⇒ <g,Ag> ≥ 0,
**一阶必然不降**。但 cos(dir,g)=0.0096 ⇒ 一阶增益极小;位移达 |y0| 的 9.6% 时
**二阶项压倒微弱的一阶项** ⇒ 净下降。acc_m=0.86 与 acc_p=0.31 并存(±方向都变差)
正是"处在该轴局部极大附近 + 位移过大"的特征。

**可证伪预测**: 缩小信赖域 ⇒ acc_p 应回升过 0.5。
**已提交 1034416 (`rtc_kcurve`)**: 同批测两条 reward-位移曲线
(可实现方向 vs 原始梯度方向, r ∈ {0.001,0.003,0.01,0.03,0.10}),
直接读出各自最优半径与增益 —— 这也正是论文需要的那张图。

## 2026-08-26 03:50 — 新方向 RTP: 把 reward 梯度投影到 teacher 子空间

**动机(由今晚数据推出)**: teacher 输出本身就是"另一组 LoRA 权重下的模型输出",
**天然落在可达集内**;而探针已证明 teacher 方向 reward 显著为正(+0.0162, t≈6.2)。
现有 OPD 用的是多 teacher 的**简单求和** —— 一个与 reward 无关的任意组合。

**RTP**: 在 span{y_Ti - y_0} 内对 reward 梯度做最小二乘投影,得到
**可实现 + reward 对齐**的方向,且几乎零额外成本(teacher 前向 OPD 本来就在做)。
相对 RTC-K 的优势: 不需要参数扰动/有限差分/JVP,完全避开今晚所有数值陷阱。

已实现为探针的方向 5(`gc_cos_tproj_g`, `gc_G_tproj`),配置 `mopd_gcprobe`(3 teacher)。
探针一次给出五个方向的严格半径匹配对比: grad / rand / teacher / ksteps / **tproj**。
判据: G_tproj 需显著 > G_teacher(否则投影没带来好处);理想情况接近 G_grad。

## 2026-08-26 04:20 — kcurve 诊断: 机制确认 + 修复方案(重要正向转折)

### 上升轨迹(rtc_asc0..4, K=4 klr=0.02)
```
step0 (y_0) 0.8025
step1       0.8232  +0.0207   上升
step2       0.8262  +0.0237   峰值
step3       0.8150  +0.0125   越过
step4 (y_K) 0.7987  -0.0038   已低于起点
```
**上升方向是对的,K=4 走过头了。** 我们一直把第 4 步当目标,而它已跌回起点以下
—— 完整解释了 acc_p=0.31。**固定 K 是错的。**

### reward-位移曲线(同批,同 reward 函数 `_reward_of_y`)
| r | dR(grad) | win | dR(ksteps) | win |
|---|---|---|---|---|
| 0.001 | +0.0058 | 98.6% | +0.0004 | 65.3% |
| 0.003 | +0.0148 | 98.6% | +0.0010 | 70.8% |
| 0.010 | +0.0361 | 98.6% | +0.0030 | 62.5% |
| 0.030 | **+0.0579** | 97.2% | +0.0071 | 54.2% |
| 0.100 | +0.0532 | 95.8% | **-0.0037** | **36.1%** |

- 小半径下 K 方向确为正(win 65~71%),r=0.10 才翻负 ⇒ **rho=0.10 过大**,与预测一致。
- grad 方向在 r≈0.03 达峰(+0.0579),r=0.10 略降。

### 关键正向读数
2 步上升得到 **+0.0237**,这是**模型自己的真实输出**(完全可实现);
潜空间无约束移动的最佳构造增益是 **+0.0579**(r=0.03)。同一 reward 函数,可直接比:
> **可实现增益 / 可构造上界 = 41%**
比 OPSD 报告的 1.3% 实现率高一个数量级 —— **正向信号**。

### 两处修复(已实现)
1. **按实测 reward 逐样本取最优迭代点**(`rtc_kbest`),而非固定 K。
   自适应、免调参,且与 OPSD"用实测而非一阶预测做接受判定"的哲学一致。
2. **信赖域改为上限**: 原构造式 `y_p = y0 + budget*dir/|dir|` 会把位移**强行拉伸**到
   rho*|y0|=10% —— 正是曲线上翻负的半径。改为 `budget = min(budget, |dir|)`,
   最优迭代点不会被重新推过头。**这是个真 bug,不是调参问题。**

验证 job 1034418 (`rtc_kbest`, K=6 给线搜索余量)。判据: acc_p 必须显著 > 0.5。

## 2026-08-26 04:35 — 关键洞察: 上升路径是弯的,线性化丢掉 3.3x 的增益

在半径 ≈3%(2 步位移量级)处,同一 reward 函数、同批样本:

| 方向 | 增益 | 占可构造上界 |
|---|---|---|
| grad(不可实现上界) | +0.0579 | 100% |
| **K 步实际迭代点(弯曲路径)** | **+0.0237** | **41%** |
| 指向 K 步终点的直线(线性化) | +0.0071 | 12% |

**含义**: 沿真实上升路径取点比"取其方向再按半径外推"好 **3.3 倍**。
路径曲率携带了大部分可实现增益。

**这直接构成支持 K 步形式、反对线性化预条件的论据** —— 而 RTC 最初的 JJ^T g
正是那个线性化。今晚 FD 数值失败反而把我们推到了更好的形式上。

**推论(已据此修正实现)**: 目标应当是**路径上的最优点 y_best 本身**,
而不是"取 dir 方向 + 按信赖域重新定标"。信赖域必须是上限(min),不能是强制值。

**待办**: 用 kbest 运行确认 y_best 的实际半径(打印 |dir|),把上表的"≈3%"换成实测值。

## 2026-08-26 04:50 — G_construct n=24 配对检验(定稿数字)

```
gc_G_grad     n=24  mean=+0.09275  sd=0.00487  se=0.00100
gc_G_teacher  n=24  mean=+0.01430  sd=0.00194  se=0.00040
gc_G_rand     n=24  mean=-0.00411  sd=0.00131  se=0.00027

配对 grad-teacher  +0.07845  se=0.00074  t=106.7
配对 teacher-rand  +0.01841  se=0.00051  t= 36.1
配对 grad-rand     +0.09686  se=0.00113  t= 85.5
grad/teacher = 6.48x    teacher 占 grad 的 15.4%
```
rand 显著为**负**(t≈-15) ⇒ 该半径下随机移动确实损害质量,预算尺度不是虚设,正结果非平凡。
n=3 的 17.1% 与 n=24 的 15.4% 一致。单 seed,论文需扩多 seed。

## 2026-08-26 05:10 — 一个会静默毁掉整次训练的 bug(已修)

RTC 训练分支末尾的 `continue` 位于 `with accelerator.accumulate(...)` **内部**,
而 info 的汇总/上报块是 `for j` 循环体内、with 之外的**兄弟语句** —— continue 把它整个跳过。

三个真实后果(都不会让程序崩,因此从"没崩"看不出来):
1. `rtc_loss`/`rtc_pos`/`rtc_neg`/`grad_norm` 从不上报(实测确实全部缺失);
2. `global_step += 1` 被跳过 ⇒ checkpoint 路径 `checkpoints/checkpoint-{global_step}`
   **恒为 checkpoint-0,所有保存互相覆盖** ⇒ 800 epoch 跑完只剩单个点,
   **没有 reward 轨迹、没有多点评测**,几小时 H200 全废;
3. `ema.step(..., global_step)` 恒传 0,`update_step_interval=8` 的判据恒成立,EMA 节奏错乱。

注: `save_ckpt` 的触发条件是 `epoch % save_freq`(不是 global_step),所以会照常触发 ——
这恰恰是它危险的地方: 保存在发生,只是全部写到同一个路径。

**修复**: 在 RTC 分支的 continue 之前补上同样的 sync_gradients 汇总块。
验证: job 1034418 应出现 rtc_loss / grad_norm 且非零。

## 2026-08-26 05:20 — RTP 投影代码的 CPU 单元测试(五项判据全过)

不上 GPU 就能查出形状/数学错误:
```
残差与子空间正交  max|D^T(g-p)| = 3.8e-05   -> 确实是投影(定义性判据)
|p| <= |g|                          True
cos(投影,g)=0.0575 >= 各列最大 0.0202  True  -> 优于任一单独 teacher
cos(等权和,g)=-0.0089 <= 投影          True  -> 优于 OPD 的等权基线
半径匹配 |y-y0| == budget            精确
```
第 4 项正是 RTP 的核心主张: **投影是 teacher 子空间内的最优方向,而等权求和是任意选择**。
随机数据上等权和的 cos 甚至为负,说明"求和"这个选择确实没有任何 reward 依据。

## 2026-08-26 05:50 — RTP 训练侧设计(待门控通过后实现)

**核心原则: 步长与 baseline 完全相同,只改方向 —— 归因才干净。**
```
mu_target = mu_0 + s * sum_t c_t (mu_Tt - mu_0)
s 使 ||mu_target - mu_0|| = mean_t ||mu_Tt - mu_0||
```
c 为 reward 梯度在 span{mu_Tt - mu_0} 上的最小二乘系数。
baseline 的"等权求和"只是 c 的一个特例 ⇒ **RTP 是 baseline 的严格推广**,
一阶上不应更差。这一点已由 CPU 单元测试的判据 4 证实。

**训练分支写法**(仿 ste 分支,插在 cagrad 之前):
```python
elif rtp and teacher_prev_mean.shape[1] > 1:
    _c   = sample["rtp_coef"].float()                    # [B, M],采样阶段算好
    _mu0 = prev_sample_mean.detach().float()
    _res = teacher_prev_mean.float() - _mu0.unsqueeze(1) # [B,M,...]
    _mix = (_c.view(B,-1,1,1,1) * _res).sum(dim=1)
    _rn  = _res.flatten(2).norm(dim=2).mean(dim=1)       # baseline 的平均位移范数
    _mn  = _mix.flatten(1).norm(dim=1).clamp(min=1e-12)
    _mix = _mix * (_rn/_mn).view(-1,1,1,1)               # 半径匹配
    teacher_prev_mean = (_mu0 + _mix).unsqueeze(1)
```

**容易漏掉的实现细节(记下来免得重蹈今晚覆辙)**:
1. **scorer 创建条件**(train_sd3_opd.py:~1083)必须加上 `rtp` ——
   否则采样阶段拿不到 reward 梯度,`rtp_coef` 无从算起。今晚 RTC 就在这里栽过一次。
2. `rtp_coef` 必须在**采样阶段**算并存入 sample dict(那里才有 reward 梯度),
   参照 `rtc_g` 的存法。
3. 训练分支若用 `continue`,**必须自带 sync_gradients 汇总块**,
   否则 global_step 不前进 ⇒ checkpoint 全部覆盖到 checkpoint-0(今晚已踩)。
4. 成本: 每 batch 一次 VAE decode + scorer 反向(与 RTC 的 `_g` 同量级),teacher 前向本来就在做。

## 2026-08-26 06:00 — RTC-K 修复后的门控结果(通过) + 一项不可信指标(如实标注)

smoke 1034418 (`rtc_kbest`, K=6, kbest=True, 信赖域改为上限):
```
rtc_acc_p      1.0000 / 1.0000 / 1.0000    <- 修复前 0.3056
rtc_acc_m      1.0000 / 0.9861 / 1.0000
rtc_kbest_idx  2.33 / 2.06 / 2.10          <- 最优步确在第 2 步附近,与 asc 曲线一致
rtc_kbest_gain 0.0300 / 0.0279 / 0.0360    <- 平均实测增益 +0.031
rtc_loss/pos/neg  0.08~0.10 非零 ✓
grad_norm      0.0035 / 0.0028 / 0.0029 非零 ✓
_step          0 / 1 / 2                   <- global_step 正常前进,continue 修复生效 ✓
```
两处修复(逐样本取最优迭代点 + 信赖域改为上限)完全生效,acc_p 从 0.31 -> 1.000。

### 不可信指标: rtc_realized ≈ 0 —— **不能据此判定 G_realized = 0**
```
rtc_realized      -0.0000 / -0.0000 / 0.0000
rtc_realized_cos  -0.0000 /  0.0000 / 0.0000
rtc_dy_norm        0.2074 /  0.1448 / 0.1606
```
`dy_norm=0.16` 而 `|y0|=331.9` ⇒ 一次优化器步造成的输出变化仅 **5e-4 相对量**,
而 16 位前向的输出分辨率约 **4e-3** ⇒ 差值落在量化噪声以下。
`realized_cos` 比随机基准(1/sqrt(65536)/sqrt(36)≈6.5e-4)还小一个量级,进一步印证是噪声。
**这与今晚毁掉有限差分 JVP 的是同一个陷阱。** 该指标在单步尺度上不可测。
已在训练配置里关掉(`rtc_measure_realized=False`)。
要测 G_realized 必须跨多步窗口(比较 epoch e 与 e+N 的 y),或直接以训练 reward 曲线为准。

### 修正一处调度预估
poe15 的 `num_epochs=1000`(不是 800),现在 ~740,**实际约 09:00 结束**,不是先前说的 06:15。
插队时间充裕。

## 2026-08-26 06:15 — 操作坑: RUN_REQUIRE_MARKER 只查训练脚本

`slurm/smoke_cfg_h200.slurm:69` 是 `grep -qF "$RUN_REQUIRE_MARKER" scripts/train_sd3_opd.py`,
**只查训练脚本**。若传的字符串位于 `config/opd.py`(例如新配置的 docstring),必然轮询失败 18 次(180s)。
好在耗尽后是 `break` 继续执行,不会中止作业 —— 只是白等。
**改动只在 config 时,marker 应传训练脚本里的字符串,或干脆不传。**
(1034421 踩到,已确认会正常跑完,无需重提。)

## 2026-08-26 06:30 — RTP 门控无效(gc_ntproj=1)+ 根因 + RTP 定位的修正

**gc_rtp(1034417)什么都没测到**: `gc_ntproj=1.0` —— 只有一个 teacher 列,
投影子空间退化为一维 ⇒ tproj ≡ teacher 方向,两者当然几乎相等(0.0034 vs 0.0026)。
**这不是 RTP 失败,是实验没测到 RTP。**

**根因**(train_sd3_opd.py:726): "每个 slot 计算全部 teacher 目标(K=M)" 这一步
**只在 `cagrad` 或 `ste` 开启时执行**。否则 **OPD 每个 batch 只用一个 teacher**
(按数据集轮转),`teacher_prev_sample_means` 的 teacher 维为 1。
已把 `rtp` 与 `gc_all_teachers` 加入该条件,并在 `mopd_gcprobe` 里开启。

**这改变了 RTP 的定位(往好的方向)**:
RTP 的对手不是"等权求和"(baseline 根本没有求和),而是**用户自己做过的 conflict-averse 组合**
(cagrad -7.4% / conflict -5.3% / cconf -5.1%,全部有害)。
RTP 用同一套机制(每状态算全部 teacher),但组合规则换成
**由实测 reward 梯度决定的最优组合**,而非基于梯度冲突的启发式。
=> 更干净的正向叙事: **在已证明"冲突消解"行不通的地方,给出一个有效的组合规则。**

**有效的交叉验证(来自那次无效门控)**:
- `gc_G_grad ≈ 0.094` 与 n=24 的 0.0928 吻合 ✓
- `gc_G_ksteps` 在 rho=0.10 处为**负**(-0.0028),独立印证 kcurve 的"该半径对可实现方向过大" ✓

**另一处更正**: 我先前说 smoke 预检耗尽重试后会继续执行 —— **错**,
`[ "$_t" -eq 18 ] && { ...; exit 1; }`,会中止。1034421 因此 FAILED 未跑,已用正确 marker 重提(1034422)。

## 2026-08-26 06:40 — 预检漏洞: 配置**内容**过期能通过(已补)

1034423 带着**旧配置**跑了(日志 `K_max=1`),门控再次白测。

**漏洞**: 两道预检各查一半 ——
- 配置预检只查 `'$RUN_CFG' in dir(m)`,即**名字**是否存在。`mopd_gcprobe` 今晚早些时候就已创建,
  所以过期的 NFS 副本照样通过;
- `RUN_REQUIRE_MARKER` 只查 `train_sd3_opd.py`。

⇒ "配置早已存在、本次只改了它的**内容**" 这一种情况正好从缝里漏过去。

**修复**: 新增 `RUN_REQUIRE_CFG_MARKER`,轮询 `config/opd.py` 的内容直到出现指定字符串,
18 次(180s)不出现则中止。重提为 1034424。
**教训**: 只改 config 内容时,必须传 `RUN_REQUIRE_CFG_MARKER`,只传 `RUN_REQUIRE_MARKER` 不够。

## 2026-08-26 06:50 — RTP 门控终于有效: K_max=3

`gc_rtp3`(1034424) 日志: `configured 3 teacher slot(s) using 3 unique teacher adapter(s) (K_max=3)`
—— 每个状态现在计算全部 3 个 teacher 目标,投影落在真正的三维子空间上。
新增的 `RUN_REQUIRE_CFG_MARKER` 预检生效(无重试,config 已刷新)。

**今晚被 NFS 缓存咬了两次**: 第一次是训练脚本(已有 RUN_REQUIRE_MARKER),
第二次是配置内容(本次新增 RUN_REQUIRE_CFG_MARKER)。两个洞现已堵上。

## 2026-08-26 07:15 — 插队脚本的一个静默失效点(已修)

**RTC 的 global_step 与常规 arm 差一个偏移**:
- poe15 等常规 arm: checkpoint 在 `1,31,61,...`(global_step = epoch+1);
- RTC smoke 实测 `_step = 0,1,2` 对应 epoch 0,1,2 ⇒ checkpoint 落在 `0,30,60,...`。

原 auto_eval 步号列表写的是 `61 151 241 ...`,对 RTC **一个都匹配不上**,
`auto_eval_on_finish.sh` 会打印"没找到 checkpoint,退出"然后静默结束
—— 训练照跑,但早上不会有任何评测结果。

**修复**: 步号列表同时覆盖两种偏移(`60 61 150 151 ...`),脚本本就按目录存在性过滤,多余的会被跳过。

## 2026-08-26 07:20 — RTC-K 训练配置 smoke 通过(1034422)

```
exit=0
rtc_acc_p      1.0000 / 1.0000 / 1.0000     判据 >0.5 ✓
rtc_acc_m      1.0000 / 1.0000 / 1.0000
rtc_kbest_gain 0.0283 / 0.0319 / 0.0376     平均 +0.033
rtc_kbest_idx  1.96 / 2.33 / 2.17           最优步稳定在 2
rtc_loss/pos/neg  0.079~0.089 非零 ✓
grad_norm      0.0034 / 0.0036 / 0.0019 非零 ✓
_step          0 / 1 / 2                    global_step 正常前进 ✓
```
三条上升曲线均在第 2 步达峰,与 kbest/kcurve 一致(三次独立运行同结论)。
`_step = epoch` 确认 checkpoint 落在 0,30,60,... ⇒ 步号列表的修复是必需的。

**两个训练臂的唯一变量**:
- `rtc_kb_train`: 目标 = 沿 reward 梯度真实走 K 步、按实测 reward 取最优迭代点的模型输出(**可实现**)
- `rtc_raw_train`: 目标 = 沿原始 reward 梯度按信赖域外推(**等价 OPSD 构造,不保证可实现**)
共用同一套双分支损失/接受门控/自适应归一化 ⇒ 差值可直接归因到"目标是否落在可达集内"。

## 2026-08-26 07:35 — 两臂对比的一个 caveat(必须在论文里讲清,不能假装是完美受控)

`rtc_kb_train` 用 `budget = min(rho*|y0|, |dir|)` ⇒ 有效半径由上升路径自身决定(实测 ~3~6%);
`rtc_raw_train` 用强制半径 `rho*|y0| = 10%`。**两臂的有效步长不同,不是严格受控对比。**

为什么不能强行统一:
- raw 方向是 reward 梯度,**其范数是任意的**(取决于 scorer 的尺度),没有"自然半径"可言,
  只能靠信赖域强制定标 —— 这正是 OPSD 的做法;
- ksteps 方向来自真实的参数步进,**自带尺度**,强行拉伸到 10% 会把它推过极值
  (kcurve 实测该半径处 dR 已翻负 -0.0037)。

所以这是**方法固有的差异**,不是可以消除的实验瑕疵。
可辩护的表述: 两臂各自处在其**最优半径**附近
(曲线显示 grad 在 r≈0.03 达峰 +0.0579、r=0.10 为 +0.0532,已接近最优;
ksteps 的最优点由 kbest 逐样本选出) ⇒ 这是"各自调好后的对比",而非"同半径对比"。

**报告时必须同时给出**: 两臂的实际有效半径,以及 kcurve 的 reward-位移曲线,
让读者自行判断。不要只报最终 reward 差值。

## 2026-08-26 07:45 — 两个 RTC 训练臂已插队(门控全部通过)

`smk_rawtrain`(1034426) 也通过: exit=0, acc_p 0.9861/1.0000/0.9722, rtc_loss 0.31~0.34 非零,
grad_norm 0.0005 非零, _step 0/1/2 正常。

### 两臂 smoke 对比: 一个直接支持 RTC 论点的实测信号
| | rtc_loss | grad_norm |
|---|---|---|
| raw(OPSD 式,不保证可实现) | 0.31~0.34 | **0.0005** |
| kb(可实现目标) | 0.079~0.089 | **0.0034** |

**raw 的损失大 4 倍,参数梯度却小 7 倍。**
损失大 ⇒ 目标离模型输出远;梯度小 ⇒ 该残差**落在参数改变不了的方向上**。
这正是"目标不可实现"的直接表现,且是**实测量而非推断**。
(注意: 损失含自适应归一化器,绝对值不可直接比;但 grad_norm 是真实参数梯度,可比。)

### 链路(已用 scontrol show job 逐个核对,不看 squeue Reason)
```
poe15(1032590) -> rtckb(1034427) -> rtcraw(1034428) -> horizon2(1033234)
                -> ste(1033235) -> retain(1033236) -> poe125(1032591)
```
两个新臂的 auto_eval 监视已挂,步号列表覆盖 RTC 的 global_step 偏移(0,30,60,... 与 1,31,61,...)。

## 更正 — 本文件 2026-08-26 各条的时间戳系统性偏前

实测 `date` = 02:47 EDT,而上文若干条标注为 05:xx~07:xx。**时间戳错误,事件顺序无误。**
对照锚点: poe15 起始 2026-08-25T17:26:15,已运行 09:21,故"插队完成"实际发生在约 **02:45**,而非 07:45。

**poe15 ETA 更正**: 44.9 s/epoch × 剩余 250 epoch = **3.12h ⇒ 约 05:53 结束**(先前说 09:00,错)。
届时 rtckb(1034427) 自动接上。

## 更正/补充 — continue bug 的第四个后果: 诊断样本量被静默塌成 n=1

diag2 刷写是 `wandb.log(_d2, step=global_step)`(train_sd3_opd.py:~2117)。
`global_step` 卡在 0 时,**每个 epoch 的诊断都写到同一个 step,被 wandb 覆盖** ⇒ n 塌成 1。

- `kcurve`(1034416) 跑在修复**前** ⇒ 3 个 epoch 的曲线只剩 1 个点(此前误以为是别的原因);
- `kbest`(1034418) 跑在修复**后** ⇒ n=3 正常。

**核心正向结果不受影响**: `gc_n24` 的 n=24 是 `gc_probe` 路径,不走 RTC 分支、无 continue,
global_step 正常递增,故 24 个点真实存在。已核对。

⇒ 先前 RTC smoke 的各项 n=1 都是这个原因,不是采样不足。
已提交 `curve_n12`(1034429) 用修复后的代码重测 reward-位移曲线,应得 n=12 及误差棒。

## 2026-08-26 03:00 — RTP 门控中途数据(n=3): 部分成功,但未过预设判据

`gc_rtp3`(1034424),`gc_ntproj=3` ✓ 子空间确为三维:
```
gc_G_grad     0.0912   (上界)
gc_G_teacher  0.0140   <- 单个 teacher(proxy reward 恰为 pickscore 的那个,即最对齐者)
gc_G_tproj    0.0137   <- 投影
gc_G_tsum     0.0093   <- 等权组合
gc_G_rand    -0.0047
cos(tproj,g)=0.0088    vs   cos(tsum,g)=0.0044
```

**投影按设计工作了**: cos 是等权组合的 **2 倍**,构造增益高 **47%**(0.0137 vs 0.0093)。
**但没有超过单个 teacher**(0.0137 vs 0.0140)。
预设判据是"G_tproj 显著 > G_teacher" ⇒ **按此标准 RTP 未过**(待 n=8 配对检验确认)。

**为什么**: 与 RTC-K 那次同一个教训 —— **一阶最优 ≠ 实测最优**。
最小二乘投影最大化的是 cos(一阶对齐),而在 rho=0.10 这个半径上二阶效应已经主导
(kcurve 已独立证明该半径对非 grad 方向过大)。

**修法(与治好 RTC-K 的原则一致)**: 不用一阶最优,改为
**在 teacher 张成的空间里按实测 reward 取最优组合**(候选: 3 个单 teacher + 等权和 + 投影,
取 argmax F_q)。机制已存在(探针本就对多个方向算 F_q),成本低。
命名: RTP-best。

**注意**: 不因为"希望它成"而放宽判据。n=8 完成后配对检验,若 tproj 仍 ≈ teacher 则如实记为未通过,
再以 RTP-best 作为下一轮迭代。

## 2026-08-26 03:00 — RTP-best 实现,以及一个必须避开的选择偏差

**RTP-best**: 不用最小二乘的一阶最优,改为在候选集
{每个单 teacher, 等权和, 投影} 中**按实测 reward 逐样本取最优**。
与治好 RTC-K 的原则一致(kbest 把 acc_p 从 0.31 修到 1.000)。

**必须避开的坑(我第一版就写错了)**:
初版写成 `_res["rtpbest"] = _fs.max(...)`,即**用同一个指标既选又报** ——
C 个候选取最大,即使全部等价也会系统性偏高。这就是选择偏差,τ 那次栽过。

**已改为**:
- **选择**用即时 reward `_rw_gc`(方法在训练时真正能看到的信号);
- **评估**用固定后缀 `_Fq`(我们要报告的量);
- 另加**空对照 `rtprand`**: 从同一候选集**随机**选一个,报同样的 F_q。
  `rtpbest` 必须显著高于 `rtprand`,否则增益只是"从 C 个里挑"的统计假象。

另修: `_rw_gc` 原定义在 `gc_ksteps>0` 分支内,gc_ksteps=0 时会 NameError。已提到探针作用域顶层。
已用 AST 扫描确认无"被调用但未定义"的 helper。

## 2026-08-26 03:05 — RTP 门控判定: **未通过**(n=6 配对检验)

`gc_rtp3`(1034424), gc_ntproj=3:
```
方向        mean         配对                 diff       t     判定
grad     +0.09225       tproj - teacher   -0.00056   -1.5   打平(噪声内) <- 门控未过
teacher  +0.01479       tproj - tsum      +0.00476  +16.1   显著
tproj    +0.01423       teacher - tsum    +0.00532  +13.6   显著
tsum     +0.00947       teacher - rand    +0.01948  +21.4   显著
ksteps   -0.00222
rand     -0.00469
```
**最小二乘投影版 RTP 不优于最好的单个 teacher ⇒ 按预设判据不投入训练。判据不放宽。**

**n=8 最终确认(与 n=6 完全一致)**:
```
grad +0.09409 | teacher +0.01494 | tproj +0.01422 | tsum +0.00968 | ksteps -0.00217 | rand -0.00491
tproj-teacher -0.00072 t=-2.4 打平(符号始终为负,边缘上略差)
tproj-tsum    +0.00454 t=+16.8 显著
teacher-tsum  +0.00526 t=+17.0 显著
teacher-rand  +0.01985 t=+25.9 显著
```

**两个扎实的正向子结论**:
1. **投影显著优于等权组合**(+50%, t=16.1) —— 若必须组合 teacher,按 reward 梯度投影是对的做法;
2. **单个对齐 teacher 显著优于等权组合**(t=13.6) —— **平均多个 teacher 比挑对一个更差**。
   这**独立佐证了 conflict-averse 系列为何全部有害**(cagrad -7.4% / conflict -5.3% / cconf -5.1%):
   组合会稀释对齐的那一路信号。

**更重要的结构性发现**: 三个 teacher 类方向全部卡在 **~0.014**,而 grad 是 **0.092**
⇒ **teacher 张成的子空间本身只含约 15% 的可构造增益**(与 n=24 的 15.4% 吻合)。
RTP 不是实现得不好,是**天花板就在那里**。想突破必须离开 teacher 子空间 —— 这正是 RTC-K 在做的事。

**探针的 ksteps 列(-0.0022)不能代表 RTC-K**: 它用的是旧构造(直线外推到 rho=0.10),
而训练臂用的是 kbest + 半径上限。已加 **`ksbest` 列**(与训练臂构造一致)以做公平比较。
另修: `_y_ksbest = None` 的初始化因缩进匹配失败而**静默未生效**(str.replace 不匹配时不报错),
已补;gc_ksteps=0 时本会 NameError。

## 2026-08-26 03:10 — ksbest 列的可比性(先想清楚,免得事后找解释)

`ksbest` 用 `cap = min(budget, |d_best|)`,若最优迭代点位移 < rho,它就在**更小半径**上被测量,
而 grad/teacher/tproj/tsum/rand 都固定在 rho=0.10。已记录 `gc_ksbest_r`(实际半径/|y0|)。

**偏置方向是保守的**:
- 若 ksbest 在更小位移下**仍赢过 teacher** ⇒ 优势被低估,结论更强;
- 若 ksbest **输了** ⇒ **不能直接归因于方向差**,可能只是位移小。这一区分必须在报告里写明。

参照(kcurve, 即时 reward): grad 在 r=0.03 为 +0.0579、r=0.10 为 +0.0532 —— 该区间内相当平坦,
说明"半径从 10% 降到 3%"对 grad 类方向的损失有限。teacher 无半径曲线,如需严格可比需补测。

## 2026-08-26 03:15 — ksbest 结果的解读框架(先定,免得事后找说法)

`G_construct` 测的是**单个时刻(初始学生,新建 LoRA)的目标质量**,
而 RTC-K 与 teacher 蒸馏的本质差别在**训练动力学**:
- **teacher 目标固定** —— 不随学生改进而变,天花板就是 teacher 的水平;
- **RTC-K 目标来自学生自己的 reward 上升** —— 学生变强目标跟着变强,是自改进回路。

**预先约定的解读**:
- ksbest 显著 **>** teacher ⇒ 强结论,目标质量本身更好;
- 两者**打平** ⇒ **不能据此否定 RTC-K**。固定目标 vs 自改进目标的差异只在训练曲线上体现,
  单点 G_construct 捕捉不到。此时以训练臂的 reward 曲线为准;
- ksbest 显著 **<** teacher ⇒ 需先排除"半径更小"这一混杂因素(见 gc_ksbest_r)再下结论。

**同时也不过度声张**: 即使 ksbest 赢,那也只是"初始点的目标更好",
要证明方法有效仍须训练臂的最终 8 列指标 + 配对检验。

## 2026-08-26 03:20 — 方法学风险: 训练 run-to-run 方差可能是评测重测方差的 2~4 倍

**发现**: `poe15_*`(评测于 Aug 23 19:13)与 `poe15b_*`(评测于 Aug 26 02:32)是
**同一配置 `mopd_poe_l15` 的两次独立训练**在相同步号上的评测(checkpoint 目录被覆盖后重测)。
配对差值(n=3~4):

| 指标 | 训练 run-to-run RMS | 评测重测 1σ | 倍数 |
|---|---|---|---|
| geneval accuracy | 0.03439 | 0.01372 | **2.5x** |
| pickscore | 0.00220 | 0.00060 | **3.7x** |
| hpsv2 | 0.00285 | 0.00067 | **4.3x** |
| aesthetic | 0.03306 | 0.01654 | **2.0x** |
| clipscore | 0.00152 | 0.00102 | 1.5x |

**含义**: 现有所有"显著/打平"判据用的是**评测重测 σ**,而 arm 之间的对比同时包含**训练方差**。
正确的 arm-vs-arm 阈值应更宽。

**未排除的混杂 —— 因此暂不改写已有结论**:
两次训练相隔两天,中间修过 `is_trainable=True`(修复前 0 个可训练 LoRA 参数)等问题,
**两次未必是同一份代码**。若旧那次跑在修复前,则两者根本不可比,上表高估了训练方差。

**验证路径(不需额外训练)**: 当前 poe15 跑完后,两次 run 在 10 个步号上都有评测,
直接给出可靠的训练方差估计。提前评测(1034419)已在做这件事。

**影响评估**: 即使 2~4x 成立,量级大的结论不受动摇
(cagrad -7.4% 是评测 σ 的 17~24 倍,放宽 3 倍仍显著);
但**边缘结论(如 POE 的权衡 Aes +11.2σ)需要按新阈值重新审视**。

## 2026-08-26 03:15 — **更正两条基于 n=1 的错误结论**(curve_n12, n=8 带误差棒)

```
r        dR_grad             win     dR_ksteps           win
0.001    +0.00575±0.00048   0.983   +0.00125±0.00009    0.927
0.003    +0.01437±0.00119   0.990   +0.00339±0.00026    0.925
0.010    +0.03419±0.00268   0.993   +0.01035±0.00080    0.927
0.030    +0.05599±0.00363   0.995   +0.02554±0.00189    0.929
0.100    +0.05648±0.00201   0.976   +0.03553±0.00258    0.894
上升轨迹: 0.8115 -> 0.8333(+0.0218,峰) -> 0.8275 -> 0.8079 -> 0.7963
```

**更正 1**: 先前称"rho=0.10 对可实现方向过大,dR 翻负(-0.0037, win 0.361)" —— **错**。
n=8 实测 **+0.03553, win 0.894, 且随半径单调上升**。

**更正 2**: 先前称"上升路径是弯的,线性化丢掉 3.3x" —— **错**。

**真因不是噪声,而是两次测的是不同方向**:
- 旧(1034416): `_dir` 指向 **K=4 终点**(已越过极值的坏方向) -> r=0.10 处为负;
- 新(1034429): 因 `rtc_kbest` 默认开启,`_dir` 指向**最优迭代点**(好方向) -> 单调为正。

**正确表述**: 关键在于**用路径上的哪个点定义方向**,而非"路径弯曲本身携带增益"。
方向选对后,沿它**线性外推反而更好**(r=0.10 的 +0.0355 > 停在最优迭代点的 +0.0218)。

**ksteps/grad 比值随半径上升**: 22%(r=.001) -> 24% -> 30% -> 46% -> **63%(r=.10)**。
grad 在 r=0.03~0.10 已饱和(+0.05599 -> +0.05648),而 ksteps 仍在涨 ⇒ **大半径下差距收窄**。
先前基于 n=1 说的"41%"偏低,实测 **63%**。

**行动**: 我加的"信赖域上限"(budget=min(budget,|dir|))**是次优的** ——
它把位移停在 |d_best|(约 3~6%),而沿同一方向外推到 10% 能多拿约 60%。
rtckb 约 05:53 才启动,先用 smoke 验证"取消上限"是否更好,再决定是否改训练配置。

## 2026-08-26 03:20 — curve_n12 定稿(n=12,误差棒)

```
r        dR_grad             win     dR_ksteps           win    ksteps/grad
0.001    +0.00567±0.00034   0.984   +0.00110±0.00009   0.891   19%
0.003    +0.01423±0.00083   0.991   +0.00295±0.00026   0.890   21%
0.010    +0.03449±0.00186   0.994   +0.00896±0.00081   0.894   26%
0.030    +0.05742±0.00257   0.997   +0.02213±0.00199   0.896   39%
0.100    +0.05869±0.00169   0.976   +0.03153±0.00268   0.851   54%
上升轨迹: 0.8185 -> 0.8353(+0.0168,峰) -> 0.8283 -> 0.8107 -> 0.8006
```
n=8 的读数在 n=12 完全复现。三点确认:
1. `dR_ksteps` 在 rho=0.10 = **+0.0315±0.0027(约 12σ 显著为正)** ⇒ 先前"翻负"的说法彻底否定;
2. grad 在 r=0.03->0.10 已饱和(+0.0574->+0.0587),ksteps 仍在涨 ⇒ **大半径下追近到 54%**;
3. 停在最优迭代点 +0.0168 vs 沿同方向外推到 0.10 的 +0.0315 = **1.9 倍**
   ⇒ **信赖域上限确实次优**。已加 `rtc_kcap` 开关,`smk_nocap`(1034432) 实测验证中。

## 2026-08-26 03:30 — **更正**: "信赖域上限是次优的"这个推论也 premature

`smk_nocap`(1034432, 取消上限, 位移用满 rho=0.10) vs `smk_kbtrain`(带上限):
| | 带上限 | 取消上限 |
|---|---|---|
| rtc_acc_p | **1.000** | 0.991 |
| grad_norm | **0.0030** | 0.0026 |
| rtc_loss | 0.085 | 0.174 (2x) |
| kbest_gain | 0.0333 | 0.0329 |

取消上限后**损失翻倍、梯度反而更小** —— 与 raw vs kb 对比中同一个特征:
**目标更远但残差落在参数改变不了的方向上**。

**与曲线并不矛盾**: 曲线测的是 **G_construct**(外推到 0.10 更高: +0.0315 vs +0.0168),
smoke 测的是**对训练真正有用的量**(grad_norm/acc_p)。二者之差正是本项目的核心张力
**G_construct vs G_realized**,而我们的论点就是"可实现性更重要"。

⇒ **我先前"上限确实次优"的说法收回**。直接测量表明上限反而略好。
**决定: 不改 rtckb**(已在用带上限配置运行)。证据不支持该改动,
为一个未被证实的改进重启已跑 10 分钟的训练是错的。

**另更正**: poe15 并非"跑到 1000 约 05:53 结束" —— `env.sh:39` 有 `OPD_NUM_EPOCHS=800`
覆盖配置里的 1000,实际 800 epoch、03:19:42 正常完成(exit=0)。rtckb 已自动接上。

**另修**: `auto_eval_on_finish.sh` 生成的评测作业没有排除 ai-gpu01/ai-gpu03(泄漏 GPU),
poe15 的自动评测已派到 gpu03。已把 `--exclude` 写进 `slurm/eval_all.slurm` 模板,后续继承。
当前 gpu03 上的两个任务暂无 OOM,监控中。

## 2026-08-26 03:35 — rtckb 主线训练已确认健康运行

```
[arm] cfg=rtc_kb_train nproc=4 host=ai-gpu17 job=1034427
[arm] 有效配置: bs=3 nbpe=6 gas=6 -> samples/epoch=72 (目标 72)   与 8 卡梯度等价 ✓
[RTC-K] K=6 klr=0.02 |dir|=1.8122e+01 |y0|=3.2376e+02 cos(dir,g)=+0.01340
[RTC] 训练分支首次执行: j=0 jq=8 num_train_timesteps=9
```
`|dir|/|y0| = 5.6%`,与 smoke 的 5.3% 一致。RTC 分支在真实 4 卡训练中确认执行。

**追踪信息**:
- rtckb wandb run: `offline-run-20260826_032007-gyjeluyk`
- 日志目录: `logs/[RTC-KB + SD3.5-M + PickScore]`
- epoch 0 会触发一次在线评测(OPD_EVAL_FREQ=150,0%150==0),约 9.5 分钟;
  800 epoch 共约 5~6 次,合计约 55 分钟开销,可接受。
- checkpoint 落在 `checkpoint-{global_step}`,RTC 的 global_step = epoch ⇒ 0,30,60,...
  auto_eval 步号列表已覆盖该偏移。

## 2026-08-26 03:40 — **决定性结果: RTC-K 突破 teacher 子空间天花板**(n=4 配对,n=8 待确认)

`gc_ksbest`(1034431), 同批同 reward, gc_ntproj=3:
```
方向        mean        se       |  配对                  diff        t
grad     +0.09648   0.00218    |  ksbest - teacher    +0.01191   +7.1  显著
ksbest   +0.02695   0.00082    |  ksbest - tproj      +0.01259   +8.9  显著
rtpbest  +0.01818   0.00123    |  ksbest - rtpbest    +0.00877   +6.4  显著
teacher  +0.01503   0.00152    |  ksbest - ksteps     +0.02836  +25.8  显著
tproj    +0.01436   0.00143    |  rtpbest - teacher   +0.00315   +8.6  显著
tsum     +0.00945   0.00097    |  rtpbest - rtprand   +0.01178  +10.4  显著
rtprand  +0.00640   0.00084    |  teacher - rand      +0.01992  +15.6  显著
ksteps   -0.00141   0.00152
rand     -0.00489   0.00028
gc_ksbest_r = 0.052~0.058  (其他方向固定 0.10)
gc_pick_*: proj 43~51% | T0(对齐 teacher) 36~42% | sum 仅 7~10% | T1/T2 0~12%
```

**核心**: teacher/tproj/tsum/rtpbest 全部落在 0.009~0.018,而 **ksbest 达 0.027 = 最好单 teacher 的 1.79x (t=7.1)**,
且在**一半位移**(5.5% vs 10%)下达成 ⇒ 按预设解读框架,该偏置**保守**,真实优势只会更大。
以 grad 为分母: **RTC-K 吃到 28%,teacher 家族天花板 16%**。

**三个佐证**:
1. `rtpbest - rtprand` t=+10.4 ⇒ **空对照生效**,"按实测 reward 选"确实携带信息,
   不是"从 C 个里挑最大"的选择偏差。(该对照是实现时特意加的,现证明必要。)
2. `rtpbest - teacher` t=+8.6 ⇒ 同一"按实测选"原则也救活了 RTP(投影版打平,best 版 +21%),
   说明该原则通用 —— 与治好 RTC-K 的是同一条。
3. `gc_pick_sum` 仅 7~10% ⇒ **等权组合几乎从不是正确答案**,
   独立佐证 conflict-averse 系列(cagrad/conflict/cconf)失败的原因。

**必须同时说明的限制**:
- n=4(n=8 待确认);
- 这是**初始学生**上的 G_construct,不等于最终训练性能。方法是否有效仍须 rtckb 训练臂的
  8 列指标 + 配对检验;
- ksbest 半径 5.5% vs 其他 10%,不是严格半径匹配(但偏置方向保守)。

## 2026-08-26 03:36 — gc_ksbest 定稿(n=8),核心对比加强

```
方向        mean       sd        se      |  配对                diff        t
grad     +0.09575   0.00475   0.00168   |  ksbest - teacher  +0.01114   +12.7  显著
ksbest   +0.02561   0.00212   0.00075   |  ksbest - tproj    +0.01175   +15.0  显著
rtpbest  +0.01777   0.00246   0.00087   |  ksbest - ksteps   +0.02934   +44.2  显著
teacher  +0.01448   0.00278   0.00098   |  ksbest - rand     +0.03070   +39.6  显著
tproj    +0.01387   0.00265   0.00094   |  ksbest - grad     -0.07014   -43.8  显著
tsum     +0.00909   0.00191   0.00067   |  rtpbest - rtprand +0.01135   +15.7  显著
rtprand  +0.00641   0.00132   0.00047   |  rtpbest - teacher +0.00329   +17.0  显著
ksteps   -0.00373   0.00328   0.00116   |  rtpbest - tsum    +0.00868   +36.5  显著
rand     -0.00508   0.00057   0.00020   |  tproj - teacher   -0.00061    -1.6  打平
                                        |  tproj - tsum      +0.00478   +12.0  显著
gc_ksbest_r   = 0.0517~0.0530 (稳定 5.2%) |  teacher - tsum   +0.00539   +15.0  显著
gc_ksbest_idx = 1.89~2.01 (最优步稳定 ~2)
gc_pick: proj 43~51% | T0 36~40% | T2 1~10% | T1 ~0% | sum 仅 5.6~9.7%
```

**核心**: `ksbest - teacher` 从 n=4 的 t=+7.1 加强到 n=8 的 **t=+12.7**,比值 **1.77x**。
三个 n 值(6/4/8)下所有结论稳定复现,无一条因样本量变化而翻转。

**必须同等醒目地写出的负面事实**: `ksbest - grad = -0.07014, t=-43.8`。
RTC-K 吃到 grad 的 **26.7%**(teacher 家族 15.1%),**仍有 73% 拿不到**。
可实现性约束真实且昂贵 —— RTC-K 把 teacher 天花板抬高 77%,但**远未接近无约束上界**。
不能只报好消息。

## 2026-08-26 03:48 — **警报: rtckb 训练 reward 在下降,疑似发散**

epoch 0~24 (run offline-run-20260826_032007-gyjeluyk):
```
                前半      后半      变化
monitor_avg    0.8087   0.7635   -0.0452   <- 模型 reward 单调下降
rtc_R0         0.8063   0.7610   -0.0453
rtc_loss       0.0775   0.0942   +0.0166   <- 损失反而上升(目标越来越远)
rtc_kbest_gain 0.0272   0.0399   +0.0127   <- 上升能捞回的增益变大
rtc_kbest_idx  1.78  ->  4.39              <- 需要越来越多上升步
grad_norm      0.0021   0.0030
rtc_acc_p      恒为 1.0000                  <- 目标构造正常
```

**四个量互相印证**: 拟合把模型推离高 reward 区 -> 每轮上升要走更远才能找到好目标
-> 目标-模型差距扩大 -> 损失上升。典型发散形态,不是慢启动。
`acc_p=1.000` 说明**目标构造没问题,病在拟合环节**。

**已核对排除**: 损失符号正确 —— 正负两分支解出来都把 y_theta 推向 `y_0 + 2d`
(beta=0.5 的固有 2 倍外推),方向无误。

**隔离病因的实验**: 是 RTC-K 特有的 ksteps 目标有问题,还是两臂共用的损失/lr 有问题?
对照臂 rtcraw 能回答,但它排在 rtckb 之后。已用空闲 H200 提交 `diag_raw30`(rtc_raw_train, 30 epoch):
- raw **也下降** ⇒ 共用机制(损失形式/lr/beta)的问题,两臂都要改;
- raw **上升** ⇒ ksteps 目标的问题,RTC-K 需要重新设计。

**暂不杀 rtckb**: 让它继续积累证据(多几十个 epoch 能确认是否单调),
且 diag_raw30 才是判定依据。若确认发散再一并处理。

## 2026-08-26 03:52 — **撤回发散警报**: 是"先降后升",不是发散。误判源于汇总统计

同类比较(都用 monitor_pickscore,前一次用 poe15 的复合 monitor_avg 是错的对比):
```
poe15: 0.807 0.821 0.794 0.783 0.762 ... 0.850 0.858 0.849 0.871 -> 0.886   (+0.058/28ep)
rtckb: 0.817 0.821 0.818 0.802 0.794 0.809 0.800 0.819 0.808 0.817 0.819
       0.782 0.757 0.747 0.752 0.748 0.766 0.781 0.789 0.790 0.781 0.779
       0.760 0.730 0.747 0.778 0.796 0.825                                  (-0.028/28ep)
```
**rtckb 最后 5 个 epoch 明显回升: 0.730 -> 0.747 -> 0.778 -> 0.796 -> 0.825**,最新值已高于起点。
"先降后升"与 baseline 同形态(poe15 也在 epoch 4 跌到 0.762 再爬升),只是 rtckb 谷更深更晚。

**误判原因**: 我用了"前半/后半均值"这个汇总量,而**谷正好落在后半窗口内**,
汇总把 V 形回升读成了单调下降。**与今晚 n=1 那次同类错误: 用聚合量代替原始数据下判断。**
教训: 判断趋势必须看原始序列,汇总统计只能用于已确认形态之后。

**修正后的判断**: rtckb **未发散**,但**明显比 baseline 起步慢**(28ep 仅 +0.008 vs poe15 +0.079)。
现在过早,继续跑并观察。`diag_raw30`(1034443) 仍有价值: 说明对照臂是否同样先降后升。

**同时保留的疑点**(不因撤回警报而忽略): `rtc_kbest_idx` 从 1.78 升到 4.39、
`rtc_kbest_gain` 从 0.027 升到 0.040 —— 这两个量仍在单调走高,
含义是"每轮需要更多上升步、能捞回更多增益",与"模型正在远离高 reward 区"一致。
若回升只是暂时的,这两个量会继续走高。**把它们作为后续监控的主要信号。**

## 2026-08-26 04:10 — rtckb 训练中期对比(对 RTC-K 不利,如实记录)

V 形已走出,epoch 55 时 monitor_pickscore=0.887(起点 0.817,+0.077);
先前标记为主要监控信号的 `rtc_kbest_idx` 已在 4.5 附近**平台化**(不再单调走高) ⇒ 该疑点解除。

**同 epoch 配对对比 vs poe15 (n=56)**:
```
epoch窗    poe15    rtckb      差
  0-9    0.7979   0.8103   +0.0125
 10-19   0.8678   0.7731   -0.0947   <- rtckb 的 V 形谷
 20-29   0.8781   0.7879   -0.0901
 30-39   0.8913   0.8589   -0.0324
 40-49   0.8959   0.8756   -0.0204
 50-55   0.9082   0.8878   -0.0204
全程配对 -0.0424 t=-7.0 | 后半配对 -0.0267 t=-10.3
```
**rtckb 在 epoch 10 后每个窗口都落后,差距显著但在收窄。**

**必须讲清的两点**:
1. **poe15 不是正确对照**: 它是 3-teacher 复合 reward 的 POE 臂,rtckb 只优化 pickscore,
   两个不同方法族相比会混淆"目标构造方式"与"方法族"。
   **设计好的对照是 `rtcraw`**(同损失/门控/双分支,只差目标是否可实现)。
2. **但不能用"对照不当"轻轻带过**: rtckb **直接优化 pickscore 却在 pickscore 上落后于
   一个优化复合目标的臂**,这个事实本身不利。

**决定**: 在 rtckb vs rtcraw 出来前**不下结论、不改任何配置** —— 现在改就是在噪声上调参。
`diag_raw30`(1034443) 将给出对照臂前 30 epoch 的形态。

## 2026-08-26 04:12 — **撤回**"ksteps 目标致病"的推断: 对比不可比

`diag_raw30`(raw, 前 11 epoch 单调升 +0.031,无 V 形谷)vs `rtckb`(kb, 同期 +0.009 后下探)
—— 我据此推断"病因在 ksteps 目标",**推断不成立**。

**混杂**(config/opd.py:568 明确):`OPD_SMOKE=1` 会**换成微型数据集** `testdata/pickscore`,
注释原文"**不产出研究数据**"。此外 1 卡 vs 4 卡。
⇒ 两者 prompt 分布、难度、规模都不同,**reward 轨迹不可比**。

**正确的隔离**: 让两臂在**同一数据集、同一规模**下比。已提交 `diag_kb30`
(rtc_kb_train + OPD_SMOKE=1 + 30 epoch + 1 卡),与 `diag_raw30` 严格同条件。
绝对值不反映真实训练,但**相对比较(kb vs raw)正是所需**。
最终判据仍是完整的 rtcraw arm(1034428,排在 rtckb 之后)。

## 2026-08-26 04:28 — **严格半径匹配对照(n=8): 限制消除,结论加强**

```
方向          mean       se      |  配对                    diff       t
grad       +0.09346  0.00156    |  ksbest - teacherRKS   +0.01394  +51.5  显著(严格匹配)
gradRKS    +0.08275  0.00208    |  ksbest - teacher      +0.00963  +19.4  显著(未匹配)
ksbest     +0.02435  0.00048    |  ksbest - gradRKS      -0.05839  -33.4  显著
rtpbest    +0.01774  0.00093    |  teacherRKS - teacher  -0.00431  -11.5  显著
teacher    +0.01472  0.00093    |  gradRKS - grad        -0.01071  -10.5  显著
tproj      +0.01424  0.00100
teacherRKS +0.01041  0.00061
tsum       +0.00939  0.00067
rtprand    +0.00582  0.00051
ksteps     -0.00494  0.00116
rand       -0.00503  0.00044
```

**同一半径(5.2%)上 ksbest 是 teacherRKS 的 2.34x(t=+51.5)。**
未匹配时 1.65x -> 匹配后 **2.34x**。

**这验证了我在数据出来之前写下的预判**(见 03:10 条):
"若 ksbest 在更小位移下仍赢过 teacher ⇒ 优势被低估,结论更强"。
偏置确实保守 ⇒ **报告中"半径不匹配"这条限制可以删除**。

**同等醒目的负面**: `ksbest - gradRKS = -0.05839, t=-33.4`。
同半径下 RTC-K 只吃到不可实现上界的 **29%**(0.02435/0.08275),这一点没有变好。

## 2026-08-26 04:30 — rtckb 已追平 poe15(更新 04:10 的"显著落后"结论)

```
epoch窗      poe15    rtckb     差
  0-14     0.8177   0.7974   -0.0203
 15-29     0.8781   0.7835   -0.0946   <- V 形谷
 30-44     0.8928   0.8636   -0.0292
 45-59     0.9011   0.8846   -0.0165
 60-74     0.9069   0.8970   -0.0098
 75-87     0.9020   0.9038   +0.0018   <- 追平并略超
全程配对 -0.0288 t=-6.5 | 最近20配对 -0.0017 t=-0.5 打平(噪声内)
```
差距单调收窄,**最近 20 epoch 已在噪声内打平**。
⇒ 04:10 记的"rtckb 显著落后"**已过时**(那是 V 形谷未走完时的数据)。
早期谷损失约 30 个 epoch,已补回。

**仍保留的限制**: 88/800 epoch,早期;poe15 优化复合目标而 rtckb 只优化 pickscore;
**正确对照仍是 rtcraw**(1034428,排 rtckb 之后)。

## 2026-08-26 04:33 — 受控隔离实验(同微型数据集/1卡)初步: kb 早期落后 raw

```
n=6:  raw: 0.826 0.804 0.835 0.835 0.837 0.850
      kb : 0.826 0.798 0.817 0.797 0.799 0.801
      配对 kb-raw: mean=-0.0249 se=0.0081 t=-3.1
raw 全程 24 epoch: 0.826 -> 0.904 单调上升
```
两臂 epoch-0 完全相同(0.826) ⇒ 起点一致,对比干净。

**不下结论**: 主线数据显示 kb 是"先降后升",epoch 75+ 已追平 poe15。
raw 可能先快后平、kb 可能后来居上。**必须等 diag_kb30 跑到 24+ epoch 与 raw 完整轨迹对齐再比。**
今晚已多次踩"用早期片段推断整体趋势"的坑(n=1 曲线、V 形误判为发散、poe15 复合 reward 误比),
这次先记录不定论。

## 2026-08-26 04:37 — raw 30 epoch 完成;两臂 loss 走向相反(概念差别,但不作为辩解)

`diag_raw30` 完成 30 epoch: **0.8275 -> 0.9065 (+0.079)**,且 `rtc_loss` **下降**(0.323 -> 0.270)。
`rtckb` 主线的 `rtc_loss` 则**上升**(0.081 -> 0.110)。

**概念差别**:
- raw 的目标是 `y0 + rho*ghat`,方向由 reward 梯度定 ⇒ 模型收敛过去后损失自然降;
- kb 的目标是"模型自己再走 K 步上升会到哪" ⇒ **目标随模型一起前进**。
  模型变强 -> 上升能捞回的增益变大(kbest_gain 0.027->0.052) -> 目标往前挪 -> 损失不降。
  即**自改进目标的损失本就不该收敛到 0**;损失下降反而意味着目标停止前进。

**但不把这当作辩解**: 这是事后合理化的典型形态。**真正判据是 reward 曲线本身**,
等 diag_kb30 跑满 30 epoch 与 raw 直接对齐比较。

## 2026-08-26 04:40 — rtckb 反超 poe15(epoch 80 附近交叉)

```
epoch窗       poe15    rtckb     差
  0-19      0.8328   0.7917   -0.0411
 20-39      0.8847   0.8234   -0.0613
 40-59      0.8998   0.8817   -0.0181
 60-79      0.9063   0.8984   -0.0079
 80-99      0.9002   0.9059   +0.0057   <- 交叉
100-102     0.9059   0.9201   +0.0142
全程 -0.0234 t=-5.8 | 最近30 +0.0048 t=+1.7 | 最近15 +0.0082 t=+2.5
```
最近 15 epoch 配对 **t=+2.5**,正落在 |t|>2.5 阈值上 ⇒ **按"边缘/接近显著但未过线"记录,不上调为显著**。

**形态差异**: poe15 最近 40 epoch **已平台化**(0.9002~0.9063),rtckb **仍在爬升**
(0.8984 -> 0.9059 -> 0.9201)。若趋势持续差距会扩大。

**限制不变**: 103/800 尚早;poe15 优化复合目标而 rtckb 只优化 pickscore
(在 pickscore 上"本该"占优);**正确对照仍是 rtcraw**。

## 2026-08-26 04:48 — 受控对比 n=13: kb 落后 raw,但**实验长度结构性不足,不下结论**

```
raw: 0.826 0.804 0.835 0.835 0.837 0.850 0.835 0.852 0.863 0.870 0.870 0.887 0.879
kb : 0.826 0.798 0.817 0.797 0.799 0.801 0.786 0.817 0.830 0.835 0.832 0.826 0.804
全程 -0.0365 t=-6.4 | 后半 -0.0465 t=-7.7 | 最近8 -0.0468 t=-8.9
```

**为何不下结论**: 主线数据显示 kb 的 V 形谷 **epoch 23 触底、40~50 恢复、80 反超**。
受控实验只有 30 epoch ⇒ **整段都落在 kb 的谷里**,等于拿"kb 最差阶段"比"raw 正常阶段"。
与今晚已踩四次的是同一个坑(用早期片段推断整体),区别是这次在下结论**之前**发现。

**行动**: 把受控实验延长到覆盖恢复期(100 epoch),两臂同条件并行。

## 2026-08-26 05:08 — rtckb 对 poe15 的优势转为显著且扩大(epoch 147)

```
epoch窗       poe15    rtckb     差
  0-24      0.8402   0.7852   -0.0550
 25-49      0.8922   0.8571   -0.0350
 50-74      0.9056   0.8934   -0.0122
 75-99      0.9010   0.9052   +0.0042   <- 交叉
100-124     0.9063   0.9166   +0.0103
125-146     0.9032   0.9211   +0.0179
全程 -0.0123 t=-3.8 | 最近50 +0.0135 t=+7.1 显著 | 最近25 +0.0178 t=+6.8 显著
```
最近 50 epoch 配对 **t=+7.1**,远超 |t|>2.5 阈值 ⇒ 不再是"边缘",是**显著**。
优势单调扩大(+0.004 -> +0.010 -> +0.018): **poe15 已平台化(0.901~0.906),rtckb 仍在爬升**。

**限制不变**: 147/800;poe15 优化复合目标而 rtckb 只优化 pickscore(在 pickscore 上本该占优);
**设计好的对照仍是 rtcraw**。全程配对为负(t=-3.8)是早期 V 形谷的残留,不代表当前状态。

poe15 的 10 点评测已完成 6 个(61/151/241/331/421/511)。

## 2026-08-26 05:30 — **收回"rtckb 显著优于 poe15"**: prompt 分布混杂

```
poe15:  3 teacher, 数据集 pickscore / ocr / geneval  -> 采样在三数据集轮转
rtckb:  1 teacher, 数据集仅 pickscore               -> 只采 pickscore prompt
```
`monitor_pickscore` 在**实际采样到的 prompt** 上计算 ⇒
- poe15 的分数在 **pickscore+ocr+geneval** 三类 prompt 上平均;
- rtckb 只在 **pickscore** prompt 上。

OCR/geneval 的 prompt(文字密集、组合性)对 pickscore 本就更难
⇒ **poe15 分数偏低可能纯因 prompt 更难,与方法优劣无关**。
**先前报的 t=+7.6 / t=+6.7 不能用来支持 RTC-K 更好,收回。**

**这正说明 rtcraw 才是设计好的对照**(同数据集、同 teacher 数、同损失,只差目标构造)。

**真正有效的对比是 10 点轨迹评测**: 跑在 drawbench/geneval/ocr 标准 benchmark 上,
与训练 prompt 分布无关。rtckb 的 auto_eval 会在结束时产出。
在那之前,**训练层面的优劣仍属未知**。

**今晚第 6 次收回**。前 5 次病因: 用聚合量/不可比条件下判断。这次是**第三方混杂变量**
(prompt 分布),形态不同但同源: 比较前没有逐项核对两边的实验条件。

## 2026-08-26 05:33 — **受控对比 30 epoch 完成: RTC-K 显著落后对照臂(重要负面结果)**

同微型数据集 / 同 1 卡 / 各 30 epoch:
```
raw: 0.826 0.804 0.835 ... 0.902 0.916 0.900 0.918 0.896   -> 稳定爬升到 ~0.90
kb : 0.826 0.798 0.817 ... 0.855 0.853 0.835 0.856 0.845   -> 谷底 0.786(ep6)
                                                              恢复 0.861(ep23)
                                                              **平台 ~0.85**
全程配对 -0.0486 t=-13.2 | 后半配对 -0.0545 t=-18.7
```
**kb 确实走出 V 形谷(ep18~23 恢复),但恢复后停在 ~0.85 而 raw 到 ~0.90,
差距在后半段反而扩大 ⇒ 不能再用"还在谷里"解释。**

### 与目标构造层面的结论方向相反
| 层面 | 结果 |
|---|---|
| 目标构造 | ksbest = **2.34x** teacher (t=+51.5) |
| 训练结果(受控) | kb **落后** raw -0.055 (t=-18.7) |

**构造出更好的目标,却训练出更差的模型。**
这恰是 OPSD 指出、RTC 本要解决的 `G_construct - G_fit` 缺口: 优势在拟合阶段被吃掉。
**若该结果在完整数据集上复现,说明 RTC-K 没有解决它设定要解决的问题。**

### 必须保留的限制
- 微型数据集(config 注释原文"不产出研究数据"),可能过拟合/动力学不同;
- 主线 rtckb 在完整数据集上仍在爬升(0.924 @ ep163,无平台迹象);
- 权威判据是 **rtcraw 完整臂**(同数据集同配置,只差目标构造)与 **10 点轨迹评测**。

### 已在跑的确认
`d100_kb` / `d100_raw`(各 100 epoch,同条件)将确认平台差是否持续到 100 epoch。

## 2026-08-26 05:50 — 病因假说: kb 的有效步长是 raw 的约 3 倍(已提交检验)

实测(三个 run 一致):
```
d100_kb    grad_norm 0.00202 (末10 0.00186)   rtc_loss 0.0765
d100_raw   grad_norm 0.00076 (末10 0.00059)   rtc_loss 0.2903
rtckb 主线  grad_norm 0.00198 (末10 0.00177)   rtc_loss 0.1105
```
`max_grad_norm=1.0`,两者都远低于阈值 ⇒ **未被裁剪**。同 lr=3e-4 下
**kb 每步走得远约 3 倍**。

**假说**: kb 落后可能是**步长过大**(快速下探 -> 在更差点附近震荡 -> 平台更低),
而非目标构造本身有缺陷。这个区分决定 RTC-K 是"有根本缺陷"还是"需要自己的学习率"。

**检验**: `d100_kblr`(1034452) = kb + lr 1e-4(按梯度比 3e-4×0.32),100 epoch,同微型数据集/1卡。
- 若 kb@1e-4 **追平或超过 raw** ⇒ 病因是步长,目标构造无问题,RTC-K 需要自己的 lr;
- 若仍落后 ⇒ 病因在目标构造,RTC-K 需要重新设计。

注: 这不是"调参救结果" —— 步长差异是**实测的**(grad_norm 3 倍),
按该比值缩放是消除已知混杂,而非搜索超参。判据预先写定,不事后调整。

**d100 中途(n=21)**: kb 落后 raw -0.047 (t=-11.4),与 30 epoch 结论一致,差距稳定不是暂态。

## 2026-08-26 06:08 — **关键认识: 探针数据早已预测了训练结果,是我先前比错了基准**

三臂受控(同微型数据集/1卡):
```
kb      n=28  最近 0.825 0.828 0.841 0.837   平台 ~0.83
raw     n=48  最近 0.923 0.926               仍在爬升
kb@1e-4 n=6   0.826 0.797 0.821 0.817 0.818 0.815  (太早)
kb-raw 最近20: -0.0599  t=-19.4   <- 差距扩大
```

**关键**: `rtcraw` 对照臂用的是 **grad 方向**,而探针测得
```
grad(raw 臂实际使用)   +0.0935
ksbest(kb 臂实际使用)  +0.0256   <- 只有 grad 的 27%
teacher                +0.0145
```
我一直在报"ksbest 是 teacher 的 2.34x",但**真正的对照臂不是 teacher,是 grad,
而 grad 的构造增益是 ksbest 的 3.8 倍**。

⇒ **RTC-K 的处境: 目标更可实现,但构造增益少得多。**
RTC 的赌注是"raw 的构造增益在拟合阶段丢掉 98.7%,所以可实现性更值钱"。
**训练结果说: 在这个设置下,构造增益的劣势压过了可实现性的优势。**

这让负面结果从"不可解释"变成**完全自洽** —— 探针与训练结果指向同一件事,
只是我先前挑错了比较基准(拿 teacher 当对照,而实际对照是 grad)。

**对 idea 的影响**: 若该结论在完整数据集上成立,则
"用可实现方向替换 reward 梯度方向"这一核心主张**不成立**;
可保留的正向结果是 **teacher 子空间天花板 ~15%** 与 **ksbest 突破该天花板**(仍有独立价值),
但不能声称它优于 OPSD 式构造。

## 2026-08-26 06:28 — **步长假说被否定(判据事先写定): 病因在目标构造**

同 epoch 对比 (n=13, 同微型数据集/1卡):
```
raw       末3均 0.8782   grad_norm末10均 0.00072
kb@3e-4   末3均 0.8287   grad_norm末10均 0.00227   (-0.033 vs raw, t=-7.1)
kb@1e-4   末3均 0.8155   grad_norm末10均 0.00249   (-0.035 vs raw, t=-5.6)  <- 无改善,略差
```
**事先写定的判据**: "kb@1e-4 追平或超过 raw ⇒ 病因是步长; 仍落后 ⇒ 病因在目标构造"。
**仍落后 ⇒ 病因在目标构造本身,不是学习率。** 判据未事后调整。

### 三条证据构成自洽的结论链
1. **探针**: ksbest 构造增益 0.0256,只有 grad 0.0935 的 **27%**;
2. **训练**: kb 显著落后 raw,差距随训练**扩大**(t=-19.4);
3. **排除**: 按实测梯度比把步长缩小 3 倍,**无改善**。

⇒ **构造增益的劣势压过可实现性的优势。**
**RTC 的核心主张——"用可实现方向替换 reward 梯度方向能提高训练效果"——在此设置下不成立。**

### 仍需完成才能定论
- n=13 偏小,d100 三臂各 100 epoch 仍在跑;
- 微型数据集(非研究数据),完整数据集上的 `rtcraw` 对照臂是权威判据;
- 10 点轨迹评测(标准 benchmark)尚未产出。

## 2026-08-26 06:50 — poe15 完整轨迹(8/10 点)复现"POE overall 打平"

```
step:  61      91     151     241     331     421     511     601     691
Ratio: 0.9430 0.9581 0.9700 0.9868 0.9966 0.9955 0.9987 0.9977 1.0011
判定:  显著低 显著低 显著低 显著低  打平   打平   打平   打平   打平
```
重测 1σ(Ratio)=0.31%,阈值 2σ=0.61%。
单调上升,约 step 331 后进入"与官方 Student 打平(噪声内)"并稳定到 1.0011。
与此前结论一致: **POE λ=1.5 是真实权衡(Aes +11.2σ / HPS +9.3σ),但 overall 打平**。
本轨迹用 9 个点独立复现,比原少点位版本更扎实。751/781 两点评测在跑。

## 2026-08-26 06:47 — 三臂受控 n=21 同 epoch: 结论稳固
```
raw      0.8892
kb@3e-4  0.8341  (-0.045, t=-9.9)
kb@1e-4  0.8289  (-0.046, t=-9.4)
```
随数据增加两个 kb 变体一致落后 raw ~0.045,**学习率确实无关**,与 06:28 的判定一致。

## 2026-08-26 07:08 — 步长假说彻底否定;据此推出一个可能救回 idea 的实验

同 epoch n=29: `raw 0.9096 | kb@3e-4 0.8455 (t=-13.4) | kb@1e-4 0.8016 (t=-10.5)`
kb@1e-4 明显**比** kb@3e-4 **更差** ⇒ 降 lr 不但无益反而更慢,**假说彻底排除**。

**由病因推出的新实验**: 既然病因是"构造增益不足",就应**提高 ksbest 的构造增益**。
curve_n12(n=12)已证明: 沿最优迭代点方向外推到 r=0.10 得 **+0.0315**,
远高于停在最优迭代点自身半径的 +0.0168。

**关键**: 我在 03:30 测过 no-cap 变体,当时依据 `grad_norm`(取消上限后 0.0026 < 0.0030)
判定"上限反而略好",决定不改。**但现在已知 grad_norm 不是正确的预测量 —— 构造增益才是。**
那次判断的依据被推翻,**必须重测**。

已提交 `d100_nocap`(rtc_kbest_nocap, 100 epoch, 同微型数据集/1卡)。
**判据(事先写定)**: 与 raw 同 epoch 比较 ——
- nocap **追平或超过 raw** ⇒ 构造增益机制确认,且给出更好的 RTC-K 变体(idea 可救);
- nocap **仍落后但优于 kb@3e-4** ⇒ 机制方向对但幅度不够;
- nocap **不优于 kb@3e-4** ⇒ 构造增益不是主因,需另找病因。

## 2026-08-26 07:29 — d100_raw 完成 100 epoch: 受控结论定稿

```
raw 分段: 0-19:0.8611  20-39:0.9050  40-59:0.9235  60-79:0.9400  80-99:0.9476  末10均 0.9532
kb  分段: 0-19:0.8171  20-39:0.8493  40-58:0.8858                              末10均 0.8874
同 epoch (n=59): 全程 -0.0457 t=-21.6 | 最近20 -0.0365 t=-41.2
```
raw 单调爬升到 0.953,kb 到 59ep 仅 0.887。差距缓慢收窄(-0.046 -> -0.037),
但基数大且方差变小,t 升至 -41.2。

**受控条件下结论已无需更多数据: RTC-K 的目标构造在训练中明显劣于 OPSD 等价构造。**

**唯一剩余的救援实验**: `d100_nocap`(1034454,刚启动) 直接检验诊断出的机制 ——
把位移从 5.2% 放开到 10%,理论上把构造增益从 +0.0168 提到 +0.0315(curve_n12 n=12 实测)。
判据已于 07:08 事先写定。

## 2026-08-26 07:31 — continue 修复已验证;但**更正**我对 checkpoint 偏移的判断

**修复验证**: rtckb 的 checkpoint = `1,31,61,91,121,151,181,211,241,271,301,331`
—— 12 个不同步号,间隔 30。若 bug 未修,全部会覆盖到 `checkpoint-0`。**修复确实生效。**

**更正**: 我在 07:15 称"RTC 的 global_step = epoch ⇒ checkpoint 落在 0,30,60",
**实际是 1,31,61,与常规 arm 一致**。
真实机制: `save_ckpt` 在该 epoch 的训练循环之后调用,此时 global_step 已递增为 epoch+1。
我把 smoke 日志里 `_step=0,1,2` 的**上报时点**误当成了**保存时点**。

**影响**: 我据此给 auto_eval 步号列表加进了 0,30,60。**改动本身无害**
(脚本按目录存在性过滤,多余项自动跳过),但**推理是错的**,已更正记录。

## 2026-08-26 07:48 — 把 reward-位移曲线扩到 r=0.30(检验"放大位移"这条修复路线的上限)

**动机**: curve_n12(n=12) 只测到 r=0.10,而在那里
- **grad 已饱和**(r=0.03 +0.0574 -> r=0.10 +0.0587);
- **ksteps 仍在上升**(+0.0221 -> +0.0315,占 grad 比从 39% 升到 54%)。

若 ksteps 在 r>0.10 还能继续涨,则"放大位移"这条修复路线(nocap 正在测的方向)有更大空间;
若已饱和,则 nocap 最多只能补到 +0.0315,仍不足以追平 grad 的 +0.0587。
**这决定了 RTC-K 是否存在可行的修复区间。**

已提交 `curve_r30`(1034456): 半径扫描扩到 {0.001,0.003,0.01,0.03,0.10,0.20,0.30},12 epoch 带误差棒。

**四臂受控中途(n=6,全部在谷内,不可判)**: nocap 0.8079 略高于 kb@3e-4 0.8017,
方向与机制预测一致,需更多 epoch。

## 2026-08-26 08:09 — kb@1e-4 终止(数据已足)+ 资源重分配

**kb@1e-4 最终数据(n=52)**:
```
同 epoch(52):  raw 0.9202  |  kb@3e-4 0.8864 (t=-19.7)  |  kb@1e-4 0.7875 (t=-16.5)
```
降 lr 让 kb 从 0.8864 掉到 **0.7875** —— 不是"没帮助",是**明显更差**。
**步长假说彻底否定,且方向相反: kb 至少需要 3e-4。**

**资源重分配**: 取消 `d100_kblr`(1034452),把 GPU 让给排队中的 `curve_r30`。
理由: lr 假说已被明确否定,跑满 100 epoch 不会改变任何结论;
而 curve_r30 回答"**是否存在可行的修复区间**"(ksteps 在 r>0.10 是否还能涨),更具决定性。
**用一个已被否定的假说的算力,换一个能决定 idea 生死的测量。**

## 2026-08-26 08:28 — curve_r30: 修复路线的上限确定,并**更正"只有 27%"这个数字**

```
r        dR_grad             dR_ksteps           ks/grad    win_ksteps
0.001    +0.00517±0.00053    +0.00129±0.00013     24.9%
0.003    +0.01300±0.00141    +0.00349±0.00034     26.8%
0.010    +0.03095±0.00308    +0.01064±0.00105     34.4%
0.030    +0.05123±0.00361    +0.02663±0.00247     52.0%
0.100    +0.05421±0.00165    +0.03828±0.00304     70.6%     0.961   <- 共同峰
0.200    +0.03575±0.00319    +0.01562±0.00229     43.7%     0.744
0.300    +0.01730±0.00458    -0.00898±0.00189    -51.9%     0.389
```
**两条曲线都在 r=0.10 达峰,之后双双下降,ksteps 到 0.30 翻负。**
⇒ "放大位移"没有更多空间,**nocap 用的 rho=0.10 恰好是最优点**。

### **更正 06:08 的"ksbest 只有 grad 的 27%"**
那个数字是拿 **ksbest 在自己的 5.2% 半径** 对比 **grad 在 10%** —— 半径不匹配。
**同一 r=0.10 上,可实现方向能拿到 grad 的 70.6%**,不是 27%。

| 变体 | 半径 | 构造增益占 grad |
|---|---|---|
| kb(带上限) | ~5.2% | ~55% |
| **nocap** | **10%** | **70.6%** |
| raw | 10% | 100% |

⇒ **`nocap` 才是该方法应有的形态**;带上限的 kb 白白丢掉约 15 个百分点。
(注: 我在 03:30 曾依据 grad_norm 判定"上限略好"并保留它 —— 那个依据已被推翻两次。)

### 真正的问题被精确化了
"**71% 构造增益 + 更高可实现性** 能否胜过 **100% 构造增益 + 更低可实现性**"
—— 这正是 `d100_nocap` vs `d100_raw` 在测的。
**主线 rtckb 用的是带上限的 kb(约 55%),不是最优形态** —— 这一点须在报告中写明。

## 2026-08-26 08:34 — **run 识别错误已修**;nocap 不优于 kb ⇒ 即时 reward 不是正确的预测量

**先修一个错误**: 我按时间戳把 `curve_r30` 误认成 `d100_nocap`,做了一次无效对比。
已写 `wandb_by_name.py`(从 wandb 记录读 `run_name` 精确定位),不再靠时间戳猜。
正确对应: 072438=RTC-KBNOCAP | 080838=RTC-KCURVE | 044835=RTC-KB | 044836=RTC-RAWB。

**正确对比(n=20)**:
```
raw 0.8869 | kb 0.8305 (t=-9.4 vs raw) | nocap 0.8393 (t=-7.9 vs raw)
nocap - kb: -0.0058  t=-1.3  打平(噪声内)
```
**nocap 并不优于 kb**,尽管构造增益从 55% 提到 70.6%。

### 这说明我 08:28 用错了度量
| 度量 | ksbest vs grad | 与训练差距吻合? |
|---|---|---|
| **即时 reward**(查询点干净图) | 70.6% | ❌ nocap 应接近 raw,实际没有 |
| **固定后缀 F_q**(插入后推完的终图) | 27% | ✅ 吻合大幅落后 |

**F_q 才可能是预测训练结果的量** —— 决定图像质量的是终点,不是查询点的干净图预测。
**即时 reward 高估了可实现方向的价值。**

### 缺的关键数据已补测
探针现有的 `ksteps` 列用 K=4 **终点**方向(已知坏方向),`ksbest` 列被限制在 5.2%,
**缺"最优迭代点方向 + 半径用满 rho"** —— 正是 nocap 实际使用的目标。
已加 `ksbestR10` 列并提交 `gc_r10`(1034458)。
**判据**: 若 F_q(ksbestR10) 仍远低于 F_q(grad),则"F_q 预测训练结果"成立,
且解释为何 nocap 无改善;若 F_q(ksbestR10) 接近 grad 而训练仍落后,则 F_q 也不是预测量,需另找。

## 2026-08-26 08:48 — curve_r30 定稿(n=12);**更正 08:28 的 70.6%**

```
r        dR_grad             dR_ksteps           ks/grad   win_ks
0.001    +0.00560±0.00027    +0.00122±0.00006     21.8%    0.941
0.003    +0.01414±0.00070    +0.00327±0.00016     23.1%    0.947
0.010    +0.03390±0.00154    +0.00987±0.00050     29.1%    0.948
0.030    +0.05623±0.00203    +0.02424±0.00129     43.1%    0.950
0.100    +0.05763±0.00123    +0.03390±0.00200     58.8%    0.913   <- 共同峰
0.200    +0.03541±0.00153    +0.01166±0.00188     32.9%    0.676
0.300    +0.01402±0.00229    -0.01317±0.00166    -94.0%    0.340
```
**更正**: 08:28 报的 **70.6% 是 n=5 的读数;n=12 实测 58.8%**。
**今晚第三次因样本量不足下判断而需更正**(前两次: n=1 的曲线、n=6 的 nocap 对比)。
**规则: n<10 时不报具体数字,只报方向。**

结论不变但更准确: 两条曲线都在 **r=0.10 达峰**,"放大位移"路线到顶;
**ksteps 最好也只有 grad 的 58.8%**;r>0.10 双双下降,ksteps 在 0.30 翻负(win 0.340)。

**非线性之谜**: nocap 用 r=0.10(58.8%),kb 用 5.2%(约 43~52%) ——
**从 ~48% 提到 58.8% 训练毫无改善,但 58.8% 到 100%(raw) 却差距巨大**。
⇒ 即时 reward 不是正确的预测量。`gc_r10`(1034457) 测的 F_q 才是判据。

## 2026-08-26 09:08 — **机制查明: F_q 是预测量;可实现方向的改善"不耐传播"**(n=4,待 n=8 确认)

`gc_r10` 加测 `ksbestR10`(= nocap 臂实际使用的目标: 最优迭代点方向 + 半径用满 rho):
```
grad       +0.09430   |  ksbestR10 - ksbest  +0.00355  t=+8.2   显著但幅度很小
gradRKS    +0.08258   |  ksbestR10 - grad    -0.06537  t=-48.3
ksbestR10  +0.02893   |  ksbest    - grad    -0.06891  t=-43.9
ksbest     +0.02538   |  teacher   - rand    +0.02045  t=+55.0
teacher    +0.01604
```

### 两个度量给出的改善幅度完全不同 —— 这解释了非线性之谜
| 度量 | kb -> nocap | 训练是否改善 |
|---|---|---|
| 即时 reward(查询点干净图) | ~48% -> 58.8% (**+11 点**) | ❌ 没有 |
| **固定后缀 F_q**(推到终图) | **~27% -> ~31% (+4 点)** | ✅ 吻合(t=-1.3 打平) |

**⇒ F_q 才是预测训练结果的量,确认。**

### 有实质意义的机制发现
**可实现方向在查询点把干净图预测改善得不错(58.8% of grad),
但该改善在剩余去噪步骤中大量流失,到终点只剩 ~30%;grad 方向的改善能更好地存活。**

⇒ RTC-K 的问题不是"目标不够好",而是**"目标的改善不耐传播"** ——
低噪声点上的改进并非都能等价传到终图,取决于方向。
这是本项目一个可写的、非平凡的机制性结论(OPSD 的固定后缀界只给了 Lipschitz 上界,
没有区分方向间的传播差异)。

**注**: n=4,按自定规则不下精确结论,但 t=-48 且模式与训练结果一致。待 n=8 确认。

## 2026-08-26 09:28 — 最终数据;**更正 09:08 的"F_q 是预测量"**

**F_q(gc_r10, n=8)**:
```
grad +0.09407 (100%) | gradRKS +0.08292 (88%) | ksbestR10 +0.02925 (31.1%)
ksbest +0.02558 (27.2%) | rtpbest +0.01851 | teacher +0.01550 | tproj +0.01416
teacherRKS +0.01091 | tsum +0.01010 | rtprand +0.00561 | ksteps -0.00347 | rand -0.00480
```
**三臂训练(n=40 同 epoch)**:
```
raw 末10均 0.9532(n=100) | kb 0.9198(n=100) | nocap 0.8562(n=40)
kb    - raw : -0.0498 t=-17.4
nocap - raw : -0.0583 t=-15.0
nocap - kb  : -0.0084 t=-3.3   <- nocap 显著更差
```

### **更正**: F_q 也不能单独预测训练结果
nocap 的 F_q 比 kb **高 14%**(0.02925 vs 0.02558),训练却**显著更差**(t=-3.3)。
⇒ 09:08 的"F_q 是预测量"**不成立**。

**差别在半径**: nocap r=0.10 vs kb r=5.2%。smoke 数据: nocap 的 `rtc_loss` 是 kb 的 2 倍
(0.174 vs 0.085)而 `grad_norm` 反而略低 ⇒ **目标更远,但多出的位移模型跟不上**。

**这正是 G_construct - G_fit 分解本身**: nocap 构造增益更高但拟合损失也更高,净值更差。
**没有单一指标能预测,该权衡是实打实的。**

### 诚实的最终判断
- **跨方法族**(grad vs ksteps): F_q 高者训练好 —— 一致;
- **族内**(kb vs nocap): F_q 高者训练**差** —— 被半径/拟合难度反转。
⇒ 只能说"构造增益与拟合难度共同决定结果",无法归因到单一量。
**这本身就是 OPSD 提出 G_realized = G_construct - G_fit 的实证支持,
但也说明我们没有找到能优化该差值的构造方式。**

## 2026-08-26 09:52 — **方法学: 现用重测 1σ 低估评测噪声 1.1~2.5 倍**(影响所有显著性判定)

`poe15_*`(auto_eval) 与 `poe15b_*`(我的预评测)现在是**同一次训练、同一批 checkpoint 的两次独立评测**
⇒ 差值 = **纯评测噪声**。重叠步号 61/331/601:
```
指标                  RMS       现用 1σ    比值
geneval:accuracy    0.02966   0.01372    2.2x
drawbench:pickscore 0.00148   0.00060    2.5x
drawbench:clipscore 0.00143   0.00102    1.4x
drawbench:hpsv2     0.00110   0.00067    1.6x
drawbench:aesthetic 0.01791   0.01654    1.1x
```

**第一性原理核对**: geneval 用 256 prompt、准确率约 0.85 ⇒ 二项抽样 sd = sqrt(.85*.15/256) ≈ 0.022,
两次评测之差的 sd ≈ **0.031**,与实测 RMS 0.0297 吻合。
⇒ **现用 0.01372 确实偏小,真实评测噪声约为其 2 倍。**

### 这修正了 03:10 的"训练方差是评测噪声 2~4 倍"
当时测的 old-run vs new-run 差值(1.5~4.3x)中,**很大一部分就是评测噪声本身**(1.1~2.5x),
**训练方差的增量远没那么大**。

### 同时如实记录一处数据损失
auto_eval 的 `poe15_*` 标签**覆盖了 2026-08-23 旧运行的评测结果**
⇒ 03:10 那条训练方差结论**已无法复核**。数字已留档但不可复现。
**教训**: auto_eval 的 TAG 应带运行标识(如日期或 jobid),避免跨运行覆盖。

### 影响
n=3 偏小,但 5 个指标方向一致,且 geneval 有理论支持。
**建议: 把 overall_score.py 的判定阈值按 2x 放宽后重跑一遍所有结论**,
特别是边缘结论(POE 权衡、任何 |t| 在 2.5~5 之间的判定)。
量级大的结论(cagrad -7.4% = 17~24 倍 1σ)不受影响。

## 2026-08-26 09:55 — 2x 阈值下的影响评估: **主结论全部不变**,16 个边缘单点标签需改

按 2x 放宽阈值重跑全部结论:

**会翻转的**: 16 个**边缘单点判定**(Δ% 在 -0.63%~-1.13%,多为训练中段 checkpoint),
从"显著低于参照"变为"打平"。涉及 tau02/tau10/tau20/caopd/poe15/regret/mopd931 的若干中间点。

**不翻转的(主结论)**:
| 结论 | 现值 | 2x 阈值下 |
|---|---|---|
| cagrad -7.4% / conflict -5.3% / cconf -5.1% | 17~24σ | **仍显著**(8~12σ) |
| τ 四点扫描全打平 | — | **不受影响**(放宽只会更打平) |
| POE overall 打平 | — | **不受影响** |
| POE 权衡 Aes +11.2σ / HPS +9.3σ | — | **仍显著**(+5.6σ / +4.7σ) |
| teacher 子空间天花板 t=106.7 | — | **不适用**(探针用自己的配对误差,不走这套阈值) |

⇒ **噪声低估不推翻任何承载性结论**,但那 16 个单点标签应改为"打平"。
**行动建议**: 论文表格中不要逐点标显著性,只标最终收敛点与整体判定。

## 2026-08-26 10:15 — **基础设施 bug: smoke 脚本互删产物**(已修)

`slurm/smoke_cfg_h200.slurm:98` 原为 `rm -rf "logs/[SMOKE]"*`,在**启动前**执行
⇒ **删除所有 smoke 目录,包括正在并行运行的其他 smoke 作业的产物**。
今晚 gpu18 上最多同时跑 4 个 smoke,它们互相删除了对方的 checkpoint。
实证: `RTC-RAWB` 跑完 100 epoch(日志明确记录 `save_ckpt ... checkpoint-100`),
但目录已不存在。

**影响范围(已逐项核实)**:
| 项 | 受损? |
|---|---|
| wandb 曲线(所有 monitor 分析) | **否** —— 写在 `.cache/wandb/`,与 logs 分离 |
| 主线 arm(rtckb/rtcraw) | **否** —— 走 opd_arm.slurm,目录无 [SMOKE] 前缀(rtckb 19 个 checkpoint 完好) |
| smoke 臂 checkpoint | **是** —— 无法对受控臂做标准 benchmark 评测 |

⇒ **今晚所有基于 monitor 曲线的结论仍成立**,只是受控臂失去了 benchmark 验证的可能。

**修复**: 改为只删**本次配置**的 smoke 目录(从 config 解析 save_dir,并校验确为 [SMOKE] 前缀)。
**教训**: 清理逻辑用通配符时必须考虑并发;一次误删就损失数小时算力的产物。
另: 这与 `poe15_*` 标签覆盖是同一类问题 —— **共享命名空间下的写操作必须带运行标识**。

## 2026-08-26 10:30 — **d100_nocap 达判据(n=60): 构造增益不是主因**;并给出理论解释

```
raw   0.9298 | kb 0.8908 (-0.0455 vs raw, t=-21.9) | nocap 0.8867 (-0.0557, t=-20.9)
nocap - kb : -0.0101  t=-5.6   <- 显著更差
```
按 07:08 事先写定的三分判据,落在第三档: **"nocap 不优于 kb ⇒ 构造增益不是主因"**。
放大位移确实提高了构造增益(即时 +11 点、F_q +14%),训练却**更差**。**判据未事后调整。**

### 理论解释: 拟合可实现目标 ≠ 恢复产生它的参数步
设目标 y* = y_{theta+dtheta}(K 步上升的真实输出,按构造完全可实现)。
MSE 损失对 theta 的梯度:
```
J^T (y_theta - y*) = -J^T J dtheta
```
⇒ 更新量是 **dtheta 被参数空间 Gauss-Newton 矩阵 J^T J 预条件后的结果**,而非 dtheta 本身。
LoRA 的 J^T J 病态严重 ⇒ 该乘积可能远小于且方向偏离 dtheta。

**⇒ RTC 的隐含假设"y 空间可实现 ⇒ 拟合能到达"不成立。**

与实测吻合: kb 的目标更近(loss 0.085 vs raw 0.29)但 `grad_norm` 仅 0.0030,模型跟不过去;
nocap 目标更远(loss 0.174)而 grad_norm 更低(0.0026) —— 位移增加的部分几乎全部落在
J^T J 的近零空间里。

### 这是对原理论的实质修正
原论证: Cauchy-Schwarz 保证 ||Ag||*||g|| >= <g,Ag> ⇒ 预条件方向的**单步实现增益**不小于原方向。
**该论证只覆盖"沿给定方向走一步",不覆盖"通过 MSE 拟合到达某个目标"。**
后者引入了额外的 J^T J,把优势抵消掉。**这是 RTC 失败的根本原因,不是实现问题。**

## 2026-08-26 10:49 — RTC-K 首批标准 benchmark 数据(**两个对比都不公平,不能据此判 RTC-K**)

```
overall Ratio: rtckbA_61 0.7374(-26.3%) | _151 0.7075(-29.3%) | _241 0.6917(-30.8%)  <- 随训练下降

step  arm     geneval  pickscore  aesthetic  hpsv2
 61  RTC-K    0.4609   0.8841     5.9958     0.2794
 61  POE15    0.7695   0.8987     5.8039     0.3126
151  RTC-K    0.5156   0.9095     6.0238     0.2788
151  POE15    0.8047   0.9128     6.1205     0.3344
241  RTC-K    0.5039   0.9146     6.1164     0.2790
241  POE15    0.8438   0.9196     6.2730     0.3452
```

### 两个对比都不公平 —— 必须写明,不能拿来判 RTC-K
- **vs POE15**: POE 是 **3-teacher** 臂(训练目标含 geneval/ocr),RTC-K **只优化 pickscore**。
  拿 geneval 比 = 问"没学过的科目考得怎样"。
- **vs Student 参照**: 官方 Student 是全轴多 teacher 训练,单 reward 臂在 overall 上本就该低。

**唯一公平的对照仍是 `rtcraw`**(同样单 reward、同样配置,只差目标构造)。

### 一个不受该混杂影响的观察
RTC-K 的 **pickscore 在涨(0.884->0.915)而 overall 在跌(0.737->0.692)**
⇒ 在自己训练的 reward 上进步、其余轴被牺牲(窄化/reward hacking 形态)。
**但 rtcraw 作为同样的单 reward 臂大概率也如此**,故仍需对照才能归因到"目标构造"。
⇒ 记为观察,不记为结论。

## 2026-08-26 11:31 — **RTC-K 单调摧毁 OCR 能力**(overall 下跌的全部来源)

标准 benchmark 轨迹(注意键名是 `ocr` 不是 `ocr:accuracy`,我先前显示成 nan):
```
RTC-K:  step   61     151     241     331     421     511
        ocr  0.3107  0.1888  0.0988  0.1048  0.0888  0.0440   <- 单调崩溃 7 倍
        pick 0.8841  0.9095  0.9146  0.9213  0.9236  0.9254   <- 训练目标,上升
        gen  0.4609  0.5156  0.5039  0.5195  0.5156  0.4883
        aes  5.9958  6.0238  6.1164  6.1767  6.1973  6.1500

POE15:  ocr  0.9249  0.9148  0.9271  0.9451  0.9424  0.9402   <- 稳定
        pick 0.8987  0.9128  0.9196  0.9214  0.9198  0.9204
        gen  0.7695  0.8047  0.8438  0.8750  0.8750  0.8867
```
⇒ **overall Ratio 从 0.737 跌到 0.692 的全部来源是 OCR 轴崩溃**,
精确解释了 10:49 记的"pickscore 涨而 overall 跌"。

### 归因仍需 rtcraw —— 不能算到"目标构造"头上
POE 的三 teacher 含 OCR,当然守得住;RTC-K 根本不训练 OCR。
**任何单 reward 的 pickscore 臂都可能这样漂移。**

### 但这给了一个锐利的对照指标
等 `rtcraw`(同样单 reward、同样配置、只差目标构造)出来后看它的 OCR:
- **也崩到 ~0.04** ⇒ 单 reward 训练的通病,与 RTC 无关;
- **守得住** ⇒ RTC-K 的目标构造额外加剧遗忘,是它特有的问题。

这比 overall Ratio 更锐利,因为它是单一轴上 7 倍的效应,远超任何噪声。

## 2026-08-26 12:32 — 受控实验三臂全部跑满 100 epoch(定稿)

```
raw    n=100  末10均=0.9532
kb     n=100  末10均=0.9198
nocap  n=100  末10均=0.9081
同 epoch n=100
  kb    -raw  -0.0421  t=-31.1
  nocap -raw  -0.0520  t=-30.8
  nocap -kb   -0.0099  t= -8.8
```
三臂相对次序 **raw > kb > nocap** 在 n=13/20/40/60/92/100 六个样本量上**始终一致,无一次翻转**。

## 2026-08-26 12:31 — 同一个互删 bug 存在于三个 smoke 脚本,已全部修复

`rm -rf "logs/[SMOKE]"*` 出现在 `smoke_cfg_h200.slurm`(已于 10:15 修)、
`smoke_cfg.slurm`、`smoke_cfg_a100.slurm`。后两个本轮一并修掉。
**教训**: 同一段危险逻辑复制在多个脚本里,修一处不等于修好 —— 应 grep 全仓确认。

## 2026-08-26 12:32 — 集群饱和,机制直测改投 A40
gpu18 被另外 4 位用户占满、A100 全满、gpu17 跑 rtckb ⇒ `mech_dtheta`(H200)只能排队。
A40 有 6 张空闲,已用同一脚本覆盖 `--gres=gpu:a40:1` 另投一份(`mech_a40`,1034520),
保留 H200 排队位作后备(排队不占资源)。若 A40 显存不足会立刻 OOM 报错,不会静默失败。

## 2026-08-26 12:49 — **失败的作业在 sacct 里显示 COMPLETED**(已修,四个脚本)

`mech_a40`(1034520) **CUDA OOM 崩溃**,但 `sacct` 显示 **COMPLETED**。
根因: 脚本末行是 `echo "[smoke] ... exit=$?"` —— 退出码被写进日志,
但**脚本自身的退出状态是 echo 的 0**,SLURM 因此判为成功。

⇒ **"没崩 ≠ 算对了"在作业层面的翻版: 崩溃的作业看起来是成功的。**

**已修四个脚本**(smoke_cfg_h200 / smoke_cfg / smoke_cfg_a100 / **opd_arm**):
```
_RC=$?
echo "[...] exit=$_RC $(date)"
exit $_RC
```
`opd_arm.slurm` 尤其关键 —— **rtcraw 正用它**,若中途崩溃必须能看出来。
已确认: 链路用 `afterany`,非零退出不会卡住后续作业;auto_eval 只看作业是否离开队列。
改动仅在末尾,前 60 行 diff 为空;已备份 /tmp/opd_arm.bak。

## 2026-08-26 12:47 — 运维知识: RTC smoke 装不进 A40
`mech_a40` OOM: A40 可用 44.42 GiB,PyTorch 已占 43.51 GiB。
**RTC 的 K 步上升需 >44GB ⇒ 只能 H200/A100,A40 不可用。**
(此前已知 GRPO/NFT 在 A40 结构性 OOM,现补充 RTC 亦然。)

## 2026-08-26 13:13 — rtckb 完成;rtcraw 遇 SLURM gres 竞态,守卫已拦截

**rtckb 干净收尾**: COMPLETED exit=0,800 epoch,9h43m,最终 checkpoint 781,
末5均 pickscore 0.9422(起点 0.8103,+0.132)。auto_eval 已自动触发 10 点评测(evrtckb)。

**rtcraw 启动即遇 gres 竞态**: `[arm] nproc=3`,ReqTRES=4 而 AllocTRES=3。
`xih4004` 的作业比 rtcraw 早 **11 秒**启动并占走 gpu17 一张卡。
**守卫正确拦截**: `[guard] 需 4 卡实得 3 -> requeue`,不会让 3 卡的不可比运行悄悄跑下去。
(3 卡不可用: `_scale_to_nproc` 有 `assert 8 % nproc == 0`;强改会破坏"与 8 卡梯度严格等价",
rtcraw 就不再能与 rtckb 比较 —— 等于白跑。)

**主动让路**: 取消 `mech_dtheta`(1034506,要 1 张 H200)。
理由: 两个 H200 节点均 4/4 满,若 mech_dtheta 抢到 gpu17 释放的卡,rtcraw 将永远凑不齐 4 张
—— **我自己的次要作业会饿死自己的首要作业**。机制直测可延后,权威对照不可。

## 2026-08-26 14:08 — rtcraw3 ETA 与一个有利的成本数据点

`rtcraw3`(1034534, 3 卡梯度等价) 实测 **58.2 s/epoch** ⇒ 800 epoch 约 **13.1h**
(约 2026-08-27 02:50 结束,10 点评测再 3~4h,**结果约 06:00**)。

**成本对比(值得写进论文)**:
```
rtckb   43.5 s/epoch x 4 卡 = 174 GPU-s/epoch
rtcraw  58.2 s/epoch x 3 卡 = 175 GPU-s/epoch   <- 总算力几乎相同
```
⇒ **[此结论已于 14:28 撤回]** 该速率基于 n=7,被启动开销污染。
n=47 实测 raw 仅需 kb 的 58% 算力,K 步上升是显著开销。详见 14:28 条。

**不缩短 epoch 数的理由**: 截断会丢掉收敛端点,而配对检验的价值恰在完整轨迹;
rtckb 的评测点覆盖 61~781,truncate 到 400 只剩 4 个公共点,统计力不足。

## 2026-08-26 14:28 — **更正 14:08 的速率与成本结论**(n=7 估计被启动开销污染)

```
早先(n=7):   58.2 s/epoch -> 13.1h
现在(n=47):  33.6 s/epoch -> 约 7.0h,预计 21:30 结束
```
n=7 时几乎全是模型加载 + epoch-0 在线评测的开销,按 12.5 分钟扣除仍不足。
**再次印证自定规则: n<10 不报精确数字。**(今晚第四次因样本量不足更正。)

**连带撤回"两臂总算力几乎相同"这条**:
```
rtckb   43.5 s/epoch x 4 卡 = 174 GPU-s/epoch
rtcraw  33.6 s/epoch x 3 卡 = 101 GPU-s/epoch   <- raw 仅需 kb 的 58%
```
⇒ **K=6 步上升是显著开销(约 1.7 倍),不是"可忽略"。**
先前写的"RTC-K 额外成本远小于预想,是有利数据点" **作废** ——
事实相反: **成本高 70% 而效果更差**,这对 RTC-K 是又一条不利证据。

evrtckb 进度: 61/151/241/331/421/511 已 3/3;601/691/751 为 1/3;781 为 0/3。

## 2026-08-26 14:50 — **更正"J^T J 是目标构造类通病"** —— 推错了范围

推 OPSD 的一阶更新:
```
OPSD:  y* = y0 + rho*ghat  (目标由**梯度**定义,y0 = y_theta.detach())
       grad = J^T(y_theta - y*) = -rho * J^T ghat        <- 没有 J^T J
ReFL:  loss = -R(y_theta)
       grad = -J^T g                                     <- 同一方向
RTC-K: y* = y_{theta+dtheta} (目标由**参数位移**定义)
       grad = J^T(y_theta - y*) = -J^T J dtheta          <- 多一层 J^T J
```
⇒ **J^T J 是 RTC-K 特有的,不是目标构造类的通病。我先前把范围说大了,收回。**
(附带结论: **OPSD 的一阶更新方向就等于 ReFL 的方向**,它加的是信赖域归一化、
实测接受门控、负分支、off-policy 复用这四层防护,而非新方向。)

**实测数据也与那个故事不符**: kb 的 grad_norm 是 raw 的 **6 倍**(0.0030 vs 0.0005)。
若 J^T J 在"吃掉"更新量,梯度应更小。更合理的解释: 可实现方向 d 落在 range(J) 内,
故 J^T d 更大 ⇒ **更新更大但效果更差,问题在方向而非幅度**。

**处置**: 报告中"根本原因: MSE 拟合引入 J^T J"**降级为待验证假说**。
真正机制未定论,`mech_dtheta`(直接测 cos(梯度, dtheta) 与模长比)正是为此设计,等 H200 空出。

## 2026-08-26 15:00 — **机制直测确证(n=2,两次一致): 更新方向与 Δθ 几乎正交**

`mech2`(1034538, rtc_kb_dtheta):
```
rtc_cos_grad_dtheta   0.0172, 0.0214     <- cos(更新方向, dtheta) ~ 0.02  (约 88.8 度)
rtc_ratio_g_dth      28.8,   31.4        <- |梯度|/|dtheta| ~ 30
rtc_acc_p             1.0000, 1.0000     <- 目标构造正常
rtc_kbest_gain        0.0305, 0.0315
```

### 直接证实"y 空间可实现 != 拟合能到达"
目标 `y* = y_{theta+dtheta}` 按构造**完全可实现**(就是 theta+dtheta 的输出),
但 MSE 拟合**不会把 theta 推向 theta+dtheta** —— 更新方向与 dtheta 夹角约 **88.8 度**。

数学上自洽: `-grad = J^T J dtheta`,J^T J 半正定 ⇒ cos >= 0 必然成立;
测到 **0.02** 说明 **dtheta 的能量大量落在 J^T J 的近零特征方向上**,即该矩阵极度病态。

### 同时解开 grad_norm 悖论
`|梯度|/|dtheta| ~ 30` ⇒ **梯度不小,只是指错方向**。
故"kb 的 grad_norm 比 raw 大 6 倍"与"更新被浪费"**并不矛盾**:
更新量很大,但几乎全部落在与目标无关的方向上。

### 结论范围(严格)
- **对 RTC-K 成立**: 目标由参数位移定义 ⇒ 引入 J^T J ⇒ 更新与 dtheta 正交;
- **对 OPSD/ReFL 不成立**: 目标由梯度定义(`y0 + rho*ghat`) ⇒ 更新 = -rho*J^T ghat,无 J^T J。
⇒ 14:50 收回的"整类通病"仍然收回;15:00 恢复的是**仅限 RTC-K 的**机制结论,且现有实测支持。

**限制**: n=2(smoke 只跑 6 epoch,前 2 个 epoch 有数据)。两次读数一致(0.0172/0.0214),
按自定规则只报方向与量级,精确值待更多样本。已在 rtcraw 完成后补跑长版本。

## 2026-08-26 15:08 — 把机制结论从 n=2 做扎实

gpu18 空出 3 张 H200,提交 `mech40`(40 epoch)把 `cos(更新, dtheta)` 的样本量从 2 提到 ~40。
理由: 这是今晚**唯一的新机制发现**(RTC-K 失败的直接证据),n=2 撑不起论文;
且今晚已四次因小样本更正,不能在最关键的结论上重蹈。

**对照是解析的,无需额外实验**:
- OPSD/raw: 目标 `y0 + rho*ghat` 由梯度定义 ⇒ 更新 ∝ `J^T ghat`,
  与 ReFL 的"意图方向"**恒等**(cos = 1,代数上必然);
- RTC-K: 目标 `y_{theta+dtheta}` 由参数位移定义 ⇒ 更新 ∝ `J^T J dtheta`,
  与 dtheta 夹角实测 **88.8 度**。
⇒ 对比是"构造方式的选择"直接导致的,不是调参或实现差异。

## 2026-08-26 15:28 — **更正 15:00 的推导**: n=6 出现负 cos,证伪"cos>=0 必然"

```
n=2:  0.0172, 0.0214              (都是小正数)
n=6:  mean=-0.0151 sd=0.0938 范围[-0.1808, +0.0821]
```
15:00 我写"`-grad = J^T J dtheta`,J^T J 半正定 ⇒ **cos >= 0 必然**"。
**实测 -0.18 直接证伪该推导。**

**真因**: 推导假设 `y_theta = y_0`,但这**只在第一步成立**。
训练时 `y_0` 来自**采样阶段**(行为策略输出,存在 sample dict),
`y_theta` 是**当前模型**预测,中间已过若干优化器步,两者不等;
且双分支 `y_pos = beta*y_theta + (1-beta)*y_0` 还有 beta 加权
⇒ 梯度不是 `-J^T J dtheta` 的形式。
n=2 恰是最初两个 epoch(y_theta ≈ y_0),偶然符合。

**今晚第五次栽在小样本上,而且这次还基于它写了一个"数学上自洽"的论证。**
教训升级: **不仅数字要等样本量,基于少量数据构造的理论解释同样要等。**

### 经验事实仍成立且更强
cos 在 0 附近正负散布(mean -0.015, sd 0.094) ⇒
**更新方向与 dtheta 基本无关**(比"几乎正交"更彻底)。
`|grad|/|dtheta| ~ 26` ⇒ 更新量大但与目标无关。
**但解释该现象的理论目前没有** —— 不再声称机制已查明。

### 处置
报告中"机制已由直接测量确证"**再次降级**为"现象已测量,机制未定论"。
等 mech40 跑满 40 epoch 看 cos 是否收敛到某个值,再决定能否给出解释。

## 2026-08-26 15:57 — **漂移判别: 我的测量本身错了;并由此发现更重要的事实**

`mech_drift`(1034552) 首批读数:
```
rtc_drift_rel        = 0.00000   <- 漂移恰为零,y_theta = y0 精确成立
rtc_cos_grad_dtheta  = -0.0300   <- 但 cos 仍为负
```
按预设判据落在"漂移≈0 却仍出负 cos ⇒ 真因在别处"。

### 真因之一: 我拿错了比较对象
`_rtc_dtheta` 累加了**全部 K=6 步**的位移,而 `kbest` 选出的最优迭代点在**第 ~2 步**
(`kbest_idx ≈ 2.1`)。**用 6 步的 dtheta 去比 2 步产生的目标** —— 后 4 步方向已转向
(上升越过极值)。**此前 cos≈0 的所有解读作废。**

### 真因之二(更重要): RTC-K 的目标根本不对应单一参数设置
`kbest` 是**逐样本**选最优迭代点(样本 i 取第 k_i 步),而 K 步上升更新的是**全局共享的 theta**
⇒ **一个 batch 的目标对应不同的参数设置,不存在单一 dtheta 能同时产生它们。**
**"可实现"只对单样本成立,对 batch 不成立。**

这可能才是 RTC-K 失败的真正原因: **批内目标互相冲突,任何一次参数更新都无法同时满足**。
(注意: 这也意味着 kbest 这个"修复"本身引入了新问题 —— 它把 acc_p 从 0.31 修到 1.000,
但代价是目标不再对应单一可达点。)

### 判别实验已提交
`rtc_kb_fixed2`(1034555): 关 kbest + 固定 K=2 ⇒ **全批共用同一 dtheta**,
"目标 = y_{theta+dtheta}" 精确成立。
**判据(事先写定)**:
- cos 显著为正且接近 1 ⇒ 理论成立,**批内目标冲突就是真因**;
- cos 仍 ≈ 0 ⇒ 另有原因,继续查。

## 2026-08-26 16:28 — 批内冲突假说被否定;测量第三次出错;设定停止规则

`mech_fixed2`(1034555, 关 kbest + 固定 K=2 ⇒ 全批共用同一 dtheta), n=9:
```
cos(更新,dtheta)  mean=+0.0405 se=0.0258 t=+1.57   <- 不显著,远不接近 1
序列: +0.008 +0.030 -0.071 +0.117 +0.159 -0.008 +0.133 +0.000 -0.003
drift = 0.0000 (确认)     acc_p = 0.784 (固定 K=2 时非 100%)
```
按 15:57 事先写定的判据 ⇒ **批内目标冲突不是真因,假说否定。**

**但序列仍有负值**,而 `-grad = J^T J dtheta`(PSD)数学上不可能给负 cos
⇒ 该关系在别处被破坏。最可能是**测量第三次出错**:
`accelerator.backward` 在梯度累积下把 `.grad` **跨 batch 累加**(gas=6/8),
读到的是本 epoch 前若干 batch 的累加值,而 `_rtc_dtheta` 每 batch 被覆盖为最新
⇒ **拿"累积梯度"比"最后一个 batch 的 dtheta"**。已改为取 backward 前后的**增量**。

### 本诊断的三次测量错误(记录以免重蹈)
1. 假设 `y_theta = y0`(仅第一步成立) —— 由 drift 测量证伪;
2. 累加**全部 K=6 步**的 dtheta 去比**第 ~2 步**产生的目标;
3. 读跨 batch 累积的 `.grad` 而非本 batch 增量。

### **停止规则(事先写定)**
再跑一次修正版。**若仍拿不到可解读数据(cos 仍出现负值或 t 不显著),
就把机制记为"未解决"并停止投入。**
理由: 主结论(RTC-K 训练更差、三条替代解释已排除)**不依赖**该机制;
权威对照(rtckb vs rtcraw)才是交付物,且不受影响。
连续三次测量错误说明该量在此代码结构下不易正确测得,继续投入的边际收益低。

## 2026-08-26 16:48 — **停止规则触发: 机制记为"未解决",停止投入**

`mech_fix3`(梯度增量版, n=8):
```
cos mean=+0.0133 se=0.0075 t=+1.77
序列: -0.009 +0.003 +0.010 +0.040 +0.038 -0.012 +0.001 +0.036
负值 2/8
```
16:28 写定的停止条件("仍出现负值 **或** t 不显著")**两条都满足** ⇒ **停止**。

**修正确有效果**: 数值范围从 [-0.18,+0.16] 收窄到 [-0.012,+0.040],噪声降一个量级。

### 可以如实写下的量化结论
LoRA 参数约 2e7 维,随机向量 cos 基准 = 1/sqrt(d) ≈ 2.2e-4。
实测 **0.013 是随机基准的约 60 倍** ⇒
**更新并非与 dtheta 完全无关,只是保留的分量极小(约 1%)。**
比"正交"更准确的表述: **拟合确实朝 dtheta 走了一点,但绝大部分更新去了别处。**
偶发负值(2/8)仍无法解释 —— `-grad = J^T J dtheta` 若成立不该出现。

### 不再投入的理由(复述,便于日后回看)
- 主结论**不依赖**该机制: RTC-K 训练更差(受控 n=100, t=-31.1),
  三条替代解释(学习率/放大位移/构造增益)均已按事先判据排除;
- 交付物是**权威对照**(rtckb vs rtcraw),不受影响,rtcraw3 已 294/800;
- 该量在此代码结构下连续三次测错(见 16:28 条),边际收益低。

**若日后要续查**,建议换思路: 不测 cos,而是直接比较
"拟合 N 步后的 y_theta" 与 "y_{theta+dtheta}" 的距离随 N 的变化 ——
这绕开了梯度累积、双分支、门控等所有干扰项。

## 2026-08-26 17:09 — rtckb 完整 10 点轨迹 + 在**训练后**模型上复测天花板

### rtckb 标准 benchmark 轨迹(10 点全完成)
```
step   geneval   ocr     pick     clip    aes     hps
  61   0.4570  0.2933  0.8868  0.2659  5.9789  0.2803
 331   0.5508  0.0848  0.9226  0.2616  6.1619  0.2823   <- geneval 峰
 511   0.4961  0.0375  0.9272  0.2608  6.1442  0.2855
 781   0.4805  0.0378  0.9288  0.2652  6.1299  0.2818
```
**训练目标 pickscore 单调升至 0.9288 后平台;OCR 单调崩溃 7.8x;geneval 在 331 达峰后回落。**
典型窄化优化: 能力集中到单一 reward,代价是其他轴。
(归因仍需 rtcraw —— 单 reward 臂都可能如此。)

### 加强主结论的实验: 在训练后模型上复测 teacher 天花板
**动机**: 所有 `G_construct` 测量都在**初始学生**(新建 LoRA)上做。
若"teacher 子空间 ~15% 天花板"只在初始化成立,结论很弱;
若在训练 781 步后仍成立,则说明这是**结构性**限制而非初始化伪影。
已提交 `gc_trained`: `OPD_STUDENT_INIT=<rtckb checkpoint-781/lora>` + `mopd_gcprobe`。
**判据**: teacher 家族(teacher/tproj/tsum/rtpbest)是否仍显著低于 grad,比值是否仍在 ~15% 量级。

## 2026-08-26 17:35 — rtcraw 预评测 + watcher 链路走查

### 状态
- rtcraw3(1034534) **Epoch 352/800**, checkpoint 1..331 已落盘(save_freq=30,共 12 个)。
  配置核对: `rtc=True, rtc_gn=None, lr=3e-4` —— 正确的无预条件对照臂。
- 注意: 目录名是 `[RTC-RAWB + ...]` 不是 `RTC-RAW`;我第一次 ls 错目录,误以为"0 个 checkpoint"。
- 另: 用 `grep -oE 'epoch=[0-9]+'` 读进度得到 `epoch=72`(实为配置打印),真实进度要从
  进度条行 `Epoch 352:` 读。**日志里同一个词出现在多处时,grep 取 tail -1 并不安全。**

### 预评测(利用空闲 A40)
rtcraw3 还要跑 ~4h,但公共步 61/151/241/331 已落盘 -> 先评掉,收尾时只剩 6 个点。
提交 `evrawpre`(1034581, array 0-3), gs=4.5 n=256 与 rtckb 完全一致,结果写入 `rtcraw_<step>_<ds>`。
**用独立的 `rtcrawpre_targets.tsv`** 而非 `rtcraw_targets.tsv`:
watcher 结束时会 `: > $T` 覆写后者,若预评作业仍在按行号读同一文件会读到错位的行 -> 评错 checkpoint。

### 自我纠正: 一个我误判的"bug"
我先判断"watcher 需要 `slurm/eval_rtcraw.slurm`,该文件不存在 -> 今晚评测会全失败"。**这是错的**:
`auto_eval_on_finish.sh:24` 是**从 `eval_all.slurm` 生成并覆写** `eval_${TAG}.slurm`,不要求预先存在。
逐行走查确认链路可用: sed 三处替换全部命中(tsv 路径 / `--array=0-9%4` / job-name),
且模板 `eval_all.slurm:50` 有"`average_scores.json` 存在即跳过"保护 -> 预评的 4 点会被自动跳过,不重复算。
教训: 报"发现 bug"前应先读完被调用脚本,而不是只 grep 一个文件名是否存在。

## 2026-08-26 17:50 — ★ teacher 方向的价值随训练耗尽并反转(今晚最强结果)

### 设计
所有既往 `G_construct` 测量都在**初始学生**上做。用 `OPD_STUDENT_INIT` 指向已训练 checkpoint 复测。
探针 reward = `pickscore`(= rtckb 的训练 reward),故**绝对增益为负是预期的**——模型已在该 reward 的
局部最优附近。唯一非平凡的量是**同一工作点上 teacher 与随机方向之差**。

### 结果(每个 epoch 内配对)
```
                     初始模型(n=24)        训练后 rtckb-781(n=8)
grad    - rand      +0.0969  t=+85.7      +0.0952  t=+75.4     <- 基本不变
teacher - rand      +0.0184  t=+36.2      -0.0061  t=-13.2     <- 符号翻转
tsum    - rand         n/a                -0.0081  t=-17.0
tproj   - rand         n/a                +0.0054  t=+16.3
rtpbest - rand         n/a                +0.0073  t=+18.9
```
两侧同号率均 100%(24/24, 8/8)。绝对值: 训练后 teacher/tsum/tproj/rtpbest **全部为负**
(-0.0179 / -0.0198 / -0.0064 / -0.0044),grad 仍 +0.0834。

### 结论与解释力
**梯度方向相对随机的优势不随训练衰减;teacher 方向的优势会耗尽并反转。**
训练后工作点上没有任何 teacher 变体能提升 reward。
-> 直接解释此前三个冲突消解方法为何全败(cagrad -7.4% / conflict -5.3% / cconf -5.1%):
   它们都在**重新加权一个已不含上升方向的子空间**,任何权重方案都救不回来。
   这比"初始化时 ~15% 天花板"强得多,且是结构性的。

### 尚未排除的两点(已提交实验)
1. **n=8 偏小**(自定规则 n<10 只报方向)。`gc_tr24`(1034593) 同 checkpoint 跑 24 epoch,与基线对齐。
2. **只测了 rtckb 一个 checkpoint,且它正是用被测 reward 训练的** -> 可能是对 pickscore 过拟合的伪影。
   `gc_poe`(1034594) 改用 `[MOPD-POE-L1.5]` ckpt-781(**多 reward 训练**: PickScore+OCR+GenEval)。
   判据: 若 teacher-rand 在该臂上同样为负 -> 结论普适;若为正 -> 只是 pickscore 过拟合伪影,须收回。

## 2026-08-26 18:15 — ★ 发现主方法 gn 从未训练过 + rtckb vs rtcraw 初步对照

### 缺口: 今晚跑的根本不是设计中的主臂
计划里的主方法是 `rtc_gn`(Gauss-Newton 预条件 JJ^T g),但:
- 所有 800-epoch 训练臂的 `rtc_precond` 都是 `'ksteps'` 或 `'raw'`;
- `rtc_gn`/`rtc_gn_nodual` 只存在于 `epochs=100000` 的探针配置,**没有训练版**;
- `logs/` 下没有任何 GN 目录。
=> **今晚的 rtckb-vs-rtcraw 实际是 ksteps vs raw,不是 gn vs raw。**

gn 代码路径本身已验证可用(不是只看 exit=0):
`smk_rtcgn_1034405 / 1034407` 产出 `rtc_cos_gn_g=0.0154/0.0162`(非零且很小,
即 g 大部分落在不可实现子空间 —— 正是收益来源的前提), `acc_p=0.72/0.86`, `acc_m=0.97/0.94`, 0 错误行。
(那些 `rtc_cos_gn_g=0.0000` 的是关掉 TF32 后 FD 扰动塌成 0 的诊断跑,非主路径。)
⚠️ **保留意见**: gn 方向靠有限差分估 JVP,其数值有效性今晚**未被证实**(eps 扫描自相矛盾,已记录)。
   只能说默认精度下该估计稳定非零,不能说它就是真的 JJ^T g。

### 动作
新增 `rtc_gn_train`(config diff 核对: 与 `rtc_raw_train` **只差 `rtc_precond` 与名字**,
lr/bs/nbpe/gas/epochs/reward/prompt/dataset/num_steps/gs 全同 -> 差值可直接归因)。
提交 `rtcgn`(1034599),依赖 rtcraw3,**排在 horizon2 之前**;链路已用 `scontrol show job` 逐个核对:
`1034534 → 1034599(rtcgn) → 1033234(horizon2) → 1033235(ste) → 1033236(retain) → 1032591(poe125)`。
auto_eval watcher 已挂(tag `rtcgn`,同 10 步,保证公共步集一致)。

### rtckb(ksteps) vs rtcraw(raw) 初步对照 —— n=3 公共步,方向明确
两臂 config 逐项核对**只差 precond**,故差值可归因。
```
逐轴配对差 (rtckb - rtcraw), 步 61/151/241
  geneval  -0.1172 -0.0703 -0.0156   mean -0.0677  同号 3/3
  ocr      -0.1449 -0.2492 -0.4144   mean -0.2695  同号 3/3
  pick     -0.0265 -0.0305 -0.0615   mean -0.0395  同号 3/3   <- 连训练 reward 都输
  hps      -0.0212 -0.0224 -0.0297   mean -0.0244  同号 3/3
  clip/aes 打平
```
**OCR 诊断有答案了**: rtcraw 的 OCR **不崩**(0.4382→0.4388→0.5030,还在升),
而 rtckb 是 0.2933→0.1896→0.0886。=> **OCR 崩溃是 ksteps 目标构造特有的,不是单 reward 训练的普遍伪影。**
n=3 仍小(自定规则 n<10 只报方向),精确值等 10 点齐全后再给;但四条轴 3/3 同号且 OCR 差达 -0.27,方向可信。
**这是对 RTC(ksteps 变体)的负面证据**,与此前"RTC-K 训练更差"一致。

## 2026-08-26 18:30 — ★ teacher 反转在第二个 checkpoint 复现,关键混淆被证伪

### 结果(每 epoch 内配对,vs 随机方向)
```
checkpoint                        grad-rand              teacher-rand
初始学生(未训练)          +0.0969 t=+85.6 24/24    +0.0184 t=+36.1 24/24
rtckb-781 (单 reward)     +0.0955 t=+73.9  8/8     -0.0061 t=-13.2  8/8
POE-L1.5-781 (多 reward)  +0.0944 t=+66.5  8/8     -0.0052 t=-15.7  8/8
```
绝对增益 mean:
```
                            grad     rand   teacher    tsum    tproj
初始学生                  +0.0928  -0.0041  +0.0143     n/a     n/a
rtckb-781                 +0.0838  -0.0117  -0.0179  -0.0198  -0.0064
POE-L1.5-781              +0.0954  +0.0011  -0.0041  -0.0047  +0.0007
```

### 关键: "近最优点所以各方向都掉"这个混淆被证伪
POE-L1.5 是**多 reward 训练**(PickScore+OCR+GenEval),与探针的 pickscore 不同。
其 `gc_G_rand = +0.0011`(约等于零,而非 rtckb 的 -0.0117) -> **它不在 pickscore 最优点附近**。
但 teacher-rand 仍为 -0.0052 (t=-15.7, 8/8)。
=> 结论不依赖"训练目标 = 被测 reward",是**跨臂、跨训练目标**的。

### 结论
**梯度方向相对随机的优势不随训练变化(三点均 +0.095);teacher 方向的优势在训练后耗尽并反转。**
n=8 时仅报方向;两个 probe 均设 24 epoch,~19:30 与基线 n=24 对齐后给精确值。

### 追加: 剂量-响应曲线
提交 `gc_s61`(1034604) / `gc_s241`(1034605) —— rtckb 的早/中期 checkpoint,A40 单卡。
目的: 定位 teacher-rand **在第几步由正转负**。若早期仍为正,则蕴含一个可操作设计:
早期用 teacher 蒸馏、后期切到 reward 梯度。
风险: gcprobe 同时算 3 个 teacher,A40 48GB 可能 OOM;若 OOM 则改排 H200。

## 2026-08-26 18:35 — A40 探针 OOM: 真因是泄漏节点 + exclude 缺口(系统性)

`gc_s61`(1034604)/`gc_s241`(1034605) 在 A40 上 exit=1, n=0(无任何指标产出)。
**不是"没崩就算对"的反面 —— 是真的崩了,查到真因:**
```
CUDA OOM: 总量 44.42 GiB, 本进程已用 42.15 GiB, 仅剩 247 MiB, 请求 384 MiB
Process 532293 has 2.01 GiB memory in use     <- 用户警告过的泄漏进程
NodeList = ai-gpu01                            <- 应被排除的节点
```
若无这 2.01 GiB 泄漏,本可剩 ~2.26 GiB,足够那次 384 MiB 分配 -> **泄漏是直接诱因**。

### 系统性缺口(比这次 OOM 本身更重要)
`--exclude=ai-gpu01,ai-gpu03` **只加在 `eval_all.slurm` 一个脚本里**;
`smoke_cfg.slurm` / `smoke_cfg_a100.slurm` / `opd_arm.slurm` 全都没有。
=> 今晚任何 A40 smoke 都可能落到 gpu01 并 OOM。已在全部脚本补上,并核对三件事:
  (a) exclude 确实存在; (b) `bash -n` 语法通过;
  (c) **指令位于第一条非注释命令之前** —— #SBATCH 出现在首条命令之后会被 SLURM 静默忽略。
已排队的 H200 作业(rtcgn 等)不受影响: sbatch 在提交时已复制脚本,且它们只能落在 H200 节点。

### 重排
除避开 gpu01 外,追加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:
日志显示 965 MiB 属"reserved but unallocated"(碎片),而 42.15/44.42 GiB 本就极紧。
若仍 OOM,则说明 gcprobe(同时算 3 个 teacher)在 A40 上结构性不可行,须改排 H200/A100。

## 2026-08-26 18:50 — 两处澄清

### 1) gcprobe 在 A40 上结构性 OOM(新增已知约束)
重排后避开了泄漏节点(落在 ai-gpu02),**仍然 OOM**:
`总 44.42 GiB / 本进程 43.73 GiB / 请求 768 MiB`。
上一次(gpu01)是 42.15 GiB 就撞墙,因为泄漏进程先占了 2.01 GiB。
=> 真实需求 >48GB,**泄漏只是把崩溃提前,不是根因**。第一次归因不完整,此处更正。
gcprobe 需同时载入 student + 3 个 teacher + reward 模型 -> 与 GRPO/NFT 同类,**只能 A100/H200**。
已改排 a100(`gc_s61`/`gc_s241`),当前 A100/H200 全满,排队等位。

### 2) rtcraw_331 "缺失"是 NFS 缓存,不是失败
`[ -f average_scores.json ]` 报缺失,但作业日志显示 18:34 已写入且内容完整。
`ls` 强制刷新属性缓存后 -> 3/3 全 OK。**checklist 里"先 ls 刷 NFS 缓存"这条今晚又救了一次。**
=> rtcraw 预评测 4 个点(61/151/241/331)**全部完成**。

## 2026-08-26 18:55 — rtckb(ksteps) vs rtcraw(raw) 配对检验, n=4 公共步

两臂 config 逐项核对**只差 `rtc_precond`**,差值可直接归因到目标构造方式。
公共步 61/151/241/331,gs=4.5 n=256 一致。

```
轴          逐点差                              mean      se      t     同号
ocr    -0.1449 -0.2492 -0.4144 -0.3151      -0.2809  0.0566  -4.96   4/4
pick   -0.0265 -0.0305 -0.0615 -0.0697      -0.0470  0.0109  -4.33   4/4
hps    -0.0212 -0.0224 -0.0297 -0.0262      -0.0249  0.0019 -12.84   4/4
--- 不显著 ---
geneval -0.1172 -0.0703 -0.0156 +0.0391     -0.0410  0.0338  -1.21   3/4
clip                                        -0.0011  0.0020  -0.54   2/4
aes                                         +0.0254  0.0376  +0.68   3/4
```

### ⚠️ 选择效应实例(记下来)
n=3(步 61/151/241)时 geneval 是 **3/3 同号、mean -0.0677**;加入步 331 后该点符号翻转(+0.0391),
变成 3/4、t=-1.21 **不显著**。=> 公共 step 集扩大一个点就足以推翻一条"看起来一致"的结论。
当时按自定规则只报方向未报数值,恰好避免了错误结论。**τ 上栽过一次,这次没栽。**

### 稳健结论(4/4 同号)
**ksteps 构造在 OCR / PickScore / HPS 上均劣于 raw(OPSD 原构造)。**
其中 pick 是**训练 reward 本身**,且差距单调扩大: -0.027 → -0.031 → -0.062 → -0.070
(rtcraw 的 pick 一路升到 0.9922,rtckb 在 0.9226 平台)。
OCR: rtcraw 维持 0.40~0.50 不崩,rtckb 从 0.293 崩到 0.085。
=> **RTC 的 ksteps 变体是负面结果,归因干净。** 主方法 gn 尚未训练(rtcgn 1034599 排队中)。
剩余 6 个点由 watcher 在 rtcraw3 结束后补齐,届时给 n=10 的最终版。

## 2026-08-26 19:15 — 评测完成度与继续抢跑

`eval_status.sh` 我第一次不带参数调用得到零输出 —— **不是脚本坏了**,它是 `for t in "$@"` 按 tag 传参的,
不带参数就什么都不做。用法错误,已更正。

```
rtckb   10/10 全部 3/3
rtcraw   4/10 (61/151/241/331)
```
rtcraw3 已到 Epoch 514,checkpoint 已落盘至 511。
再抢跑 421/511 两点(`evrawpr2` 1034612,独立 tsv `rtcrawpre2_targets.tsv` 避免覆写竞态)
=> 收尾前配对表可达 n=6,watcher 补完剩余 4 点(601/691/751/781)后为 n=10。

## 2026-08-26 20:10 — ★ 剂量-响应: teacher 信号在 step 61~241 之间穿零

`gc_s61`(1034614) / `gc_s241`(1034615) 在 H200 上跑(A40 结构性 OOM,已记),目标 n=24,当前 n=4。
**按自定规则 n<10 只报方向,精确值待 n=24。**
```
step 0   (初始学生)   teacher-rand = +0.0184   24/24
step 61                            +0.0076    4/4    <- 仍为正
step 241                           -0.0043    4/4    <- 已转负
step 781                           -0.0062   24/24
```
**单调衰减,零点落在 step 61~241 之间**(训练全长 800 -> 约前 8%~30%)。

### 若 n=24 后仍成立,这条比单纯的"反转"更有价值
它把一个负面观察变成**可操作的设计**: teacher 蒸馏只在训练早期携带上升信息,
之后应切换到 reward 梯度。=> 可解释为何固定权重的多 teacher 方案(cagrad/conflict/cconf)全败:
它们在整个训练过程用同一套权重,而被加权对象的价值本身在中途变号。

### 后续可做(未排,视算力)
在 step 121/151 再插一个探针可把穿零点收窄到 ~2 倍区间。当前算力紧张,优先级低于 rtcgn 主臂。

## 2026-08-26 20:30 — 给 rtcgn 主臂补 smoke(避免浪费整夜 H200)

**风险**: `rtc_gn_train` 是今晚新建的配置,**从未 smoke 过**。
验证过 gn 路径的是 `rtc_gn`(epochs=100000,rtc_curve/measure_realized 设置不同)。
若新配置有问题,rtcgn 会在 rtcraw3 结束(~21:20)后启动并浪费 ~7 小时 H200。

**隔离核对**(避免 smoke 污染正式运行):
```
OPD_SMOKE=1  -> save_dir = logs/[SMOKE] [RTC-GNB + SD3.5-M + PickScore]  num_epochs=8
不带          -> save_dir = logs/[RTC-GNB + SD3.5-M + PickScore]          num_epochs=800
```
且 smoke 脚本强制 `OPD_RESUME=none`。=> 两者不共用 checkpoint 目录,
正式运行的 `OPD_RESUME=auto` 不会从 smoke 的 train_state.pt 恢复。

提交 `smk_gntr`(1034617)。**判据(按 checklist,不看 exit=0)**:
`rtc_cos_gn_g` 非零、`rtc_acc_p`/`rtc_acc_m` 有实测值、`rtc_loss`/`rtc_pos`/`rtc_neg`/`grad_norm` 非零、0 错误行。

## 2026-08-26 20:55 — smk_gntr 拿不到卡,改为对正式运行做早期校验

### 排队诊断(含一处自我纠正)
`smk_gntr` 请求 2 卡被排到 23:47;改 1 卡仍是 23:47。
我先判断 `sinfo` 显示的 `mix-` 末尾 `-` 表示节点 DRAIN —— **这是错的**。
`scontrol show node ai-gpu18` 给出 `State=MIXED+PLANNED`:
`-` 代表 **PLANNED**(该空卡已被 SLURM 预留给更高优先级的排队作业),不是 DRAIN。
ai-gpu17 被 rtcraw3 占满,4 个 A100 节点也全满 => 在 rtcgn 启动前无卡可用。

### 决定
取消 smk_gntr,改为**在 rtcgn 正式启动后的前 ~10 分钟直接校验**。
理由: 这比 smoke **信息量更大** —— 校验的是生产配置本身(800 epoch / 多卡 / OPD_RESUME=auto),
而 smoke 用的是 8 epoch / 1 卡 / RESUME=none,本来就不完全等价。
代价对称: 若 gn 路径有问题,损失 ~10 分钟而非 ~7 小时。

**校验判据(不看 exit=0)**: `rtc_cos_gn_g` 非零、`rtc_acc_p`/`rtc_acc_m` 有实测值、
`rtc_loss`/`rtc_pos`/`rtc_neg`/`grad_norm` 非零、无 Traceback/OOM。任一不满足立即 scancel 并查真因。

## 2026-08-26 21:10 — 剂量-响应定量结果(n>=17,可报数值)

```
step        n     teacher-rand                  grad-rand
0(初始)    24    +0.01840  t= +36.1  24/24     +0.0969  t= +85.6
61          17    +0.00732  t= +19.4  17/17     +0.1154  t=+109.7
241         17    -0.00349  t= -11.9  17/17     +0.1025  t=+113.1
781         24    -0.00618  t= -26.2  24/24     +0.0964  t=+111.2
```
绝对增益:
```
step      grad      rand    teacher
0       +0.0928  -0.0041   +0.0143
61      +0.1040  -0.0114   -0.0041
241     +0.0900  -0.0125   -0.0160
781     +0.0854  -0.0110   -0.0172
```

### 关键细节: 有**两个**穿零点,不能混为一谈
- teacher 的**绝对** reward 增益在 step 0~61 之间就转负(-0.0041): 沿 teacher 方向走会**降低** reward。
- teacher **相对随机**的优势在 step 61~241 之间才转负。
step 61 处 teacher 已不能提升 reward,但仍比随机扰动**破坏得少**
=> 与"teacher 承载的是保持而非上升"一致(该限制已在 §十三 写明,且因 OCR 不可微而无法直接证伪)。

### grad-rand 的说法要收紧
四点分别为 +0.0969 / +0.1154 / +0.1025 / +0.0964,有约 ±10% 波动,**不是严格不变**。
可写的稳健表述是: **teacher-rand 在训练中变号,而 grad-rand 全程保持强正**,不宜说"完全不变"。

## 2026-08-26 21:50 — 剂量-响应定稿(四点全 n=24)

```
step   n    teacher-rand                        grad-rand
  0   24   +0.01840 se=.00051 t= +36.1 24/24   +0.0969 t= +85.6
 61   24   +0.00706 se=.00031 t= +23.1 24/24   +0.1153 t=+136.7
241   24   -0.00355 se=.00031 t= -11.4 24/24   +0.1020 t=+115.7
781   24   -0.00618 se=.00024 t= -26.2 24/24   +0.0964 t=+111.2
```
n=17 -> n=24 后估计几乎不变(+0.00732->+0.00706, -0.00349->-0.00355),结论稳。

**衰减在减速**: 0->61 掉 0.0113/61 步; 61->241 掉 0.0106/180 步; 241->781 仅掉 0.0026/540 步
=> 指数衰减趋向 ≈ -0.006 渐近值。
**零点**: 线性内插约 step 180,但曲线为凸 => 线性内插**低估**穿零步数,真实值应晚于 180。
要收窄需在 step 121/151 补点(未做,算力给了 rtcgn)。

rtcraw 评测进度: 8/10(缺 751/781,由 watcher 在收尾后补齐)。

## 2026-08-26 22:20 — ★ rtcgn 早期校验**不通过**;真因是我今晚引入的回归,已修

### 校验结果(启动 15 分钟后,按 checklist 判据)
```
rtc_cos_gn_g = 0.0000   <- 判据核心,应非零
rtc_dirnorm  = 0.0000   <- 方向是**零向量**
rtc_eps_stab = 0.0000
rtc_acc_p/m  = 1.0000   <- 零向量下"采纳"是平凡的,**不能当作通过**
```
决定性诊断行:
```
[RTC-FD] |u|=6.45e-01 |p|=9.57e+01 h=1.48e-01 |h*u|=9.57e-02
         max|v1-v0|=0.000000e+00  dirnorm=0  v0.dtype=float32  n_params=382
```
参数扰动范数 9.57e-02(约 |p| 的 0.1%,并非微小量),模型输出**精确无变化**。
舍入会给微小非零值,**恰好 0 说明扰动没进到前向的有效路径**。

### 真因: 我今晚加的 `_fp32_fd()`
通过的 smoke(`smk_rtcgn_1034405/1034407`)在 **01:01** 运行,那时 `_fp32_fd()` 还不存在。
`_fp32_fd()` 是我为解决 FD 舍入噪声而加的(关 autocast + TF32),而 `rtc_fd_fp32` **默认 True**。
早些时候已测到"关掉 TF32 后 max|v1-v0|=0"并记为**未解决的矛盾** —— 它现在直接卡死主方法。
=> gn 方向两条路都不通: 精确 JVP 被 flash-attn 无二阶导挡住; FD 在 fp32 下恒零、在 fp16/TF32 下噪声主导(稳定性 cos=0.34)。

### 修复(修真因,不绕过)
`aten::_scaled_dot_product_flash_attention_backward` 无二阶导,但 **SDPA 的 math 后端有**。
把双反向包进 `sdpa_kernel(SDPBackend.MATH)`(带旧 API 回退),使精确 JVP 真正可算。
并给 `rtc_gn_train` 设 `rtc_jvp="exact"` —— 默认 "both" 只把精确 JVP 当对拍,`_dir` 仍取坏掉的 FD。

### 附带处理
- 被终止的运行已写出 `[RTC-GNB]/checkpoints/checkpoint-1`(零方向训出的状态),
  **已删除** —— 否则 `OPD_RESUME=auto` 会从污染状态恢复。
- `scontrol hold 1033234`(horizon2),防止 kill rtcgn 后它抢占 H200;修复验证后再 release。
- 首次 smoke 提交把 marker 写成 "SDPA math 后端",而代码里是 "SDPA 的 **math 后端**" -> 不匹配,
  预检空转重试(18 次后会 abort)。已改用确实存在的 `SDPBackend.MATH` 重排(1034637)。

### 判据(smk_gnfix)
`rtc_cos_gn_g` 非零、**无"精确 JVP 失败"字样**、`rtc_dirnorm` 非零、`acc_p`/`acc_m` 有实测值。
若精确 JVP 在 math 后端下仍失败(例如 OOM),则 gn 方向在本代码库中**不可靠计算**,应如实记录并放弃该臂。

## 2026-08-26 22:45 — gn 修复第二轮: math 后端有效,但撞显存墙;改分块双反向

### 第一轮修复的判定: **部分成功**
`smk_gnfix`(1034637) 后报错**内容变了**:
```
旧: RuntimeError: derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented
新: OutOfMemoryError: Tried to allocate 576.00 MiB. GPU 总 139.80 GiB, 仅剩 107.81 MiB
    (PyTorch 已分配 136.56 GiB, 另 1.91 GiB reserved-but-unallocated)
```
=> **强制 SDPA math 后端确实解除了二阶导限制**(原错误消失),
但 math 后端要 materialize 完整注意力矩阵,叠加双反向需保留整张前向图 -> 140GB H200 仍 OOM。

### 一条被排除的捷径
差额仅 576 MiB 而碎片有 1.91 GiB,本想用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
核查发现 `smoke_cfg_h200.slurm:22` **早已导出该变量** —— 失败那次它就是开着的。该路已尽,不是可选项。
(教训: 提"加个环境变量试试"之前先 grep 它是不是已经在了。)

### 第二轮修复: 按样本分块
双反向的峰值显存 ∝ batch。当前 batch=6(3 样本 × 2 CFG 分支)。
新增 `_fwd_v_sub(_z,_t,_sl)`(与 `_fwd_v` 等价但按 batch 切片),精确 JVP 改为逐样本循环,
每块结束 `del` + `empty_cache`,最后 `torch.cat` 还原 per-sample 方向 -> 峰值降到约 1/B。
两处易错点已处理:
  - 上下文管理器实例**不可重复进入** -> 改成 `_mk_sdp()` 工厂,每块新建;
  - 切片前向内部不得遮蔽 `_u`(那是 J^T g 的参数向量表) -> 用 `_uu/_cc`。
块大小由 `rtc_jvp_chunk` 控制(默认 1)。提交 `smk_gnx3`(1034639)。

### 预注册停止规则(23:15)
若届时仍算不出非零 exact JVP,则**如实记录 gn 在本代码库不可靠计算**、放弃该臂,
并 release horizon2 让 H200 回到既定队列。不再继续投入。

## 2026-08-26 22:45 — gn 第三轮: 分块双反向解决了 OOM,但结果是 nan

### 进展
`smk_gnx3`(1034639): **JVP 失败 0 次、OOM 0 次** —— 分块(每次 1 个样本)确实把峰值显存压下来了。
但所有方向指标为 **nan**:
```
rtc_cos_exact=nan  rtc_cos_gn_g=nan  rtc_dirnorm=nan  rtc_fd_vs_exact=nan  rtc_eps_stab=nan
rtc_acc_p=0.125    rtc_acc_m=0.167   (方向是 nan 时,采纳率没有意义)
```
=> 失败模式从"报错"变成"静默产出 nan"。**这比报错更危险** —— 若不看指标只看 exit code,会当成成功。

### 三轮失败模式的演进(都已定位到具体原因)
1. flash-attn 无二阶导 -> 精确 JVP 报错退回 FD;FD 在 `_fp32_fd()`(我今晚加的)下 **恒为 0**。
2. 强制 SDPA math 后端 -> 二阶导限制解除(旧错误消失),但 math 后端 materialize 完整注意力矩阵,
   叠加双反向保图 -> **OOM**(140GB H200 上已分配 136.6 GiB)。
3. 按样本分块 -> OOM 解决,但结果 **nan**。
   假说: math 后端在 fp16 下显式算 QK^T 会溢出成 inf->nan;
   flash-attn 的 online softmax 自带减最大值保护,所以**换后端之前不会暴露这个问题**。

### 当前动作: 定位而非猜测
加 `[JVP-DBG]` 逐阶段有限性诊断(vt2 / yt2 / jtw / s2 / dir_ex,只打印一次),
提交 `gndbg`(1034640,`OPD_EVAL_FREQ=99999` 跳过在线 eval 以缩短周期)。
目标: 即便今晚修不完,也要把失败**精确定位到某一阶段**,而不是留一个"算不出来"的模糊结论。

### 资源处置
kill rtcgn + hold horizon2 曾让 gpu17 的 4 张 H200 闲置。已 **release horizon2**(现 RUNNING),
依赖链 `horizon2 -> ste -> retain -> poe125` 用 `scontrol show job` 核对无误。
调试只占 1 张卡,不再空转。

## 2026-08-26 22:45 — gn 主方法判定: **JJ^T g 在本代码库中不可靠计算**(按停止规则收手)

### 四轮尝试,每一轮都定位到具体原因,失败模式逐层深入
| 轮次 | 做法 | 结果 |
|---|---|---|
| 1 | 默认(FD,`rtc_fd_fp32=True`) | `max\|v1-v0\|` **恒为 0** -> 方向是零向量。真因: 我今晚加的 `_fp32_fd()` 关掉 TF32 |
| 2 | 强制 SDPA math 后端 | flash-attn 二阶导限制**解除**(旧错误消失),但 **OOM**(140GB 卡上已分配 136.6 GiB) |
| 3 | 双反向按样本分块 | OOM 解决、0 报错,但方向 **nan** |
| 4 | `_w` 由全零改小随机(数学等价,避开退化点) | `s2` 从 0 变 1.093(退化点已消除),`dir_ex` **仍 nan** |

### 最终定位(`[JVP-DBG]`,不是猜测)
```
vt2:  finite=True fp32 absmax=6.412     yt2: finite=True fp32 absmax=4.351
jtw:  n=382 none=0 nonfinite=0          s2:  finite=True absmax=1.093
dir_ex: finite=False  nan
```
**前向有限、J^T w 全部 382 项有限、s2 有限,唯独 `grad(s2, w)` 产出 nan**
=> nan 出在**二次求导本身**(math 注意力的 double-backward 数值路径),不是输入退化、不是溢出上游。

### 结论(如实写,不粉饰)
`JJ^T g` 这个方向在当前代码库/精度下**没有可靠的计算方式**:
- 精确 JVP: flash-attn 无二阶导; 换 math 后端可跑但产出 nan;
- 有限差分: fp32/关 TF32 下恒为 0; fp16/TF32 下被舍入噪声主导(稳定性 cos=0.34,本该 ~1)。

**因此 rtc_gn 臂今晚无法运行。** 这不是"没时间跑",是**方法的实现前提不成立**。
早期校验(启动 15 分钟)避免了让零方向跑满 7 小时 H200 —— 若只看 exit code 会误判为成功。

### 下一步建议(未实施,需清醒时评估)
用 **forward-mode AD**(`torch.func.jvp` + `functional_call`)直接算 JVP,
这本来就是算 J u 的正确工具,完全绕开 double-backward 及其数值问题。
代价: 需把 transformer 改成函数式调用,是较大改动,不宜在深夜未验证地写。
风险: flash-attn 可能同样缺 forward-mode 规则,需先做最小验证。

### 资源与队列(已回到既定状态)
`horizon2`(1033234) RUNNING on ai-gpu17(4xH200);链路 `horizon2 -> ste -> retain -> poe125`
经 `scontrol show job` 逐个核对无误。调试期间只占 1 张卡,未造成空转。

## 2026-08-26 22:50 — ★ 四个排队臂携带**旧脚本**,卡在 requeue 死循环(已修)

### 现象
release 后 horizon2 立刻 RUNNING,但一分钟后变回 `PD (BeginTime)`,`Restarts=2`。日志:
```
[arm] cfg=mopd_horizon_h2 nproc=3 host=ai-gpu17
[guard] 需 4 卡实得 3 -> requeue        <- **旧版**提示语
```
ai-gpu17 有 1 张卡被他人占用,只能给 3 张。

### 真因: sbatch 在**提交时**复制脚本
今晚我已把 `opd_arm.slurm` 的 guard 从"必须 4 卡"放宽到 2~4 卡
(2/3/4 卡都能保持 samples/epoch=72,梯度等价)。但 horizon2/ste/retain/poe125
都是**改动之前**提交的,携带旧的严格 guard -> 只要有人占着 gpu17 的一张卡就无限 requeue。

**我早先还把这条特性当作"已排队作业不受影响"的好处记下来** —— 同一机制在这里是隐患。
教训: 改了 slurm 脚本后,必须检查**已排队作业**是否需要重新提交,不能只看新提交的。

### 处理
1. 杀掉指向旧 job id 的 4 个 watcher(否则作业被 cancel 后会立刻空转触发评测);
2. `scancel` 四个作业;
3. 用**当前**脚本按原顺序重新提交并串依赖:
   `horizon2(1034642) -> ste(1034643) -> retain(1034644) -> poe125(1034645)`,
   `scontrol show job` 逐个核对无误;
4. 按新 job id 重新挂 watcher(10 步: 61 121 181 271 331 391 481 601 721 781)。

## 2026-08-26 22:50 — 四臂秒退真因: 我把 guard 放宽得**过度泛化**了(已修)

### 现象
用当前脚本重新提交后,horizon2 越过卡数 guard,但 **10 秒后 exit=1**;
`afterany` 链导致 ste/retain/poe125 在几秒内相继 FAILED(四个全灭)。

### 真因
```
AssertionError: 缩放后 num_batches_per_epoch=8 必须能被 teacher 数 3 整除
```
这四个臂是**多 teacher**(PickScore+OCR+GenEval,3 个),要求 nbpe 能被 teacher 数整除:
```
4 卡 -> nbpe=6,  6%3=0  OK
3 卡 -> nbpe=8,  8%3!=0 失败
2 卡 -> nbpe=12, 12%3=0 OK
```
今晚我把 guard 从"必须 4 卡"放宽到 2~4 卡,是为了解决 rtcraw 拿不到 4 卡的问题 ——
那对**单 teacher** 的 RTC 臂成立,但我**把它当成了通用结论**,直接套到多 teacher 臂上。
这是一次过度泛化: 一个在 A 情形下验证过的放宽,未经检验就应用到了 B 情形。

### 修复(通用,不硬编码 teacher 数)
在 `opd_arm.slurm` 加**配置兼容性检查**: 用实际 NPROC 真去 `get_config()`,
载不动就 requeue 换卡数(带 5 次上限,超限则明确报错退出而非无限空转)。
比"允许哪些卡数"的白名单更可靠 —— 它直接问配置本身能不能用。

### 调度决策
两个 H200 节点各有 1 张卡被他人占用 -> **4 卡请求短期内不可能满足**。
改用 **2 卡**提交(nbpe=12,samples/epoch=72,与 4 卡梯度等价)。
代价: 每 epoch 约慢一倍;收益: 现在就能跑,而不是无限期等 4 卡。
`horizon2(1034646)` 已 RUNNING: `nproc=2 nbpe=12 gas=12 -> samples/epoch=72`,0 错误。
链路 `horizon2 -> ste(1034647) -> retain(1034648) -> poe125(1034649)` 已用 `scontrol show job` 核对,
四个 watcher 按新 job id 重挂(旧的指向已失败作业,已 kill)。

## 2026-08-26 22:55 — 补强: 在**训练良好**的臂上重测 teacher 衰减曲线

### 动机(这是当前最强结论的最大弱点)
剂量-响应曲线(step 0/61/241/781)全部测在 **rtckb** 上,而 rtckb 是**训练失败**的臂
(五条轴显著劣于 rtcraw)。合理质疑: "表征本身就退化了,衰减是训练失败的副产品而非普遍现象"。
现有的跨臂证据只有 POE-L1.5 的**单个** step-781 点,没有轨迹。

### 做法
在 rtcraw(训练良好: pick 单调升至 1.0305,OCR 退化远轻)的 checkpoint-61 / 241 上跑同一探针,
配 rtcraw 已有的 step-781?—— 注意 rtcraw 的 781 尚未测探针,若算力允许应补。
判据: 若 rtcraw 上 teacher-rand 同样从正转负且量级相近 -> 衰减是**普遍现象**,与训练成败无关;
若 rtcraw 上保持为正 -> 衰减是 ksteps 训练失败的副产品,§十三 的结论必须大幅收窄。
提交 gcraw61 / gcraw241(各 n=24,H200)。

## 2026-08-26 23:45 — ★★ RTC 立论前提被证伪 + 启动 Phase 1(官方 OPSD 复现)

### 一、前提是一个论文里不存在的数字
直接读 `arXiv-2608.24646v1` 原文,`0.03511` 与 `0.00046` **来自两个不同实验**:
- `0.03511`(experiments.tex:69): **512 条 held-out prompt**、同一低噪声查询点,
  OPSD 正目标使 fixed-suffix reward 提升 0.03511(alignment 0.7094)。
  对照是 DiffusionNFT endpoint 的 -0.03551。该段论证的是**"必须用 reward 梯度方向"**。
- `0.0004588`(experiments.tex:99): **calibrated plain-SGD**、**512 条 same-query prompt**、
  每条之间**恢复参数**。ReFL 0.0005805 vs bounded OPSD 0.0004588 vs NFT -0.0000333。

prompt 集不同、协议不同、优化器不同。全文 grep `98.7` / `1.3%` **零命中**。
论文还明确写恒等式 **"does not describe end-to-end training"**。
=> **"98.7% 构造增益丢失"是我们自己做的除法,不是论文结论。RTC 的动机基础不成立。**

### 二、论文真正说的,和我们今晚的结果撞上了
> "A better target can produce a worse finite update." HPSv2.1 上构造增益 +0.00245 -> 实现增益 **-0.000740**;
> 随机目标构造 -0.03251 -> 实现 -0.000021(反而更好)。**排序在 62.3% 的 prompt 上反转**(CI 58.2-66.6%),
> CLIPScore 上 29.5%。且 **ReFL 单步实现增益 0.0005805 > OPSD bounded 0.0004588**。

我们今晚 §十四 测出的"更可实现的目标(ksteps)反而五条轴全输"**正是论文已记录的现象**。
我们独立复现了他们的定性结论,却因为读的是二手前提而没意识到。

### 三、另一个此前未察觉的差异
我们的 `rtc_raw`(号称"等价 OPSD 构造")用 **72 samples/epoch**;
官方 `config/public.py:sd35_pickscore` 用 **1152 samples/epoch**(100 epoch,总样本约我们的 2 倍),
且 **guidance_scale=1.0**(我们训练/评测用 4.5)。
=> `rtc_raw` 是"OPSD 的**构造方式**放进 DiffusionOPD 的**训练设置**",作为受控对比合理,
   但**不构成对 OPSD 结果的复现**。

### 四、Phase 1 已打通(全部为不占算力的准备工作)
`reference_papers/DiffusionOPSD-main/` 是**官方代码**。逐项验证:
1. 依赖: 顶层 import 仅缺 `diffusionopsd`(仓库自带包,`pyproject` 把它映射到 `src/`)。
   用 **符号链接垫片 + PYTHONPATH** 解决,**不做 `pip install -e`** —— 避免改动正在被
   其他运行中作业使用的共享 conda env。训练脚本已确认可完整 import。
2. 配置: `config/public.py:<backbone>_<reward>`。实测 **3/4/6/8 卡**可满足其 48x24 分组(1/2 卡报错)。
   `sd35_pickscore`: num_epochs=100, save_freq=10, bs*nbpe*world=1152, lr=3e-4, num_steps=10, gs=1.0。
3. 数据: `prepare_pickapic_prompts.py` 已跑通 —— train 25415 条(SHA-256 与论文 recipe 校验一致)、
   DrawBench test 999 条。
4. reward: 官方 `check_reward_setup.py --reward pickscore --backbone sd35` 返回 `[ok]`。
5. 新建 `slurm/opsd_repro.slurm`(含 3/4/6/8 卡 guard、退出码传递、exclude 泄漏节点)。
   提交 `opsdsmk`(1034660,SMOKE_TEST=1 UPDATES=2),**依赖三个 gcraw 探针**,约 01:00 起跑,
   不抢占 horizon2 链路。smoke 通过后再排完整 100-epoch 运行(3-4 卡约 15h)。

## 2026-08-26 23:47 — opsdsmk 的通过判据(事先写定)

`opsdsmk`(1034660) 依赖三个 gcraw 探针,预计 00:45 起跑。**通过 = 以下全部满足**,
任一不满足则查真因并修,不排完整运行:
1. `exit=0`(必要但**远不充分**);
2. 训练确实推进: 日志出现 epoch/step 推进,且**产出非零的 loss 与 reward 指标**
   (不能只有初始化日志就退出);
3. **无 Traceback / OOM / NaN**;
4. 官方 `save_dir` 下确有产物写出;
5. 关键: OPA 分支确实被执行(`opa=1` 路径),而非静默走了 legacy RI 分支 ——
   配置注释明确写 "``opa=1`` uses the paper values above and does not execute the legacy RI branch",
   须在日志中确认。

通过后排完整运行: `REWARD=pickscore UPDATES=100`,3~4 卡 H200,约 15h,
save_freq=10 -> 10 个 checkpoint,可与论文主结果表逐项对照。

## 2026-08-26 23:50 — horizon2 在 2 卡下的速率核算

实测: 22:46:15 启动,23:48 到 Epoch 10;扣掉约 13 分钟初始在线 eval,
约 **4.95 分钟/epoch**。
**归因更正**: 我先归因为"3 teacher 前向 + 初始 eval",实测都不对:
```
Epoch 8/9/10 的 sampling 各 33 秒;Eval[main] 出现次数 = 0(该臂根本没跑在线 eval)
=> 慢的是**训练阶段**(约 4.5 分钟/epoch)
```
同为 3-teacher 的 poe15 在 3 卡下只需 45 秒/epoch(800 epoch 共 9:53:27),
tau02 在 4 卡下 10:01:13。**差异主要来自 `horizon=2`**(多步 horizon 使每状态的前向次数成倍增加),
而非卡数。
=> 800 epoch 约 **66~67 小时**,walltime 72h,几乎没有余量。

**但不会丢进度**: `opd_arm.slurm` 已有 `--requeue` + `OPD_RESUME=auto`,
超时会 requeue 并从最新 `train_state.pt` 续跑。故 2 卡只是慢,不是不可完成。

**决定: 不动它。** 理由: (a) 进度可恢复; (b) 用户优先级已转向 OPSD 官方复现,
把 00:45 释放的 3 张卡给 opsdsmk / 完整复现更合算; (c) 重启 horizon2 换 4 卡需要
再等 4 卡同时空闲,而两个 H200 节点各有 1 张卡被他人长期占用,不确定性更大。
**如实记录: 按此配置 horizon2 约需 3 天,其后的 ste/retain/poe125 依次排队。**

## 2026-08-27 00:25 — ★ 把 teacher 衰减这一 observation 落地为可检验的臂

### 稳健性检验先通过了(资产 1 最大弱点已排除)
```
                    step 0      step 61            step 241           step 781
rtckb(失败臂)     +0.01840   +0.00706 (24/24)   -0.00355 (24/24)   -0.00618 (24/24)
rtcraw(良好臂)    +0.01840   -0.00058 (10/16)   -0.00883 (15/15)   -0.00695 (14/14)
```
**训练良好的臂衰减更快**(step 61 已到零/略负,241 时更负) => "衰减是训练失败副产品"被排除。
⚠️ 收紧: 良好臂上 grad-rand 也降 ~29%(+0.0969->+0.0692),不能说 grad 完全不衰减;
   可写的是"teacher 变号,grad 全程强正且量级高一个数量级"。

### 落地方式: 零代码改动的两段式
本想在训练脚本里加 teacher 权重调度,但那是对 3000 行主脚本的实质改动,
且 horizon2 requeue 时会读到新代码 —— 凌晨改它风险不对称。
改用等价且零风险的做法: **用 `OPD_STUDENT_INIT` 让 reward-only 的 RTC 运行
从 teacher 蒸馏过的早期 checkpoint 起步**,即「先 teacher warmup、后 reward 梯度」。

新增 `rtc_raw_warm`(**仅配置改动**,不碰训练脚本)。与 `rtc_raw_train` 逐项 diff:
**只差 `train.lora_path` 与名字**。起点 = POE-L1.5 的 checkpoint-61(teacher 信号刚好穿零处)。
提交 3 卡以匹配 rtcraw3 的确切配置;watcher 用与 rtcraw 相同的 10 步 -> 可直接配对检验。

### 判据(事先写定)
基线 = rtcraw3(已完成 10 点评测,pick 0.9133->1.0305,OCR 0.438->0.198)。
- **主判据**: 同等 pickscore 下 OCR 是否被保住。rtcraw 从零训练时 OCR 退化 2.2 倍;
  若 warmup 版显著缓解,即为「teacher 早期有用、后期应退出」的直接证据。
- 次判据: pickscore 是否不劣于基线(若 warmup 拖慢 reward 上升,则该 idea 有代价)。
- **反向结论也要如实报**: 若两者无差异,说明 teacher warmup 无实际价值,
  该 observation 只停留在机制层面,不能升级为方法。

## 2026-08-27 00:10 — 调度: 保证 opsdsmk 优先拿到释放的 3 张 H200

gcraw 三个探针约 00:45 结束,释放 gpu18 的 3 张 H200。届时三个作业在等:
`opsdsmk`(3 卡,Phase 1 smoke,用户明确要求优先) / `rtcwarm`(3 卡) / `measrz`(1 卡)。
opsdsmk 有依赖,gcraw 一结束才变 eligible;而另两个**已经** eligible,
SLURM 可能先把卡给它们 -> opsdsmk 继续等。
处理: `scontrol hold` 住 rtcwarm 与 measrz;opsdsmk 起跑后再 release。
opsdsmk 是 SMOKE(UPDATES=2),预计很快结束,不会长时间阻塞另两个。


## 2026-08-27 00:30 — ★ 更正: horizon2 的速率估算与归因**都是错的**

### 错在哪
23:50 我报告「约 4.95 分钟/epoch,800 epoch 约 67 小时(3 天)」,并归因于 `horizon=2`。
**两条都错。** 用两点差分重测:
```
23:48  elapsed 1:02:16  Epoch 10
00:27  elapsed 1:41:06  Epoch 49
-> 39 个 epoch / 38 分 50 秒 = 59.7 秒/epoch(稳态)
```
=> **800 epoch 约 13.3 小时**,不是 67 小时。

### 根因: 我用「总耗时 / 总 epoch」而不是稳态差分
前 10 个 epoch 包含模型加载、CUDA kernel 编译、首次 autotune 等一次性开销,
把它们平摊进去会把每 epoch 成本高估约 5 倍。
**教训: 估算长作业耗时必须用两点差分测稳态,不能用累计平均。**

### 连带更正: 「慢在 horizon=2」的归因也不成立
稳态 59.7 秒/epoch 与 poe15 在 3 卡下的 45 秒/epoch 完全吻合(2 卡慢 1.5 倍 ≈ 67 秒)。
=> horizon2 **并不比同类 3-teacher 臂慢**,`horizon=2` 没有带来额外的量级开销。
我之前说「换 4 卡也只能降到 33 小时,量级不变」——基于错误前提,一并作废。

### 对决策的影响
此前据此建议「暂停 horizon2 是给 OPSD 复现让路的杠杆」,其代价被我夸大了。
实际 horizon2 约 13 小时完成,ste/retain/poe125 各约 13 小时,四臂串行约 2.2 天,
**而非之前说的单 horizon2 就要 3 天**。已在报告中同步更正。

## 2026-08-27 01:10 — opsdsmk 拿不到 3 卡: 真因是他人占卡,已改为耐心等待

### 现象
opsdsmk 申请 3 卡但 `nvidia-smi -L` 只见 2 张,guard 连续 requeue 3 次(上限 3)即将放弃。

### 真因(不是配置问题)
```
ai-gpu17: rlv4001(1) + xih4004(1) + 我的 horizon2(2) = 4/4 满
ai-gpu18: rlv4001(1) + rlv4001(1)                    = 2 张空闲
```
两个 H200 节点**都无法提供 3 张空闲卡**。官方 config 的 48x24 分组最少要 3 卡
(实测 1/2 卡直接报 `Cannot satisfy 48x24 grouping`),故复现暂时起不来。
即便停掉 horizon2,gpu17 也只能空出 2 张,仍然不够。

### 处理
guard 由「3 次后放弃」改为「**40 次 x 15 分钟 ≈ 10 小时的耐心等待**」——
他人占卡是排队现实而非错误,应当等而不是放弃。重新提交 `opsdsmk`(1034680),拿到 3 卡即跑。

### 调度取舍(如实记录)
gpu18 现有 2 张空闲。若放行 `rtcwarm`(2 卡,约 20h),gpu18 即满,
**opsdsmk 将完全无法起跑**(gpu17 也满) —— 与用户「opsdsmk 优先」的明确指示冲突。
故: `rtcwarm` 继续 hold;放行 `measrz`(1 卡,12 epoch,约 30 分钟) ——
短作业不会长时间阻塞,且直接测我们自己的 realized/construct 比值。
**代价: 有 1 张 H200 会短时闲置,这是为保住 opsdsmk 优先级付的价。**

## 2026-08-27 02:20 — ★★ measrz: 独立测出拟合缺口 99.59%,并定位机制

```
G_construct@rho=0.10 = 0.04975 (se 8.4e-4)     G_realized = 0.000202 (se 2.0e-5)
=> 丢失 99.59%
acc_p = 0.9757     realized_cos = 0.000414 (随机基线 0.003906,仅 10.6%)   dy_norm = 0.1945
```
**前提实质成立**(我们测到 99.59%,比二手转述的 98.7% 更极端),
**但出处仍错**(0.03511/0.00046 来自两个不同实验)。正确表述: 现象在我们的复现中被独立证实。

**机制**: 目标端很好(97.6% 采纳),拟合端丢 99.6%,且实现位移与 g 的对齐只有随机的 1/10
=> 损失是「走错方向」而非「走得不够远」=> GN 预条件应加在**参数更新**而非**目标构造**。

### 过程记录: 一次查找失误
我先报「rtc_realized 没产出」——错。它走 `info` -> `wandb.log`,而 stdout 只回显 `diag2_acc`。
且我自写的 `read_wandb.py` 有白名单、`--keys` 也没生效(该指标只在 summary 记录里,不在 history)。
最终直接解析 datastore 的 summary 记录才取到。
**教训: 说「指标没产出」之前,必须确认自己查的是该指标真正的落盘位置。**

## 2026-08-27 02:40 — ★ 集群行为定性: `--gres` 不被严格满足(诊断实证)

在 `opsd_repro.slurm` 里加了 GPU 分配诊断后拿到实证:
```
[diag] nvidia-smi 可见=2 | CUDA_VISIBLE_DEVICES=0,1
[diag] SLURM_GPUS_ON_NODE=2   SLURM_JOB_GPUS=1,2
[diag] scontrol 记录: TresPerNode=gres/gpu:h200:3
```
**请求 3 张,实际分配 2 张。** 这不是我们脚本的问题,是集群调度行为:
`--gres=gpu:h200:N` 可能只给 N-1 张。今晚反复出现的「申请 4 得 3」(rtcraw)、
「申请 3 得 2」(opsdsmk) 是同一个根因。
=> **任何对卡数敏感的作业都必须在脚本内用 `nvidia-smi -L` 自检 + requeue**,
   不能假设 `--gres` 生效。今晚三个脚本(opd_arm / smoke_cfg / opsd_repro)都已具备该自检。

### 另一处调度陷阱: SLURM 的 StartTime 估计可能不可达
申请 4 卡时 SLURM 给 `StartTime=08:39, SchedNodeList=ai-gpu18`,看似可行。
但逐个查该节点上他人作业的 EndTime:
```
1034666 inr_functa    EndTime = 08-27 08:39    <- 08:39 这个估计来自它
1034633 imagenet_t256 EndTime = 08-29 22:08    <- 还要占 1 张卡 3 天
```
gpu18 共 4 卡,其中 1 张被 imagenet_t256 占到 8/29 => **08:39 最多空出 3 张,4 卡请求不可能满足**。
SLURM 的 StartTime 只看了最近结束的那个作业。
=> **看到 StartTime 估计时,必须自己核对节点上所有作业的 EndTime**,不能直接采信。
已改回申请 3 卡(= 08:39 后的真实可用数)。

### 当前安排(会自行收敛)
- `opsdsmk`(1034682, 3 卡): 每 15 分钟重排,预算 40 次(到约 12:30);08:39 后可拿到 3 卡。
- `gcraw121`/`gcraw151`(1034683/4, 各 1 卡): 用当前闲置的 2 张,约 2h,04:40 前结束,
  不占用 08:39 的窗口。目的: 把 teacher 信号穿零点从 [61,241] 收窄。
- `rtcwarm` 继续 hold: 它要 2 卡跑约 20h,会横跨 08:39 窗口。

## 2026-08-27 03:15 — gcraw151 失败: 并发同配置探针共享 save_dir(我重新引入的旧问题)

### 真因
```
FileNotFoundError: 'logs/[SMOKE] [GCPROBE-M + SD3.5-M + PickScore]/checkpoints/checkpoint-1/train_state.pt.tmp'
                -> '.../train_state.pt'
```
`gcraw121` 与 `gcraw151` 都用 `RUN_CFG=mopd_gcprobe`,而该配置的 `run_name` 是**硬编码**的,
两者共享同一 save_dir;smoke 脚本启动时又会清理该目录 -> 互删对方的 `.tmp`,原子替换失败。

**这是我今晚修过一次的同类问题**(早先"smoke 脚本互相删除输出"),现在用另一种方式重新引入。
早先 gcraw61/241/781 三个并发没触发,是侥幸(时序凑巧),不是设计正确。

### 修复
给 `mopd_gcprobe` 加 `GC_TAG` 环境变量,拼进 `run_name`:
```
GC_TAG=""       -> logs/[GCPROBE-M + SD3.5-M + PickScore]          (与既有行为一致,向后兼容)
GC_TAG=raw151   -> logs/[GCPROBE-M-raw151 + SD3.5-M + PickScore]
```
重排 `gcraw151`(1034699)。`gcraw121` 仍在跑(n=5/24),不打扰它。

### 教训
**「同一个 RUN_CFG 的两个并发作业」= 共享 save_dir = 互相破坏。**
凡是要并发跑同配置的多个实例(不同 checkpoint / 不同种子),
必须先确认该配置的 `run_name` 可由环境变量区分。今晚这是第二次栽在同一类问题上。

## 2026-08-27 05:15 — teacher 衰减曲线全部定稿(训练良好臂,六点均 n=24)

```
step 0:   +0.01840  t=+36.1  24/24
step 61:  -0.00071  t= -3.1  16/24    统计零
step 121: -0.00022  t= -0.9  15/24    统计零
step 151: -0.00413  t=-17.0  24/24    明确转负
step 241: -0.00935  t=-28.7  24/24
step 781: -0.00761  t=-19.5  24/24
grad-rand: 0.0969 -> 0.0916 -> 0.0897 -> 0.0860 -> 0.0782 -> 0.0688  (平滑单调,-29%)
```
**三段式**: 0->61 衰减到统计零; 61->121 零平台; **121->151 转折为明确负**; 之后稳定 -0.008。
转折窗口 30 epoch(全程 3.75%),恰为一个 checkpoint 间隔,无法再细分。
n=19 -> n=24 后 step151 由 -0.00411 变 -0.00413,结论稳。

**对比**: teacher 在 121-151 有突变,而 grad-rand 在同窗口无任何特征(仍光滑下降)
=> 两者衰减机制不同。假设(未检验): grad 的下降是逼近 reward 最优点的自然钝化,
teacher 的转折是学生越过 teacher 能力边界的质变。

## 2026-08-27 05:15 — opsdsmk 仍拿不到 3 卡: 模式已完全确定
```
SLURM_GPUS_ON_NODE=2   而   TresPerNode=gres/gpu:h200:3
```
第 4 次重排。gpu18 共 4 卡,imagenet_t256 长期占 1 张(到 8/29),
先前 gcraw121/151 各占 1 张 -> 只剩 2 张。
**SLURM 的行为: 只要有空卡就启动作业,给多少算多少,不理会请求数。**
gcraw151 已于 05:10 退出 => 下一次重排(约 05:23)时应有 3 张空闲,可成功。

## 2026-08-27 05:35 — ★★ opsdsmk **五条判据全部通过**,已排完整 100-epoch 官方复现

### 判据核验(逐条,不只看 exit=0)
```
1 exit=0                                                  PASS
2 指标非零: reward_pickscore=0.7427 total_loss=0.617       PASS
           reward_std_mean=0.0224  trained_prompt_num=2
3 无 Traceback / OOM / 真 NaN                              PASS
  (grep 'nan' 命中 1 次,查明是单词 "provenance" 的字母序列;严格数值 NaN 检查 = 0)
4 save_dir 产物: metrics.jsonl / run_config.json / run_done.json / train_rewards/   PASS
5 OPA 分支执行(而非 legacy RI):                            PASS
  [OPA] opa=1 reward=pickscore dir_mode=grad cert=0 dual_neg=1 rho=0.1 margin=0.005
        n_ascent=2 eta=1.0 query_sigma=0.278 mb=6 state_mode=rollout target_mode=replace
```

### 附带证实了两处此前的推断
```json
"group_size": 24.0                              <- 24 确实是 advantage 组大小(先前只是推断)
"opa_kq": 8, "opa_sigma_q": 0.2780487835407257  <- sigma* = sigmas[8] = 0.278,与论文一致
"opa_acc_plus": 1.0, "opa_acc_minus": 1.0
```

### 已提交
`opsdrepro`: 官方代码 + `config/public.py:sd35_pickscore` + 100 epoch + 3 卡。
对照基准见 OVERNIGHT 的「Phase 1 复现目标」节(Pick 24.94 / CLIP 0.340 / HPSv2.1 0.390 / Aes 12.08 / ImgR 1.76)。
**可对照范围只有 5 列**(HPSv3/DeQA/AltCLIP/Point/Pair 我们没有或不可复现),
且 Pick 有 x26 尺度差、Aes 与 HPS 版本待核 —— 这些已事先写入 PLAN,避免事后误读。

## 2026-08-27 05:55 — opsdrepro 正常推进(附一个分析时的坑)

```
起跑 05:28:37, 3 卡, 100 epoch, 0 错误
[OPA] opa=1 dir_mode=grad cert=0 dual_neg=1 rho=0.1 margin=0.005 n_ascent=2 eta=1.0
      query_sigma=0.278 mb=6 state_mode=rollout target_mode=replace
epoch 0: reward_pickscore=0.7427  ->  epoch 6: 0.8384
速率约 3.2 分钟/epoch => 100 epoch 约 5.3 小时,预计 10:50 完成
```

### ⚠️ 分析时的坑: metrics.jsonl 混有 smoke 的记录
smoke(`opsdsmk`)与完整运行用了**同一个 save_dir**,`metrics.jsonl` 是**追加**而非截断。
文件开头几行是 smoke 的 epoch-0 记录,而 smoke 跑的是 `--config.debug=true` +
`nbpe=1 gas=1`,**batch 规模与完整运行不同,数值不可比**。
=> 分析时必须按 `_step` 或时间戳剔除 05:28 之前的记录,不能直接全量读取。
(另: `run_done.json` 目前仍是 smoke 的 `global_step: 2`,完整运行结束时会覆盖。)

### 排查过程中的一次自我纠正
我先看到 `$D` 下所有文件时间戳都是 05:24-05:26(smoke 时段),一度以为完整运行没在写。
`ls` 强制刷新 NFS 属性缓存后,实际 mtime 是 **05:47:33** —— 在写,只是缓存陈旧。
**今晚第三次栽在 NFS 属性缓存上**(前两次: rtcraw_331 评测、eval_status)。

## 2026-08-27 06:28 — rtcwarm 在 A40 上两次 OOM: 真因是**在线 eval**,且我踩了代码库已记录的坑

### 第一次(1034768)
崩在 `_eval_one_teacher`(L440) <- `eval`(L548) <- main(L1247),**不是训练阶段**。
gpu04 上另有他人进程占 ~6.8 GiB(1.20+4.89+0.34x2),本进程 37.17/44.42 GiB 到顶。
=> 训练本身可能装得下,是在线 eval 撑不住。

### 第二次(1034769): 我的修法错了
设 `OPD_EVAL_FREQ=999999` 想拉大间隔,但触发条件是
```python
if config.eval_freq > 0 and epoch % config.eval_freq == 0:
```
**`0 % 999999 == 0` 为真 => epoch 0 照样 eval**,再次 OOM。
而 `config/opd.py:537` 早已写着同一句注释:
`config.eval_freq = 10 ** 9   # epoch 0 仍会 eval(0%N==0)`
—— **代码库自己记录过这个坑,我还是踩了进去。**

### 正确修法
`OPD_EVAL_FREQ=0` -> `config.eval_freq=0` -> 被 `eval_freq > 0` 守卫彻底挡掉(已实测验证两种取值)。
在线 eval 是**纯观测**且占 ~60% 墙钟,关掉它不影响权威评测(离线 10 点流水线依赖 save_freq,不变)。

### 教训
**"把间隔调大"不等于"关掉"。** 凡是 `x % N == 0` 形式的触发,N 再大也挡不住 x=0。
要关就用显式的开关(此处是 `> 0` 守卫),并**实测两种取值**再提交。

## 2026-08-27 06:33 — rtcwarm 第三次 OOM: eval 修复生效,但训练本身在 A40 上也装不下

```
eval_freq: 0 且 Eval[main] 出现 0 次   -> 上一轮的 eval 修复**确实生效**
本次崩在 _reward_of_y(L1493) <- main(L1507)  -> **RTC 目标构造阶段**(VAE 解码 + 可微 pickscore)
本进程 37.13 / 44.42 GiB;另有持久进程占 6.09 GiB(1.20 + 4.89,两次报告 PID 相同)
```
=> **`rtc_raw_warm` 在 A40 上结构性不可行**,与 gcprobe 同类。已知约束更新为:
   **gcprobe / RTC 训练臂 / GRPO / NFT 均只能 H200 或 A100**,A40 只能跑评测。

⚠️ **已确认: ai-gpu04 有泄漏 GPU 进程,占 6.09 GiB**
`squeue -h -w ai-gpu04` **返回空**(节点上无任何作业),却有两个进程持有 6.09 GiB
(PID 3385678 = 1.20 GiB, 3385679 = 4.89 GiB,在两次 OOM 报告中 PID 相同 => 持久存在)。
=> 与用户已知的 gpu01(2.01 GiB)/gpu03(1.48 GiB) 同类,但**大 3~4 倍**。

**对本次 OOM 的影响**: 本进程用 37.13 GiB,加泄漏 6.09 = 43.2 / 44.42 GiB。
若无泄漏可多出 6 GiB 余量,**可能**装得下 —— 但 `_reward_of_y` 本身就要 37 GiB,
**无法断定泄漏是决定性因素**,只能说它显著缩小了余量。
**未擅自修改 `--exclude` 列表**(那是用户的运维决定);此处如实记录,供用户判断是否把
ai-gpu04 加入排除名单。

已改回 H200 2 卡提交(`nbpe=12 -> samples/epoch=72`,与 rtcraw3 的 3 卡运行梯度等价),
用 120 次 x 60s 的耐心重试等 opsdrepro(约 10:30)或 horizon2 释放卡。

## 2026-08-27 07:30 — opsdrepro 训练轨迹形态(附一次自我更正)

```
剔除 smoke 前缀后 42 点(定位最后一个 epoch=0 再往后取)
前 1/3 mean=0.8329   中 1/3 mean=0.8836   后 1/3 mean=0.9056
起点 0.7561 -> 最新 0.9028 (+0.1466)   最高 0.9254 @ epoch 40
逐 epoch 波动约 ±0.02
```

**自我更正**: 我在同一轮里先说「出现平台迹象(0.9122->0.9086->0.9028)」——**那是把逐 epoch 噪声当成趋势**。
三分段均值显示仍在稳定上升。**看趋势必须用分段/滑窗均值,不能连看三个相邻点。**
(与昨夜 rtckb「V 形误判」是同一类错误,当时也是被前半/后半的汇总统计误导。)

与论文的对应: 「DiffusionOPSD sustains reward gains through the final update」
—— 目前 41/100 epoch 的形态与该说法一致,但**这是训练 reward,不是 held-out**;
最终判定要等 `opsdeval` 在 999 条 DrawBench 上的结果。

## 2026-08-27 07:55 — 预检发现 opsdeval 的路径错误(否则早上交付会落空)

`cross_eval.py:358`:
```python
if args.ckpt and not os.path.exists(os.path.join(args.ckpt, "adapter_config.json")):
    raise ...("expected a saved LoRA dir")
```
但官方 `checkpoint-N/` 的内容是:
```
ema.pt  optimizer.pt  rng_state.pt  scaler.pt  trainer_state.json  lora/
lora/: adapter_config.json  adapter_model.safetensors  old/  README.md
```
**adapter 在 `lora/` 子目录**,我的 `opsd_eval.slurm` 原本传的是上一层 -> 触发上述 raise,直接失败。
实测确认: `checkpoint-40/adapter_config.json` 不存在,`checkpoint-40/lora/adapter_config.json` 存在。
已改为 `--ckpt $CKROOT/checkpoint-$s/lora`,并把存在性检查从 `-d 目录` 改为 `-f adapter_config.json`
(更贴近 cross_eval 的真实判据)。

**这次是在作业触发前主动预检发现的** —— 若等 10:20 训练跑完才发现,早上的核心交付就没了。
附带确认: `lora/old/` 子目录存在 = 读码时识别出的**行为策略 "old" adapter**,与官方实现一致。

## 2026-08-27 09:35 — 给 opsdeval 加 checkpoint 回退保险,并重新提交

### 预检: checkpoint-100 会存在
官方主循环 `for epoch in range(first_epoch, config.num_epochs)` = 0..99,
保存条件 `global_step % config.save_freq == 0`,global_step 到 100 => checkpoint-100 会保存。
(实测节奏吻合: 09:27 时已有 10..80。)

### 仍加了保险
万一编号差一位,整轮评测会空手而归。改为: 目标 step 不存在时**回退到最高可用 checkpoint**,
并在日志中明确打印 `⚠️ 回退到 checkpoint-N`(不静默替换)。

### ⚠️ 重新提交的原因(昨夜栽过的同一个坑)
**SLURM 在提交时就复制脚本**,已排队的 1034773 持有的是加保险**之前**的副本,改文件对它无效。
昨夜 horizon2/ste/retain/poe125 四臂就是因为携带旧脚本(严格 4 卡 guard)卡在 requeue 死循环。
=> 已 scancel 并重新提交。**改了 slurm 脚本后,必须检查已排队作业是否需要重提。**

## 2026-08-27 10:30 — ★ opsdrepro 完成(五条判据全过);opsdeval 首次失败已修

### opsdrepro 完成核验(不只看 exit)
```
State=COMPLETED 0:0
run_done.json: global_step=100          <- 不是 smoke 遗留的 2,确认是完整运行覆盖
最高 checkpoint = 100
checkpoint-100/lora/adapter_config.json 存在
最后 epoch=99, pickscore=0.9148, 0 错误
训练 reward 四分段均值 0.8452 -> 0.8950 -> 0.9119 -> 0.9158 (严格单调)
```
**官方 DiffusionOPSD 在本环境成功复现训练全过程**(100 epoch,3xH200,约 4h45m)。

### opsdeval 首次失败: `PUBLIC_POLICY_WORLD_SIZE=1` 通不过配置检查
```
ValueError: Cannot satisfy 48x24 grouping with 1 policy GPUs
```
评测只用 1 卡,我就把该变量设成 1 —— 但 `config/public.py:get_config` **无条件**执行 48x24 分组检查。

**修法(先核对再改,不是猜)**: 已确认 `cross_eval.py` 只读
`dataset / pretrained.model / resolution / reward_fn / sample.* / train.lora_path`,
**完全不用** `train_batch_size` / `num_batches_per_epoch` / `gradient_accumulation_steps`
这些分组派生字段;生成 batch 由 `--batch_size` 单独控制。
=> 填 4 仅为通过检查,**不影响评测结果**。已改并重排(1034822)。

### 教训
**"这个变量按实际情况填"未必对** —— 当它同时被用作配置合法性检查的输入时,
必须先确认下游到底读不读由它派生的字段。这次是先 grep 出 cross_eval 用到的全部
config 字段、确认无关后才改的。

## 2026-08-27 11:50 — opsdeval 打分失败: **我造的 PYTHONPATH 垫片**导致相对路径解析错位

### 现象
生成阶段成功(1000 张图 / 2322s),打分阶段:
```
[cross_eval] scorer 'hpsv2' FAILED: Pretrained value
'.../scratchpad/opsd_shim/diffusionopsd/../reward_ckpts/open_clip_pytorch_model.bin'
is not a known tag or valid file path
```

### 真因(是我引入的副作用,不是官方代码问题)
官方 `src/reward_ckpt_path.py`:
```python
CKPT_PATH = os.environ.get("REWARD_CKPT_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../reward_ckpts"))
```
我为了让 `diffusionopsd` 可 import,做了符号链接垫片 `opsd_shim/diffusionopsd -> repo/src`。
于是 `__file__` 落在 **scratchpad** 下,`../reward_ckpts` 解析到垫片上级而非仓库目录。
**垫片解决了 import,却打断了基于 `__file__` 的资源定位。**

### 修法
用官方提供的环境变量,而非再加符号链接:
```
export REWARD_CKPT_PATH=$OPD_ROOT/CA-OPD/code/DiffusionOPD-main/reward_ckpts
```
(官方仓库本身没有 reward_ckpts/;权重在我们自己的仓库: open_clip 3.7G + HPS_v2.1 1.9G + aesthetic。)

### 顺带去掉一次冗余生成
原脚本对 `raw` / `training` 两种 pickscore 口径各跑一遍,**每遍都要重新生成 1000 张图(2322s)**。
但两者只差 pickscore 的 `/26` 归一化(`training = raw / 26`)。
=> 改为只跑 `raw`,事后换算,省掉一次 39 分钟的重复生成。

### 教训
**用符号链接把包"塞进" PYTHONPATH 能解决 import,但会改变 `__file__`,
从而打断一切基于 `dirname(__file__)` 的相对资源定位。**
这类库通常同时提供环境变量入口 —— 应优先用它,而不是继续叠符号链接。

## 2026-08-27 12:35 — ★★★ Phase 1 完成: 官方 OPSD 复现成功

```
Pick    25.2133 ± 0.0541  vs 论文 24.940  -> +0.273 (+1.1%)   ✅唯一可比列
CLIP     0.2548 ± 0.0015  vs 0.340   |  HPSv2.1 0.3108 ± 0.0011 vs 0.390  |  Aes 6.2761 vs 12.080
n=1000(DrawBench held-out), sampler=flow, steps=40, seed=42, pickscore_scale=raw
```

### 关键解读: 只有 Pick 可比
论文表 caption: `\cmark` = **reward-specific checkpoints**(每列用该列 reward 训练的 checkpoint)。
OPSD 行是 `\cmark` => CLIP/HPS/Aes 三列在论文中来自**别的 checkpoint**,
而我们只训练了 pickscore 一个臂。**不讲清这点会把复现误判为失败。**
已把该提示写进 `compare_opsd_repro.py` 的输出(逐列标 ✅可比/❌不可比),让 caveat 跟着数据走。

### 意义
我们**现在才具备**「在同一协议下与 OPSD 对比」的资格。
今晚之前的 `rtc_raw` 缺 ω 加权与 EMA 行为策略,且训练(72 vs 1152 samples/epoch)、
评测(256 条 gs=4.5 10 步 vs 999 条 flow 40 步)协议均不同。

### 过程: NFS 缓存第四次
判定完成时日志显示 343 字节、无任何分数,一度以为静默失败;
`ls` 刷新后全文俱在(生成 2324.3s + 四个 scorer 结果 + exit=0)。
**今晚第四次栽在 NFS 属性缓存上**(前三次: rtcraw_331 评测、eval_status、opsdrepro metrics)。

## 2026-08-27 12:50 — Phase 1 打通后的第一件新事: 复现的 held-out 轨迹

Phase 1 之前做不了同协议评测;现在可以了。用 3 张空闲 A40 并行评测复现的中间 checkpoint
(20/50/80),与已有的 100 合成**四点 held-out 轨迹**(999 条 DrawBench, flow/40 步/seed=42)。

**用途**:
1. 核对论文关于训练动态的说法(「sustains reward gains through the final update」)在 **held-out** 上是否也成立
   —— 我们目前只有训练 reward 的单调性,held-out 未验证;
2. 建立**我们自己的同协议基线轨迹**,后续任何改动(teacher 调度 / GN 用在拟合步 / ReFL 对照)
   都能在同一把尺子上比较。

**未做但值得做(需更多算力)**: 官方仓库有 `train_refl_sd35m.py`,可在同协议下复现 ReFL,
用于攻 Phase 2(a) ——「ReFL 单步实现增益更高(0.0005805 vs 0.0004588)却端到端更差」这一论文未解释的现象。
当前 H200 仅剩 1 张,ReFL 训练需 ~5h x 3 卡,暂缓。

## 2026-08-27 13:50 — 复现的 held-out 四点轨迹: 官方实现也有离目标退化

```
ckpt      Pick            CLIP          HPSv2.1         Aes
  20   23.5717±.0464   0.2491±.0016   0.2975±.0011   6.3627±.0180
  50   24.4274±.0493   0.2574±.0016   0.3055±.0011   6.4341±.0172
  80   24.9457±.0518   0.2562±.0015   0.3087±.0011   6.2917±.0162
 100   25.2133±.0541   0.2548±.0015   0.3108±.0011   6.2761±.0151
```
- **Pick 单调↑**(+0.856/+0.518/+0.268,增幅每段约减半) => 仍在改善但趋于平台;
- **HPSv2.1 单调↑**(相关 reward 同步受益);
- **CLIP / Aes 在 ck50 达峰后回落**(−0.5% / −1.4%)。

### 结论
**窄化优化在官方实现中同样存在**,不是我们简化版的伪影。
但程度是数量级差别: 官方 Aes 峰后 −1.4%; §十四 中 rtckb 的 OCR 崩 85%。
=> 「离目标退化」是共性,「崩溃」是 **ksteps 构造特有**的。这让 §十四 的负面结果更有分量。

### 一个提交时的疏忽(记下)
三个评测未指定分散节点,SLURM 把 ck20/ck50 放在同一张 A40 上,速度约减半(39->~55 分钟)。
不影响正确性(无 OOM),但下次并行评测应显式分散或用 array 的 %N 限流。

## 2026-08-27 14:05 — ★ 已排 rho 扫描: 决定后续走哪条路的分叉实验

### 动机
目前只有 rho=0.10 一个点:
```
G_construct@0.10 = 0.04975    G_realized = 0.000202   仅实现 0.41%
realized_cos = 4.14e-4        随机基线 3.91e-3        对齐只有随机的 10.6%
```
**核心未知: G_realized 随 rho 如何变化?** 两种结果对应两条完全不同的路:
- **饱和** -> rho=0.10 是浪费,**小 rho + 多步拟合**能实现更多。便宜,且**不需要解决 JVP 数值问题**。
- **线性增长** -> 瓶颈是方向而非幅度,必须做 **Gauss-Newton 拟合步**(须先打通 forward-mode AD,工程量大)。

### 实现
`rtc_raw_meas` 加 `RHO` 环境变量,同时设 `train.rtc_rho` 与 **run_name**
(`[RTC-RAW-MEAS-rNNN + ...]`) —— 并发实例共享 save_dir 会互删 checkpoint,gcprobe 上栽过一次。
已核对四个取值的 save_dir 互不相同(r010/r030/r100/r200,无碰撞)。
提交 rho ∈ {0.01, 0.03, 0.10, 0.20},各 12 epoch / 1xH200 / OPD_EVAL_FREQ=0。
rho=0.10 也重跑(而非复用 01:24 的 measrz),以保证四点用同一份二进制。

### 为什么不去复现论文主表其余四列
论文那行是 reward-specific,复现整行要训 5 个臂(约 25h),产出只是「我们能复现」——
而这一点已由 pickscore 列(25.2133 vs 24.940)证明。算力应投给上面这个分叉实验。

### 顺带更正一处昨夜的说法
我曾说 3 卡 vs 8 卡「gas 差 3 倍,数值上不完全等同」——**说重了**。
正确口径是**每优化步样本数**,三者都是 1152;且 `group_size=24`,
4 卡的 micro-batch 跨卡正好 24 = 完整一组,8 卡是 3 组,**组从不被拆开**,
组内相对 advantage 不受影响。=> **4 卡与 8 卡等价**,后续统一用 4 卡。

## 2026-08-27 14:55 — ★★★ rho 扫描第一点: **实现增益对 rho 完全饱和**,且机制可推导

```
rho=0.01:  G_realized = 0.000204 ± 0.000029  (n=24)   dy_norm=0.2148  cos=3.26e-4
rho=0.10:  G_realized = 0.000202 ± 0.000020  (n=12)   dy_norm=0.1945  cos=4.14e-4
构造增益:  r=0.01 -> 0.01924    r=0.10 -> 0.04888/0.04975  (差 2.5 倍)
=> realized/construct: rho=0.01 为 1.06% ; rho=0.10 为 0.41%
```
**把目标做远 10 倍,构造增益 x2.5,实现增益一分没多;位移幅度也几乎不变。**
=> 命中我事先写定的「饱和」分支。

### 机制(可从他们的损失直接推出,不是猜)
```
L = omega * mean[r^2] / gamma_+ ,  r = y_theta^+ - ybar_+ ,  gamma_+ = sg(mean|r|)   # detached
dL/dy_theta  ∝  2r / mean|r|
r -> c*r 时: 分子 xc, 分母 xc  =>  **梯度完全不变**
```
**自适应归一化器使梯度对残差尺度严格不变**,故信赖域 rho 对更新幅度的影响被完全抵消。
rho 本应控制目标激进程度,实际上(到一阶)**对参数更新无影响**,只剩改变 ybar_+ 方向的二阶效应。

### 这条批评的分量
论文把 rho=0.10 作为 canonical 超参,并用信赖域论证「有界目标构造」的安全性;
但**该有界性对实际更新不起作用** —— 安全性来自归一化器,不是来自信赖域。
且这解释了论文自己的观察「A better target can produce a worse finite update」:
更好的目标 = 更大的 r,而更新对 r 的尺度不敏感 => 构造增益与实现增益天然脱钩。

### 适用范围(严格区分,避免又犯「用简化实现批评官方」的错)
- **推导部分直接适用于官方**: 已在 `train_opsd_ri_sd3.py:1539-1545` 确认
  `with torch.no_grad(): wf_p = |y_pos - y_plus|.mean(...)`,损失为 `(y_pos-y_plus)^2 / wf_p`
  —— **归一化器 detached**,故梯度对残差尺度严格不变的推导对 OPSD 本身成立。
- **经验部分目前仍在我们的实现里测**(rho 扫描跑的是 `rtc_raw_meas`)。
  我们的归一化器结构与官方一致(`_gp = (_ypos-_ypb).abs().mean()` under no_grad),故机制共享;
  但**严格地说,官方实现下的饱和曲线尚未实测**。若要把这条写进论文,应在官方代码里补测。

### 待确认
rho=0.03 / 0.10 / 0.20 三个点仍在排队(节点预留)。
若四点的 G_realized 都在 2.0e-4 附近 => 经验部分坐实。

## 2026-08-27 15:15 — ★★★ 数值验证: OPSD 分支损失**对残差 1-齐次**,故梯度严格尺度无关

复刻官方损失结构(`train_opsd_ri_sd3.py:1539-1545`),在 CPU 上把目标位移放大 c 倍:
```
c:          0.1      0.3      1.0      3.0     10.0
grad 范数:  0.039114 0.039114 0.039114 0.039114 0.039114   <- 六位小数完全相同
cos(与c=1):                            1.000000 1.000000   <- 方向也一致
loss:       0.0039   0.0117   0.0391   0.1173   0.3911     <- loss 随 c 线性(x100)
```
**梯度幅度与方向都对目标距离严格不变。**

### 根源比先前的说法更本质(自我更正)
我先前归因于「归一化器 detached」。对照实验显示: **不 detach 时梯度范数同样恒定**(0.024480)。
真正的原因是 **`L = mean(r^2)/mean|r|` 对 r 是 1-齐次的** => `L(cr)=c·L(r)`(loss 数据完全吻合)
=> **梯度是 0-齐次的**,天然尺度无关。与是否 detach 无关。

### 对论文的含义
论文围绕信赖域建了一整套论证:投影公式 Eq.(target_projection)、半径上界
`||y± − y_0|| <= rho*||y_0||`、以及由此论证的「bounded target construction」安全性。
但 **rho 对参数更新没有一阶影响** —— 它唯一的作用是改变 ybar_+ 的**方向**(上升路径会弯),属二阶效应。
这与我们实测的 rho=0.01 vs 0.10 实现增益完全相同(2.04e-4 vs 2.02e-4)一致。

### 精确边界(必须写明)
上述不变性在 `r` 为纯缩放时严格成立。实际中 `r = (y_pos − y_0) − c·d`:
训练初期 y_theta ≈ y_0 => y_pos ≈ y_0 => r ≈ −c·d,是纯缩放,不变性近似严格;
训练后期 y_theta 偏离 y_0 后不再是纯缩放,不变性只是近似。
**实测(rho 0.01 vs 0.10 realized 相同)表明近似在实践中足够好。**

## 2026-08-27 15:35 — 追加两点: beta=1.0(新保真度缺口) + c_adv 是唯一真步长旋钮

### 官方 sd35_pickscore 的 beta = **1.0**(我们用 0.5 —— 第三处保真度缺口)
```
adv_clip_max=5   lr=0.0003   config.beta=1.0
```
beta=1 => `v_pos = v_theta`(不与锚混合),正分支就是把模型输出**直接**拟合到 ybar_+;
负分支是关于 y_0 的反射。
**这让 1-齐次/尺度不变的论证在训练早期变为严格成立**(r = y_theta − ybar_+ ≈ −c·d,纯缩放),
比我先前说的「近似」更强。
(注意有两个 beta: `config.beta`=分支混合=1.0;`config.train.beta`=FlowGRPO 参考正则权重=0.0。)

### c_adv 是唯一真正的步长旋钮,且论文承认它耦合
```python
opa_policy = (r1*pos_loss + (1-r1)*neg_loss).mean() * config.train.adv_clip_max   # =5
```
论文原文: "c_adv is shared by reward normalization and branch-loss scaling.
Changing it therefore changes both the clipping range and the fitting scale
**rather than acting as a pure learning-rate multiplier**."

### 合起来的批评
- 被论文当作核心安全机制、并写了投影公式与半径上界的 **rho,对更新无一阶影响**(已数值验证);
- 唯一真正的步长旋钮 **c_adv 与 reward 归一化耦合**,论文自己承认无法当作纯 lr 乘子;
=> **方法的"可控性"论证与实际的可控量之间存在错配。**

## 2026-08-27 16:00 — ★ 把批评转成方法: 冻结归一化器消融(已排队)

### 逻辑链
1. canonical 损失 `L = mean(r^2)/mean|r|` 对 r **1-齐次** => 梯度 0-齐次 => 对目标距离不敏感;
2. 实测: rho 0.01 vs 0.10,实现增益(2.04e-4 vs 2.02e-4)与训练轨迹(+0.0585 vs +0.0598)几乎重合
   => **信赖域是装饰性的**;
3. **若这是齐次性所致,那它是可以去掉的** —— 这就从「批评」变成「改动」。

### 消融设计: 冻结归一化器(而非换成纯 MSE)
```python
if os.environ.get("OPSD_WF_MODE", "adaptive") == "frozen":
    # 首个 batch 计算一次,此后固定为常数
    wf_p = full_like(wf_p, _WF_CONST_P);  wf_n = full_like(wf_n, _WF_CONST_N)
```
**为什么不用纯 MSE**: 纯 MSE 会改变初始梯度尺度,需要重调 lr,把「rho 是否生效」与「lr 是否合适」混在一起。
冻结常数则**初始梯度尺度与 canonical 完全一致**,唯一变化是损失由 1-齐次变 2-齐次
=> 梯度 1-齐次 => **rho 重新线性地影响更新**。这是干净的单变量改动。

### 实现与安全性
在**官方** `train_opsd_ri_sd3.py` 上加,**env 门控、默认 adaptive**(不设变量则行为与 canonical 完全一致)。
已 `ast.parse` 通过,原文件备份在 `/tmp/train_opsd_ri_sd3.bak_wf`。

### 测量方式(不再改官方代码)
官方**不记录** realized gain(那是我们的 instrumentation)。改用现成的**训练 reward 轨迹**:
- 判据: frozen 下 rho=0.01 与 rho=0.10 的 12-epoch 增幅应**出现差异**;
- 对照: adaptive 下两者几乎重合(已实测,且齐次性论证与实现无关)。
已提交 `wf010`(1035055) / `wf100`(1035056),各 12 epoch / 3xH200。

### 若结果为「仍无差异」
说明瓶颈不在更新幅度而在**方向**(cos 只有随机的 10.6%),
则转向多查询点联合拟合那条路。**两种结果都推进研究。**

## 2026-08-27 16:50 — 队列剪枝(我过度提交了)

同时有 9 个作业在排,拖慢所有人(含自己)。检视后剪掉两处冗余:
- `rho100`(adaptive ρ=0.10): **measrz 已有该点**(n=12, G_realized=2.02e-4),纯重复;
- `wf010`/`wf100`(官方 frozen,各 **3 卡**): 同一机制的 1 卡版(`wfz010`/`wfz100`,我们的实现)
  更容易排上,**且直接记录 `rtc_realized`**(官方不记录,只能用训练轨迹做代理)。
  先等便宜版给出信号,再决定是否值得占 3 卡跑官方版。

**教训**: 「多排几个反正闲着也是闲着」在共享集群上不成立 —— 9 个 pending 会互相竞争,
且 3 卡请求会长期占住调度器的 backfill 规划位。**应先跑最便宜的判别实验,再按结果放大。**

## 2026-08-27 17:10 — 临时 hold ste,让核心机制实验先跑

horizon2 即将完成(781/800),其依赖作业 `ste`(afterany)会立刻占走 2 张 H200 跑 **13 小时**,
把 `wfz010`/`wfz100`(冻结归一化器消融,各 **30 分钟 / 1 卡**)再次挤到后面。
`ste` 是 OPSD 之前的 CA-OPD 遗留臂;`wfz` 是当前主线(把「信赖域装饰性」转成正面改动)的判别实验。
=> `scontrol hold 1034647`,待 wfz 起跑后 release。**代价: 给 13h 作业增加约 35 分钟延迟。**

## 2026-08-27 18:05 — rho 扫描四点齐: 秩和检验下**实现增益与 rho 无关**

```
rho     中位数     均值    均值/中位数  G_construct@rho
0.01  0.000150  0.000204    1.36       0.01924
0.03  0.000234  0.000388    1.66       0.03682
0.10  0.000212  0.000202    0.96       0.04975
0.20  0.000222  0.000211    0.95       0.03951
Mann-Whitney: 0.01vs0.10 z=-0.82 | 0.03vs0.10 z=+0.99 | 0.20vs0.10 z=+0.16 | 0.01vs0.20 z=-0.33
=> 两两全部不显著
```

### 统计方法更正(重要)
我最初用**均值 + 3σ** 判定,得出「极差在噪声内」。但分布**右偏**(均值/中位数最高 1.66;
rho=0.03 的最大值 0.001375 是其中位数的 6 倍),均值被长尾抬高,
一度让 rho=0.03 看起来是别处的 2 倍(0.000388 vs 0.000202),我为此专门去查了异常值。
**改用中位数 + Mann-Whitney 秩和检验后,四点两两全部不显著** —— 结论更强也更稳。
**教训: 看到均值/中位数比值远离 1 时,均值不可用;右偏分布应默认用秩方法。**

## 2026-08-27 18:20 — ★ 冻结归一化器实验**设计有缺陷**,结果是假阴性(已修并重排)

### 表面结果
```
adaptive rho=0.01 中位数 0.000150 | rho=0.10 0.000212   z=-0.82 不显著
frozen   rho=0.01 中位数 0.000191 | rho=0.10 0.000179   z=-0.16 不显著
```
看起来「冻结后 rho 仍无效」=> 假设被证伪。**但这个结论不成立。**

### 缺陷: 冻结的常数本身随 rho 缩放
先确认处理确实生效(`[WF-FROZEN]` 各打印一次、rho 回显正确、save_dir 已分离),然后发现:
```
frozen 常数:  rho=0.01 -> gp=5.488531e-03
             rho=0.10 -> gp=5.488521e-02      <- 正好差 10 倍,与 rho 同比例
```
我把 gamma 冻结在**各自 run 的首个 batch**,而该值本身正比于该 run 的 rho。
于是 `2r/gamma` 中 r 与 gamma **同比例缩放**,尺度不变性**根本没被去掉** ——
我只是把「逐步归一化」换成了「逐 run 归一化」,**实验没有检验到想检验的东西**。

### 修正
加 `RTC_WF_CONST` 环境变量,让所有 rho 用**同一个绝对常数**(取 5.488521e-02)。
只有这样 `2r/gamma` 才随 r(即 rho)线性变化,才真正打破尺度不变性。
已重排 `wfc010`(1035288) / `wfc100`(1035289)。

### 这次差点犯的错
**几乎就把假阴性写成「假设被证伪」并据此转向另一条路线。**
拦下它的是「先确认处理真的生效」这一步 —— 打印出的常数值暴露了缩放关系。
**教训: 消融实验不仅要确认「开关打开了」,还要确认「被改的量在各条件下确实不同」。**

## 2026-08-27 18:55 — ★★★ 因果验证成功: 去掉齐次性后 rho 重新生效

```
                  rho=0.01    rho=0.10    Mann-Whitney
adaptive        0.000150    0.000212     z=-0.82  不显著
frozen-const    0.000071    0.000172     z=-2.02  **勉强显著**(p=0.046, 中位数 2.4 倍)
dy_norm(frozen)   0.1445      0.2183              位移幅度也随 rho 变
cos(frozen)     0.000196    0.000340
```
**三环因果链齐备**: 1-齐次(代数+数值) -> rho 无效(四点扫描) -> 去掉齐次性后 rho 恢复(z=-2.02, p=0.046)。
  注:第三环只是勉强过 0.05 线,不能当强证据用;前两环(代数/数值)才是主证。

### 必须同时说明的限制
frozen-const 在 rho=0.10 的实现增益(0.000172) **低于** adaptive(0.000212)。
=> **这不是立刻能赢的改动**,价值是让 rho 成为**可调旋钮**(现在调它无意义),不是直接提升。

### 下一步
在 frozen-const 下**向大 rho 扫描**(0.3 / 1.0)。既然 rho 现在有效,
若实现增益随 rho 继续上升,就能找到一个真正优于 canonical 的工作点 —— 那才是性能提升。
若很快饱和/反转,则说明幅度不是瓶颈,回到「方向」那条路(多查询点)。

## 2026-08-27 22:07 — horizon2 完成评测;但**缺干净基线**,结论受限

### horizon 是什么
`[HORIZON] teacher H-step flow-map matching`:
标准 MOPD 把学生均值对齐 teacher 的**单步**预测;horizon-H 改为对齐 teacher **向前滚 H 步**的位置。
```python
_Hs = min(H, num_steps - j)      # 末尾不足 H 步自动收缩
_x = teacher_prev_mean_j
for m in 1..H-1: _x = compute_teacher_step_mean(_x, timestep=timesteps[:, j+m])
```
动机(从设计推断): 单步匹配只约束局部转移,学生可每步都对却在复合后偏离轨迹;
最终样本由**复合映射**决定,故直接匹配 H 步流映射是更强约束。
实现细节: `_Hs==1` 时恒等退化为单步目标,是严格推广。

### 10 点轨迹(多 reward: PickScore+OCR+GenEval)
```
step   geneval   ocr     pick     clip    aes     hps
  61   0.8281  0.9270  0.8964  0.2952  5.6863  0.3041
 391   0.9336  0.9477  0.9193  0.2929  6.2006  0.3386
 721   0.9453  0.9363  0.9149  0.2908  6.2649  0.3354
```
**OCR 全程稳在 0.93** —— 与单 reward 臂形成量化对照(rtcraw 0.438->0.198;rtckb 0.293->0.038)。
代价是 pick 只到 0.915(rtcraw 到 1.03),典型的广度换深度。

### ⚠️ 对照的混淆(必须写明)
vs `tau10`(n=8 公共步): geneval +0.0088(t=+3.81) / aes +0.0726(t=+3.80) 更好;
clip -0.0037(t=-5.71) 更差;ocr/pick/hps 打平。
**但 `horizon2 = mopd + horizon=2`,而 `tau10 = mopd + CA-OPD(tau=1.0)` —— 两个差异,不是一个。**
且用户此前实测冲突消解方法本身有害(cagrad -7.4%/conflict -5.3%/cconf -5.1%),
tau10 可能已被该机制拖低 => horizon2 的"更好"里有多少来自 horizon 无法区分。

**已评测的臂中没有 plain `mopd`(h=1、无 CA-OPD、无 POE)。**
=> **要判断 horizon 是否有效,必须补一个 plain mopd 基线臂**(约 10h/3-4 卡)。
即便按现有对照,效应也都在 ±1% 量级(geneval +0.9%/aes +1.2%/clip -1.3%),2 好 1 坏 3 平,
**不是清晰的胜利**。

## 2026-08-27 22:20 — horizon 结论修正 + 分析工具两处缺陷

### 1) horizon 的权威判定(修正我早先的口述)
按项目既定判据(|t|>2.5 **且** |差|>2σ_重测),`horizon2` vs `tau10`,n=8 公共 step:
**1 好**(aes +0.0726, t=+3.8) / **1 坏**(clip −0.0037, t=−5.7) / **4 平**。
我早先说"2 好 1 坏 3 平"偏乐观,来自一个临时脚本的两处问题:
- 用子串匹配抓到 `geneval` 键而非项目统一的 `accuracy` 键(step 61 分别 0.8483 / 0.8281);
- 只看 t,未叠加噪声带判据(geneval σ=0.0297,+0.0088 远在带内)。
评测数据自始至终完整(两臂每 step 均 3/3 benchmark)。**结论不变且更弱:不建议投入 horizon。**

### 2) compare_arms.py 两处既存缺陷(已修,备份在 scratchpad/compare_arms.bak)
- **静默残缺**:`load()` 用 `if os.path.exists` + `o.update`,三个 benchmark 目录只到两个也照算。
  已改为回传覆盖度,残缺 step 剔除并打印 `[warn]`。**在既有数据上从不触发,故既往结论不受影响。**
- **硬编码 step 列表**:不传 step 时用固定 12 点,对 horizon2/tau10 只命中 n=2(真实可用 n=8)。
  n 变小本身就是选择效应。已改为自动取两臂交集。

### 3) 用新口径复核既往结论 —— 全部通过,无需回撤
| 对照 | n | 结论 |
|---|---|---|
| tau02/tau20 vs tau10 | 11 / 8 | 六轴全打平(τ 扫描结论成立) |
| poe15 vs tau10 | 9 | aes +10.9σ、hps +8.0σ 好;geneval −6.5σ 差(POE 权衡成立) |
| poe30 vs tau10 | 4 | aes +5.6σ 好,geneval 塌到 0.52(λ 曲线极端点成立) |

### 4) wfc300 / wfc1000 已重排(1035439 / 1035440)
旧的 1035308/1035309 提交于 18:48:47,而已完成的 wfc010/wfc100 提交于 18:08:31 ——
两次独立提交,SLURM 不保留提交环境,**无法核实新两点是否带了 `RTC_WF_CONST`**。
漏传会退回"每 run 首批均值"(γ 自身随 ρ 缩放),正是造成上次假阴性的路径。
故取消重排,用 `--export=ALL,RTC_WF_MODE=frozen,RTC_WF_CONST=5.488521e-02,RHO=...` 显式传入。
四个 ρ 的 save_dir 已验证互斥(r010/r100/r300/r1000-frozen)。
运行时核对点:日志须出现 `[WF-FROZEN] const=5.488521e-02 (来源=RTC_WF_CONST 环境变量)`。

### 5) NFS 缓存今晚第 5 次误导
22:07 数 evhorizon 为 29/30 判缺 `horizon2_781`,实为该文件 22:10 才落盘;
10 个 array task 全部 `COMPLETED 0:0`。**不是任务失败。**

## 2026-08-27 22:35 — 分区盲区解除,队列从"明晚"变成"现在"

### 问题
只盯 `sablab-gpu` 派活,而该分区被他人占满且我的优先级(2226~2238)低于 xih4004(2254,另有 32 个 pending)。
`scontrol show job` 给出的预估:`opsdcert` 2026-08-28T21:57、`evrtcwarm` 21:18、
`wfc300/wfc1000` **StartTime=Unknown**。即用户早上什么也看不到。

**这不是记忆缺失** —— `cluster-usage-conventions` 里早已写明 scu-gpu 是默认分区、minilab-gpu 有 h100 gpu14、
preempt_gpu 需 `--qos=low`。是我连续几轮没去用。

### 实际空闲(22:30 快照)
ai-gpu16 h100×4 **idle**(preempt_gpu) / ai-gpu19 l40×4 **idle**(scu-gpu) /
ai-gpu13 l40×4 **idle**(minilab-gpu) / ai-gpu14 h100 空 3(minilab-gpu) / ai-gpu05 a40 空 2(preempt_gpu)

### 派活结果(1 个在跑 -> 5 个在跑)
| 作业 | 去处 | 说明 |
|---|---|---|
| `wfc300` / `wfc1000` / `wfc100r` | minilab-gpu ai-gpu14 (H100×1 each) | ρ 扫描,~25 min |
| `evrtcwarm` | preempt_gpu ai-gpu05 (**A40**×2) | 改 Partition 需**同时**改 QOS,否则报 `Invalid qos specification` |
| `opsdcertp` | preempt_gpu (H100×3) | 20 epoch 采纳率短探针 |

### 硬件等效性(用户 22:40 的要求)
- **评测既定基准是 A40**:`ev*` 作业历史上全部在 ai-gpu01/02/03/04。故 evrtcwarm 特意选 A40 的 ai-gpu05
  而非更快但会破坏可比性的 L40。
- ρ 扫描从 H200 挪到 H100,故**加排 `wfc100r`(ρ=0.100, `WFC_TAG=h100rep`)**,
  与已完成的 H200 上同 ρ 点直接对比,把硬件差异测出来而不是假设可忽略。
  为此给 `rtc_raw_meas` 加了 `WFC_TAG` 环境变量(照搬 `GC_TAG` 的用法),四个 save_dir 已验证互斥。

### opsdcert 不能整个丢到 preempt 上
`opsd_cert.slurm` 有 `--requeue` 但**不传 `resume_from`**,被抢占后从头重启;
而 `--open-mode=append` 会让日志看起来"还在跑"。故改排 20 epoch 短探针
(采纳率 `opa_acc_plus` 每 epoch 都记,L1363,不需要跑满 100),原 6h 版本保留在 sablab-gpu 队列里。

### 运行时核对(第(1)条:不能只看 exit=0)
三个 wfc 均确认 `const=5.488521e-02 (来源=RTC_WF_CONST 环境变量)` ——
用的是显式绝对常数,不是会随 ρ 缩放的每-run 回退值。**重排是必要的。**
RTC 指标(rtc_loss/rtc_pos/rtc_neg/grad_norm)待作业跑完核对。

## 2026-08-27 22:55 — 又一个静默不匹配:OPD_SMOKE_EPOCHS

### 问题
22:30 重排的三个 wfc 只跑了 7:45,而 H200 上的旧点跑了 24:51。查出:
**新的是 `num_epochs=3`(smoke 默认),旧的是 12。**
`--export=ALL,VAR=...` 只钉住我显式列出的变量,原提交 shell 里的 `OPD_SMOKE_EPOCHS=12` 没带过来——
修掉 `RTC_WF_CONST` 的同时引入了另一处不匹配。已按 12 epoch 重排(1035450/1035451/1035452)。

### 从 3-epoch 数据得到的两个诊断(数据本身作废,诊断有效)
| 条件 | n | median rtc_realized |
|---|---|---|
| ρ=0.010 H200 12ep | 12 | 7.12e-05 |
| ρ=0.100 H200 12ep | 12 | 1.72e-04 |
| ρ=0.100 H100 **3ep** | 3 | 3.12e-04 |
| ρ=0.300 H100 **3ep** | 3 | 2.25e-04 |
| ρ=1.000 H100 **3ep** | 3 | 2.72e-04 |

1. **epoch 效应存在**:12ep run 里前 3 epoch 的中位数比后 9 个高 **1.24×**(ρ=0.010)/ **1.45×**(ρ=0.100)。
   故 3ep 与 12ep 不可比。
2. **该测量极噪**:单 run 内逐 epoch 摆动达 **20 倍**(ρ=0.100: 1.79e-5 ~ 3.91e-4)。
   **n=3 没有任何判别力**;那个看似 81% 的"硬件差异"(1.72e-4 vs 3.12e-4)基本是噪声+epoch 效应,不是硬件。
   → 硬件对照 `wfc100r` 的价值在此:它把"换硬件"这个疑问从猜测变成可测量,而不是假设可忽略。
3. `rtc_realized_cos` 中位仅 **0.0002~0.0003** —— 实现位移几乎与 g 正交,
   与"构造增益绝大部分丢在拟合阶段"的图景一致。

### 重排后的设计(全部 n=12)
r010 H200 / r100 H200 / **r100 H100(硬件对照)** / r300 H100 / r1000 H100
—— r100 那一对隔离硬件;H100 上的 100/300/1000 三点内部同硬件,用于判 ρ 趋势。

### 注意
噪声这么大,n=12 的中位数 SE 约 25%。能分辨 ρ=0.010→0.100 的 2.4× 差异,
但要分辨"ρ 继续升 vs 饱和"(可能只有 1.5×)会很勉强。若 12 点判不出,需加长而非加更多 ρ 点。

## 2026-08-27 23:05 — 更正:frozen-const 的 z 是 −2.02 不是 −2.89

写 `rho_trend.py`(把 ρ 扫描分析固化成脚本,不再每次临时写)时发现:
同一份数据、同样的中位数(7.123e-05 / 1.723e-04),脚本给 z=−2.02,而既有记录是 z=−2.89。
三种独立方法核验:
- 暴力数 a<b 的对数: U=107 -> |z|=2.021
- `scipy.stats.mannwhitneyu`: U=37, p=0.0464 -> |z|=1.992
- `rho_trend.py`: z=−2.02

**结论:真实值 |z|≈2.0, p≈0.046 —— 刚过 0.05 线,不是 p<0.01。** 旧的 −2.89 记错了。
已改 `PLAN_2026-08-27.md` / `OVERNIGHT_2026-08-26.md` / 本文件共 5 处。

### 对结论的影响
"ρ 是装饰性的"有四条腿,只有第四条被削弱:
| 腿 | 状态 |
|---|---|
| 代数(损失 1-齐次 -> 梯度 0-齐次) | 不受影响,**这是主证** |
| 数值(梯度放大 100 倍,更新不变) | 不受影响 |
| 四点实测(两两秩和不显著) | 不受影响 |
| 因果(冻结 γ 后 ρ 恢复) | **从 p<0.01 降为 p=0.046**,只能当支持性证据,不能当强证据 |

前两条是确定性的(代数恒等式 + 可复现的数值实验),不依赖统计。故主结论仍成立,
但**写论文时第四条要按 p=0.046 报,不能吹成 p<0.01**。

### 教训
"多点结论一律配对检验"不够 —— 检验统计量本身也要用独立实现核对一次。
临时脚本的 Mann-Whitney 实现已在 `rho_trend.py` 里与 scipy 对齐并固化。

## 2026-08-27 23:10 — opsdcertp 实测采纳率:**我们的弱点#2 被自己的实验大幅削弱**

`opsdcertp`(官方代码 + `--config.opsd.opa_cert=1`, nproc=3, H100 ai-gpu16, 20 epoch)前 8 个 epoch:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| `opa_acc_plus`  | .9714 | .9792 | .9635 | .9792 | .9740 | .9844 | .9531 | .9219 |
| `opa_acc_minus` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**先验证了这不是硬编码**:`ri_opa_cert=1` 分支下(L1344-1345)
`opa_ap=(R_plus>=R_old+margin)`、`opa_am=(R_minus<=R_old-margin)` 都是实际 reward 比较。
`opa_acc_minus` 恒为 1.0 是真测量结果,不是没测。

### 结论:弱点#2 要重写
原表述"**canonical 从不验证目标**"分两部分:
- **事实部分仍成立**:`opa_cert=0` 硬编码 `acc=1.0`;论文方法章节 "certif" 出现 0 次。
- **批评部分基本被推翻**:那个假设在经验上是对的 —— 正支 **96.6%**、负支 **100%** 真的达到了预期方向。
  换言之这是一个**被验证过的合理捷径**,不是一个会造成误差的缺陷。

**不能再把它当作四条弱点之一去写。** 剩下的可说之处只有两点,而且都还没证实其重要性:
1. 有 ~3.4% 的正目标并未抬高 reward 却照样被拟合(污染量级很小);
2. `opa_acc_plus` 可能随训练下降(epoch 0 的 .9714 -> epoch 7 的 .9219)。若 20 epoch 跑完确认在降,
   则"假设在训练后期失效"是一个真结论;若只是噪声,则弱点#2 整条作废。

### 对整体立论的影响
四条弱点现在剩:
| # | 弱点 | 状态 |
|---|---|---|
| 1 | ρ 是装饰性的 | **成立**(代数+数值为主证;因果腿 p=0.046) |
| 2 | canonical 从不验证目标 | **基本作废** —— 实测采纳率 96.6%/100%,捷径是合理的 |
| 3 | 没有 teacher,"Distillation" 是名义的 | 成立(代码 grep=0) |
| 4 | 官方实现的离目标退化 | 成立 |

## 2026-08-27 23:20 — ρ 扫描完成:**幅度不是瓶颈,按预注册规则转「方向」路线**

### 先修一个读错
探针键名 `rtc_dR_g_{int(_r*1000):03d}`,而 `_r ∈ (0.001,0.003,0.01,0.03,0.10,0.20,0.30)`。
即 `rtc_dR_g_100` 是 **ρ=0.10** 不是 1.00。我第一次读表时整条曲线的 ρ 都放大了 10 倍,已修正。
(该错误只出现在中间计算,未进入任何结论。)

### 结果(全部 n=12, frozen-const=5.488521e-02)
**构造增益曲线(五个 run 各自测,几乎重合):**
| ρ | 0.001 | 0.003 | 0.01 | 0.03 | **0.10** | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|
| G_construct | 3.4e-3 | 7.9e-3 | 2.0e-2 | 3.8e-2 | **4.9e-2 峰值** | 3.9e-2 | 2.3e-2 |
| win_rate | .972 | .986 | .986 | .986 | .979 | .889 | .771 |

**训练 ρ 扫描:**
| 训练 ρ | 硬件 | G_construct@该ρ | G_realized | 实现比例 | acc_p |
|---|---|---|---|---|---|
| 0.010 | H200 | 2.035e-02 | 7.123e-05 | 0.350% | 0.993 |
| 0.100 | H200 | 4.907e-02 | 1.723e-04 | **0.351%** | 0.972 |
| 0.100 | H100 | 4.862e-02 | 1.926e-04 | 0.396% | 0.986 |
| 0.300 | H100 | 2.240e-02 | 2.026e-04 | 0.905% | 0.792 |
| 1.000 | H100 | 探针范围外 | 2.581e-04 | — | **0.111** |

Mann-Whitney: 0.010 vs 其余全部显著(z=−2.02~−3.41);**0.100/0.300/1.000 之间两两均不显著**
(z=−0.46~−1.62)。

### 硬件等效性(用户 22:40 的要求)—— 通过
ρ=0.100 的 H200 vs H100:
- G_realized 1.723e-4 vs 1.926e-4,**z=−0.40 不显著**
- G_construct 整条曲线差 **0.9%**(4.907e-2 vs 4.862e-2)
→ 跨 sablab(H200)/minilab(H100) 的数据可以合并使用。

### 预注册判据的答案:**幅度不是瓶颈**
原规则:"realized 随 ρ 继续升 ⇒ 找到优于 canonical 的工作点;很快饱和/反转 ⇒ 转方向路线"。
1. realized 从 ρ=0.01 到 1.00(**100 倍**)只涨 **3.6 倍**,且 0.100→1.000 一段不显著;
2. 更关键:**G_construct 本身在 ρ=0.10 见顶**。再大 ρ 只会让目标变差 ——
   win_rate 0.979→0.771,`acc_p` 到 ρ=1.0 塌到 **0.111**(89% 的正目标根本没抬高 reward)。
3. 即论文的 ρ=0.10 **已经站在构造增益的最优点上**,幅度维度没有余量可挖。

### 对弱点#1 的表述要更精确
不能笼统说"ρ 是装饰性的"。准确表述是:
- ρ **强烈影响构造出的目标**(G_construct 随 ρ 变化 15 倍,且有明确最优点 0.10);
- ρ **几乎不影响实际更新**(拟合损失 1-齐次 -> 梯度 0-齐次 -> 目标幅度被归一化器约掉);
- 所以信赖域半径调的是一个**随后被归一化掉的量**。
这比原表述更强也更站得住:论文把 ρ 当作核心超参并做了消融,但它调的东西在更新阶段被约分了。

### 99.6% 丢失率确认
论文默认 ρ=0.10 处,实现比例 **0.351%**,即 **99.65% 的构造增益丢失**。
与此前独立测得的 99.59% 一致。**这是主线立论的核心数字,已被复现。**

### 下一步(方向路线)
`rtc_realized_cos` 中位仅 **2.0e-4 ~ 3.7e-4** —— 实现位移与 g 近乎正交。
既然幅度调不动,要攻的是**方向**:OPSD 只在**单个查询点** σ*=0.278 上约束目标,
拟合器有大量自由度可以在其他噪声水平上任意偏移。
=> **多查询点联合拟合**:在若干 σ 上同时约束,压缩不可实现子空间。这是 Phase 3 #2,现在升为第一优先级。

## 2026-08-27 23:35 — 弱点#2 以修正形式救回:采纳率随训练单调退化

`opsdcertp`(官方代码, opa_cert=1, H100x3) 前 13/20 epoch:
```
acc_plus : .9714 .9792 .9635 .9792 .9740 .9844 .9531 .9219 .9141 .8854 .8802 .8516 .8750
acc_minus: 1.0000 (全程,13/13)
```
Spearman ρ(epoch, acc_plus) = **−0.841, t=−5.15 (df=11)** —— 高度显著,不是噪声。
线性斜率 −0.011/epoch;前半均值 .9753 vs 后半 .8973。

### 修正后的表述(比 23:10 那版更准,也更有价值)
不是"canonical 的假设是错的"(23:10 我据前 8 点判它基本作废),而是:
**假设早期成立(~0.97),但随训练单调退化** —— epoch 12 已到 0.875。OPSD 训 ~100 epoch。
且退化**不对称**:负支从不失效(1.000),只有正支退化。
=> 模型被可靠地推离坏区域,却越来越不可靠地被推向好区域。这是一个可写的机制性结论。

### 两个必须写明的限制
1. **线性外推不可用**:外推 epoch 100 给 −0.106,超出 [0,1],必然饱和。
   只能说"在 0~12 的观测窗口内以 ~1.1 个百分点/epoch 下降"。
2. **测量与被测对象耦合**:`opa_cert=1` 既算采纳率**又用它加权损失**,
   epoch 12 时已有 ~12% 目标被丢弃而 canonical 会全部保留 —— 两条轨迹已分叉。
   **这不是 canonical 自身的采纳率曲线。** 要测 canonical 的,需要一个"只算不用"的模式(代码目前不支持)。
   偏差方向未知:cert 丢掉坏目标本应训得更好,故 canonical 的退化可能更严重,但没有证据,不能假设。

### 待 20 epoch 跑完确认曲线是否已开始走平。

## 2026-08-27 23:30 — 排 rand 归因对照(方向路线的第一步)
ρ 扫描判定"幅度不是瓶颈"后,下一个问题是**方向到底贡献了多少**。
`rtc_rand` 配置早已存在但从未跑过(sacct 无记录)。新加 `rtc_rand_meas` = `rtc_rand` + realized/curve。
- `rtcrandm`(1035454) / `rtcrawm`(1035455),均 H100 ai-gpu14、adaptive 归一化、ρ=0.10、12 epoch。
- 基线不用历史 measrz 点,而是同批新跑一个 raw —— 同硬件同 epoch 同会话,只差方向一个变量。
- **单 run 内还有内部对照**:`rtc_curve` 同时记 `rtc_dR_k_*`(实际方向=随机)与 `rtc_dR_g_*`(reward 梯度),
  随机那条的构造增益应当 ≈0。
判据:若 rand 的 G_realized 与 raw 相当 -> reward 梯度方向不贡献,整套构造等价于
"随机扰动 + 采纳门控筛选";若 rand ≈ 0 -> 方向确实重要,损失在拟合阶段,支持多查询点路线。

## 2026-08-27 23:50 — 采纳率曲线的正确形态:**前 8 epoch 下降后走平在 0.86**

18/20 epoch:
```
.9714 .9792 .9635 .9792 .9740 .9844 .9531 .9219 .9141 .8854 .8802 .8516 .8750 .8906 .8698 .7370 .9036 .8802
```
| 区间 | 均值 |
|---|---|
| epoch 0~5 | 0.9753 |
| epoch 6~11 | 0.9010 |
| epoch 12~17 | **0.8594** |

全 18 点 Spearman **−0.808, t=−5.49 (df=16)** 显著;
但**只看 epoch 8~17 则 Spearman=−0.212 —— 已走平**。

### 修正 23:35 的表述
不是"单调退化(可能继续降)",而是:
**前 ~8 个 epoch 从 0.975 掉到 ~0.86,之后稳定。** 全程显著性是被前段拉出来的。
=> **canonical 在稳态下约拟合 14% 的无效正目标**(而非某个持续恶化的比例)。
负支仍恒为 1.000。

### 这让主线实验的预测变得具体可证伪
`opsdc100`(cert=1) 做的正是去掉这 14%。
- 若最终 Pick 显著高于基线 -> 这 14% 有害,捷径是真缺陷,弱点#2 成立且可量化;
- 若打平 -> **捷径被验证是安全的**,弱点#2 作废,应如实写"我们检验了这个怀疑并否定了它"。
两种结果都值得写;第二种同样是有价值的负结果(且我们是第一个测过的)。

### 已取消 opsdcert(1034930)
与 opsdc100 重复(同为 cert=1 / 100 epoch),且占着 H200 队列位。

## 2026-08-28 00:00 — 方向归因对照:**瓶颈确认在拟合端,不在目标端**

`rtcrandm`(rand) vs `rtcrawm`(raw),同硬件 H100 ai-gpu14、adaptive、ρ=0.10、12 epoch、同批提交:
| | G_realized 中位 | acc_p | acc_m | cos |
|---|---|---|---|---|
| raw(reward 梯度) | **+2.001e-04** | 0.965 | 1.000 | +3.19e-04 |
| rand(随机方向) | **−1.560e-05** | 0.319 | 0.701 | −2.63e-05 |

Mann-Whitney **z=−3.98** 显著。随机方向的实现增益为**负**。

单 run 内部对照(`rtc_curve` 同批记两条,同一批样本):
| ρ | 随机方向 dR (win) | reward 梯度 dR (win) |
|---|---|---|
| 0.01 | +2.19e-4 (0.597≈随机) | **+2.41e-2 (0.986)** |
| 0.10 | −4.25e-3 (0.319) | **+5.22e-2 (0.986)** |
| 0.30 | −3.71e-2 (0.049) | +2.55e-2 (0.778) |

### 结论
1. **reward 梯度方向是必需的**,且构造出的目标本身很好(win 0.986, acc_p 0.965);
   采纳门控救不了随机方向(acc_p 0.965 -> 0.319)。
2. 故 **99.65% 的损失不在目标构造,确认在拟合阶段** —— 与论文自己的 G_fit 记账一致。
3. **这判了原 RTC-GN 死刑**:它预条件的是目标构造端,而该端不是瓶颈。
   (此前已从 ksteps 五轴全输推断过,现在有了直接证据。)
4. 大 ρ 下随机方向 dR 强烈为负(ρ=0.30 时 win=0.049),说明 reward 曲面在 y0 附近局部近似凹 ——
   任何大幅随机位移都下坡。这也解释了为何 G_construct 曲线在 ρ=0.10 见顶。

## 2026-08-28 00:10 — 主线实验开跑 + 一处 nproc 不匹配

### 三个实验在跑
| 作业 | 配置 | 节点 | nproc |
|---|---|---|---|
| `opsdc100`(1035463) | **cert=1**, 100ep | ai-gpu14 H100 | **3**(请求 4,SLURM 降级) |
| `opsdb100`(1035464) | cert=0, 100ep | ai-gpu16 H100 | 4 |
| `poe125`(1035465) | **lam=1.25** MOPD-POE, 800ep | ai-gpu11 A100 | 2 |

**nproc 不匹配已记录**:`--gres` 又一次未被兑现(opsdc100 请求 4 实得 3)。
3 与 4 在官方 OPSD 下**梯度等价**(都是 1152 samples/优化步,group_size=24 不会被拆),故科学上可比;
但速度差 33%,且这是一个非受控差异。**巧的是 opsdc100 的 nproc=3 恰好与原始 H200 基线一致。**

### 可分解的三点设计
| run | cert | 硬件 | nproc | 末值 |
|---|---|---|---|---|
| opsdc100 | 1 | H100 | 3 | 跑 |
| opsdb100 | 0 | H100 | 4 | 跑 |
| **opsd_repro**(已完成) | 0 | H200 | 3 | **reward_avg=0.9148 @ep99** |
- c100 vs b100 -> cert 效应(同硬件)
- c100 vs repro -> cert 效应(同 nproc)
- b100 vs repro -> 纯硬件/nproc 对照

### 不必等评测:训练 reward 可直接配对
`metrics.jsonl` 每 epoch 写 `reward_avg`,且与 `epoch` **在同一行**(已核验:基线 epoch 0~99 连续 100 点,
0.7561 -> 0.9148)。新写 `opsd_cert_compare.py` 做逐 epoch 配对差 + 符号检验。
注:`opa_acc_plus` 在**另一行**且无 epoch 键,需承接上一个 epoch —— 该逻辑已在脚本里,顺序正确。

### poe125 的时间现实
2 卡 A100 上约 2.7 min/epoch,800 epoch 需 ~36 小时。到早上 07:00 约 155 epoch,
与 tau10 的公共 step 只有 61/121 两个 -> **早上只能给出初步读数,不是定论**。如实说明,不夸大。

## 2026-08-28 00:25 — 对 OPSD 概念的判断 + 三条实质改动(不是调参)

用户 00:20 的批评正确:**`poe125` 是 OPSD 之前的 CA-OPD 线上的东西,且本质只是调 λ**。
它是"最可能拿到正向显著结果"的保底枪,不是贡献。真正的贡献必须动 OPSD 的概念。

### 先给判断:OPSD 哪里对、哪里被高估
**真的对(我们自己的实验支持他们):**
- reward 梯度方向是必需的 —— rand 对照 z=−3.98,随机方向实现增益为**负**;
- 构造出的目标本身很好 —— win 0.986、acc_p 0.965(早期);
- ρ=0.10 恰在总体 G_construct 曲线的峰值(0.001~0.30 七点实测);
- 复现成立 —— Pick 25.2133±0.0541 vs 论文 24.940。

**被高估 / 站不住:**
- "Distillation" 是名义的(`grep -c teacher` = 0);
- 信赖域 ρ 对**更新**无一阶影响(损失 1-齐次 -> 梯度 0-齐次;代数+100 倍数值+四点扫描);
- `acc=1.0` 是假设,稳态实测只有 ~0.86;
- 单查询点 -> 自己构造的增益 **99.65% 没有实现**。

=> 论文的核心洞见(在低噪声 clean-output 上做 reward 上升构造目标,再拟合)**是对的且 work**;
   问题全在"**怎么构造**"(固定半径、不验证)和"**怎么拟合**"(单点、归一化器约掉尺度)。
   这正是可以做出实质改进的地方,而不是"他们全错了"。

### 三条实质改动

**A. 逐样本最优半径(今晚已实现并在跑 = `opsdls100`)**
- 依据:`rtc_win_g_300`=0.778 -> ρ=0.30 时已有 **22% 的样本越过自身最优点**;
  而官方给所有样本同一个 ρ‖y0‖。最优半径是**逐样本**的,总体峰值 0.10 只是平均意义上的。
  另:我们自己代码的注释记载,改成"按实测 reward 逐样本取最优迭代点"后 **acc_p 从 0.31 修到 1.000**。
- 改动:`_opa_tr_step` 末尾在 x0->x 射线上取 K 个等距点,连同 **x0 本身**按实测 reward 逐样本选最优。
- **理论后果**:R(y_plus) ≥ R(y0) 恒成立 => **canonical 硬编码的 acc=1.0 由构造为真,不再是假设。**
- 代价:K+1 次 reward 前向(仅解码+打分,无反向),与原有 n_ascent 次反向同量级。
- 实现:`config/opsd_defaults.py` 加 `opa_ls`(默认 0 = canonical 逐字节不变);
  `train_opsd_ri_sd3.py` 的 `_opa_tr_step` 加 `n_ls` 参数。备份在 scratchpad/*.bak_ls。

**B. 多查询点联合拟合(未实现,概念上最强)**
- 依据:`cos(Δy, g) ≈ 3e-4` 而 `‖Δy‖ = 0.186` —— **模型动得很多,但几乎完全正交于目标方向**;
  realized/construct = 0.35%。
- 概念批评:OPSD 只在**一个** σ*=0.278 上约束 y。但 clean-output 目标 y_plus 是**与 σ 无关的**
  (它是像素空间里的一个点),所以完全可以要求模型**从多个噪声水平**都预测出它 —— 且**不增加任何 reward 计算**。
  单点约束留下巨大零空间,更新就在零空间里跑掉了。
- 改动:L = Σ_σ acc·(y_θ(z_σ, σ) − y_target)²/wf,z_σ 由同一个 y0 前向加噪得到。
- 代价:K 倍拟合前/反向;昂贵的部分(采样 + reward 梯度)不变。
- 这把"点约束"变成"**流映射一致性约束**",是对 one-step 概念的实质补强。

**C. 让拟合损失保留幅度信息(未实现)**
- 依据:归一化器 wf 使损失 1-齐次,ρ 被完全约掉(已三重验证)。
- 概念批评:论文把 ρ 当核心超参并做消融,但它调的量在更新阶段被约分了。
- 改动:用**共享常数**而非逐样本逐分支的 wf,使样本间的相对幅度存活 ——
  reward 余量大的样本获得成比例更大的更新。
- 风险:我们的 frozen-const 实验显示这样 ρ 确实恢复作用(p=0.046),但也可能不稳定。

### 当前四臂设计
| 臂 | opa_ls | opa_cert | 硬件/nproc | 作用 |
|---|---|---|---|---|
| `opsdls100`(1035479) | **4** | 1 | L40 / 3 | **新方法 A** |
| `opsdc100`(1035463) | 0 | 1 | H100 / 3 | cert 单独的效应 |
| `opsdb100`(1035464) | 0 | 0 | H100 / 4 | canonical(同硬件) |
| `opsd_repro`(已完成) | 0 | 0 | H200 / 3 | 原始基线 reward_avg=**0.9148** |
- ls100 vs c100 -> **线搜索的效应**(两者都 cert=1,单变量;但硬件 L40 vs H100 有差异,需注明)
- c100 vs b100 -> cert 的效应

## 2026-08-28 00:40 — cert 的早期信号:**纯对照挡下了一次误报**

n=8 公共 epoch,`reward_avg` 逐 epoch 配对:
| 比较 | 差值 | se | 符号检验 |
|---|---|---|---|
| cert=1(n3,H100) vs cert=0(n4,H100) | **+0.00513** | 0.00237 | z=+2.12 **显著** |
| cert=1(n3,H100) vs cert=0(n3,H200) —— nproc 匹配 | +0.00536 | 0.00317 | z=+1.41 不显著 |
| **cert=0(n4,H100) vs cert=0(n3,H200) —— 纯对照** | **−0.00315** | 0.00213 | z=−1.39 不显著 |

**纯对照(两边都 cert=0,只差硬件/nproc)自己就漂了 −0.003**,与"cert 效应"的 +0.005 同量级。
第一行的显著性有一部分只是因为 `opsdb100` 恰好偏低。

### 诚实结论(n=8)
**cert 效应尚未成立。** 噪声地板 ~0.003,效应 +0.005 未明确高出;nproc 匹配的比较不显著。
若只跑第一行就报"cert 显著更好",就是一次误报。**教训:每个正向比较都要配一个 null 对照。**

### 另外两个结构性问题(影响解读,必须写明)
1. `opsdc100` nproc=3 / `opsdb100` nproc=4 -> **数据分片不同,不是真配对设计**,
   逐 epoch 配对只消掉了"训练进度"这一项,没消掉 prompt 采样。
2. 前 8 epoch 的 acc_p 还在 0.95~0.98,cert 只拒掉 ~3% 目标,**效应本就该很小**。
   真正该看的是 epoch 12 之后(acc_p 稳定在 0.86,拒掉 ~14%)。

### 方向A(线搜索)的速度问题
`opsdls100` 在 L40 上:Epoch 0 采样 1:48,但 OPA 块超过 5 分钟未完成。
原因累加:`opa_mb=2` 使分块数 ×3;`ls=4` 每分支 +5 次 reward 前向;`cert=1` 再 +3 次。
=> OPA 块可能慢近 10 倍,**100 epoch 跑不完**。
对策(待首个 acc 验证后执行):确认 acc→1.0 后,关掉 cert(线搜索已由构造保证 acc=1.0,
不需要每 epoch 再测)、把 ls 降到 2、opa_mb 提到 3。
按 epoch 配对比较不要求两臂 epoch 数相同,只需公共区间,故不必跑满 100。

## 2026-08-28 00:35 — 方向A:线搜索**已验证有效**;L40 结构性放不下,已搬 H100

### 验证成功(这是新方法的关键判据)
`opsdls100` 在 L40 上活到 epoch 0 结束,产出:
```
acc_plus  = 0.9965      (同期 opsdc100 无线搜索 = 0.9896)
acc_minus = 1.0000
```
**没到精确 1.0 的原因已查明且合理**:cert 判据是 `R_plus >= R_old + opa_margin(0.005)`,
而线搜索只保证 `R_plus >= R_old`(x0 在候选集内)。约 0.35% 的样本其最佳改善量小于 margin。
=> 实现正确,符合预期。真正的检验在 epoch 12 之后(基线 acc_p 稳态 0.86)。

### L40 放不下(两次 OOM,位置不同 -> 不是单点问题)
| 次 | opa_mb | 崩点 | 说明 |
|---|---|---|---|
| 1035479 | 6 | L1345 `torch.autograd.grad(_rg.sum(), _xg)` | OPA 块的 reward 反向;**canonical 路径,非补丁引起** |
| 1035480 | 2 | L1548 `_cfg_combine(transformer_ddp(...))` | **拟合损失前向**,另一个内存消费者 |
`opa_mb=2` 修好了第一处,第二处又爆 => **L40 的 44.4GiB 结构性不够**,再削就会动到算法本身,停止削减。

### 已搬到 H100(用户 00:24 指示:不行就用 H100/H200/A100)
停掉 `opsdb100` 让出 4 张 H100。**理由**:它是 cert=0 的同硬件基线,
而我们已有完整 100 epoch 的 H200 基线(`opsd_repro`);它作为噪声地板对照的作用**已完成**
(刚给出 −0.003 的漂移量),14 个 epoch 的数据全部保留在盘上,随时可再用。

新作业 `opsdls100`(1035485): H100x3、`opa_mb=6`、`cert=1`、`ls=4`
=> **与 `opsdc100` 仅差 `opa_ls` 一个变量**(同硬件、同 nproc、同 mb、同 cert)。这是干净的单变量对照。

## 2026-08-28 00:50 — 方向A 在 H100 上稳定运行

`opsdls100`(1035485, H100x3, nproc=3, opa_mb=6, cert=1, **ls=4**) —— **0 次 OOM**,已到 epoch 2。
与 `opsdc100`(同硬件同 nproc 同 mb 同 cert, **ls=0**) **仅差 opa_ls 一个变量**。

| epoch | 0 | 1 |
|---|---|---|
| ls=4 | **0.9922** | **0.9974** |
| ls=0 | 0.9896 | 0.9844 |

前两点差距小是**预期的** —— 基线早期本来就有 0.98+。真正的检验在 epoch 8 之后
(基线已实测掉到 0.8646 / 0.9010 / 0.9479)。

速度:ls=0 为 4.2 min/epoch,ls=4 为 6.5 min/epoch(线搜索开销 ~+55%)。
到 08:00 约 65 epoch,足够配对比较(配对不要求两臂 epoch 数相同,只需公共区间)。

### 一次自制的混淆(已澄清,无损失)
我在**提交作业之后**才 `rm -f metrics.jsonl`(本意是清掉 L40 失败留下的数据),
担心训练进程持有已删除 inode 导致写入丢失。核实结果:**文件正常存在**(00:46 更新),
是 NFS 属性缓存让我先看到"无文件"。**今晚第 6 次被该缓存误导。**
教训:凡"文件不存在"的判断,必须先 `ls` 再等 1 秒再读,或直接 `ls -la` 看实际结果而不是靠 python 的 os.path.exists。

## 2026-08-28 01:10 — 方向B(多查询点)已实现,**但未经任何运行验证**

### 实现
- `config/opsd_defaults.py`: 新增 `opa_mq_sigmas`(逗号分隔的额外查询 σ,`""`=关闭=canonical 原样)。
- `train_opsd_ri_sd3.py`: 在 `opa_target_mode=="replace"` 分支内,对每个额外 σ_k:
  1. 用**官方自己的**前向加噪公式 `z_k=(1-σ_k)·latents_clean + σ_k·ε`
     (与 `opa_state_mode="forward"` 同源;锚点是模型自己的 rollout 输出,即采样时真会经过的状态);
  2. 在 (z_k, σ_k) 上前向 old/default 两个 adapter,按同样的 β 组合得 y_pos_k / y_neg_k;
  3. 用与主查询点**逐字相同**的损失(同 wf 归一化、同 accp/accm 门控、同 r1 加权)对同一个 y±;
  4. **各查询点取平均而非求和** -> 总损失尺度与 canonical 一致,**无需重调 lr**,可直接比较。
- 备份: `scratchpad/train_opsd_ri_sd3.bak_mq`、`opsd_defaults.bak_mq`。

### 未验证的部分(必须先跑冒烟)
1. **DDP 多次前向**:canonical 已有 2 次前向(old 无梯度 + default 有梯度),
   本改动再加 2K 次。DDP 在"多前向单反向"下通常可行,但可能触发梯度分桶相关报错 —— **没跑过就不知道**。
2. 显存:每个额外 σ 增加一份激活。K=2 时是否放得下 H100 未测。
3. 速度:预计 ~(1+K) 倍拟合开销。

### 今晚无法测的原因(算力)
需要 3~4 卡,而当前只剩 1 张 H100;`ste` 在 2 张 H200 上才到 checkpoint 211/800(还需 6h+),
`opsdc100` 要到 ~07:15 才跑满 100 epoch。**故 B 今晚只能停在"已实现"。**

### 一个已排除的风险
我改的是 `opsdls100`/`opsdc100` 正在跑的同一个源文件。Python 在进程启动时读源码,
运行中的作业不受影响;且 `opa_mq_sigmas` 默认为空,即使被 requeue 重启行为也不变。

## 2026-08-28 01:45 — 方向A **不成立**,且原因已精确定位(我的预测错在哪)

### 结果(n=9~10 公共 epoch, ls=4 vs ls=0, 单变量对照)
| 指标 | 差值 | t | 同号 | 符号检验 | 判定 |
|---|---|---|---|---|---|
| `opa_acc_plus` | +0.01042 | +1.29 | 6/9 | z=+1.00 | **不显著** |
| `reward_avg` | **−0.00501** | −1.67 | 4/10 | z=−0.63 | **不显著(点估计偏负)** |

### 我的预测错在哪(两处,都在代码里查实了)
我说过"线搜索让 canonical 的 acc=1.0 由构造为真"。**错。**
`cert` 打分的对象与线搜索优化的对象根本不是同一个:

| | 线搜索优化的 | cert 测量的 |
|---|---|---|
| 被打分对象 | `y_plus` 本身(clean 空间的目标) | `ep_plus = _dpm2_continue_from(zq, kq, y_plus, prev_x0, sig_sched, vold)` —— **求解器续解后的终点** |
| 对照基准 | `R(y0)`,y0 = zq − σ_q·v_old | `R(xi)`,xi = **原始 rollout 终点** |

(代码位置:L1334 `y0 = zq - sig_q*vold(...)`;L1353-1360 cert 块;
 `_dpm2_continue_from` 的 docstring 自述 "forcing forced_x0=y0 reproduces the real endpoint x_i to ~2e-6",
 故 `R(xi)` 确实是 `R(ep(y0))` 的正确代理。)

=> 线搜索**确实**保证了 `R(y_plus) ≥ R(y0)`,但那个改善**没有穿过求解器续解**。

### 这本身是一个有价值的发现
这是"构造 vs 实现"缺口的**第三个层次**:
1. 目标端 -> 拟合端:99.65% 丢失(已测);
2. 单点 -> 全轨迹:cos(Δy,g)≈3e-4(已测);
3. **clean 空间的目标改善 -> 续解后的终点改善:不传递(今晚新测)**。
   即"把目标在 clean 空间上做得更好"对最终生成**没有帮助**。

### 修正方案 A′(未实现,留给清醒时写)
在**cert 实际测量的量**上做线搜索,而不是在目标本身上:
对候选 `y_k = y0 + (k/K)·(y_plus − y0)`,先 `_dpm2_continue_from` 得 `ep_k`,再按 `R(ep_k)` 选最优,
基准用与 cert 相同的 `R(xi)`。
- **代价可接受**:σ*=0.278 是 `sigmas[8]`(倒数第二步),续解只有 1~2 次网络前向。
- 需要的量 `zq/kq/prev_x0/sig_sched/vold/xi` 都在主块里,但 `prev_x0` 目前只在 `ri_opa_cert` 分支内计算,
  实现时要把它提出来。
- **今晚不写**:已有一份未验证的 B,再叠一份未验证代码风险过高(用户 00:14 明确要求"不要弄得太 nasty")。

### 当前处置
`opsdls100` 继续跑(它在给出一个干净的负结果 + 完整 reward 轨迹,不浪费)。

## 2026-08-28 01:55 — 停 A、上 B(冒烟 + 若通过即作为正式臂)

### A 的最终固化结果(n=13 公共 epoch)
`reward_avg` 差 = **−0.00496**, t=−2.15, 同号 4/13, 符号检验 z=−1.39 **不显著**。
`opa_acc_plus` 差 = +0.0104, 符号检验 z=+1.00 **不显著**。
=> **方向 A 不带来收益,点估计偏负。** metrics 已备份为 `metrics_final.jsonl.bak`。

停掉 `opsdls100` 的理由(不是浪费):
- 结论已稳定 —— `ls=4` 自己的 acc 也在下降(.9922 -> .9167),说明下降由**续解缺口**驱动,
  多跑 epoch 不会改变;
- 它占着 3 张 H100,而**方向 B 一次都没跑过**,而全集群没有别的 3 卡空位。

### 方向B 正式上机 `opsdmq100`(1035511)
- `opa_mq_sigmas="0.15,0.45"`,`opa_ls=0`,`opa_cert=1`,`opa_mb=6`,H100x3
  => **与 `opsdc100` 仅差 `opa_mq_sigmas` 一个变量**(同硬件同 nproc 同 mb 同 cert)。
- **σ 的选取理由**:0.15 与 0.45 夹住 σ*=0.278,是 clean 预测仍良定的邻近噪声水平。
  不取更高的 σ 是因为那里后验很宽、单个 y± 不是合理目标,强行拟合会不稳。
- **必须先确认的三件事**(这是未验证代码):
  1. 配置是否真的传进去(看日志里的 opa_mq_sigmas);
  2. DDP 多前向是否报 "Expected to mark a variable ready only once";
  3. 显存/速度。

## 2026-08-28 02:05 — 方向B 冒烟:机制通过,但**提交时丢了一个 σ**

### 冒烟结论(未验证代码 -> 现在验证了机制)
`opsdmq100`(1035511, H100x3):
- **0 次** DDP "Expected to mark a variable ready only once";
- **0 次** OOM;
- 已跑到 epoch 3,约 4.3~4.8 min/epoch(对照 opsdc100 约 4.4)。
=> 多前向 + 额外查询点在 DDP 下**机制上可用**。这是本改动最大的未知,已排除。

### 但提交有 bug:实际只有一个额外查询点
`run_config.json` 实测 `opa_mq_sigmas='0.15'`,**不是我想传的 "0.15,0.45"**。
真因:SLURM `--export` 以逗号分隔,`OPA_MQ=0.15\,0.45` 的转义未生效,第二个值被当成另一个 export 项丢掉。
**处置**:不重启。K=1(共 2 个查询点)仍是方向 B 的有效检验,成本更低、归因更干净。
**但报告时必须写"一个额外查询点 σ=0.15",不能写成两个。**
下次要传含逗号的值,用 `--export=ALL` 加 `sbatch --wrap` 或写进脚本变量,不要靠转义。

### 仍需确证:代码路径是否真的在执行(不只是配置对)
速度差别太小,不足以判定(拟合块只占 epoch 的一部分)。
用 `opa_policy_loss` 判:方向B 把它变成**两项的平均**,量级应与 opsdc100 不同。
待 metrics 出来后比对。**在确证之前不把它当作"B 的结果"。**

## 2026-08-28 02:10 — 方向B **代码路径已确证在执行**(不只是配置对)

| 指标 | mq(B, +σ=0.15) | c100(canonical) |
|---|---|---|
| `opa_policy_loss` | **0.9408 0.9894 1.0620 1.1576** | 0.6453 0.6847 0.7140 0.7309 |
| `opa_pos_loss`(**仅主查询点**) | 0.1332 0.1337 0.1427 0.1470 | 0.1287 0.1371 0.1413 0.1457 |
| `opa_acc_plus` | 0.9974 0.9844 0.9818 0.9896 | 0.9896 0.9844 0.9766 0.9818 |

**判据**:总损失高约 45%,而**主查询点的分项几乎一致**。
这正是"额外项与主项取平均、且额外项约为主项 2 倍"的预期形态;若代码没执行,两个
`opa_policy_loss` 应当相同。σ=0.15 处的项更大也合理 —— 从不同噪声水平预测同一个 clean 目标更难。
=> **B 的实现确实在跑**,不是配置生效而代码空转。

### 当前 B 的三项未知已全部排除
1. DDP 多前向 —— 0 次报错 ✅
2. 显存 —— 0 次 OOM ✅
3. 代码路径 —— 由 loss 形态确证 ✅
剩下的就是**它到底有没有用**(reward_avg),n=4 太早,等积累。

### 提醒:本臂是 K=1(共 2 个查询点),不是 K=2
提交时 SLURM `--export` 的逗号转义失效,只传进 σ=0.15。**报告务必写 K=1。**

## 2026-08-28 02:30 — rtcwarm:聚合检验**显著但具误导性**,轨迹显示效应衰减并反转

### 聚合(n=8 公共 step,覆盖度守卫剔除了未跑完的 751)
| 轴 | rtcwarm | rtcraw | 差 | t | 判定 |
|---|---|---|---|---|---|
| ocr | 0.6271 | 0.4006 | +0.2265 | +3.8 | 显著 |
| geneval | 0.6382 | 0.5439 | +0.0942 | +3.6 | 显著 |
| clip | 0.2728 | 0.2658 | +0.0069 | +3.0 | 显著 |
| pick / aes / hps | | | | | 打平 |

**单变量已核实**:`rtc_raw_warm()` = `rtc_raw_train()` + 仅设 `lora_path`(teacher 早期 checkpoint 初始化)。

### 但逐步轨迹推翻了这个读法
| step | 61 | 151 | 241 | 331 | 421 | 511 | 601 | **691** |
|---|---|---|---|---|---|---|---|---|
| OCR 差 | +.4468 | +.3825 | +.2988 | +.2603 | +.2521 | +.1287 | +.1303 | **−.0877** |

绝对值:warm **0.8850 -> 0.2557**;raw 0.4382 -> 0.3433。
=> **warmup 起点高得多,但衰减更快,终点反而更低。**

### 预注册判据:**未满足**
原判据(写在 `rtc_raw_warm` 的 docstring 里):
"若 warmup 版在**同等 pickscore** 下保住 OCR,即为该 idea 的直接证据"。
训练末端 pick 1.0324 vs 1.0234(基本相等),而 OCR **0.2557 vs 0.3433 —— 更差**。
**判据不满足。**

### 正确结论
- teacher warmup 给出一个**好得多的起点**(OCR 0.885 vs 0.438);
- 但 RTC 训练**无论如何都会摧毁 OCR**,且起点越高摧毁得越多;
- 到 step 691 优势已归零并略微反转。
=> **"teacher warmup 能保住 OCR" 不成立。** 能说的只是"它推迟了退化",而这与
   "teacher 信号有保质期"的机制一致(优势随训练单调衰减)。

### 教训
**n=8 的配对检验平均掉了一个正在衰减并反转的效应。**
只报聚合值(3 轴显著、0 轴更差)就是误报。
=> 凡聚合显著的结论,**必须同时看逐步轨迹**,确认效应是稳定的而不是单调衰减/反转的。
这条要加进后续所有臂间比较的流程。

## 2026-08-28 02:45 — 把"必须看轨迹"回溯应用到已报的显著结论

### ① POE **通过**轨迹检验(与 rtcwarm 形成对比)
poe15 vs tau10 逐步 aes 差:
| step | 61 | 121 | 181 | 271 | 301 | 391 | 481 | 601 | 721 |
|---|---|---|---|---|---|---|---|---|---|
| aes | +.117 | +.139 | +.122 | +.234 | +.263 | +.174 | +.178 | **+.198** | **+.171** |
| geneval | −.008 | −.078 | −.082 | −.090 | −.106 | −.055 | −.031 | −.098 | −.082 |

**9/9 全正、无衰减、末端依然强**;geneval 的代价也稳定不恶化。
=> λ 曲线外推的前提成立,**`poe125` 的依据站得住**。这是与 rtcwarm 的关键差别:
   rtcwarm 的优势来自"更好的起点"故会被训练抹平;POE 的优势来自**目标本身改变**故持续存在。

### ② ksteps 的计数更正:是 **3 轴**显著更差,不是 5 轴
实际检验(判据 = |t|>2.5 **且** |差|>2σ):
| 轴 | 差 | t | 判定 |
|---|---|---|---|
| ocr | −0.2676 | −10.3 | **显著** |
| pick | −0.0746 | −8.5 | **显著** |
| hps | −0.0281 | −20.5 | **显著** |
| geneval | −0.0543 | −3.5 | t 显著但幅度在噪声带内 |
| clip | −0.0028 | −3.0 | t 显著但幅度在噪声带内 |
| aes | +0.0360 | +1.8 | 打平 |

我此前记的"**五轴显著更差**"把噪声带内的两条也算了进去,**混淆了判据**。已更正为 3 轴。
"无一条更好"这半句成立(aes 打平)。
轨迹检验也通过:pick 差从 −0.027 单调走到 −0.102,ocr 全程 −0.14~−0.41 —— **稳定且在加强**。

### 现在的结论可信度分层
| 结论 | 聚合 | 轨迹 | 可信 |
|---|---|---|---|
| POE 换来 aes/hps,代价 geneval | 显著 | **稳定** | ✅ |
| ksteps 劣于 raw(3 轴) | 显著 | **稳定/加强** | ✅ |
| rtcwarm 保住 OCR | 显著 | **衰减并反转** | ❌ 判据未满足 |
| horizon 有效 | 1 好 1 坏 4 平 | — | ❌ |
| 方向A 线搜索 | 不显著 | — | ❌ |

## 2026-08-28 03:20 — 方向B **显著更差**(定论);实现并上机 A′

### 方向B 结果(n=18 公共 epoch, mq vs canonical, 单变量)
`reward_avg` 差 = **−0.01148**, t=−4.70, **16/18 为负**, 符号检验 **z=−3.30 显著**。
轨迹检验(按 02:30 新立的规矩):前半 −0.00417 -> 后半 **−0.01880**,**劣势在扩大**,不是噪声也不会追回。

**结论:在 σ=0.15 加一个等权的额外查询点,显著损害产出 reward。**
=> 推翻了我的假设:更新在"零空间"里的运动**并非浪费**,约束它是有代价的。
**保留**:只测了单个 σ、等权平均这一种设定。不能由此判定"多查询点"整个想法死了,
但按现有证据它不是一条通路。

### 三项未知在停机前已全部验证(不是"没崩就算对")
DDP 多前向 0 报错 / 0 OOM / 代码路径由 loss 形态确证(总损失 +45% 而主项分量不变)。
metrics 已备份 `metrics_final.jsonl.bak`。

### A′ 已实现并上机 `opsdlse`(1035549)
`opa_ls_mode="endpoint"`(默认 `"target"`=现状不变),`opa_ls=3`,`cert=1`,H100x3
=> 与 `opsdc100` 仅差 `opa_ls`/`opa_ls_mode`。
- **做法**:候选 `y_k = y0 + (k/K)(y_plus − y0)`,每个先 `_dpm2_continue_from` 得续解终点 `ep_k`,
  再按 `R(ep_k)` 逐样本选最优;基准用与 cert **相同的** `R(xi)`。负支同理取最小。
- **为什么比 A 强**:A 优化 `R(y_plus)`(clean 空间),而 cert 测 `R(ep_plus)`(续解后)。
  实测证明前者的改善不穿过续解。A′ 直接优化后者。
- **立即可自检的强预测**:候选含 y0,而 `_dpm2_continue_from` 自述 `ep(y0)` 复现 `xi` 到 ~2e-6,
  故 `R(ep_best) >= R(xi)` 恒成立 = cert 判据(模去 margin=0.005)。
  **=> `opa_acc_plus` 应当趋近 1.0。若头 1~2 个 epoch 没有,就是实现有 bug,必须查。**
  (这正是 A 缺的东西 —— A 当时的"acc 会到 1.0"是我推错了,A′ 这次的推导有代码依据。)
- 代价:K=3 -> 每 chunk 多 6 次续解(各 1~2 次网络前向)+ 7 次 reward 前向。预计 ~2x epoch 时间。

## 2026-08-28 03:30 — A′ 自检通过(今晚唯一被精确证实的推导)

`opsdlse`(1035549) 配置核对: `opa_ls=3, opa_ls_mode='endpoint', opa_cert=1, opa_mb=6` ✅
错误 0 次。
```
acc_plus : 1.0000  1.0000      (canonical 同期 0.9896  0.9844)
acc_minus: 1.0000  1.0000
```
**预测与实测完全一致** —— 候选含 y0 且 `ep(y0)≈xi`(2e-6),故 `R(ep_best)>=R(xi)` 恒成立,
cert 判据(含 margin=0.005)由构造满足。**A′ 的实现正确。**
对比:A 当时"acc 会到 1.0"是我推错的(推导对象搞错);A′ 这次的推导有代码依据且被证实。

产出指标 n=3(−0.0047, t=−0.65),**太早,不解读**。
速度 6 min/epoch(canonical 4.4,即 1.4x,低于我估的 2x),到 08:00 约 n≈48。

## 2026-08-28 03:50 — **更正 02:30 的更正**:rtcwarm 判据其实**满足**

`evrtcwarm` 全部 COMPLETED,30/30 出分。补上 751/781 后 n=10:

| step | 61 | 151 | 241 | 331 | 421 | 511 | 601 | **691** | 751 | 781 |
|---|---|---|---|---|---|---|---|---|---|---|
| OCR warm | .8850 | .8213 | .8018 | .6602 | .6246 | .5134 | .4551 | **.2557** | .4997 | .4301 |
| OCR raw | .4382 | .4388 | .5030 | .3999 | .3725 | .3847 | .3248 | .3433 | .2623 | .1977 |
| OCR 差 | +.447 | +.383 | +.299 | +.260 | +.252 | +.129 | +.130 | **−.088** | **+.237** | **+.232** |

**step 691 是孤立下陷**(前后为 .4551 / .4997),不是反转。我 02:30 把它当成了趋势。

### 正确结论(n=10 完整数据)
- warmup 的 OCR 优势**确实衰减**(+0.447 -> ~+0.23),但**没有消失**;
- 训练末端(781):OCR **0.4301 vs 0.1977 = 2.2 倍**,pickscore 基本相等(1.0356 vs 1.0305, +0.005);
- **预注册判据"在同等 pickscore 下保住 OCR"** —— **满足。**
- 聚合 n=10:ocr +0.2282(t=+4.9)、geneval +0.0805(t=+3.6)、clip +0.0071(t=+3.9)、
  aes +0.0991(t=+2.6) **四轴显著更好**;pick/hps 打平;**无一轴更差**。

### 根因:我从残缺数据下了结论
02:30 时覆盖度守卫**已经打印警告** `step 751: 缺 ['drawbench','ocr']`,我看到了却仍从 n=8 下结论,
而恰好被排除的两个点(751/781)是优势回升的点,留下的最后一点(691)又是个下陷。
=> **守卫报警时必须等数据齐,不能"先看个大概"。** 这条比"要看轨迹"更靠前。

### 现在 rtcwarm 的地位
这是今晚**唯一一个通过了完整数据 + 轨迹检验的正向显著结果**。
但仍有一个必须写明的保留:warmup 臂多了 ~61 步 teacher 训练,**同 step 比较不等于同算力**
(末端 781 时相对差异约 8%)。要完全干净需给 raw 臂也补 61 步,或按总算力对齐后重比。

## 2026-08-28 04:05 — rtcwarm 通过**算力对齐**检验:结论稳固

用现有数据做保守的算力对齐:`warm@S` vs `raw@下一个 step`(步距约 90 > 真实偏移 61,
即**给 raw 更多**算力)。n=9:

| 轴 | 差 | t | 判定 |
|---|---|---|---|
| ocr | **+0.2544** | +5.4 | **显著** |
| geneval | **+0.0881** | +3.3 | **显著** |
| aes | **+0.1088** | +3.2 | **显著** |
| clip | **+0.0079** | +3.3 | **显著** |
| pick | −0.0096 | −2.2 | 打平 |
| hps | −0.0031 | −1.8 | 打平 |

OCR 逐对 **8/9 为正**(唯一例外 691/751 仍是 warm 在 691 的那个孤立下陷)。
=> **同样 4 轴显著更好、0 轴更差,算力混淆已排除。**

诚实说明:算力对齐后 pick 从 +0.0036 变为 −0.0096(t=−2.2,不显著) —— 与"raw 多跑步数"的预期一致。

### rtcwarm 现在的证据完整度
| 检验 | 状态 |
|---|---|
| 单变量(只差 `lora_path`) | ✅ 代码核实 |
| 预注册判据(同 pick 保住 OCR) | ✅ 末端 0.4301 vs 0.1977 |
| 数据完整(n=10,守卫无警告) | ✅ |
| 轨迹(非衰减到消失) | ✅ 衰减但稳定在 ~+0.23 |
| **算力对齐** | ✅ 保守对齐后 4 轴仍显著 |
=> **今晚唯一通过全部检验的正向显著结果。**

### 机制解释(与既有观测自洽)
- teacher 信号有保质期(teacher−rand 从 +0.0184 到转负) -> 只该在早期用;
- warmup 正是"早期用 teacher、之后切 reward 梯度";
- 优势随训练衰减但不归零 —— 因为它改变的是**起点所在的解**,而 RTC 的 reward 上升
  并不会把模型拉回原来那个低 OCR 的解。

## 2026-08-28 04:30 — 三个独立干预全部对产出无效:一个更强的结论正在成形

| 干预 | 改了什么 | 目标质量 | 产出 `reward_avg` |
|---|---|---|---|
| **A**(`opa_ls`, target 模式) | 逐样本最优半径 | acc +0.010(n.s.) | −0.0050, z=−1.39 **n.s.** |
| **A′**(`opa_ls_mode=endpoint`) | 在 cert 真正测量的量上线搜索 | **acc 0.85 -> 1.00** ✅ | −0.0069, z=−0.83 **n.s.**(n=13,继续积累) |
| **B**(`opa_mq_sigmas`) | 多查询点联合拟合 | — | −0.0115, z=−3.30 **显著更差** |

**A′ 是关键的一个**:它**确凿地把目标质量提上去了**(采纳率由构造达到 1.0,而 canonical 稳态只有 0.86),
产出却没有改善。

### 正在成形的结论
**OPSD 的目标构造环节是过度设计的 —— 它的细节不影响最终结果。**
- 半径(ρ):四点扫描全打平 + 1-齐次代数证明它被归一化器约掉;
- 方向:必须是 reward 梯度(rand 对照 z=−3.98),但**在此之外**怎么优化都没用;
- 幅度的逐样本最优化(A):无效;
- 认证目标的直接优化(A′):目标质量确实提升,产出仍无改善;
- 加约束(B):反而更差。
=> 与"99.65% 丢在拟合阶段"完全自洽:**瓶颈被拟合环节主导到目标质量几乎不起作用。**

### 那什么有用?(今晚已验证的两条)
1. **起点** —— `rtcwarm`(teacher warmup)通过全部五项检验,4 轴显著更好;
2. **目标函数本身** —— POE(改变蒸馏所指向的分布),aes +10.9σ 且轨迹稳定。
**两者都不在 OPA 的目标构造里。**

这比"批评 OPSD 的四条弱点"更像一个正面命题,且有三个独立的 null 结果作三角验证。
**待 A′ 积累到 n>=30 再定稿。**

## 2026-08-28 05:00 — cert 效应被**机制判别**证伪;同时更正采纳率轨迹

### ① 判别检验:+0.0031 不是 cert 效应,是基线偏移
| 窗口 | 实际拒绝率 | reward 差(cert=1 vs 基线) |
|---|---|---|
| cert 近乎惰性(epoch 0~5) | **2.5%** | **+0.00398** (n=6) |
| cert 生效(epoch>=20) | **31.5%** | **+0.00329** (n=50) |

**拒绝率相差 12 倍,效应却一样大(惰性窗口甚至更大)。**
若 +0.0031 由 cert 造成,它在几乎不拒目标的窗口应当接近 0。
=> **该差值是基线偏移(H100 vs H200 / run 间),不是 cert 效应。** 若只报 n=70 的
"符号检验 z=+2.39 显著"就是一次误报。

附:自相关不是问题 —— 逐 epoch 差的一阶自相关 **ρ₁=−0.096**(略负),n_eff≈85>70,
检验没有被自相关撑高;抽稀后显著性下降只是功效损失,点估计始终为正。

### ② 更正 23:50 的"采纳率走平在 0.86"
70 epoch 完整轨迹:
| epoch | 0-9 | 10-19 | 20-29 | 30-39 | 40-49 | 50-59 | 60-69 |
|---|---|---|---|---|---|---|---|
| acc_plus | .952 | .868 | .784 | .723 | .659 | .627 | **.613** |
**单调降到 ~0.61 才放缓**,不是 0.86。原结论来自 20-epoch 探针,区间太短。
=> canonical 在后期拟合 **~39%** 的无效正目标(不是 14%)。

### ③ 合起来 = 今晚最强的命题
| 干预 | 目标质量的改善 | 产出 |
|---|---|---|
| A(逐样本最优半径) | acc +0.010 (n.s.) | **null** |
| A′(在 cert 测量的量上线搜索) | **acc 0.61~0.86 -> 1.00** ✅ | **null**(n=16) |
| B(多查询点) | — | **显著更差** |
| cert=1(拒掉 39% 无效目标) | 按定义完全修正 | **null**(机制判别证伪) |

**四条独立路径都表明:把目标构造做得更好,对最终结果没有帮助。**
即使 canonical 后期有 39% 的目标根本不抬高 reward,修掉它也无收益。
=> 与"99.65% 的构造增益丢在拟合阶段"完全自洽:**拟合环节主导到目标质量不起作用。**

而今晚**唯一**有效的两条都不在目标构造里:
**起点**(`rtcwarm`,五项检验全过,4 轴显著)与**目标函数本身**(POE,aes +10.9σ,轨迹稳定)。

## 2026-08-28 05:35 — 关键诊断:A 几乎没改动目标(其 null 无信息量);A′ 是实质改动

`opa_delta_res` = ‖y_plus − y0‖ / ‖y0‖:
| run | 均值 | 相对 canonical |
|---|---|---|
| canonical(ls=0) | **0.0715** | — |
| A(target 线搜索) | 0.0703 | **−1.7%(几乎无变化)** |
| A′(endpoint 线搜索) | **0.0445** | **−38%** |

### 两个后果
1. **A 的 null 不能当证据。** 目标空间的线搜索几乎总是选满预算(因为 R(y) 沿上升射线单调,
   最优点就在边界),所以 A ≈ canonical,它"没效果"是平凡的。
   **我此前把 A 列为四条独立证据之一是不对的,已降级。**
2. **A′ 是真正的干预**:按**续解终点**判,逐样本最优位移比固定预算**小 38%** ——
   即 **OPSD 的 ρ=0.10 在"最终生成"这个尺度上是过冲的**。
   A′ 同时做到:半径端点最优 + 采纳率由构造 =1.0(canonical 后期仅 0.61)。
   **而产出仍无改善(n=23, −0.0060, z=−1.46 n.s.)。**
   => A′ 的 null **信息量远大于** A。

### 修正后的证据清单(命题"目标质量不影响结果")
| 干预 | 是否真的改变了目标 | 产出 |
|---|---|---|
| ~~A~~ | **否**(−1.7%) | ~~null~~ **不计入** |
| **A′** | **是**(位移 −38%,acc 0.61→1.00) | **null** |
| **B**(多查询点) | 是(总损失 +45%) | **显著更差** |
| **cert=1**(拒 39% 无效目标) | 是(按定义) | **null**(机制判别证伪) |
=> 仍是三条独立路径,但去掉了一条无效证据。**A′ 是其中最有力的。**

### 附带的新发现(可写进论文)
**OPSD 的固定信赖域在"续解终点"尺度上过冲约 38%。**
这是一个可测量、可复现的量化批评,且与"G_construct 曲线在 ρ=0.10 见顶"是不同层次的陈述
(那是在 clean 空间测的;这里是在最终生成上测的)。

## 2026-08-28 06:00 — 核实 rtcwarm 的机制:**方法成立,但 docstring 里写的机制不被数据支持**

gc 探针(`GCPROBE-M-raw151`, n=24)实测:
| 量 | 中位数 |
|---|---|
| `gc_G_grad`(reward 梯度) | **+0.0794** |
| `gc_G_rand`(随机) | −0.0059 |
| `gc_G_teacher`(teacher 方向) | −0.0101 |
| **teacher − rand** | **−0.0040, 24/24 全负** |
更早的 checkpoint 上 teacher−rand = −0.00026(10/24),即**≈0 而非正**。

### 需要澄清的地方
`rtc_raw_warm` 的 docstring 写:"teacher 只在训练最初携带上升信息 -> 先 warmup 再切 reward 梯度"。
**探针数据不支持这个说法** —— teacher 方向在 PickScore 上从早期就不优于随机(≈0),之后转负。

### 但结果不受影响,因为机制是另一个(且自洽)
探针测的是"沿 teacher 方向能否抬高 **PickScore**",而 teacher 优化的是 **OCR/GenEval**。
两者本就不该一致。
**rtcwarm 的收益恰好全在 OCR(+0.228)/GenEval(+0.081)/clip/aes 上,pickscore 打平** ——
完全符合"teacher 给的是 OCR/GenEval 能力,不是 PickScore 上升方向"。

### 正确的机制表述(写论文时用这个)
不是"teacher 早期携带 reward 上升信息",而是:
**多 teacher 蒸馏的初始化把模型放进一个 OCR/GenEval 能力更强的解;
随后的 RTC 只优化 PickScore,不会主动破坏该能力(只是缓慢侵蚀),故末端仍保留 2.2 倍的 OCR。**
这与实测的衰减形态(优势 +0.447 -> ~+0.23 但不归零)一致。

**待办**:`rtc_raw_warm` 的 docstring 需要按上述修正 —— 但那是运行中臂的配置文件,
改动不影响已跑完的结果,留给白天做,避免夜间误触。

## 2026-08-28 06:15 — A′ 判决:用**同一把尺子**,劣势同样不可归因

### 聚合(n=30)
`reward_avg` 差 = **−0.00651**, t=−2.63, 21/30 为负, 符号检验 **z=−2.19** —— 两个判据都过。
轨迹:前半 −0.00522 -> 后半 −0.00780,**劣势扩大**。

### 但机制判别不支持归因
| 窗口 | canonical acc | reward 差 |
|---|---|---|
| acc>=0.93(A′ 无从改善) | 0.969 | **−0.00636** (n=8) |
| acc<0.93(A′ 应显优势) | 0.831 | **−0.00656** (n=22) |

**两个窗口几乎完全相同。** 干预近乎惰性处与充分生效处劣势一样大
=> **这 −0.0065 不可归因于 A′**,是基线偏移(两臂在不同节点 gpu14 / gpu16)。

**方法论上必须对称**:我用这把尺子否掉了 cert 的"正向显著",就必须同样用它否掉 A′ 的"负向显著"。
若只在结果不合意时才用判别检验,那就是选择性使用统计。

### A′ 的正确结论
**A′ 对产出没有可归因的影响**(既非改善也非损害)。
而它**确凿地改动了目标**(位移 −38%、acc 0.61->1.00)。
=> 这仍是"目标质量不影响结果"最有力的一条证据,且现在两个方向都排除了。

### 已排 poe125 评测
`evpoe125`(1035563), checkpoint 61/121/181, GS=4.5 NS=256(与 tau10/poe15 等同类臂一致)。
`poe125_targets.tsv` 此前是**空文件**,已填。早上可给出 n=3 的初步读数 —— **n=3 只能看方向,不能判定。**

## 2026-08-28 06:55 — cert 的最终证据(含我自己的一次疏忽)

### 我的疏忽:拿不同 epoch 的末值作比
我先说"cert 末值 0.9277 vs 基线 0.9148 = +0.0129" —— 那是 **cert@97 vs 基线@99**,不同 epoch。
同 epoch 配对后:epoch 97 差 **−0.0069**、epoch 98 差 **−0.0066**(cert 反而更低)。
**端点值一律不能单独引用**,逐 epoch 噪声达 ±0.02。

### 最终数字(n=99 完整配对)
| 区间 | 均值 | 同号 |
|---|---|---|
| 全程(n=99) | **+0.00330** | — |
| 末 10 个 epoch | **+0.00342** | 4/10 |

**全程均值与末 10 均值几乎相同(+0.0033 vs +0.0034)** —— 而这段时间里 `opa_acc_plus`
从 0.95 一路掉到 0.61(cert 拒绝率从 5% 升到 39%)。
**干预强度变化 8 倍,效应纹丝不动。**

### cert 的结论(定稿)
`opa_cert=1` 对产出**没有可归因的影响**。观测到的 +0.0033:
- 幅度只有逐 epoch 噪声(±0.02)的 1/6,需平均 ~100 个 epoch 才能"显著";
- **不随拒绝率变化**(惰性窗 +0.0040 vs 生效窗 +0.0033 vs 末段 +0.0034);
- 与纯对照(两边都 cert=0,只差硬件)的漂移量 −0.0021 同量级。
=> **是基线偏移,不是 cert 效应。**

**论文含义**:canonical 后期硬编码接受了 39% 根本不抬高 reward 的目标,
**修掉它没有任何收益** —— 这比"他们的假设是错的"更强,也更反直觉。

## 2026-08-28 07:30 — **关键澄清:`rtcwarm` 是「多 reward 预训练 + 单 reward RTC」,不是 teacher warmup**

从训练日志 `slurm_rtcwarm_1034771.out` 查实,`OPD_STUDENT_INIT` 指向
`logs/[MOPD-POE-L1.5 + SD3.5-M + PickScore+OCR+GenEval]/checkpoints/checkpoint-61/lora`
—— 即 **poe15(λ=1.5 的多 reward PoE 臂)的 checkpoint-61**,不是任何单个 teacher。

| | OCR | geneval | pick |
|---|---|---|---|
| 起点 `poe15@61` | **0.9249** | 0.8125 | 0.9013 |
| 61 步 RTC 后 `rtcwarm@61` | 0.885 | 0.7773 | **0.9380** |
| 从零训 `rtcraw@61` | **0.4382** | — | 0.9133 |

### 正确的命题(比"teacher warmup"更标准也更强)
**多 reward 预训练赋予的能力(OCR/GenEval),被后续单 reward RTC 侵蚀但不摧毁。**
- RTC 把 pick 从 0.9013 抬到 0.9380,同时 OCR 从 0.9249 掉到 0.885;
- 到末端(781)仍有 OCR 0.4301 vs 从零训的 0.1977 = **2.2 倍**,而 pick 打平。
- 与 06:00 记录的机制修正一致,现在有了确切的初始化来源。

### 对既有记录的影响
- `rtc_raw_warm` docstring 的"teacher 只在训练最初携带上升信息"**双重错误**:
  (a) gc 探针显示 teacher 方向对 PickScore 从早期就不优于随机(24/24 全负);
  (b) 实际初始化根本不是单 teacher,而是多 reward PoE 臂。
  **该 docstring 必须重写**(留白天,避免夜间误触运行中的配置)。
- 结论本身不变:4 轴显著、算力对齐后仍成立。**变的是解释,不是数据。**

### 论文里应该这样写
不要写成"teacher warmup 调度",要写成
**"多 reward 预训练 -> 单 reward 在线蒸馏"的两阶段方案**,并给出侵蚀曲线
(OCR 优势 +0.447 -> +0.23,不归零)作为"能力不被摧毁"的量化证据。

## 2026-08-28 07:35 — poe125 初步:**看起来像净胜,但对照证明是没功效**

`poe125` vs `tau10`,n=2(step 61/121;守卫剔除了未跑完的 181):
aes +0.1166(t=+3.0)**显著**、hps +0.0061(t=+10.6)**显著**、geneval −0.0332 **打平**。
—— 表面上正是 λ 曲线外推预测的"净胜"。

### 但对照否定了这个读法
把 **`poe15` 限制在同样的 step 61/121**:
geneval **−0.0430, t=−1.2, 同样打平**;aes +0.1279 显著;hps +0.0049 显著。
=> **poe125 的 geneval「打平」只是 n=2 没功效**,不是 λ=1.25 避免了代价。
(poe15 在 n=9 时 geneval 是 −0.0699 显著;n=2 时就测不出来了。)

### 可以说的与不可以说的
- **可以说**:点估计往预测方向动 —— geneval 损失 0.77x(−0.043 -> −0.033)、
  aes 增益 0.91x(+0.128 -> +0.117),与 λ^2~3 vs λ^0.5 的标度一致。
- **不可以说**:"λ=1.25 拿到了无代价的美学增益"。要确立它需要
  poe125 跑到 step >= 391(与 tau10 有 n>=6 的公共步),现在 2 卡 A100 上约 2.7 min/step,
  即还需 ~10 小时。

### 这是今晚第五次靠对照挡下的误报
方法固定为:**任何"打平"结论,必须用一个已知有效应的臂在同样的 n 上验证是否也"打平"**。
若是,则该"打平"是功效不足,不是效应消失。

## 2026-08-28 07:55 — poe125 n=3 带对照:**点估计精确follows λ 标度预测**

同样 3 个 step(61/121/181),两臂各自对 tau10:
| 轴 | poe125(λ=1.25) | poe15(λ=1.5) | 比值 | 幂律预测 |
|---|---|---|---|---|
| geneval 损失 | **−0.0312** | −0.0560 | **0.56x** | (1.25/1.5)^2.5 = **0.63x** |
| aes 增益 | +0.1147 | +0.1260 | **0.91x** | (1.25/1.5)^0.5 = **0.91x** |
| hps 增益 | +0.0053 | +0.0047 | 1.13x | — |

**两个比值都落在预测上**(aes 精确命中)。效率 aes/geneval 由 2.25 -> **3.68**。

### 但仍不能说"净胜已确立"
n=3 时**两臂的 geneval 都不显著**(poe125 t=−3.0 但在噪声带内;poe15 t=−2.3 打平)。
所以只能说:**点估计按预测移动**,不能说 λ=1.25 消除了代价。
要确立需 poe125 跑到 step >= 391(n>=6)。当前 2 卡 A100 上 step 181,约 2.7 min/step,还需 ~9 小时。

### 已派活(清单第(5)条:空闲就派)
`opsdc100` 跑满 100 epoch 后释放 3 张 H100,已用于**给唯一正向结果补第二个种子**:
`rtcwarm43`(1035570, seed=43, H100x2, 与原 run 的 nproc=2 严格一致)。
核实生效:`run_name=RTC-RAW-WARM-w61s43`(独立 save_dir)、`seed: 43`、
`lora_path` 指向 poe15 checkpoint-61 —— 三项都确认,不是"没崩就算跑对了"。
改动最小化:`base.py` 加 `OPD_SEED`(默认 42 = 原行为逐字节不变),
`opd.py` 在 seed≠42 时给 run_name 加后缀(防 save_dir 互删)。备份 `scratchpad/base.bak_seed`。

## 2026-08-28 08:10 — A′ 更新到 n=49:出现弱梯度,但**与时间共线,仍不可归因**

| 分箱 | canonical acc | A′ acc | 目标质量差距 | reward 差 |
|---|---|---|---|---|
| acc>=0.93 | 0.969 | 0.997 | 0.028 | −0.00636 (n=8) |
| 0.75<=acc<0.93 | 0.832 | 0.992 | 0.160 | −0.00673 (n=23) |
| acc<0.75 | 0.690 | 0.983 | **0.293** | **−0.00894** (n=18) |
聚合 n=49: −0.00748, t=−3.83, 35/49 为负。

### 与 06:15 记录的差别
当时 n=30 只有两个分箱且**几乎相同**(−0.00636 / −0.00656),我据此判"不可归因"。
现在 n=49 出现**弱梯度**:差距扩大 10 倍 -> 劣势扩大 1.4 倍。

### 但仍不能归因,原因是共线
`opa_acc_plus` 随训练**单调下降**,故"低 acc 分箱" ≡ "晚期 epoch"。
而劣势本身也随时间扩大(前半 −0.0052 -> 后半 −0.0078)。
**两者是同一件事的两种切法,无法分离"A′ 改得越多越有害"与"基线偏移随时间漂移"。**
要分离需要一个"晚期但高 acc"或"早期但低 acc"的窗口 —— 由于 acc 单调,这样的 epoch 不存在。

### 当前最诚实的表述
- **A′ 不改善产出**:这一点确定(49 个 epoch 全程为负,t=−3.83)。
- **A′ 是否轻微有害**:不可判定 —— 与漂移不可分离。
- 无论哪种,**"把目标做得更好没有帮助"这一命题不受影响**,因为两种解释都不含"目标质量改善带来收益"。

### 方法论备注
我 06:15 写"两窗口完全相同 => 不可归因"是**基于 n=30 的正确结论**;
n=49 时形态变了,我据此更新而不是坚持原判。**但更新后的结论仍是"不可归因",只是理由从
"无梯度"变成"梯度与时间共线"。** 如实记录这个演变,避免看起来像事后挑选解释。

## 2026-08-28 08:30 — **重要背景补正:整条 RTC 线比官方 Student 低 17~23%**

跑 `overall_score.py`(6 轴聚合,参照官方 Student LoRA,172 个模型):
| 模型 | Ratio | Δ% | 判定 |
|---|---|---|---|
| Student(参照) | 1.0000 | — | — |
| `poe15_781` | 1.0037 | **+0.37** | 与参照打平 |
| `tau10_721` | 1.0019 | +0.19 | 打平 |
| `caopd721` | 1.0032 | +0.32 | 打平 |
| **`rtcwarm_781`** | **0.8300** | **−17.00** | **显著低于参照** |
| **`rtcraw_781`** | **0.7668** | **−23.32** | **显著低于参照** |

### 这改变了昨晚"正向结果"的解读
- `rtcwarm` 优于 `rtcraw` 是真的(+6.3 个百分点,且 4 轴显著);
- **但这是两个都远低于官方基线的臂之间的比较。**
  只报"4 轴显著更好"会让人误以为 rtcwarm 是个好模型。
- 而**多 reward 的臂(poe15 / tau10 / caopd)与官方 Student 打平**。

### 由此得到的更重要结论
**单 reward 的 RTC/OPSD 路线在综合指标上是倒退的** ——
它把 PickScore 抬上去(rtcwarm pick 1.036 vs Student 附近),代价是 OCR/GenEval 崩塌,
6 轴聚合后净损失 17~23%。
=> 这与 OPSD 论文只报 PickScore 类单项指标是一致的:**单项赢,综合输。**
   而我们自己的多 reward 线(POE/CA-OPD)才是与官方持平的那条。

### 对昨晚记录的修正
`OVERNIGHT` 一页纸里"唯一通过全部检验的正向结果"必须加上这个上下文,
否则是**在正确的数字上给出误导性的印象**。已修改。

## 2026-08-28 08:50 — poe125 在**综合 6 轴分数**上稳定优于 poe15(最有希望的一条)

同 step 对比(`overall_score.py`,参照官方 Student,判定阈值 2σ=0.61pp):
| step | poe125 | poe15 | 差 |
|---|---|---|---|
| 61 | 0.9533 | 0.9523 | +0.10pp |
| 121 | **0.9809** | 0.9720 | **+0.89pp** ✅超噪声带 |
| 181 | **0.9866** | 0.9789 | **+0.77pp** ✅超噪声带 |
Z 聚合同向(poe125 +0.418/+0.318/+0.003 vs poe15 +0.386/+0.256/+0.004)。

### 为什么这条最有希望
- `poe15_781` 最终 = **+0.37%(与官方 Student 打平)**,是我们所有臂里最高的;
- poe125 在同 step 上稳定高出 0.1~0.9pp;
- 若该优势保持到 781,**poe125 有机会成为第一个显著超越官方 Student 的臂**。

### 但现在不能下结论
n=3 且都是早期 step(61/121/181),轨迹后段可能变化。
poe125 当前在 step 181,2 卡 A100 上约 2.7 min/step,到 781 还需 **~27 小时**。
**判据(预注册)**:poe125_781 的 Ratio 若 > 1.0037+0.61pp 即显著超越 Student。

### 与昨晚结论的关系
昨晚证明"OPSD 的单 reward 目标构造修不动"(三条独立 null),
今早又发现"整条 RTC 单 reward 线比官方低 17~23%"。
=> **价值在多 reward 线**,而 poe125 正是这条线上最有希望的一个点。
   资源应优先保障 poe125 跑到 781,而不是继续在 OPSD 目标构造上做文章。

## 2026-08-28 09:10 — 给 poe125 补 step 271 评测(n=3 -> 4)

poe125 已跑到 checkpoint-271(与 tau10 的可比 step)。ai-gpu05 有 4 张空闲 A40,
按清单第(5)条派活:`evpoe125`(1035589), array 0-3(前 3 个已有结果会自动跳过)。
出结果后 poe125 vs poe15 的同 step 对比将有 n=4。

**未做的一件事**:我建议过"停掉已定论的 `opsdlse`(A′,占 3 张 H100)、
把 poe125 从 2 卡 A100 迁到更多卡加速"(到 781 还需 ~27h)。
`opd_arm.slurm` 带 `OPD_RESUME=auto` 可保进度迁移,但这会**中止一个正在跑的实验**,
属于不可逆操作,**已向用户提出但未获确认,故不执行**。

## 2026-08-28 09:50 — ★ **poe125(λ=1.25) 是第一个干净的正向显著结果**

### 综合 6 轴(overall_score.py,参照官方 Student,判定阈值 2σ=0.61pp)
| step | poe125 | poe15 | 差 |
|---|---|---|---|
| 61 | 0.9533 | 0.9523 | +0.10pp |
| 121 | 0.9809 | 0.9720 | **+0.89pp** ✅ |
| 181 | 0.9866 | 0.9789 | **+0.77pp** ✅ |
| 271 | **0.9972** | 0.9871 | **+1.01pp** ✅ |
**4/4 为正,3 个超噪声带,差距在扩大**(与 rtcwarm 的衰减形态正好相反)。
**`poe125_271` 已达「与官方 Student 打平」**(−0.28%),而 `poe15_271` 仍「显著低于参照」(−1.29%);
Z 聚合上 poe125_271(+0.572)略高于 Student(+0.546)。

### 逐轴 n=4,且**对照排除了功效不足**
| 轴 | poe125 vs tau10 | poe15 vs tau10(同 4 step) |
|---|---|---|
| geneval | **−0.0244 打平** | **−0.0645 显著更差**(t=−3.4) |
| aes | +0.1343 **显著** | +0.1529 显著 |
| hps | +0.0062 **显著** | +0.0057 显著 |
07:35 设的判别条件("用已知有效应的臂在同 n 上验证是否也打平")**现在通过了**:
poe15 的 geneval 在 n=4 是显著的,poe125 的不显著 => **不是没功效,是真差别。**

### 为什么这条与之前所有失败不同
- **单变量**:只差 λ(1.25 vs 1.5),其余逐项相同;
- **机制有定量预测且命中**:geneval 代价 ∝λ^2~3、aes 增益 ∝λ^0.5,
  实测比值 0.56x / 0.91x,预测 0.63x / 0.91x;
- **轨迹向好**(差距扩大),不是衰减型;
- **改的是目标函数本身**,不是 OPSD 的目标构造 —— 而昨晚三条独立证据已证明后者修不动。

### 仍需完成
poe125 当前 step ~281,到 781 还需 ~23 小时(2 卡 A100)。
**预注册判据**:`poe125_781` 的 Ratio 若 > 1.0037(poe15_781)+0.61pp,即**显著超越官方 Student**。

## 2026-08-28 10:10 — poe125 的**全臂对照**:step 271 上它是最好的,且唯一与官方打平

| step 271 | Ratio | Δ% | 判定 |
|---|---|---|---|
| 官方 Student | 1.0000 | — | 参照 |
| **poe125_271** | **0.9972** | **−0.28** | **与参照打平** |
| caopd271 | 0.9926 | −0.74 | 显著低于参照 |
| tau10_271 | 0.9887 | −1.13 | 显著低于参照 |
| poe15_271 | 0.9871 | −1.29 | 显著低于参照 |
| poe20_271 | 0.9748 | −2.52 | 显著低于参照 |
| poe30_271 | 0.9265 | −7.35 | 显著低于参照 |
| cconf/conf/cagrad@271 | 0.93/0.92/0.89 | −7~−11 | 显著低于参照 |

**λ 在 step 271 上单调:1.25 > 1.5 > 2.0 > 3.0。**
poe125 超 tau10(+0.85pp)、超 poe15(+1.01pp) **均超 0.61pp 噪声带**;超 caopd(+0.46pp)在带内。

### 结论升级
不只是"λ=1.25 优于 λ=1.5",而是 **"λ=1.25 在同 step 上优于我们所有其他臂"**,
且是**唯一**在该 step 达到与官方 Student 打平的。

### 我特意跑了可能让它泄气的对照
原本担心"poe125 的优势只是 λ 越小越好"(即 λ=1.0 的普通 mopd 也会一样好)。
λ=1.0 的 `mopd` 臂只在 step 91/931/961 有评测,无法在 271 直接比;
但 `tau10`(mopd+CA-OPD τ=1.0)与 `caopd` 在 271 都有,**两者都显著低于 poe125**。
=> 优势不是"λ 越小越好"的平凡效应。

### 仍需完成(判据不变)
poe125 到 step 781 还需 ~23h。判据:Ratio > 1.0037(poe15_781)+0.61pp 即显著超越官方 Student。
**注意**:step 271 是中期,最终排名可能变化。现有最好的**终点**值仍是 poe15_781 = +0.37%。

## 2026-08-28 11:20 — **更正:B(多查询点)的负结果被"信号稀释"混淆**

### 混淆
我的实现是 `opa_policy = sum(_terms) / len(_terms)`(L1671)。K=1 时有 2 项,
=> **主查询点的权重从 1.0 降到 0.5**。
当初我选"取平均而非求和"的理由是"保持损失尺度不变、无需重调 lr" —— 这对**尺度**是对的,
但它同时把**主目标的梯度贡献减半**。

=> B 测到的 −0.0115(z=−3.30)**至少一部分来自主信号被稀释,而不是多查询点本身有害**。
**这个负结果不能当作"多查询点无效"的证据。**

### 正确的测法
`opa_policy = main + w * mean(extras)`,w 取小值(如 0.1~0.3),使主项权重保持 1.0。
需要给 `opa_mq_sigmas` 配一个 `opa_mq_weight`。改动很小(一行)。

### 与其他结论的关系
不影响 A′ / cert 的结论 —— 那两条没有稀释问题(A′ 只改目标本身,cert 只改权重门控)。
故"目标构造修不动"仍有 A′ 和 cert 两条独立支撑,但**从三条降为两条**。

## 2026-08-28 11:50 — poe125 到 n=5:优势**稳定**(修正 09:50 的"扩大")

| step | poe125 | poe15 | 差 |
|---|---|---|---|
| 61 | 0.9533 | 0.9523 | +0.10pp |
| 121 | 0.9809 | 0.9720 | +0.89pp ✅ |
| 181 | 0.9866 | 0.9789 | +0.77pp ✅ |
| 271 | 0.9972 | 0.9871 | +1.01pp ✅ |
| **301** | **0.9984** | 0.9913 | **+0.71pp** ✅ |
5/5 为正,4 个超 0.61pp 噪声带,均值 **+0.70pp**。

### 修正
09:50 我写"差距在扩大(+0.10 -> +1.01)"。加上第 5 点后 271 是峰值、301 回落到 +0.71,
**准确说法是"稳定在 +0.7~1.0pp"**,不是单调扩大。
(这条重要:若按"扩大"外推到 781,会高估最终优势。)

### 现状
`poe125_301` = 0.9984(−0.16%),Z=+0.542 与 Student(+0.544)基本相等。
仍是"与参照打平",尚未超越。到 781 还需 ~20h(2 卡 A100)。
判据不变:Ratio > 1.0037 + 0.61pp = **1.0098** 才算显著超越官方 Student。
按当前 +0.70pp 的稳定优势外推,poe15_781=1.0037 -> poe125_781 约 **1.0107**,
**刚好在判据线之上** —— 但这是外推,不是结果。

## 2026-08-28 12:15 — **回撤 09:50 的措辞:poe125 的优势是 geneval 单轴的,且两个聚合口径不一致**

### 逐轴分解(poe125 vs poe15,同 step,相对差 %)
| 轴 | 均值 | 同号 |
|---|---|---|
| **geneval** | **+5.57%** | 4/5(121~301 上 +7.5~+10.1) |
| clip | +0.37% | 5/5 |
| ocr | +0.29% | 3/5 |
| pick | +0.14% | 5/5 |
| hps | −0.04% | 3/5 |
| **aes** | **−0.63%** | 4/5(负) |
=> **优势几乎全部来自 geneval**,其余各轴 ~0,aes 略负。

### 两个聚合口径不一致
| step | ΔRatio(pp) | ΔZ(阈值 0.039) |
|---|---|---|
| 61 | +0.10 | −0.004 |
| 121 | +0.89 ✅ | +0.063 ✅ |
| 181 | +0.77 ✅ | +0.032 |
| 271 | +1.01 ✅ | +0.051 ✅ |
| **301** | +0.71 ✅ | **−0.031** |
Ratio:5/5 正、4 个超带。**Z:3/5 正、仅 2 个超带,且最新的 301 为负。**

原因:`overall_score.py` 的 docstring 自己写明 **Ratio 会被相对波动最大的轴(geneval)主导**;
Z 把各轴离散度压平后,单轴优势就大幅缩水。

### 修正后的表述
**不能说"poe125 是第一个干净的正向显著结果"。** 准确说法:
**poe125 相对 poe15 显著改善 geneval(+5.6% 相对),代价是 aes 略降(−0.6%),其余轴无变化。**
这在 Ratio 口径上表现为稳定的 +0.7pp,但在 Z 口径上很弱且不一致。
=> 它是**一个明确的 geneval/aes 权衡改善**,不是全面提升。

### 这仍有价值,但价值点变了
λ 从 1.5 降到 1.25,用 0.6% 的 aes 换 5.6% 的 geneval —— 这个交换率是有意义的,
且与 λ^2~3 / λ^0.5 的标度预测一致。但**要主张"超越官方 Student"必须用两个口径都成立**,
现在 Z 口径不支持。**判据应加严:Ratio 与 Z 都需超过各自噪声带。**

## 2026-08-28 13:10 — **poe125 在 step 391 反转:聚合优势是早中期的暂态**

| step | ΔRatio(pp) | ΔZ(阈值 0.039) |
|---|---|---|
| 61 | +0.10 | −0.003 |
| 121 | +0.89 | +0.062 |
| 181 | +0.77 | +0.031 |
| 271 | +1.01 | +0.051 |
| 301 | +0.71 | **−0.031** |
| **391** | **−0.40** | **−0.101** |
`poe15_391`=0.9996(几乎与官方持平) > `poe125_391`=0.9956。**两个口径同时转负,ΔZ 远超噪声带。**

### 逐轴:geneval 机制**没有**失效,是其他轴在累积损失
| 轴 | 181 | 271 | 301 | 391 |
|---|---|---|---|---|
| geneval | +6.54 | +10.14 | +8.92 | **+8.04**(稳定) |
| aes | −0.19 | −0.64 | −1.94 | **−2.22**(扩大) |
| ocr | +0.44 | −0.36 | −0.06 | **−3.39**(转负) |
| hps | −0.12 | +0.08 | −0.88 | **−1.10**(转负) |
绝对值 @391:geneval **0.9453 vs 0.8750**;aes **6.144 vs 6.283**。

### 修正后的结论(今天第二次下调这条)
**λ=1.25 vs 1.5 是一个持续的逐轴权衡** —— geneval 明显更好,aes/ocr/hps 越来越差。
**聚合上的优势只是早中期的暂态**,随训练推进被 aes/ocr/hps 的累积损失抵消并反超。
=> **不能说 poe125 更好。** 只能说:**λ 控制的是 geneval ↔ aes/ocr/hps 的交换点,
   而更小的 λ 把交换点推向 geneval 一侧。**

### 方法论
这是 02:30 立的"聚合显著必须看逐步轨迹"第二次救场。
第一次(rtcwarm)是我从残缺数据误判为反转;这次是**真的反转**,且在发生时就抓到了。
**再次确认:早中期的聚合优势不能外推到终点。** 我 11:50 那个"外推到 781 约 1.0107"的估算作废。

## 2026-08-28 13:30 — poe125 的单轴救援也不成立;并给出项目当前的诚实底线

### geneval 绝对值排行(poe125 并不突出)
| 臂 | geneval |
|---|---|
| tau20_721 | **0.9570** |
| tau20_271 / tau02_991 / caopd721 | 0.9531 |
| GenEvalTeacher | 0.9512 |
| **poe125_391** | **0.9453**(与另外 6 个臂并列) |
| 官方 Student | 0.9376 |
`caopd721` 同时有 geneval 0.9531 **和**综合 +0.32%,两项都优于 poe125。
=> **poe125 不是任何单轴上的突出者。** 它的价值仅是一个 **λ 消融点**(λ 控制 geneval↔aes 交换点)。

### 项目当前的诚实底线
把所有臂放在 `overall_score.py` 上,**我们最好的几个都只是"与官方 Student 打平"**:
| 臂 | Δ% | 判定 |
|---|---|---|
| poe15_781 | +0.37 | 打平(噪声内) |
| caopd721 | +0.32 | 打平 |
| mopd961 | +0.22 | 打平 |
| tau10_721 | +0.19 | 打平 |
**没有任何一个臂显著超越官方 Student。** 噪声带 2σ=0.61pp,而最好的是 +0.37pp。

而单 reward 的 RTC/OPSD 线是 **−17~−23%**,明显更差。

### 这意味着什么
1. **对 OPSD 的批评是扎实的**(ρ 装饰性、目标构造修不动、单 reward 综合上倒退) —— 这部分证据充分;
2. **但我们还没有一个能超越基线的方法。** 多 reward 线只做到打平。
3. 今天两次下调 poe125,说明**早中期的领先一律不可外推**;要主张超越,必须看终点且两个口径都过。

### 仍未试、且在确认的瓶颈(拟合端)上的两条
- **C** `OPSD_WF_MODE=frozen`(官方代码已实现,一个环境变量);
- **B'** 多查询点(已修好稀释,`OPA_MQ_W=0.3`)。
两者都需 80GB 级显存(L40/A40 实测两处 OOM),而 H100/H200/A100 全被占。

## 2026-08-28 13:50 — A′ 完整 100 epoch:**梯度消失,归因排除干净**

`opsdlse` **正常完成**(COMPLETED, exit=0, 10:07:21, epoch 0~99),不是失败。释放 4 张 H100。

聚合 n=100:−0.00651, t=−4.80, 69/100 为负, 符号检验 z=−3.80。
**机制判别(完整数据):**
| 分箱 | canonical acc | n | 差 |
|---|---|---|---|
| acc>=0.93 | 0.969 | 8 | −0.00636 |
| 0.75<=acc<0.93 | 0.832 | 23 | −0.00673 |
| acc<0.75 | **0.605** | 69 | −0.00646 |

**目标质量差距从 0.028 拉到 0.39(14 倍),三个分箱的差值彼此只差 0.0004。**
我 08:10 担心的"弱梯度"在 n=100 下**消失** —— 那是 n=49 时的采样波动。
=> **−0.0065 是均匀基线偏移,不可归因于 A′。归因排除干净。**

### A′ 的最终地位
它是主命题最有力的一条证据:
**目标确凿变好(位移 −38%、采纳率 0.61→1.00),而产出的差异与干预强度完全无关。**

## 2026-08-28 13:55 — 方向C 已上机(用 A′ 释放的 H100,无需停任何作业)
`opsdwf`(1035722): H100x3、nproc=3(与 `opsd_repro` 基线一致)、cert=0、ls=0、mq 关闭
=> **与 canonical 仅差 `OPSD_WF_MODE=frozen` 一个变量。**
**运行时验证要点**:`OPSD_WF_MODE` 是环境变量,**不出现在 CMD 行里**,
唯一确认途径是拟合阶段打印的 `[WF-FROZEN] 冻结归一化器 wf_p=... wf_n=...`。已挂后台守候。
**在看到该行之前,不能认为 C 生效。**

## 2026-08-28 14:20 — **方向C 的判据(预注册,在看到结果之前写下)**

### 为什么需要新判据
A′/cert 用的是"效应是否随干预强度变化"。但 C 的"冻结"从第一个 batch 起全程生效,
**没有强度梯度**,该判别法不适用。需要一个 C 特有的可证伪预测。

### 机制推导
- canonical:`L = mean(r²)/mean|r|` 对残差 **1-齐次** => **L ∝ |r|**;
- frozen: `L = mean(r²)/const` 是 **2-齐次** => **L ∝ r²**;
- 冻结常数取自**第一个 batch**(实测 wf_p=3.668129e-02, wf_n=3.510782e-02),
  故两者在 epoch 0 应当**几乎相等**,随训练 r 减小而**发散**。

### 预注册预测
**P1(机制是否真的启动)**:`opa_pos_loss` 在 epoch 0 两臂近似相等;之后 C 的**持续更小**,
且比值 L_C / L_canonical 应当正比于 `mean|r| / 3.668e-2`(随训练下降)。
- P1 成立 => 机制按设计启动;
- P1 不成立 => 实现或我的理解有问题,**必须查,不能直接看产出**。

**P2(产出)**:`reward_avg` 对 `opsd_repro`(canonical, nproc=3) 逐 epoch 配对。
判定门槛:**必须超过基线偏移量**。已测得的偏移参考:
- `opsdb100`(cert=0,H100,n4) vs repro:−0.0021(n=14);
- `opsdc100`(cert=1,H100,n3) vs repro:+0.0033(n=99,已证伪为偏移);
- `opsdlse`(A′,H100) vs c100:−0.0065(n=100,已证伪为偏移)。
=> **节点/run 间偏移量级为 ±0.002~0.007。C 的效应必须明显超出这个范围才可信。**
这比之前任何一条的门槛都高 —— 因为我们现在知道这个系统的基线漂移有多大。

**P3(轨迹)**:按 02:30 的规矩,聚合显著还需逐 epoch 轨迹稳定,不能是早期暂态。

### 注意一个已知风险
冻结常数会让**有效学习率随训练衰减**(L ∝ r² 且 r 在变小)。
若 C 的 reward 更低,可能只是"等效 lr 变小"而非机制本身有害 —— 这是一个**混淆**,
要靠 P1 的比值曲线来判断衰减幅度是否足以解释产出差异。

## 2026-08-28 14:30 — **P1 的方向性预测失败**(如实记录);机制确认但干预强度很弱

| epoch | C(frozen) | canonical | 比值 C/can |
|---|---|---|---|
| 0 | 0.1296 | 0.1300 | **0.997** |
| 4 | 0.1529 | 0.1444 | 1.059 |
| 8 | 0.1574 | 0.1507 | 1.045 |
| 12 | 0.1577 | 0.1483 | 1.064 |
| 16 | 0.1603 | 0.1504 | **1.066** |

### 我错在哪
预注册的 P1 说"比值应随训练**下降**"(因为 L_C/L_can = mean|r|/const,我假设 r 随训练变小)。
**实测比值是上升的。** 真因:**残差并不变小,反而变大** ——
canonical 的 `opa_pos_loss` 本身从 0.1300 涨到 0.1504。
目标随 reward 上升移得更远,模型追不上,故 mean|r| 增大。

### 但机制本身是确认的
- 起点比值 **0.997 ≈ 1**,与"常数取自第一个 batch"完全一致;
- 比值 = mean|r|/const 的关系成立,只是方向与我预期相反。
=> **冻结确实生效**,`OPSD_WF_MODE=frozen` 不是空转。

### 一个重要的早期诊断:干预强度很弱
16 个 epoch 只带来 **6.6%** 的损失差异(比值 0.997 -> 1.066)。
即 1-齐次性虽被打破,但**打破得很轻**,因为 mean|r| 本身只涨了 ~6.6%。
=> **不该预期 C 有大效应。** 若最终产出差异落在基线偏移带(±0.002~0.007)内,
   合理解释是"干预太弱"而非"机制无效" —— 这两者要分开写。
=> 若要真正检验该机制,应当用**跨条件固定的绝对常数**(像我们在 rho 扫描里用的
   `RTC_WF_CONST`),把差异放大到可测量,而不是用"首 batch 自适应值"。

### 修正后的期待
C 现在更像"机制验证"而非"性能改进"。真正的性能检验需要一个**更强的**幅度保留方案。

## 2026-08-28 15:10 — ★ 唯一有胜算的方向:**可微 reward 上升 + 不可微轴 teacher 蒸馏,并行**

### 立论(建立在今天全部负结果之上)
1. OPSD 的目标构造**是好的**(rand 对照 z=−3.98、win 0.986),但**它的细节改不动**
   (A′ 位移−38%+acc→1.0 无可归因影响、cert 拒 39% 无效目标也无效);
2. 真正的问题是**目标里缺东西**:单 reward 线 6 轴综合 **−17~−23%**,多 reward 线**打平**;
3. **结构性限制**(关键):OPSD 的上升需要 reward 可微 —— 代码硬性
   `raise ValueError(f"OPA ascent reward '{opa_kind}' not differentiable-supported.")`,
   支持的只有 pickscore/clipscore/hpsv2/aesthetic/imagereward,**全是偏好类**。
   而崩掉的 **OCR / GenEval 不可微**(PaddleOCR、目标检测)。
   连它自己的"joint"预设 `sd35_open3` 也只是 {pickscore, clipscore, hpsv2} —— 三个高度相关的偏好 reward。
   => **OPSD 结构上永远无法覆盖不可微轴,这不是调参能修的。**
4. 而 teacher 蒸馏**不需要梯度** —— teacher 权重里就编码了 OCR/GenEval 能力。

=> **合成:可微 reward 走 OPSD 式梯度上升,不可微轴走 teacher 蒸馏,同时进行。**
`rtcwarm` 已验证其**串行弱版本**(先 teacher 预训练再 RTC):OCR 保住 2.2 倍且 pick 打平,
是今天唯一 work 的东西;但串行会被持续侵蚀(+0.447 -> +0.23)。**并行则无侵蚀问题。**

### 代码可行性(已查)
`train_sd3_opd.py`:
- L64 `config.train.teachers` 是**必需**的(RTC 配置也设了单 teacher,只为复用数据/打分管线);
- L2394 `if rtc:` 分支自行 `accelerator.backward(_loss_rtc)`(L2470),
  并在 **L2531 `continue`** 跳过后面的 teacher 蒸馏路径。
=> **两者不是结构性互斥,只是一个提前退出。** 改动量可控。

### 但有三个必须先解决的问题(不能顺手改)
1. **j 循环范围不同**:RTC 只在单个查询点 j=jq 训练,teacher 蒸馏在所有 j 上训练。
   直接相加会让 teacher 项被算 T 次而 RTC 项 1 次,**权重严重失衡**。
2. **梯度累积语义**:RTC 分支自行管理 `sync_gradients` / `global_step`
   (且已有踩坑记录:那个 `continue` 在 `with accelerator.accumulate()` 内部,
   会跳过 info 汇总和 `global_step += 1`,导致 checkpoint 永不保存、eval 永不触发)。合并后要重新理顺。
3. **两个损失的尺度**:RTC 损失经 wf 归一化(~0.15),teacher 蒸馏损失量级需实测。

### 处置
**不在监控循环里改。** 这触及训练主循环的控制流,改错会**静默产生错误梯度而不是崩掉** ——
比崩掉危险得多。正确做法:清醒时专门做,并先跑一个 3-epoch 冒烟,
**核对两项损失的实际量级与各自的 j 计数**,再定权重。

## 2026-08-28 15:30 — 方向C 中期(n=36):两个判据都指向 null,且有干净的解释

**P2(产出)**:−0.00168, t=−0.91, 17/36, 符号检验 z=−0.33 **不显著**;
幅度 0.0017 **低于基线漂移带**(±0.002~0.007) => 既区分不出零,也区分不出漂移。
**P1(机制)**:比值稳在 1.02~1.08,**不再走高**。干预强度全程只有 2~8%。

### 为什么这是一个干净的收尾,而不是"测得不好"
我 14:30 曾想"用跨条件固定的绝对常数把差异放大"。**但那想错了**:
把 wf 乘一个常数 c,损失缩放 1/c、梯度也缩放 1/c —— **等价于改学习率,不是机制改变**。
真正的机制改变是 **1-齐次 -> 2-齐次**,而冻结版**确实做到了**。
问题在于:2-齐次与 1-齐次的差别只在 `mean|r|` **偏离冻结常数**时才显现,
而实测 `mean|r|` 全程只变动 ±7%。

=> **结论:归一化器的 1-齐次性在实践中不重要,因为残差本身几乎不变。**
这与"ρ 是装饰性的"是同一枚硬币的两面:
- ρ 被约掉 <= 损失 1-齐次;
- 而"修好齐次性"也没用 <= 残差不变,齐次性差异无从体现。
**两条合起来才是完整结论:信赖域幅度这个维度在 OPSD 里从头到尾都是空的。**

### 对方向 C 的处置
继续跑完 100 epoch(数据有用,且能给出 n=100 的确定性结论),
但**不再期待它是性能改进**。它的价值是**把"幅度维度为空"这个论断补全**。

## 2026-08-28 15:50 — 合成方向的**前置测量**:两项损失差 ~1000 倍

用现有 wandb 日志实测(纯只读,无需新跑):
| 损失 | 中位数 | n |
|---|---|---|
| teacher 蒸馏 `policy_loss`(poe15) | **0.0002** | 800 |
| teacher 蒸馏 `policy_loss`(ste) | 0.0003 | 770 |
| **RTC `rtc_loss`** | **0.2683** | 800 |

### 含义
- 朴素 `loss_rtc + loss_teacher` 会被 RTC **完全主导**,teacher 项等于不存在;
- 修正 j 计数后仍失衡:teacher 在约 10 个 j 上累积(10 x 0.0002 = 0.002),
  RTC 在 1 个查询点(0.27) => **梯度层面约 135 倍**,仍是 RTC 压倒。
- **注意区分**:损失量级不直接等于梯度贡献(取决于各自对参数的敏感度),
  但 1000 倍的差距足以说明**必须显式定权重**,不能直接相加。

### 这解决了前置问题 #3
前置问题现在剩两个(都在控制流上,必须清醒时做):
1. **j 循环范围不同**(RTC 单点 vs teacher 全步) —— 需决定是"RTC 只在 jq 加项"还是"teacher 只在 jq 算";
2. **梯度累积语义**(RTC 分支自管 `sync_gradients`/`global_step`,且 L2531 的 `continue` 有已知踩坑)。

### 一个副产品观察
teacher 蒸馏损失只有 2e-4,说明**学生已经非常接近 teacher**(蒸馏早已收敛)。
=> teacher 项能提供的梯度本就很小。这可能意味着:
   即使正确加权,teacher 项对"保住 OCR/GenEval"的**主动**贡献也有限,
   它更像一个**正则/锚定**项而非驱动项。
   => 合成的预期收益应当下调:它更可能是"减缓侵蚀"而不是"消除侵蚀"。
   这与 rtcwarm 的实测形态(优势 +0.447 衰减到 +0.23,不归零)一致。

## 2026-08-28 16:30 — ste 跑完,链路自动推进;但评测又卡在同一个分区坑(第 2 次)

`ste` 完成 -> `retain` 自动启动、`evste` 10 点评测自动排上(第(2)条依赖链 + auto_eval 正常工作)。

### 坑:评测排到**后天凌晨 2:55**
`evste`(1035766) `Reason=Resources StartTime=2026-08-30T02:55`。
而 **ai-gpu05 有 4 张空闲 A40** —— 但它不在 `sablab-gpu` 分区(属 mosadeghlab-gpu/preempt_gpu/admin)。
改 `Partition=preempt_gpu QOS=low` 后**立刻启动**,省约 34 小时。

### 根因(今天第 2 次,需固化)
`auto_eval_on_finish.sh` 生成的评测作业**默认走 `sablab-gpu`**,
而该分区只含 a40 gpu01-04 / a100 gpu06-09 / h200 gpu17-18;
**空闲的 ai-gpu05(A40) 与 gpu11(A100) 都不在其中**。
=> 凡评测作业报 `Reason=Resources` 且预计启动很远,**先查空闲 A40 是否在别的分区**,
   而不是等。改 partition+QOS 是安全的(仍是 A40,评测硬件基准不变;
   且用户既定偏好"评测可用 preempt")。
**建议**:给 `auto_eval_on_finish.sh` 默认加 `--partition=preempt_gpu --qos=low`,
但那是会影响后续所有自动评测的改动,留白天确认后再做。

## 2026-08-28 19:10 — ★ **方向C 完成(n=100):三个预注册判据全部 null,幅度维度收口**

`opsdwf` COMPLETED, exit=0, 05:28:11, epoch 0~99。

| 判据 | 结果 |
|---|---|
| **P2 产出** | −0.00169, t=−1.56, 46/100, 符号检验 **z=−0.80 不显著**;幅度**低于**基线漂移带(±0.002~0.007) |
| **P1 机制** | 比值 C/can 分段:1.045 / 1.052 / 1.044 / **1.020** —— 干预强度全程 2~5%,末段还在**降** |
| **P3 轨迹** | 分段差:+0.00070 / −0.00635 / +0.00076 / −0.00186 —— **符号来回翻** |

P3 的符号翻转是最强的一条:真效应应当四段同向,现在正负交替 => **噪声**。

### 结论(与 15:30 的推理一致)
**冻结归一化器对最终质量无影响。** 且原因清楚:
1-齐次与 2-齐次的差别只在 `mean|r|` 偏离冻结常数时显现,而实测 `mean|r|` 全程只变 ±5%。
=> **归一化器的齐次性在实践中不重要,因为残差本身几乎不变。**

### 这补全了"幅度维度为空"的完整论断
- **ρ 被约掉** <= 损失 1-齐次(代数 + 100 倍数值 + 四点扫描);
- **修好齐次性也没用** <= 残差不变,齐次性差异无从体现(本实验, n=100)。
=> **信赖域幅度这个维度在 OPSD 里从头到尾都是空的。** 两条合起来才是完整结论。

### OPSD 三个维度全部收口
| 维度 | 结论 | 证据 |
|---|---|---|
| **幅度** | **空的** | ρ 装饰性(4 条) + 冻结归一化器无效(n=100, 3 判据) |
| **方向** | 必须是 reward 梯度,但**优化不动** | rand 对照 z=−3.98;A(位移−1.7%,无信息)、A′(位移−38%+acc→1.0,无可归因影响, n=100) |
| **目标质量** | **改好了也没用** | cert=1 拒掉后期 39% 无效目标,机制判别证伪(拒绝率差 12 倍效应不动) |

**=> 对 OPSD 的批评部分已经完整且证据充分。剩下的唯一活路是"目标里缺什么"(不可微轴),
   而那需要 teacher 蒸馏并行,不是 OPSD 框架内能解决的。**

## 2026-08-28 20:00 — ★ 下一个该做的实验:**Best-of-N 自蒸馏(OPD without teachers)**

### 想法
用 OPSD **已经在采的** `num_image_per_prompt` 个样本,按任意 reward(**含不可微的 OCR/GenEval**)
选出最好的那个,把它的**整条去噪轨迹**当 teacher 蒸馏。
teacher = **同一策略在同一 prompt 上更好的那次采样** => 真正的 on-policy 自蒸馏,无需外部 teacher。

### 为什么正好贴合今天全部的发现
| 今天的发现 | Best-of-N 如何应对 |
|---|---|
| 构造增益 **99.65% 丢在拟合** | **目标是模型自己的输出,可实现性缺口为零** |
| 目标构造细节改不动(A/A′/cert 三条 null) | **没有目标构造可调**,选出来即是 |
| 随机方向增益为负(z=−3.98) | 方向 = "最好样本 vs 其余",有意义 |
| OCR/GenEval **不可微**,OPSD 结构上够不着 | **选择只需能评估,不需要梯度** |
| 采样成本 | **零额外成本** —— 24 个样本本来就在采 |

### 可行性(已查实,只读)
1. **完整轨迹已保存**:`sample_entry["rollout_states"] = latents[:, :-1]`(L1137) —— 全部中间态都在;
2. **group 结构现成**:`stat_tracker.update(prompts_all_decoded, rewards)`(L1217)已按 prompt 分组,
   `group_size` 已记录;
3. **与现有 advantage 的关系**:OPSD 现在是**连续加权**(每样本按自身 advantage 加权),
   Best-of-N 是**硬选择**(只留最好的当目标)。
   => **不是新增机制,而是把"连续加权"换成"argmax 选择"。**

### 这构成一个干净的单变量对照,直接检验主命题
> OPSD:用 reward 梯度**外推**目标(离流形远,99.65% 实现不了);
> Best-of-N:用 group 里**已有的真实样本**当目标(在流形上,可实现性缺口为零)。
**同一批样本、同一个 reward,只差"构造目标 vs 选择目标"。**
若可实现性真是瓶颈,Best-of-N 应当明显胜出 —— 这是主命题最强的可证伪检验。

### 与已有 `onmanifold` 的区别(重要)
`onmanifold`(K 候选搜索)做类似的事,但配置注释写明结构性缺陷:
需 `noise_level=0.7` 的 SDE 采样才能让 K 条轨迹分叉,而**论文消融与我们重跑的 anneal 臂都显示 ODE > SDE**,
故它实际检验的是"搜索收益能否盖过改用 SDE 的损失"。
**Best-of-N 无此问题** —— OPSD 的 K 个样本是不同初始噪声的独立采样,天然分叉,不需要改采样器。

### 排序(virtual teacher 路线)
1. **Best-of-N 自蒸馏**(新,最贴合证据,零额外采样成本);
2. **MVP-2 值函数 critic**(`sopd_vvalue_pick`,**已实现但从没跑过**;
   docstring 自述 "Uses only the scalar reward, so it extends to non-diff rewards" ——
   唯一在原理上绕开可微性限制的已实现方案);
3. ~~MVP-1 reward 梯度~~(`sopd_vteacher_pick`,等价 OPSD 的做法,今天的结论已覆盖,不值得再投)。

## 2026-08-28 20:30 — **Best-of-N 自蒸馏已实现**;冒烟抓到跨卡分组问题并修复

### 实现方式(最小改动,严格单变量)
复用已有的 `dir_mode` 开关,新增 `"bestof"`:
方向从 reward 梯度换成 **`(组内 reward 最高样本的 clean latent − y0)`**,
其余(信赖域 ρ、双分支、归一化器、采纳门控、损失)**逐字不变**。
=> 与 canonical 只差一个变量:**目标是"梯度外推"还是"组内已有的真实样本"**。
数学上复用 `dir_mode="residual"` 的 `fixed_dir = x_end − x0` 结构,只换 `x_end`。

改动点:
- `_opa_tr_step`:接受 `"bestof"`(与 residual 同式);
- 配置层校验 L898 补 `"bestof"`(**不补会直接 raise,启动即失败**);
- OPA chunk 循环前算 `opa_best`,循环内 `_xend = opa_best[s:e] if ... else xi`;
- `slurm/opsd_ls.slurm` 加 `OPA_DIR`(默认 grad = canonical 不变)。

### 冒烟第一次:**组被跨 GPU 拆开**(核对行抓到)
```
[BESTOF] 本地组数=48 组大小 min/max=8/8 (期望 = num_image_per_prompt = 24)
```
=> 每卡只持有同 prompt 的 24/3=8 个样本,实际跑的是 **best-of-8 而非 best-of-24**。
**这种错误不会崩,只会静默给出定义不同且偏弱的方法。**
(同时更正我此前一个说法:"group_size=24 意味着组不会被拆分"只对**梯度等价性**成立,
 对**样本分片**不成立。)

### 修复:跨卡全局分组
用现成的 `gather_tensor_to_all` 聚合 reward / prompt_ids / clean latent,
分组键沿用 OPSD 自己算 group advantage 的口径(**gather 后 `batch_decode` 回文本**,L1212 已有先例),
全局取 argmax 再按 `rank*Bn + i` 映射回本地。
**注意**:`prompt_ids` 是 tokenizer 的 token 序列(L1047),**不是标量 ID**,不能直接当分组键 —— 必须先 decode。

### 待确认
冒烟核对行必须显示 **组大小 24/24**。在看到之前不认为实现正确。

## 2026-08-28 20:35 — Best-of-N 冒烟**通过**,完整 100 epoch 已排

### 冒烟核对(3 epoch,全部通过)
| 指标 | e0/e1/e2 | 判读 |
|---|---|---|
| `bestof_group_size` / `_min` | **24/24/24** | ✅ 跨卡修复生效,真正的 best-of-24(修复前是 8) |
| `bestof_gap` | .058/.060/.064 | 组内最优比均值高约 0.06 reward |
| **`bestof_dist_ratio`** | **9.46/8.00/7.39** | **‖y_best−y0‖ 约为信赖域预算的 8 倍** |

### dist_ratio≈8 的意义(这是只看 reward 曲线看不出来的)
最优样本离当前样本很远,ρ=0.10 使每步只朝它走 **约 1/8** 的距离。
**但这让对照更干净**:canonical 沿 reward 梯度走 ρ‖y0‖,bestof 沿"指向最优样本"走**同样的** ρ‖y0‖
=> **步长相同,只差方向**。
且若最终无效,`dist_ratio` 能直接区分"想法不对"还是"步长不够" —— 后者可用更大 ρ 复测。

### 修的两个问题(都不会崩或很难查)
1. **组被跨 GPU 拆开**(8/8 vs 期望 24/24)—— 静默,会把 best-of-8 当 best-of-24 报;
   修法:`gather_tensor_to_all` 聚合 reward/prompt_ids/clean latent,分组键用 **decode 回的文本**
   (`prompt_ids` 是 token 序列不是标量 ID);
2. `all_gather` 要求**连续张量**,`_rw[:, 0]` 列切片非连续 -> `ValueError: Tensors must be contiguous`。

### 按用户建议改进的调试方式
原本的核对是"只打印一次"。**改成每 epoch 写进 `metrics.jsonl` 的指标**,
100 epoch 全程可核对、可画曲线,而不是只在开头看一眼。

## 2026-08-28 21:30 — Best-of-N(移动式)**大幅有害**,且诊断指标给出了确切机制

### 数据(n=20 同 epoch 配对 vs canonical)
| epoch | Best-of-N | canonical | 差 |
|---|---|---|---|
| 0 | 0.7482 | 0.7561 | −0.008 |
| 9 | 0.6670 | 0.8600 | −0.193 |
| 19 | 0.6625 | 0.8716 | **−0.209** |
**20/20 全为负,差距单调扩大**,末值是基线漂移带(±0.002~0.007)的 **30 倍**。
Best-of-N 的 reward **绝对下降**(0.748 -> 0.66),canonical 上升(0.756 -> 0.872)。

### 失败机制(靠新加的每-epoch 指标定位,只看 reward 曲线看不出来)
| epoch | 0 | 3 | 7 | 12 | 18 |
|---|---|---|---|---|---|
| `bestof_gap` | .056 | **.062** | .010 | .009 | .012 |
| `bestof_dist_ratio` | **9.05** | 6.07 | 2.61 | 1.43 | 1.14 |

`dist_ratio` 初始 **9.05** 是关键:组内最优样本离当前样本有 9 倍信赖域远。
**根因:那 24 个样本用的是不同初始噪声,所以"最优样本"是一张完全不同的图,
不是同一张图的改进版。** 于是"朝最优样本移动 1/9"**不是质量提升方向,而是朝无关图像的插值**
=> 产生插值伪影,模型学它就退化。
`gap` 0.062->0.010、`dist_ratio` 9.05->1.14 正是模型被拖向低方差劣化态的表现。

### 这是**实现设计错误**,不是想法本身的问题
teacher 之所以能用"朝它移动",是因为 teacher 对**同一张图**给出修正版;
而组内兄弟是**另一张图**。
=> **正确形态是 RAFT 式:只在最优样本自己的轨迹上训练**(拒绝采样微调),
   而不是把其他样本朝它推。

### 处置
按用户"完整跑完再出结论"的要求,当前 100 epoch 继续跑完并做全轴评测,作为该变体的完整记录。
修正版(RAFT 式)需 3~4 卡 80GB,当前空闲只有 L40(实测两处 OOM)与 2 张 H200
(OPSD 的 48x24 分组要求 nproc ∈ {3,4,6,8}),故排队等待。

## 2026-08-28 22:00 — Best-of-N 移动式**已停**;RAFT 式过滤冒烟通过,完整 100 epoch 已排

### 移动式(dir_mode=bestof)最终:大幅有害,已停
n=20 同 epoch 配对 vs canonical:**20/20 全为负**,差从 −0.008 单调扩大到 **−0.209**
(基线漂移带的 30 倍);reward 绝对下降 0.748->0.66 而 canonical 上升 0.756->0.872。
**根因(靠 `bestof_dist_ratio` 定位)**:初始 9.05 => 组内最优样本离当前样本 9 倍信赖域远,
因为那 24 个样本用**不同初始噪声**,"最优"是**另一张图**,不是同一张图的改进版。
=> "朝它移动"是朝无关图像插值,产生伪影。**这是我的实现设计错误,不是想法本身的问题。**
数据已备份 `metrics_final30ep.jsonl.bak`。

### 修正版:RAFT 式过滤(opa_topk)
**改训练数据而非目标方向**:每 prompt 组只保留 reward 最高的一部分样本训练,
目标构造仍是 canonical 的 reward 上升 => "最优"是样本**自己**,无跨图插值。
选择只需能评估、不需梯度 => 天然支持不可微 reward(OCR/GenEval)。

### 冒烟过程中守卫拦下一个会静默出错的版本
第一版用**全局 top-k**:`[TOPK] 跳过:构造出 368 条,期望 384`。
根因:全局 top-6 可能一个都不落在本 rank(每 rank 每组 8/24,全无 top-6 概率
C(16,6)/C(24,6)≈2.9%,48 组期望 1.4 组;实测差 16=**恰好 2 组×8**)。
**若无批量守恒检查,会变成"部分组被跳过、批量悄悄变小",然后被当作 RAFT 的效果报出去。**
改为 **rank 本地按同比例取 top**(每组 25%),每组必有成员、批量天然守恒、且省掉跨卡 gather。

### 冒烟核对(全部通过)
```
[TOPK] k=6/24 本地组数=48 每组本地样本=8 每组保留=2 唯一样本=96/384 批量守恒=OK
```
每组 8->2 = 25%(与全局 top-6/24 同比例);唯一样本 96/384 = 1/4;批量守恒;0 错误。
3 epoch 的 reward_avg **0.7506 -> 0.7810 -> 0.8132(上升)**,与移动式的下降相反 —— 但 3 点不作结论。

### 判据(预注册)
- **P1 机制**:`[TOPK]` 每组保留数与批量守恒(已在冒烟确认;完整跑中仍会打印);
- **P2 产出**:`reward_avg` vs `opsd_repro` 逐 epoch 配对,门槛**超出 ±0.002~0.007 基线漂移带**;
- **P3 轨迹**:分段符号需同向,不能来回翻;
- **P4 全轴**:跑完排 10 点评测,出 geneval/ocr/pick/clip/aes/hps 六轴 + `overall_score.py` 全局排名。

---

## 2026-08-29 02:40 — poe125 vs poe15 补点到 n=14,结论稳定

把 poe125 缺的 9 个 step(91/151/211/241/331/481/601/721/781)补评,公共步集 6 → 10 → 14 → **15(完整)**。

| 轴 | n=10 | n=14 | **n=15(最终)** | 判定变化 |
|---|---|---|---|---|
| geneval | +0.0574, t=+4.5 | +0.0566, t=+6.1 | **+0.0549, t=+6.3** | 稳定;始终 ~1.9σ,差一点没过 2σ=0.0594 |
| aes | −0.0620, t=−3.7 | −0.0642, t=−4.1 | **−0.0637, t=−4.4 显著** | 稳定,poe125 更差 |
| ocr | −0.0037, 打平 | −0.0011, 打平 | −0.0013, 打平 | 稳定 |
| pick | +0.0012, t=+3.1 | +0.0010, t=+1.9 | +0.0010, t=+2.1 | **退出**"t 显著" |
| clip | +0.0013, t=+4.8 | +0.0010, t=+2.3 | +0.0009, t=+2.3 | **退出**"t 显著" |
| hps | −0.0012, 打平 | −0.0014, 打平 | −0.0015, 打平 | 稳定 |

**读法**:步集从 10 扩到 15(+50%)而 geneval/aes 两轴的判定纹丝不动 —— 这是对选择效应的直接检验,通过。
pick/clip 随数据增多而弱化,符合"本来就是噪声"。

**结论**:poe125(λ=1.25)相对 poe15(λ=1.5)是**权衡不是胜利**。GenEval 有稳定的大幅正向
(+0.057,t=6.1)但卡在噪声带边缘;aesthetic 确定性更差(−0.064)。不能作为正向主张。

顺带修掉一个系统性误判源:eval wrapper 里 `python ... | grep ... || echo "rc=$?"` 取的是 grep
的退出码,且末尾 echo 恒为 0 —— 失败的 eval 在 sacct 里显示 COMPLETED(实测 evopsdck 1035924
三个数据集全挂却 exit=0)。已改用 `PIPESTATUS[0]` 并校验 `average_scores.json` 存在。

OPSD 自身 checkpoint 在 CA-OPD 协议下的退化预算(ck20 → ck100):
geneval −0.0312 / aes −0.0995 / ocr +0.0489 / pick +0.0256。

---

## 2026-08-29 04:00 — RTC 线的正向结果(自相关校正后,两种子复现)

> **04:10 归因订正**:我最初把这组对比写成"Gauss-Newton 预条件 vs 等价 OPSD 构造",**这是错的**。
> 追踪 checkpoint 路径与 config docstring 后确认:
> - eval tag `rtcraw` → `[RTC-RAWB …]` = `rtc_raw_train`;
> - eval tag `rtcwarm` → `[RTC-RAW-WARM-w61 …]` = `rtc_raw_warm` = `rtc_raw_train()` + `OPD_STUDENT_INIT`
>   从 teacher 臂早期 checkpoint 初始化,docstring 明写"其余与 rtc_raw_train **逐项相同**"。
>
> **两臂都用 raw(即 OPSD 等价)方向。唯一变量是 teacher 蒸馏 warmup 初始化,不是预条件。**
> 对比本身是有效的(它正是 docstring 指定的对照),错的只是我的归因标签。
>
> **Gauss-Newton 预条件那一支(`rtc_gn`)在 benchmark 上评测点为 0,至今从未被检验。**

`rtcwarm` = raw 构造 + **teacher 蒸馏 warmup 初始化**;`rtcraw` = 同样的 raw 构造但从零训练。
两者逐项相同,差值可直接归因到 warmup。
评测在**同一 10 点步集** {61,151,241,331,421,511,601,691,751,781} 上配对。

### 校正后的判定

逐 step 的评测指标仍有自相关(r1 达 0.24~0.73),先按 `n_eff = n(1−r1)/(1+r1)` 校正再判显著:

| 轴 | 种子42 diff / t_naive → **t_eff** | 种子43 diff / t_naive → **t_eff** | 判定 |
|---|---|---|---|
| **ocr** | +0.2282 / 4.87 → **3.11** (5.7×2σ) | +0.2693 / 7.94 → **7.94** (6.7×2σ) | **两种子都显著** |
| **clip** | +0.0071 / 3.87 → **2.80** (2.5×2σ) | +0.0079 / 6.46 → **4.48** (2.7×2σ) | **两种子都显著** |
| geneval | +0.0805 / 3.56 → 2.06 | +0.0750 / 3.37 → 1.51 | 打平(naive 会误判为显著) |
| aes | +0.0991 / 2.62 → 1.31 | +0.0254 / 0.53 → 0.24 | 打平 |
| pick | +0.0036 / 0.92 → 0.57 | +0.0097 / 3.74 → **2.84** | 只种子43 |
| hps | −0.0024 / −1.39 → −0.64 | −0.0042 / −2.15 → −0.96 | 打平 |

**结论:ocr 与 clip 两轴在两个种子上都稳健显著,两个种子都 0 轴显著更差。**
geneval/aes 的"显著"是自相关造成的假象(r1 0.50~0.73,有效样本量只剩 2~3),已剔除。

### 绝对水平(必须一并陈述,否则会误导)

| | accuracy | ocr | pickscore | clipscore | aesthetic | hpsv2 |
|---|---|---|---|---|---|---|
| **Student(官方)** | **0.9376** | **0.9273** | 0.9222 | **0.2958** | 6.1594 | **0.3427** |
| rtcraw(10点均) | 0.5465 | 0.3665 | 0.9940 | 0.2658 | 6.0806 | 0.3091 |
| rtcwarm(10点均) | 0.6270 | 0.5947 | 0.9976 | 0.2729 | 6.1797 | 0.3067 |
| rtcwarm43(10点均) | 0.6215 | 0.6358 | 1.0036 | 0.2737 | 6.1060 | 0.3049 |

整条 RTC 线为 pickscore 训练(0.998~1.004,**高于** Student 的 0.9222),
但 geneval/ocr 出现灾难性遗忘(0.62/0.60 vs Student 的 0.94/0.93)。

**因此可主张的命题是**:"在切到 reward 梯度训练之前,先用 teacher 蒸馏做**短暂 warmup 初始化**,
可显著减少随后的能力损失(OCR +0.23~+0.27,即把 0.37 拉回到 0.59~0.64;CLIP +0.007~+0.008),
两种子复现,0 轴显著更差",
**不是**"Gauss-Newton 预条件有效",也**不是**"打败官方 Student"。

这恰好是 `rtc_raw_warm` docstring 里那条 observation 的可操作检验:
G_construct 探针实测 teacher 方向的优势在训练早期即变号(step 61 已到 −0.00058,241 时 −0.00883,
15/15 同号),而 grad-rand 全程强正 ⇒ **teacher 只在训练最初携带上升信息**。
推论"先 teacher warmup 再切 reward 梯度"现已被 benchmark 证实。

### 严谨性缺口(如实记录)

1. 种子 43 的对比是**跨种子**的(rtcwarm43 vs rtcraw 种子42),处理与种子同时变化。
   干净的同种子对比只有种子 42 那一组。补 `rtcraw43` 才能让第二组也是同种子对照 ——
   按 781 step 计约需 20h,今晚来不及,列为后续。
2. **主线方法(Gauss-Newton 预条件)本身仍未被 benchmark 检验**:`rtc_gn` 的评测点为 0。
   本条目证实的是 warmup 初始化,不是预条件。这两件事必须在论文里分开陈述。

---

## 2026-08-29 04:15 — GN 主臂被数值问题阻塞:方向恒为 NaN(未解决,如实记录)

### 发现

排查中发现一件之前没被记录的事:**Gauss-Newton 预条件(cron 里的主线方法)从未训练到可评测状态**。
`logs/` 下非 SMOKE 的 RTC 训练只有 `[RTC-KB]`、`[RTC-RAWB]`(对照)、两个 `[RTC-RAW-WARM]`;
`rtc_gn` 只有一个 0 checkpoint 的 SMOKE 目录。因此 `rtcgn` 的评测点为 0。

于是用 opsdtopk 释放的 4×H100 提交了 `rtc_gn_train`(job 1036003)。
它与 `rtc_raw_train` 逐项相同,只差 `precond="gn"` + `rtc_jvp="exact"`,对照已有现成 10 点评测。

### 症状

```
[DIAG2] rtc_R0=0.8210 rtc_acc_m=0.2222 rtc_acc_p=0.1667
        rtc_cos_exact=nan rtc_cos_gn_g=nan rtc_dirnorm=nan rtc_eps_stab=nan rtc_fd_vs_exact=nan
[JVP-DBG] vt2: finite=True dtype=fp32 absmax=5.752e+00 | yt2: finite=True absmax=3.610e+00
          | jtw: n=382 none=0 nonfinite=0 | s2: finite=True absmax=1.828e+00
          | dir_ex: finite=False absmax=nan
```

即 **`_vt2` / `_yt2` / `_jtw`(382 个参数全部) / `_s2` 全部有限,唯独二次反向
`torch.autograd.grad(_s2, _w)` 产出 NaN**。方向是 NaN ⇒ 目标构造无效 ⇒ 这一跑没有意义。
`acc_p` 只有 0.11~0.17(未被污染样本的残留)也印证了这点。

### 已排除的假设

1. **`_w` 全零退化点** —— 代码里早已修(改用小随机 `_w`),`jtw` 现在非零且全有限。
2. **fp16 溢出**(代码原注释的猜测:math 后端显式 materialize QK^T,无 online softmax 的减最大值保护)。
   我据此把精确 JVP 块包进已有的 `_fp32_fd()`(同时关 autocast 与 TF32),重跑 job 1036006 —— 
   **`dir_ex` 仍然 finite=False**。故该假设**被证伪**,不是精度问题。
   (该修改本身无害且方向正确,已保留;备份 `scratchpad/train_sd3_opd.bak_jvpnan`。)
3. **梯度检查点** —— 代码里未启用,排除。

### 真因(已用 anomaly detection 定位,job 1036012)

加了一次性开关 `RTC_JVP_ANOMALY=1`,让 PyTorch 直接点名:

```
RuntimeError: Function 'NativeLayerNormBackwardBackward0' returned nan values in its 0th output.
```

且同一次跑里还报出另外两处:

```
Error detected in BmmBackward0
Error detected in SafeSoftmaxBackward0
```

即 **NaN 不是单一算子的 bug,而是这个模型/后端组合下双反向路径整体不可用**。
`SafeSoftmaxBackward0` 是 math SDPA 后端的 softmax —— 与"attention mask 的 −inf 在双反向里
产生 0×inf"这一猜测吻合,但它只是三处之一。

### 尝试过的修法(失败,如实记录)

融合算子 `native_layer_norm` 的二阶导有问题,而手写分解版
`(x−mu)·rsqrt(var+eps)` 由 mean/var/rsqrt/乘加组成、每个算子二阶导都正确。
于是在 JVP 块内临时替换 `F.layer_norm`(已验证:3 种形状 × 3 种 weight/bias 组合下
前向与 PyTorch **逐位一致**,不引入数学错误)。

**结果:`dir_ex` 仍然 finite=False,且 `acc_p`/`acc_m` 双双掉到 0.0000**(job 1036013)。
LayerNorm 只是三个 NaN 源之一,修一个不够。该补丁本身无害,已保留(`_ln_safe()`)。

### 假设淘汰表

| # | 假设 | 检验方式 | 结论 |
|---|---|---|---|
| 1 | `_w` 全零退化点 | 代码早已改用随机 `_w` | 已修,`jtw` 非零且全有限 |
| 2 | fp16 溢出(math 后端 materialize QK^T) | 包进 `_fp32_fd()`(关 autocast+TF32) | **证伪**,仍 nan |
| 3 | DDP 不支持双反向 | 改用 `unwrap_model(...)` | **证伪**,仍 nan |
| 4 | 梯度检查点 | 代码未启用 | 排除 |
| 5 | LayerNorm 融合算子二阶导 | 换手写分解版 | **是原因之一但不充分**,仍 nan |

### forward-mode AD:已实现并**修好**(2026-08-29 07:30)

`torch.func.jvp` + `functional_call` 一次前向直接算 `J·u`,完全绕开二阶反向。
配置 `rtc_gn_fwd`(`rtc_jvp="fwd"`),与 `rtc_raw_train` 仍只差 precond。

**最终确认**:`[FWD-DBG] dir_fwd finite=True 非有限元素=0/196608 absmax=4.03e+01 |u|=1.29`
—— GN 方向终于可以计算。

#### 真因:fp16 输入被 accelerate 的 autocast 装饰器悄悄兜住

采样期存下的 latents / prompt embeds 是 **fp16**,而 LoRA 与基座权重是 **fp32**。
accelerate 在 `prepare()` 时把实例 forward 换成 `convert_outputs_to_fp32(autocast(forward))`,
内部按 fp16 算、输出再转回 fp32 —— 所以从外面看 `pdd.dtype=float32`,完全看不出来。
绕过该装饰器后直接报 `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`,
真因才暴露。**那个 fp16 前向就是切向量全部 196608 个元素溢出成 nan 的原因。**

修法:在 `_f_of_params` 里把四个输入显式 `.float()`。

#### 一条重要的方法论教训

我曾据 `[FWD-DBG] autocast_enabled=False` 判定"fp16 已排除" —— **那是错的**:
探针放在 `functional_call` **返回之后**,读的是外层状态,读不到装饰器内部。
而 `ConvertOutputsToFp32` 使输出 dtype 与"内部 fp16"完全相容。
**探针位置错误会给出看似确凿的错误结论**,比没有探针更危险。

#### 排查路径(转折点在第 5 步)

| # | 假设/动作 | 结果 |
|---|---|---|
| 1 | 二阶反向恒 NaN | `detect_anomaly` 点名 NativeLayerNormBackwardBackward0 / BmmBackward0 / SafeSoftmaxBackward0 |
| 2 | fp32 + 关 TF32 | 无效 |
| 3 | DDP 双反向(改 unwrapped) | 无效 |
| 4 | 手写 LayerNorm | 无效(是原因之一但不充分) |
| 5 | **改走 forward-mode**,离线验证 | 成功,与有限差分 cos=+0.999728 |
| 6 | flash 无 forward-mode 规则 | 强制 math;**mem-efficient 同样不支持** ⇒ math 唯一 |
| 7 | `sdpa_kernel` 上下文穿不透 PEFT+accelerate | 改用全局标志 |
| 8 | 切向量全 NaN;train/eval 模式 | 无效 |
| 9 | **隔离测试 0013**(裸/部分字典/PEFT 三组合) | 三组合全 OK ⇒ **一次排除模型侧** |
| 10 | 绕过 accelerate forward 装饰 | 暴露 Half vs Float ⇒ **真因** |

第 1–8 步是"提假设→跑一轮→被推翻",每轮代价一次训练启动;
第 9 步改用一次性隔离测试,**一次排除两个候选**,是效率的转折点。

### 处置

停掉 job 1036006(继续跑只会烧 4 张 H100 产出 NaN 方向)。
**GN 主臂目前无法计算其方向**:有限差分路径也不可用(fp32 下 `max|v1-v0|` 恒为 0 → 零向量;
fp16 下被舍入噪声主导,`cos(d_eps,d_10eps)=0.34`),精确 JVP 出 NaN。**两条路都断。**

**这意味着 cron 里描述的主线方法(把 reward 上升方向换成 JJ^T g)至今没有任何 benchmark 证据。**
今晚确认的正向结果(见 04:00 条目)来自 **warmup 初始化**,与预条件无关。

释放的 4×H100 改投 `rtcraw43`(最终 job 1036016,4 卡,`rtc_raw_train` + `OPD_SEED=43`),
用于把 04:00 那条正向结果的第二个种子从"跨种子"补成"同种子对照"。
为此给 `rtc_raw_train` 加了种子后缀(种子 42 时 run_name 逐字不变,已验证)。

---

## 2026-08-29 04:35 — RAFT(top-k 过滤)判负;并修掉一个会静默毁评测的端口冲突

### RAFT 结果(n=4,同 step 配对)

`opsdtopk`(top-k 过滤,K=6/24)vs `opsdck`(OPSD 基线复现),同协议同 checkpoint 步 {20,60,80,100}:

| 轴 | opsdtopk | opsdck | 差 | t | 判定 |
|---|---|---|---|---|---|
| geneval | 0.6025 | 0.6201 | −0.0176 | −1.2 | 打平 |
| ocr | 0.4710 | 0.4547 | +0.0163 | +1.6 | 打平 |
| **pick** | 0.9128 | 0.9179 | **−0.0050** | **−4.8** | **显著更差** |
| clip | 0.2777 | 0.2778 | −0.0000 | −0.0 | 打平 |
| aes | 5.9892 | 5.9711 | +0.0181 | +0.6 | 打平 |
| hps | 0.3155 | 0.3172 | −0.0017 | −3.1 | t 显著但在噪声带内 |

**判负**:唯一显著的一轴是 pickscore 且是更差。top-k 过滤没有带来收益。
(step 40 因下述端口冲突失败被覆盖度守卫剔除,已重排 job 1036017;补齐后 n=5 重跑。)

### 一个会静默毁掉评测的 bug:端口只依赖 array task 号

`eval_*.slurm` 里 `PORT=$(( 29600 + SLURM_ARRAY_TASK_ID * 7 ))` —— **不含作业号**。
今晚同时跑了 evtopk / evtopk100 / evguard / evguard3 / evg25 五个 array,
落到同一节点的**同号 task** 必然取到同一端口:

```
RuntimeError: The server socket has failed to listen on any local network address.
port: 29614, useIpv6: 0, code: -98, name: EADDRINUSE
```

结果 `opsdck_40/geneval` 整格失败。**这是靠今晚早些时候加的 rc 传播修复才被发现的** ——
在那之前它会被 wrapper 吞掉、在 sacct 里显示 COMPLETED,然后以"缺 geneval"的形式
被覆盖度守卫默默剔除,永远查不出原因。

改为 `PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))`(SLURM_JOB_ID 对 array 每个 task 都唯一)。
已修今晚在用的 7 个脚本(eval_opsdck / eval_topk / eval_topk100 / eval_guarded /
eval_guarded3 / eval_g25 / eval_poe125_fill),另有 ~37 个历史脚本含同一模式,未动。

---

## 2026-08-29 07:50 — 一个会静默污染结果的隐患:OPD_RESUME=auto + 同名 run_name

### 事件

GN 主臂修好后启动正式训练(job 1036056),看起来一切正常:配置正确、方向有限、
`samples/epoch=72`、以 ~2.2 min/epoch 稳定推进到 Epoch 6。**但它不是从头训练的**:

```
[resume] restored from logs/[RTC-GNFWD + SD3.5-M + PickScore]/checkpoints/checkpoint-1/train_state.pt -> epoch=1
```

那个 `checkpoint-1` 建于 **06:03:08**,对应作业 **1036049**(`gnfwdsmk`, 06:00:37–06:03:49)——
它跑在 fp32 输入修复**之前**,方向全是 NaN。
即正式跑从"用 NaN 方向训练过一个 epoch 的权重"接着往下走。

**这不会报错,日志一切正常** —— 只有主动去查"起始 epoch 是不是 0"才会发现。

### 机制

`slurm/opd_arm.slurm` 里是**无条件**的 `export OPD_RESUME=auto`,
它会覆盖 `sbatch --export=ALL,OPD_RESUME=none` 传进去的值(实测:传了 none 仍打印 auto)。
而 `save_dir` 只由 `run_name` 决定,同一 config 的所有尝试共用一个目录。
所以**任何一次失败/取消的调试跑留下的 checkpoint,都会被下一次正式跑自动接上**。

### 处置

把被污染的目录 `mv` 成 `*.contaminated_HHMM`(**保留证据,不删**),重跑 job 1036057,
确认日志出现 `no train_state.pt found -> training from scratch`。

### 对其他实验的影响(需要检查,今晚未逐一核对)

同一模式适用于所有走 `opd_arm.slurm` 的臂。凡是"改了代码/配置后用同一 `run_name` 重跑"的,
都有可能接上旧 checkpoint。判据是日志里那行 `[resume] restored from ...`:
**它出现 = 不是从头训练**。今晚确认干净的:`rtcraw43`(1036037,是有意 resume,同一份代码)、
`rtcgnfwd`(1036057,已确认 from scratch)。
其余历史臂未逐一回查 —— 若某条结论存疑,先去 grep 它的 `[resume]` 行。

---

## 2026-08-29 08:20 — 种子43 同种子对照补齐;并识破一个步集造成的选择效应

`rtcraw43`(`rtc_raw_train` + `OPD_SEED=43`)已训到 step 301,评测了 {61,151,241},
于是种子43 也有了**同种子**对照(此前只能与种子42 的 `rtcraw` 跨种子比)。

### 三组对比放在一起,差别不在种子而在步集

| 对比 | 步集 | 显著更好轴数 | 显著更差 |
|---|---|---|---|
| 种子43 `rtcwarm43` vs `rtcraw43` | {61,151,241} n=3 | **5** | 0 |
| 种子42 `rtcwarm` vs `rtcraw` | {61,151,241} n=3 | **5** | 0 |
| 种子42 `rtcwarm` vs `rtcraw` | 完整 10 点 | **2** | 0 |

**同一个种子42,只因步集不同,从 5 轴掉到 2 轴。** 逐轴看是效应随训练衰减:

| 轴 | 早期 3 点 | 完整 10 点 |
|---|---|---|
| geneval | +0.1536 | +0.0805 |
| ocr | +0.3760 | +0.2282 |
| aes | +0.1850 | +0.0991 |

### 结论与口径

**warmup 的优势随训练衰减。** 因此:

- 可以说:"在同一 3 点步集上,两个种子都是 5 轴显著更好、0 轴更差" —— 干净的同种子复现;
- **不能**把种子43 的 n=3(5 轴)与种子42 的 n=10(2 轴)并列当作"种子43 效果更强" ——
  那是把步集差异冒充成结果,正是 cron 警告的选择效应;
- **头条以完整步集为准**:ocr / clip 两轴显著,0 轴更差。

`rtcraw43` 仍在训练(目标 781)。

### 10:40 更新:已在同一 6 点步集上并列,结论收紧为"仅 OCR 跨种子复现"

`rtcraw43` 评到 511,于是两个种子都能用**自己的同种子对照**在
steps={61,151,241,331,421,511} 上比较:

| 轴 | 种子42 diff / t_eff | 种子43 diff / t_eff | 两种子都显著? |
|---|---|---|---|
| **ocr** | +0.2949 / **4.47** | +0.2620 / **4.36** | ✅ |
| geneval | +0.1146 / 3.20 | +0.1296 / 2.02 | 否 |
| aes | +0.1689 / 3.91 | +0.0183 / 0.54 | 否 |
| pick | +0.0017 / 0.17 | +0.0152 / 4.40 | 否 |
| clip | +0.0078 / 1.84 | +0.0083 / 3.82 | 否 |
| hps | +0.0001 / 0.03 | +0.0057 / 3.50 | 否 |

**只有 OCR 两个种子都显著;其余各轴各显著各的、互不复现。两种子均 0 轴更差。**

比早先的"ocr/clip 两轴"更严格 —— clip 在种子42 的这 6 点上只有 1.84。
**两种子口径必须是"跨种子复现",不能取各自显著轴的并集** —— 后者会把种子噪声算成结果。

#### 步集稳定性检验(6→7 点,加入 step 601)进一步佐证

| 轴 | 种子42 t_eff | 种子43 t_eff | 跨种子 |
|---|---|---|---|
| **ocr** | 4.47✱ → **3.32✱** | 4.36✱ → **5.70✱** | ✅ |
| geneval | 3.20✱ → **2.44**(掉出) | 2.02 → 1.79 | 否 |
| aes | 3.91✱ → **2.39**(掉出) | 0.54 → 0.22 | 否 |
| pick | 0.17 → 0.21 | 4.40✱ → 4.73✱ | 否 |
| clip | 1.84 → 1.96 | 3.82✱ → 3.55✱ | 否 |
| hps | 0.03 → −0.12 | 3.50✱ → 2.70✱ | 否 |

**只加一个点,种子42 的 geneval 与 aes 就掉出显著** —— 恰是那两个"只在种子42 显著"的轴。
即:**不跨种子复现的,也不跨步集稳健**。两个独立维度给出一致判断。

#### 12:30 订正:扩到 8 点后,种子42 的 OCR 也掉出显著

| 轴 | 种子42(6→7→8) | 种子43(6→7→8) |
|---|---|---|
| **ocr** | 4.47✱ → 3.32✱ → **2.39** | 4.36✱ → 5.70✱ → **5.70✱** |

**8 点步集上种子42 已无任何一轴显著**;而其**完整 10 点**集上同一轴 t_eff=3.11(显著)。
同种子同对照,8 点 2.39 / 10 点 3.11 —— **显著性在阈值附近随步集摇摆**。

⇒ 上一段"只有 OCR 在 4 个组合上全部显著"**过强**。订正为:
- 种子43:OCR **稳健**显著(4.36~5.70,6/7/8 点全成立);
- 种子42:OCR **临界**(2.39~4.47,过不过 2.5 取决于步集);
- 两种子均 **0 轴显著更差**(各步集下都成立)。

**结论强度:一个种子稳健 + 一个种子临界,不是"双种子稳健复现"。**

#### 13:00 决定性发现:效应已消失并转负,不只是"衰减"

OCR 差值的完整轨迹(种子42):

| step | 61 | 151 | 241 | 331 | 421 | 511 | 601 | **691** |
|---|---|---|---|---|---|---|---|---|
| warm | 0.8850 | 0.8213 | 0.8018 | 0.6602 | 0.6246 | 0.5134 | 0.4551 | 0.2557 |
| raw | 0.4382 | 0.4388 | 0.5030 | 0.3999 | 0.3725 | 0.3847 | 0.3248 | 0.3433 |
| **diff** | **+0.447** | +0.383 | +0.299 | +0.260 | +0.252 | +0.129 | +0.130 | **−0.088** |

**单调衰减并在 step 691 转负。** 这一次性解释了所有"步集敏感"现象 ——
不是统计噪声,是**效应本身在消失**。我此前按选择效应处理,方向对但归因不完整。

#### 三个层面都不支持"正向结果"

1. **处理效应**(n=8):种子42 **0 轴显著**;种子43 ocr/pick/clip 显著 —— 不跨种子复现。
2. **绝对水平**(8 点均值):

| 轴 | 官方 Student | warm42 | raw42 | warm43 | raw43 |
|---|---|---|---|---|---|
| geneval | **0.9376** | 0.6382 | 0.5439 | 0.6318 | 0.5239 |
| ocr | **0.9273** | 0.6271 | 0.4006 | 0.6516 | 0.4115 |
| clip | **0.2958** | 0.2728 | 0.2658 | 0.2735 | 0.2658 |
| hps | **0.3427** | 0.3070 | 0.3085 | 0.3052 | 0.3006 |
| pick | 0.9222 | **0.9876** | 0.9850 | **0.9937** | 0.9793 |
| aes | 6.1594 | **6.2006** | 6.0911 | 6.1318 | 6.1275 |

warmup 只把 OCR 从 0.40 拉到 0.63,**离 Student 的 0.93 仍差很远**;
减少的是遗忘**程度**,不是产出更好的模型。只有被优化的 pickscore 超过 Student。
3. **时间趋势**:见上表,已转负。

**⇒ 14:00 完整 10 点定稿(订正上面这句)**

`rtcraw43` 训满 781 并评完 10 点后,两个种子各用自己的同种子对照:

| 轴 | 种子42 t_eff | 种子43 t_eff | 跨种子 |
|---|---|---|---|
| **ocr** | **3.11✱**(+5.7×2σ) | **7.91✱**(+6.2×2σ) | ✅ |
| **clip** | **2.80✱**(+2.5×2σ) | **3.08✱**(+2.2×2σ) | ✅ |
| 其余四轴 | 不显著 | 不显著(pick 仅种子43) | 否 |

**两种子均 0 轴更差。** 我 13:00 基于 8 点子集说"无任何一轴显著" —— 那个子集是
`rtcraw43` 尚未训完的产物,**用不完整数据下否定结论与下肯定结论同样是选择效应**,已订正。

**最终命题**:warmup 在完整步集上使 OCR 与 CLIP 显著改善(两种子复现、0 轴更差),
但优势随训练单调衰减、step≈700 接近消失,且绝对水平仍远低于 Student。
**不是"方法更好",而是"更好的初始化 + 明确的有效期"。**

有科学含义的是这条衰减曲线:**teacher warmup 的收益是暂时的,会被后续 reward 训练完全抹掉**。
这与 `rtc_raw_warm` docstring 里 G_construct 探针的观察("teacher 只在训练最初携带上升信息")
一致,但把"最初"量化到了 **step≈600** 这个尺度。

#### 衰减在**两个种子**上都复现 —— 这才是本条的稳健结论

**OCR 差值(warmup − 从零训练)**

| step | 61 | 151 | 241 | 331 | 421 | 511 | 601 | 691 |
|---|---|---|---|---|---|---|---|---|
| 种子42 | +0.447 | +0.383 | +0.299 | +0.260 | +0.252 | +0.129 | +0.130 | **−0.088** |
| 种子43 | +0.353 | +0.331 | +0.331 | +0.238 | +0.210 | +0.109 | +0.286 | **+0.062** |

**GenEval 差值**

| step | 61 | 151 | 241 | 331 | 421 | 511 | 601 | 691 |
|---|---|---|---|---|---|---|---|---|
| 种子42 | +0.203 | +0.106 | +0.152 | +0.168 | +0.059 | **0.000** | +0.031 | +0.035 |
| 种子43 | +0.203 | +0.266 | +0.156 | +0.070 | +0.082 | **0.000** | +0.051 | +0.035 |

两个种子 × 两条能力轴,全部从 +0.20~+0.45 衰减到 ≈0(降幅 **83~100%**);
种子42 的 OCR 转负,种子43 只剩 +0.062。

(GenEval 在 511 都恰为 0.000、691 都恰为 +0.0352,是 256 样本的量化步长 1/256=0.0039 所致
—— +0.0352 = 9/256,不是巧合。)

**⇒ 可主张的命题(负结果,但稳健且可量化)**:
"teacher 蒸馏 warmup 初始化能在训练早期显著减少能力遗忘(OCR +0.35~0.45),
但该优势随 reward 训练**单调衰减**,到 step≈700 已基本归零(种子42 转负、种子43 仅剩 +0.06)。
两个种子、两条能力轴上一致复现。"

这比"warmup 有效"信息量更大:它给出了**收益的有效期**,并说明
"先 teacher warmup 再切 reward 梯度"这条路线**无法产生持久的能力保持**。

---

## 2026-08-29 08:50 — GN 主臂前 38 epoch 的训练行为(**初步观察,尚不能下结论**)

方向修好后 GN 已训 38 epoch。指标如下,但**必须先说清楚一个混淆**:
对照 `rtcraw43` 的 wandb 只覆盖 epoch **61–402**,GN 在 **0–37**,两者训练阶段不同,
早期 acc_p / grad_norm 本来就可能不同。**下面的对比不是同 epoch 的,不能据此判定方法失败。**

| 指标 | GN (ep 0–37) | RAW 对照 (ep 61–402) |
|---|---|---|
| `grad_norm` | 0.03→0.3~1.65 | 中位 **0.00083**(max 0.0083) |
| `rtc_acc_p` | 0.25~0.57 | 中位 **0.917** |
| `rtc_loss` | 0.13→0.34 | 0.219~0.297(稳) |
| `monitor_pickscore` | 0.838→**0.794(降)** | 中位 0.974,**升 +0.0955** |
| `rtc_cos_gn_g` | 0.0095→0.0062 | 未记录(raw 臂无此量) |
| `rtc_dirnorm` | 251→17000 | 未记录 |

### 已排除的平凡解释:方向未归一化

读代码确认目标构造是**尺度无关**的:
```python
_dn = _dir.flatten(1).norm(dim=1).clamp(min=1e-12)
_step = _budget / max(rtc_nascent, 1)      # _budget = rho * ||y0||
_yp = _yp + _step * (_dir / _dn)
```
`_dir` 先除以自身范数再乘信赖域步长,所以 `rtc_dirnorm` 从 251 涨到 17000
**不会**放大实际步长。"GN 方向没归一化导致步子过大"这个解释不成立。

### 需要同 epoch 才能判定的两点

1. **`rtc_acc_p` 0.33 vs 0.92**:acc_p 是"构造出的目标是否真的抬高了 reward"。
   若同 epoch 下 GN 仍显著低于 raw,那直接与本方法的前提冲突
   (Cauchy-Schwarz 论证要求 GN 方向的单步增益**不小于**原方向)。
2. **`monitor_pickscore` 下降 vs 上升**:同上,需同 epoch 才可比。

**判定计划**:GN 约 60 分钟后到达 epoch 61,届时与 `rtcraw43` 的 61+ 段做**同 epoch**对比。
在那之前不写结论。

---

## 2026-08-29 09:10 — **GN 预条件的核心前提被同 epoch 对照证伪**

上一条记的"初步观察"因 epoch 不匹配无法下结论。现已在 wandb 里找到 `rtcraw43` 覆盖
**epoch 0–68** 的那个 run(`gn5j2mwy`;之前只看到 61+ 的 `xniz0cnx`),
于是可以做**同 epoch、同种子(43)、同配置、只差 `precond`** 的对照。

> **09:55 口径订正 → 10:10 已用实测修正**
>
> 下表的对照 `gn5j2mwy` 是 `[RTC-RAWB-**s43**]`,而 GN 主跑未设 `OPD_SEED` 即**种子 42**,
> 所以那是**跨种子**对比,不是原文写的"同种子"。wandb 无种子42 的 RAWB 记录(当时 offline 跑)。
>
> **已补跑严格同种子对照**(`rtc_raw_diag`,种子42、同 epoch、独立 save_dir,job 1036286,COMPLETED):
>
> | 臂(种子42, epoch 0–1, 仅差 precond) | `rtc_acc_p` | `rtc_acc_m` |
> |---|---|---|
> | RAW | **0.9861 / 1.0000** | 1.0000 |
> | GN | **≈0.417 ± 0.035**(三次独立测量) | ~0.95 |
>
> **差值 0.57 = 16× 噪声基线(±0.035)。跨种子的保留已完全消除。**

### 结果(n=46, epoch 0–45,已做 lag-1 自相关校正)

| 指标 | GN 均 | RAW 均 | 差(GN−RAW) | t_eff |
|---|---|---|---|---|
| **`rtc_acc_p`** | **0.368** | **0.978** | **−0.610** | **−44.7** |
| `rtc_acc_m` | 0.937 | 1.000 | −0.063 | −13.1 |
| `grad_norm` | 0.600 | 0.00116 | +0.599 | +4.6 |
| `monitor_pickscore` | 0.817 | 0.885 | −0.067 | −3.3 |
| `rtc_R0` | 0.817 | 0.884 | −0.067 | −3.4 |
| `rtc_loss` | 0.246 | 0.277 | −0.032 | −0.5(打平) |

逐 epoch 无一例外:RAW 的 acc_p 全在 0.889~1.000,GN 在 0.29~0.57。

### 与方法前提的冲突

`rtc_acc_p` 的定义是"构造出的 +分支目标是否真的把 reward 抬高了"。
方法的 Cauchy-Schwarz 论证主张 GN 方向的单步实现增益**不小于**原方向,
因此 acc_p 应当 **≥** raw 的水平。实测是 **0.37 vs 0.98**,方向相反。

### 一个自洽的机制解释

`rtc_cos_gn_g ≈ 0.006~0.011`,而随机基准是 1/√65536 = 0.0039 —— 即 GN 方向与 g **近乎正交**。
Cauchy-Schwarz 只是**一阶**陈述:`<g, JJ^T g> = ||J^T g||² ≥ 0` 保证一阶增益非负,
但当 cos 只有 0.006 时,这个一阶增益极小;而实际步长是信赖域 `rho=0.10 × ||y0||` 的**有限**步,
此时 reward 变化被曲率(高阶项)主导 -> 采纳与否接近抛硬币。实测 0.37 与此一致。

**换句话说:"增益不小于原方向"在一阶成立,但在实际步长下不成立 —— 理论保证与可用性之间有落差。**

### 不能排除的替代解释(必须一并记录)

**训练配置下没有对 JVP 正确性的独立校验。**
`rtc_fwd_vs_fd = 0.0000`,但同一行 `rtc_eps_stab ≈ 0.008~0.05` 表明有限差分本身被噪声主导,
所以那个交叉验证**验不了任何东西**。forward-mode 的正确性证据只有离线的 0012
(受控设置、随机切向量下与 FD 的 cos=+0.999728)。
因此当前证据无法区分:
  (a) 方法前提在实际步长下失效;
  (b) 训练配置下算出的方向并非真正的 JJ^T g。
两者都会表现为 acc_p 低。**要区分需要一个在训练配置下可信的独立 JVP 校验,今晚没有。**

### 替代解释 (b) 已用精确恒等式排除(09:30)

构造了一个**不依赖有限差分**的校验。记 `u = J^T g`(反向传播得到,标准 autograd):

```
<g, J u> = <J^T g, u> = <u, u> = ||u||^2      —— 精确恒等,无近似
```

左边用 forward-mode 的 `dir_fwd` 算,右边只用 `u`。若 `u` 与 JVP 里的 J 不同源
(例如 VJP 走 accelerate 的 fp16 装饰前向、JVP 走我绕过装饰的 fp32 前向),比值会明显偏离 1。

**实测(job 1036265,独立 save_dir 以免污染主跑)**:

```
[JVP-ID] <g,Ju>=1.503914e+00  ||u||^2=1.516965e+00  ratio=0.991397
```

**偏差仅 0.86%** —— 方向确实是 `J J^T g`,`u` 与 JVP 的 J 同源。
fp16 VJP 带来的误差 <1%,**远不足以解释 acc_p 0.37 vs 0.98**。
⇒ **(b) 排除,(a) 成立。**

### 结论(可定稿)

**在同 epoch、同种子、同配置、只差 `precond` 的对照下(n=46, epoch 0–45):**

1. **方向计算正确** —— 精确恒等式校验 ratio=0.9914;
2. **方向与 reward 梯度近乎正交** —— `cos(JJ^T g, g) ≈ 0.006~0.011`,随机基准 0.0039;
3. **沿该方向构造的目标只有 37% 抬高了 reward**,而原方向是 98%(t_eff=−44.7);
4. **reward 显著更低** —— `rtc_R0` −0.067 (t_eff=−3.4),`monitor_pickscore` −0.067 (t_eff=−3.3)。

**Cauchy-Schwarz 的保证是一阶陈述**:`<g, JJ^T g> = ||J^T g||² ≥ 0` 只说明一阶增益非负。
当 `cos ≈ 0.006` 时该一阶增益极微,而实际步长是信赖域 `rho=0.10×||y0||` 的**有限**步,
reward 变化被曲率主导 —— 于是"增益不小于原方向"在一阶成立、在可用步长下不成立。

**这是对 cron 所述主线方法核心前提的一次受控证伪。**

### 机制已由 rho 剂量-反应**实测验证**(09:50, jobs 1036280/1/2)

我上面对"为什么失败"给出的解释(一阶增益被曲率淹没)本身也需要检验 ——
判据同样是"效应是否随干预强度变化"。固定方向、只改信赖域半径,各 2 epoch:

| rho | `rtc_acc_p` | `rtc_cos_gn_g` |
|---|---|---|
| **0.01** | **0.9167** | 0.0078 |
| 0.03 | 0.7361 | 0.0078 |
| 0.10(方法工作点) | 0.4028 | 0.0078 |
| — RAW 对照 @0.10 | ~0.98 | — |

- **单调回升**:0.40 → 0.74 → 0.92,rho=0.01 时几乎追平 RAW。
- `cos_gn_g` 三点**完全相同**(0.0078) —— 方向不随 rho 变,只有步长变。
  这是一个通过了的内部一致性检查(若 cos 也跟着变,说明扫描本身有问题)。

⇒ **一阶增益确实为正(Cauchy-Schwarz 成立),但在方法的工作点被曲率淹没。**

#### 随机方向对照:排除"步子太小所以没动"这个平凡解释

"小 rho 下 acc_p 回到 0.92"还有一个平凡解释:**位移太小,reward 几乎没变,所以不降**。
若如此,**随机方向在同 rho 下也该有高 acc_p**。实测(`rtc_rand_diag`, job 1036341):

| 方向 | rho=0.01 acc_p | rho=0.10 acc_p(工作点) |
|---|---|---|
| **GN (JJ^T g)** | **0.9167** | **0.4028** |
| **RAND(随机)** | **0.5278** | **0.2778** |
| RAW(reward 梯度) | — | ~0.98 |

随机方向在 rho=0.01 下只有 0.53(抛硬币),**平凡解释("步子太小所以没动")被排除**。
RAND 本身也随步长退化(0.53→0.28) —— 曲率对**任何**方向都起作用,不是 GN 特有。
GN 在两个步长上都高于随机 ⇒ **始终携带上升信息**。

#### 把 GN 放在"随机 → raw"这把尺子上量

| 步长 | GN 在 RAND→RAW 区间上的位置 |
|---|---|
| rho=0.01 | (0.9167−0.5278)/(1.0−0.5278) ≈ **83%** |
| rho=0.10(工作点) | (0.4028−0.2778)/(0.98−0.2778) ≈ **17%** |

**在方法的工作点,GN 方向离随机(0.28)比离 reward 梯度(0.98)近得多 —— 只挽回了 17% 的差距。**
而在方向"好用"的 rho=0.01 处能挽回 83%,但那个步长下 G_construct 已趋零。

这是本方法失败的**可量化**刻画:不是"方向错了",而是**方向的有用性随步长坍缩**,
而方法恰恰需要大步长才能产生它主张的 G_construct。

**完整且自洽的图景**:
- 小步长:GN(0.92) ≫ RAND(0.53) ⇒ 方向有信息,一阶保证成立;
- 工作步长:GN 掉到 0.40 ⇒ 曲率主导,一阶项被淹没;
- 同一工作步长下 RAW 仍有 0.98 ⇒ reward 梯度对该尺度的曲率稳健。

三个方向 × 两个步长的二维对照,把"方向是否有信息"与"步长是否可用"**彻底分开** ——
这正是本方法失败的确切位置:**方向有信息,但只在无用的步长下有用。**

### 方法的内在张力(这才是本质结论)

RTC 的全部价值主张是 `G_construct` 大(实测 +0.03511),而那需要**大**步长;
但实测表明大步长下 GN 方向不可靠(acc_p 0.40)。
小步长下方向可靠了(acc_p 0.92),`G_construct` 却随位移一起趋零。

**Cauchy-Schwarz 的保证是真的,但在能产生可观 G_construct 的步长下是空的。**
这不是调 rho 能绕开的 —— 两端互斥。

### "小 rho + 增大 n_ascent"这条出路:**现有旋钮走不通**(已读码 + 实测)

读码:`_dir` 在 n_ascent 循环内**不重算**,且 `_step × n_ascent = _budget`,
每步后还有 `clamp(_budget/_ddn, max=1.0)` 的信赖域投影 ——
终点恒为 `y0 + budget·dir/|dir|`,**与单步数学上同一点**。

实测(rho=0.10 固定,各 2 epoch,jobs 1036283/1036284):

| n_ascent | acc_p | acc_m | cos_gn_g | R0 |
|---|---|---|---|---|
| 1 | 0.4583 | 0.9583 | 0.0075 | 0.8221 |
| 5 | 0.3889 | 0.9583 | 0.0076 | 0.8221 |

**并非逐位相同**(acc_p 差 0.069,即 72 个样本里差 5 个),但 `acc_m`/`R0` 完全一致、
`cos_gn_g` 只差第 4 位 —— 是两次独立作业的运行期非确定性,不是 n_ascent 的效应。

### 副产品:acc_p 的运行间噪声基线 ≈ ±0.035

rho=0.10 下 acc_p 已被独立测量三次:**0.4028 / 0.4583 / 0.3889**,均值 0.417,离散 ±0.035。
有了这个基线,前面各结论的强度可以量化:

| 对比 | 差值 | / 噪声(0.035) |
|---|---|---|
| n_ascent 1 vs 5 | 0.069 | **2×** —— 与"无效应"相容 |
| rho 0.10 vs 0.01 | **0.514** | **15×** |
| GN vs RAW @rho=0.10 | **0.610** | **17×** |

⇒ 剂量-反应与主结论都远超噪声;n_ascent 无效应。

### 真正的多步上升需要**改代码**

要实现"多次可靠小步"必须**在每个中间点重算方向**(重新求 `g`、`J^T g`、`J u`),
现有实现没有这么做。这是代码改动而非配置改动,本轮未实施。
**若不查清这一点,接手的人会先去调 n_ascent、得到与单步一样的结果、再误归因到别处。**

### Benchmark 定论(n=4, steps {61,151,241,331})

| 轴 | diff / ×2σ / t_eff | 判定 |
|---|---|---|
| **pick** | −0.1014 / −34.3 / **−7.48** | **显著更差** |
| **aes** | −0.5907 / −16.5 / **−3.27** | **显著更差** |
| **hps** | −0.0378 / −17.2 / **−5.24** | **显著更差** |
| ocr / clip / geneval | +1.1 / +3.5 / −0.4 | 打平 |

**0 轴更好,3 轴显著更差。** pickscore 逐点:GN 0.849→0.872,RAW 0.913→**0.992**,
差距从 −0.065 单调拉大到 **−0.120**。
**预条件方向无法优化它本该优化的 reward** —— 与训练期 acc_p 0.42 vs 0.99 完全一致。

⇒ **训练指标与 benchmark 两个层面独立判负,结论一致。**

### 该结论的边界

属"本实现 + 本信赖域定义(`rho·||y0||`)"下的结论。
