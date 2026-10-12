# Flow-OPSD — Multi-State target Propagation (MSP) on DiffusionOPSD

This repository is the official [DiffusionOPSD](https://github.com/) code base with one
method added: **Multi-State target Propagation (MSP)**.

Everything except three files is the upstream code, byte-for-byte. The upstream README is
kept as [`README_DiffusionOPSD_upstream.md`](README_DiffusionOPSD_upstream.md) — use it for
installation, reward-model setup, checkpoint downloads and the Pick-a-Pic manifest.

> **Full research archive:** [`research/`](research/README.md) holds the complete project record as of
> 2026-10-11: every idea tried and its verdict, the experiment log, evaluation results, mechanism
> analyses, the research fork's code (all experimental switches, as a 3-file overlay on this root)
> and the MSP paper draft. Start at [`research/README.md`](research/README.md) (mostly in Chinese).
> The code at the repository root is unchanged from the original MSP-only release (tag `v0.1-msp-release`).

---

## What is MSP

Canonical OPSD turns a reward gradient into a detached regression target `y±` at a **single**
query state of the sampling trajectory (`σ* ≈ 0.278` on the 10-step schedule) and applies the
distillation loss only there.

`y±` lives in **x0 space**, so it is a type-correct regression target for the implied-x0 of
*any* state on the same rollout. MSP reuses the same target at additional, higher-noise
states:

```
L_MSP = 1/(1+|K|) · [ L_OPSD(s_q) + Σ_{k∈K} L_OPSD(s_k ; same ȳ±) ]
```

with `K = {3, 6}` (σ = 0.858, 0.602) by default — exactly the two interior states of the
**4-step deployment grid** {1.000, 0.858, 0.602, 0.009}, which the single query state is not
on.

* **no extra reward evaluation, no extra VAE decode** — the two dominant costs are paid once
* per-state backward is mathematically identical to one joint backward, so **peak memory is
  unchanged**; measured cost is **1.21× / 1.22×** canonical on H200 / H100

---

## Results (SD3.5-Medium, joint three-reward `open3` objective)

Paired per-image comparison, `n = 1000` per cell, identical hardware / world size / seed /
checkpoint. Composite = PickScore/26 + CLIPScore + HPSv2.1, as in the paper.

| prompt set | NFE | OPSD | **+ MSP** | gain |
|---|---|---|---|---|
| DrawBench | **4** | 1.5732 | **1.6085** | **+10.70 MDE** (t = +23.6, win 83%) |
| DrawBench | 40 | 1.6353 | **1.6538** | **+5.62 MDE** (t = +14.2, win 71%) |
| Unseen-1k | **4** | 1.6040 | **1.6311** | **+8.20 MDE** (t = +22.6, win 80%) |
| Unseen-1k | 40 | 1.6611 | **1.6722** | **+3.34 MDE** (t = +11.0, win 64%) |

`MDE = 0.0033` is the minimum detectable effect calibrated against 28 canonical-vs-canonical
run pairs (median |Δ| = 0.39 MDE, max 1.23 MDE). The effect reproduces across two random
seeds and across independent training runs at the same seed.

### Reproduction of the published baseline

Our reproduction of DiffusionOPSD at **its own published 300-update budget** matches the
reported row on all five evaluators (DrawBench, 40-step flow, guidance 1.0, 512²):

| row | Pick | CLIP | HPSv2.1 | Aes | ImgR |
|---|---|---|---|---|---|
| DiffusionOPSD (paper, 300 updates) | 25.51 | 0.333 | 0.389 | 6.03 | 1.51 |
| **OPSD reproduced here (300)** | **25.46** | **0.332** | **0.388** | **5.95** | **1.51** |
| OPSD reproduced here (100) | 24.66 | 0.316 | 0.370 | 5.90 | 1.43 |
| **+ MSP (100)** | **24.84** | **0.323** | **0.376** | **5.94** | **1.51** |

**Scope.** All results above are on the joint three-reward objective. Trained on a *single*
reward under the published reward-specific recipe (β = 1.0), MSP does **not** reproduce this
gain — it is neutral on HPSv2 and harmful at four steps on PickScore. We are still
investigating which difference between the two settings is responsible.

---

## What we changed (92 lines, 3 files)

Upstream commit: `9c4c48cdfe5c3fa712d32957345d75fae9295696`.

| file | lines added | what |
|---|---|---|
| `scripts/train_opsd_ri_sd3.py` | +76 | MSP: store the extra rollout states, build the extra-state list, one independent backward per state |
| `config/public.py` | +8 | `OPSD_SEED` hook (needed for paired A/B runs; `config.seed` cannot be set via `--config.seed` because it defaults to `None`) |
| `config/mixed.py` | +8 | same hook |

Every other file — `src/`, `opd/`, the other 20 scripts, `data/`, `pyproject.toml`,
`LICENSE` — is byte-identical to upstream. You can verify with:

```bash
git clone <upstream>  upstream && diff -rq upstream . | grep -v '\.git'
```

With `MSP_STATES` unset the code path is exactly canonical OPSD.

---

## Running it

Install and download rewards/weights per the upstream README, then:

```bash
# Canonical OPSD (joint three-reward objective, 100 updates)
OPSD_SEED=20260830 \
torchrun --standalone --nproc_per_node=4 scripts/train_opsd_ri_sd3.py \
  --config config/public.py:sd35_open3 \
  --config.num_epochs=100 --config.save_dir=outputs/opsd_open3

# + MSP  (the only difference is MSP_STATES)
MSP_STATES=3,6 OPSD_SEED=20260830 \
torchrun --standalone --nproc_per_node=4 scripts/train_opsd_ri_sd3.py \
  --config config/public.py:sd35_open3 \
  --config.num_epochs=100 --config.save_dir=outputs/opsd_open3_msp
```

Evaluation uses the upstream script unchanged:

```bash
python scripts/cross_eval.py \
  --ckpt outputs/opsd_open3_msp/checkpoints/checkpoint-100/lora \
  --config sd35_open3 --config_file config/public.py \
  --prompts data/drawbench/test.txt \
  --num_steps 4 --sampler flow --guidance_scale 1.0
```

Set `--num_steps 40` for the full-budget column. The few-step axis (`--num_steps 4`) is where
MSP helps most.

### Notes

* `MSP_STATES` indexes the 10-step rollout schedule. `3,6` → σ = 0.858, 0.602.
* If `MSP_STATES` is set but the rollout side did not store the states, training **aborts**
  rather than silently falling back to single-state supervision.
* `nproc_per_node` must satisfy the upstream 48-group × 24-image constraint; we use 4.

---

## License and attribution

Apache 2.0, inherited from DiffusionOPSD — see [`LICENSE`](LICENSE). This is a fork; all
credit for the base method and code goes to the DiffusionOPSD authors. MSP and the
evaluation protocol around it are ours.
