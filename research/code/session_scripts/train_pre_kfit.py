# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import diffusionopsd.rewards
from diffusionopsd import profiling  # paper efficiency-profiling harness (env PROFILE=1)
from diffusionopsd.stat_tracking import PerPromptStatTracker, calculate_prompt_group_dispersion
from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusionopsd.diffusers_patch.solver import dpm_step, DPMState  # OPA: exact-dpm2 suffix continuation
from diffusionopsd.clip_scorer import get_image_transform  # OPA cross-reward: differentiable CLIP preprocessing
from diffusionopsd.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from diffusionopsd.ema import EMAModuleWrapper
from diffusionopsd.experiment_io import (
    resolve_resume_checkpoint, resume_position, restore_ema_and_rng,
    save_trainer_state, write_raw_reward_jsonl,
)
from diffusionopsd.metrics import install_wandb_jsonl_tee
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS

# ── RPI Gate A 探针(RPI_GATEA_PROBE=1 才启用;未启用时与 upstream 9c4c48c 行为完全一致)──
# 目的:测 Open3 的**标量求和**上升方向,是否在三个 reward 上同时改善(向量可行率 VFR)。
# 判据与 kill 条件见 codex_research/0040_GATEA_VECTOR_FEASIBILITY_PREREG.md,先于数据写定。
_GATEA_ON = os.environ.get("RPI_GATEA_PROBE", "0") == "1"
# ── A1:向量可行性 veto(RPI_A1=1 才启用)────────────────────────────────────
# 官方 Open3 把三个 reward **求和**后再上升,一轴大涨可抵消另一轴大跌。
# A1 改为:只有当 y_plus 在**每一轴**上都不劣于 y0 时,该 state 才产生损失。
# 落点是官方自己的接受权重 opa_ap —— canonical 路径把它硬编码成 1.0(L~1345),
# 我们换成逐 reward 判定,不新增任何损失项。判据见 0040。
_A1_ON = os.environ.get("RPI_A1", "0") == "1"
_A1_TOL = float(os.environ.get("RPI_A1_TOL", "0.0"))   # 0 = 严格(与 VFR 定义一致)
# A1-v2:绝对判据("每轴不劣于 y0")实测只否决 **1.39%** 的 state ⇒ 等于空转,
# 那次"打平"其实没测到 idea 本身(EXPERIMENT_LOG §4.107)。
# §4.21 实测:换成**相对改善**判据(每轴的相对改善 ≥ 各轴中位数的一半),
# 可行率从 0.9935 掉到 0.7717 ⇒ **约 22.8% 的 state 会被否决**,才是有效的干预。
_A1_MODE = os.environ.get("RPI_A1_MODE", "abs")          # abs(原版)| rel(A1-v2)
_A1_RELFRAC = float(os.environ.get("RPI_A1_RELFRAC", "0.5"))  # 相对改善需达到中位数的几成
_QGRAD_ON = os.environ.get("RPI_QGRAD_PROBE", "0") == "1"   # Gate Q1:suffix action-value 梯度探针
_Q1B_ON = os.environ.get("RPI_Q1B_PROBE", "0") == "1"       # Gate Q1b:额外记录 d_Q 的多个回溯步长
_Q2A_ON = os.environ.get("RPI_Q2A_PROBE", "0") == "1"       # Gate Q2a:逐样本 LoRA 梯度相容性
_Q2A_SINK = None
_Q2A_NSAMP = int(os.environ.get("RPI_Q2A_NSAMP", "6"))      # 每 chunk 取几个样本做逐样本 backward
# ── Q-OPSD 训练臂(**不是探针**):把 reward 上升方向 ∇_y R(D(y0)) 换成
# d_Q = ∇_y R(D(F_old(s,y0)))——穿过**冻结 old-policy 后缀**反传的方向,
# 并按 Gate Q1b(0061,四条判据全过)的**在线回溯**选最大可行步长 α。
# 与 canonical 只差"用哪个方向构造 y±",其余(rho 信赖域、n_ascent、损失、优化器)完全相同。
_QOPSD_ON = os.environ.get("RPI_QOPSD", "0") == "1"
# Q-OPSD-Core(0069 §10):**保留官方两步构造与 advantage 语义**,只把构造用的 oracle
#   ∇_y R(D(y))  ->  ∇_y R(D(F_old(s,y)))
# 其余全同:step_len=budget/2、同 trust ball、**无回溯**、同 dual loss、同 r1 加权、同 seed。
# 与 _QOPSD_ON 的区别:那一版是**一步严格镜像 + 标量回溯**,y±=y0±a·d 严格反号,
# 与策略侧本就镜像的 y_pos/y_neg 相消 => pos_loss≡neg_loss => r1 对梯度贡献恒为零(§4.145 ①)。
# 两步构造的第二步在正、负分支的**不同位置**重算梯度,reward 曲率天然打破镜像。
_QCORE_ON = os.environ.get("RPI_QCORE", "0") == "1"
if _QCORE_ON and _QOPSD_ON:
    raise SystemExit("[guard] RPI_QCORE 与 RPI_QOPSD 互斥,不能同时开")
_QOPSD_ALPHAS = tuple(float(x) for x in
                      os.environ.get("RPI_QOPSD_ALPHAS", "1.0,0.5,0.25,0.125").split(","))
_QOPSD_DIAG = {}          # 诊断:选中的 α 分布、no-op 率 —— 必须能验证干预真的生效
# 消融:Q-OPSD 同时改了两件事 —— (i) 用 d_Q 方向、(ii) 在线回溯选步长。
# dir=canon 保留 (ii) 但把方向换回 canonical 的 ∇_y R(y0),用来判断收益来自哪一半。
# 若 canon+回溯 就拿到大部分收益,则"穿过冻结后缀反传"这个核心主张不成立。
_QOPSD_DIR = os.environ.get("RPI_QOPSD_DIR", "q")        # q(Q-OPSD)| canon(仅回溯消融)

_QGRAD_SINK = None
_CERTP_ON = os.environ.get("RPI_CERT_PROBE", "0") == "1"
_CERTP_SINK = None        # cert 探针(只测不门控)的 jsonl 句柄
_GATEP_ON = os.environ.get("RPI_GATEP_PROBE", "0") == "1"
_GATEP_SINK = None        # Gate P(LSR 方向组合)探针的 jsonl 句柄
_N2_ON = os.environ.get("RPI_N2", "0") == "1"
_N2_MOM = float(os.environ.get("RPI_N2_MOM", "0.99"))
_N2_EMA = {}              # N2: |r| 归一化分母的全局 EMA(逐 rank 同步)
_N2_DIAG = {}             # N2: 诊断量,写入 metrics
# ── Gate B:梯度噪声尺度 / 临界批量(预注册 codex_research/0064)──────────────
# 半批 McCandlish 估计:把一次更新的样本切成两个不相交半批 A/B,
#   E[ḡ_A·ḡ_B] = ‖G‖²          (噪声在两半间独立,内积无偏滤噪)
#   E‖ḡ_full‖² = ‖G‖² + trΣ/B
# ⇒ B_noise = trΣ/‖G‖² = B·(‖ḡ_full‖² − ḡ_A·ḡ_B)/(ḡ_A·ḡ_B)
# 纯读 .grad,不写回、不进 loss、不改训练数学 ⇒ 不需要 _NEEDS_SUFFIX/_NEEDS_AXIS_SCORES。
_BN_ON = os.environ.get("RPI_BNOISE_PROBE", "0") == "1"
_BN_SINK = None           # Gate B 的 jsonl 句柄
_BN_HALF = None           # 累积到一半时的 .grad 快照
# 半批切分方式:"random"(默认,与既有行为一致)或 "prompt"(0064 §⑧,按 prompt 切)。
# 随机切分下两半共享全部 48 个 prompt,prompt 抽样噪声被误算成信号(0064 §⑦);
# 按 prompt 切才能测「加 prompt 数」那一半问题。
# ⚠️ 重排 perm **不改训练**:梯度累积是线性的,整批梯度与 microbatch 如何切分无关(§4.81)。
_BN_SPLIT = os.environ.get("RPI_BNOISE_SPLIT", "random")
_BN_SPLIT_OK = None       # 本 epoch 的 prompt 切分是否严格均衡(不均衡则退回随机并记录)
_BN_SPLIT_DIAG = {}       # 诊断:唯一 prompt 数、A/B 计数
# ── RTC 可达性探针(预注册 codex_research/0067 §⑦ 修订版)────────────────
# R(d) = ‖Jᵀd‖²/‖d‖²  是 JJᵀ 在方向 d 上的 Rayleigh 商,**一次 VJP 即得**。
# ratio = R(g)/E_u[R(u)]:>1 表示 reward 方向比随机方向更可达,<1 则更不可达。
# 用比值而非 cos(g,JJᵀg):(a) 只需 VJP,fp16 下稳定(有限差分 JVP 在 fp16 下误差 69~101%);
# (b) J 整体收缩时分子分母同步、比值不变 ⇒ **零模型内建**,不需另开 rand 臂。
# g 在训练循环内**就地重算**(照抄 :1467 的调用式),故**不触碰 collation 生产→消费链**。
_RTCR_ON = os.environ.get("RPI_RTCREACH_PROBE", "0") == "1"
_RTCR_K = int(os.environ.get("RPI_RTCREACH_K", "8"))   # 随机方向数(0067 §⑦d 要求 ≥8)
_RTCR_SINK = None
# ⚠️ **单一真相源**:凡"需要 rollout 后缀机制"(prev_x0 / opa_rollout_z_prev / exact suffix)
# 的探针,一律加进这一个派生标志,**不要再在各处手写 `A or B or C`**。
# 本项目已因"只改了链条一端"栽过 4 次(A1 的 _GATEA_LAST、Gate P 的 _ga_r0 取用时机、
# certp 的 opa_rollout_z_prev 生产端、Q2a 的 prev_x0)。加新开关时只改这一行。
_NEEDS_SUFFIX = _CERTP_ON or _GATEP_ON or _QGRAD_ON or _Q2A_ON or _QOPSD_ON or _QCORE_ON
# 第二条链:凡需要**逐轴 reward 分量**(_GATEA_LAST)的探针,加进这一个。同样只改这一行。
_NEEDS_AXIS_SCORES = _GATEA_ON or _A1_ON or _GATEP_ON or _QGRAD_ON or _Q2A_ON


def _n2_freeze(wf, tag):
    """N2: 把逐样本瞬时的 |r| 归一化分母换成全局 EMA 标量。

    动机: L = mean(r^2)/mean|r| 对残差 r 是 1-齐次 => dL/dr 是 0-齐次,
    即把构造目标推远 c 倍,梯度数值完全不变(解析可验;实测亦如此 ——
    reward 权重相差 3.6 倍的三条臂 opa_delta_res 只差 0.2%)。
    冻住分母后 loss 变 2-齐次、梯度变 1-齐次,拟合阶段才会按"目标有多远"
    分配力度;同时用 EMA 保住损失的平均尺度,使有效学习率与 baseline 可比。
    """
    cur = wf.mean()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(cur, op=torch.distributed.ReduceOp.AVG)
    if _N2_EMA.get(tag) is None:
        _N2_EMA[tag] = cur.clone()
    else:
        _N2_EMA[tag].mul_(_N2_MOM).add_(cur, alpha=1.0 - _N2_MOM)
    # 诊断: 若 cv≈0 且 drift≈1,N2 与 baseline 等价(预注册的"无效"判据)。
    # ⚠️ 必须存 0 维 **tensor** 而非 python float —— 下游是
    # `torch.mean(torch.stack(v))`(L1698),给它一列 float 会直接 TypeError。
    _N2_DIAG[f"n2_wf_cv_{tag}"] = (wf.std() / cur.clamp(min=1e-12)).detach()
    _N2_DIAG[f"n2_wf_drift_{tag}"] = (cur / _N2_EMA[tag].clamp(min=1e-12)).detach()
    _N2_DIAG[f"n2_wf_ema_{tag}"] = _N2_EMA[tag].detach().clone()
    return _N2_EMA[tag].detach().expand_as(wf).clone()


_GATEA_LAST = {}          # 最近一次 open3 打分的逐 reward 分量(求和前)
_GATEA_SINK = None        # jsonl 写入句柄,首次使用时打开
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Public training uses the default distributed process group.
POLICY_GROUP = None

# ===================== DiffusionOPSD reward-improvement (RI) helpers =====================
# Legacy RI compatibility path: per-sample HPS gradient plus optional certification
# refinement of the clean endpoint x_i -> x_i*. Public DiffusionOPSD presets use
# the OPA path below instead (rho=0.10, two target steps, no certification).
# The reward gradient touches ONLY the target latent; the policy learns x_i* via NFT's
# forward/rollout distillation loss (the gradient never enters the policy/sampler).
def _hps_scores_grad(scorer, images01, prompts):
    """Differentiable HPS (HPSv2Scorer.__call__ body, without its @torch.no_grad)."""
    profiling.reward_fwd_inc()  # §6: reward forward (differentiable HPS ascent/refine)
    image = scorer.preprocess_val(images01.to(scorer.dtype).to(scorer.device))
    text = scorer.processor(prompts).to(scorer.device)
    outputs = scorer.model(image, text)
    logits = outputs["image_features"] @ outputs["text_features"].T
    return torch.diagonal(logits, 0).float()


def _decode01(pipeline, x_latent):
    lat = (x_latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    img = pipeline.vae.decode(lat.to(pipeline.vae.dtype), return_dict=False)[0]
    return (img / 2 + 0.5).clamp(0, 1).float()


def _hps_of_latents_grad(pipeline, scorer, x_latent, prompts):
    return _hps_scores_grad(scorer, _decode01(pipeline, x_latent), prompts)


@torch.no_grad()
def _hps_of_latents(pipeline, scorer, x_latent, prompts):
    return scorer(_decode01(pipeline, x_latent), list(prompts)).float()


# --- OPA cross-reward: generalized DIFFERENTIABLE reward scorer (gradient w.r.t. the input image) ---
# Reward adapters have different call signatures and some route images through an HF processor
# that detaches the graph. This reimplements each supported scorer's forward with differentiable
# preprocessing (reusing clip_scorer.get_image_transform for the CLIP-family), so the OPA target ascent
# can follow the TRAINING reward's gradient (not hardcoded HPS).
_OPA_TFORM_CACHE = {}


def _reward_scores_grad(scorer, kind, images01, prompts):
    """Differentiable reward score for images01 [B,3,H,W] in [0,1]."""
    if kind in ("open3", "multi_open3", "mixed"):
        if not isinstance(scorer, dict):
            raise ValueError("Composite OPA scorer must be a dict.")
        if kind == "mixed":
            weights = scorer.get("weights")
            scorers = scorer.get("scorers")
            if not isinstance(weights, dict) or not isinstance(scorers, dict):
                raise ValueError("Mixed OPA scorer requires weights and scorers dictionaries.")
        else:
            weights = {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
            scorers = scorer
        total = None
        for sub_kind, weight in weights.items():
            sub_scores = _reward_scores_grad(scorers[sub_kind], sub_kind, images01, prompts)
            if _NEEDS_AXIS_SCORES:
                # 三个分量本来就算出来了,只是被求和丢弃。截获它们不产生任何额外前向。
                # ⚠️ 条件必须与下游**所有**取数处一致(_GATEA_ON / _A1_ON / _GATEP_ON):
                # 2026-08-31 初版这里只写 `_GATEA_ON`,导致 A1 单跑(探针关闭)时
                # _GATEA_LAST 永远为空 -> _ga_r0={} -> 循环不执行 -> _feas 保持 None
                # -> `_feas.to(...)` 抛 AttributeError(job 1037521,跑了 2:39 就崩)。
                _GATEA_LAST[sub_kind] = sub_scores.detach().float().cpu()
            weighted = float(weight) * sub_scores
            total = weighted if total is None else total + weighted
        return total.float()
    profiling.reward_fwd_inc()  # §6: reward forward (OPA ascent / certification scoring)
    dev = scorer.device
    if kind == "hpsv2":
        image = scorer.preprocess_val(images01.to(scorer.dtype).to(dev))
        text = scorer.processor(prompts).to(dev)
        out = scorer.model(image, text)
        return torch.diagonal(out["image_features"] @ out["text_features"].T, 0).float()
    if kind == "clipscore":
        texts = scorer.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(dev)
        pixels = scorer._process(images01).to(dev)
        out = scorer.model(pixel_values=pixels, **texts)
        return (out.logits_per_image.diagonal() / 100).float()
    if kind == "pickscore":
        if "pickscore" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["pickscore"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["pickscore"](images01).to(dtype=scorer.dtype, device=dev)
        text_inputs = scorer.processor(text=list(prompts), padding=True, truncation=True,
                                       max_length=77, return_tensors="pt").to(dev)
        img_e = scorer.model.get_image_features(pixel_values=pixels)
        img_e = img_e / img_e.norm(p=2, dim=-1, keepdim=True)
        txt_e = scorer.model.get_text_features(**text_inputs)
        txt_e = txt_e / txt_e.norm(p=2, dim=-1, keepdim=True)
        return ((scorer.model.logit_scale.exp() * (txt_e @ img_e.T)).diag() / 26).float()
    if kind == "aesthetic":
        if "aesthetic" not in _OPA_TFORM_CACHE:
            _OPA_TFORM_CACHE["aesthetic"] = get_image_transform(scorer.processor.image_processor)
        pixels = _OPA_TFORM_CACHE["aesthetic"](images01).to(dtype=scorer.dtype, device=dev)
        embed = scorer.clip.get_image_features(pixel_values=pixels)
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return scorer.mlp.layers(embed).squeeze(1).float()  # .layers bypasses MLP.forward's @no_grad
    if kind == "hpsv3":
        return scorer._scores(images01, prompts).float()  # differentiable Qwen2-VL-7B ranknet mu
    if kind == "deqa":
        return scorer._scores(images01, prompts).float()  # differentiable mPLUG-Owl2 rating-token MOS
    if kind == "imagereward":
        return scorer._scores(images01, prompts).float()  # differentiable ImageReward BLIP score_gard
    raise ValueError(f"OPA differentiable ascent: unsupported reward kind '{kind}'")


def _reward_of_latents_grad(pipeline, scorer, kind, x_latent, prompts):
    return _reward_scores_grad(scorer, kind, _decode01(pipeline, x_latent), prompts)


def _load_reward_scorer(kind, device, reward_weights=None):
    """Load the differentiable scorer matching the training reward (weights frozen)."""
    if kind == "mixed":
        weights = {name: float(weight) for name, weight in dict(reward_weights or {}).items()}
        if len(weights) < 2:
            raise ValueError("Mixed OPA requires at least two weighted rewards.")
        return {
            "weights": weights,
            "scorers": {name: _load_reward_scorer(name, device) for name in weights},
        }
    if kind in ("open3", "multi_open3"):
        s = {
            "pickscore": _load_reward_scorer("pickscore", device),
            "clipscore": _load_reward_scorer("clipscore", device),
            "hpsv2": _load_reward_scorer("hpsv2", device),
        }
        return s
    if kind == "hpsv2":
        from diffusionopsd.hpsv2_scorer import HPSv2Scorer

        s = HPSv2Scorer(dtype=torch.float32, device=device)
    elif kind == "clipscore":
        from diffusionopsd.clip_scorer import ClipScorer
        s = ClipScorer(device=device); s.dtype = torch.float32
    elif kind == "pickscore":
        from diffusionopsd.pickscore_scorer import PickScoreScorer
        s = PickScoreScorer(device=device, dtype=torch.float32)
    elif kind == "aesthetic":
        from diffusionopsd.aesthetic_scorer import AestheticScorer
        s = AestheticScorer(dtype=torch.float32, device=device)
    elif kind == "hpsv3":
        from diffusionopsd.hpsv3_scorer import get_hpsv3_scorer
        s = get_hpsv3_scorer(device=device)  # shared 7B singleton (avoid 3× load -> OOM)
    elif kind == "deqa":
        from diffusionopsd.deqa_scorer import get_deqa_scorer
        s = get_deqa_scorer(device=device)  # process singleton (shares the frozen 7B across reward_fn/eval/ri_scorer)
    elif kind == "imagereward":
        from diffusionopsd.imagereward_scorer import ImageRewardScorer
        s = ImageRewardScorer(device=device, dtype=torch.float32)  # differentiable BLIP score_gard
    else:
        raise ValueError(f"OPA: no differentiable scorer for reward '{kind}'.")
    s.requires_grad_(False)
    return s


def ri_refine(pipeline, scorer, x_end, prompts, rho, n_ascent, eta, margin, direction=1.0):
    """Trust-region HPS gradient step on the clean latent + re-decode certification.

    direction=+1: ascent target x_i^+ accepted iff HPS(x*) >= HPS(x)+margin.
    direction=-1: descent target x_i^- accepted iff HPS(x*) <= HPS(x)-margin.

    The descent mode is the required negative-sample dual: DiffusionNFT's negative
    branch can then match v_neg to a certified-worse target, which pushes v_theta
    away from that bad direction.
    """
    x0 = x_end.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_len = eta * budget / max(n_ascent, 1)
    x = x0.clone()
    for _ in range(n_ascent):
        x = x.detach().requires_grad_(True)
        hps = _hps_of_latents_grad(pipeline, scorer, x, prompts)
        (g,) = torch.autograd.grad(hps.sum(), x)
        profiling.reward_bwd_inc()  # §6: reward-gradient backward (RI refine ascent)
        gnorm = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gnorm)
        delta = x - x0
        dnorm = delta.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        x = (x0 + delta * torch.clamp(budget / (dnorm + 1e-12), max=1.0)).detach()
    h0 = _hps_of_latents(pipeline, scorer, x0, prompts)
    hs = _hps_of_latents(pipeline, scorer, x, prompts)
    if float(direction) >= 0:
        accept = (hs >= h0 + margin).view(-1, 1, 1, 1)
    else:
        accept = (hs <= h0 - margin).view(-1, 1, 1, 1)
    return torch.where(accept, x, x0), h0, hs, accept.view(-1)


def _dpm2_continue_from(z_k, k, forced_x0, prev_x0, sigmas, v_old_fn):
    """OPA fork(a): EXACT-dpm2 suffix continuation from rollout state z_k with a forced first action.

    forced_x0 sets x0_pred[k] (via v_forced=(z_k-forced_x0)/sigma_k); prev_x0=x0_pred[k-1] seeds the
    2nd-order dpm_state so step k is the exact multistep-dpm2 update; steps k+1.. re-evaluate v_old_fn.
    Validated by the 1-GPU probe null-sanity: forcing forced_x0=y0 reproduces the real endpoint x_i to
    ~2e-6 relative error. Returns the continued endpoint latent (no grad)."""
    st = DPMState(order=2)
    st.model_outputs[-1] = prev_x0.detach().float()
    st.lower_order_nums = 2
    z = z_k.detach().float()
    v_forced = (z - forced_x0.detach().float()) / sigmas[k]
    z, _, _ = dpm_step(2, v_forced, z, k, sigmas[:-1], sigmas, st)
    for i in range(k + 1, len(sigmas) - 1):
        v = v_old_fn(z, sigmas[i]).detach().float()
        z, _, _ = dpm_step(2, v, z.float(), i, sigmas[:-1], sigmas, st)
    return z


def _dpm2_continue_from_grad(z_k, k, forced_x0, prev_x0, sigmas, v_old_fn_grad):
    """`_dpm2_continue_from` 的**可微**版本:梯度可回传到 `forced_x0`。

    与原版逐行同构,只去掉两处 detach:
      (a) `v_forced` 里的 `forced_x0` 保持 attached —— 这是我们要求导的变量;
      (b) 后续步的 `v` 不再 detach —— 梯度须经冻结 old-policy 的 velocity 回流。
    `z_k` / `prev_x0` 仍 detach(它们是给定的 rollout 状态,不是自变量)。

    用途:构造真正的 action value  Q_old(s,y) = R(D(F_old(s,y))),
    从而取 ∇_y Q 而非 canonical 的 ∇_y R(D(y))(0058 §4)。
    kq=8 + 10 步采样 ⇒ 循环只跑 1 次,即后缀只需一次 transformer 前向的图。"""
    st = DPMState(order=2)
    st.model_outputs[-1] = prev_x0.detach().float()
    st.lower_order_nums = 2
    z = z_k.detach().float()
    v_forced = (z - forced_x0) / sigmas[k]              # (a) 不 detach
    z, _, _ = dpm_step(2, v_forced, z, k, sigmas[:-1], sigmas, st)
    for i in range(k + 1, len(sigmas) - 1):
        v = v_old_fn_grad(z, sigmas[i])                 # (b) 不 detach
        z, _, _ = dpm_step(2, v, z.float(), i, sigmas[:-1], sigmas, st)
    return z


def _opa_tr_step(pipeline, scorer, kind, y0, prompts, rho, n_ascent, eta, direction,
                 dir_mode="grad", x_end=None, first_grad=None, q_ctx=None):
    """OPA target: trust-region step (direction=+1 ascent / -1 descent) at the low-noise x0 anchor y0.

    dir_mode selects the step direction (ablation knob; 'grad' IS the method):
      'grad'     : the TRAINING reward `kind`'s gradient at y0 (hpsv2/clipscore/pickscore/aesthetic).
      'rand'     : a fixed random unit direction, same trust region, NO reward info (attribution control).
      'residual' : the denoising residual (x_end - y0) direction (ATC-style), inside the OPA harness.
      'noop'     : no displacement (y+ = y- = y0), a no-reward-gradient/no-perturbation control.
    'rand'/'residual' need no scorer gradient; everything downstream (dual branch, wf, loss) is identical,
    so this isolates 'is the win from the reward direction, or just trust-region perturbation + NFT machinery?'"""
    x0 = y0.detach().float()
    budget = (rho * x0.flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
    step_len = eta * budget / max(n_ascent, 1)
    fixed_dir = None
    if dir_mode == "noop":
        return x0
    if dir_mode == "rand":
        fixed_dir = torch.randn_like(x0)
    elif dir_mode == "residual":
        if x_end is None:
            raise ValueError("_opa_tr_step dir_mode='residual' requires x_end (rollout endpoint).")
        fixed_dir = x_end.detach().float() - x0
    elif dir_mode != "grad":
        raise ValueError(f"_opa_tr_step dir_mode '{dir_mode}' unknown (grad|rand|residual|noop).")
    x = x0.clone()
    for _i in range(n_ascent):
        if dir_mode == "grad":
            if _i == 0 and first_grad is not None:
                g = first_grad  # Shared y0 gradient is identical for positive/negative first steps.
            else:
                x = x.detach().requires_grad_(True)
                if q_ctx is None:
                    r = _reward_of_latents_grad(pipeline, scorer, kind, x, prompts)
                else:
                    # Q-OPSD-Core:同一个两步构造,只把 oracle 换成穿过冻结 old-policy
                    # 后缀的 action value R(D(F_old(s,x)))。**每一步都在当前 x 处重算**,
                    # 于是正负分支第二步的位移不再互为反号 —— 这正是修掉镜像退化的机制。
                    _zq, _kq, _prev, _sig, _vold = q_ctx
                    _ep = _dpm2_continue_from_grad(_zq, _kq, x, _prev, _sig, _vold)
                    r = _reward_of_latents_grad(pipeline, scorer, kind, _ep, prompts)
                (g,) = torch.autograd.grad(r.sum(), x)
                profiling.reward_bwd_inc()  # §6: reward-gradient backward (OPA trust-region ascent/descent)
        else:
            g = fixed_dir
        gn = g.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12
        x = x.detach() + float(direction) * step_len * (g / gn)
        d = x - x0
        dn = d.flatten(1).norm(dim=1).view(-1, 1, 1, 1)
        x = (x0 + d * torch.clamp(budget / (dn + 1e-12), max=1.0)).detach()
    return x.detach()


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    # The text encoder may be offloaded during memory-intensive reward backward passes;
    # ensure every encoder is back on `device` before embedding.
    for _te in text_encoders:
        _te.to(device)
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)




def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=getattr(config.sample, "eval_guidance_scale", config.sample.guidance_scale),  # §4.4 inference CFG
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            # Persist eval samples for offline visual-collapse inspection.
            persist_dir = os.path.join(config.save_dir, "eval_samples", f"step_{global_step}")
            os.makedirs(persist_dir, exist_ok=True)
            captions = []
            for idx in range(num_samples_to_log):
                Image.open(os.path.join(tmpdir, f"{idx}.jpg")).save(os.path.join(persist_dir, f"{idx}.jpg"))
                rw = " ".join(f"{k}:{sampled_rewards_log[idx][k]:.3f}" for k in sampled_rewards_log[idx])
                captions.append(f"{idx}\t{rw}\t{sampled_prompts_log[idx][:200]}")
            with open(os.path.join(persist_dir, "captions.txt"), "w") as cf:
                cf.write("\n".join(captions))

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world


def save_ckpt(
    save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler,
    epoch_completed=None,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        save_trainer_state(
            save_root, epoch_completed=(global_step if epoch_completed is None else epoch_completed),
            global_step=global_step, ema=ema,
        )

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    global POLICY_GROUP
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        os.makedirs(config.save_dir, exist_ok=True)
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project="diffusionopsd", name=config.run_name, config=config.to_dict(), dir=log_dir)

        install_wandb_jsonl_tee(wandb, os.path.join(config.save_dir, "metrics.jsonl"))
    logger.info(f"\n{config}")

    # --- Seed policy: DiffusionOPSD requires NO fixed seed (naturally random runs). ---
    # Draw a random epoch-grouping nonce on rank 0 and broadcast it so all ranks agree
    # on K-repeat prompt grouping WITHOUT making sampling reproducible. If a user
    # explicitly sets config.seed, we honor it (reproducible mode) for debugging only.
    if config.seed is not None:
        run_random_nonce = int(config.seed)
        set_seed(config.seed, rank)
        logger.info(f"[seed] FIXED seed={config.seed} (reproducible/debug mode)")
    else:
        nonce_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main_process(rank):
            nonce_tensor[0] = int.from_bytes(os.urandom(8), "little") % (2**31 - 1)
        if world_size > 1:
            dist.broadcast(nonce_tensor, src=0, group=POLICY_GROUP)  # POLICY_GROUP=None => default world
        run_random_nonce = int(nonce_tensor.item())
        logger.info(f"[seed] NO fixed seed; run_random_nonce={run_random_nonce} (prompt grouping only)")

    # --- Persist run provenance (seed policy, sampling regime, reward ckpts, wall-clock). ---
    if is_main_process(rank):
        run_meta = {
            "run_name": config.run_name,
            "code_variant": os.environ.get("CODE_VARIANT", "diffusionnft_baseline"),
            "seed_policy": ("no_fixed_seed_random_run" if config.seed is None else f"fixed_seed_{config.seed}"),
            "run_random_nonce": run_random_nonce,
            "sample": {
                "deterministic": bool(config.sample.deterministic),
                "solver": config.sample.solver,
                "num_steps": int(config.sample.num_steps),
                "eval_num_steps": int(config.sample.eval_num_steps),
                "guidance_scale": float(config.sample.guidance_scale),
                "noise_level": float(config.sample.noise_level),
                "num_image_per_prompt": int(config.sample.num_image_per_prompt),
            },
            "reward_fn": {k: float(v) for k, v in dict(config.reward_fn).items()},
            "opsd": (
                {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in dict(config.opsd).items()}
                if hasattr(config, "opsd") else {}
            ),
            "reward_ckpt_path": os.environ.get("REWARD_CKPT_PATH", "<repo-default reward_ckpts>"),
            "model": config.pretrained.model,
            "resolution": int(config.resolution),
            "world_size": world_size,
            "git_commit": os.environ.get("CODE_COMMIT", "unknown"),
            "wall_clock_start": datetime.datetime.now().isoformat(),
        }
        with open(os.path.join(config.save_dir, "run_config.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
        logger.info(f"[provenance] wrote {os.path.join(config.save_dir, 'run_config.json')}")

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank,
                          find_unused_parameters=False, process_group=POLICY_GROUP)  # None => default world
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    if config.prompt_fn != "general_ocr":
        raise NotImplementedError("Prompt function not supported with dataset")
    train_dataset = TextPromptDataset(config.dataset, "train")
    test_dataset = TextPromptDataset(config.dataset, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=run_random_nonce,  # random per-run nonce (grouping only, not reproducibility)
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(diffusionopsd.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(diffusionopsd.rewards, "multi_score")(device, config.reward_fn)  # Pass device

    # --- Paper efficiency profiling (env PROFILE=1) ---
    # Enable per-optimizer-step timing/counting, wrap the rollout reward scorer so each
    # call counts as one reward forward, and force debug mode so periodic/final save_ckpt
    # and eval_fn are skipped (no checkpoint spam, no 100-epoch training).
    if profiling.profile_enabled():
        profiling.enable()
        config.debug = True
        reward_fn = profiling.count_reward_fn(reward_fn)

    # --- Legacy-OPSD RI: config + a separate differentiable HPS scorer for refinement ---
    ri = config.opsd
    ri_train_state = str(ri.train_state)          # "forward" (RI-NFT) | "rollout" (OPD, on-policy)
    ri_refine_on = bool(int(ri.refine))  # int flag (0/1) for reliable ml_collections CLI override
    ri_rho, ri_n_ascent, ri_eta = float(ri.rho), int(ri.n_ascent), float(ri.eta)
    ri_margin, ri_mb = float(ri.margin), int(ri.ri_mb)
    ri_aux_mode = bool(int(ri.get("aux_mode", 0)))
    ri_dual_negative = bool(int(ri.get("dual_negative", 0)))
    ri_lambda_aux = float(ri.get("lambda_aux", 0.0))
    ri_gate_mode = str(ri.get("gate_mode", "sdar"))
    ri_gate_beta = float(ri.get("gate_beta", 5.0))
    if ri_gate_mode not in ("sdar", "accept"):
        raise ValueError(f"Unknown opsd.gate_mode={ri_gate_mode}; expected 'sdar' or 'accept'")
    # ATC-X0-NFT: on-policy X0-consistency target replaces the off-policy endpoint target.
    #   y0 = z_t - t*v_old ; y_tgt = y0 + clip(x_i - y0, rho_x0*||y0||) ; train only in the t-band.
    ri_atc_x0 = bool(int(ri.get("atc_x0", 0)))
    ri_rho_x0 = float(ri.get("rho_x0", 0.03))
    ri_t_lo = float(ri.get("t_lo", 0.0))
    ri_t_hi = float(ri.get("t_hi", 1.0))
    if ri_atc_x0 and ri_train_state != "rollout":
        raise ValueError("opsd.atc_x0=1 requires opsd.train_state='rollout' (on-policy z_t states).")
    if ri_atc_x0 and (ri_refine_on or ri_aux_mode):
        raise ValueError("opsd.atc_x0=1 is a target-replacement method; set refine=0 and aux_mode=0.")
    # DiffusionOPSD: on-policy reward-gradient target in x0 space.
    #   query the low-noise rollout state sigma~=opa_query_sigma; y0=z_q-sig*v_old; build y+/y- via
    #   reward ascent/descent at y0; train NFT branches toward y+/y-. Optional opa_cert=1 enables
    #   exact-dpm2 suffix certification; canonical default uses the no-cert fast path.
    ri_opa = bool(int(ri.get("opa", 0)))
    opa_rho = float(ri.get("opa_rho", 0.10))
    opa_margin = float(ri.get("opa_margin", 0.005))
    opa_n_ascent = int(ri.get("opa_n_ascent", 2))
    opa_eta = float(ri.get("opa_eta", 1.0))
    opa_query_sigma = float(ri.get("opa_query_sigma", 0.278))
    opa_mb = int(ri.get("opa_mb", 6))
    ri_opa_dual_neg = bool(int(ri.get("opa_dual_neg", 1)))  # 0 = positive-branch only (dual-negative ablation)
    ri_opa_cert = bool(int(ri.get("opa_cert", 0)))          # 0 = no-cert fast path (skip suffix continuation)
    opa_dir_mode = str(ri.get("opa_dir_mode", "grad"))      # grad(method) | rand | residual | noop
    if ri_opa and opa_dir_mode not in ("grad", "rand", "residual", "noop"):
        raise ValueError(f"OPA opa_dir_mode '{opa_dir_mode}' unknown (grad|rand|residual|noop).")
    # Mechanism ablations from the paper appendix, independent of the direction knob above. Defaults
    # ("rollout"/"replace") reproduce the method exactly; each flips a REAL branch in the OPA loss.
    opa_state_mode = str(ri.get("opa_state_mode", "rollout"))    # rollout(method) | forward (offline re-noise z_q)
    opa_target_mode = str(ri.get("opa_target_mode", "replace"))  # replace(method) | aux (NFT-endpoint + λ·reward-grad)
    opa_aux_lambda = float(ri.get("opa_aux_lambda", 1.0))
    if ri_opa and opa_state_mode not in ("rollout", "forward"):
        raise ValueError(f"OPA opa_state_mode '{opa_state_mode}' unknown (rollout|forward).")
    if ri_opa and opa_target_mode not in ("replace", "aux"):
        raise ValueError(f"OPA opa_target_mode '{opa_target_mode}' unknown (replace|aux).")
    if ri_opa and opa_state_mode == "forward" and (ri_opa_cert or _NEEDS_SUFFIX):
        raise ValueError("opa_state_mode='forward' is incompatible with opa_cert=1 / RPI_CERT_PROBE=1: "
                         "forward-noised states have no rollout trajectory for the exact-dpm2 suffix "
                         "certification.")
    primary_reward_kind = list(config.reward_fn.keys())[0]
    opa_kind = str(ri.get("opa_reward_kind", primary_reward_kind)) if ri_opa else "hpsv2"  # ascent follows the training reward
    if ri_opa and ri_train_state != "rollout":
        raise ValueError("opsd.opa=1 requires opsd.train_state='rollout' (on-policy z_t states).")
    if ri_opa and opa_kind not in ("hpsv2", "clipscore", "pickscore", "open3", "multi_open3",
                                    "mixed", "aesthetic", "hpsv3", "deqa", "imagereward"):
        raise ValueError(f"OPA ascent reward '{opa_kind}' not differentiable-supported.")
    ri_scorer = None
    if ri_refine_on or ri_opa:
        _grad_kind = opa_kind if ri_opa else "hpsv2"
        if ri_opa:
            ri_scorer = _load_reward_scorer(
                opa_kind,
                device,
                reward_weights=config.reward_fn if opa_kind == "mixed" else None,
            )  # ascent scorer matches the training reward
        else:
            from diffusionopsd.hpsv2_scorer import HPSv2Scorer

            ri_scorer = HPSv2Scorer(dtype=torch.float32, device=device)
            ri_scorer.requires_grad_(False)
    if ri_opa:
        # OPA reward gradients backprop through the VAE decode on top of the transformer
        # training graph -> OOM without help. Gradient-checkpoint the VAE so its decode stores
        # minimal activations (recomputed in backward); the OPA micro-batch bounds peak
        # memory. No-op for the no_grad sampling/eval decodes.
        try:
            pipeline.vae.enable_gradient_checkpointing()
        except Exception as _e:
            logger.warning(f"pipeline.vae.enable_gradient_checkpointing() unavailable: {_e}")
    # Optional transformer checkpointing reduces peak memory for heavyweight public rewards.
    if os.environ.get("SD3_TRANSFORMER_GRADCKPT", "0") == "1":
        try:
            pipeline.transformer.enable_gradient_checkpointing()
            if is_main_process(rank):
                logger.info("[mem] SD3 transformer gradient_checkpointing ENABLED (SD3_TRANSFORMER_GRADCKPT=1)")
        except Exception as _e:
            logger.warning(f"pipeline.transformer.enable_gradient_checkpointing() unavailable: {_e}")
    if is_main_process(rank):
        logger.info(f"[RI] train_state={ri_train_state} refine={ri_refine_on} rho={ri_rho} "
                    f"n_ascent={ri_n_ascent} eta={ri_eta} margin={ri_margin} mb={ri_mb} "
                    f"aux_mode={ri_aux_mode} dual_negative={ri_dual_negative} "
                    f"lambda_aux={ri_lambda_aux} gate_mode={ri_gate_mode} gate_beta={ri_gate_beta}")
        logger.info(f"[ATC] atc_x0={int(ri_atc_x0)} rho_x0={ri_rho_x0} t_band=[{ri_t_lo},{ri_t_hi}]")
        logger.info(f"[OPA] opa={int(ri_opa)} reward={opa_kind} dir_mode={opa_dir_mode} cert={int(ri_opa_cert)} "
                    f"dual_neg={int(ri_opa_dual_neg)} rho={opa_rho} margin={opa_margin} n_ascent={opa_n_ascent} "
                    f"eta={opa_eta} query_sigma={opa_query_sigma} mb={opa_mb} "
                    f"state_mode={opa_state_mode} target_mode={opa_target_mode} aux_lambda={opa_aux_lambda}")

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        config.resume_from = resolve_resume_checkpoint(config.resume_from)
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict
            lora_state = load_peft_weights(lora_path, device=str(device))
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="default")
            set_peft_model_state_dict(transformer_ddp.module, lora_state, adapter_name="old")
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(config.resume_from, "transformer_model.pt")  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(torch.load(model_ckpt_path, map_location=device))

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        first_epoch, global_step = resume_position(config.resume_from, config, world_size)
        logger.info(f"Resume position: first_epoch={first_epoch}, global_step={global_step}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
    if config.resume_from:
        restore_ema_and_rng(config.resume_from, ema)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    # --- §6 profiler: 1 optimizer step == 1 epoch (gradient_step_per_epoch=1) ---
    prof = profiling.Profiler(config, world_size, rank, device) if profiling.is_enabled() else None
    prof_epoch0 = first_epoch

    def _profile_sanity_eval():
        return profiling.run_sanity_eval(
            pipeline=pipeline, reward_fn=eval_reward_fn, compute_text_embeddings=compute_text_embeddings,
            text_encoders=text_encoders, tokenizers=tokenizers, config=config, device=device,
            rank=rank, world_size=world_size,
        )

    for epoch in range(first_epoch, config.num_epochs):
        if prof is not None:
            prof.epoch_begin(epoch - prof_epoch0)
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []
        epoch_prompts_text = []  # RI: per-sample prompt text (aligned with collated samples order)

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)
            epoch_prompts_text.extend(list(prompts))  # RI: keep text aligned with stored samples

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and config.eval_freq > 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if (
                i == 0
                and global_step > 0
                and global_step % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                    epoch_completed=epoch,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale),  # §4.4 train CFG (rollout)
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            sample_entry = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                # Clone the endpoint so this record does not keep the stacked trajectory storage
                # alive through a strided view.
                "latents_clean": latents[:, -1].clone(),
                "rewards_future": rewards_future,  # Store future
            }
            if ri_train_state == "rollout":
                if ri_opa:
                    # Canonical OPSD fits one low-noise query per trajectory.  Retain only that
                    # detached state rather than carrying the full rollout-state tensor through
                    # target construction and fitting.  Endpoint certification additionally needs
                    # the immediately preceding state to seed the second-order solver suffix.
                    sig_sched = pipeline.scheduler.sigmas.float().to(device)
                    num_states = latents.shape[1] - 1
                    kq = int(torch.argmin((sig_sched[:num_states] - opa_query_sigma).abs()).item())
                    kq = max(kq, 1)
                    sample_entry["opa_rollout_z_q"] = latents[:, kq].clone()
                    # ⚠️ 消费端(L~1374)与生产端**必须同时放宽**。
                    # 2026-08-31 只放宽了消费端 => job 1038153 跑 2:20 后
                    # KeyError: 'opa_rollout_z_prev'。这是本项目第三次栽在
                    # "只改了链条的一端"(A1 的 _GATEA_LAST、Gate P 的 _ga_r0 取用时机)。
                    if ri_opa_cert or _NEEDS_SUFFIX:
                        sample_entry["opa_rollout_z_prev"] = latents[:, kq - 1].clone()
                else:
                    # Legacy rollout-state objectives train at every denoising state.
                    sample_entry["rollout_states"] = latents[:, :-1].contiguous()
            samples_data_list.append(sample_entry)
            del latents

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process); skipped in debug/profile mode to keep the §6 timing window clean.
        if epoch % 10 == 0 and is_main_process(rank) and not config.debug:
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            if is_main_process(rank):
                write_raw_reward_jsonl(
                    config.save_dir, epoch=epoch, global_step=global_step,
                    prompts=prompts_all_decoded, rewards=gathered_rewards_dict,
                )
            # Stat tracker update expects numpy arrays for rewards
            advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                dispersion_stats = calculate_prompt_group_dispersion(
                    prompts_all_decoded, gathered_rewards_dict["avg"]
                )
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        **dispersion_stats,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (avg_rewards_all.std() + 1e-4)
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if is_main_process(rank):
            logger.info(f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}")

        # ---- Legacy-OPSD RI targets / SDAR-style auxiliary targets -----------------------
        #
        # replace mode (legacy A/B):
        #   positive-adv samples replace the main DiffusionNFT target x_i with x_i^+.
        #
        # aux mode (current work-first RI-SDAR-Aux):
        #   main DiffusionNFT target remains x_i for every sample.
        #   positive samples get a gated auxiliary target x_i^+ (HPS ascent).
        #   negative samples get the required dual target x_i^- (HPS descent), used only
        #   through DiffusionNFT's negative branch so v_theta is pushed away from x_i^-.
        #
        # All targets are per-sample; never use group-best targets.
        latents_clean_local = collated_samples["latents_clean"]
        # RI/DRaFT: per-sample index into epoch_prompts_text. It rides through the shuffle+batch
        # with every other tensor, so the training loop can recover each sample's prompt text
        # (needed by the direct HPS reward gradient).
        collated_samples["prompt_idx"] = torch.arange(latents_clean_local.shape[0], device=device)
        collated_samples["x0_aux_pos"] = latents_clean_local.clone()
        collated_samples["x0_aux_neg"] = latents_clean_local.clone()
        collated_samples["gate_pos"] = torch.zeros(latents_clean_local.shape[0], device=device, dtype=torch.float32)
        collated_samples["gate_neg"] = torch.zeros(latents_clean_local.shape[0], device=device, dtype=torch.float32)

        # ---- DiffusionOPSD: build dual reward-gradient targets at the low-noise query state ----
        # For each trajectory: y0 = z_q - sig_q*v_old (low-noise x0 anchor); y+/y- = HPS ascent/descent
        # at y0. With opa_cert=1, certify by forcing the action and running the EXACT dpm2 suffix;
        # canonical opa_cert=0 accepts the local trust-region target directly.
        if ri_opa:
            sig_sched = pipeline.scheduler.sigmas.float().to(device)          # [num_steps+1], last ~0
            Bn = latents_clean_local.shape[0]
            num_states = collated_samples["timesteps"].shape[1]
            kq = int(torch.argmin((sig_sched[:num_states] - opa_query_sigma).abs()).item())
            kq = max(kq, 1)                                                   # preserves certified-suffix indexing
            sig_q = sig_sched[kq]
            emb_all = collated_samples["prompt_embeds"]; pemb_all = collated_samples["pooled_prompt_embeds"]
            opa_yp = latents_clean_local.clone(); opa_ym = latents_clean_local.clone()
            # Gate Q2a:额外存一份"Q 方向"目标,供训练段对比两种目标的逐样本参数梯度相容性
            opa_ypq = latents_clean_local.clone() if _Q2A_ON else None
            opa_ap = torch.zeros(Bn, device=device); opa_am = torch.zeros(Bn, device=device)
            opa_res = torch.zeros(Bn, device=device)
            # State-provenance ablation. Canonical OPSD queries the real on-policy
            # stored rollout query. opa_state_mode="forward" instead uses an OFFLINE forward-noised
            # state at the SAME sig_q, drawn ONCE per epoch here so that y0, the reward targets y+/y-,
            # AND the stored opa_z_q all derive from the SAME state -> the ONLY difference from the
            # method is state provenance (solver rollout vs offline forward-noise). This is the clean
            # isolation: swapping the state on only one side (prediction) would leave the target
            # anchored to the rollout state and inject a stale-target bias of order the reward step.
            if opa_state_mode == "forward":
                opa_state_src = ((1.0 - sig_q) * latents_clean_local.float()
                                 + sig_q * torch.randn_like(latents_clean_local.float()))
            else:
                opa_state_src = collated_samples["opa_rollout_z_q"].float()
            if opa_kind in ("open3", "multi_open3", "mixed"):
                # Reward BACKWARD is memory-bound. T5-XXL isn't needed until next epoch's embedding,
                # so park it on CPU (~10GB freed); compute_text_embeddings reloads it.
                pipeline.text_encoder_3.to("cpu")
            torch.cuda.empty_cache()
            for s in range(0, Bn, opa_mb):
                e = min(s + opa_mb, Bn)
                zq = opa_state_src[s:e]; xi = latents_clean_local[s:e].float()
                emb = emb_all[s:e]; pemb = pemb_all[s:e]; prm = epoch_prompts_text[s:e]
                _g_train = getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale)
                def vold(z, sigma, emb=emb, pemb=pemb, g=_g_train):
                    tt = torch.full([z.shape[0]], float(sigma) * 1000, device=device, dtype=torch.long)
                    with torch.no_grad(), torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        if g > 1.0:
                            # §4.4 train-CFG on the OPA anchor: guided old-policy velocity v_u+g(v_c-v_u),
                            # so y0 uses the SAME guided velocity as rollout (not a conditional-only anchor).
                            ne = neg_prompt_embed.to(emb.dtype).repeat(z.shape[0], 1, 1)
                            npe = neg_pooled_prompt_embed.to(pemb.dtype).repeat(z.shape[0], 1)
                            vv = pipeline.transformer(hidden_states=z.to(emb.dtype).repeat(2, 1, 1, 1),
                                                      timestep=tt.repeat(2),
                                                      encoder_hidden_states=torch.cat([ne, emb]),
                                                      pooled_projections=torch.cat([npe, pemb]),
                                                      return_dict=False)[0]
                            v_u, v_c = vv.chunk(2)
                            v = v_u + g * (v_c - v_u)
                        else:
                            v = pipeline.transformer(hidden_states=z.to(emb.dtype), timestep=tt,
                                                     encoder_hidden_states=emb, pooled_projections=pemb,
                                                     return_dict=False)[0]
                    return v.float()

                def vold_g(z, sigma, emb=emb, pemb=pemb, g=_g_train):
                    """`vold` 的**可微**版本(Gate Q1 用):与 vold 逐行同构,只去掉 `torch.no_grad()`。

                    参数仍是冻结的 "old" adapter —— 我们只需要 ∂v/∂z,不需要 ∂v/∂θ;
                    调用方用 `torch.autograd.grad(Q, y)` 取导,不会污染任何 `.grad`。"""
                    tt = torch.full([z.shape[0]], float(sigma) * 1000, device=device, dtype=torch.long)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        if g > 1.0:
                            ne = neg_prompt_embed.to(emb.dtype).repeat(z.shape[0], 1, 1)
                            npe = neg_pooled_prompt_embed.to(pemb.dtype).repeat(z.shape[0], 1)
                            vv = pipeline.transformer(hidden_states=z.to(emb.dtype).repeat(2, 1, 1, 1),
                                                      timestep=tt.repeat(2),
                                                      encoder_hidden_states=torch.cat([ne, emb]),
                                                      pooled_projections=torch.cat([npe, pemb]),
                                                      return_dict=False)[0]
                            v_u, v_c = vv.chunk(2)
                            v = v_u + g * (v_c - v_u)
                        else:
                            v = pipeline.transformer(hidden_states=z.to(emb.dtype), timestep=tt,
                                                     encoder_hidden_states=emb, pooled_projections=pemb,
                                                     return_dict=False)[0]
                    return v.float()

                with torch.no_grad():
                    y0 = zq - sig_q * vold(zq, sig_q)
                    # ⚠️ prev_x0 原本只在 ri_opa_cert 下计算。cert 探针同样需要它,
                    # 条件必须一起放宽 —— A1 首跑就是栽在"写入条件"与"读取条件"不一致(job 1037521)。
                    if ri_opa_cert or _NEEDS_SUFFIX:
                        zprev = collated_samples["opa_rollout_z_prev"][s:e].float()
                        prev_x0 = zprev - sig_sched[kq - 1] * vold(zprev, sig_sched[kq - 1])
                # The first ascent gradient at y0 is identical for the positive/negative directions
                # under the same anchor and reward, so compute it once and share it as both
                # directions' first step -> saves one reward-grad per chunk (n_ascent=2: 4/chunk -> 3).
                # grad path only (rand/residual/noop use a fixed direction / no displacement).
                opa_g0 = None
                if opa_dir_mode == "grad" and opa_n_ascent > 0:
                    _xg = y0.detach().float().requires_grad_(True)
                    _rg = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _xg, prm)
                    (opa_g0,) = torch.autograd.grad(_rg.sum(), _xg)
                    profiling.reward_bwd_inc()  # §6: the shared y0 reward-gradient backward
                # ── Gate Q1 探针(0058 §7):比较三个方向在**exact suffix 终点**上的增益 ────
                # d_clean = ∇_y R(D(y0))            canonical 用的局部代理梯度(= opa_g0)
                # d_Q     = ∇_y R(D(F_old(s,y0)))   真正的 action-value 梯度(穿过后缀反传)
                # d_rand  = 同半径随机方向          归因对照
                # 三者同半径、同 state、全部走 exact suffix 打分 ⇒ 差值可直接归因于方向。
                if (_QGRAD_ON or _Q2A_ON) and opa_dir_mode == "grad" \
                        and opa_kind in ("open3", "multi_open3") and opa_g0 is not None:
                    _q_budget = (opa_rho * y0.detach().float().flatten(1).norm(dim=1)).view(-1, 1, 1, 1)

                    def _q_unit(v):
                        return v / (v.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)

                    # (1) d_Q:穿过冻结 old-policy 后缀反传
                    _yq = y0.detach().float().requires_grad_(True)
                    _ep_q = _dpm2_continue_from_grad(zq, kq, _yq, prev_x0, sig_sched, vold_g)
                    _rq = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _ep_q, prm)
                    (_gq,) = torch.autograd.grad(_rq.sum(), _yq)
                    profiling.reward_bwd_inc()

                    if _Q2A_ON and opa_ypq is not None:
                        # α=1 的纯 Q 方向目标。Q2a 问的是**方向**是否让参数梯度更相容,
                        # 回溯(Q1b)修的是尾部增益符号,与梯度相容性是两回事,故此处不做回溯。
                        _ypq = (y0.detach().float()
                                + _q_budget * _q_unit(_gq.detach().float())).detach()
                        opa_ypq[s:e] = _ypq.to(opa_ypq.dtype)
                    _dirs = {"clean": _q_unit(opa_g0.detach().float()),
                             "q":     _q_unit(_gq.detach().float()),
                             "rand":  _q_unit(torch.randn_like(y0.detach().float()))}
                    _cos_cq = ((_dirs["clean"] * _dirs["q"]).flatten(1).sum(dim=1)).detach().cpu()

                    # (2) 锚点(y0)自身经 exact suffix 的三轴分 —— 所有增益都相对它
                    with torch.no_grad():
                        _ep0 = _dpm2_continue_from(zq, kq, y0.detach().float(), prev_x0, sig_sched, vold)
                        _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _ep0, prm)
                        _q_r0 = {k: v.clone() for k, v in _GATEA_LAST.items()}
                        # (3) clean / rand 走同半径一步;q 额外走 Gate Q1b 的多个步长
                        # Q1b(0061 §⑤):Q1 判失败的两条都是**尾部**判据,而 Codex 0058 §4.2
                        # 算法第 6 步的**回溯步长**恰恰针对尾部,却完全没被 Q1 测到。
                        # 这里对 d_Q 记录 α∈{1,.5,.25,.125} 的终点分,离线即可判"取最大可行 α"。
                        _q_rows = {}
                        for _nm, _d in _dirs.items():
                            _yj = (y0.detach().float() + _q_budget * _d).detach()
                            _epj = _dpm2_continue_from(zq, kq, _yj, prev_x0, sig_sched, vold)
                            _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _epj, prm)
                            _q_rows[_nm] = {k: v.clone() for k, v in _GATEA_LAST.items()}
                        _q_alpha = {}
                        if _Q1B_ON:
                            for _al in (0.5, 0.25, 0.125):     # α=1 已在上面算过
                                _yj = (y0.detach().float() + _al * _q_budget * _dirs["q"]).detach()
                                _epj = _dpm2_continue_from(zq, kq, _yj, prev_x0, sig_sched, vold)
                                _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _epj, prm)
                                _q_alpha[str(_al)] = {k: v.clone() for k, v in _GATEA_LAST.items()}

                    global _QGRAD_SINK
                    if _QGRAD_SINK is None and is_main_process(rank):
                        _QGRAD_SINK = open(os.path.join(config.save_dir, "gateQ1_probe.jsonl"), "a")
                    if _QGRAD_SINK is not None:
                        _K3q = ("pickscore", "clipscore", "hpsv2")
                        _gn = {"clean": opa_g0.detach().float().flatten(1).norm(dim=1).cpu(),
                               "q": _gq.detach().float().flatten(1).norm(dim=1).cpu()}
                        for _i in range(len(_cos_cq)):
                            _QGRAD_SINK.write(json.dumps({
                                "epoch": int(epoch), "kq": int(kq), "sigma_q": float(sig_q),
                                "cos_clean_q": float(_cos_cq[_i]),
                                "gnorm": {k: float(v[_i]) for k, v in _gn.items()},
                                "r0_suffix": {k: float(_q_r0[k][_i]) for k in _K3q},
                                "dir": {_n: {k: float(_v[k][_i]) for k in _K3q}
                                        for _n, _v in _q_rows.items()},
                                **({"q_alpha": {_a: {k: float(_v[k][_i]) for k in _K3q}
                                                for _a, _v in _q_alpha.items()}} if _q_alpha else {}),
                            }) + "\n")
                        _QGRAD_SINK.flush()
                    del _yq, _ep_q, _rq, _gq

                # ── Gate A 探针:y0 处的逐 reward 分量在上面算 opa_g0 时**已经算过**,直接取走。
                # ⚠️ 必须在 y_plus/y_minus **之前**取:_opa_tr_step 内部会在上升途中反复调用 scorer,
                # 把 _GATEA_LAST 覆盖成"上升点"的分数。在其之后再读,拿到的就不是锚点 y0 的 r0 了
                # —— 那是**静默的错数据**,比崩掉更难发现。
                _ga_r0 = {k: v.clone() for k, v in _GATEA_LAST.items()} if (
                    _NEEDS_AXIS_SCORES and opa_kind in ("open3", "multi_open3")
                    and opa_g0 is not None) else None
                # Q-OPSD-Core:把两步构造的 oracle 整体换成 action value。
                # y0 处第一步的梯度对正负分支相同,沿用官方的 first_grad 共享(省一次反传);
                # 第二步由 q_ctx 在各自的新位置重算 —— 镜像在这里被打破。
                _qc_ctx = None
                _qc_g0 = None
                if _QCORE_ON and opa_dir_mode == "grad" and opa_n_ascent > 0:
                    _qc_ctx = (zq, kq, prev_x0, sig_sched, vold_g)
                    _yq0 = y0.detach().float().requires_grad_(True)
                    _ep0 = _dpm2_continue_from_grad(zq, kq, _yq0, prev_x0, sig_sched, vold_g)
                    _rq0 = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _ep0, prm)
                    (_gq0,) = torch.autograd.grad(_rq0.sum(), _yq0)
                    profiling.reward_bwd_inc()
                    _qc_g0 = _gq0.detach().float()
                    del _yq0, _ep0, _rq0, _gq0
                _fg = _qc_g0 if _QCORE_ON else opa_g0
                y_plus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                      +1.0, opa_dir_mode, xi, first_grad=_fg, q_ctx=_qc_ctx)
                y_minus = _opa_tr_step(pipeline, ri_scorer, opa_kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                       -1.0, opa_dir_mode, xi, first_grad=_fg, q_ctx=_qc_ctx)

                # ── Q-OPSD:用 d_Q 替换 y± 的构造方向,并在线回溯选最大可行 α ──────────
                if _QOPSD_ON and opa_dir_mode == "grad" and opa_g0 is not None:
                    _qb = (opa_rho * y0.detach().float().flatten(1).norm(dim=1)).view(-1, 1, 1, 1)
                    if _QOPSD_DIR == "canon":
                        # 消融臂:方向用 canonical 的 ∇_y R(y0)(即 opa_g0),只保留在线回溯。
                        # 省掉 d_Q 的那次穿后缀反传,故本臂也更便宜。
                        _dq = opa_g0.detach().float()
                    else:
                        # d_Q:穿过冻结 old-policy 后缀反传(与 Gate Q1 探针同一算法)
                        _yq = y0.detach().float().requires_grad_(True)
                        _epq = _dpm2_continue_from_grad(zq, kq, _yq, prev_x0, sig_sched, vold_g)
                        _rqv = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _epq, prm)
                        (_gqv,) = torch.autograd.grad(_rqv.sum(), _yq)
                        profiling.reward_bwd_inc()
                        _dq = _gqv.detach().float()
                    _dq = _dq / (_dq.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
                    with torch.no_grad():
                        # 锚点在 suffix 终点的分数 —— 回溯判据以它为基准(Q1b 的口径)
                        _ep0q = _dpm2_continue_from(zq, kq, y0.detach().float(), prev_x0, sig_sched, vold)
                        _r0q = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _ep0q, prm).float()
                        _yp_q = y0.detach().float().clone()
                        _ym_q = y0.detach().float().clone()
                        _done = torch.zeros(_yp_q.shape[0], dtype=torch.bool, device=_yp_q.device)
                        _pick = {}
                        for _al in _QOPSD_ALPHAS:
                            _cand = y0.detach().float() + _al * _qb * _dq
                            _epj = _dpm2_continue_from(zq, kq, _cand, prev_x0, sig_sched, vold)
                            _rj = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _epj, prm).float()
                            _ok = (~_done) & (_rj > _r0q)          # 取最大可行 alpha(按序先到先得)
                            # 掩码赋值必须**无条件**执行:`if _ok.any()` 会让各 rank 走不同分支。
                            _m4 = _ok.view(-1, 1, 1, 1)
                            _yp_q = torch.where(_m4, _cand, _yp_q)
                            _ym_q = torch.where(_m4, y0.detach().float() - _al * _qb * _dq, _ym_q)
                            _pick[str(_al)] = int(_ok.sum())
                            _done = _done | _ok
                            # 提前跳出必须**跨 rank 同步**。首版写的是 `if bool(_done.all()): break`,
                            # 而 _done 是逐 rank 的 => 不同 rank 在不同轮次跳出 => 各 rank 的
                            # backward 次数不同 => DDP all-reduce 配不上 => NCCL checkTimeout /
                            # SIGABRT(job 1038560,跑 27 分钟后崩)。
                            _ad = _done.all().float().reshape(1)
                            dist.all_reduce(_ad, op=dist.ReduceOp.MIN, group=POLICY_GROUP)
                            if bool(_ad.item() > 0.5):
                                break
                        # 全部 α 都不给正增益 ⇒ no-op(y±=y0),与 Q1b 的处理一致
                        _pick["noop"] = int((~_done).sum())
                    for _k2, _v2 in _pick.items():
                        _QOPSD_DIAG[f"qopsd_pick_{_k2}"] = _QOPSD_DIAG.get(f"qopsd_pick_{_k2}", 0) + _v2
                    y_plus, y_minus = _yp_q, _ym_q
                if _ga_r0 is not None:
                    # y_plus 处需要 1 次 decode + 3 次 scorer 前向(无 backward)。这是探针的**全部**开销。
                    # 测的是 clean target y_plus,不是 rollout 终点 —— Gate A 问的正是"构造出的目标"。
                    with torch.no_grad():
                        _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, y_plus.detach().float(), prm)
                    _ga_rp = {k: v.clone() for k, v in _GATEA_LAST.items()}
                    global _GATEA_SINK
                    if _GATEA_SINK is None and is_main_process(rank):
                        _GATEA_SINK = open(os.path.join(config.save_dir, "gateA_probe.jsonl"), "a")
                    if _GATEA_SINK is not None:
                        _keys = sorted(_ga_r0)
                        for _i in range(len(_ga_r0[_keys[0]])):
                            _GATEA_SINK.write(json.dumps({
                                "epoch": int(epoch), "kq": int(kq), "sigma_q": float(sig_q),
                                "r0": {k: float(_ga_r0[k][_i]) for k in _keys},
                                "rp": {k: float(_ga_rp[k][_i]) for k in _keys},
                            }) + "\n")
                        _GATEA_SINK.flush()
                # ── Gate P 探针(LSR-OPSD 的零训练 oracle):在**同一个** y0 上比较
                # official 聚合方向与 7 个候选方向,量各轴的 local headroom。
                # 只读,不改 y_plus / y_minus,不影响训练路径。
                if _GATEP_ON and opa_dir_mode == "grad" and opa_kind in ("open3", "multi_open3"):
                    _K3 = ("pickscore", "clipscore", "hpsv2")
                    # 候选方向:u0=official 聚合;u1..u3=单轴;u4..u6=两两之和。
                    # 单轴/两两都用官方同一个 _opa_tr_step(同 rho / n_ascent / eta / trust ball),
                    # 所以候选之间**只差方向**,半径完全一致 —— 差值可直接归因。
                    _cands = [("u0_official", opa_kind, ri_scorer)]
                    for _k in _K3:
                        _cands.append((f"u_{_k}", _k, ri_scorer[_k]))
                    for _a, _b in (("pickscore", "clipscore"), ("pickscore", "hpsv2"),
                                   ("clipscore", "hpsv2")):
                        _cands.append((f"u_{_a[:4]}+{_b[:4]}", "mixed",
                                       {"weights": {_a: 1.0, _b: 1.0}, "scorers": ri_scorer}))
                    _gp_r0 = _ga_r0          # 见上:锚点分数只能用 y_plus 之前捕获的那一份
                    if _gp_r0 is None or len(_gp_r0) != 3:
                        raise RuntimeError(
                            "Gate P: 锚点 r0 缺失(_ga_r0=%r)。检查 _GATEA_LAST 的写入条件是否已包含 "
                            "_GATEP_ON,以及 opa_dir_mode 是否为 grad。" % (_ga_r0,))
                    if True:
                        with torch.no_grad():   # rollout 层的锚点 = 原始 rollout 终点 xi
                            _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind,
                                                        xi.detach().float(), prm)
                        _gp_xi = {k: v.clone() for k, v in _GATEA_LAST.items()}
                        _gp_rows = {}
                        for _name, _kind, _sc in _cands:
                            _yj = y_plus if _name == "u0_official" else _opa_tr_step(
                                pipeline, _sc, _kind, y0, prm, opa_rho, opa_n_ascent, opa_eta,
                                +1.0, opa_dir_mode, xi,
                                first_grad=(opa_g0 if _kind == opa_kind else None))
                            with torch.no_grad():
                                # v1(clean 层):在构造目标 y_j 上打分 —— 0055 已证实该量
                                # 与"最终图更好"在训练后期严重脱钩(98.6% vs 52%),故仅留作对照。
                                _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind,
                                                            _yj.detach().float(), prm)
                                _clean = {k: v.clone() for k, v in _GATEA_LAST.items()}
                                # v2(rollout 层):把 y_j 强制进第 kq 步再走完剩余扩散,
                                # 在**最终图**上打分。这才是与"图是否真的更好"对齐的量。
                                _epj = _dpm2_continue_from(zq, kq, _yj, prev_x0, sig_sched, vold)
                                _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind,
                                                            _epj.detach().float(), prm)
                                _roll = {k: v.clone() for k, v in _GATEA_LAST.items()}
                            _gp_rows[_name] = {"clean": _clean, "roll": _roll}
                        global _GATEP_SINK
                        if _GATEP_SINK is None and is_main_process(rank):
                            _GATEP_SINK = open(os.path.join(config.save_dir, "gateP_probe.jsonl"), "a")
                        if _GATEP_SINK is not None:
                            for _i in range(len(_gp_r0["pickscore"])):
                                _GATEP_SINK.write(json.dumps({
                                    "epoch": int(epoch), "kq": int(kq), "sigma_q": float(sig_q),
                                    "r0": {k: float(_gp_r0[k][_i]) for k in _K3},
                                    "r0_roll": {k: float(_gp_xi[k][_i]) for k in _K3},
                                    "cand": {_n: {"clean": {k: float(_v["clean"][k][_i]) for k in _K3},
                                                  "roll": {k: float(_v["roll"][k][_i]) for k in _K3}}
                                             for _n, _v in _gp_rows.items()},
                                }) + "\n")
                            _GATEP_SINK.flush()

                # ── Cert 探针(RPI_CERT_PROBE=1):**只测不门控** ──────────────────
                # 官方 opa_cert=1 会把未认证样本置零 => 改变训练轨迹,
                # 于是"认证率随训练下降"无法与"门控自身造成的下降"分开(0053 §④ caveat 3)。
                # 这里复制同一套计算(_dpm2_continue_from + 终点打分),但**绝不写 opa_ap/opa_am**,
                # 只把认证率落盘。与 Gate A 之于 A1 的关系完全相同。
                if _CERTP_ON and not ri_opa_cert:
                    with torch.no_grad():
                        _cp_ep = _dpm2_continue_from(zq, kq, y_plus, prev_x0, sig_sched, vold)
                        _cp_em = _dpm2_continue_from(zq, kq, y_minus, prev_x0, sig_sched, vold)
                        _cp_ro = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, xi, prm)
                        _cp_rp = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _cp_ep, prm)
                        _cp_rm = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _cp_em, prm)
                    global _CERTP_SINK
                    if _CERTP_SINK is None and is_main_process(rank):
                        _CERTP_SINK = open(os.path.join(config.save_dir, "certP_probe.jsonl"), "a")
                    if _CERTP_SINK is not None:
                        for _i in range(_cp_ro.shape[0]):
                            _CERTP_SINK.write(json.dumps({
                                "epoch": int(epoch), "kq": int(kq), "sigma_q": float(sig_q),
                                "R_old": float(_cp_ro[_i]), "R_plus": float(_cp_rp[_i]),
                                "R_minus": float(_cp_rm[_i]), "margin": float(opa_margin),
                            }) + "\n")
                        _CERTP_SINK.flush()

                if ri_opa_cert:
                    with torch.no_grad():
                        ep_plus = _dpm2_continue_from(zq, kq, y_plus, prev_x0, sig_sched, vold)
                        ep_minus = _dpm2_continue_from(zq, kq, y_minus, prev_x0, sig_sched, vold)
                        R_old = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, xi, prm)
                        R_plus = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, ep_plus, prm)
                        R_minus = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, ep_minus, prm)
                    opa_ap[s:e] = (R_plus >= R_old + opa_margin).float()
                    opa_am[s:e] = (R_minus <= R_old - opa_margin).float()
                elif _A1_ON and _ga_r0 is not None:
                    # ── A1:逐 reward 可行性 veto ────────────────────────────────
                    # 复用官方接受权重:不可行的 state 令 acc_plus=0 => 该样本正损失为 0
                    # (见 pos_loss = accp * (y_pos - y_plus)**2 / wf_p)。不新增损失项。
                    with torch.no_grad():
                        _ = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind,
                                                    y_plus.detach().float(), prm)
                    _a1_rp = {k: v.clone() for k, v in _GATEA_LAST.items()}
                    _feas = None
                    if _A1_MODE == "rel":
                        # A1-v2:逐轴的**相对改善** (rp-r0)/|r0|,要求每轴都 >= 该 state
                        # 三轴相对改善**中位数**的 _A1_RELFRAC 倍。捕捉"某轴被明显冷落"
                        # (§4.21:标量和在原始尺度上做,clipscore 被推的力度是 pickscore 的 3.6 倍)。
                        _rel = {}
                        for _k in sorted(_ga_r0):
                            _rel[_k] = (_a1_rp[_k] - _ga_r0[_k]) / (_ga_r0[_k].abs() + 1e-8)
                        _stk = torch.stack([_rel[_k] for _k in sorted(_rel)], dim=0)
                        _med = _stk.median(dim=0).values
                        _thr = _A1_RELFRAC * _med
                        for _k in sorted(_rel):
                            _ok = (_rel[_k] >= _thr).float()
                            _feas = _ok if _feas is None else _feas * _ok
                    else:
                        for _k in sorted(_ga_r0):
                            _ok = ((_a1_rp[_k] - _ga_r0[_k]) >= -_A1_TOL).float()
                            _feas = _ok if _feas is None else _feas * _ok
                    if _feas is None:
                        raise RuntimeError(
                            "A1: _GATEA_LAST 为空,无法构造可行性掩码。"
                            "通常是 _reward_scores_grad 的填充条件与此处取数条件不一致。")
                    opa_ap[s:e] = _feas.to(opa_ap.device).to(opa_ap.dtype)
                    opa_am[s:e] = 1.0
                    # 不另设指标:既有的 wandb 记录点(本文件 ~L1414)已经在记
                    #   "opa_acc_plus": float(opa_ap.mean())
                    # 而 A1 下 opa_ap 就是可行性掩码 => 该指标自动变成**可行率**。
                    # (canonical 路径下它恒为 1.0,是硬编码常量;A1 下它才第一次含信息。)
                    # ⚠️ 曾想在此 append 到 info_accumulated,但那个变量到 ~L1526 才定义,
                    #    会 NameError —— compile() 查不出来,只有跑起来才炸。
                else:
                    # Canonical no-cert path: skip suffix continuation/scoring and use the
                    # bounded reward-gradient targets directly.
                    opa_ap[s:e] = 1.0
                    opa_am[s:e] = 1.0
                opa_yp[s:e] = y_plus.to(opa_yp.dtype); opa_ym[s:e] = y_minus.to(opa_ym.dtype)
                opa_res[s:e] = (y_plus - y0).flatten(1).norm(dim=1) / (y0.flatten(1).norm(dim=1) + 1e-8)
            transformer_ddp.module.set_adapter("default")
            collated_samples["opa_z_q"] = opa_state_src.contiguous()
            collated_samples["opa_y_plus"] = opa_yp
            if opa_ypq is not None:
                collated_samples["opa_y_plus_q"] = opa_ypq
            collated_samples["opa_y_minus"] = opa_ym
            collated_samples["opa_acc_plus"] = opa_ap
            collated_samples["opa_acc_minus"] = opa_am
            collated_samples["opa_sig_q"] = torch.full((Bn,), float(sig_q), device=device)
            collated_samples.pop("opa_rollout_z_q", None)
            collated_samples.pop("opa_rollout_z_prev", None)
            if is_main_process(rank):
                wandb.log({"opa_kq": kq, "opa_sigma_q": float(sig_q),
                           "opa_acc_plus": float(opa_ap.mean()), "opa_acc_minus": float(opa_am.mean()),
                           "opa_delta_res": float(opa_res.mean())}, step=global_step)

        if ri_refine_on:
            adv_local = collated_samples["advantages"]
            adv_local = adv_local[:, 0] if adv_local.ndim > 1 else adv_local
            x0_target = latents_clean_local.clone()
            pos_idx = torch.nonzero(adv_local > 0, as_tuple=True)[0]
            neg_idx = torch.nonzero(adv_local < 0, as_tuple=True)[0]
            ri_h0, ri_hs, ri_acc = [], [], []
            ri_neg_h0, ri_neg_hs, ri_neg_acc = [], [], []
            reward_std = torch.clamp(collated_samples["rewards"]["avg"].detach().float().std(), min=1e-4)
            for s in range(0, len(pos_idx), ri_mb):
                idx = pos_idx[s:s + ri_mb]
                pr = [epoch_prompts_text[k] for k in idx.tolist()]
                xs, h0, hs, acc = ri_refine(pipeline, ri_scorer, latents_clean_local[idx], pr,
                                            ri_rho, ri_n_ascent, ri_eta, ri_margin, direction=1.0)
                if not ri_aux_mode:
                    x0_target[idx] = xs.to(x0_target.dtype)
                collated_samples["x0_aux_pos"][idx] = xs.to(collated_samples["x0_aux_pos"].dtype)
                gain = (hs - h0 - ri_margin) / reward_std
                if ri_gate_mode == "accept":
                    collated_samples["gate_pos"][idx] = acc.float()
                else:
                    collated_samples["gate_pos"][idx] = acc.float() * torch.sigmoid(ri_gate_beta * gain).detach()
                ri_h0.append(h0); ri_hs.append(hs); ri_acc.append(acc)
            if ri_dual_negative:
                for s in range(0, len(neg_idx), ri_mb):
                    idx = neg_idx[s:s + ri_mb]
                    pr = [epoch_prompts_text[k] for k in idx.tolist()]
                    xs, h0, hs, acc = ri_refine(pipeline, ri_scorer, latents_clean_local[idx], pr,
                                                ri_rho, ri_n_ascent, ri_eta, ri_margin, direction=-1.0)
                    collated_samples["x0_aux_neg"][idx] = xs.to(collated_samples["x0_aux_neg"].dtype)
                    gap = (h0 - hs - ri_margin) / reward_std
                    if ri_gate_mode == "accept":
                        collated_samples["gate_neg"][idx] = acc.float()
                    else:
                        collated_samples["gate_neg"][idx] = acc.float() * torch.sigmoid(ri_gate_beta * gap).detach()
                    ri_neg_h0.append(h0); ri_neg_hs.append(hs); ri_neg_acc.append(acc)
            collated_samples["x0_target"] = x0_target
            if is_main_process(rank):
                ri_log = {
                    "ri_n_pos": float(len(pos_idx)),
                    "ri_n_neg": float(len(neg_idx)),
                    "ri_gate_mode_accept": float(ri_gate_mode == "accept"),
                    "ri_gate_pos_mean": float(collated_samples["gate_pos"].mean()),
                    "ri_gate_neg_mean": float(collated_samples["gate_neg"].mean()),
                    "ri_gate_pos_p90": float(torch.quantile(collated_samples["gate_pos"], 0.90)),
                    "ri_gate_neg_p90": float(torch.quantile(collated_samples["gate_neg"], 0.90)),
                }
                if len(ri_h0) > 0:
                    _h0 = torch.cat(ri_h0); _hs = torch.cat(ri_hs); _acc = torch.cat(ri_acc)
                    ri_log.update({"ri_hps_before": float(_h0.mean()),
                                   "ri_hps_after": float(_hs.mean()),
                                   "ri_gain": float((_hs - _h0).mean()),
                                   "ri_accept_rate": float(_acc.float().mean())})
                if len(ri_neg_h0) > 0:
                    _nh0 = torch.cat(ri_neg_h0); _nhs = torch.cat(ri_neg_hs); _nacc = torch.cat(ri_neg_acc)
                    ri_log.update({"ri_neg_hps_before": float(_nh0.mean()),
                                   "ri_neg_hps_after": float(_nhs.mean()),
                                   "ri_neg_drop": float((_nh0 - _nhs).mean()),
                                   "ri_neg_accept_rate": float(_nacc.float().mean())})
                wandb.log(ri_log, step=global_step)
        else:
            collated_samples["x0_target"] = latents_clean_local

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            # ── 0064 §⑧:按 prompt 切半批(仅 RPI_BNOISE_PROBE=1 且 RPI_BNOISE_SPLIT=prompt)
            # 把 prompt 集合按**文本排序后**对半分(各 rank 见到同一顺序 ⇒ 同一分配),
            # 前一半 microbatch 只含 A 组 prompt,后一半只含 B 组 ⇒ 两半 prompt 不相交。
            # 组内仍随机。不均衡则退回随机切分,并在探针记录里标明 split="random"。
            if _BN_ON and _BN_SPLIT == "prompt":
                global _BN_SPLIT_OK, _BN_SPLIT_DIAG
                _pidx = filtered_samples["prompt_idx"].tolist()
                _ptxt = [epoch_prompts_text[_k] for _k in _pidx]
                _uq = sorted(set(_ptxt))
                _setA = set(_uq[: len(_uq) // 2])
                _isA = torch.tensor([_t in _setA for _t in _ptxt], device=device)
                _a = perm[_isA[perm]]
                _b = perm[~_isA[perm]]
                _BN_SPLIT_DIAG = {"n_uniq_prompt": len(_uq), "nA": int(_a.numel()), "nB": int(_b.numel())}
                _BN_SPLIT_OK = bool(_a.numel() == _b.numel() and _a.numel() > 0)
                if _BN_SPLIT_OK:
                    perm = torch.cat([_a, _b])
                elif is_main_process(rank):
                    logger.info(f"[gateB] prompt 切分不均衡 {_BN_SPLIT_DIAG} -> 退回随机切分")
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]
            if ri_train_state == "rollout" and "rollout_states" in shuffled_filtered_samples:
                # Keep legacy rollout states z_t paired with their shuffled timesteps.
                shuffled_filtered_samples["rollout_states"] = shuffled_filtered_samples["rollout_states"][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale) > 1.0:  # §4.4 train CFG: SAME _g_train as the velocity double below (single source of truth)
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # ---- DiffusionOPSD loss: single reward-gradient query state per trajectory (no timestep loop) ----
                if ri_opa:
                    opa_accum = max(1, config.train.gradient_accumulation_steps)
                    # z_q (the query state the policy conditions on). In the state-provenance ablation
                    # (opa_state_mode="forward") opa_z_q was already built as an offline forward-noised
                    # state AT COLLATION -- so both this state AND its reward target y± come from the
                    # same forward state, and the only difference from the method is state provenance.
                    zq = train_sample_batch["opa_z_q"].float()
                    y_plus = train_sample_batch["opa_y_plus"].float()
                    y_minus = train_sample_batch["opa_y_minus"].float()
                    acc_p = train_sample_batch["opa_acc_plus"].float()
                    acc_m = train_sample_batch["opa_acc_minus"].float()
                    sig_q = float(train_sample_batch["opa_sig_q"][0].item())
                    adv = train_sample_batch["advantages"]
                    adv = (adv[:, 0] if adv.ndim > 1 else adv).float()
                    # §4.4 train-CFG: when g_train>1, embeds/pooled_embeds are already [uncond;cond] (2B)
                    # from the guidance block above -> double the query state, forward once, and combine
                    # v_u + g_train*(v_c - v_u) so old/current training velocities use the SAME guided
                    # velocity as rollout + the OPA anchor. g_train=1 -> _cfg False -> byte-identical.
                    _g_train = getattr(config.sample, "train_guidance_scale", config.sample.guidance_scale)
                    _cfg = _g_train > 1.0
                    zq_in = torch.cat([zq, zq]) if _cfg else zq
                    tt = torch.full([zq_in.shape[0]], sig_q * 1000, device=device, dtype=torch.long)
                    def _cfg_combine(v):
                        if not _cfg:
                            return v
                        v_u, v_c = v.chunk(2)
                        return v_u + _g_train * (v_c - v_u)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            v_old_q = _cfg_combine(transformer_ddp(hidden_states=zq_in, timestep=tt, encoder_hidden_states=embeds,
                                                      pooled_projections=pooled_embeds, return_dict=False)[0]).detach()
                        transformer_ddp.module.set_adapter("default")
                        v_theta_q = _cfg_combine(transformer_ddp(hidden_states=zq_in, timestep=tt, encoder_hidden_states=embeds,
                                                    pooled_projections=pooled_embeds, return_dict=False)[0])
                    v_old_q = v_old_q.float()
                    v_pos = config.beta * v_theta_q.float() + (1 - config.beta) * v_old_q
                    v_neg = (1.0 + config.beta) * v_old_q - config.beta * v_theta_q.float()
                    y_pos = zq - sig_q * v_pos
                    y_neg = zq - sig_q * v_neg
                    rd = tuple(range(1, y_pos.ndim))

                    # ── RTC 可达性探针(0067 §⑦)。每个累积周期只做一次(首个 microbatch)。
                    # 纯读:retain_graph 保住主 backward 的图,不写 .grad、不进 loss、不动参数。
                    if _RTCR_ON and current_accumulated_steps % opa_accum == 0:
                        global _RTCR_SINK
                        try:
                            # 先释放"已保留未分配"的缓存(A100 80GB 上实测训练基线已占 79.2/79.25 GiB,
                            # 探针多要 32MiB 就 OOM;错误信息里 reserved-but-unallocated 有 1.3GiB)。
                            # 每 epoch 只调一次,开销可忽略。
                            torch.cuda.empty_cache()
                            _rt_lp = [_p for _p in transformer_ddp.module.parameters() if _p.requires_grad]
                            _rt_pt = [epoch_prompts_text[_k]
                                      for _k in train_sample_batch["prompt_idx"].tolist()]
                            # g = ∇_y R(y0):就地重算(与 :1467 同一调用式),y0 = rollout 的 clean x0
                            _rt_y0 = train_sample_batch["latents_clean"].detach().float().requires_grad_(True)
                            _rt_r = _reward_of_latents_grad(pipeline, ri_scorer, opa_kind, _rt_y0, _rt_pt)
                            (_rt_g,) = torch.autograd.grad(_rt_r.sum(), _rt_y0)
                            _rt_g = _rt_g.detach().float()

                            def _rt_R(_d):
                                """R(d) = ‖Jᵀd‖²/‖d‖²,一次 VJP。J = ∂y_pos/∂θ。"""
                                _gs = torch.autograd.grad((y_pos * _d).sum(), _rt_lp,
                                                          retain_graph=True, allow_unused=True)
                                _n2 = sum(float((_x.detach().float() ** 2).sum())
                                          for _x in _gs if _x is not None)
                                return _n2 / (float((_d ** 2).sum()) + 1e-30)

                            # 0067 §⑩ 判别 (A) 逐样本衰减 vs (B) 跨样本抵消:
                            # 单样本上重算,和的范数里没有跨样本项 ⇒ 只反映逐样本衰减。
                            def _rt_R1(_d, _i):
                                _gs = torch.autograd.grad((y_pos[_i] * _d[_i]).sum(), _rt_lp,
                                                          retain_graph=True, allow_unused=True)
                                _n2 = sum(float((_x.detach().float() ** 2).sum())
                                          for _x in _gs if _x is not None)
                                return _n2 / (float((_d[_i] ** 2).sum()) + 1e-30)

                            _R_g = _rt_R(_rt_g)
                            # 0067 §⑨ 加强零模型:沿 batch 维循环移位。保留 g 的空间频谱、
                            # 范数尺度与"真实 reward 梯度"属性,只破坏"g 与样本的配对"。
                            # 各向同性高斯分不开"reward 特异"与"光滑方向本就难达"。
                            _R_gp = _rt_R(torch.roll(_rt_g, 1, dims=0)) if _rt_g.shape[0] > 1 else None
                            _R_u = [_rt_R(torch.randn_like(_rt_g)) for _ in range(_RTCR_K)]
                            # 单样本版:对前 2 个样本各算 匹配/错配 一次(共 4 次 VJP)
                            _r1m = []; _r1p = []
                            if _rt_g.shape[0] > 1:
                                _grl = torch.roll(_rt_g, 1, dims=0)
                                for _i0 in range(min(2, _rt_g.shape[0])):
                                    _r1m.append(_rt_R1(_rt_g, _i0))
                                    _r1p.append(_rt_R1(_grl, _i0))
                            _mu = sum(_R_u) / len(_R_u)
                            _sd = (sum((x - _mu) ** 2 for x in _R_u) / max(len(_R_u) - 1, 1)) ** 0.5
                            if _RTCR_SINK is None and is_main_process(rank):
                                _RTCR_SINK = open(os.path.join(config.save_dir, "rtcReach_probe.jsonl"), "a")
                            if _RTCR_SINK is not None and _mu > 0:
                                _RTCR_SINK.write(json.dumps({
                                    "epoch": int(epoch), "upd": int(gradient_update_times),
                                    "R_g": _R_g, "R_u_mean": _mu, "R_u_sd": _sd, "K": len(_R_u),
                                    "ratio": _R_g / _mu,
                                    **({"R_gperm": _R_gp, "ratio_perm": _R_g / _R_gp}
                                       if _R_gp else {}),
                                    **({"R1_match": _r1m, "R1_perm": _r1p,
                                        "ratio_R1": (sum(_r1m)/len(_r1m))/(sum(_r1p)/len(_r1p))}
                                       if _r1m and _r1p and sum(_r1p) > 0 else {}),
                                }) + "\n")
                                _RTCR_SINK.flush()
                            del _rt_g, _rt_y0, _rt_r
                        except Exception as _rt_e:        # 探针不得影响训练
                            if is_main_process(rank):
                                logger.info(f"[rtcReach] 探针异常(已跳过): {type(_rt_e).__name__}: {_rt_e}")
                    adv_clip = torch.clamp(adv, -config.train.adv_clip_max, config.train.adv_clip_max)
                    r1 = torch.clamp((adv_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)
                    with torch.no_grad():
                        wf_p = torch.abs(y_pos.double() - y_plus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                        wf_n = torch.abs(y_neg.double() - y_minus.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                        if _N2_ON:
                            wf_p = _n2_freeze(wf_p, "p")
                            wf_n = _n2_freeze(wf_n, "n")
                    accp = acc_p.view(-1, *([1] * (y_pos.ndim - 1)))
                    accm = acc_m.view(-1, *([1] * (y_pos.ndim - 1)))
                    pos_loss = (accp * (y_pos - y_plus) ** 2 / wf_p).mean(dim=rd)
                    neg_loss = (accm * (y_neg - y_minus) ** 2 / wf_n).mean(dim=rd)
                    # ── Gate Q2a:共享 LoRA 的**逐样本参数梯度**相容性(0062 §②)────────
                    # Codex 0058 §7:Gate A 的 +0.715 是 reward-gain 相关,**不是** parameter-gradient
                    # cosine;共享 LoRA 的梯度相容性至今没在干净官方 baseline 上测过。
                    # 这里对同一批样本,分别在 canonical 目标与 Q 目标下求 ∇_θ(逐样本拟合损失),
                    # 报告两两余弦 —— 回答"第二段损失是否来自跨样本梯度抵消"。
                    if _Q2A_ON and "opa_y_plus_q" in train_sample_batch:
                        _ypq = train_sample_batch["opa_y_plus_q"].float()
                        with torch.no_grad():
                            _wfq = torch.abs(y_pos.double() - _ypq.double()).mean(
                                dim=rd, keepdim=True).clip(min=1e-5)
                        _pl_q = (accp * (y_pos - _ypq) ** 2 / _wfq).mean(dim=rd)
                        _lp = [_p for _p in transformer_ddp.module.parameters() if _p.requires_grad]
                        _ns = min(_Q2A_NSAMP, int(pos_loss.shape[0]))

                        # Q2c:同 prompt vs 跨 prompt 的相干性。训练批在 L1777 被 randperm
                        # 全局打乱,故随机 6 个样本里同 prompt 的概率仅 ~2% —— 统计量太少。
                        # 这里**优先挑出成对同 prompt 的样本**,再随机补足;
                        # 每对是否同 prompt 都逐条记录,故对比仍然有效。
                        _sel = list(range(_ns))
                        _phash = None
                        try:
                            _pi = train_sample_batch["prompt_idx"].tolist()
                            _pt = [epoch_prompts_text[_k] for _k in _pi]
                            _by = {}
                            for _j, _t in enumerate(_pt):
                                _by.setdefault(_t, []).append(_j)
                            _pairs = [v for v in _by.values() if len(v) >= 2]
                            _pref = []
                            for _grp in _pairs[:2]:            # 最多取 2 组同 prompt
                                _pref.extend(_grp[:2])
                            _rest = [_j for _j in range(len(_pt)) if _j not in _pref]
                            _sel = (_pref + _rest)[:_ns]
                            _phash = [abs(hash(_pt[_j])) % 1000000 for _j in _sel]
                        except Exception:
                            _sel = list(range(_ns)); _phash = None


                        def _persample_grad(_losses, _i):
                            _gs = torch.autograd.grad(_losses[_i], _lp, retain_graph=True,
                                                      allow_unused=True)
                            return torch.cat([_g.reshape(-1) for _g in _gs
                                              if _g is not None]).detach().float()

                        def _pairwise_cos(_losses, _out_nrm=None):
                            _V = [_persample_grad(_losses, _i) for _i in _sel]
                            # 0065 §⑤b:余弦法丢掉范数,而 ρ=‖E[ĝ]‖² 与 ‖E[g]‖²/E‖g‖²
                            # 只在范数齐次时相等。记下范数才能与半批法对账。
                            if _out_nrm is not None:
                                _out_nrm.extend(float(_v.norm()) for _v in _V)
                            _V = [_v / (_v.norm() + 1e-12) for _v in _V]
                            return [float((_V[_a] * _V[_b]).sum())
                                    for _a in range(_ns) for _b in range(_a + 1, _ns)]

                        _nrm_canon = []
                        _cos_canon = _pairwise_cos(pos_loss, _nrm_canon)
                        _cos_q = _pairwise_cos(_pl_q)

                        # Q2b-lite:AdamW 的对角预条件 P = 1/(√v+eps) 会**逐坐标缩放**梯度,
                        # 从而旋转方向。问:它把逐样本梯度转得**更拢**还是**更散**?
                        # 这是 0062/0063 转向处方的第 2 条,也是最后一个便宜的诊断。
                        _cos_pc = None
                        try:
                            _v = []
                            for _pp in _lp:
                                _st = optimizer.state.get(_pp, {})
                                _eas = _st.get("exp_avg_sq")
                                _v.append(_eas.reshape(-1) if _eas is not None
                                          else torch.zeros(_pp.numel(), device=_pp.device))
                            _vv = torch.cat(_v).float()
                            if float(_vv.abs().sum()) > 0:      # 优化器已有状态才做
                                _P = 1.0 / (_vv.sqrt() + 1e-8)
                                _Vp = [_persample_grad(pos_loss, _i) * _P for _i in _sel]
                                _Vp = [_x / (_x.norm() + 1e-12) for _x in _Vp]
                                _cos_pc = [float((_Vp[_a] * _Vp[_b]).sum())
                                           for _a in range(_ns) for _b in range(_a + 1, _ns)]
                                del _Vp, _P
                            del _vv, _v
                        except Exception as _e:                # 探针不得影响训练
                            _cos_pc = None

                        global _Q2A_SINK
                        if _Q2A_SINK is None and is_main_process(rank):
                            _Q2A_SINK = open(os.path.join(config.save_dir, "gateQ2a_probe.jsonl"), "a")
                        if _Q2A_SINK is not None:
                            _Q2A_SINK.write(json.dumps({
                                "epoch": int(epoch), "n_samp": _ns,
                                "cos_canon": _cos_canon, "cos_q": _cos_q,
                                **({"nrm_canon": _nrm_canon} if _nrm_canon else {}),
                                **({"phash": _phash} if _phash else {}),
                                **({"cos_precond": _cos_pc} if _cos_pc else {}),
                            }) + "\n")
                            _Q2A_SINK.flush()
                        del _pl_q

                    if not ri_opa_dual_neg:
                        neg_loss = neg_loss * 0.0   # ablation: positive-branch only
                    if opa_target_mode == "aux":
                        # Ablation (target replacement vs auxiliary): instead of REPLACING
                        # the NFT endpoint with the refined target, keep the vanilla-NFT loss (predict the
                        # UN-refined rollout endpoint x_i = latents_clean) and ADD the reward-gradient
                        # displacement (pos_loss/neg_loss above, which pull toward the refined y±) as a
                        # SEPARATE, scale-matched auxiliary term weighted by opa_aux_lambda. Tests whether
                        # OPSD helps because it is a target replacement, or merely an aux reward regularizer.
                        # Scale match: identical |·|-normalization (/wf, clip 1e-5) and *adv_clip_max as the
                        # policy term, so opa_aux_lambda=1 makes the aux comparable to the main NFT loss.
                        xi_tgt = train_sample_batch["latents_clean"].float()
                        with torch.no_grad():
                            wf_p0 = torch.abs(y_pos.double() - xi_tgt.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                            wf_n0 = torch.abs(y_neg.double() - xi_tgt.double()).mean(dim=rd, keepdim=True).clip(min=1e-5)
                        # Vanilla NFT loss toward the un-refined endpoint x_i: no accept-gating (there is
                        # no ascended target here to certify), unlike the reward-target aux term below.
                        pos_main = ((y_pos - xi_tgt) ** 2 / wf_p0).mean(dim=rd)
                        neg_main = ((y_neg - xi_tgt) ** 2 / wf_n0).mean(dim=rd)
                        if not ri_opa_dual_neg:
                            neg_main = neg_main * 0.0
                        main_policy = (r1 * pos_main + (1.0 - r1) * neg_main).mean() * config.train.adv_clip_max
                        aux_policy = (r1 * pos_loss + (1.0 - r1) * neg_loss).mean() * config.train.adv_clip_max
                        opa_policy = main_policy + opa_aux_lambda * aux_policy
                    else:  # "replace" (method default): the refined target IS the NFT endpoint
                        opa_policy = (r1 * pos_loss + (1.0 - r1) * neg_loss).mean() * config.train.adv_clip_max
                    # Canonical OPSD uses only target fitting.  A frozen-base prediction
                    # MSE can be tried as an optional FlowGRPO-style stabilizer, but the
                    # tested coefficient (1e-4) had no measurable effect on the result.
                    loss = opa_policy
                    scaled_loss = loss / opa_accum
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    profiling.train_bwd_inc()  # §6: diffusion (training) backward count — OPA single-query-state path
                    current_accumulated_steps += 1

                    # ── Gate B 探针(0064)。位置必须在 unscale_/clip_grad_norm_ 之前:
                    # 两次快照要处于同一 GradScaler 缩放下(B_noise 是比值量,scale 自动约掉),
                    # 且 clip 会重标定 g_full 却不动快照。DDP 每次反传已跨秩 all-reduce,
                    # 故 .grad 就是全局量;两个半批仍是不相交样本集,估计量不变。
                    if _BN_ON and opa_accum >= 2:
                        global _BN_HALF, _BN_SINK
                        _bn_h = opa_accum // 2
                        _bn_k = current_accumulated_steps % opa_accum
                        _bn_lp = [_p for _p in transformer_ddp.module.parameters() if _p.requires_grad]
                        if _bn_k == _bn_h:
                            _BN_HALF = [(_p.grad.detach().float().clone() if _p.grad is not None else None)
                                        for _p in _bn_lp]
                        elif _bn_k == 0 and _BN_HALF is not None and len(_BN_HALF) == len(_bn_lp):
                            _bn_dot = 0.0
                            _bn_f2 = 0.0
                            for _p, _gh in zip(_bn_lp, _BN_HALF):
                                if _p.grad is None or _gh is None:
                                    continue
                                _gf = _p.grad.detach().float()
                                _bn_dot += float((_gh.double() * (_gf - _gh).double()).sum())
                                _bn_f2 += float((_gf.double() * _gf.double()).sum())
                            # 半批均值 ḡ_A = g_half·(accum/h),ḡ_B = (g_full−g_half)·(accum/(accum−h));
                            # 一般写法,不假设 accum 为偶数。
                            _bn_ab = _bn_dot * (opa_accum / _bn_h) * (opa_accum / (opa_accum - _bn_h))
                            if _BN_SINK is None and is_main_process(rank):
                                _BN_SINK = open(os.path.join(config.save_dir, "gateB_noise.jsonl"), "a")
                            if _BN_SINK is not None and np.isfinite(_bn_ab) and np.isfinite(_bn_f2):
                                _BN_SINK.write(json.dumps({
                                    "epoch": int(epoch), "upd": int(gradient_update_times),
                                    "dot_ab": _bn_ab, "nrm_full2": _bn_f2,
                                    "accum": int(opa_accum), "half": int(_bn_h),
                                    "bs": int(config.train.batch_size), "world": int(world_size),
                                    "split": ("prompt" if (_BN_SPLIT == "prompt" and _BN_SPLIT_OK) else "random"),
                                    **({"split_diag": _BN_SPLIT_DIAG} if _BN_SPLIT_DIAG else {}),
                                }) + "\n")
                                _BN_SINK.flush()
                            _BN_HALF = None
                    _qd = {}
                    if _QOPSD_ON and _QOPSD_DIAG:
                        _tot = sum(_QOPSD_DIAG.values()) or 1
                        _qd = {f"{k}_frac": torch.tensor(v / _tot, device=device)
                               for k, v in _QOPSD_DIAG.items()}
                    for kk, vv in {"opa_policy_loss": opa_policy.detach(), **_qd,
                                   **(_N2_DIAG if _N2_ON else {}),
                                   "total_loss": loss.detach(), "opa_pos_loss": pos_loss.mean().detach(),
                                   "opa_neg_loss": neg_loss.mean().detach(),
                                   # 实现门:opa_policy=(r1·pos+(1-r1)·neg).mean(),其对 r1 的
                                   # 敏感度**恰好**是 (pos-neg)。所以"r1 是否真进梯度"的充要条件
                                   # 就是**逐样本** |pos-neg| 显著非零。只比两者的均值会被抵消,
                                   # §4.145 ① 的退化正是这样漏掉的。
                                   "opa_posneg_absdiff": (pos_loss - neg_loss).abs().mean().detach(),
                                   "opa_acc_plus_b": acc_p.mean().detach(),
                                   "opa_acc_minus_b": acc_m.mean().detach()}.items():
                        info_accumulated[kk].append(vv)
                    if current_accumulated_steps % opa_accum == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer); scaler.update()
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        optimizer.zero_grad()
                        log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG, group=POLICY_GROUP)  # None => default world
                        reduced = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log({"step": global_step, "gradient_update_times": gradient_update_times,
                                       "epoch": epoch, "inner_epoch": inner_epoch, **reduced})
                        global_step += 1
                        info_accumulated = defaultdict(list)
                        if config.train.ema and ema is not None:
                            ema.step(transformer_trainable_parameters, global_step)
                    continue

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    # Main DiffusionNFT target.
                    # - RI-SDAR-Aux: keep the stable original endpoint x_i as the main target.
                    # - Legacy replace mode: use x0_target, where positive samples may be x_i^+.
                    x0 = train_sample_batch["latents_clean"] if ri_aux_mode else train_sample_batch["x0_target"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    if ri_train_state == "rollout":
                        # OPD: train on the REAL inference-rollout state z_t (train-inference consistency).
                        # E_t(z_t, v) = z_t - t*v is the scheduler's own one-step x0 estimate at this state.
                        xt = train_sample_batch["rollout_states"][:, j_idx].float()
                    else:
                        # RI-NFT: forward-process re-noising toward the (refined) endpoint.
                        noise = torch.randn_like(x0.float())
                        xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False

                    # --- ATC-X0-NFT target override ---------------------------------------
                    # Replace the (B/B0) endpoint target with a per-state clipped self-consistency
                    # target y_tgt = y0 + clip(x_i - y0, rho_x0*||y0||), anchored at this rollout
                    # state's own old-policy x0 estimate y0 = z_t - t*v_old. The endpoint x_i is
                    # used only to define a tiny trust-region direction, never as the target itself.
                    # Restricted to the low-noise t-band (atc_band gates the gradient DDP-safely,
                    # since t is shared across ranks -> identical mask on every rank).
                    atc_band = 1.0
                    if ri_atc_x0:
                        with torch.no_grad():
                            atc_rd = tuple(range(1, x0.ndim))
                            y0 = (xt - t_expanded * old_prediction).float()
                            x_end_i = train_sample_batch["latents_clean"].float()
                            d_raw = x_end_i - y0
                            y0norm = y0.pow(2).sum(dim=atc_rd, keepdim=True).sqrt().clamp(min=1e-8)
                            dnorm = d_raw.pow(2).sum(dim=atc_rd, keepdim=True).sqrt().clamp(min=1e-12)
                            budget = ri_rho_x0 * y0norm
                            d = d_raw * torch.clamp(budget / dnorm, max=1.0)
                            x0 = (y0 + d).to(x0.dtype)
                        tval = float(t.reshape(-1)[0].item())
                        atc_band = 1.0 if (ri_t_lo <= tval <= ri_t_hi) else 0.0
                    loss_terms = {}
                    if ri_atc_x0:
                        loss_terms["atc_in_band"] = torch.as_tensor(atc_band, device=device, dtype=torch.float32)
                        loss_terms["atc_x0_res"] = (dnorm / y0norm).mean().detach()
                        loss_terms["atc_x0_delta"] = (torch.minimum(budget, dnorm) / y0norm).mean().detach()
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    if ri_aux_mode and ri_lambda_aux > 0:
                        # SDAR-style gated auxiliary:
                        #   keep L_NFT(x_i, A_i) as the backbone, and add only a small local
                        #   target-distillation term. Positive samples attract v_pos toward better
                        #   x_i^+; negative samples use the DiffusionNFT implicit negative branch
                        #   to repel v_theta from worse x_i^-.
                        reduce_dims = tuple(range(1, x0.ndim))
                        x0_aux_pos = train_sample_batch["x0_aux_pos"].to(x0_prediction.dtype)
                        x0_aux_neg = train_sample_batch["x0_aux_neg"].to(negative_x0_prediction.dtype)
                        gate_pos = train_sample_batch["gate_pos"].to(x0_prediction.dtype)
                        gate_neg = train_sample_batch["gate_neg"].to(x0_prediction.dtype)

                        with torch.no_grad():
                            pos_aux_weight = (
                                torch.abs(x0_prediction.double() - x0_aux_pos.double())
                                .mean(dim=reduce_dims, keepdim=True)
                                .clip(min=0.00001)
                            )
                            neg_aux_weight = (
                                torch.abs(negative_x0_prediction.double() - x0_aux_neg.double())
                                .mean(dim=reduce_dims, keepdim=True)
                                .clip(min=0.00001)
                            )
                        pos_aux_loss = ((x0_prediction - x0_aux_pos) ** 2 / pos_aux_weight).mean(dim=reduce_dims)
                        neg_aux_loss = ((negative_x0_prediction - x0_aux_neg) ** 2 / neg_aux_weight).mean(dim=reduce_dims)
                        ri_aux_raw = (gate_pos * pos_aux_loss + gate_neg * neg_aux_loss).mean()
                        ri_aux_loss = ri_aux_raw * config.train.adv_clip_max
                        ri_aux_weighted = ri_lambda_aux * ri_aux_loss
                        loss = loss + ri_aux_weighted
                        loss_terms["ri_aux_loss"] = ri_aux_loss.detach()
                        loss_terms["ri_aux_weighted"] = ri_aux_weighted.detach()
                        loss_terms["ri_aux_to_policy"] = (
                            ri_aux_weighted.detach() / (policy_loss.detach().abs() + 1e-8)
                        )
                        loss_terms["gate_pos_mean"] = gate_pos.mean().detach()
                        loss_terms["gate_neg_mean"] = gate_neg.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    # ATC-X0-NFT: only the low-noise t-band contributes gradient. Zero the whole
                    # per-state loss out-of-band (t is identical across ranks -> DDP-safe; the
                    # accumulation counter is untouched so grad-accum cadence is unchanged).
                    if ri_atc_x0 and atc_band == 0.0:
                        loss = loss * 0.0
                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    profiling.train_bwd_inc()  # §6: diffusion training backward count
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG, group=POLICY_GROUP)  # None => default world
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        if prof is not None:
            prof.epoch_end(epoch - prof_epoch0, global_step=global_step)
            if prof.done(epoch - prof_epoch0):
                prof.finalize(_profile_sanity_eval)
                break

    if prof is not None and not prof.finalized:  # safety net if num_epochs < warmup+measure
        prof.finalize(_profile_sanity_eval)

    if not config.debug:
        save_ckpt(
            config.save_dir, transformer_ddp, global_step, rank, ema,
            transformer_trainable_parameters, config, optimizer, scaler,
            epoch_completed=config.num_epochs,
        )
    if world_size > 1:
        dist.barrier(group=POLICY_GROUP)  # POLICY_GROUP=None => default world

    if is_main_process(rank):
        try:
            with open(os.path.join(config.save_dir, "run_done.json"), "w") as f:
                json.dump({"wall_clock_end": datetime.datetime.now().isoformat(), "global_step": global_step}, f, indent=2)
        except Exception:
            pass
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
