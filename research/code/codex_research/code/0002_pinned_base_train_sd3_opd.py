from collections import defaultdict
import contextlib
import os
import glob
import re as _re
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.solver import VALID_SOLVERS, solver_step
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
import wandb
from functools import partial, wraps
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import torch.nn.functional as F
from flow_grpo.pickscore_scorer import PickScoreScorer
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

def make_no_grad(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return fn(*args, **kwargs)
    return wrapper

def _build_teachers_config(config):
    global_proxy = dict(getattr(config, "proxy_reward_fn", {}) or {})
    global_prompt_fn = getattr(config, "prompt_fn", "general_ocr")
    global_teacher_gs = float(
        getattr(
            config.sample,
            "teacher_guidance_scale",
            getattr(config.sample, "guidance_scale", 1.0),
        )
    )

    def _resolve_gs(value, default):
        return float(default if value is None else value)

    teachers = getattr(config.train, "teachers", None)
    if not teachers:
        raise ValueError("OPD training requires `config.train.teachers`.")

    out = []
    for t in teachers:
        entry = dict(t)
        for required in ("name", "dataset", "lora_path"):
            if required not in entry:
                raise ValueError(
                    f"Teacher entry {entry!r} is missing required key '{required}'."
                )
        entry["proxy_reward"] = dict(entry.get("proxy_reward") or global_proxy)
        entry["prompt_fn"] = entry.get("prompt_fn") or global_prompt_fn
        entry["guidance_scale"] = _resolve_gs(
            entry.get("guidance_scale"), global_teacher_gs
        )
        out.append(entry)

    names = [t["name"] for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"Teacher slot names must be unique, got {names}.")
    return out

from accelerate.state import AcceleratorState

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
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

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

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
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:

            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

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

class DistributedGroupedKSampler(Sampler):
    """同 prompt 的 k 个副本**保持在同一 rank 的连续位置**(原 DistributedKRepeatSampler 会全局 shuffle,
    导致一个 group 被拆到不同 rank,无法在 rank 内做组内 reward 加权)。

    ⚠️ 这是一个**新增**类,原 DistributedKRepeatSampler 完全不动 —— baseline / mopd / caopd 等已有实验
    仍走原采样器,行为不受影响。仅 on-manifold 搜索(config.train.onmanifold)使用本类。

    要求 batch_size % k == 0:每 rank 拿 batch_size//k 个不同 prompt,各重复 k 次且相邻。
    """

    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        assert batch_size % k == 0, f"batch_size({batch_size}) 必须是 k({k}) 的整数倍"
        self.dataset, self.batch_size, self.k = dataset, batch_size, k
        self.num_replicas, self.rank, self.seed = num_replicas, rank, seed
        self.groups_per_rank = batch_size // k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
            need = self.groups_per_rank * self.num_replicas
            idx = torch.randperm(len(self.dataset), generator=g)[:need].tolist()
            start = self.rank * self.groups_per_rank
            mine = idx[start:start + self.groups_per_rank]
            yield [i for i in mine for _ in range(self.k)]     # 每个 prompt 连续重复 k 次

    def set_epoch(self, epoch):
        self.epoch = epoch


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:

        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def _resolve_solver(config):

    solver = str(getattr(config.sample, "solver", "flow"))
    deterministic = bool(getattr(config.sample, "deterministic", False))
    if solver not in VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver: {solver!r}. Expected one of {VALID_SOLVERS}."
        )
    return solver, deterministic

def compute_student_step(transformer, pipeline, sample, j, embeds, pooled_embeds, config):

    solver, deterministic = _resolve_solver(config)

    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    if solver == "flow" and not deterministic:

        _, _, prev_sample_mean, std_dev_t = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            sample["timesteps"][:, j],
            sample["latents"][:, j].float(),
            prev_sample=sample["next_latents"][:, j].float(),
            noise_level=config.sample.noise_level,
        )
    else:
        _, prev_sample_mean, std_dev_t, _ = solver_step(
            scheduler=pipeline.scheduler,
            model_output=noise_pred.float(),
            timestep=sample["timesteps"][:, j],
            sample=sample["latents"][:, j].float(),
            solver=solver,
            noise_level=config.sample.noise_level,
            deterministic=deterministic,
            prev_sample=sample["next_latents"][:, j].float(),
        )
    return prev_sample_mean, std_dev_t

def compute_base_velocity(transformer, accelerator, latents_j, timesteps_j, embeds, pooled_embeds, config):
    """Frozen BASE (no-LoRA) velocity at one step, for the value-gradient virtual teacher.
    Mirrors compute_student_step's forward but with the LoRA adapter disabled and no grad."""
    with torch.no_grad():
        with unwrap_model(transformer, accelerator).disable_adapter():
            if config.train.cfg:
                noise_pred = transformer(
                    hidden_states=torch.cat([latents_j] * 2),
                    timestep=torch.cat([timesteps_j] * 2),
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + config.sample.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            else:
                noise_pred = transformer(
                    hidden_states=latents_j,
                    timestep=timesteps_j,
                    encoder_hidden_states=embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]
    return noise_pred.float()

def _proj_simplex(v):
    """Euclidean projection of vector v onto the probability simplex."""
    v = v.float()
    u, _ = torch.sort(v, descending=True)
    css = torch.cumsum(u, 0) - 1.0
    ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
    cond = u - css / ind > 0
    rho = int(cond.to(torch.int).sum().clamp(min=1).item())
    theta = css[rho - 1] / rho
    return (v - theta).clamp(min=0.0)

def _mgda_minnorm_weights(G, iters=60):
    """Min-norm point in the convex hull of the per-teacher gradients (MGDA): argmin_w w^T G w over the
    simplex. G = Gram matrix of per-teacher gradient directions. Returns weights w (sum to 1)."""
    M = G.shape[0]
    w = torch.full((M,), 1.0 / M, device=G.device, dtype=torch.float32)
    lr = 1.0 / (2.0 * G.diagonal().abs().max().clamp(min=1e-8))
    for _ in range(iters):
        w = _proj_simplex(w - lr * (2.0 * (G @ w)))
    return w

def compute_teacher_step_mean(
    transformer,
    pipeline,
    latents_at_j,
    next_latent_at_j,
    timestep_at_j,
    cond_embeds,
    cond_pooled,
    neg_embeds,
    neg_pooled,
    guidance_scale,
    noise_level,
    solver: str = "flow",
    deterministic: bool = False,
):

    if guidance_scale > 1.0:
        noise_pred = transformer(
            hidden_states=torch.cat([latents_at_j] * 2),
            timestep=torch.cat([timestep_at_j] * 2),
            encoder_hidden_states=torch.cat([neg_embeds, cond_embeds]),
            pooled_projections=torch.cat([neg_pooled, cond_pooled]),
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
    else:
        noise_pred = transformer(
            hidden_states=latents_at_j,
            timestep=timestep_at_j,
            encoder_hidden_states=cond_embeds,
            pooled_projections=cond_pooled,
            return_dict=False,
        )[0]

    if solver == "flow" and not deterministic:
        _, _, prev_sample_mean, _ = sde_step_with_logprob(
            pipeline.scheduler,
            noise_pred.float(),
            timestep_at_j,
            latents_at_j.float(),
            prev_sample=next_latent_at_j.float(),
            noise_level=noise_level,
        )
    else:
        _, prev_sample_mean, _, _ = solver_step(
            scheduler=pipeline.scheduler,
            model_output=noise_pred.float(),
            timestep=timestep_at_j,
            sample=latents_at_j.float(),
            solver=solver,
            noise_level=noise_level,
            deterministic=deterministic,
            prev_sample=next_latent_at_j.float(),
        )
    return prev_sample_mean

def _eval_one_teacher(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    proxy_reward_fn,
    executor,
    autocast,
    log_prefix,
    save_subdir,
):
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    local_images = []
    local_captions = []
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc=f"Eval[{log_prefix or 'main'}]: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )

        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                eval_kw = dict(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    guidance_scale=config.sample.eval_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                    deterministic=True,
                    num_inference_steps=config.sample.eval_num_steps,
                )
                images, _, _ = pipeline_with_logprob(pipeline, **eval_kw)
            with torch.no_grad():
                primary_fut = executor.submit(
                    proxy_reward_fn, images, prompts, prompt_metadata, only_strict=False
                )
                time.sleep(0)
                proxy_rewards, _ = primary_fut.result()

        proc_id = accelerator.process_index
        for idx, i in enumerate(range(len(images))):
            image = images[i]
            pil = Image.fromarray(
                (image.detach().float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            )
            pil = pil.resize((config.resolution, config.resolution))

            avg_proxy = proxy_rewards["avg"][i].item() if hasattr(proxy_rewards["avg"][i], "item") else float(proxy_rewards["avg"][i])
            caption = f"GPU:{proc_id} | {prompts[i]} | proxy: {avg_proxy:.2f}"

            local_images.append(pil)
            local_captions.append(caption)

        for key, value in proxy_rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).detach().float().cpu().numpy()
            all_rewards[f"proxy_{key}"].append(rewards_gather)

    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size = dist.get_world_size()
        all_images = [None] * world_size
        all_captions = [None] * world_size
        dist.all_gather_object(all_images, local_images)
        dist.all_gather_object(all_captions, local_captions)
        all_images = [item for sublist in all_images for item in sublist]
        all_captions = [item for sublist in all_captions for item in sublist]
    else:
        all_images = local_images
        all_captions = local_captions
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    if accelerator.is_main_process:
        num_samples = min(100, len(all_images))
        sample_indices = list(range(num_samples))
        sampled_images = [all_images[i] for i in sample_indices]
        sampled_captions = [all_captions[i] for i in sample_indices]
        eval_images = [wandb.Image(pil, caption=caption) for pil, caption in zip(sampled_images, sampled_captions)]
        for key, value in all_rewards.items():
            print(f"{log_prefix}{key}", value.shape)
        wandb.log(
            {
                f"eval_{log_prefix}images": eval_images,
                **{
                    f"eval_{log_prefix}monitor_{key}": np.mean(value[value != -10])
                    for key, value in all_rewards.items()
                },
                **{
                    f"eval_{log_prefix}monitor_{key}_std": np.std(value[value != -10])
                    for key, value in all_rewards.items()
                },
            },
            step=global_step,
        )
        save_eval_dir = getattr(config, "save_eval_dir", None)
        if save_eval_dir:
            save_dir = os.path.join(save_eval_dir, f"step_{global_step:04d}")
            if save_subdir:
                save_dir = os.path.join(save_dir, save_subdir)
            os.makedirs(save_dir, exist_ok=True)
            for i, image in enumerate(all_images[:num_samples]):
                image.save(os.path.join(save_dir, f"image_{i:04d}.png"))
    metrics: dict[str, float] = {}
    if accelerator.is_main_process:
        for key, value in all_rewards.items():
            if "_strict_accuracy" in key or "_accuracy" in key:
                continue
            valid = value[value != -10]
            if valid.size > 0:
                metrics[key] = float(np.mean(valid))
    return metrics

def eval(
    pipeline,
    test_dataloaders,
    teachers,
    text_encoders,
    tokenizers,
    config,
    accelerator,
    global_step,
    proxy_reward_fns_per_teacher,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    use_prefix = len(teachers) > 1
    metrics_per_slot: dict[int, dict[str, float]] = {}
    for slot_idx, (tc, loader, proxy_reward_fn) in enumerate(
        zip(teachers, test_dataloaders, proxy_reward_fns_per_teacher)
    ):
        if proxy_reward_fn is None:

            continue
        log_prefix = f"{tc['name']}_" if use_prefix else ""
        save_subdir = tc["name"] if use_prefix else None
        slot_metrics = _eval_one_teacher(
            pipeline=pipeline,
            test_dataloader=loader,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            config=config,
            accelerator=accelerator,
            global_step=global_step,
            proxy_reward_fn=proxy_reward_fn,
            executor=executor,
            autocast=autocast,
            log_prefix=log_prefix,
            save_subdir=save_subdir,
        )
        metrics_per_slot[slot_idx] = slot_metrics

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

    return metrics_per_slot

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config,
              optimizer=None, epoch=None, extra_state=None):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
        # ---- resume 训练态 (本集群新增) ----
        # 注意: 上面存进 lora/ 的是 EMA 权重(适合 eval),此处参数已被 copy_temp_to 换回 RAW 权重,
        # 续训必须用 RAW 权重 + 优化器动量 + EMA 影子,否则等于每次 resume 都把 EMA 叠一遍。
        if optimizer is not None:
            state = {
                "params": [p.detach().to("cpu", copy=True) for p in transformer_trainable_parameters],
                "optimizer": optimizer.state_dict(),
                "ema": ({"decay": ema.decay,
                         "ema_parameters": [q.detach().to("cpu", copy=True) for q in ema.ema_parameters]}
                        if config.train.ema else None),
                "epoch": epoch,
                "global_step": global_step,
            }
            state.update(extra_state or {})
            tmp = os.path.join(save_root, "train_state.pt.tmp")
            torch.save(state, tmp)
            os.replace(tmp, os.path.join(save_root, "train_state.pt"))  # 原子替换:半写的文件不会被当成有效 resume 点
            # 每份训练态 ~300MB,只留最新一份;历史 lora/ 全保留(收敛曲线要用)
            # save_dir 里含 "[...]"(run_name),glob 会把它当字符类 -> 必须 escape 目录部分
            for d in glob.glob(os.path.join(glob.escape(save_dir), "checkpoints", "checkpoint-*")):
                if os.path.abspath(d) != os.path.abspath(save_root):
                    stale = os.path.join(d, "train_state.pt")
                    if os.path.exists(stale):
                        try:
                            os.remove(stale)
                        except OSError:
                            pass

def main(_):

    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,

        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:

        accelerator.init_trackers(
            project_name="flow_grpo",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )

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
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device)
    teachers = _build_teachers_config(config)
    unique_adapter_lora = {}
    unique_adapter_guidance_scale = {}
    slot_adapter_names = []
    slot_adapter_guidance_scales = []

    def _register_adapter(name, lora_path, guidance_scale, slot_label):
        adapter = f"teacher_{name}"
        if adapter in unique_adapter_lora:
            if unique_adapter_lora[adapter] != lora_path:
                raise ValueError(
                    f"Teacher name '{name}' (slot '{slot_label}') "
                    f"references lora_path={lora_path!r} but the same "
                    f"name was previously registered with "
                    f"{unique_adapter_lora[adapter]!r}. Each "
                    f"`teacher_<name>` adapter must map to exactly one "
                    f"LoRA across the whole `config.train.teachers` "
                    f"list."
                )
            if unique_adapter_guidance_scale[adapter] != guidance_scale:
                raise ValueError(
                    f"Teacher name '{name}' (slot '{slot_label}') "
                    f"references guidance_scale={guidance_scale} but the "
                    f"same name was previously registered with "
                    f"guidance_scale="
                    f"{unique_adapter_guidance_scale[adapter]}. Each "
                    f"`teacher_<name>` adapter is one PEFT adapter on the "
                    f"shared backbone, so it must use exactly one CFG "
                    f"scale across the whole `config.train.teachers` "
                    f"list."
                )
        else:
            unique_adapter_lora[adapter] = lora_path
            unique_adapter_guidance_scale[adapter] = guidance_scale
        return adapter
    for tc in teachers:
        adapter_list = [
            _register_adapter(tc["name"], tc["lora_path"], tc["guidance_scale"], tc["name"])
        ]
        gs_list = [tc["guidance_scale"]]
        slot_adapter_names.append(adapter_list)
        slot_adapter_guidance_scales.append(gs_list)
    if (bool(getattr(config.train, "cagrad", False))
            or bool(getattr(config.train, "ste", False))
            or bool(getattr(config.train, "rtp", False))
            or bool(getattr(config.train, "gc_all_teachers", False))):
        # CAGrad/STE: every slot computes ALL teacher targets per state (K=M) so they can be combined per state
        _all_a = [a for sub in slot_adapter_names for a in sub]
        _all_g = [g for sub in slot_adapter_guidance_scales for g in sub]
        slot_adapter_names = [list(_all_a) for _ in slot_adapter_names]
        slot_adapter_guidance_scales = [list(_all_g) for _ in slot_adapter_guidance_scales]
    max_teachers_per_slot = max(len(a) for a in slot_adapter_names)
    if accelerator.is_local_main_process:
        logger.info(
            f"OPD: configured {len(teachers)} teacher slot(s) using "
            f"{len(unique_adapter_lora)} unique teacher adapter(s) "
            f"(K_max={max_teachers_per_slot}):"
        )
        for tc, adapters, gs_list in zip(
            teachers,
            slot_adapter_names,
            slot_adapter_guidance_scales,
        ):
            logger.info(
                f"  slot {tc['name']}: prompts={tc['dataset']}, "
                f"proxy={list(tc['proxy_reward'].keys())}, "
                f"teachers={adapters}, "
                f"guidance_scales={gs_list}"
            )
        for adapter, path in unique_adapter_lora.items():
            logger.info(
                f"  adapter {adapter} <- {path} "
                f"(guidance_scale={unique_adapter_guidance_scale[adapter]})"
            )

    original_reset = AcceleratorState._reset_state
    AcceleratorState._reset_state = lambda *args, **kwargs: None
    proxy_reward_fns_per_teacher = []
    for tc in teachers:
        if tc["proxy_reward"]:
            fn = getattr(flow_grpo.rewards, "multi_score")(
                accelerator.device, tc["proxy_reward"]
            )
            fn = make_no_grad(fn)
        else:

            fn = None
        proxy_reward_fns_per_teacher.append(fn)

    AcceleratorState._reset_state = original_reset
    torch.cuda.empty_cache()

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
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            # is_trainable=True 是必须的:peft 0.10 的 from_pretrained 默认 is_trainable=False,
            # 会把加载进来的 'default' adapter 全部 requires_grad=False;
            # 于是可训练参数列表为空,optimizer 直接报 'empty parameter list'。
            # (get_peft_model 分支没有这个问题,所以只有 lora_path 非空的 run 会踩到。)
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path, is_trainable=True)

            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

        for adapter_name, lora_path in unique_adapter_lora.items():
            pipeline.transformer.load_adapter(lora_path, adapter_name=adapter_name)

        for n, p in pipeline.transformer.named_parameters():
            if "teacher_" in n:
                p.requires_grad = False

        pipeline.transformer.set_adapter("default")

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    def _resolve_dataset_cls(prompt_fn):
        if prompt_fn == "general_ocr":
            return TextPromptDataset
        elif prompt_fn == "geneval":
            return GenevalPromptDataset
        else:
            raise NotImplementedError(
                "Only `general_ocr` and `geneval` prompt_fns are supported."
            )

    teacher_dataset_classes = [
        _resolve_dataset_cls(tc["prompt_fn"]) for tc in teachers
    ]

    train_datasets = [
        cls(tc["dataset"], "train")
        for cls, tc in zip(teacher_dataset_classes, teachers)
    ]
    test_datasets = [
        cls(tc["dataset"], "test")
        for cls, tc in zip(teacher_dataset_classes, teachers)
    ]

    train_samplers = []
    train_dataloaders = []
    for cls, ds in zip(teacher_dataset_classes, train_datasets):
        _sampler_cls = (DistributedGroupedKSampler
                        if bool(getattr(config.train, "onmanifold", False))
                        else DistributedKRepeatSampler)
        sampler = _sampler_cls(
            dataset=ds,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=cls.collate_fn,

        )
        train_samplers.append(sampler)
        train_dataloaders.append(loader)

    test_dataloaders = [
        DataLoader(
            ds,
            batch_size=config.sample.test_batch_size,
            collate_fn=cls.collate_fn,
            shuffle=False,
            num_workers=0,
        )
        for cls, ds in zip(teacher_dataset_classes, test_datasets)
    ]

    if len(teachers) > 1:
        assert config.sample.num_batches_per_epoch % len(teachers) == 0, (
            f"`config.sample.num_batches_per_epoch` "
            f"(={config.sample.num_batches_per_epoch}) must be divisible by "
            f"the number of teachers (={len(teachers)}) so each teacher gets "
            f"the same number of sampling batches per epoch."
        )

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    prepared = accelerator.prepare(
        transformer,
        optimizer,
        *train_dataloaders,
        *test_dataloaders,
    )
    n_t = len(teachers)
    transformer = prepared[0]
    optimizer = prepared[1]
    train_dataloaders = list(prepared[2 : 2 + n_t])
    test_dataloaders = list(prepared[2 + n_t : 2 + 2 * n_t])

    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    diag2_acc = defaultdict(list)   # diag2 探针在**采样阶段**记录,而 info 只在训练阶段才初始化

    train_iters = [iter(loader) for loader in train_dataloaders]

    slot_idx_to_name = [tc["name"] for tc in teachers]

    slot_idx_to_teacher_short_names = [
        [name.replace("teacher_", "", 1) for name in adapters]
        for adapters in slot_adapter_names
    ]

    # --- CA-OPD (conflict-averse / Pareto multi-teacher OPD) state ---
    ca_opd = bool(getattr(config.train, "ca_opd", False))
    ca_opd_tau = float(getattr(config.train, "ca_opd_tau", 0.5))
    ca_opd_ema = float(getattr(config.train, "ca_opd_ema", 0.9))
    ca_opd_regret = bool(getattr(config.train, "ca_opd_regret", False))  # weight reducible gap r_m = ema_m - floor_m instead of absolute KL
    teacher_kl_ema = torch.ones(n_t, device=accelerator.device)  # neutral init -> starts uniform (=mopd), then adapts
    teacher_kl_floor = torch.full((n_t,), float("inf"), device=accelerator.device)  # running-min KL per teacher (attainable floor)
    if ca_opd and accelerator.is_main_process:
        logger.info(f"[CA-OPD] Chebyshev min-max multi-teacher reweighting ON (tau={ca_opd_tau}, ema={ca_opd_ema}, n_teachers={n_t})")

    # --- VIRTUAL teacher (no RL expert). mode="reward" (MVP-1, myopic reward-grad) | "value" (MVP-2, learned-value grad) ---
    vteacher = bool(getattr(config.train, "vteacher", False))
    vteacher_mode = str(getattr(config.train, "vteacher_mode", "reward"))
    vteacher_alpha = float(getattr(config.train, "vteacher_alpha", 1.0))
    vteacher_nsteps = int(getattr(config.train, "vteacher_nsteps", 2))
    vteacher_sign = float(getattr(config.train, "vteacher_sign", 1.0))
    value_lr = float(getattr(config.train, "value_lr", 1e-4))
    value_steps = int(getattr(config.train, "value_steps", 16))
    value_bs = int(getattr(config.train, "value_bs", 64))
    vteacher_scorer = None
    value_net = None
    value_optimizer = None
    # Product-of-Experts OPD: distill to v_base + lambda*(v_teacher - v_base). lambda=1 -> uniform mopd,
    # lambda=M -> product-of-experts (sum-reward) optimum. Targets the JOINT multi-reward distribution
    # p* ~ p_ref*exp(sum_m R_m/beta) instead of uniform's average-reward one.
    poe = bool(getattr(config.train, "poe", False))
    poe_lambda = getattr(config.train, "poe_lambda", 1.0)  # float, or per-teacher list indexed by teacher_idx (weighted-POE)
    # CAGrad/MGDA: resolve inter-teacher conflict by an output-space min-norm weighted combination of the
    # M teacher targets (targets stay at REAL teacher velocities => no amplification => cannot reward-hack).
    cagrad = bool(getattr(config.train, "cagrad", False))
    cagrad_c = float(getattr(config.train, "cagrad_c", 0.5))  # 0 -> uniform mopd, 1 -> pure MGDA min-norm
    cagrad_mode = str(getattr(config.train, "cagrad_mode", "mgda"))  # "mgda" | "pcgrad"
    # Spatio-Temporal Expert OPD: route each teacher's residual to the (timestep, token) locations where that
    # teacher actually carries signal. Uniform mean over H*W tokens dilutes a spatially concentrated teacher
    # (OCR cares about the text region only) by ~1/N; routing restores its gradient there.
    ste = bool(getattr(config.train, "ste", False))
    ste_mode = str(getattr(config.train, "ste_mode", "spatial"))   # "spatial" | "temporal" | "both"
    ste_tau = float(getattr(config.train, "ste_tau", 1.0))          # ->inf: uniform average (=mopd); ->0: hard routing
    ste_norm = str(getattr(config.train, "ste_norm", "zscore"))     # "zscore" | "raw"
    # Multi-Step / Horizon OPD: 单步 KL 只对齐"这一步的速度",是短视的;真正要对齐的是 flow map。
    # 让 teacher 从 student 当前状态自己往前滚 H 步,再把这 H 步位移按 sigma 比例折算回当前这一步:
    #   mu_H(x_j) = x_j + (s_{j+1}-s_j)/(s_{j+H}-s_j) * (x^T_{j+H} - x_j)
    # H=1 时该式恒等于原始 teacher 目标(可直接与基线对比)。
    opd_horizon = int(getattr(config.train, "horizon", 1))
    if opd_horizon > 1 and accelerator.is_main_process:
        logger.info(f"[HORIZON] teacher {opd_horizon}-step flow-map matching")
    # Continual Capability OPD: round-robin 轮到别的 teacher 时,上一个能力会被挤掉。
    # 维护一个每 retain_every 个 epoch 刷新一次的 student 慢照快(自蒸馏锚),
    # 在 loss 里加 retain_w * ||mu_S - mu_snapshot||^2,把"已经学会的东西"钉住。
    # retain_w=0 时完全不产生任何开销,基线行为逐位不变。
    retain_w = float(getattr(config.train, "retain_w", 0.0))
    retain_every = int(getattr(config.train, "retain_every", 100))
    retain_snapshot = None
    retain_snapshot_step = -1

    # ---- G_construct 探针 (对照 DiffusionOPSD, arXiv 2608.24646) ----
    # 目的: 用**正确的判据**重做 teacher-free 诊断。我们此前测的是
    #   cos(reward 梯度, teacher delta) ≈ 0.003 ≈ 随机 -> 据此判定 teacher-free 不成立。
    # 但 OPSD 在他们的低噪声查询点上测 NFT 终点方向的同一个量,得到 -0.0002,同样接近零 ——
    # 说明"与 teacher/终点方向正交"是终点式目标的性质,不能用来否定 reward 梯度。
    # 正确判据是 fixed-suffix 构造增益:
    #   G_construct = F_q(y_+) - F_q(y_0),  F_q(y) = R(decode(把 y 插进查询点后用冻结策略跑完))
    # 本探针在单个低噪声查询点上同时测四个方向,共享同一后缀,因此可直接相减:
    #   noop / reward 梯度 / 半径匹配随机 / **teacher 方向**(后者是 OPSD 没做的对照)
    # ================= RTC: Realizable Target Construction =================
    # 动机(来自 DiffusionOPSD, arXiv 2608.24646 自己的量化结果):
    #   G_realized = G_construct - G_fit,  实测 G_construct=+0.03511 而 G_realized=+0.00046
    #   -> 构造出的增益有 ~98.7% 在有限拟合阶段丢失,且他们把这列为开放问题。
    # 诊断: 目标 y_+ = y_0 + h*ghat 是在**图像空间**构造的,不管网络在该查询点能否产生它。
    #   rank-32 LoRA 在 y 空间只能触及 range(J) (J = d y_theta / d theta),
    #   ghat 落在 range(J)^perp 的分量原理上无法实现,只会让更新旋转(他们实测 drift=0.371)。
    # 方法: 把方向换成 Gauss-Newton 预条件方向  d = J J^T g,  它天然落在可实现子空间内。
    # 一阶保证(Cauchy-Schwarz): 设 A=JJ^T (PSD),单步实现增益 ∝ <g, A·dir>,
    #   dir=ghat  -> g^T A g / ||g||
    #   dir=Ag/|| -> g^T A^2 g / ||Ag|| = ||Ag||
    #   而 ||Ag||·||g|| >= <g, Ag>  ==>  预条件方向的实现增益**严格不小于**原方向,
    #   等号仅当 g 是 A 的特征向量。提升倍数 = ||Ag||·||g|| / <g,Ag>,即 g 与模型响应算子的失配度。
    # 计算: A g 不显式构造。u = J^T g 一次反向; d = J u 用有限差分一次扰动前向。
    rtc = bool(getattr(config.train, "rtc", False))
    rtc_sigma = float(getattr(config.train, "rtc_sigma", 0.278))   # 低噪声查询点(OPSD 的 SD3.5-M 取值)
    rtc_rho = float(getattr(config.train, "rtc_rho", 0.10))        # 信赖域 = rho*||y0||
    rtc_nascent = int(getattr(config.train, "rtc_nascent", 2))
    rtc_precond = str(getattr(config.train, "rtc_precond", "gn"))  # "gn"(方法) | "raw"(消融=OPSD) | "rand"(归因对照)
    rtc_eps = float(getattr(config.train, "rtc_eps", 1e-3))        # JVP 有限差分步长(相对参数范数)
    rtc_beta = float(getattr(config.train, "rtc_beta", 0.5))       # 双分支混合系数
    rtc_dual = bool(getattr(config.train, "rtc_dual", True))       # 负分支(排斥)
    rtc_margin = float(getattr(config.train, "rtc_margin", 0.0))   # 采纳门限: R(y+) >= R(y0)+margin
    if rtc and accelerator.is_main_process:
        logger.info(f"[RTC] precond={rtc_precond} sigma*={rtc_sigma} rho={rtc_rho} "
                    f"n_ascent={rtc_nascent} beta={rtc_beta} dual={rtc_dual} margin={rtc_margin}")

    gc_probe = bool(getattr(config.train, "gc_probe", False))
    gc_sigma = float(getattr(config.train, "gc_sigma", 0.278))   # OPSD 的 SD3.5-M 取值
    gc_rho = float(getattr(config.train, "gc_rho", 0.10))        # 信赖域 = rho*||y0||
    gc_nascent = int(getattr(config.train, "gc_nascent", 2))     # 梯度上升步数
    if gc_probe and accelerator.is_main_process:
        logger.info(f"[GCPROBE] sigma*={gc_sigma} rho={gc_rho} n_ascent={gc_nascent} "
                    f"(fixed-suffix G_construct: noop / grad / rand / teacher)")
    if retain_w > 0.0 and accelerator.is_main_process:
        logger.info(f"[RETAIN] self-distillation anchor w={retain_w} refresh_every={retain_every} epochs")

    def _swap_params(params, new_vals):
        """就地把 params 换成 new_vals,返回被换下来的旧值(用于事后还原)。

        只在 no_grad 的采样阶段调用,不触碰计算图;LoRA r32 的参数量很小,克隆开销可忽略。
        """
        old = [p.detach().clone() for p in params]
        with torch.no_grad():
            for _p, _v in zip(params, new_vals):
                _p.data.copy_(_v)
        return old
    if ((vteacher and vteacher_mode == "reward")
            or bool(getattr(config.train, "itopd", False))
            or bool(getattr(config.train, "diag2_alpha", False))
            or bool(getattr(config.train, "gc_probe", False))
            or bool(getattr(config.train, "rtc", False))):   # diag2 / G_construct 探针 / RTC 都要可微打分器
        # itopd_reward_fn 选择 IT-OPD 用哪个可微 reward。用 "aesthetic" 可以做**同 reward 对照**:
        # Aes teacher 是用 aesthetic reward 训出来的,拿 aesthetic 的梯度去比对它的方向才是 apples-to-apples;
        # 用 pickscore 比 Aes teacher 会混入目标失配,低余弦无法归因。
        if str(getattr(config.train, "itopd_reward_fn", "pickscore")) == "aesthetic":
            from flow_grpo.aesthetic_scorer import DiffAestheticScorer
            vteacher_scorer = DiffAestheticScorer(device=accelerator.device, dtype=torch.float32)
        else:
            vteacher_scorer = PickScoreScorer(device=accelerator.device, dtype=torch.float32)
    # 启动即校验: 上面这个条件漏掉任一分支,都会在跑到采样阶段才以含糊的
    # 'NoneType has no attribute score_from_images' 崩掉(diag2 和 rtc 各踩过一次)。
    # 必须放在创建**之后**,否则恒为 None、恒触发(这本身也踩过一次)。
    if (bool(getattr(config.train, "rtc", False))
            or bool(getattr(config.train, "gc_probe", False))) and vteacher_scorer is None:
        raise RuntimeError(
            "rtc/gc_probe 需要可微打分器,但 vteacher_scorer 为 None —— "
            "请把该 flag 加进上面 vteacher_scorer 的创建条件。")
    if vteacher and vteacher_mode == "value":
        from flow_grpo.value_net import ValueNet
        value_net = ValueNet().to(accelerator.device)  # per-rank critic (policy grad is DDP-synced downstream)
    if vteacher and accelerator.is_main_process:
        logger.info(
            f"[VTEACHER] ON mode={vteacher_mode} alpha={vteacher_alpha} nsteps={vteacher_nsteps} "
            f"sign={vteacher_sign} -- NO RL-expert velocity signal used"
        )

    # --- Diagnostic 2: teacher correction 的 alpha 探针(纯测量,不改训练目标) ---
    # 问题:完整 teacher target 是否局部过量/超出 student 当前可实现范围?
    # 做法:在**采样阶段已有的 student 轨迹**上,于步 j 用 mu_alpha = mu_S + alpha*(mu_T - mu_S) 替换该步,
    #      其余步保持 student,走完后打分。判据:R(alpha) 峰值在 alpha<1 -> 完整 teacher 步过量。
    # ⚠️ 放在训练脚本内而非独立脚本:独立脚本需自建 pipeline/adapter/dtype 环境,与训练侧假设频繁不一致
    #    (实测踩了 7 次:argv 顺序、PeftModel 包装、dtype 布局x3、scheduler 属性、prepare_latents 用 vae.dtype)。
    #    这里天然继承训练已建好的 pipeline、adapter、dtype 与 solver 设置。
    diag2_alpha = bool(getattr(config.train, "diag2_alpha", False))
    diag2_alphas = list(getattr(config.train, "diag2_alphas", [0.0, 0.5, 1.0, 1.5]))
    diag2_js = list(getattr(config.train, "diag2_js", [8]))
    if diag2_alpha and accelerator.is_main_process:
        logger.info(f"[DIAG2] alpha 探针 ON  alphas={diag2_alphas} steps={diag2_js} (纯测量,不改训练)")

    # --- On-Manifold Search OPD (teacher-free 的"唯一还说得通的形式") ---
    # 诊断结论:teacher(rank-32 LoRA)的速度增量落在模型可实现的低维子空间;像素空间 reward 梯度
    # 与之基本正交(cos≈0.002,只解释 0.0004% 的能量),Jacobian 投影在严格统计下也无提升。
    # 所以候选**必须是模型自己生成的东西**:同 prompt 的 K 条真实 rollout 天然在流形上。
    # virtual teacher = 组内 reward 加权的 next-latent 均值(相对组均值的改进方向)。
    onmanifold = bool(getattr(config.train, "onmanifold", False))
    onm_tau = float(getattr(config.train, "onm_tau", 1.0))      # 组内 reward z-score 后的 softmax 温度
    onm_step = float(getattr(config.train, "onm_step", 0.10))   # 信赖域: ‖Δμ‖ ≤ 该比例 × ‖μ_S‖
    if onmanifold and accelerator.is_main_process:
        logger.info(f"[ON-MANIFOLD] K={config.sample.num_image_per_prompt} tau={onm_tau} step={onm_step} "
                    f"-- 候选=同 prompt 的真实 rollout(在流形上),不用任何 latent 扰动/梯度")

    # --- Realizability-Aware OPD: student 锚定的 teacher 目标插值 ---
    # target = mu_S + alpha*(mu_T - mu_S)。alpha=1 即标准 OPD;alpha<1 表示"只走 teacher 方向的一部分";
    # alpha>1 是 STEP 式外推。核心问题:完整 teacher target 是否 over-demanding / 超出 student 当前可实现范围?
    # ⚠️ 与已有的 poe 不同:poe 锚在 **base**(v_base + lam*(v_T-v_base)),这里锚在 **student**。
    # 训练早期两者近似,后期 student 远离 base 后含义分岔 —— 问"student 还能沿 teacher 方向走多远"必须用 student 锚。
    opd_alpha = getattr(config.train, "opd_alpha", None)   # None = 关闭(基线行为不变)
    if opd_alpha is not None and accelerator.is_main_process:
        logger.info(f"[REALIZ] student-anchored teacher interpolation alpha={opd_alpha}")

    # --- IT-OPD: Implicit-Teacher OPD (no explicit RL expert; teacher solved locally) ---
    # KL-regularized local policy improvement:  q* ∝ p_θ(a|s)·exp(Q(s,a)/τ)
    # Gaussian ⇒ μ* = μ_θ + Δμ, distilled by the SAME closed-form velocity regression as OPD.
    # 与仓库里已有的 `vteacher`(MVP-1) 的关键区别: MVP-1 把 target 锚在 v_base(base 模型速度),
    # 即 "先拉回 base 再朝 reward 推一点" —— 这不是 on-policy improvement,实测结果 ≈ base+CFG
    # (vt_* pickscore 0.858-0.861 vs base_gs4.5 0.863,零增益)。IT-OPD 锚在 student 自己的 μ_θ。
    itopd = bool(getattr(config.train, "itopd", False))
    itopd_mode = str(getattr(config.train, "itopd_mode", "search"))   # "search"(黑盒) | "grad"(可微)
    itopd_k = int(getattr(config.train, "itopd_k", 4))                # search: 候选数 K
    itopd_tau = float(getattr(config.train, "itopd_tau", 0.05))       # search: Boltzmann 温度
    # ⚠️ 理论偏差(必须在论文里写明): 推导假设随机策略 N(μ,Σ),但 OPD 最佳设置是 ODE
    # (config.sample.noise_level=0 ⇒ Σ=0),KL 正则的 improvement 在 Σ=0 时退化。
    # 因此这里把 improvement operator 定义在一个**辅助探索核**上,尺度以 ‖μ_θ‖ 归一
    # (在 65k 个 latent 元素上做单位归一会让扰动 ~0.4%/元素 -> student 塌回 base,见 MVP-1 注释)。
    itopd_explore = float(getattr(config.train, "itopd_explore", 0.05))  # 候选散布 = 该比例 × ‖μ_θ‖
    itopd_step = float(getattr(config.train, "itopd_step", 0.05))        # 信赖域: ‖Δμ‖ ≤ 该比例 × ‖μ_θ‖
    itopd_nsteps = int(getattr(config.train, "itopd_nsteps", 2))         # 只在最低噪声的若干步施加
    itopd_gate = bool(getattr(config.train, "itopd_gate", True))         # advantage 门控: 无改进则不学
    # 纯诊断模式:在**每个** timestep 上测量"真 teacher 增量"与"局部 reward 梯度方向"的关系,
    # 但**不修改训练目标**(训练照常按真 teacher 走)。用来回答两个问题:
    #   (a) teacher 到底在哪些噪声层发力? (teacher_delta_rel_j*)
    #   (b) 各层的对齐度分别多少? 是否只是我选错了 timestep? (itopd_cos_j*)
    itopd_diag_only = bool(getattr(config.train, "itopd_diag_only", False))
    if itopd and accelerator.is_main_process:
        logger.info(f"[IT-OPD] ON mode={itopd_mode} K={itopd_k} tau={itopd_tau} explore={itopd_explore} "
                    f"step={itopd_step} nsteps={itopd_nsteps} gate={itopd_gate} -- NO RL-expert velocity used")

    # --- extra OPD-improvement toggles (all default OFF -> baseline mopd unchanged) ---
    opd_huber = bool(getattr(config.train, "opd_huber", False))      # Idea: robust/trust-region residual
    huber_delta = float(getattr(config.train, "huber_delta", 1.0))
    snr_weight = bool(getattr(config.train, "snr_weight", False))    # Idea: Min-SNR-gamma per-step weighting
    snr_gamma = float(getattr(config.train, "snr_gamma", 5.0))
    anneal_noise = bool(getattr(config.train, "anneal_noise", False))  # Idea: SDE->ODE noise annealing
    noise_start = float(getattr(config.train, "noise_start", 0.7))
    noise_end = float(getattr(config.train, "noise_end", 0.0))
    anneal_epochs = int(getattr(config.train, "anneal_epochs", 400))
    max_epochs = int(getattr(config, "num_epochs", 10**9))
    if accelerator.is_main_process and (opd_huber or snr_weight or anneal_noise):
        logger.info(f"[OPD-EXT] huber={opd_huber}(d={huber_delta}) snr={snr_weight}(g={snr_gamma}) anneal={anneal_noise}({noise_start}->{noise_end}@{anneal_epochs}) max_epochs={max_epochs}")

    # ================= RESUME (本集群新增;上游 train_sd3_opd.py 只存不读) =================
    # OPD_RESUME=auto  -> 扫 config.save_dir 找 global_step 最大的 train_state.pt
    # OPD_RESUME=<dir> -> 指定 checkpoint-N 目录 (或直接给 train_state.pt 路径)
    # 未设置 / none     -> 从头训练 (与上游行为完全一致)
    _resume_req = os.environ.get("OPD_RESUME", "").strip()
    if _resume_req and _resume_req.lower() != "none":
        _rp = ""
        if _resume_req.lower() == "auto":
            _cands = glob.glob(os.path.join(glob.escape(config.save_dir), "checkpoints", "checkpoint-*", "train_state.pt"))
            if _cands:
                _rp = max(_cands, key=lambda q: int(_re.search(r"checkpoint-(\d+)", q).group(1)))
        elif os.path.isdir(_resume_req):
            _rp = os.path.join(_resume_req, "train_state.pt")
        else:
            _rp = _resume_req
        if _rp and os.path.exists(_rp):
            # 每个 rank 都从共享盘读同一份(存的是 main process 的参数),保证 8 个 rank 起点一致
            _st = torch.load(_rp, map_location="cpu", weights_only=False)
            with torch.no_grad():
                for _p, _q in zip(transformer_trainable_parameters, _st["params"], strict=True):
                    _p.data.copy_(_q.to(device=_p.device, dtype=_p.dtype))
            optimizer.load_state_dict(_st["optimizer"])
            if config.train.ema and _st.get("ema"):
                ema.load_state_dict(_st["ema"])
                ema.to(accelerator.device)
            global_step = int(_st["global_step"])
            epoch = int(_st["epoch"]) + 1
            if _st.get("teacher_kl_ema") is not None:
                teacher_kl_ema = _st["teacher_kl_ema"].to(accelerator.device)
            if _st.get("teacher_kl_floor") is not None:
                teacher_kl_floor = _st["teacher_kl_floor"].to(accelerator.device)
            if accelerator.is_main_process:
                logger.info(f"[resume] restored from {_rp} -> epoch={epoch} global_step={global_step}")
                if vteacher and vteacher_mode == "value":
                    logger.warning("[resume] value-critic (sopd_vvalue_*) state is NOT checkpointed; "
                                   "the critic restarts from scratch -- avoid resume for that arm.")
        elif accelerator.is_main_process:
            logger.info(f"[resume] OPD_RESUME={_resume_req} but no train_state.pt found -> training from scratch")
    accelerator.wait_for_everyone()

    while True:

        if epoch >= max_epochs:
            if accelerator.is_main_process:
                logger.info(f"[stop] reached num_epochs={max_epochs} at epoch {epoch}; finishing.")
            break
        if anneal_noise:
            frac = min(1.0, epoch / max(1, anneal_epochs))
            config.sample.noise_level = noise_start + (noise_end - noise_start) * frac
            if accelerator.is_main_process:
                logger.info(f"[anneal] epoch {epoch} noise_level={config.sample.noise_level:.4f}")

        if retain_w > 0.0 and (retain_snapshot is None or epoch - retain_snapshot_step >= retain_every):
            # 注意刷新点在采样之前:本 epoch 用的锚就是"retain_every 个 epoch 之前的自己"。
            retain_snapshot = [p.detach().clone() for p in transformer_trainable_parameters]
            retain_snapshot_step = epoch
            if accelerator.is_main_process:
                logger.info(f"[RETAIN] snapshot refreshed at epoch {epoch}")

        pipeline.transformer.eval()
        if config.eval_freq > 0 and epoch % config.eval_freq == 0:
            eval_metrics_per_slot = eval(
                pipeline,
                test_dataloaders,
                teachers,
                text_encoders,
                tokenizers,
                config,
                accelerator,
                global_step,
                proxy_reward_fns_per_teacher,
                executor,
                autocast,
                num_train_timesteps,
                ema,
                transformer_trainable_parameters,
            )

        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):

            teacher_idx = i % len(teachers)
            slot_adapters = slot_adapter_names[teacher_idx]
            slot_guidance_scales = slot_adapter_guidance_scales[teacher_idx]
            train_samplers[teacher_idx].set_epoch(
                epoch * config.sample.num_batches_per_epoch + i
            )
            prompts, prompt_metadata = next(train_iters[teacher_idx])

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():

                    sample_solver, sample_deterministic = _resolve_solver(config)
                    sample_kw = dict(
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        solver=sample_solver,
                        deterministic=sample_deterministic,
                    )
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        **sample_kw,
                    )

            latents = torch.stack(
                latents, dim=1
            )
            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )
            if timesteps.shape[0] != latents.shape[0]:
                timesteps = timesteps[: latents.shape[0]]

            sample_neg_prompt_embeds_b = sample_neg_prompt_embeds[: prompt_embeds.shape[0]]
            sample_neg_pooled_prompt_embeds_b = sample_neg_pooled_prompt_embeds[
                : pooled_prompt_embeds.shape[0]
            ]

            slot_teacher_means_per_teacher = []
            with autocast():
                with torch.no_grad():
                    for k_idx, adapter_name in enumerate(slot_adapters):
                        unwrap_model(transformer, accelerator).set_adapter(adapter_name)
                        with contextlib.nullcontext():

                            teacher_gs = slot_guidance_scales[k_idx]
                            teacher_prev_sample_mean_list = []
                            for j in range(config.sample.num_steps):
                                teacher_prev_mean_j = compute_teacher_step_mean(
                                        transformer=transformer,
                                        pipeline=pipeline,
                                        latents_at_j=latents[:, j],
                                        next_latent_at_j=latents[:, j + 1],
                                        timestep_at_j=timesteps[:, j],
                                        cond_embeds=prompt_embeds,
                                        cond_pooled=pooled_prompt_embeds,
                                        neg_embeds=sample_neg_prompt_embeds_b,
                                        neg_pooled=sample_neg_pooled_prompt_embeds_b,
                                        guidance_scale=teacher_gs,
                                        noise_level=config.sample.noise_level,
                                        solver=sample_solver,
                                        deterministic=sample_deterministic,
                                    )
                                if opd_horizon > 1:
                                    # 从 teacher 自己的第一步结果继续往前滚,时间步随之推进。
                                    # 末尾不足 H 步时 _Hs 自动收缩;_Hs==1 时下面的系数恰为 1,
                                    # 目标退化为标准单步 teacher 目标(恒等,无需特判)。
                                    _Hs = min(opd_horizon, config.sample.num_steps - j)
                                    _x = teacher_prev_mean_j
                                    for _m in range(1, _Hs):
                                        _x = compute_teacher_step_mean(
                                            transformer=transformer,
                                            pipeline=pipeline,
                                            latents_at_j=_x,
                                            # noise_level=0(mopd 就是 0)时 prev_sample_mean 与该参数无关,
                                            # 它只参与 logprob 计算,这里丢弃。
                                            next_latent_at_j=_x,
                                            timestep_at_j=timesteps[:, j + _m],
                                            cond_embeds=prompt_embeds,
                                            cond_pooled=pooled_prompt_embeds,
                                            neg_embeds=sample_neg_prompt_embeds_b,
                                            neg_pooled=sample_neg_pooled_prompt_embeds_b,
                                            guidance_scale=teacher_gs,
                                            noise_level=config.sample.noise_level,
                                            solver=sample_solver,
                                            deterministic=sample_deterministic,
                                        )
                                    _sg = pipeline.scheduler.sigmas
                                    _x0 = latents[:, j].float()
                                    _c = ((_sg[j + 1] - _sg[j]) / (_sg[j + _Hs] - _sg[j])).to(_x0.device)
                                    teacher_prev_mean_j = _x0 + _c * (_x.float() - _x0)
                                teacher_prev_sample_mean_list.append(
                                    teacher_prev_mean_j
                                )
                            slot_teacher_means_per_teacher.append(
                                torch.stack(
                                    teacher_prev_sample_mean_list, dim=1
                                ).float()
                            )

            unwrap_model(transformer, accelerator).set_adapter("default")

            teacher_prev_sample_means_slot = torch.stack(
                slot_teacher_means_per_teacher, dim=2
            )
            slot_K = teacher_prev_sample_means_slot.shape[2]

            if slot_K < max_teachers_per_slot:
                B_dim, T_dim, _, C_dim, H_dim, W_dim = (
                    teacher_prev_sample_means_slot.shape
                )
                pad = torch.zeros(
                    B_dim,
                    T_dim,
                    max_teachers_per_slot - slot_K,
                    C_dim,
                    H_dim,
                    W_dim,
                    dtype=teacher_prev_sample_means_slot.dtype,
                    device=teacher_prev_sample_means_slot.device,
                )
                teacher_prev_sample_means = torch.cat(
                    [teacher_prev_sample_means_slot, pad], dim=2
                )
            else:
                teacher_prev_sample_means = teacher_prev_sample_means_slot

            # ===== Diagnostic 2: alpha 探针(纯测量,不影响训练目标) =====
            # 在已有的 student 轨迹上,于步 j 用 mu_alpha = mu_S + alpha*(mu_T - mu_S) 替换该步,
            # 其余步仍用 student 走完,再打分。R(alpha) 峰值若在 alpha<1,说明完整 teacher 步局部过量。
            if diag2_alpha and accelerator.is_main_process:
                with torch.no_grad():
                    _ts_d = timesteps
                    _lat_d = latents.float()                       # [B, N+1, C, H, W]
                    _mt_all = teacher_prev_sample_means[:, :, 0].float()   # 取第 0 个 teacher
                    for _j in diag2_js:
                        if _j >= config.sample.num_steps - 1:
                            continue
                        _muS = _lat_d[:, _j + 1]                   # ODE 下 student 均值即实际下一状态
                        _muT = _mt_all[:, _j]
                        for _al in diag2_alphas:
                            _z = (_muS + _al * (_muT - _muS))
                            for _jj in range(_j + 1, config.sample.num_steps):
                                _tt = _ts_d[:, _jj]
                                _npd = transformer(
                                    hidden_states=torch.cat([_z] * 2),
                                    timestep=torch.cat([_tt] * 2),
                                    encoder_hidden_states=torch.cat(
                                        [sample_neg_prompt_embeds_b, prompt_embeds]),
                                    pooled_projections=torch.cat(
                                        [sample_neg_pooled_prompt_embeds_b, pooled_prompt_embeds]),
                                    return_dict=False)[0]
                                _nu, _nt = _npd.chunk(2)
                                _vv = _nu + config.sample.guidance_scale * (_nt - _nu)
                                _, _, _z, _ = sde_step_with_logprob(
                                    pipeline.scheduler, _vv.float(), _tt, _z.float(),
                                    noise_level=config.sample.noise_level)
                            _dec = (_z.float() / pipeline.vae.config.scaling_factor) \
                                   + pipeline.vae.config.shift_factor
                            _im = pipeline.vae.decode(
                                _dec.to(pipeline.vae.dtype), return_dict=False)[0].float()
                            _im01 = (_im / 2 + 0.5).clamp(0, 1)
                            _sc = vteacher_scorer.score_from_images(
                                vteacher_scorer.encode_text(prompts), _im01)
                            diag2_acc[f"diag2_R_j{_j}_a{_al}"].append(float(_sc.mean()))

            # ===== RTC: 在低噪声查询点构造"可实现"目标 =====
            rtc_pack = None
            if rtc:
                _sg = pipeline.scheduler.sigmas
                _jq = min([(abs(float(_sg[_j]) - rtc_sigma), _j) for _j in range(config.sample.num_steps)])[1]
                _sq = float(_sg[_jq])
                _zq = latents[:, _jq].float()
                _tq = timesteps[:, _jq]
                _params = transformer_trainable_parameters

                def _fwd_v(_z, _t):
                    _pd = transformer(
                        hidden_states=torch.cat([_z] * 2), timestep=torch.cat([_t] * 2),
                        encoder_hidden_states=torch.cat([sample_neg_prompt_embeds_b, prompt_embeds]),
                        pooled_projections=torch.cat([sample_neg_pooled_prompt_embeds_b, pooled_prompt_embeds]),
                        return_dict=False)[0]
                    _u, _c = _pd.chunk(2)
                    return (_u + config.sample.guidance_scale * (_c - _u)).float()

                def _fwd_v_sub(_z, _t, _sl):
                    """与 _fwd_v 等价,但只算 batch 的一个切片。
                    双反向(精确 JVP)要保留整张前向图,且 math 后端会 materialize 完整注意力矩阵,
                    全 batch 下实测在 140GB H200 上 OOM(136.6 GiB 已分配)。按样本分块可把峰值降到约 1/B。"""
                    _pd = transformer(
                        hidden_states=torch.cat([_z] * 2), timestep=torch.cat([_t] * 2),
                        encoder_hidden_states=torch.cat([sample_neg_prompt_embeds_b[_sl], prompt_embeds[_sl]]),
                        pooled_projections=torch.cat([sample_neg_pooled_prompt_embeds_b[_sl], pooled_prompt_embeds[_sl]]),
                        return_dict=False)[0]
                    _uu, _cc = _pd.chunk(2)
                    return (_uu + config.sample.guidance_scale * (_cc - _uu)).float()

                def _reward_of_y(_y):
                    _dc = (_y / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                    _im = pipeline.vae.decode(_dc.to(pipeline.vae.dtype), return_dict=False)[0].float()
                    return vteacher_scorer.score_from_images(
                        vteacher_scorer.encode_text(prompts), (_im / 2 + 0.5).clamp(0, 1))

                with autocast():
                    with torch.no_grad():
                        _v0 = _fwd_v(_zq, _tq)
                    _y0 = (_zq - _sq * _v0).detach()          # 锚点: 干净图预测
                    _shp = (-1, *([1] * (_y0.ndim - 1)))
                    _budget = (rtc_rho * _y0.flatten(1).norm(dim=1)).view(_shp)

                    # ---- 方向 ----
                    with torch.enable_grad():
                        _a = _y0.clone().requires_grad_(True)
                        _g = torch.autograd.grad(_reward_of_y(_a).sum(), _a)[0].detach()

                    if rtc_precond == "rand":
                        _dir = torch.randn_like(_y0)
                    elif rtc_precond == "raw":
                        _dir = _g                                   # 消融: 等价 OPSD 的原始 reward 梯度
                    elif rtc_precond == "ksteps":
                        # RTC-K: 可实现目标 = 模型沿 reward 梯度真实走 K 步后的输出。
                        # 相比有限差分估 JJ^T g 的三个决定性优势:
                        #  1) 不做"两个相近量相减",位移由步长直接定标 -> 不受 bf16 输出量化影响
                        #     (实测 bf16 前向输出分辨率 ~4e-3,0.1% 扰动被完全舍入);
                        #  2) 目标是真实的模型输出,天然落在可达集内,不依赖线性化假设;
                        #  3) 捕捉的是真实非线性可达轨迹,比一阶的 JJ^T g 更强。
                        _K = int(getattr(config.train, "rtc_ksteps", 4))
                        _klr = float(getattr(config.train, "rtc_klr", 0.02))
                        _orig = [_p.detach().clone() for _p in _params]
                        _asc_log = []
                        _asc_ys = []      # 逐步保存迭代点,便于按实测 reward 取最优
                        try:
                            for _k in range(_K):
                                with torch.enable_grad():
                                    _vk = _fwd_v(_zq, _tq)
                                    _yk = _zq - _sq * _vk
                                _ak = _yk.detach().requires_grad_(True)
                                with torch.enable_grad():
                                    _rk = _reward_of_y(_ak)
                                    _gk = torch.autograd.grad(_rk.sum(), _ak)[0].detach()
                                _asc_log.append(float(_rk.mean()))   # 必须验证上升真的在上升
                                _asc_ys.append((_yk.detach(), _rk.detach()))
                                with torch.enable_grad():
                                    _uk = torch.autograd.grad((_yk * _gk).sum(), _params,
                                                              retain_graph=False, allow_unused=True)
                                _uk = [torch.zeros_like(_p) if _x is None else _x.detach()
                                       for _p, _x in zip(_params, _uk)]
                                _ukn = torch.sqrt(sum((_x.double() ** 2).sum() for _x in _uk)).clamp(min=1e-12)
                                _pn2 = torch.sqrt(sum((_p.double() ** 2).sum() for _p in _params)).clamp(min=1e-12)
                                _step = float(_klr * (_pn2 / _ukn))     # 每步参数相对位移 = _klr
                                # 机制直测(默认关闭): 累计 K 步上升的总参数位移 dtheta。
                                # 若 MSE 梯度 = -J^T J dtheta,则 cos(梯度, dtheta) 应显著 <1
                                # (J^T J 病态 -> 方向偏离)。这是"y 空间可实现 != 拟合能到达"的直接证据。
                                if bool(getattr(config.train, "rtc_measure_dtheta", False)):
                                    if _k == 0:
                                        globals()["_rtc_dtheta"] = [torch.zeros_like(_p) for _p in _params]
                                    with torch.no_grad():
                                        for _a2, _x2 in zip(globals()["_rtc_dtheta"], _uk):
                                            _a2.add_(_x2, alpha=_step)
                                with torch.no_grad():
                                    for _p, _x in zip(_params, _uk):
                                        _p.data.add_(_x, alpha=_step)   # reward 上升
                            with torch.no_grad():
                                _yK = _zq - _sq * _fwd_v(_zq, _tq)
                        finally:
                            _swap_params(_params, _orig)               # 必须还原,否则污染训练权重
                        with torch.no_grad():
                            _rK = _reward_of_y(_yK)
                            _asc_log.append(float(_rK.mean()))
                            _asc_ys.append((_yK.detach(), _rK.detach()))
                        # 实测显示上升在第 2 步左右达峰、之后越过极值反而变差(K=4 时 y_K 已低于 y_0),
                        # 固定 K 是错的。改为**按实测 reward 逐样本取最优迭代点** ——
                        # 自适应、无需调 K,且与 OPSD"用实测而非一阶预测做接受判定"的哲学一致。
                        if bool(getattr(config.train, "rtc_kbest", True)):
                            with torch.no_grad():
                                _rs = torch.stack([_r for _, _r in _asc_ys], dim=0)      # [K+1, B]
                                _ys = torch.stack([_yv for _yv, _ in _asc_ys], dim=0)    # [K+1, B, ...]
                                _bi = _rs.argmax(dim=0)                                  # [B] 逐样本最优步
                                _yK = _ys[_bi, torch.arange(_ys.shape[1], device=_ys.device)]
                                diag2_acc["rtc_kbest_idx"].append(float(_bi.float().mean()))
                                diag2_acc["rtc_kbest_gain"].append(
                                    float((_rs.max(dim=0).values - _rs[0]).mean()))
                        _dir = (_yK - _y0).detach()
                        # 信赖域上限(可关): 早先基于 n=1 认为 rho=0.10 处 dR 已翻负,故取 min。
                        # 但 curve_n12(n=8,带误差棒)实测该半径处 dR=+0.0355 且随半径单调上升,
                        # 沿最优迭代点方向外推到 0.10 比停在 |d_best| 多约 60% 增益
                        # ⇒ **上限可能是次优的**,改为可配置(rtc_kcap),由 smoke 实测判定。
                        if bool(getattr(config.train, "rtc_kcap", True)):
                            with torch.no_grad():
                                _dnk = _dir.flatten(1).norm(dim=1).view(_shp)
                                _budget = torch.minimum(_budget, _dnk)
                        for _i2, _rv in enumerate(_asc_log):
                            diag2_acc[f"rtc_asc{_i2}"].append(_rv)
                        if accelerator.is_main_process and not globals().get("_rtc_k_dbg", False):
                            globals()["_rtc_k_dbg"] = True
                            print(f"[RTC-K] K={_K} klr={_klr} |dir|={float(_dir.flatten(1).norm(dim=1).mean()):.4e} "
                                  f"|y0|={float(_y0.flatten(1).norm(dim=1).mean()):.4e} "
                                  f"cos(dir,g)={float(torch.nn.functional.cosine_similarity(_dir.flatten(1), _g.flatten(1), dim=1).mean()):+.5f}",
                                  flush=True)
                    else:
                        # GN 预条件: d = J J^T g。 u = J^T g 一次反向; d = J u 一次扰动前向(有限差分)。
                        # y_theta = z_q - sigma_q * v_theta  =>  <g, y_theta> 对 theta 的梯度 = -sigma_q * d<g,v>/dtheta
                        with torch.enable_grad():
                            _vt = _fwd_v(_zq, _tq)
                            _u = torch.autograd.grad((-_sq * (_g * _vt)).sum(), _params,
                                                     retain_graph=False, allow_unused=True)
                        _u = [torch.zeros_like(_p) if _x is None else _x.detach() for _p, _x in zip(_params, _u)]
                        _un = torch.sqrt(sum((_x.double() ** 2).sum() for _x in _u)).clamp(min=1e-12)
                        _pn = torch.sqrt(sum((_p.double() ** 2).sum() for _p in _params)).clamp(min=1e-12)
                        _h = float(rtc_eps * (_pn / _un))           # 相对步长,数值稳定
                        # 有限差分必须在 fp32 下做: autocast 的 fp16 只有 ~3 位有效数字,
                        # 而这里要对两个相近的大数作差,实测 cos(d_eps, d_10eps) 只有 0.34(本该 ~1),
                        # 即方向被舍入噪声主导。fp32 前向更慢但 H200 显存充裕。
                        _fd_fp32 = bool(getattr(config.train, "rtc_fd_fp32", True))

                        @contextlib.contextmanager
                        def _fp32_fd():
                            # TF32 的尾数只有 10 位,和 fp16 一样 —— 只关 autocast 而不关 TF32
                            # 等于没关,有限差分仍会被舍入噪声主导。两个都要关。
                            _t1 = torch.backends.cuda.matmul.allow_tf32
                            _t2 = torch.backends.cudnn.allow_tf32
                            torch.backends.cuda.matmul.allow_tf32 = False
                            torch.backends.cudnn.allow_tf32 = False
                            try:
                                with torch.autocast(device_type="cuda", enabled=False):
                                    yield
                            finally:
                                torch.backends.cuda.matmul.allow_tf32 = _t1
                                torch.backends.cudnn.allow_tf32 = _t2

                        with (_fp32_fd() if _fd_fp32 else contextlib.nullcontext()):
                            with torch.no_grad():
                                _v0fd = _fwd_v(_zq, _tq)
                            _saved = _swap_params(_params, [(_p.detach() + _h * _x) for _p, _x in zip(_params, _u)])
                            try:
                                with torch.no_grad():
                                    _v1 = _fwd_v(_zq, _tq)
                            finally:
                                _swap_params(_params, _saved)
                        _v0 = _v0fd if _fd_fp32 else _v0
                        _dir = (-_sq * (_v1 - _v0)) / max(_h, 1e-12)   # = J u,已在 y 空间
                        # dirnorm 曾恰好为 0 —— 必须能区分"u 塌缩"和"扰动没生效",不能靠猜。
                        if accelerator.is_main_process and not globals().get("_rtc_fd_dbg", False):
                            globals()["_rtc_fd_dbg"] = True
                            # 判据一: 拷贝是否落地(扰动后再换回,此处 _params 已是旧值,
                            #          所以重做一次 swap 并直接比对张量)。
                            # 权重是 BF16、前向 16 位,输出分辨率 ~4e-3。1x 扰动被舍入抹掉,
                            # 1000x 又远离线性区。扫一遍 eps 看是否存在"可分辨且仍线性"的窗口:
                            # 线性区内相邻尺度的方向应近乎平行(cos~1)。没有窗口就必须换 fp32 前向。
                            # 先判前向是否确定性: 同一权重连做两次。同一次运行里 scale=1 报 dv=1.212
                            # 而 FD 用相同扰动报 dv=0,自相矛盾 -> 强烈提示前向带随机性(dropout)。
                            # 若不确定性 >> 扰动响应,之前所有 JVP 数字测的都是噪声。
                            with (_fp32_fd() if _fd_fp32 else contextlib.nullcontext()):
                                with torch.no_grad():
                                    _vA = _fwd_v(_zq, _tq)
                                    _vB = _fwd_v(_zq, _tq)
                            _dp = [(n, m.p) for n, m in transformer.named_modules()
                                   if isinstance(m, torch.nn.Dropout) and m.p > 0]
                            print(f"[RTC-DET] max|vA-vB|(同权重两次)={float((_vA-_vB).abs().max()):.6e} "
                                  f"training={transformer.training} dropout>0 的模块数={len(_dp)} "
                                  f"max|vA-v0|={float((_vA-_v0).abs().max()):.6e}", flush=True)
                            _prev = None
                            for _sc in (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
                                _pt = [(_p.detach() + _sc * _h * _x) for _p, _x in zip(_params, _u)]
                                _sv = _swap_params(_params, _pt)
                                try:
                                    with (_fp32_fd() if _fd_fp32 else contextlib.nullcontext()):
                                        with torch.no_grad():
                                            _vs = _fwd_v(_zq, _tq)
                                finally:
                                    _swap_params(_params, _sv)
                                _ds = (-_sq * (_vs - _v0)) / max(_sc * _h, 1e-12)
                                _cc = (float(torch.nn.functional.cosine_similarity(
                                        _ds.flatten(1), _prev.flatten(1), dim=1).mean())
                                       if _prev is not None else float("nan"))
                                print(f"[RTC-EPS] scale={_sc:7.1f} max|dv|={float((_vs-_v0).abs().max()):.3e} "
                                      f"|dir|={float(_ds.flatten(1).norm(dim=1).mean()):.4e} "
                                      f"cos_vs_prev={_cc:+.4f} "
                                      f"cos_dir_g={float(torch.nn.functional.cosine_similarity(_ds.flatten(1), _g.flatten(1), dim=1).mean()):+.5f}",
                                      flush=True)
                                _prev = _ds
                            _dv = float((_v1 - _v0).abs().max())
                            print(f"[RTC-FD] |u|={float(_un):.6e} |p|={float(_pn):.6e} h={_h:.6e} "
                                  f"|h*u|={_h*float(_un):.6e} max|v1-v0|={_dv:.6e} "
                                  f"max|v0|={float(_v0.abs().max()):.6e} dirnorm={float(_dir.flatten(1).norm(dim=1).mean()):.6e} "
                                  f"v0.dtype={_v0.dtype} n_params={len(_params)}", flush=True)
                        # --- 精确 JVP 交叉验证(双反向,无 eps) ---
                        # 恒等式: s(w)=<w, y_theta> => grad_theta s = J^T w;
                        #        再对 w 求 <u, J^T w> 的梯度 = J u。完全精确,不依赖差分步长。
                        # 有限差分是"能不能算准"的唯一风险点,而 J u 正是方法本身要用的方向,
                        # 所以这里做一次独立实现来对拍,而不是只信一个数。
                        info_rtc_cos_exact = None
                        info_rtc_fd_vs_exact = None
                        if str(getattr(config.train, "rtc_jvp", "both")) in ("exact", "both"):
                            try:
                                # flash-attention 的 backward 没有二阶导(aten::_scaled_dot_product_
                                # flash_attention_backward is not implemented) -> 双反向必然失败。
                                # SDPA 的 **math 后端**支持二阶导,强制走它即可真正算出 J u,
                                # 而不是退回有限差分(fp32 下恒为 0,fp16 下被舍入噪声主导)。
                                # 只包住这一处双反向:每 epoch 单个查询点,math 后端的额外开销可接受。
                                # 上下文管理器实例不可重复进入 -> 用工厂,每个分块新建一次
                                def _mk_sdp():
                                    try:
                                        from torch.nn.attention import sdpa_kernel, SDPBackend
                                        return sdpa_kernel(SDPBackend.MATH)
                                    except Exception:
                                        return torch.backends.cuda.sdp_kernel(
                                            enable_flash=False, enable_mem_efficient=False, enable_math=True)
                                _ck = max(1, int(getattr(config.train, "rtc_jvp_chunk", 1)))
                                _Bq = int(_zq.shape[0])
                                _parts = []
                                for _o in range(0, _Bq, _ck):
                                    _sl = slice(_o, min(_o + _ck, _Bq))
                                    with _mk_sdp(), torch.enable_grad():
                                        _vt2 = _fwd_v_sub(_zq[_sl], _tq[_sl], _sl)
                                        _yt2 = _zq[_sl] - _sq * _vt2
                                        # _w 不能取全零: 实测 [JVP-DBG] 显示 jtw 全为 0 -> s2 恒等于 0,
                                        # 二次求导在该退化点产出 nan(vt2/yt2/jtw 本身都 finite)。
                                        # J^T w 对 w 是**线性**的 => d/dw <J^T w, u> = J u 与 w 的取值无关,
                                        # 故改用小随机 w:数学上完全等价,只是避开全零退化点。
                                        # 用固定 generator 保证可复现,并可用两个不同 w 交叉验证结果一致。
                                        _wg = torch.Generator(device=_yt2.device).manual_seed(1234 + _o)
                                        _w = (torch.randn(_yt2.shape, generator=_wg, device=_yt2.device,
                                                          dtype=_yt2.dtype) * 1e-2).requires_grad_(True)
                                        _jtw = torch.autograd.grad((_w * _yt2).sum(), _params,
                                                                   create_graph=True, allow_unused=True)
                                        _s2 = sum((a * b).sum() for a, b in zip(_jtw, _u) if a is not None)
                                        _dx = torch.autograd.grad(_s2, _w)[0].detach().float()
                                        # nan 定位: 逐阶段查有限性,只打印一次。
                                        # math 后端显式 materialize QK^T,fp16 下可能溢出成 inf->nan,
                                        # 而 flash-attn 的 online softmax 自带减最大值保护 —— 故换后端后才暴露。
                                        if accelerator.is_main_process and not globals().get("_jvp_dbg", False):
                                            globals()["_jvp_dbg"] = True
                                            _nn = lambda t: ("None" if t is None else
                                                f"finite={bool(torch.isfinite(t).all())} dtype={t.dtype} "
                                                f"absmax={float(t.abs().max()) if torch.isfinite(t).any() else float('nan'):.3e}")
                                            _ng = sum(1 for a in _jtw if a is None)
                                            _nf = sum(1 for a in _jtw if a is not None and not bool(torch.isfinite(a).all()))
                                            print(f"[JVP-DBG] vt2: {_nn(_vt2)} | yt2: {_nn(_yt2)} | "
                                                  f"jtw: n={len(_jtw)} none={_ng} nonfinite={_nf} | "
                                                  f"s2: {_nn(_s2 if torch.is_tensor(_s2) else None)} | dir_ex: {_nn(_dx)}",
                                                  flush=True)
                                        _parts.append(_dx)
                                    del _vt2, _yt2, _w, _jtw, _s2
                                    torch.cuda.empty_cache()
                                _dir_ex = torch.cat(_parts, dim=0)
                                with torch.no_grad():
                                    info_rtc_cos_exact = float(torch.nn.functional.cosine_similarity(
                                        _dir_ex.flatten(1), _g.flatten(1), dim=1).mean())
                                    info_rtc_fd_vs_exact = float(torch.nn.functional.cosine_similarity(
                                        _dir.flatten(1), _dir_ex.flatten(1), dim=1).mean())
                                if str(getattr(config.train, "rtc_jvp", "both")) == "exact":
                                    _dir = _dir_ex
                            except Exception as _e:
                                # 双反向要保留整张前向图,可能 OOM;失败就退回有限差分并如实记录
                                if accelerator.is_main_process:
                                    print(f"[RTC] 精确 JVP 失败,退回有限差分: {type(_e).__name__}: {_e}")
                                info_rtc_cos_exact = None
                        info_rtc_cos = None
                        info_rtc_eps_stab = None
                        with torch.no_grad():
                            _c1 = torch.nn.functional.cosine_similarity(
                                _dir.flatten(1), _g.flatten(1), dim=1).mean()
                            info_rtc_cos = float(_c1)
                        # 数值稳定性检验(必要,不是可选):有限差分在 fp16 autocast 下可能被舍入噪声主导,
                        # 而 cos(JJ^T g, g)=0.015 只有随机基准 1/sqrt(65536)=0.0039 的 4 倍。
                        # 用 10x 的 eps 再算一次:两者若近似平行(cos~1)则 JVP 可信;若近似正交则是噪声。
                        if bool(getattr(config.train, "rtc_epscheck", True)):
                            _h2 = _h * 10.0
                            with (_fp32_fd() if _fd_fp32 else contextlib.nullcontext()):
                                _saved2 = _swap_params(_params, [(_p.detach() + _h2 * _x) for _p, _x in zip(_params, _u)])
                                try:
                                    with torch.no_grad():
                                        _v2 = _fwd_v(_zq, _tq)
                                finally:
                                    _swap_params(_params, _saved2)
                            with torch.no_grad():
                                _dir2 = (-_sq * (_v2 - _v0)) / max(_h2, 1e-12)
                                info_rtc_eps_stab = float(torch.nn.functional.cosine_similarity(
                                    _dir.flatten(1), _dir2.flatten(1), dim=1).mean())
                                info_rtc_dirnorm = float(_dir.flatten(1).norm(dim=1).mean())

                    # 诊断: reward 随位移的完整曲线。acc_p=0.31(<0.5)提示信赖域过大 ——
                    # 可实现方向与 g 近乎正交(cos~0.01),一阶增益极小,大位移下二阶项压倒它。
                    # 同批测"可实现方向"与"原始梯度方向"两条曲线,直接读出各自最优半径与增益。
                    if bool(getattr(config.train, "rtc_curve", False)):
                        with torch.no_grad():
                            _R0c = _reward_of_y(_y0)
                            _y0n = _y0.flatten(1).norm(dim=1).view(_shp)
                            for _nm, _dd in (("k", _dir), ("g", _g)):
                                _ddn2 = _dd.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                                for _r in (0.001, 0.003, 0.01, 0.03, 0.10, 0.20, 0.30):
                                    _yy = _y0 + (_r * _y0n) * (_dd / _ddn2)
                                    _dR = (_reward_of_y(_yy) - _R0c)
                                    diag2_acc[f"rtc_dR_{_nm}_{int(_r*1000):03d}"].append(float(_dR.mean()))
                                    diag2_acc[f"rtc_win_{_nm}_{int(_r*1000):03d}"].append(float((_dR > 0).float().mean()))

                    # ---- 归一化步 + 信赖域投影,构造正/负目标 ----
                    with torch.no_grad():
                        _dn = _dir.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                        _step = _budget / max(rtc_nascent, 1)
                        _yp, _ym = _y0.clone(), _y0.clone()
                        for _m in range(rtc_nascent):
                            _yp = _yp + _step * (_dir / _dn)
                            _ym = _ym - _step * (_dir / _dn)
                            for _t_ in ("p", "m"):
                                _y = _yp if _t_ == "p" else _ym
                                _d = _y - _y0
                                _ddn = _d.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                                _y = _y0 + _d * torch.clamp(_budget / _ddn, max=1.0)
                                if _t_ == "p": _yp = _y
                                else: _ym = _y
                        # 采纳门控: 目标必须**实测**抬高/压低 reward(OPSD 的做法;
                        # 我们此前 IT-OPD 用的一阶预测 <g, dy> 对归一化步恒为正,形同虚设)
                        _R0 = _reward_of_y(_y0)
                        _accp = (_reward_of_y(_yp) >= _R0 + rtc_margin).float()
                        _accm = (_reward_of_y(_ym) <= _R0 - rtc_margin).float()

                rtc_pack = {
                    "rtc_jq": torch.full((prompt_embeds.shape[0],), _jq, device=accelerator.device, dtype=torch.long),
                    "rtc_zq": _zq.detach(), "rtc_yp": _yp.detach(), "rtc_ym": _ym.detach(),
                    "rtc_accp": _accp.detach(), "rtc_accm": _accm.detach(),
                    "rtc_y0": _y0.detach(), "rtc_g": _g.detach(),
                }
                if accelerator.is_main_process:
                    diag2_acc["rtc_acc_p"].append(float(_accp.mean()))
                    diag2_acc["rtc_acc_m"].append(float(_accm.mean()))
                    diag2_acc["rtc_R0"].append(float(_R0.mean()))
                    if rtc_precond == "gn" and info_rtc_cos is not None:
                        # cos(JJ^T g, g): 越小说明 g 落在不可实现子空间的分量越多 —— 本方法的收益来源
                        diag2_acc["rtc_cos_gn_g"].append(info_rtc_cos)
                        if info_rtc_cos_exact is not None:
                            diag2_acc["rtc_cos_exact"].append(info_rtc_cos_exact)
                            diag2_acc["rtc_fd_vs_exact"].append(info_rtc_fd_vs_exact)
                        if info_rtc_eps_stab is not None:
                            # ~1 => 有限差分稳定,JVP 可信;~0 => 被数值噪声主导,上面那个 cos 无意义
                            diag2_acc["rtc_eps_stab"].append(info_rtc_eps_stab)
                            diag2_acc["rtc_dirnorm"].append(info_rtc_dirnorm)

            # ===== G_construct 探针: fixed-suffix 协议,四方向共享同一后缀 =====
            if gc_probe and accelerator.is_main_process:
                _sg = pipeline.scheduler.sigmas
                # 选 sigma 最接近 gc_sigma 的那个查询步(OPSD 的 argmin_j |sigma_j - sigma*|)
                _cand = [(abs(float(_sg[_j]) - gc_sigma), _j) for _j in range(config.sample.num_steps)]
                _jq = min(_cand)[1]
                _sq = float(_sg[_jq])
                _zq = latents[:, _jq].float()
                _tq = timesteps[:, _jq]

                def _Fq(_y):
                    """fixed-suffix reward: 把干净图预测 y 插进查询点,其余步用冻结策略跑完再打分。
                    v_y = (z_q - y)/sigma_q 是 y 对应的速度(rectified flow 的代数等价)。"""
                    with torch.no_grad():
                        _vy = (_zq - _y) / max(_sq, 1e-8)
                        _, _, _z, _ = sde_step_with_logprob(
                            pipeline.scheduler, _vy.float(), _tq, _zq,
                            noise_level=config.sample.noise_level)
                        for _jj in range(_jq + 1, config.sample.num_steps):
                            _tt = timesteps[:, _jj]
                            _np = transformer(
                                hidden_states=torch.cat([_z] * 2), timestep=torch.cat([_tt] * 2),
                                encoder_hidden_states=torch.cat([sample_neg_prompt_embeds_b, prompt_embeds]),
                                pooled_projections=torch.cat([sample_neg_pooled_prompt_embeds_b, pooled_prompt_embeds]),
                                return_dict=False)[0]
                            _nu, _nt = _np.chunk(2)
                            _vv = _nu + config.sample.guidance_scale * (_nt - _nu)
                            _, _, _z, _ = sde_step_with_logprob(
                                pipeline.scheduler, _vv.float(), _tt, _z.float(),
                                noise_level=config.sample.noise_level)
                        _dec = (_z.float() / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                        _im = pipeline.vae.decode(_dec.to(pipeline.vae.dtype), return_dict=False)[0].float()
                        return vteacher_scorer.score_from_images(
                            vteacher_scorer.encode_text(prompts), (_im / 2 + 0.5).clamp(0, 1)).float()

                with torch.no_grad():
                    unwrap_model(transformer, accelerator).set_adapter("default")
                    _pd = transformer(
                        hidden_states=torch.cat([_zq] * 2), timestep=torch.cat([_tq] * 2),
                        encoder_hidden_states=torch.cat([sample_neg_prompt_embeds_b, prompt_embeds]),
                        pooled_projections=torch.cat([sample_neg_pooled_prompt_embeds_b, pooled_prompt_embeds]),
                        return_dict=False)[0]
                    _u, _t = _pd.chunk(2)
                    _v0 = (_u + config.sample.guidance_scale * (_t - _u)).float()
                    _y0 = _zq - _sq * _v0                      # 锚点: 干净图预测
                    _shp = (-1, *([1] * (_y0.ndim - 1)))
                    _budget = (gc_rho * _y0.flatten(1).norm(dim=1)).view(_shp)

                def _rw_gc(_y):
                    # 即时 reward(解码+打分),与 grad 方向的定义完全一致 —— 保证可比。
                    # 定义在探针作用域顶层: ksteps 分支与 rtpbest 都要用,不能藏在某个 if 里。
                    _d2 = (_y / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                    _i2 = pipeline.vae.decode(_d2.to(pipeline.vae.dtype), return_dict=False)[0].float()
                    return vteacher_scorer.score_from_images(
                        vteacher_scorer.encode_text(prompts), (_i2 / 2 + 0.5).clamp(0, 1))

                # 方向 1: reward 梯度上升(归一化步 + 信赖域投影),即 OPSD 的方法
                _g0 = None
                _y = _y0.clone()
                for _m in range(gc_nascent):
                    with torch.enable_grad():
                        _a = _y.detach().requires_grad_(True)
                        _dc = (_a / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                        _im = pipeline.vae.decode(_dc.to(pipeline.vae.dtype), return_dict=False)[0].float()
                        _r = vteacher_scorer.score_from_images(
                            vteacher_scorer.encode_text(prompts), (_im / 2 + 0.5).clamp(0, 1))
                        _g = torch.autograd.grad(_r.sum(), _a)[0]
                        if _m == 0:
                            _g0 = _g.detach().clone()   # y0 处的梯度,供 tproj 投影使用
                    with torch.no_grad():
                        _gn = _g.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                        _y = _y.detach() + (_budget / max(gc_nascent, 1)) * (_g / _gn)
                        _d = _y - _y0
                        _dn = _d.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                        _y = (_y0 + _d * torch.clamp(_budget / _dn, max=1.0)).detach()
                _y_grad = _y

                with torch.no_grad():
                    # 方向 2: 半径匹配的随机方向(归因对照 —— 有无 reward 信息之差)
                    _rr = torch.randn_like(_y0)
                    _rn = _rr.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                    _y_rand = _y0 + _budget * (_rr / _rn)

                    # 方向 3: teacher 方向(OPSD 没有的对照)。teacher 在该查询点的干净图预测,
                    # 按同一半径归一 —— 回答"teacher 的方向本身是否 reward-improving"。
                    _y_teach = None
                    _tm = teacher_prev_sample_means
                    if _tm is not None and _tm.shape[2] > 0:
                        _dsig = float(_sg[_jq + 1]) - _sq if _jq + 1 < len(_sg) else -_sq
                        if abs(_dsig) > 1e-8:
                            _muT = _tm[:, _jq, 0].float()          # teacher 的下一状态均值
                            _vT = (_muT - _zq) / _dsig             # 反解 teacher 速度
                            _yT = _zq - _sq * _vT                  # teacher 的干净图预测
                            _dt = _yT - _y0
                            _dtn = _dt.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                            _y_teach = _y0 + _budget * (_dt / _dtn)

                    # 方向 5: RTP —— 把 reward 梯度最小二乘投影到 teacher 位移张成的子空间。
                    # 依据: teacher 输出本身就是"另一组 LoRA 权重下的模型输出",天然落在可达集内;
                    # 而现有 OPD 用的是多个 teacher 的**求和**,一个与 reward 无关的任意组合。
                    # 投影给出该子空间内与 reward 梯度最对齐的方向:可实现 + reward 对齐,
                    # 且几乎零额外成本(teacher 前向 OPD 本来就在做)。
                    _y_tproj = None
                    _y_tsum = None
                    if _tm is not None and _tm.shape[2] >= 1 and _g0 is not None:
                        _dsig2 = float(_sg[_jq + 1]) - _sq if _jq + 1 < len(_sg) else -_sq
                        if abs(_dsig2) > 1e-8:
                            _cols = []
                            for _ti in range(_tm.shape[2]):
                                _muTi = _tm[:, _jq, _ti].float()
                                if float(_muTi.abs().sum()) == 0.0:
                                    continue                      # 空槽位(teacher 数不足时的 padding)
                                _yTi = _zq - _sq * ((_muTi - _zq) / _dsig2)
                                _cols.append((_yTi - _y0).flatten(1))
                            if _cols:
                                _D = torch.stack(_cols, dim=2)                # [B,N,K]
                                _gf = _g0.flatten(1).unsqueeze(2)             # [B,N,1] 用 y0 处的梯度
                                _DtD = torch.bmm(_D.transpose(1, 2), _D)
                                _Dtg = torch.bmm(_D.transpose(1, 2), _gf)
                                _Kd = _DtD.shape[1]
                                _sc0 = _DtD.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1).clamp(min=1e-12)
                                _eye = torch.eye(_Kd, device=_D.device, dtype=_D.dtype).unsqueeze(0)
                                _coef = torch.linalg.solve(_DtD + 1e-6 * _sc0 * _eye, _Dtg)
                                # 公平基线: 等权组合(= OPD 实际使用的目标方向)。
                                # 探针原有的 "teacher" 列只用了第一个 teacher,不是真正的 baseline。
                                _dsum = torch.stack(_cols, dim=0).mean(dim=0)     # [B,N]
                                _dsn = _dsum.norm(dim=1).clamp(min=1e-12).view(-1, 1)
                                _y_tsum = (_y0.flatten(1)
                                           + _budget.view(-1, 1) * (_dsum / _dsn)).view_as(_y0)
                                diag2_acc["gc_cos_tsum_g"].append(float(
                                    torch.nn.functional.cosine_similarity(_dsum, _g0.flatten(1), dim=1).mean()))
                                _pj = torch.bmm(_D, _coef).squeeze(2)         # [B,N] = 投影后的方向
                                _pjn = _pj.norm(dim=1).clamp(min=1e-12).view(-1, 1)
                                _y_tproj = (_y0.flatten(1)
                                            + _budget.view(-1, 1) * (_pj / _pjn)).view_as(_y0)
                                diag2_acc["gc_cos_tproj_g"].append(float(
                                    torch.nn.functional.cosine_similarity(_pj, _g0.flatten(1), dim=1).mean()))
                                diag2_acc["gc_ntproj"].append(float(len(_cols)))

                # 方向 4: RTC-K 可实现目标 —— 沿 reward 梯度真实走 K 步后的模型输出,
                # 按同一半径归一,与 grad/rand/teacher 严格半径匹配可比。
                # 回答论文最关键的一问: 限制在模型可达集内还能构造出多少增益?
                _y_ks = None
                _y_ksbest = None
                _y_ksbestR10 = None
                if int(getattr(config.train, "gc_ksteps", 0)) > 0:
                    _gK = int(config.train.gc_ksteps)
                    _gklr = float(getattr(config.train, "gc_klr", 0.02))
                    _gp = transformer_trainable_parameters

                    def _fwdv_gc(_z, _t):
                        _q = transformer(
                            hidden_states=torch.cat([_z] * 2), timestep=torch.cat([_t] * 2),
                            encoder_hidden_states=torch.cat([sample_neg_prompt_embeds_b, prompt_embeds]),
                            pooled_projections=torch.cat([sample_neg_pooled_prompt_embeds_b, pooled_prompt_embeds]),
                            return_dict=False)[0]
                        _qa, _qb = _q.chunk(2)
                        return (_qa + config.sample.guidance_scale * (_qb - _qa)).float()

                    _gorig = [_x.detach().clone() for _x in _gp]
                    _ks_iters = []          # 保存上升过程的迭代点,供 ksbest 按实测 reward 挑选
                    try:
                        for _gk in range(_gK):
                            with torch.enable_grad():
                                _yk = _zq - _sq * _fwdv_gc(_zq, _tq)
                                _ak2 = _yk.detach().requires_grad_(True)
                                _gg = torch.autograd.grad(_rw_gc(_ak2).sum(), _ak2)[0].detach()
                                _gu = torch.autograd.grad((_yk * _gg).sum(), _gp,
                                                          retain_graph=False, allow_unused=True)
                            _gu = [torch.zeros_like(_x) if _z2 is None else _z2.detach()
                                   for _x, _z2 in zip(_gp, _gu)]
                            _gun = torch.sqrt(sum((_z2.double() ** 2).sum() for _z2 in _gu)).clamp(min=1e-12)
                            _gpn = torch.sqrt(sum((_x.double() ** 2).sum() for _x in _gp)).clamp(min=1e-12)
                            _gst = float(_gklr * (_gpn / _gun))
                            _ks_iters.append(_yk.detach())
                            with torch.no_grad():
                                for _x, _z2 in zip(_gp, _gu):
                                    _x.data.add_(_z2, alpha=_gst)
                        with torch.no_grad():
                            _yKg = _zq - _sq * _fwdv_gc(_zq, _tq)
                            _ks_iters.append(_yKg.detach())
                    finally:
                        _swap_params(_gp, _gorig)      # 必须还原,否则污染后续测量
                    with torch.no_grad():
                        _dk = _yKg - _y0
                        _dkn = _dk.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                        _y_ks = _y0 + _budget * (_dk / _dkn)   # 旧构造: 直线外推到同一半径
                        # ksbest: 与**训练臂实际使用的**构造一致 —— 按实测即时 reward 逐样本取最优
                        # 迭代点,信赖域只作上限。探针原有的 ksteps 列是直线外推,会低估 RTC-K。
                        if _ks_iters:
                            _rs4 = torch.stack([_rw_gc(_y) for _y in _ks_iters], dim=0)
                            _ys4 = torch.stack(_ks_iters, dim=0)
                            _bi4 = _rs4.argmax(dim=0)
                            _yb4 = _ys4[_bi4, torch.arange(_ys4.shape[1], device=_ys4.device)]
                            _db4 = _yb4 - _y0
                            _dn4 = _db4.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                            _cap = torch.minimum(_budget, _dn4)
                            _y_ksbest = _y0 + _cap * (_db4 / _dn4)
                            # ksbestR10 = nocap 臂实际使用的目标(最优迭代点方向,但半径用满 rho)。
                            # 需要它的 F_q 来检验"F_q 能否预测训练结果":
                            # 即时 reward 说该方向在 r=0.10 有 grad 的 70.6%,但 nocap 训练并不优于 kb
                            # ⇒ 即时 reward 不是正确的预测量,F_q 才可能是。缺这一列就无法验证。
                            _y_ksbestR10 = _y0 + _budget * (_db4 / _dn4)
                            diag2_acc["gc_ksbest_idx"].append(float(_bi4.float().mean()))
                            diag2_acc["gc_ksbest_r"].append(float(
                                (_cap / _y0.flatten(1).norm(dim=1).view(_shp)).mean()))

                with torch.no_grad():
                    _F0 = _Fq(_y0)
                    _res = {"noop": _F0, "grad": _Fq(_y_grad), "rand": _Fq(_y_rand)}
                    if _y_ks is not None:
                        _res["ksteps"] = _Fq(_y_ks)
                    if _y_ksbest is not None:
                        _res["ksbest"] = _Fq(_y_ksbest)     # kb 臂(带上限)实际使用的构造
                        _res["ksbestR10"] = _Fq(_y_ksbestR10)   # nocap 臂实际使用的构造
                        # 严格半径匹配对照: 把 grad / teacher 也放到 ksbest 的实际半径上重测。
                        # ksbest 的半径由上升路径自身决定(实测 ~5.2%),而其余方向固定在 rho=0.10。
                        # 虽然该偏置对 ksbest 不利(保守),但审稿人会问,所以直接补测消除这条限制。
                        with torch.no_grad():
                            _rks = (_cap / _y0.flatten(1).norm(dim=1).view(_shp))   # ksbest 的实际半径比例
                            _bm = _rks * _y0.flatten(1).norm(dim=1).view(_shp)
                            _dg = _y_grad - _y0
                            _dgn = _dg.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                            _res["gradRKS"] = _Fq(_y0 + _bm * (_dg / _dgn))
                            if _y_teach is not None:
                                _dt2 = _y_teach - _y0
                                _dt2n = _dt2.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                                _res["teacherRKS"] = _Fq(_y0 + _bm * (_dt2 / _dt2n))
                    if _y_tsum is not None:
                        _res["tsum"] = _Fq(_y_tsum)      # OPD 的真实基线方向
                    # 方向 6: RTP-best —— 在 teacher 张成的空间里按**实测 reward** 取最优组合。
                    # 依据: 最小二乘投影最大化的是 cos(一阶对齐),而 kcurve 已证明在 rho=0.10
                    # 这个半径上二阶效应主导 ⇒ 一阶最优 ≠ 实测最优(RTC-K 栽过同一个跟头,
                    # 后来改成"按实测 reward 逐样本取最优迭代点"才把 acc_p 从 0.31 修到 1.000)。
                    # 候选集: 每个单 teacher + 等权和 + 投影,逐样本取 F_q 最大者。
                    if bool(getattr(config.train, "gc_rtpbest", True)):
                        _cands, _cnames = [], []
                        if _tm is not None and _tm.shape[2] >= 1:
                            _dsig3 = float(_sg[_jq + 1]) - _sq if _jq + 1 < len(_sg) else -_sq
                            if abs(_dsig3) > 1e-8:
                                for _ti in range(_tm.shape[2]):
                                    _mu3 = _tm[:, _jq, _ti].float()
                                    if float(_mu3.abs().sum()) == 0.0:
                                        continue
                                    _y3 = _zq - _sq * ((_mu3 - _zq) / _dsig3)
                                    _d3 = _y3 - _y0
                                    _n3 = _d3.flatten(1).norm(dim=1).clamp(min=1e-12).view(_shp)
                                    _cands.append(_y0 + _budget * (_d3 / _n3)); _cnames.append(f"T{_ti}")
                        if _y_tsum is not None:
                            _cands.append(_y_tsum); _cnames.append("sum")
                        if _y_tproj is not None:
                            _cands.append(_y_tproj); _cnames.append("proj")
                        if len(_cands) >= 2:
                            # 选择与评估必须用**不同**的指标,否则是自我印证:
                            # C 个候选按同一指标取最大,即使全部等价也会系统性偏高(选择效应)。
                            # 选择用**即时 reward**(方法在训练时真正能看到的信号),
                            # 评估用**固定后缀 F_q**(我们要报告的量)。
                            _sel = torch.stack([_rw_gc(_c) for _c in _cands], dim=0)   # [C,B] 选择信号
                            _bi3 = _sel.argmax(dim=0)                                   # [B]
                            _ys3 = torch.stack(_cands, dim=0)
                            _idx3 = torch.arange(_ys3.shape[1], device=_ys3.device)
                            _y_rtpbest = _ys3[_bi3, _idx3]
                            _res["rtpbest"] = _Fq(_y_rtpbest)                           # 评估信号
                            for _ci, _cn in enumerate(_cnames):
                                diag2_acc[f"gc_pick_{_cn}"].append(float((_bi3 == _ci).float().mean()))
                            # 选择效应的空对照: 随机选一个候选,报同样的 F_q。
                            # rtpbest 必须显著高于它,否则增益只是"从 C 个里挑"的统计假象。
                            _rnd3 = torch.randint(0, len(_cands), (_ys3.shape[1],), device=_ys3.device)
                            _res["rtprand"] = _Fq(_ys3[_rnd3, _idx3])
                    if _y_tproj is not None:
                        _res["tproj"] = _Fq(_y_tproj)
                    if _y_teach is not None:
                        _res["teacher"] = _Fq(_y_teach)
                    for _k, _v in _res.items():
                        diag2_acc[f"gc_F_{_k}"].append(float(_v.mean()))
                        if _k != "noop":
                            diag2_acc[f"gc_G_{_k}"].append(float((_v - _F0).mean()))
                    diag2_acc["gc_sigma_q"].append(_sq)
                    diag2_acc["gc_jq"].append(float(_jq))

            retain_prev_sample_means = None
            if retain_w > 0.0 and retain_snapshot is not None:
                # 快照是 student 自己,所以用 student 的 CFG(sample.guidance_scale),不是 teacher 的。
                _saved = _swap_params(transformer_trainable_parameters, retain_snapshot)
                try:
                    with autocast():
                        with torch.no_grad():
                            _rl = []
                            for j in range(config.sample.num_steps):
                                _rl.append(compute_teacher_step_mean(
                                    transformer=transformer,
                                    pipeline=pipeline,
                                    latents_at_j=latents[:, j],
                                    next_latent_at_j=latents[:, j + 1],
                                    timestep_at_j=timesteps[:, j],
                                    cond_embeds=prompt_embeds,
                                    cond_pooled=pooled_prompt_embeds,
                                    neg_embeds=sample_neg_prompt_embeds_b,
                                    neg_pooled=sample_neg_pooled_prompt_embeds_b,
                                    guidance_scale=config.sample.guidance_scale,
                                    noise_level=config.sample.noise_level,
                                    solver=sample_solver,
                                    deterministic=sample_deterministic,
                                ))
                            retain_prev_sample_means = torch.stack(_rl, dim=1).float()
                finally:
                    _swap_params(transformer_trainable_parameters, _saved)

            batch_proxy_fn = proxy_reward_fns_per_teacher[teacher_idx]
            if batch_proxy_fn is None:
                monitor_fut = None
            else:
                with torch.no_grad():
                    monitor_fut = executor.submit(
                        batch_proxy_fn,
                        images,
                        prompts,
                        prompt_metadata,
                        only_strict=True,
                    )
                time.sleep(0)

            _sample_dict = {
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],
                    "next_latents": latents[
                        :, 1:
                    ],

                    "teacher_prev_sample_means": teacher_prev_sample_means,
                    "teacher_idx": torch.full(
                        (prompt_embeds.shape[0],), teacher_idx,
                        device=accelerator.device, dtype=torch.long,
                    ),
                    "monitor": monitor_fut,
            }
            if (vteacher and vteacher_mode == "reward") or bool(getattr(config.train, "itopd", False)):
                # (reward mode only) precompute detached CLIP-H text embeddings for the reward gradient
                _sample_dict["pickscore_text_embs"] = vteacher_scorer.encode_text(prompts)
            if retain_prev_sample_means is not None:
                _sample_dict["retain_prev_sample_means"] = retain_prev_sample_means
            if rtc_pack is not None:
                _sample_dict.update(rtc_pack)
            samples.append(_sample_dict)

        for sample in tqdm(
            samples,
            desc="Waiting for monitor scores",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            if sample["monitor"] is None:
                sample["monitor"] = {}
            else:
                with torch.no_grad():
                    proxy_out, _proxy_meta = sample["monitor"].result()
                sample["monitor"] = {
                    key: torch.as_tensor(value, device=accelerator.device).float()
                    for key, value in proxy_out.items()
                }

        all_monitor_keys = sorted({k for s in samples for k in s["monitor"].keys()})
        for s in samples:
            n = s["timesteps"].shape[0]
            for k in all_monitor_keys:
                if k not in s["monitor"]:
                    s["monitor"][k] = torch.full(
                        (n,), -10.0, device=accelerator.device, dtype=torch.float32
                    )

        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:

            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]

                if "avg" in samples["monitor"]:
                    mon_avg = samples["monitor"]["avg"][-len(images) :]
                    sampled_monitor = [mon_avg[i].item() for i in sample_indices]
                else:
                    sampled_monitor = [float("nan")] * len(sample_indices)
                def _format_caption(prompt, proxy_avg):
                    return f"{prompt:.100} | proxy_avg: {float(proxy_avg):.2f}"

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=_format_caption(prompt, proxy_avg),
                            )
                            for idx, (prompt, proxy_avg) in enumerate(
                                zip(sampled_prompts, sampled_monitor)
                            )
                        ],
                    },
                    step=global_step,
                )

        gathered_monitor = {
            key: accelerator.gather(value).detach().float().cpu().numpy()
            for key, value in samples["monitor"].items()
        }
        if accelerator.is_main_process:
            if diag2_acc:   # diag2 / gc_probe / rtc 共用这个累加器,条件不能写死在某一个 flag 上
                _d2 = {k: (sum(v) / len(v)) for k, v in diag2_acc.items() if v}
                wandb.log(_d2, step=global_step)
                logger.info("[DIAG2] " + "  ".join(f"{k.split('diag2_R_')[-1]}={v:.4f}"
                                                   for k, v in sorted(_d2.items())))
                diag2_acc.clear()
            monitor_log = {"epoch": epoch}
            for key, value in gathered_monitor.items():
                if "_strict_accuracy" in key or "_accuracy" in key:
                    continue
                valid = value[value != -10]
                if valid.size > 0:
                    monitor_log[f"monitor_{key}"] = float(valid.mean())
            wandb.log(monitor_log, step=global_step)

        if vteacher and vteacher_mode == "value":
            # keep the per-trajectory terminal reward as the value-regression target (Monte-Carlo return)
            _tr = samples["monitor"].get("pickscore", samples["monitor"].get("avg"))
            samples["terminal_reward"] = _tr.detach().float().clone()

        # --- On-Manifold: 在 **shuffle 之前**按 prompt 组算好组相对优势 ---
        # ⚠️ 修正(2026-08-24):初版用"组内 K 个成员 next_latents 的加权平均"当目标 —— 错误。
        # 即使 same_latent=True,K 条轨迹在 SDE 下会逐步分叉,时刻 j 时各成员处在**不同状态**;
        # 把成员 k 从自己的状态推向"其他轨迹下一状态的平均",既不在该状态的流形上,也不是合法的
        # policy improvement(实测 cos 由 +0.02 变到 -0.12、reward 从 0.81 掉到 0.74)。
        #
        # 正确形式(对照 GRPO:组相对优势作用在**每个成员自身的动作**上,从不跨成员平均状态):
        #     target_k = mu_k + eta * A_k * (a_k - mu_k)
        # a_k 是成员 k 自己采样到的 next_latent(模型自己的输出 -> 天然在流形上),
        # A_k 是组内标准化优势:>0 朝自己的动作走(那次采样好),<0 远离。
        # 这里只存 A_k;target 在训练循环里按各步的 mu_k 现算(mu_k 依赖当前参数,不能预存)。
        if onmanifold:
            _K = int(config.sample.num_image_per_prompt)
            _Bn = samples["next_latents"].shape[0]
            _mon = samples["monitor"]
            _rw = _mon.get("avg")
            if _rw is None:
                _rw = next(iter(_mon.values()))
            _rw = _rw.float()
            if _Bn % _K == 0 and _K > 1:
                _Gn = _Bn // _K
                _rg = _rw.view(_Gn, _K)
                _adv = (_rg - _rg.mean(1, keepdim=True)) / _rg.std(1, keepdim=True).clamp(min=1e-6)
                samples["onm_adv"] = _adv.reshape(-1).detach()          # [B] 组内标准化优势
                if accelerator.is_main_process and epoch % 20 == 0:
                    logger.info(f"[ON-MANIFOLD] epoch {epoch}: G={_Gn} K={_K} "
                                f"|adv|mean={_adv.abs().mean().item():.3f} "
                                f"reward spread={_rg.std(1).mean().item():.4f}")
            else:
                if accelerator.is_main_process:
                    logger.warning(f"[ON-MANIFOLD] B={_Bn} 不能被 K={_K} 整除,本 epoch 优势置零(无操作)")
                samples["onm_adv"] = torch.zeros(_Bn, device=accelerator.device)

        del samples["monitor"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps

        # --- value-critic update phase (MVP-2): regress V_psi(x_t,t,c) -> terminal reward ---
        if vteacher and vteacher_mode == "value":
            _tb, _nts = samples["timesteps"].shape
            _Rter = samples["terminal_reward"]
            value_net.train()
            _vloss_last = None
            for _vs in range(value_steps):
                bi = torch.randint(0, _tb, (value_bs,), device=accelerator.device)
                tj = torch.randint(0, _nts, (value_bs,), device=accelerator.device)
                xv = samples["latents"][bi, tj]
                tv = samples["timesteps"][bi, tj]
                cv = samples["pooled_prompt_embeds"][bi]
                Rv = _Rter[bi]
                pred = value_net(xv, tv, cv)
                vloss = torch.nn.functional.mse_loss(pred, Rv)
                if value_optimizer is None:
                    value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=value_lr)
                value_optimizer.zero_grad(); vloss.backward(); value_optimizer.step()
                _vloss_last = vloss
            value_net.eval()
            if accelerator.is_main_process and _vloss_last is not None:
                logger.info(
                    f"[VVALUE] epoch {epoch} value_mse={_vloss_last.item():.5f} "
                    f"pred~{pred.mean().item():.4f} target~{Rv.mean().item():.4f}"
                )

        for inner_epoch in range(config.train.num_inner_epochs):

            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:

                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                                prev_sample_mean, std_dev_t = compute_student_step(
                                    transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )

                        if rtc:
                            if not globals().get("_rtc_seen", False):
                                globals()["_rtc_seen"] = True
                                if accelerator.is_main_process:
                                    logger.info(f"[RTC] 训练分支首次执行: j={j} jq={int(sample['rtc_jq'][0].item())} "
                                                f"num_train_timesteps={num_train_timesteps}")
                            # 只在低噪声查询点训练(其余步不产生梯度) —— 与 OPSD/DanceOPD 的
                            # "每条轨迹一个查询点"设计一致,也是它们相对 NFT 省 9x 反向的来源。
                            _jq_b = int(sample["rtc_jq"][0].item())
                            if j != _jq_b:
                                continue
                            _z = sample["latents"][:, j].float()
                            _sgs = pipeline.scheduler.sigmas
                            _sq = float(_sgs[j])
                            _dsg = float(_sgs[j + 1]) - _sq if j + 1 < len(_sgs) else -_sq
                            if abs(_dsg) < 1e-8:
                                continue
                            # mu = z + dsigma*v  =>  v = (mu-z)/dsigma  =>  y = z - sigma*v
                            _yth = _z - _sq * (prev_sample_mean.float() - _z) / _dsg
                            _y0b = sample["rtc_y0"].float()
                            _ypb = sample["rtc_yp"].float()
                            _ymb = sample["rtc_ym"].float()
                            _ap = sample["rtc_accp"].float().view(-1, *([1] * (_yth.ndim - 1)))
                            _am = sample["rtc_accm"].float().view(-1, *([1] * (_yth.ndim - 1)))
                            # 双分支: 同一个输出同时编码"靠近正目标"与"远离负目标"(OPSD Eq.14)
                            _ypos = rtc_beta * _yth + (1.0 - rtc_beta) * _y0b
                            _yneg = (1.0 + rtc_beta) * _y0b - rtc_beta * _yth
                            _rdim = tuple(range(1, _yth.ndim))
                            with torch.no_grad():   # 自适应归一化器(detached),使损失无量纲
                                _gp = (_ypos - _ypb).abs().mean(dim=_rdim, keepdim=True).clamp(min=1e-5)
                                _gn2 = (_yneg - _ymb).abs().mean(dim=_rdim, keepdim=True).clamp(min=1e-5)
                                # 消融(默认关闭): RTC_WF_MODE=frozen
                                # `L = mean(r^2)/mean|r|` 对 r 是 1-齐次 => 梯度 0-齐次 => 对目标距离不敏感。
                                # 实测 rho 0.01 vs 0.10 的 realized(2.04e-4 vs 2.02e-4)与训练轨迹几乎重合。
                                # 冻结归一化器为首个 batch 的常数 => 损失变 2-齐次 => 梯度 1-齐次
                                # => **rho 重新线性影响更新**。初始梯度尺度与 canonical 一致(不需重调 lr)。
                                if os.environ.get("RTC_WF_MODE", "adaptive") == "frozen":
                                    # ⚠️ 必须用**跨 rho 相同的绝对常数**。
                                    # 第一版把 gamma 冻结在各自 run 的首个 batch,而该值本身正比于该 run 的 rho
                                    # (实测 rho=0.01 -> 5.4885e-3;rho=0.10 -> 5.4885e-2,正好差 10 倍)。
                                    # 于是 2r/gamma 中 r 与 gamma 同比例缩放,**尺度不变性根本没被去掉**,
                                    # 实验没有检验到想检验的东西(z=-0.16 是假阴性)。
                                    _wc = os.environ.get("RTC_WF_CONST", "").strip()
                                    if _wc:
                                        _cv = float(_wc)          # 跨 rho 固定 -> 真正打破尺度不变性
                                    else:
                                        if globals().get("_RTC_WF_CONST") is None:
                                            globals()["_RTC_WF_CONST"] = float(_gp.mean())
                                        _cv = globals()["_RTC_WF_CONST"]
                                    if not globals().get("_RTC_WF_LOGGED"):
                                        globals()["_RTC_WF_LOGGED"] = True
                                        if accelerator.is_main_process:
                                            print(f"[WF-FROZEN] const={_cv:.6e} "
                                                  f"(来源={'RTC_WF_CONST 环境变量' if _wc else '首 batch(会随 rho 缩放,不可用于 rho 对比)'})",
                                                  flush=True)
                                    _gp = torch.full_like(_gp, _cv); _gn2 = torch.full_like(_gn2, _cv)
                            _lp = (_ap * (_ypos - _ypb) ** 2 / _gp).mean(dim=_rdim)
                            _ln = (_am * (_yneg - _ymb) ** 2 / _gn2).mean(dim=_rdim)
                            if not rtc_dual:
                                _ln = _ln * 0.0
                            # omega=0.5: 本仓库 num_image_per_prompt=1,没有 prompt 组可做组内归一化,
                            # 因此不用 OPSD 的 group-normalized 权重。核心方法(GN 预条件)与此无关。
                            _loss_rtc = (0.5 * _lp + 0.5 * _ln).mean()
                            info["rtc_loss"].append(_loss_rtc.detach())
                            info["rtc_pos"].append(_lp.mean().detach())
                            info["rtc_neg"].append(_ln.mean().detach())
                            # 梯度累积下 `.grad` 会**跨 batch 累加**(gas=6/8),直接读到的是本 epoch
                            # 前若干 batch 的累加值,而 _rtc_dtheta 每 batch 被覆盖为最新
                            # ⇒ 会拿"累积梯度"比"最后一个 batch 的 dtheta"。必须取 backward 前后的**增量**。
                            _msr = (bool(getattr(config.train, "rtc_measure_dtheta", False))
                                    and globals().get("_rtc_dtheta") is not None)
                            if _msr:
                                with torch.no_grad():
                                    _g_before = [(_p.grad.clone() if _p.grad is not None
                                                  else torch.zeros_like(_p))
                                                 for _p in transformer_trainable_parameters]
                            accelerator.backward(_loss_rtc)
                            if _msr:
                                with torch.no_grad():
                                    _dth = globals()["_rtc_dtheta"]
                                    _gs = [((_p.grad if _p.grad is not None else torch.zeros_like(_p)) - _b)
                                           for _p, _b in zip(transformer_trainable_parameters, _g_before)]
                                    if len(_gs) == len(_dth):
                                        _gf2 = torch.cat([_x.flatten() for _x in _gs]).double()
                                        _df2 = torch.cat([_x.flatten() for _x in _dth]).double()
                                        _gn2b = _gf2.norm().clamp(min=1e-30)
                                        _dn2b = _df2.norm().clamp(min=1e-30)
                                        # 梯度是 -J^T J dtheta,故取负号后与 dtheta 比方向
                                        info["rtc_cos_grad_dtheta"].append(
                                            (-(_gf2 @ _df2) / (_gn2b * _dn2b)).float())
                                        info["rtc_ratio_g_dth"].append((_gn2b / _dn2b).float())
                                        # 关键判别: y_theta 与采样期的 y0 是否已漂移。
                                        # 我 15:00 的推导 (-grad = J^T J dtheta,故 cos>=0) 假设 y_theta = y0;
                                        # n=6 出现负 cos 证伪了它。若漂移≈0 而 cos 仍散布,说明真因在别处
                                        # (采样走 _fwd_v、训练走 compute_student_step,两条前向路径可能不一致);
                                        # 若漂移大,则推导只是被漂移项淹没。这一测量能定论。
                                        _y0f = sample["rtc_y0"].float().flatten(1)
                                        _ythf = _yth.detach().float().flatten(1)
                                        info["rtc_drift_rel"].append(
                                            ((_ythf - _y0f).norm(dim=1) / _y0f.norm(dim=1).clamp(min=1e-12)).mean())
                            if accelerator.sync_gradients:
                                grad_norm = accelerator.clip_grad_norm_(
                                    transformer.parameters(), config.train.max_grad_norm)
                                info["grad_norm"].append(grad_norm)
                            optimizer.step()
                            optimizer.zero_grad()
                            # --- 直接测 G_realized(整个立论的核心量) ---
                            # OPSD 报 G_realized 只有 G_construct 的 1.3%,98.7% 丢在拟合阶段。
                            # 更新后重新前向得 y_new,则 <g, y_new - y_old> 就是理论里的实现增益。
                            # 这同时给出一个可证伪预测: gain_gn / gain_raw 应当 = 1/cos(Ag,g)。
                            if bool(getattr(config.train, "rtc_measure_realized", True)):
                                with torch.no_grad():
                                    with autocast():
                                        _pm2, _ = compute_student_step(
                                            transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                                    _yth2 = _z - _sq * (_pm2.float() - _z) / _dsg
                                    _dy = _yth2 - _yth.detach()
                                    _gb = sample["rtc_g"].float()
                                    _gbf, _dyf = _gb.flatten(1), _dy.flatten(1)
                                    info["rtc_realized"].append((_gbf * _dyf).sum(1).mean())
                                    info["rtc_dy_norm"].append(_dyf.norm(dim=1).mean())
                                    info["rtc_realized_cos"].append(
                                        torch.nn.functional.cosine_similarity(_gbf, _dyf, dim=1).mean())
                            # 这个 continue 位于 `with accelerator.accumulate(...)` 内部,而 info 的汇总/上报
                            # 块是 for-j 循环体内、with 之外的兄弟语句 —— continue 会把它整个跳过。
                            # 后果不只是 rtc_loss/grad_norm 丢失: `global_step += 1` 也被跳过,
                            # 导致 checkpoint 永不保存、eval 永不触发。必须在此补上同样的汇总。
                            if accelerator.sync_gradients:
                                info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                                info = accelerator.reduce(info, reduction="mean")
                                info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                                if accelerator.is_main_process:
                                    wandb.log(info, step=global_step)
                                global_step += 1
                                info = defaultdict(list)
                            continue

                        if vteacher and vteacher_mode == "value":
                            # MVP-2: LEARNED-VALUE-GRADIENT virtual teacher. grad_x V_psi is cheap + smooth,
                            # so we steer EVERY step toward higher predicted reward-to-go (no decode, no subset).
                            x_t = sample["latents"][:, j].float()
                            t_j = sample["timesteps"][:, j]
                            cvec = sample["pooled_prompt_embeds"]
                            v_base = compute_base_velocity(
                                transformer, accelerator, sample["latents"][:, j], t_j,
                                embeds, pooled_embeds, config,
                            )
                            xg = x_t.detach().requires_grad_(True)
                            val = value_net(xg, t_j, cvec)
                            g = torch.autograd.grad(val.sum(), xg)[0]
                            g_norm = g.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, *([1] * (g.ndim - 1)))
                            vb_norm = v_base.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, *([1] * (v_base.ndim - 1)))
                            v_tgt = v_base + vteacher_sign * vteacher_alpha * (vb_norm / g_norm) * g
                            _, _, target_prev_mean, _ = sde_step_with_logprob(
                                pipeline.scheduler, v_tgt.float(), t_j, x_t,
                                prev_sample=sample["next_latents"][:, j].float(),
                                noise_level=config.sample.noise_level,
                            )
                            teacher_prev_mean = target_prev_mean.detach().unsqueeze(1)
                            info["vvalue_gnorm"].append(g_norm.mean().detach())
                        elif onmanifold:
                            # ===== On-Manifold: target_k = mu_k + eta * A_k * (a_k - mu_k) =====
                            # a_k = 该成员自己采样到的 next_latent(在流形上);A_k = 组内标准化优势。
                            with torch.no_grad():
                                _mu = prev_sample_mean.detach().float()          # mu_k
                                _ak = sample["next_latents"][:, j].float()        # a_k(自己的动作)
                                _A = sample["onm_adv"].float().view(-1, 1, 1, 1)  # A_k
                                _d = _A * (_ak - _mu)                             # 优势>0 朝自己动作,<0 远离
                                _mn = _mu.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1, 1)
                                _dn = _d.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1, 1)
                                _d = _d * torch.clamp(onm_step * _mn / _dn, max=1.0)   # 信赖域
                                _rl = sample.get("teacher_prev_sample_means")
                                if _rl is not None:
                                    _rt = _rl[:, j].float()
                                    if _rt.ndim == _mu.ndim + 1: _rt = _rt[:, 0]
                                    info["onm_cos_real_teacher"].append(
                                        torch.nn.functional.cosine_similarity(
                                            _d.flatten(1), (_rt - _mu).flatten(1), dim=1).mean().detach())
                                info["onm_adv_abs"].append(_A.abs().mean().detach())
                                teacher_prev_mean = (_mu + _d).unsqueeze(1)

                        elif itopd and (itopd_diag_only or j in set(range(max(0, num_train_timesteps - itopd_nsteps), num_train_timesteps))):
                            # ============ IT-OPD: 在 student 自己访问的 state 上"就地解出" teacher ============
                            # q*(a|s) ∝ p_θ(a|s)·exp(Q(s,a)/τ)  ⇒ Gaussian 下 μ* = μ_θ + Δμ
                            # 锚点是 μ_θ(student 自己),不是 v_base —— 这正是 MVP-1 失败的地方。
                            _mu = prev_sample_mean.detach().float()
                            _xt = sample["latents"][:, j].float()
                            _tj = sample["timesteps"][:, j]
                            _emb = sample["pickscore_text_embs"]
                            _shape = (-1,) + (1,) * (_xt.ndim - 1)
                            _sched = pipeline.scheduler
                            _si = [_sched.index_for_timestep(tt) for tt in _tj]
                            _si1 = [min(k + 1, len(_sched.sigmas) - 1) for k in _si]
                            _sig = _sched.sigmas[_si].view(_shape).to(_xt.device).float()
                            _sig_n = _sched.sigmas[_si1].view(_shape).to(_xt.device).float()
                            _dsig = _sig_n - _sig
                            _dsig = torch.where(_dsig.abs() < 1e-8, torch.full_like(_dsig, -1e-8), _dsig)
                            _munorm = _mu.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)

                            def _itopd_reward(_a):
                                # Euler 步的闭式反解: μ = x_t + (σ_next-σ)·v 且 x̂₀ = x_t - σ·v
                                #   ⇒ x̂₀(a) = x_t - σ·(a - x_t)/(σ_next - σ)
                                # 这样每个候选都不用再跑一次 transformer(K 次 forward -> 0 次)。
                                _xh = _xt - _sig * (_a - _xt) / _dsig
                                _dec = (_xh / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                                _im = pipeline.vae.decode(_dec.to(pipeline.vae.dtype), return_dict=False)[0].float()
                                _sc = vteacher_scorer.score_from_images(_emb, (_im / 2 + 0.5).clamp(0, 1))
                                # 逐候选立刻释放解码中间量:K 次 decode 的峰值应只有 1 次,
                                # 否则在显存紧张的卡上会 OOM(A40-48G 实测基线能跑、加 IT-OPD 就爆)
                                del _im, _dec, _xh
                                return _sc

                            if itopd_mode == "search":
                                # ---- Version B: 黑盒 reward-weighted local search(可用于不可微 reward) ----
                                with torch.no_grad():
                                    _cs = []
                                    for _ in range(itopd_k):
                                        _e = torch.randn_like(_mu)
                                        _e = _e / _e.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)
                                        _cs.append(_mu + itopd_explore * _munorm * _e)
                                    _R = torch.stack([_itopd_reward(_a) for _a in _cs], dim=1)   # [B,K]
                                    _A = torch.stack(_cs, dim=1)                                  # [B,K,C,H,W]
                                    # 候选间 reward 差异只有 ~1e-3 量级(PickScore ~0.8),直接 R/tau 会让 softmax
                                    # 几乎均匀 -> mu* - mu 退化成 K 个随机扰动的平均(纯噪声)。先按候选维度 z-score,
                                    # 使 tau 与 reward 量纲无关。
                                    _Rz = (_R - _R.mean(1, keepdim=True)) / _R.std(1, keepdim=True).clamp(min=1e-6)
                                    _w = torch.softmax(_Rz / max(itopd_tau, 1e-6), dim=1)
                                    _mu_star = (_w.view(_w.shape[0], _w.shape[1], 1, 1, 1) * _A).sum(1)
                                    # advantage 必须比较 R(mu*) 与 R(mu_theta),而不是"加权均值 vs 均匀均值"
                                    # —— 后者因 softmax 权重与 R 正相关而恒 >=0(Chebyshev 求和不等式),门控恒真、形同虚设。
                                    _adv = _itopd_reward(_mu_star) - _itopd_reward(_mu)
                                    info["itopd_reward"].append(_R.mean().detach())
                            else:
                                # ---- Version A: 可微 reward 的梯度方向 ----
                                with torch.enable_grad():
                                    _a = _mu.detach().requires_grad_(True)
                                    _r = _itopd_reward(_a)
                                    _g = torch.autograd.grad(_r.sum(), _a)[0]
                                with torch.no_grad():
                                    _gn = _g.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)
                                    _mu_star = _mu + itopd_step * _munorm * (_g / _gn)
                                    _adv = (_g.flatten(1) * (_mu_star - _mu).flatten(1)).sum(1)   # 一阶预测提升
                                    info["itopd_reward"].append(_r.mean().detach())

                            with torch.no_grad():
                                _d = _mu_star - _mu
                                # 信赖域: ‖Δμ‖ ≤ itopd_step·‖μ_θ‖ (Σ=σ²I 时 KL = ‖Δμ‖²/2σ²,
                                # 但 ODE 设置下 σ=0,故以 ‖μ_θ‖ 为尺度 —— 见 IT-OPD 开关处的说明)
                                _dn = _d.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)
                                _d = _d * torch.clamp(itopd_step * _munorm / _dn, max=1.0)
                                if itopd_gate:
                                    _d = _d * (_adv > 0).float().view(_shape)
                                    info["itopd_gate_frac"].append((_adv > 0).float().mean().detach())
                                # 诊断(不参与训练): virtual teacher 方向与真 RL teacher 方向的余弦
                                _real = sample.get("teacher_prev_sample_means")
                                if _real is not None:
                                    # 形状是 [B, T, M, C, H, W]:先按时间步 j 取,再取第 0 个 teacher。
                                    # (写成 _real[:,0] 会取到时间步 0 -> [B,M,C,H,W] 与 [B,C,H,W] 广播成
                                    #  [B,M,C,H,W],flatten 后维度 M 倍,cosine 报 65536 vs 196608)
                                    _rt = _real[:, j].float()
                                    if _rt.ndim == _mu.ndim + 1:
                                        _rt = _rt[:, 0]
                                    _dr = (_rt - _mu).flatten(1)
                                    _cos = torch.nn.functional.cosine_similarity(_d.flatten(1), _dr, dim=1)
                                    info["itopd_cos_real_teacher"].append(_cos.mean().detach())
                                    # 关键对照:teacher 增量本身有多大? 若 ‖d_teacher‖/‖mu‖ ≈ 0,
                                    # 低余弦只说明"这些 state 上 teacher 没意见",而不是"teacher 方向与 reward 梯度无关"。
                                    info["teacher_delta_rel"].append(
                                        (_dr.norm(dim=1) / _munorm.flatten()).mean().detach())
                                    # 随机方向的余弦基线(同维度) —— 用来判断 0.004 是否就是噪声水平
                                    _rnd = torch.randn_like(_d).flatten(1)
                                    info["cos_random_baseline"].append(
                                        torch.nn.functional.cosine_similarity(_rnd, _dr, dim=1).abs().mean().detach())
                                info["itopd_step_rel"].append((_d.flatten(1).norm(dim=1) / _munorm.flatten()).mean().detach())
                                if itopd_diag_only:
                                    # 逐 timestep 分桶记录(j=0 噪声最高, j=N-1 最低)
                                    _real2 = sample.get("teacher_prev_sample_means")
                                    if _real2 is not None:
                                        _rt2 = _real2[:, j].float()
                                        if _rt2.ndim == _mu.ndim + 1:
                                            _rt2 = _rt2[:, 0]
                                        _dr2 = (_rt2 - _mu).flatten(1)
                                        info[f"diag_cos_j{j}"].append(
                                            torch.nn.functional.cosine_similarity(_d.flatten(1), _dr2, dim=1).mean().detach())
                                        info[f"diag_tdelta_j{j}"].append(
                                            (_dr2.norm(dim=1) / _munorm.flatten()).mean().detach())
                                        # ---- 流形投影假说的直接检验 ----
                                        # 假说:teacher 是**模型**,其增量必然落在"模型可实现"的方向子空间里;
                                        # 而像素空间的原始 reward 梯度是通用方向,两者在 65536 维里几乎正交。
                                        # 检验:把 reward 梯度当作对 x_t 的扰动,让 denoiser 自己重新出速度,
                                        #      d_proj = v(x_t + eps*ĝ) - v(x_t) 就是"被模型投影过的 reward 方向"。
                                        # 若 cos(d_proj, d_teacher) >> cos(ĝ, d_teacher),假说成立,且给出了正确的 estimator。
                                        _gh = (_mu_star - _mu)
                                        _gh = _gh / _gh.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)
                                        _xn = _xt.flatten(1).norm(dim=1).clamp(min=1e-8).view(_shape)
                                        _eps = 0.02 * _xn
                                        _v0 = compute_base_velocity(transformer, accelerator, _xt, _tj,
                                                                    embeds, pooled_embeds, config)
                                        _v1 = compute_base_velocity(transformer, accelerator, _xt + _eps * _gh, _tj,
                                                                    embeds, pooled_embeds, config)
                                        # ⚠️ 空间转换:d_proj 在 **v 空间**,而 d_teacher 在 **mu 空间**。
                                        # Euler 步 mu = x_t + (sigma_next - sigma)*v,且 (sigma_next - sigma) < 0,
                                        # 所以两者天然反平行 —— 不做转换直接算余弦会得到符号相反的结果。
                                        _dproj = ((_v1 - _v0) * _dsig).flatten(1)      # -> mu 空间
                                        info[f"diag_cosproj_j{j}"].append(
                                            torch.nn.functional.cosine_similarity(_dproj, _dr2, dim=1).mean().detach())
                                        # ---- teacher 增量是"逐 state 特异"还是"跨 prompt 共享的固定偏移"? ----
                                        # 这决定 teacher-free 该做什么:
                                        #   共享成分高 -> 不需要逐 state 梯度,只需学到那个偏移;
                                        #   接近 0     -> 确实需要逐 state 估计器。
                                        # 零额外开销:teacher 增量已算出。B 个样本是**不同 prompt**、同一 timestep。
                                        _dn2 = _dr2 / _dr2.norm(dim=1, keepdim=True).clamp(min=1e-8)
                                        _B = _dn2.shape[0]
                                        if _B > 1:
                                            _G = _dn2 @ _dn2.t()                       # 两两余弦
                                            _off = (_G.sum() - _G.diag().sum()) / (_B * (_B - 1))
                                            info[f"diag_tt_cos_j{j}"].append(_off.detach())
                                            # 共享成分占比:平均方向的模长(=1 表示完全共享,~1/sqrt(B) 表示互相独立)
                                            info[f"diag_tt_shared_j{j}"].append(_dn2.mean(0).norm().detach())
                                    # 训练目标保持原样(真 teacher),本模式只做测量
                                    teacher_prev_mean = sample["teacher_prev_sample_means"][:, j].detach()
                                else:
                                    teacher_prev_mean = (_mu + _d).unsqueeze(1)

                        elif vteacher:
                            # MVP-1: reward-gradient (myopic; decode-based, last `nsteps` only). Kept for reference.
                            vt_subset = set(
                                range(max(0, num_train_timesteps - vteacher_nsteps), num_train_timesteps)
                            )
                            if j in vt_subset:
                                x_t = sample["latents"][:, j].float()
                                t_j = sample["timesteps"][:, j]
                                text_embs = sample["pickscore_text_embs"]
                                v_base = compute_base_velocity(
                                    transformer, accelerator, sample["latents"][:, j], t_j,
                                    embeds, pooled_embeds, config,
                                )
                                sched = pipeline.scheduler
                                step_index = [sched.index_for_timestep(tt) for tt in t_j]
                                sigma = sched.sigmas[step_index].view(
                                    -1, *([1] * (x_t.ndim - 1))
                                ).to(x_t.device).float()
                                xhat0_base = x_t - sigma * v_base
                                xhat0 = xhat0_base.detach().requires_grad_(True)
                                dec_in = (
                                    xhat0 / pipeline.vae.config.scaling_factor
                                ) + pipeline.vae.config.shift_factor
                                img = pipeline.vae.decode(
                                    dec_in.to(pipeline.vae.dtype), return_dict=False
                                )[0].float()
                                img01 = (img / 2 + 0.5).clamp(0, 1)
                                reward = vteacher_scorer.score_from_images(text_embs, img01)
                                g = torch.autograd.grad(reward.sum(), xhat0)[0]
                                # scale the reward direction RELATIVE to the base-velocity magnitude, so
                                # alpha is an interpretable fraction of the velocity scale (alpha=0.5 = half
                                # a |v_base| step toward higher reward). Unit-norming over all 65k latent
                                # elements made alpha=1 a ~0.4% per-element nudge -> student collapsed to base.
                                g_norm = g.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, *([1] * (g.ndim - 1)))
                                vb_norm = v_base.flatten(1).norm(dim=1).clamp(min=1e-8).view(-1, *([1] * (v_base.ndim - 1)))
                                # move x_hat0 up-reward  <=>  v decreases along +g : v_tgt = v_base - alpha*(|v_base|/|g|)*g
                                v_tgt = v_base - vteacher_alpha * (vb_norm / g_norm) * g
                                _, _, target_prev_mean, _ = sde_step_with_logprob(
                                    pipeline.scheduler, v_tgt.float(), t_j, x_t,
                                    prev_sample=sample["next_latents"][:, j].float(),
                                    noise_level=config.sample.noise_level,
                                )
                                teacher_prev_mean = target_prev_mean.detach().unsqueeze(1)
                                info["vteacher_reward"].append(reward.mean().detach())
                            else:
                                teacher_prev_mean = prev_sample_mean.detach().unsqueeze(1)
                        else:
                            teacher_prev_mean = sample[
                                "teacher_prev_sample_means"
                            ][:, j].detach()
                            if cagrad and teacher_prev_mean.shape[1] > 1:
                                # output-space conflict resolution over the M real teacher targets (no amplification).
                                with torch.no_grad():
                                    _stud = prev_sample_mean.float()
                                    _tk = teacher_prev_mean.float()  # [B, M, C, H, W]
                                    _Mt = _tk.shape[1]
                                    if cagrad_mode == "pcgrad":
                                        # per-sample PCGrad: remove pairwise-conflicting components, then AVERAGE
                                        # (mean, not sum -> bounded like uniform, so it cannot reward-hack).
                                        _gf = (_tk - _stud.unsqueeze(1)).flatten(2)  # [B, M, D] per-teacher descent dirs
                                        _gp = _gf.clone()
                                        for _i in range(_Mt):
                                            for _jj in range(_Mt):
                                                if _i != _jj:
                                                    _dot = (_gp[:, _i] * _gf[:, _jj]).sum(-1, keepdim=True)
                                                    _nj = (_gf[:, _jj] * _gf[:, _jj]).sum(-1, keepdim=True).clamp(min=1e-8)
                                                    _gp[:, _i] = _gp[:, _i] - (_dot / _nj).clamp(max=0.0) * _gf[:, _jj]
                                        _d = _gp.mean(1)  # [B, D] averaged conflict-projected descent
                                        teacher_prev_mean = (_stud.flatten(1) + _d).view_as(_stud).unsqueeze(1)
                                    else:
                                        # MGDA/CAGrad: min-norm common-descent weights, blended toward uniform by cagrad_c
                                        _g = (_tk - _stud.unsqueeze(1)).mean(0).flatten(1)  # [M, D]
                                        _G = _g @ _g.t()
                                        _w = _mgda_minnorm_weights(_G)
                                        _w = (1.0 - cagrad_c) / _Mt + cagrad_c * _w
                                        _w = (_w / _w.sum().clamp(min=1e-8)).to(_tk.dtype)
                                        teacher_prev_mean = (_w.view(1, _Mt, 1, 1, 1) * _tk).sum(dim=1, keepdim=True)
                                        info["cagrad_w_max"].append(_w.max().detach())
                            elif ste and teacher_prev_mean.shape[1] > 1:
                                # Per-(timestep, token) softmax over the M teacher residuals, then collapse K->1
                                # so the loss scale matches round-robin mopd exactly (same as the cagrad branch).
                                with torch.no_grad():
                                    _stud = prev_sample_mean.float()                      # [B,C,H,W]
                                    _tk = teacher_prev_mean.float()                       # [B,K,C,H,W]
                                    _res = _tk - _stud.unsqueeze(1)                       # per-teacher residual
                                    _mag = _res.pow(2).sum(dim=2, keepdim=True).sqrt()    # [B,K,1,H,W]
                                    if ste_norm == "zscore":
                                        # per-teacher spatial z-score: "where is THIS teacher unusually active
                                        # relative to its own baseline" -> immune to global scale gaps between
                                        # teachers (same lesson as the IT-OPD near-uniform-softmax bug).
                                        _m = _mag.mean(dim=(3, 4), keepdim=True)
                                        _sd = _mag.std(dim=(3, 4), keepdim=True).clamp(min=1e-8)
                                        _score = (_mag - _m) / _sd
                                    else:
                                        _score = _mag
                                    if ste_mode == "temporal":
                                        _score = _score.mean(dim=(3, 4), keepdim=True).expand_as(_mag)
                                    elif ste_mode == "both":
                                        _score = _score + _score.mean(dim=(3, 4), keepdim=True)
                                    _w = torch.softmax(_score / max(ste_tau, 1e-6), dim=1)  # [B,K,1,H,W], sums to 1 over K
                                    _d = (_w * _res).sum(dim=1)                             # [B,C,H,W] routed residual
                                    teacher_prev_mean = (_stud + _d).unsqueeze(1)
                                    info["ste_w_max"].append(_w.max(dim=1).values.mean().detach())
                                    info["ste_w_ent"].append(
                                        (-(_w.clamp(min=1e-8).log() * _w).sum(dim=1)).mean().detach())
                            elif poe:
                                # amplify each teacher's residual over the base velocity: target =
                                # base + lambda*(teacher - base). Over the round-robin the student
                                # converges to base + lambda*mean(residual); lambda=M => sum-of-residuals
                                # (product-of-experts / joint sum-reward optimum), the true multi-reward target.
                                x_t = sample["latents"][:, j].float()
                                t_j = sample["timesteps"][:, j]
                                v_base = compute_base_velocity(
                                    transformer, accelerator, sample["latents"][:, j], t_j,
                                    embeds, pooled_embeds, config,
                                )
                                _, _, base_prev_mean, _ = sde_step_with_logprob(
                                    pipeline.scheduler, v_base.float(), t_j, x_t,
                                    prev_sample=sample["next_latents"][:, j].float(),
                                    noise_level=config.sample.noise_level,
                                )
                                base_b = base_prev_mean.detach().unsqueeze(1)
                                if isinstance(poe_lambda, (list, tuple)):
                                    _lam = float(poe_lambda[int(sample["teacher_idx"][0].item())])
                                else:
                                    _lam = float(poe_lambda)
                                teacher_prev_mean = base_b + _lam * (teacher_prev_mean.float() - base_b)

                        # --- Realizability: 对最终 teacher 目标做 student 锚定插值(在所有组合方式之后统一施加) ---
                        if opd_alpha is not None:
                            with torch.no_grad():
                                _ms = prev_sample_mean.detach().float().unsqueeze(1)   # [B,1,C,H,W]
                                _mt = teacher_prev_mean.float()
                                teacher_prev_mean = _ms + float(opd_alpha) * (_mt - _ms)
                                # 记录 teacher 要求的步长相对 student 的幅度,用于分析"要求有多大"
                                info["realiz_dT_rel"].append(
                                    ((_mt - _ms).flatten(2).norm(dim=2).mean()
                                     / _ms.flatten(2).norm(dim=2).mean().clamp(min=1e-8)).detach())

                        delta = (
                            prev_sample_mean.float().unsqueeze(1)
                            - teacher_prev_mean.float()
                        )

                        # residual (optionally Huber/trust-region -> robust to RL-teacher reward-hacking outliers)
                        if opd_huber:
                            ad = delta.abs()
                            sq = torch.where(
                                ad <= huber_delta, delta ** 2, huber_delta * (2.0 * ad - huber_delta)
                            )
                        else:
                            sq = delta ** 2

                        if float(config.sample.noise_level) <= 0.0:
                            per_step_kl = 0.5 * sq
                        else:

                            sigma_sq = (
                                std_dev_t.float() ** 2
                            ).clamp(min=1e-8).unsqueeze(1)
                            per_step_kl = sq / (2.0 * sigma_sq)

                        per_step_kl_per_teacher_scalar = per_step_kl.mean(
                            dim=tuple(range(2, per_step_kl.ndim))
                        )
                        per_step_kl_scalar = per_step_kl_per_teacher_scalar.sum(dim=1)

                        if retain_w > 0.0 and "retain_prev_sample_means" in sample:
                            # 自蒸馏锚: 与 teacher 项同一形式(ODE 用 0.5*d^2, SDE 除以 2 sigma^2),
                            # 只是目标换成"retain_every 个 epoch 之前的自己",且没有 teacher 维度。
                            _mr = sample["retain_prev_sample_means"][:, j].detach().float()
                            _dr = prev_sample_mean.float() - _mr
                            _sqr = _dr ** 2
                            if float(config.sample.noise_level) <= 0.0:
                                _kr = 0.5 * _sqr
                            else:
                                _kr = _sqr / (2.0 * (std_dev_t.float() ** 2).clamp(min=1e-8))
                            _kr = _kr.mean(dim=tuple(range(1, _kr.ndim)))
                            # 保留项占比诊断: 快照每 retain_every 个 epoch 才刷新一次,
                            # student 漂移越久 _kr 越大。若 retain_frac 逼近 1,说明锚已经压过
                            # teacher 项、把 student 冻住 —— 那是超参没调好,不是"想法不成立",
                            # 必须能在训练中实时看见,而不是等 20 小时后从结果反推。
                            _rt = retain_w * _kr
                            info["retain_frac"].append(
                                (_rt.mean() / per_step_kl_scalar.mean().clamp(min=1e-12)).detach())
                            per_step_kl_scalar = per_step_kl_scalar + _rt
                            info["retain_kl"].append(_kr.mean().detach())

                        # Min-SNR-gamma per-step weighting: down-weight high-noise steps (t~1)
                        if snr_weight:
                            with torch.no_grad():
                                t_norm = (sample["timesteps"][:, j].float() / 1000.0).clamp(1e-3, 1.0 - 1e-3)
                                w_snr = torch.clamp(((1.0 - t_norm) / t_norm) ** 2, max=snr_gamma) / snr_gamma
                            per_step_kl_scalar = per_step_kl_scalar * w_snr

                        if ca_opd and "teacher_idx" in sample:
                            # CA-OPD: adaptive Chebyshev (min-max) reweighting over teachers so the
                            # student is pulled toward the Pareto frontier rather than the (Pareto-
                            # dominated) uniform teacher-mean velocity. Weights come from an EMA of
                            # each teacher's per-step KL -> up-weight the worst-matched teacher.
                            tidx = sample["teacher_idx"].long()
                            with torch.no_grad():
                                cur = torch.zeros(n_t, device=per_step_kl_scalar.device)
                                cnt = torch.zeros(n_t, device=per_step_kl_scalar.device)
                                cur.index_add_(0, tidx, per_step_kl_scalar.detach())
                                cnt.index_add_(0, tidx, torch.ones_like(per_step_kl_scalar))
                                cur = cur / cnt.clamp(min=1.0)
                                for m in torch.unique(tidx).tolist():
                                    teacher_kl_ema[m] = (
                                        ca_opd_ema * teacher_kl_ema[m] + (1.0 - ca_opd_ema) * cur[m]
                                    )
                                if ca_opd_regret:
                                    # regret weighting: up-weight teachers whose current KL sits
                                    # ABOVE their own attainable floor (reducible gap), so min-max
                                    # stops chasing an intrinsically-hard-but-irreducible teacher.
                                    for m in torch.unique(tidx).tolist():
                                        teacher_kl_floor[m] = torch.minimum(teacher_kl_floor[m], teacher_kl_ema[m])
                                    regret = (teacher_kl_ema - teacher_kl_floor).clamp(min=0.0) + 1e-8
                                    score_w = regret / regret.mean()
                                else:
                                    ema_valid = teacher_kl_ema.clamp(min=1e-8)
                                    score_w = ema_valid / ema_valid.mean()
                                w = torch.softmax(score_w / ca_opd_tau, dim=0)
                                w_sample = w[tidx]
                            distill_loss = (w_sample * per_step_kl_scalar).sum() / w_sample.sum().clamp(min=1e-8)
                            info["ca_opd_w_max"].append(w.max().detach())
                        else:
                            distill_loss = per_step_kl_scalar.mean()
                        loss = distill_loss

                        info["policy_loss"].append(distill_loss)
                        info["train_reverse_kl"].append(per_step_kl_scalar.mean())
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            grad_norm = accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                            info["grad_norm"].append(grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:

                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        if config.save_freq and epoch % config.save_freq == 0:
            accelerator.wait_for_everyone()
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config,
                      optimizer=optimizer, epoch=epoch,
                      extra_state={"teacher_kl_ema": teacher_kl_ema.detach().cpu(),
                                   "teacher_kl_floor": teacher_kl_floor.detach().cpu()})
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info(f"[save_ckpt] epoch {epoch} step {global_step} -> {config.save_dir}/checkpoints/checkpoint-{global_step}")

        epoch+=1

if __name__ == "__main__":
    app.run(main)
