"""SD3.5 LoRA backend and a CPU-only causal toy for contract tests."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import nullcontext
import copy
import importlib
from pathlib import Path
import sys

import torch


def activate_upstream(runtime):
    runtime = Path(runtime).resolve()
    for name, module in list(sys.modules.items()):
        if name == "flow_grpo" or name.startswith("flow_grpo."):
            file = getattr(module, "__file__", None)
            if file and not Path(file).resolve().is_relative_to(runtime):
                raise RuntimeError(f"Refusing an already imported old fork: {file}")
    sys.path.insert(0, str(runtime))
    importlib.invalidate_caches()


class ToyBackend:
    def __init__(self, config, site=None, runtime=None, device="cpu", training=True):
        self.config, self.device = config, torch.device(device)
        self.weight = torch.nn.Parameter(torch.zeros(2, device=self.device))
        self.behavior = self.weight.detach().clone()
        self.counters = Counter()
        self.schedule(config.steps)

    def parameters(self):
        return [self.weight]

    def encode(self, prompt):
        return prompt

    def initial(self, seed):
        return torch.randn((1, 2), generator=torch.Generator().manual_seed(seed)).to(self.device) * 0.6

    def schedule(self, steps):
        self.sigmas = torch.linspace(1, 0, steps + 1, device=self.device)
        return self.sigmas

    def predict(self, state, index, context, adapter, grad=False):
        self.counters[f"forward_{adapter}"] += 1
        self.counters["forward_examples"] += 1
        if adapter in {"student", "behavior"}:
            weight = self.weight if adapter == "student" else self.behavior
            return 0.15 * state + weight
        v = 0.15 * state.clone()
        axis = self.config.teachers.index(adapter) % 2
        v[:, axis] = state[:, axis] / self.sigmas[index].clamp_min(0.1)
        # A local expert can damage the other constraint.
        v[:, 1 - axis] += 0.12
        return v.detach()

    def step(self, state, action, index):
        return state.float() + (self.sigmas[index + 1] - self.sigmas[index]) * action.float()

    def decode(self, endpoint):
        self.counters["decode_images"] += 1
        return endpoint.detach().cpu().clone()

    def refresh_behavior(self):
        self.behavior.copy_(self.weight.detach())

    def student_state(self):
        return {"weight": self.weight.detach().cpu().clone()}

    def behavior_state(self):
        return {"weight": self.behavior.detach().cpu().clone()}

    def load_student(self, state, refresh_behavior=True):
        self.weight.data.copy_(state["weight"].to(self.device))
        if refresh_behavior:
            self.refresh_behavior()

    def export(self, path, state=None):
        Path(path).mkdir(parents=True, exist_ok=False)
        torch.save(state or self.student_state(), Path(path) / "toy.pt")

    @torch.no_grad()
    def generate(self, record, seed, steps, sampler="flow", adapter="student"):
        if sampler != "flow":
            raise ValueError("Toy backend implements flow only")
        self.schedule(steps)
        z = self.initial(seed)
        context = self.encode(record["prompt"])
        for i in range(steps):
            z = self.step(z, self.predict(z, i, context, adapter), i)
        return self.decode(z)


class SD3Backend:
    def __init__(self, config, site, runtime, device="cuda", training=True):
        if not torch.cuda.is_available():
            raise RuntimeError("SD3 needs an allocated CUDA GPU; use backend=toy for CPU contract tests")
        activate_upstream(runtime)
        from diffusers import StableDiffusion3Pipeline
        from peft import LoraConfig, PeftModel, get_peft_model
        self.config, self.device = config, torch.device(device)
        self.dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[config.precision]
        self.counters = Counter()
        self.cache = OrderedDict()
        self.pipeline = StableDiffusion3Pipeline.from_pretrained(site["model"])
        self.pipeline.set_progress_bar_config(disable=True)
        for module in (self.pipeline.vae, self.pipeline.text_encoder, self.pipeline.text_encoder_2,
                       self.pipeline.text_encoder_3, self.pipeline.transformer):
            module.requires_grad_(False)
            module.eval()
        self.pipeline.vae.to(self.device, dtype=torch.float32)
        for encoder in (self.pipeline.text_encoder, self.pipeline.text_encoder_2, self.pipeline.text_encoder_3):
            encoder.to(self.device, dtype=self.dtype)
        base = self.pipeline.transformer.to(self.device, dtype=torch.float32)
        targets = ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
                   "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]
        if config.initial_adapter:
            self.model = PeftModel.from_pretrained(base, config.initial_adapter, is_trainable=training)
            if training and (self.model.peft_config["default"].r != config.lora_rank or
                             self.model.peft_config["default"].lora_alpha != config.lora_alpha):
                raise ValueError("Initial LoRA architecture does not match the declared config")
        else:
            self.model = get_peft_model(base, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha,
                                          init_lora_weights="gaussian", target_modules=targets))
        self.pipeline.transformer = self.model
        self.trainable = [p for n, p in self.model.named_parameters() if ".default." in n]
        if not self.trainable:
            raise RuntimeError("No student LoRA parameters found")
        self.trainable_ids = {id(p) for p in self.trainable}
        self.model.add_adapter("behavior", copy.deepcopy(self.model.peft_config["default"]))
        self.adapters = {"student": "default", "behavior": "behavior"}
        if training:
            for name in config.teachers:
                internal = f"expert_{name}"
                self.model.load_adapter(site["teachers"][name], adapter_name=internal, is_trainable=False)
                self.adapters[name] = internal
            self.model.enable_gradient_checkpointing()
        self.refresh_behavior()
        self.schedule(config.steps)

    def parameters(self):
        return self.trainable

    def _activate(self, adapter, grad=False):
        if grad and adapter != "student":
            raise ValueError("Only student predictions may carry gradients")
        self.model.set_adapter(self.adapters[adapter])
        # PEFT set_adapter may re-enable gradients. Explicitly enforce ownership.
        for p in self.model.parameters():
            p.requires_grad_(grad and id(p) in self.trainable_ids)
        self.model.train(grad)

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.dtype != torch.float32 else nullcontext()

    @torch.no_grad()
    def encode(self, prompt):
        from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
        if prompt not in self.cache:
            encoders = [self.pipeline.text_encoder, self.pipeline.text_encoder_2, self.pipeline.text_encoder_3]
            tokenizers = [self.pipeline.tokenizer, self.pipeline.tokenizer_2, self.pipeline.tokenizer_3]
            with self.autocast():
                embeds, pooled = encode_prompt(encoders, tokenizers, ["", prompt], self.config.max_sequence_length)
            self.counters["text_encode_prompts"] += 2
            self.cache[prompt] = (embeds.detach().cpu(), pooled.detach().cpu())
            if len(self.cache) > 64:
                self.cache.popitem(last=False)
        return self.cache[prompt]

    def schedule(self, steps):
        self.pipeline.scheduler.set_timesteps(steps, device=self.device)
        self.sigmas = self.pipeline.scheduler.sigmas.to(self.device)
        if len(self.sigmas) != steps + 1 or not torch.all(self.sigmas[:-1] > self.sigmas[1:]):
            raise ValueError("Expected a strictly decreasing Euler sigma schedule")
        return self.sigmas

    def initial(self, seed):
        scale = self.pipeline.vae_scale_factor
        if self.config.resolution % (scale * 2):
            raise ValueError("Resolution must match SD3 latent patch dimensions")
        shape = (1, self.model.config.in_channels, self.config.resolution // scale, self.config.resolution // scale)
        # Protocol-specific CPU FP32 per-image noise; independent of batching/rank.
        return torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(self.device)

    def predict(self, state, index, context, adapter, grad=False):
        self._activate(adapter, grad)
        guidance = self.config.teacher_guidance[adapter] if adapter in self.config.teachers else self.config.guidance_scale
        embeds, pooled = (v.to(self.device) for v in context)
        do_cfg = guidance > 1
        if not do_cfg:
            embeds, pooled = embeds[1:], pooled[1:]
        inputs = torch.cat([state, state]) if do_cfg else state
        self.counters[f"forward_{adapter}"] += 1
        self.counters["forward_examples"] += len(inputs)
        scope = nullcontext() if grad else torch.no_grad()
        with scope, self.autocast():
            value = self.model(hidden_states=inputs,
                               timestep=self.pipeline.scheduler.timesteps[index].expand(len(inputs)),
                               encoder_hidden_states=embeds, pooled_projections=pooled,
                               return_dict=False)[0]
            if self.config.cfg_combine_precision == "float32":
                # Native Accelerate.prepare converts model outputs before the
                # caller combines CFG. Explicit opt-in preserves legacy replay.
                value = value.float()
            if do_cfg:
                uncond, cond = value.chunk(2)
                value = uncond + guidance * (cond - uncond)
        return value.float()

    def step(self, state, action, index):
        return state.float() + (self.sigmas[index + 1] - self.sigmas[index]) * action.float()

    @torch.no_grad()
    def decode(self, endpoint):
        self.counters["decode_images"] += len(endpoint)
        z = endpoint.float() / self.pipeline.vae.config.scaling_factor + self.pipeline.vae.config.shift_factor
        image = self.pipeline.vae.decode(z, return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]

    @torch.no_grad()
    def refresh_behavior(self):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model, adapter_name="default").items()}
        set_peft_model_state_dict(self.model, state, adapter_name="behavior")
        self._activate("behavior")

    def student_state(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model, adapter_name="default").items()}

    def behavior_state(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model, adapter_name="behavior").items()}

    def load_student(self, state, refresh_behavior=True):
        from peft import set_peft_model_state_dict
        expected = self.student_state()
        if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
            raise ValueError("LoRA checkpoint keys/shapes do not match")
        set_peft_model_state_dict(self.model, state, adapter_name="default")
        if refresh_behavior:
            self.refresh_behavior()

    def export(self, path, state=None):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        current = self.student_state()
        if state is not None:
            # Saving an alternate (e.g. EMA) student must not refresh the frozen
            # behavior, which can also own the deployment anchor.
            self.load_student(state, refresh_behavior=False)
        try:
            self.model.save_pretrained(path, selected_adapters=["default"])
        finally:
            if state is not None:
                self.load_student(current, refresh_behavior=False)

    @torch.no_grad()
    def generate(self, record, seed, steps, sampler="flow", adapter="student"):
        self.schedule(steps)
        z, context = self.initial(seed), self.encode(record["prompt"])
        if sampler == "flow":
            for i in range(steps):
                z = self.step(z, self.predict(z, i, context, adapter), i)
        else:
            from flow_grpo.diffusers_patch.solver import run_sampling
            # Reuse the pinned upstream implementation, including DPM history.
            times = self.pipeline.scheduler.timesteps
            def velocity(state, timestep):
                index = int((times == timestep).nonzero(as_tuple=False)[0])
                return self.predict(state, index, context, adapter)
            z, _, _ = run_sampling(velocity, z, self.pipeline.scheduler, times,
                                   solver=sampler, deterministic=True, eta=0.0,
                                   generator=torch.Generator(device=self.device).manual_seed(seed))
        return self.decode(z)


def build_backend(config, site, runtime, device, training=True):
    cls = ToyBackend if config.backend == "toy" else SD3Backend
    return cls(config, site, runtime, device, training)
