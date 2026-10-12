from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import re

from .io import read_json


@dataclass
class Experiment:
    # This local-intervention engine is NOT the native MOPD recipe.
    backend: str = "sd3"
    method: str = "verified"
    selection_protocol: str = "legacy"
    selection_reference: str | None = None
    dataset: str = ""
    split: str = "train"
    seed: int = 42
    resolution: int = 512
    steps: int = 10
    guidance_scale: float = 4.5
    max_sequence_length: int = 256
    precision: str = "fp16"
    # Historical runs combined half-precision predictions before casting. New
    # parity studies can explicitly match Accelerate's FP32-output CFG arithmetic.
    cfg_combine_precision: str = "model"
    teachers: list[str] = field(default_factory=lambda: ["ocr", "geneval"])
    teacher_guidance: dict[str, float] = field(default_factory=lambda: {"ocr": 4.5, "geneval": 1.0})
    alphas: list[float] = field(default_factory=lambda: [0.5, 1.0])
    fixed_teacher: str = "ocr"
    fixed_alpha: float = 1.0
    accept_probability: float = 0.5
    disagreement_min: float = 0.0
    utility_joint_weight: float = 0.5
    min_gain: float = 0.0
    protect_satisfied: bool = True
    quality_floor: float | None = None
    quality_drop_tolerance: float | None = None
    outer_epochs: int = 100
    queries_per_rank: int = 12
    fitting_epochs: int = 1
    accumulation: int = 12
    timestep_fraction: float = 0.99
    learning_rate: float = 0.0003
    optimizer_step_rule: str = "constant"
    record_selected_states: bool = False
    record_fit_blocks: bool = False
    adam_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.0001
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    initial_adapter: str | None = None
    ema_decay: float = 0.9
    ema_interval: int = 8
    save_every: int = 10
    audit_postfit: bool = True
    diagnostic_manifest: str | None = None
    training_manifest: str | None = None
    save_candidates: bool = False

    @classmethod
    def load(cls, path):
        values = read_json(path)
        config = cls(**values)  # Unknown options are errors, never ignored.
        if config.dataset:
            config.dataset = str(Path(config.dataset).expanduser().resolve())
        if config.initial_adapter:
            config.initial_adapter = str(Path(config.initial_adapter).expanduser().resolve())
        if config.diagnostic_manifest:
            config.diagnostic_manifest = str(Path(config.diagnostic_manifest).expanduser().resolve())
        if config.training_manifest:
            config.training_manifest = str(Path(config.training_manifest).expanduser().resolve())
        if config.selection_reference:
            config.selection_reference = str(Path(config.selection_reference).expanduser().resolve())
        config.validate()
        return config

    def validate(self):
        if self.cfg_combine_precision not in {"model", "float32"}:
            raise ValueError("cfg_combine_precision must be model or float32")
        if self.backend not in {"toy", "sd3"}:
            raise ValueError("backend must be toy or sd3")
        if self.method not in {"verified", "verified_fixed", "fixed", "single_query", "round_robin", "random", "disagreement", "harm_veto"}:
            raise ValueError("Unknown method")
        if self.selection_protocol not in {"legacy", "semantic_batch_v1", "semantic_veto_v1"}:
            raise ValueError("Unknown selection protocol")
        if self.method == "harm_veto" and self.selection_protocol != "semantic_veto_v1":
            raise ValueError("harm_veto requires its own versioned protocol")
        if self.optimizer_step_rule not in {"constant", "accepted_fraction"}:
            raise ValueError("Unknown optimizer step rule")
        if type(self.record_selected_states) is not bool:
            raise ValueError("record_selected_states must be a boolean")
        if type(self.record_fit_blocks) is not bool:
            raise ValueError("record_fit_blocks must be a boolean")
        if self.record_fit_blocks and (not self.record_selected_states or
                self.selection_protocol != "semantic_batch_v1" or
                self.accumulation != self.queries_per_rank):
            raise ValueError("Fit-block recording requires selected states, semantic selection and one group")
        if (self.optimizer_step_rule != "constant" or self.record_selected_states) and self.selection_protocol != "semantic_batch_v1":
            raise ValueError("Sparse step protocols require semantic_batch_v1")
        if self.selection_protocol in {"semantic_batch_v1", "semantic_veto_v1"}:
            source_method = "harm_veto" if self.selection_protocol == "semantic_veto_v1" else "verified_fixed"
            if self.method not in {source_method, "random", "single_query"}:
                raise ValueError("Unsupported method for the declared semantic protocol")
            if self.selection_protocol == "semantic_veto_v1" and (
                    self.optimizer_step_rule != "constant" or self.fitting_epochs != 1 or
                    self.audit_postfit or not self.training_manifest or self.accumulation != self.queries_per_rank):
                raise ValueError("Veto v1 requires constant LR, one fit, a training manifest, one group and no legacy postfit audit")
            if self.teachers != [self.fixed_teacher] or self.alphas != [1.0] or self.fixed_alpha != 1.0:
                raise ValueError("Semantic protocol requires one fixed full-amplitude teacher")
            if bool(self.selection_reference) != (self.method == "random"):
                raise ValueError("Semantic random requires a frozen source count reference; other arms forbid it")
            if (not self.protect_satisfied or self.min_gain != 0 or self.quality_floor is not None
                    or self.quality_drop_tolerance is not None):
                raise ValueError("Semantic acceptance uses fixed clause rules, without utility/quality overrides")
            if self.accept_probability != .5 or self.utility_joint_weight != .5 or self.disagreement_min != 0:
                raise ValueError("Legacy probability/utility/disagreement knobs are inactive in semantic_batch_v1")
            if self.diagnostic_manifest:
                raise ValueError("Legacy diagnostic manifests do not support semantic batch selection")
        elif self.selection_reference:
            raise ValueError("A selection reference requires semantic_batch_v1")
        if type(self.protect_satisfied) is not bool:
            raise ValueError("protect_satisfied must be a boolean")
        if self.training_manifest and self.diagnostic_manifest:
            raise ValueError("Use separate runs for training and frozen diagnostic manifests")
        if self.precision not in {"no", "fp16", "bf16"}:
            raise ValueError("Unknown precision")
        for key in ("steps", "resolution", "max_sequence_length", "outer_epochs", "queries_per_rank",
                    "fitting_epochs", "accumulation", "lora_rank", "lora_alpha", "ema_interval", "save_every"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.queries_per_rank % self.accumulation:
            raise ValueError("queries_per_rank must be divisible by accumulation; no hidden partial accumulation")
        if not 0 < self.timestep_fraction <= 1 or int(self.steps * self.timestep_fraction) < 1:
            raise ValueError("Empty or invalid training timestep range")
        if not self.teachers or len(set(self.teachers)) != len(self.teachers):
            raise ValueError("Teachers must be nonempty and unique")
        if any(name in {"student", "behavior", "default"} or not re.fullmatch(r"[a-zA-Z0-9_-]+", name)
               for name in self.teachers):
            raise ValueError("Teacher names must be safe adapter identifiers, not reserved student/behavior names")
        if set(self.teachers) - self.teacher_guidance.keys():
            raise ValueError("Every teacher needs an explicit guidance scale")
        if not self.alphas or any(not 0 < a <= 1 for a in self.alphas) or len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must be unique values in (0,1]")
        if self.fixed_teacher not in self.teachers or self.fixed_alpha not in self.alphas:
            raise ValueError("Fixed action must be in the candidate set")
        if not 0 <= self.utility_joint_weight <= 1 or not 0 <= self.accept_probability <= 1:
            raise ValueError("Invalid probability/utility weight")
        if self.min_gain < 0 or self.disagreement_min < 0:
            raise ValueError("Margins must be nonnegative")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("Invalid EMA decay")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.adam_epsilon <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer parameters")
        if len(self.adam_betas) != 2 or any(not 0 <= b < 1 for b in self.adam_betas):
            raise ValueError("Invalid Adam betas")
        for k, v in asdict(self).items():
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"Nonfinite setting: {k}")
        if self.quality_drop_tolerance is not None and self.quality_drop_tolerance < 0:
            raise ValueError("quality_drop_tolerance must be nonnegative")
        return self

    def as_dict(self):
        return asdict(self)
