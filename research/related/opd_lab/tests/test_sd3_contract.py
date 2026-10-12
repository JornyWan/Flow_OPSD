"""Real tiny diffusers SD3 + PEFT on CPU; no downloaded model or GPU required."""
from collections import Counter, OrderedDict
from contextlib import nullcontext
import copy
from types import SimpleNamespace

import pytest
import torch

from opdlab.backends import SD3Backend
from opdlab.config import Experiment


def test_real_sd3_adapter_gradient_ownership_and_behavior_snapshot():
    diffusers = pytest.importorskip("diffusers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(7)
    base = diffusers.SD3Transformer2DModel(sample_size=8, in_channels=4, out_channels=4,
           num_layers=1, attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
           caption_projection_dim=16, pooled_projection_dim=8, pos_embed_max_size=8)
    lora = peft.LoraConfig(r=2, lora_alpha=4, init_lora_weights="gaussian",
                          target_modules=["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"])
    model = peft.get_peft_model(base, lora)
    backend = object.__new__(SD3Backend)
    backend.config = Experiment(backend="sd3", precision="no", steps=4, lora_rank=2, lora_alpha=4)
    backend.device, backend.dtype = torch.device("cpu"), torch.float32
    backend.model = model
    backend.counters, backend.cache = Counter(), OrderedDict()
    backend.trainable = [p for n, p in model.named_parameters() if ".default." in n]
    backend.trainable_ids = {id(p) for p in backend.trainable}
    model.add_adapter("behavior", copy.deepcopy(lora))
    model.add_adapter("expert_ocr", copy.deepcopy(lora))
    backend.adapters = {"student": "default", "behavior": "behavior", "ocr": "expert_ocr"}
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler(shift=3)
    backend.pipeline = SimpleNamespace(scheduler=scheduler)
    backend.schedule(4)
    backend.refresh_behavior()
    state = torch.randn(1, 4, 8, 8)
    context = (torch.randn(2, 4, 16), torch.randn(2, 8))
    frozen = backend.predict(state, 1, context, "behavior")
    prediction = backend.predict(state, 1, context, "student", grad=True)
    torch.testing.assert_close(prediction.detach(), frozen)
    prediction.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backend.trainable)
    assert all(p.grad is None for n, p in model.named_parameters() if ".default." not in n)
    with torch.no_grad():
        for p in backend.trainable:
            p.add_(0.1)
    # Behavior stays frozen until explicit refresh, even after student changes.
    torch.testing.assert_close(backend.predict(state, 1, context, "behavior"), frozen)
    backend.predict(state, 1, context, "ocr")
    assert all(not p.requires_grad for p in model.parameters())
    # Fork restoration must be able to retain the original suffix policy.
    changed = backend.student_state()
    old_behavior = backend.behavior_state()
    backend.load_student(changed, refresh_behavior=False)
    torch.testing.assert_close(backend.predict(state, 1, context, "behavior"), frozen)
    assert all(torch.equal(v, backend.behavior_state()[k]) for k, v in old_behavior.items())
    backend.refresh_behavior()
    current = backend.predict(state, 1, context, "student")
    torch.testing.assert_close(backend.predict(state, 1, context, "behavior"), current)
    assert not torch.equal(current, frozen)
