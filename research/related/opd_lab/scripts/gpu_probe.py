"""Read-only checks inside a Slurm GPU allocation. No model training."""
import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time

start = time.perf_counter()
report = {"job_id": os.getenv("SLURM_JOB_ID"), "host": platform.node(), "checks": {}}
for name in ["torch", "diffusers", "peft", "paddle", "paddleocr", "mmcv", "mmdet", "open_clip"]:
    try:
        module = importlib.import_module(name)
        report["checks"][name] = {"ok": True, "version": getattr(module, "__version__", None),
                                  "file": getattr(module, "__file__", None)}
    except Exception as exc:
        report["checks"][name] = {"ok": False, "error": repr(exc)}
try:
    import torch
    assert torch.cuda.is_available(), "No CUDA device in allocated job"
    value = torch.randn(16, 16, device="cuda")
    assert torch.isfinite(value @ value).all()
    report["cuda"] = {"ok": True, "device": torch.cuda.get_device_name(0),
                       "total_memory": torch.cuda.get_device_properties(0).total_memory,
                       "torch_cuda": torch.version.cuda}
    from mmcv.ops import nms
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device="cuda")
    nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    report["checks"]["mmcv_cuda_nms"] = {"ok": True}
except Exception as exc:
    report["checks"]["cuda_kernels"] = {"ok": False, "error": repr(exc)}
report["elapsed_seconds"] = time.perf_counter() - start
path = Path(sys.argv[1])
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
raise SystemExit(0 if all(item["ok"] for item in report["checks"].values()) else 1)
