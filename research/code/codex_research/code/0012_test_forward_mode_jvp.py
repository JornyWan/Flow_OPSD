"""判定 SD3.5-M transformer 是否支持 forward-mode AD(torch.func.jvp)。

背景:GN 主臂需要 J·u。经由**二阶反向**计算时恒为 NaN
(anomaly 定位到 NativeLayerNormBackwardBackward0 / BmmBackward0 / SafeSoftmaxBackward0),
且 fp32、unwrapped DDP、手写 LayerNorm 三种修法都无效。
forward-mode 一次前向直接得 J·u,完全不经过二阶反向 —— 但风险是算子覆盖度。
本脚本只回答这一个问题,不改训练代码。
"""
import os, sys, torch
from diffusers import SD3Transformer2DModel
from torch.func import jvp, functional_call

MODEL = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/models/stable-diffusion-3.5-medium"
dev = "cuda"

print("[1] 载入 transformer(fp32,单卡)…", flush=True)
tr = SD3Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.float32).to(dev)
tr.eval()
for p in tr.parameters():
    p.requires_grad_(False)

# 只对一小撮参数取切向量,模拟 LoRA 规模;其余切向量为 0
names = [n for n, _ in tr.named_parameters()]
pick = [n for n in names if "attn" in n and n.endswith("weight")][:8]
print(f"[2] 参数 {len(names)} 个,取 {len(pick)} 个作非零切向量", flush=True)

B, C, H, W = 1, 16, 32, 32
hs = torch.randn(B, C, H, W, device=dev)
ehs = torch.randn(B, 77, 4096, device=dev)
pool = torch.randn(B, 2048, device=dev)
ts = torch.tensor([500.0], device=dev)

params = {n: p.detach() for n, p in tr.named_parameters()}
tang = {n: (torch.randn_like(v) * 1e-3 if n in pick else torch.zeros_like(v)) for n, v in params.items()}

def f(p):
    return functional_call(tr, p, args=(),
        kwargs=dict(hidden_states=hs, timestep=ts, encoder_hidden_states=ehs,
                    pooled_projections=pool, return_dict=False))[0]

print("[3] 先验一次普通前向…", flush=True)
with torch.no_grad():
    y0 = f(params)
print(f"    out shape={tuple(y0.shape)} finite={bool(torch.isfinite(y0).all())} absmax={float(y0.abs().max()):.4e}", flush=True)

print("[4] forward-mode jvp…", flush=True)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    ctx = sdpa_kernel(SDPBackend.MATH)
except Exception:
    import contextlib; ctx = contextlib.nullcontext()
try:
    with ctx:
        y, ju = jvp(f, (params,), (tang,))
    ok = bool(torch.isfinite(ju).all())
    print(f"    JVP 成功: finite={ok} absmax={float(ju.abs().max()):.4e} "
          f"|ju|={float(ju.flatten().norm()):.4e}", flush=True)
    # 与有限差分对拍:方向应近似平行
    with torch.no_grad():
        eps = 1.0
        p2 = {n: params[n] + eps * tang[n] for n in params}
        y2 = f(p2)
        fd = (y2 - y0) / eps
        cos = torch.nn.functional.cosine_similarity(ju.flatten().unsqueeze(0), fd.flatten().unsqueeze(0)).item()
    print(f"    与有限差分对拍 cos={cos:+.6f} (应接近 +1)", flush=True)
    print("RESULT: FORWARD_MODE_OK" if ok and cos > 0.9 else "RESULT: FORWARD_MODE_SUSPECT", flush=True)
except Exception as e:
    print(f"    JVP 失败: {type(e).__name__}: {e}", flush=True)
    print("RESULT: FORWARD_MODE_UNSUPPORTED", flush=True)
