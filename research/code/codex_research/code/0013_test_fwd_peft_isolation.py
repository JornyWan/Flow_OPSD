"""隔离 forward-mode NaN 的真因:PEFT 包装 vs 部分参数字典。

0012 在**裸** SD3.5-M 上 forward-mode 成功(cos=+0.999728),
但 trainer 里 _dir_fwd 全部元素非有限。已排除:flash/mem 后端(都不支持 forward AD,
math 是唯一选项)、fp16(autocast 实测已关)、train/eval 模式。
只剩两处差别,本脚本逐一开关:
  A. 裸模型 + 全参数字典        -> 复现 0012(应成功)
  B. 裸模型 + **部分**参数字典  -> 只推少数参数
  C. PEFT 模型 + 部分参数字典   -> 与 trainer 一致
"""
import torch, itertools
from diffusers import SD3Transformer2DModel
from torch.func import jvp, functional_call

MODEL = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/models/stable-diffusion-3.5-medium"
dev = "cuda"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

base = SD3Transformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=torch.float32).to(dev)
base.eval()
for p in base.parameters(): p.requires_grad_(False)

B = 1
hs = torch.randn(B, 16, 32, 32, device=dev)
ehs = torch.randn(B, 77, 4096, device=dev)
pool = torch.randn(B, 2048, device=dev)
ts = torch.tensor([500.0], device=dev)

def run(tag, mod, partial):
    allp = dict(mod.named_parameters())
    if partial:
        names = [n for n in allp if ("lora" in n.lower() and n.endswith("weight"))] or \
                [n for n in allp if "attn" in n and n.endswith("weight")][:8]
    else:
        names = list(allp)
    pd = {n: allp[n].detach() for n in names}
    tg = {n: torch.randn_like(pd[n]) * 1e-3 for n in names}
    def f(pp):
        return functional_call(mod, pp, args=(), kwargs=dict(
            hidden_states=hs, timestep=ts, encoder_hidden_states=ehs,
            pooled_projections=pool, return_dict=False))[0]
    try:
        with torch.autocast(device_type="cuda", enabled=False):
            y, ju = jvp(f, (pd,), (tg,))
        nf = int((~torch.isfinite(ju)).sum())
        print(f"  [{tag}] 参数字典 {len(names)}/{len(allp)} 项 | y_finite={bool(torch.isfinite(y).all())} "
              f"| ju 非有限 {nf}/{ju.numel()} | {'OK' if nf==0 else 'NaN'}", flush=True)
        return nf == 0
    except Exception as e:
        print(f"  [{tag}] 异常 {type(e).__name__}: {str(e)[:120]}", flush=True)
        return False

print("A. 裸模型 + 全参数字典"); run("A", base, False)
print("B. 裸模型 + 部分参数字典"); run("B", base, True)
print("C. PEFT 模型 + 部分(LoRA)参数字典")
try:
    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                    target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    pm = get_peft_model(base, lc); pm.eval()
    run("C", pm, True)
except Exception as e:
    print(f"  [C] 构建 PEFT 失败: {type(e).__name__}: {str(e)[:160]}")
