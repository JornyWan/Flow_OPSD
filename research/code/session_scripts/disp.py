# 步长标定:LoRA 参数空间每 10 步位移 ‖θ_{k+10} − θ_k‖(Adam 作用于该空间)。锚 = lr2x / canonical(同 H200 np3)应 ≈ 2。
import sys, torch
from safetensors.torch import load_file
R = {"canon H200": "ab_open3_baseline_np3_s20260830_u100", "lr2x H200": "ab_open3_lr2x_s20260830_u100", "split1.0 H200": "sa1_h2n3_s30",
     "canon H100": "ab_open3_baseH100_s20260830_u100", "MSP H100": "ab_open3_mkF4h_s20260830_u100_r2", "split0.5 H100": "sa_h3_s30",
     "canon A100": "ab_open3_baseline_s20260905_u100", "MSP A100": "ab_open3_multikF4_s20260905_u100", "split0.5 A100": "sa_a3_s05"}
L = lambda r, k: load_file(f"{r}/checkpoints/checkpoint-{k}/lora/adapter_model.safetensors")
res = {}
for nm, r in R.items():
    W = {k: L(r, k) for k in (10, 20, 30, 40, 50)}
    seg = [sum(((W[k+10][n].float() - W[k][n].float())**2).sum() for n in W[k]).sqrt().item() for k in (10, 20, 30, 40)]
    tot = sum(((W[50][n].float() - W[10][n].float())**2).sum() for n in W[10]).sqrt().item()
    res[nm] = (seg, tot); print(f"{nm:14s} 每10步位移 " + " ".join("%.4f" % x for x in seg) + f" | 10→50 {tot:.4f}")
print("\n比值(逐段均值 / 10→50 总位移):")
for a, b in (("lr2x H200", "canon H200"), ("split1.0 H200", "canon H200"), ("split1.0 H200", "lr2x H200"), ("MSP H100", "canon H100"), ("split0.5 H100", "canon H100"), ("split0.5 H100", "MSP H100"),
             ("MSP A100", "canon A100"), ("split0.5 A100", "canon A100"), ("split0.5 A100", "MSP A100")):
    print(f"  {a:14s}/ {b:12s} {sum(res[a][0])/sum(res[b][0]):.3f} / {res[a][1]/res[b][1]:.3f}")
