import io, re, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "../shim_rpi")
from diffusionopsd.diffusers_patch.solver import dpm_step, DPMState
src = io.open("scripts/train_opsd_ri_sd3.py", encoding="utf-8").read()
ns = {"torch": torch, "dpm_step": dpm_step, "DPMState": DPMState}
for fn in ("_dpm2_continue_from_grad", "_dpm2_continue_from"):
    m = re.search(rf'^def {fn}\(.*?(?=^def )', src, re.S | re.M); assert m, fn
    exec(m.group(0), ns)
G, P = ns["_dpm2_continue_from_grad"], ns["_dpm2_continue_from"]
torch.manual_seed(0)
B_,C,H,W = 2,4,8,8
sig = torch.linspace(1.0, 0.0, 11); k = 8
z_k = torch.randn(B_,C,H,W); prev = torch.randn(B_,C,H,W)
Wl = torch.randn(C,C)
def vg(z,s): return torch.einsum("ij,bjhw->bihw", Wl, z) * float(s)
def vp(z,s): return vg(z,s).detach()
y0 = torch.randn(B_,C,H,W)
print("  (1) null sanity: 可微版 vs 原版,同一 forced_x0")
e_ref = P(z_k,k,y0,prev,sig,vp)
e_grd = G(z_k,k,y0.clone().requires_grad_(True),prev,sig,vg)
rel = ((e_grd-e_ref).norm()/e_ref.norm()).item()
print(f"      相对差 = {rel:.3e}   {'✅' if rel<1e-5 else '❌'}")
print("  (2) 梯度是否回传到 forced_x0")
y = y0.clone().requires_grad_(True)
Q = (G(z_k,k,y,prev,sig,vg)**2).sum()
(g,) = torch.autograd.grad(Q,y)
print(f"      ||∇_y Q|| = {g.norm().item():.4e}   {'✅ 非零' if g.norm()>0 else '❌ 梯度断了'}")
print("  (3) 有限差分核对")
d = torch.randn_like(y0); d/=d.norm(); eps=1e-3
Qof = lambda yv: (G(z_k,k,yv,prev,sig,vg)**2).sum().item()
num = (Qof(y0+eps*d)-Qof(y0-eps*d))/(2*eps); ana = (g*d).sum().item()
err = abs(num-ana)/max(abs(num),1e-12)
print(f"      数值={num:+.6f}  autograd={ana:+.6f}  相对误差={err:.3e}  {'✅' if err<1e-3 else '❌'}")
