import io, re, sys, torch
sys.path.insert(0,"."); sys.path.insert(0,"../shim_rpi")
from diffusionopsd.diffusers_patch.solver import dpm_step, DPMState
src = io.open("scripts/train_opsd_ri_sd3.py", encoding="utf-8").read()
ns = {"torch":torch,"dpm_step":dpm_step,"DPMState":DPMState}
m = re.search(r'^def _dpm2_continue_from_grad\(.*?(?=^def )', src, re.S|re.M)
exec(m.group(0), ns); G = ns["_dpm2_continue_from_grad"]
torch.manual_seed(0)
B_,C,H,W = 2,4,8,8
sig = torch.linspace(1.0,0.0,11).double(); k = 8
z_k = torch.randn(B_,C,H,W).double(); prev = torch.randn(B_,C,H,W).double()
Wl = torch.randn(C,C).double()
def vg(z,s): return torch.einsum("ij,bjhw->bihw", Wl, z.double()) * float(s)
y0 = torch.randn(B_,C,H,W).double()
y = y0.clone().requires_grad_(True)
e = G(z_k,k,y,prev,sig,vg); Q = (e.double()**2).sum()
(g,) = torch.autograd.grad(Q,y)
print(f"  Q(y0) 量级 = {Q.item():.4f}   ||∇Q|| = {g.norm().item():.4e}")
d = torch.randn_like(y0); d/=d.norm()
Qof = lambda yv: (G(z_k,k,yv,prev,sig,vg).double()**2).sum().item()
ana = (g*d).sum().item()
print(f"\n  autograd 方向导数 = {ana:+.10f}")
print("  %-10s %18s %14s" % ("eps","中心差分","相对误差"))
for eps in (1e-2,1e-3,1e-4,1e-5,1e-6):
    num = (Qof(y0+eps*d)-Qof(y0-eps*d))/(2*eps)
    print("  %-10.0e %18.10f %13.2e" % (eps, num, abs(num-ana)/max(abs(num),1e-30)))
print("\n  ⇒ 若 double 下各 eps 的误差都 ~1e-10,则 float32 那 1.2% 纯属精度,不是梯度 bug")
