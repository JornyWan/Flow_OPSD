"""_flow_continue_from{,_grad} 的数值自检(纯 CPU,不需要模型)。

四项:
  T1 我的 Euler 化简 == solver.flow_grpo_step(eta=0) 的逐步输出
  T2 _flow_continue_from == 手写 Euler 后缀
  T3 grad 版的 d(endpoint)/d(forced_x0) == 解析值(v_old 与 z 无关时)
  T4 flow 后缀 != dpm2 后缀(否则这个实验没有内容),但两者都终止在 sigma=0
"""
import sys, torch
sys.path.insert(0, "CA-OPD/code/DiffusionOPSD-rpi")
sys.path.insert(0, "CA-OPD/code/DiffusionOPSD-rpi/scripts")
from src.diffusers_patch.solver import flow_grpo_step, DPMState, dpm_step

# 直接从训练脚本抽出四个函数(不 import 整个脚本,避免 absl flags 副作用)
import ast, textwrap
src = open("CA-OPD/code/DiffusionOPSD-rpi/scripts/train_opsd_ri_sd3.py").read()
tree = ast.parse(src)
ns = {"torch": torch, "DPMState": DPMState, "dpm_step": dpm_step}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in (
            "_dpm2_continue_from", "_dpm2_continue_from_grad",
            "_flow_continue_from", "_flow_continue_from_grad"):
        exec(compile(ast.Module([node], []), "<x>", "exec"), ns)
F, FG = ns["_flow_continue_from"], ns["_flow_continue_from_grad"]
D, DG = ns["_dpm2_continue_from"], ns["_dpm2_continue_from_grad"]
print(f"抽出 {len([k for k in ns if k.startswith('_')])} 个函数")

torch.manual_seed(0)
# 真实 rollout 网格形状:10 步 ⇒ 11 个 sigma,末位 0;kq=8
sigmas = torch.tensor([1.0, .94, .87, .79, .70, .60, .49, .38, .278, .15, .0])
k, SH = 8, (2, 4, 8, 8)
z_k  = torch.randn(SH)
x0   = torch.randn(SH)
prev = torch.randn(SH)
ok = True

# ---- T1: Euler 化简 == solver 的 eta=0 输出 -------------------------------
i = 8
v = torch.randn(SH)
mine = z_k + v * (sigmas[i+1] - sigmas[i])
theirs, x0p, _ = flow_grpo_step(v, z_k, 0.0, sigmas, i, None, None)
d1 = (mine - theirs).abs().max().item()
print(f"T1 Euler 化简 vs flow_grpo_step(eta=0): max|Δ| = {d1:.3e}  {'PASS' if d1 < 1e-6 else 'FAIL'}")
dx = (x0p - (z_k - sigmas[i]*v)).abs().max().item()
print(f"T1b x0_pred == z - sigma*v (强制 x0 反解的约定): max|Δ| = {dx:.3e}  {'PASS' if dx < 1e-6 else 'FAIL'}")
ok &= dx < 1e-6
ok &= d1 < 1e-6

# ---- T2: _flow_continue_from == 手写后缀 ---------------------------------
CONST = torch.randn(SH)
vfn = lambda z, s: CONST * (1.0 + 0.0 * s)          # 与 z 无关,便于 T3 解析
got = F(z_k, k, x0, prev, sigmas, vfn)
z = z_k.clone(); vv = (z - x0) / sigmas[k]
z = z + vv * (sigmas[k+1] - sigmas[k])
for j in range(k+1, len(sigmas)-1):
    z = z + vfn(z, sigmas[j]) * (sigmas[j+1] - sigmas[j])
d2 = (got - z).abs().max().item()
print(f"T2 _flow_continue_from vs 手写 Euler:      max|Δ| = {d2:.3e}  {'PASS' if d2 < 1e-6 else 'FAIL'}")
ok &= d2 < 1e-6

# ---- T3: 梯度 == 解析值 ---------------------------------------------------
x0g = x0.clone().requires_grad_(True)
end = FG(z_k, k, x0g, prev, sigmas, vfn)
g = torch.autograd.grad(end.sum(), x0g)[0]
analytic = -(sigmas[k+1] - sigmas[k]) / sigmas[k]    # 下游 v 与 z 无关 ⇒ 只有第一步带梯度
d3 = (g - analytic).abs().max().item()
print(f"T3 d(endpoint)/d(x0): got {g.flatten()[0]:.6f} vs 解析 {analytic:.6f}  "
      f"max|Δ| = {d3:.3e}  {'PASS' if d3 < 1e-6 else 'FAIL'}")
ok &= d3 < 1e-6

# ---- T4: 与 dpm2 后缀确实不同 --------------------------------------------
gf = F(z_k, k, x0, prev, sigmas, vfn)
gd = D(z_k, k, x0, prev, sigmas, vfn)
rel = (gf - gd).norm().item() / gd.norm().item()
print(f"T4 flow 后缀 vs dpm2 后缀 相对差 = {rel:.4f}  "
      f"{'PASS(有差异)' if rel > 1e-3 else 'FAIL(无差异⇒实验空转)'}")
ok &= rel > 1e-3
print(f"  末端 sigma = {sigmas[-1].item()}(两者都终止在 0)")

# ---- 成本:可微后缀的 transformer 前向次数 -------------------------------
calls = [0]
vcount = lambda z, s: (calls.__setitem__(0, calls[0]+1), CONST)[1]
calls[0] = 0; FG(z_k, k, x0.requires_grad_(True), prev, sigmas, vcount); nf = calls[0]
calls[0] = 0; DG(z_k, k, x0, prev, sigmas, vcount); nd = calls[0]
print(f"成本 可微后缀前向次数: flow={nf}  dpm2={nd}  ⇒ {'零额外成本' if nf == nd else f'{nf/max(nd,1):.1f}×'}")

print("\n" + ("✅ 全部通过" if ok else "❌ 有失败项"))
sys.exit(0 if ok else 1)
