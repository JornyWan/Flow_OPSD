"""§4.517-P 的**零 GPU** 结构性检查:
后缀网格加密时,action(= forced_x0)对终点的影响如何 scale?

若 ||d(endpoint)/d(forced_x0)|| ~ 1/N,则"把后缀对齐到 flow-40"会让 action gradient 退化,
这不是"对齐没用",而是 **action 的定义与网格绑定**(codex 提的 fixed-interval action 正是针对这一点)。
"""
import sys, os, ast, torch
sys.path.insert(0, "CA-OPD/code/DiffusionOPSD-rpi")
from src.diffusers_patch.solver import DPMState, dpm_step

SRC = "CA-OPD/code/DiffusionOPSD-rpi/scripts/train_opsd_ri_sd3.py"

def load(nsteps):
    """按给定 RPI_QCORE_FLOW_STEPS 重新抽函数(_QFLOW_N 是模块级常量)。"""
    os.environ["RPI_QCORE_FLOW_STEPS"] = str(nsteps)
    tree = ast.parse(open(SRC).read())
    ns = {"torch": torch, "os": os, "DPMState": DPMState, "dpm_step": dpm_step,
          "_QFLOW_N": nsteps}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name.startswith(("_flow_continue", "_dpm2_continue")):
            exec(compile(ast.Module([n], []), "<x>", "exec"), ns)
    return ns

torch.manual_seed(0)
sigmas = torch.tensor([1.0, .94, .87, .79, .70, .60, .49, .38, .278, .15, .0])
k, SH = 8, (4, 4, 8, 8)
z_k, x0, prev = torch.randn(SH), torch.randn(SH), torch.randn(SH)

# 有 z 依赖的平滑 v_old(模拟真实 transformer 的状态依赖),与网格无关
W = torch.randn(256, 256) * 0.05
def vold(z, s):
    f = z.flatten(1) @ W
    return (torch.tanh(f) + 0.3 * float(s) * f).view_as(z)

# T5:回归 —— N=0 必须还原原网格行为
ns0 = load(0)
a = ns0["_flow_continue_from"](z_k, k, x0, prev, sigmas, vold)
z = z_k.clone(); vv = (z - x0) / sigmas[k]
z = z + vv * (sigmas[k+1] - sigmas[k])
for j in range(k+1, len(sigmas)-1):
    z = z + vold(z, sigmas[j]) * (sigmas[j+1] - sigmas[j])
d5 = (a - z).abs().max().item()
print(f"T5 RPI_QCORE_FLOW_STEPS=0 还原原网格: max|Δ| = {d5:.3e}  {'PASS' if d5 < 1e-6 else 'FAIL'}")

print(f"\n{'N':>5} {'||endpoint||':>13} {'||d端点/d(x0)||':>16} {'× N':>9} {'cos(dQ_N, dQ_2)':>17}")
print("  " + "-" * 63)
ref = None
for N in (2, 4, 8, 16, 32, 40, 80, 160):
    nsN = load(N)
    xg = x0.clone().requires_grad_(True)
    end = nsN["_flow_continue_from_grad"](z_k, k, xg, prev, sigmas, vold)
    # 用随机投影取一个标量,拿到完整的 Jacobian-vector 方向(不是逐元素求和)
    torch.manual_seed(1); u = torch.randn(SH)
    g = torch.autograd.grad((end * u).sum(), xg)[0]
    gn = g.norm().item()
    if ref is None: ref = g / (g.norm() + 1e-30)
    cs = ((g / (g.norm() + 1e-30)) * ref).sum().item()
    print(f"  {N:>3} {end.norm().item():>13.5f} {gn:>16.3e} {gn*N:>9.4f} {cs:>17.4f}")

print("\n判读:'× N' 一列若接近常数 ⇒ ||梯度|| = O(1/N) ⇒ action 影响随网格加密而退化。")
