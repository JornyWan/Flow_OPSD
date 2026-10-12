"""0067 §⑥ 前置条件 #2:验证 cos(g, JJᵀg) 的有限差分实现能还原已知答案。
Jᵀg 用 autograd(精确);Jv 用有限差分(近似)。线性模型有闭式解可对照。"""
import torch
torch.manual_seed(0)
D_out, D_par = 400, 300

def cos(a,b): return float((a*b).sum()/(a.norm()*b.norm()+1e-30))

# ---------- 线性:y = W θ,J = W(常数),闭式 ----------
W = torch.randn(D_out, D_par)/D_par**0.5
def make_lin():
    th = torch.zeros(D_par, requires_grad=True)
    return th, lambda t: W @ t
print("  === 线性模型(J=W 常数,有闭式解)===")
print(f"  {'eps':>8} {'有限差分 cos':>14} {'闭式 cos':>11} {'相对误差':>10}")
th, f = make_lin()
g = torch.randn(D_out)
y = f(th)
v = torch.autograd.grad((y*g).sum(), th)[0]           # Jᵀg(精确)
exact = cos(g, W @ v)                                  # JJᵀg 闭式
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    with torch.no_grad():
        vn = v/ (v.norm()+1e-30)
        y0 = f(th); y1 = f(th + eps*vn)
        Jv = (y1-y0)/eps
    c = cos(g, Jv)
    print(f"  {eps:>8.0e} {c:>14.6f} {exact:>11.6f} {abs(c/exact-1):>9.2%}")

# ---------- 非线性:y = tanh(W θ + b),J 依赖 θ ----------
b = torch.randn(D_out)*0.3
def make_nl(scale):
    th = (torch.randn(D_par)*scale).requires_grad_(True)
    return th, lambda t: torch.tanh(W @ t + b)
print("\n  === 非线性模型(y=tanh(Wθ+b),J 依赖 θ)===")
print(f"  {'eps':>8} {'有限差分 cos':>14} {'解析 JVP cos':>13} {'相对误差':>10}")
th, f = make_nl(1.0)
g = torch.randn(D_out)
y = f(th)
v = torch.autograd.grad((y*g).sum(), th)[0]
# 解析 JVP:J v = (1-tanh²(Wθ+b)) ⊙ (W v)
with torch.no_grad():
    pre = W @ th + b
    Jv_exact = (1-torch.tanh(pre)**2) * (W @ (v/v.norm()))
exact_nl = cos(g, Jv_exact)
for eps in (1e-2, 1e-3, 1e-4, 1e-5):
    with torch.no_grad():
        vn = v/(v.norm()+1e-30)
        Jv = (f(th+eps*vn)-f(th))/eps
    c = cos(g, Jv)
    print(f"  {eps:>8.0e} {c:>14.6f} {exact_nl:>13.6f} {abs(c/exact_nl-1):>9.2%}")

# ---------- fp16 下的稳定性(训练用 fp16) ----------
print("\n  === fp16 前向下的有限差分(训练实际是 fp16 混合精度)===")
print(f"  {'eps':>8} {'cos(fp16)':>12} {'vs fp32 解析':>12}")
for eps in (1e-2, 1e-3, 1e-4):
    with torch.no_grad():
        vn = (v/(v.norm()+1e-30)).half()
        th16 = th.detach().half(); W16=W.half(); b16=b.half()
        y0 = torch.tanh(W16@th16 + b16); y1 = torch.tanh(W16@(th16+eps*vn) + b16)
        Jv = ((y1-y0).float())/eps
    print(f"  {eps:>8.0e} {cos(g,Jv):>12.6f} {abs(cos(g,Jv)/exact_nl-1):>11.2%}")
print("\n  0067 §③ C-e 要求:两个 eps 的相对差 <10%。看上面哪个区间满足。")
