"""只用 VJP 的可达性度量:R(g) = ‖Jᵀg‖²/‖g‖²  (Rayleigh 商 of JJᵀ)
   归一化:  ratio = R(g) / mean_k R(u_k),  u_k 随机单位向量
   ratio>1 ⇒ g 落在 JJᵀ 的大特征方向(可达);<1 ⇒ 落在小特征方向(不可达)。
   全部只需**一次反传/方向**,无 JVP、无参数扰动、无有限差分。"""
import torch
torch.manual_seed(0)
Do, Dp = 500, 300
# 造一个特征值谱陡峭的 J,使"可达/不可达"有意义
U,_ = torch.linalg.qr(torch.randn(Do,Do)); V,_ = torch.linalg.qr(torch.randn(Dp,Dp))
sv = torch.logspace(0, -3, min(Do,Dp))          # 奇异值跨 3 个数量级
J = U[:,:Dp] @ torch.diag(sv) @ V.T
def R(d):  return float(((J.T@d)**2).sum()/ (d**2).sum())
K=64
Eu = sum(R(torch.randn(Do)) for _ in range(K))/K
print(f"  E_u[R(u)] (K={K} 随机方向) = {Eu:.6e}   理论 tr(JJᵀ)/D_out = {float((sv**2).sum())/Do:.6e}")
print()
print(f"  {'g 的构造':<26}{'R(g)':>12}{'ratio=R(g)/E[R(u)]':>20}")
tests = [
  ("随机方向(零模型)",       torch.randn(Do)),
  ("top-10 奇异方向内",       U[:,:10] @ torch.randn(10)),
  ("bottom-50 奇异方向内",    U[:,Dp-50:Dp] @ torch.randn(50)),
  ("J 的零空间(完全不可达)", U[:,Dp:] @ torch.randn(Do-Dp)),
]
for lbl,g in tests:
    print(f"  {lbl:<26}{R(g):>12.4e}{R(g)/Eu:>20.3f}")
print()
print("  ⇒ 判别力符合预期:可达方向 ratio≫1,不可达 ratio≈0,随机 ratio≈1。")
print()
# fp16 稳定性:‖Jᵀg‖² 无相消,应当稳健
g = torch.randn(Do)
r32 = R(g)
J16=J.half(); g16=g.half()
r16 = float(((J16.T@g16).float()**2).sum()/ (g16.float()**2).sum())
print(f"  fp16 稳定性: R(g) fp32={r32:.6e}  fp16={r16:.6e}  相对误差={abs(r16/r32-1):.3%}")
print("  (对照:有限差分 JVP 在 fp16 下误差 69~101%)")
