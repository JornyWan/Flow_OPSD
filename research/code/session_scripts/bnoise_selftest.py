"""已知答案自检:构造 trΣ/‖G‖² 已知的合成梯度,验证半批估计量能还原它。"""
import numpy as np
rng=np.random.default_rng(0)
d=2000
def run(B_noise_true, B, reps=4000):
    G=rng.normal(size=d); G/=np.linalg.norm(G)          # ‖G‖=1
    sigma=np.sqrt(B_noise_true/d)                        # trΣ = d*sigma^2 = B_noise_true*‖G‖²
    num=[];den=[]
    for _ in range(reps):
        g=G+rng.normal(scale=sigma,size=(B,d))           # 逐样本梯度
        h=B//2; A=g[:h].mean(0); Bh=g[h:].mean(0); full=g.mean(0)
        dot=A@Bh
        num.append(full@full-dot); den.append(dot)       # 分子分母分别平均(比逐次求比更稳)
    return B*np.mean(num)/np.mean(den)

print(f"  {'真值 B_noise':>12} | {'B=48':>9} {'B=192':>9} {'B=1152':>9}   <- 估计值")
for t in [100,661,1152,4000]:
    print(f"  {t:>12} | "+" ".join(f"{run(t,B):>9.0f}" for B in (48,192,1152)))
print()
print("  判据:估计值应在真值 ±10% 内,且不随 B 系统漂移。")
# 零结构对照:G=0(纯噪声)时 B_noise 应 -> 巨大
G0=np.zeros(d); num=[];den=[]
for _ in range(4000):
    g=rng.normal(scale=1.0,size=(1152,d)); h=576
    A=g[:h].mean(0);Bh=g[h:].mean(0);full=g.mean(0)
    num.append(full@full-A@Bh); den.append(A@Bh)
print(f"  零模型(G=0,应 -> 极大或负/不稳): B_noise_hat = {1152*np.mean(num)/np.mean(den):,.0f}")
