"""功效分析:在真实配置 B=1152 下,n_updates 次更新能把 B_noise 的 CI 收到多窄?
判据来自 0064 §③:B-a 需 CI 上界<1152 或 下界>2304;B-b 需 hi/lo<3。"""
import numpy as np
import sys
rng=np.random.default_rng(1); d=int(sys.argv[1])   # 噪声的有效维
def sim_updates(Bn,B,n):
    """返回 n 次更新各自的 (num, den);用 d 维高斯,只需其充分统计量"""
    vh=(Bn/d)*2.0/B                                # 半批均值每分量方差
    # dot = ‖G‖² + G·(ξA+ξB) + ξA·ξB ;  取 ‖G‖=1
    #   G·(ξA+ξB) ~ N(0, 2*vh)
    #   ξA·ξB     ~ 近似 N(0, d*vh²)
    dot = 1.0 + rng.normal(scale=np.sqrt(2*vh),size=n) + rng.normal(scale=np.sqrt(d)*vh,size=n)
    # ‖f‖² = ‖G‖² + 2G·f_noise + ‖f_noise‖²,  f_noise=(ξA+ξB)/2 ~ 每分量方差 vh/2
    fn2 = rng.chisquare(d,size=n)*(vh/2)
    nrm = 1.0 + rng.normal(scale=np.sqrt(4*(vh/2)/2),size=n)*0 + rng.normal(scale=np.sqrt(2*(vh/2)),size=n) + fn2
    return B*(nrm-dot), dot
def boot_ci(num,den,B=1152,nb=2000):
    n=len(num); idx=rng.integers(0,n,size=(nb,n))
    r=num[idx].sum(1)/den[idx].sum(1)
    return np.percentile(r,[2.5,50,97.5])
print(f"  d_eff={d:,}  B=1152。每格 = 中位估计 [95%CI],以及 hi/lo")
print(f"  {'真值':>7} |"+"".join(f"{'n=%d'%n:>26}" for n in (25,100,400)))
for t in (400,835,1152,2500,4131):
    row=""
    for n in (25,100,400):
        num,den=sim_updates(t,1152,n); lo,md,hi=boot_ci(num,den)
        row+=f"  {md:>6.0f} [{lo:>5.0f},{hi:>6.0f}] {hi/max(lo,1):>4.1f}"
    print(f"  {t:>7} |{row}")
print()
print("  B-b 门槛 hi/lo<3;B-a 判负需 hi<1152,判正需 lo>2304。")
