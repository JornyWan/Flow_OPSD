"""已知答案自检(向量化):trΣ/‖G‖² 已知,验证半批估计量能还原。
B_noise_hat = B*(‖ḡ_full‖² - ḡ_A·ḡ_B)/(ḡ_A·ḡ_B)"""
import numpy as np
rng=np.random.default_rng(0); d=2000

def est(Bn_true,B,reps):
    """逐样本全模拟(慢,验证'半批均值'假设本身)"""
    G=np.zeros(d); G[0]=1.0                       # ‖G‖=1
    sd=np.sqrt(Bn_true/d)                          # trΣ = d*sd² = Bn_true
    num=den=0.0
    for _ in range(reps):
        g=G+rng.normal(scale=sd,size=(B,d)); h=B//2
        A=g[:h].mean(0); Bh=g[h:].mean(0); f=g.mean(0)
        dot=A@Bh; num+=f@f-dot; den+=dot
    return B*num/den

def est_fast(Bn_true,B,reps):
    """半批均值直接采样(等价,快)"""
    G=np.zeros(d); G[0]=1.0
    var_half=(Bn_true/d)*2.0/B                     # 每个半批均值的逐分量方差
    A=G+rng.normal(scale=np.sqrt(var_half),size=(reps,d))
    Bh=G+rng.normal(scale=np.sqrt(var_half),size=(reps,d))
    f=(A+Bh)/2
    dot=(A*Bh).sum(1); num=(f*f).sum(1)-dot
    return B*num.mean()/dot.mean()

print("  === (a) 逐样本全模拟,小 B(验证半批均值假设)===")
print(f"  {'真值':>8} | {'B=48':>8} {'B=192':>8}")
for t in [100,661,1152,4000]:
    print(f"  {t:>8} | "+" ".join(f"{est(t,B,1500):>8.0f}" for B in (48,192)))
print()
print("  === (b) 向量化,含 B=1152(实际训练配置)===")
print(f"  {'真值':>8} | {'B=48':>8} {'B=192':>8} {'B=1152':>8} {'B=4608':>8}")
for t in [100,661,835,1152,4000]:
    row=" ".join(f"{est_fast(t,B,40000):>8.0f}" for B in (48,192,1152,4608))
    print(f"  {t:>8} | {row}")
print()
print("  === (c) 零模型:G=0(纯噪声,无真信号)⇒ B_noise 应发散 ===")
for B in (192,1152):
    A=rng.normal(size=(40000,d))/np.sqrt(B/2); Bh=rng.normal(size=(40000,d))/np.sqrt(B/2)
    f=(A+Bh)/2; dot=(A*Bh).sum(1)
    print(f"    B={B:>5}: dot 均值={dot.mean():+.4f} (应≈0)  B_noise_hat={B*(((f*f).sum(1)-dot).mean()/dot.mean()):,.0f}")
print()
print("  判据:(a)(b) 估计值应在真值 ±5% 内且不随 B 漂移;(c) dot≈0 且估计发散。")
