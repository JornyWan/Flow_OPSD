import numpy as np
rng=np.random.default_rng(0); d=400
def est_fast(Bn,B,reps=200000,chunk=20000):
    vh=(Bn/d)*2.0/B; s=np.sqrt(vh); num=den=0.0
    for i in range(0,reps,chunk):
        n=min(chunk,reps-i)
        A=rng.normal(scale=s,size=(n,d)); A[:,0]+=1.0
        Bh=rng.normal(scale=s,size=(n,d)); Bh[:,0]+=1.0
        f=(A+Bh)*0.5; dot=(A*Bh).sum(1)
        num+=((f*f).sum(1)-dot).sum(); den+=dot.sum()
    return B*num/den
def est_full(Bn,B,reps=400):
    sd=np.sqrt(Bn/d); num=den=0.0
    for _ in range(reps):
        g=rng.normal(scale=sd,size=(B,d)); g[:,0]+=1.0; h=B//2
        A=g[:h].mean(0); Bh=g[h:].mean(0); f=g.mean(0)
        dot=A@Bh; num+=f@f-dot; den+=dot
    return B*num/den
print("(a) 逐样本全模拟(验证半批均值假设本身)")
print(f"  {'真值':>7} |"+"".join(f"{'B=%d'%B:>9}" for B in (48,192)))
for t in (100,661,1152,4000):
    print(f"  {t:>7} |"+"".join(f"{est_full(t,B):>9.0f}" for B in (48,192)))
print("\n(b) 向量化,含实际配置 B=1152 与 4× 的 4608")
print(f"  {'真值':>7} |"+"".join(f"{'B=%d'%B:>9}" for B in (48,192,1152,4608)))
for t in (100,464,835,1152,4131):
    print(f"  {t:>7} |"+"".join(f"{est_fast(t,B):>9.0f}" for B in (48,192,1152,4608)))
print("\n(c) 零模型 G=0(无真信号)⇒ dot≈0,估计应发散/不稳")
for B in (192,1152):
    s=np.sqrt(2.0/B); num=den=0.0
    for i in range(10):
        A=rng.normal(scale=s,size=(20000,d)); Bh=rng.normal(scale=s,size=(20000,d))
        f=(A+Bh)*0.5; dot=(A*Bh).sum(1); num+=((f*f).sum(1)-dot).sum(); den+=dot.sum()
    print(f"    B={B:>5}: dot均值={den/200000:+.6f} (应≈0)  B_noise_hat={B*num/den:,.0f}")
print("\n判据:(a)(b) 应在真值 ±5% 内且不随 B 漂移;(c) dot≈0 且估计爆掉。")
