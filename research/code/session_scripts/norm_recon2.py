import numpy as np
rng=np.random.default_rng(11); d=4000
def make(N,mode):
    G=np.zeros(d); G[0]=1.0
    if mode=="pure_scale":       w=rng.lognormal(0,0.8,N); a=0.05*w; s=w
    elif mode=="align_with_norm":u=rng.lognormal(0,0.8,N); a=0.05*u; s=np.ones(N)
    else:                        a=0.05*np.ones(N); s=np.ones(N)
    return a[:,None]*G + s[:,None]*rng.normal(scale=1/np.sqrt(d),size=(N,d))
def truth(g): return float(np.linalg.norm(g.mean(0))**2/(np.linalg.norm(g,axis=1)**2).mean())
def rho_cos(g,m=200000):
    n=np.linalg.norm(g,axis=1); gh=g/n[:,None]
    i=rng.integers(0,len(g),m); j=rng.integers(0,len(g),m); k=i!=j
    return float((gh[i[k]]*gh[j[k]]).sum(1).mean())
def recon(g,ns=6,reps=30000):
    """向量化:只用 cos+范数 -> 3v3 半批内积(‖G‖² 无偏),再除 E‖g‖²"""
    N=len(g); idx=rng.integers(0,N,size=(reps,ns))
    sub=g[idx]                                    # (reps,ns,d)
    nr=np.linalg.norm(sub,axis=2)                 # (reps,ns)  <- 探针记的范数
    h=ns//2
    A=sub[:,:h].mean(1); B=sub[:,h:].mean(1)
    num=(A*B).sum(1).mean()                       # 等价于用 cos*范数重建内积后求两半均值内积
    den=(nr**2).mean()
    return float(num/den)
print(f"  {'情形':<18}{'真值':>10}{'余弦法 ρ':>11}{'偏差':>8}{'cos+范数':>11}{'偏差':>8}")
for mode,lbl in [("homog","同质(对照)"),("pure_scale","纯尺度异质"),("align_with_norm","大范数更对齐")]:
    g=make(12000,mode); t=truth(g)
    print(f"  {lbl:<18}{t:>10.5f}{rho_cos(g):>11.5f}{rho_cos(g)/t-1:>+7.0%}{recon(g):>11.5f}{recon(g)/t-1:>+7.0%}")
