import numpy as np
rng=np.random.default_rng(11); d=800
def make(N,mode):
    G=np.zeros(d); G[0]=1.0
    if mode=="pure_scale":       w=rng.lognormal(0,0.8,N); a=0.15*w; s=w
    elif mode=="align_with_norm":u=rng.lognormal(0,0.8,N); a=0.15*u; s=np.ones(N)
    else:                        a=0.15*np.ones(N); s=np.ones(N)
    return a[:,None]*G + s[:,None]*rng.normal(scale=1/np.sqrt(d),size=(N,d))
def truth(g): return float(np.linalg.norm(g.mean(0))**2/(np.linalg.norm(g,axis=1)**2).mean())
def rho_cos(g,m=100000):
    n=np.linalg.norm(g,axis=1); gh=g/n[:,None]
    i=rng.integers(0,len(g),m); j=rng.integers(0,len(g),m); k=i!=j
    return float((gh[i[k]]*gh[j[k]]).sum(1).mean())
def recon(g,ns=6,reps=4000):
    N=len(g); idx=rng.integers(0,N,size=(reps,ns)); sub=g[idx]
    nr=np.linalg.norm(sub,axis=2); h=ns//2
    A=sub[:,:h].mean(1); B=sub[:,h:].mean(1)
    return float((A*B).sum(1).mean()/(nr**2).mean())
print(f"  {'情形':<18}{'真值':>9}{'余弦法':>10}{'偏差':>8}{'cos+范数':>10}{'偏差':>8}")
for mode,lbl in [("homog","同质(对照)"),("pure_scale","纯尺度异质"),("align_with_norm","大范数更对齐")]:
    g=make(6000,mode); t=truth(g); r=rho_cos(g); c=recon(g)
    print(f"  {lbl:<18}{t:>9.4f}{r:>10.4f}{r/t-1:>+7.0%}{c:>10.4f}{c/t-1:>+7.0%}")
