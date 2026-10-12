import numpy as np
rng=np.random.default_rng(11); d=4000

def make(N, mode):
    """g_i = a_i*G + s_i*xi_i ,  ‖G‖=1"""
    G=np.zeros(d); G[0]=1.0
    if mode=="pure_scale":      # 范数同时缩放信号与噪声 => 归一化后同质
        w=rng.lognormal(0,0.8,N); a=0.05*w; s=w
    elif mode=="align_with_norm":  # 范数大者**更对齐**(我的假说)
        u=rng.lognormal(0,0.8,N); a=0.05*u; s=np.ones(N)
    else:                        # 同质对照
        a=0.05*np.ones(N); s=np.ones(N)
    g=a[:,None]*G + s[:,None]*rng.normal(scale=1/np.sqrt(d),size=(N,d))
    return g

def truth(g):
    return float(np.linalg.norm(g.mean(0))**2 / (np.linalg.norm(g,axis=1)**2).mean())

def rho_cos(g, m=60000):
    n=np.linalg.norm(g,axis=1); gh=g/n[:,None]
    i=rng.integers(0,len(g),m); j=rng.integers(0,len(g),m); k=i!=j
    return float((gh[i[k]]*gh[j[k]]).sum(1).mean())

def recon_from_probe(g, ns=6, reps=3000):
    """只用探针记录的量:逐样本范数 + 两两余弦。
    在 ns 个样本内做半批(3v3)内积 => ‖G‖² 的无偏估计;再除以均值范数平方。"""
    N=len(g); num=den=0.0
    for _ in range(reps):
        idx=rng.choice(N,ns,replace=False)
        sub=g[idx]; nr=np.linalg.norm(sub,axis=1)
        C=(sub@sub.T)/np.outer(nr,nr)                 # 余弦矩阵(探针记的就是它的上三角)
        IP=C*np.outer(nr,nr)                          # 由 cos+范数重建全部内积
        h=ns//2
        A=IP[:h,:h].sum()/h**2; B=IP[h:,h:].sum()/(ns-h)**2   # 未用,仅示意
        num += IP[:h,h:].sum()/(h*(ns-h))             # 两半均值的内积 = ‖G‖² 的无偏估计
        den += (nr**2).mean()
    return num/den

print(f"  {'情形':<20}{'真值':>10}{'余弦法 ρ':>12}{'偏差':>9}{'cos+范数重建':>14}{'偏差':>9}")
for mode,lbl in [("homog","同质(对照)"),("pure_scale","纯尺度异质"),("align_with_norm","大范数更对齐")]:
    g=make(12000,mode); t=truth(g); r=rho_cos(g); rc=recon_from_probe(g)
    print(f"  {lbl:<20}{t:>10.5f}{r:>12.5f}{r/t-1:>+8.0%}{rc:>14.5f}{rc/t-1:>+8.0%}")
print()
print("  判读:余弦法在同质时无偏;两种异质下**都偏**,方向相反 —— 故不能凭方向猜,必须实测范数。")
print("        'cos+范数'重建在三种情形下都还原真值 ⇒ 只要探针记下范数,就能与半批法对账。")
