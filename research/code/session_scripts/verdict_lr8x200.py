import sys,math,statistics as st,os
sys.path.insert(0,'/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related')
os.chdir('/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related')
from paired_growth import load, obj
MDE=0.0033
def P(a,b):
    A,e1=load(a); B,e2=load(b)
    if e1 or e2: return None
    ks=sorted(set(A)&set(B)); v=[obj(A[k])-obj(B[k]) for k in ks]
    n=len(v); m=st.mean(v); se=st.pstdev(v)*math.sqrt(n/(n-1))/math.sqrt(n)
    return m,se,m/se,sum(1 for x in v if x>0)/n,n
print("="*74)
print("§4.451 臂1 判定:lr8x u200 @ck200(纯 H200 训练,配 H200 基线 base200_200)")
r=P("lr8x200_200_sh","base200_200")
if r is None:
    print("  ⏳ 数据未出"); sys.exit()
m,se,t,w,n=r
print(f"  Δ = {m:+.5f} ± {se:.5f}   t={t:+.2f}   胜率={w:.1%}   n={n}")
BANDS=[("① 登记版(锚 lr8x@ep100=+0.06052,不配对)",0.06052),
       ("② 配对锚点版(§4.469 的 +0.05740)",0.05740),
       ("③ 本臂自身锚点(§4.507 实测 ck100=+0.05614)",0.05614)]
print()
for lab,anchor in BANDS:
    hi=anchor+MDE; lo=anchor-MDE
    cell="A(继续上升)" if m>hi else ("B(饱和)" if m>=lo else "C(回落)")
    print(f"  {lab}")
    print(f"     A > {hi:+.5f} | B [{lo:+.5f},{hi:+.5f}] | C < {lo:+.5f}   ⇒ **{cell}**")
print()
print("  三套锚点是否给出同一格:", end=" ")
cells=set()
for _,a in BANDS:
    cells.add("A" if m>a+MDE else ("B" if m>=a-MDE else "C"))
print("✅ 一致" if len(cells)==1 else f"⚠️ 不一致 {sorted(cells)}")
print()
print("  === 参照 ===")
for lab,a,b in [("base200 ep100→ep200 自身增长","base200_200","base200_100"),
                ("lr8x u200 ck100(§4.507)","lr8x200_100_sh","basenp3")]:
    rr=P(a,b)
    if rr: print(f"    {lab}: {rr[0]:+.5f}")
print("="*74)
