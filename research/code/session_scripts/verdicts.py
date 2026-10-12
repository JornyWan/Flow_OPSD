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

print("="*72)
print("§4.429/§4.470 判定:inner2gacc2 seed-2 @ck400")
print("  登记带 [+0.01830, +0.03810];落 B 格(超可加)需 > +0.03810")
r=P("inner2gacc2s2_400","bases2_100")
if r is None:
    print("  ⏳ 数据未出")
else:
    m,se,t,w,n=r
    cell = "**B(超可加)**" if m>0.03810 else ("A/带内(乘性~可加)" if m>=0.01830 else "C(低于带)")
    print(f"  Δ = {m:+.5f} ± {se:.5f}   t={t:+.2f}   胜率={w:.1%}   n={n}")
    print(f"  ⇒ 落 {cell}   (距上界 {m-0.03810:+.5f} = {(m-0.03810)/MDE:+.2f}×MDE)")
    print(f"  seed-1 对照:+0.03874,落 B 格,超上界 3.1×MDE")
    print("  ⚠️ 限制(§4.470 §三1):gacc2_s2 锚点是 H200 训练配 A100 基线(不配对),")
    print("     真值应更高 ⇒ 真实门槛比 +0.03810 更高。判据不改,但须同报。")
    r2=P("inner2gacc2s2_400","lr8xs2_100")
    if r2: print(f"  §4.478 补行:inner2gacc2_s2 − lr8x_s2 @ep100 = {r2[0]:+.5f}  t={r2[2]:+.2f}  胜率={r2[3]:.1%}")

print("="*72)
print("§4.443/§4.455/§4.469 判定:inner2lrhalf @ck200(基线 baseAA,A100 配对)")
print("  幅度带:C < -0.02834 | A [-0.02834,-0.00854] | B > -0.00854")
print("  形状阈(ep20=+0.01022):步长族 >= +0.00232 | 拟合族 <= -0.01168 | 之间=间隙")
r=P("inner2lrhalf_200","baseAA_100_a40")
if r is None:
    print("  ⏳ 数据未出")
else:
    m,se,t,w,n=r
    amp = "C" if m<-0.02834 else ("A" if m<=-0.00854 else "B")
    shp = "拟合族" if m<=-0.01168 else ("步长族" if m>=0.00232 else "间隙")
    print(f"  Δ = {m:+.5f} ± {se:.5f}   t={t:+.2f}   胜率={w:.1%}   n={n}")
    print(f"  衰减 ep20→100 = {m-0.01022:+.5f}")
    print(f"  ⇒ 幅度格 = {amp}   形状格 = {shp}   ⇒ **交叉格 = {amp} × {shp}**")
    READ={"C × 拟合族":"inner2 在低 lr 下主动有害;形状仍是拟合族",
          "A × 拟合族":"⚠️两判据打架:幅度说纯位移,形状说拟合族签名 —— 以形状为准但必须同报",
          "A × 间隙":"幅度 A、形状不判 ⇒ 不能下 method 结论",
          "B × 间隙":"幅度 B、形状不判 ⇒ 按 §4.452,B 本身不成立证据 ⇒ 不能下 method 结论",
          "B × 步长族":"幅度与形状都指向『只是等效 lr』⇒ 拟合轴是幻象(干净负结论)"}
    print(f"  ⇒ 判读:{READ.get(f'{amp} × {shp}','(未登记组合,须检查)')}")
    lh=P("lrhalf_100","baseAA_100_a40")
    if lh: print(f"  组合贡献 @ep100 = {m-lh[0]:+.5f}(lrhalf@ep100 = {lh[0]:+.5f})")
    print("  §4.479 的 0.5× 行 ep100 由此补齐")
print("="*72)
