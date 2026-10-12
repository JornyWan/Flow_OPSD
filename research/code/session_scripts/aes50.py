import sys, importlib.util, statistics as st
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
def f(t,p,n): return f"matched_pi_{t}_default_{p}_flow{n}"
print("Aesthetic 臂 @ck50(⚠️ 判定点是 ck100,§4.743 已证 ck50 会误判 ⇒ 不判)")
print(f"{'set':<11}{'nfe':<5}{'MSP':>9}{'canon':>9}{'Δ':>10}{'t':>8}{'win%':>7}")
res={}
for p in ("drawbench","unseen1k"):
    for n in (4,40):
        A=J.load(f("rsaestheticmsp_50",p,n),"aesthetic")
        B=J.load(f("rsaestheticcanon_50",p,n),"aesthetic")
        r=J.paired(A,B); res[(p,n)]=r
        print(f"{p:<11}{n:<5}{st.mean(A.values()):>9.4f}{st.mean(B.values()):>9.4f}{r[1]:>+10.4f}{r[3]:>+8.2f}{r[4]:>7.1f}")
print()
print("少步缺口(NFE40 − NFE4):")
for p in ("drawbench","unseen1k"):
    v={}
    for lab,t in (("MSP","rsaestheticmsp_50"),("canon","rsaestheticcanon_50")):
        a40=J.load(f(t,p,40),"aesthetic"); a4=J.load(f(t,p,4),"aesthetic")
        ks=sorted(set(a40)&set(a4)); v[lab]=sum(a40[k]-a4[k] for k in ks)/len(ks)
    print(f"  {p:<10} canon={v['canon']:+.4f}  MSP={v['MSP']:+.4f}  比值={v['MSP']/v['canon']:.3f}")
print()
print("参考:canon 侧 ck50 → ck100 自己变了多少(看 ck50 是否稳定)")
for p in ("drawbench","unseen1k"):
    for n in (4,40):
        a=J.load(f("rsaestheticcanon_50",p,n),"aesthetic"); b=J.load(f("rsaestheticcanon_100",p,n),"aesthetic")
        r=J.paired(b,a)
        print(f"  canon {p:<10} flow{n:<3} ck100−ck50 = {r[1]:+.4f} (t={r[3]:+.2f})")
