# 等效提前步数:canonical 轨迹上线性插值,找 c' 使 canon(c') = MSP(c);Δ = c' − c(超出 ck300 则按最后一段斜率外推,标 *)
import sys, os, importlib.util
SP=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("judge", SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
_a=sys.argv; sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv=_a
CK=(100,150,200,250,300)
for n in (40,4):
 for seed,(m,c) in {"s0830":("o3msp300","o3canon300"),"s0905":("msp300s2","canon300s2")}.items():
  for p in ("drawbench","unseen1k"):
    L=lambda t,k: J.load(f"matched_pi_{t}_{k}_default_{p}_flow{n}","open3")
    base=L(c,100)
    lc=[J.paired(L(c,k),base)[1]/J.MDE if k!=100 else 0.0 for k in CK]
    lm=[J.paired(L(m,k),base)[1]/J.MDE for k in CK]
    out=[]
    for i,k in enumerate(CK):
      y=lm[i]; s=None
      for j in range(4):
        if lc[j]<=y<=lc[j+1]: s=CK[j]+50*(y-lc[j])/(lc[j+1]-lc[j]); tag=""; break
      if s is None:
        if y>lc[-1]: s=300+50*(y-lc[-1])/(lc[-1]-lc[-2]); tag="*"
        else: s=100+50*(y-lc[0])/(lc[1]-lc[0]); tag="*"
      out.append("ck%d:Δ%+4.0f%s"%(k,s-k,tag))
    print(f"f{n:<2d} {seed} {p:9s} canon 水平 "+"/".join("%.1f"%x for x in lc)+" | MSP "+"/".join("%.1f"%x for x in lm)+" | "+"  ".join(out))
