# MSP(c) − canonical(c') 交叉 ck 配对矩阵(×MDE),检验"MSP 的 40 步领先 = 同轨迹提前"假设
import sys, os, importlib.util
SP=os.path.dirname(os.path.abspath(__file__))
spec=importlib.util.spec_from_file_location("judge", SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
_a=sys.argv; sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv=_a
CK=(100,150,200,250,300)
n=int(sys.argv[1]) if len(sys.argv)>1 else 40
for seed,(m,c) in {"s0830":("o3msp300","o3canon300"),"s0905":("msp300s2","canon300s2")}.items():
  for p in ("drawbench","unseen1k"):
    L=lambda t,k: J.load(f"matched_pi_{t}_{k}_default_{p}_flow{n}","open3")
    print(f"\n{seed} {p} f{n}: 行 = MSP ck,列 = canonical ck;格 = MSP − canonical ×MDE(t)")
    print("        "+"".join(f"  canon{k:<9d}" for k in CK))
    for a in CK:
      row=[]
      for b in CK:
        r=J.paired(L(m,a),L(c,b)); row.append("%+6.2f(t%+5.1f)"%(r[1]/J.MDE,r[3]))
      print(f"MSP{a:<4d} "+"  ".join(row))
    print("canonical 逐段增量:", "  ".join("%d→%d %+.2f"%(CK[i],CK[i+1],J.paired(L(c,CK[i+1]),L(c,CK[i]))[1]/J.MDE) for i in range(4)))
    print("MSP 逐段增量:      ", "  ".join("%d→%d %+.2f"%(CK[i],CK[i+1],J.paired(L(m,CK[i+1]),L(m,CK[i]))[1]/J.MDE) for i in range(4)))
