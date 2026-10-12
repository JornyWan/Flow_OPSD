import sys, importlib.util, os
ARGS=list(sys.argv[1:])
spec=importlib.util.spec_from_file_location("judge", os.path.join(os.path.dirname(os.path.abspath(__file__)),"judge.py"))
J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]                       # 阻止 judge.py 的 main 跑
try: spec.loader.exec_module(J)
except SystemExit: pass
MDE=J.MDE
def go(a,b,lbl):
    print(f"  {lbl}")
    for ax in ("open3","imagereward","aesthetic","pickscore","clipscore","hpsv2"):
        try: n,m,se,t,w=J.paired(J.load(f"matched_pi_{a}_default_drawbench_flow40",ax),
                                 J.load(f"matched_pi_{b}_default_drawbench_flow40",ax))
        except Exception as e: print(f"    {ax:<12} 读失败 {e}"); continue
        ex=f"  {m/MDE:+8.2f}×MDE 胜率={w:5.1f}%" if ax=="open3" else ""
        print(f"    {ax:<12} Δ={m:+.5f} ±{se:.5f} t={t:+8.2f}{ex}  n={n}")
for _s in ARGS:
    a,b,l=_s.split(":",2)
    go(a,b,l)
