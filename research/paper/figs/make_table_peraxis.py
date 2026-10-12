#!/usr/bin/env python
"""逐 reward 轴表(app:peraxis)。§4.671/§4.691 的教训:进论文的数必须走生成器。
multikF4_100 vs baseline_100(同硬件同 ckpt),逐图配对 t。"""
import sys, os, importlib.util
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
NFE=[2,4,6,10,20,40]
AX=[("pickscore","PickScore"),("clipscore","CLIP"),("hpsv2","HPSv2"),("imagereward","ImageReward")]
def t(p,n,ax):
    fa="matched_pi_multikF4_100_default_%s_flow%d"%(p,n); fb="matched_pi_baseline_100_default_%s_flow%d"%(p,n)
    if not(os.path.exists("outputs/%s.json"%fa) and os.path.exists("outputs/%s.json"%fb)): return None
    return J.paired(J.load(fa,ax),J.load(fb,ax))[3]
out=["% 自动生成,勿手改。源:paper_figs/make_table_peraxis.py",
     "\\begin{center}","\\small","\\begin{tabular}{llcccccc}","\\toprule",
     "set & axis & "+" & ".join("NFE$=%d$"%n if n==2 else "$%d$"%n for n in NFE)+"\\\\","\\midrule"]
for pi,(p,plab) in enumerate((("drawbench","DrawBench"),("unseen1k","Unseen-1k"))):
    if pi: out.append("\\midrule")
    out.append("\\multirow{%d}{*}{%s}"%(len(AX),plab))
    for ax,alab in AX:
        cs=[("--" if (v:=t(p,n,ax)) is None else "$%+.2f$"%v) for n in NFE]
        out.append(" & %-12s & %s\\\\"%(alab," & ".join(cs)))
out+=["\\bottomrule","\\end{tabular}","\\end{center}"]
open("paper_figs/table_peraxis.tex","w").write("\n".join(out)+"\n")
print("  ✅ paper_figs/table_peraxis.tex")
for p,plab in (("drawbench","DrawBench"),("unseen1k","Unseen-1k")):
    for ax,alab in AX:
        print("    %-10s %-12s %s"%(plab,alab,"  ".join(
            ("   --   " if (v:=t(p,n,ax)) is None else "%+8.2f"%v) for n in NFE)))
