#!/usr/bin/env python
"""Checkpoint 表(app:ckpt)。§4.671 的教训:进论文的数必须走生成器,手写表没有来源可追溯。
对照一律取**同 ckpt、同硬件**的 canonical;缺的格留 --,不跨 ckpt 凑。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
CKS = [20, 30, 50, 80, 100]
def cell(arm, ctl, p, n=4):
    fa = "matched_pi_%s_default_%s_flow%d" % (arm, p, n)
    fb = "matched_pi_%s_default_%s_flow%d" % (ctl, p, n)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE
# (显示名, arm 前缀, 每个 ck 的同硬件对照前缀, 备注)
ROWS = [(r"$\mathcal K=\{3,6\}$ (default)", "multikF4", {20:"baseline_20",30:"baseline_30",50:"baseline_50",80:"baseline_80",100:"baseline_100"}, "H200, np4"),
        (r"$\mathcal K=\{5,6,7\}$",         "multik4",  {20:"baserep_20", 30:"baserep_30", 50:"baserep_50", 80:"baserep_80", 100:"baseline_100"}, "A100, np3")]
out = ["% 自动生成,勿手改。源:paper_figs/make_table_ckpt.py",
       "\\begin{center}", "\\resizebox{0.98\\linewidth}{!}{%",
       "\\begin{tabular}{ll ccccc c ccccc}", "\\toprule",
       "& & \\multicolumn{5}{c}{\\textbf{NFE}$=$\\textbf{4}} & & \\multicolumn{5}{c}{\\textbf{NFE}$=$\\textbf{40}} \\\\",
       "\\cmidrule(lr){3-7}\\cmidrule(lr){9-13}",
       "$\\mathcal K$ & set & " + " & ".join("$%d$" % c for c in CKS) + " & & " +
       " & ".join("$%d$" % c for c in CKS) + "\\\\", "\\midrule"]
print("  %-24s %-10s %s | %s" % ("K","set","  ".join("%7s"%("ck%d"%c) for c in CKS),
                                 "  ".join("%7s"%("ck%d"%c) for c in CKS)))
for nm, arm, ctls, note in ROWS:
    for p, plab in (("drawbench", "DrawBench"), ("unseen1k", "Unseen-1k")):
        cs4, cs40, r4, r40 = [], [], [], []
        for c in CKS:
            ctl = ctls.get(c)
            for nfe, cs, raw in ((4, cs4, r4), (40, cs40, r40)):
                v = cell("%s_%d" % (arm, c), ctl, p, nfe) if ctl else None
                cs.append("--" if v is None else "$%+.2f$" % v)
                raw.append("    --  " if v is None else "%+7.2f" % v)
        out.append("%s & %s & %s & & %s \\\\" % (nm if p == "drawbench" else "", plab,
                                                    " & ".join(cs4), " & ".join(cs40)))
        print("  %-24s %-10s %s | %s" % (nm[:24] if p=="drawbench" else "", plab,
                                         "  ".join(r4), "  ".join(r40)))
    out.append("\\addlinespace[2pt]")
out += ["\\bottomrule", "\\end{tabular}}", "\\end{center}"]
open("paper_figs/table_ckpt.tex", "w").write("\n".join(out) + "\n")
print("  ✅ paper_figs/table_ckpt.tex")
