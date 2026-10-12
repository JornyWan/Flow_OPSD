#!/usr/bin/env python
"""少步缺口比值表(§4.747):同一模型自己的 40步−4步缺口,MSP ÷ canonical。
无量纲 ⇒ 联合目标与各 reward-specific 设置可**直接并排**。只画已落盘的格。"""
import sys, os, importlib.util, statistics as st
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

def defi(tag, p, ax):
    f4 = "matched_pi_%s_default_%s_flow4" % (tag, p); f40 = "matched_pi_%s_default_%s_flow40" % (tag, p)
    if not (os.path.exists("outputs/%s.json" % f4) and os.path.exists("outputs/%s.json" % f40)): return None
    a4, a40 = J.load(f4, ax), J.load(f40, ax); ks = sorted(set(a4) & set(a40))
    return st.mean(a40[k] - a4[k] for k in ks)

ROWS = [("joint three-reward objective", "baseline_100", "multikF4_100", "open3"),
        ("reward-specific, HPSv2.1", "rshpsv2canon_100", "rshpsv2msp_100", "hpsv2"),
        ("reward-specific, PickScore", "rspickscorecanon_100", "rspickscoremsp_100", "pickscore"),
        ("reward-specific, CLIPScore", "rsclipscorecanon_100", "rsclipscoremsp_100", "clipscore"),
        ("reward-specific, ImageReward", "rsimagerewardcanon_100", "rsimagerewardmsp_100", "imagereward")]
# §4.776:Aesthetic 对**不进**本表 —— 两臂的 open3 复合分都低于未训练底模(四格全败),
# 该配置没在学"质量",缺口比值无从解释。数字留在 EXPERIMENT_LOG §4.776 与正文的说明里,
# 不在这张"方法收窄缺口"的表里并列(并列会读成一个可比的负结果)。
out = ["% 自动生成,勿手改。源:paper_figs/make_table_deficit.py",
       "\\begin{center}", "\\begin{tabular}{lcc}", "\\toprule",
       "& \\multicolumn{2}{c}{\\textbf{four-step deficit, \\MSP $/$ canonical}}\\\\",
       "\\cmidrule(lr){2-3}",
       "\\textbf{training objective} & DrawBench & Unseen-1k\\\\", "\\midrule"]
print("  %-32s %10s %10s" % ("objective", "DrawBench", "Unseen-1k"))
for lab, cn, ms, ax in ROWS:
    cs = []
    for p in ("drawbench", "unseen1k"):
        a, b = defi(cn, p, ax), defi(ms, p, ax)
        cs.append("--" if (a is None or b is None or a == 0) else "$%.2f$" % (b / a))
    bold = all(c != "--" for c in cs) and lab.startswith("joint")
    row = ("\\textbf{%s}" % lab) if bold else lab
    out.append("%s & %s\\\\" % (row, " & ".join(cs)))
    print("  %-32s %10s %10s" % (lab, cs[0].replace("$", ""), cs[1].replace("$", "")))
out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
open("paper_figs/table_deficit.tex", "w").write("\n".join(out) + "\n")
print("\n  ✅ paper_figs/table_deficit.tex")
