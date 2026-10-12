#!/usr/bin/env python
"""主表(绝对数值):骨架照 DiffusionOPD 原文 tab:all_task —— 各方法在各评测轴上的绝对值,
粗体=最优、下划线=次优。缺的行写 placeholder,不编数。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

PSET = "drawbench"
AXES = [("pickscore", "PickScore", 2), ("clipscore", "CLIP", 4),
        ("hpsv2", "HPSv2", 4), ("open3", "Composite", 4), ("imagereward", "ImageReward", 3)]

def mean(tag, adapter, nfe, axis):
    f = "matched_%s_%s_%s_flow%d" % (tag, adapter, PSET, nfe)
    if not os.path.exists("outputs/%s.json" % f): return None
    A = J.load(f, axis); return sum(A.values()) / len(A)

# (显示名, tag, adapter, 备注)  tag=None ⇒ 占位行
ROWS = [
    ("SD3.5-M (no reward training)",        "dflt_basemodel",  "none",    ""),
    ("\\quad + SFT on teacher rollouts",    None,              None,      "not run"),
    ("\\quad + few-step distillation (DMD2/TDM)", None,        None,      "not run"),
    ("\\midrule", None, None, None),
    ("\\quad + OPSD \\citep{opsd2026} \\emph{(prior SOTA)}", "pi_baseline_100", "default", ""),
    ("\\quad\\quad (A) query $\\sigma$ 0.278$\\to$0.465", "pi_qstrain7_100", "default", "np3"),
    ("\\quad\\quad (B) rollout 10$\\to$6",  "pi_base6_100",    "default", ""),
    ("\\quad\\quad \\textbf{(C) $+$ \\MSP (ours)}", "pi_multikF4_100", "default", ""),
    ("\\quad\\quad (B$+$C)",                "pi_base6mk_100",  "default", ""),
]
NFES = [4, 40]

# 收数
data = {}
for nm, tag, ad, note in ROWS:
    if tag is None: continue
    for n in NFES:
        for ax, _, _ in AXES:
            data[(nm, n, ax)] = mean(tag, ad, n, ax)

# 每列的最优/次优(只在真实行里比)
best = {}
for n in NFES:
    for ax, _, _ in AXES:
        vs = sorted([(v, nm) for (nm, nn, aa), v in data.items()
                     if nn == n and aa == ax and v is not None], reverse=True)
        best[(n, ax)] = ([vs[0][1]] if vs else [], [vs[1][1]] if len(vs) > 1 else [])

def fmt(nm, n, ax, dec):
    v = data.get((nm, n, ax))
    if v is None: return "--"
    s = "%.*f" % (dec, v)
    b, sec = best[(n, ax)]
    if nm in b:   return "\\textbf{%s}" % s
    if nm in sec: return "\\underline{%s}" % s
    return s

hdr_ax = " & ".join("\\textbf{%s}" % lab for _, lab, _ in AXES)
out = ["% 主表 — 自动生成,勿手改。源:paper_figs/make_table_main.py",
       "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{3.5pt}",
       "\\caption{\\textbf{Main results on DrawBench} (SD3.5-Medium, $100$ optimizer updates, A40 eval,",
       "$n{=}1000$ prompts). Absolute scores; \\textbf{bold}: best, \\underline{underline}: second best.",
       "Composite $=$ PickScore$/26+$CLIP$+$HPSv2 is the reported metric; ImageReward is the held-out",
       "evaluator and is never trained on. Rows marked ``--'' are baselines we have not run yet.",
       "The (A) row was trained on $3$ processes rather than $4$; the offset this induces is $0.17$ MDE",
       "(App.~\\ref{app:pairing}).}",
       "\\label{tab:main}", "\\resizebox{0.98\\linewidth}{!}{%",
       "\\begin{tabular}{l %s %s}" % ("c" * len(AXES), "c" * len(AXES)), "\\toprule",
       "& \\multicolumn{%d}{c}{\\textbf{NFE}$=$\\textbf{4}} & \\multicolumn{%d}{c}{\\textbf{NFE}$=$\\textbf{40}} \\\\"
       % (len(AXES), len(AXES)),
       "\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}" % (1 + len(AXES), 2 + len(AXES), 1 + 2 * len(AXES)),
       "\\textbf{Method} & %s & %s \\\\" % (hdr_ax, hdr_ax), "\\midrule"]
for nm, tag, ad, note in ROWS:
    if nm == "\\midrule": out.append("\\midrule"); continue
    cells = [fmt(nm, n, ax, dec) for n in NFES for ax, _, dec in AXES]
    out.append("%s & %s \\\\" % (nm, " & ".join(cells)))
out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open("paper_figs/table_main.tex", "w").write("\n".join(out) + "\n")
print("  ✅ paper_figs/table_main.tex")
for nm, tag, ad, note in ROWS:
    if nm == "\\midrule": continue
    print("    %-44s %s" % (nm[:44], "  ".join(
        "%-9s" % (("%.4f" % data[(nm, n, ax)]) if data.get((nm, n, ax)) is not None else "--")
        for n in NFES for ax, _, _ in AXES)))
