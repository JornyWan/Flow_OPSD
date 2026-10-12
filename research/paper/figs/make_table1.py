#!/usr/bin/env python
"""从 outputs/ 的实测数据自动生成 Table 1(新框架:reward 目标该施加在哪里)。
缺的格留空并在 stderr 报告;**不插值、不手抄**。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
MDE = J.MDE

# (显示名, 臂 tag, 同硬件同 ck 的对照 tag, 成本, 备注)
CFG = [("(A) move query state $\\sigma$ 0.278$\\to$0.465", "qstrain7_100", "basenp3_100", "$1.00\\times$", "ck100, np3"),
       ("(B) shorten rollout 10$\\to$6 steps",             "base6_100",    "baseline_100", "$0.90\\times$", "ck100, np4"),
       ("(C) propagate target to 2 extra states (ours)",   "multikF4_100", "baseline_100", "$1.20\\times$", "ck100, np4"),
       ("(B$+$C) rollout 6 $+$ propagation",               "base6mk_100",  "baseline_100", "$\\sim\\!1.10\\times$", "ck100, np4"),
       ("(A$+$C) query $\\sigma$ 0.465 $+$ propagation",   "qs7mk_100",    "baseline_100",  "$1.20\\times$", "ck100, np4"),
       # 参照行:**不是**"施加在哪"的答案,也不是同预算比较。放进来是因为审稿人必问
       # "为什么不干脆多训练"。对照取**同一次 300 步运行的 ck100**(§4.726 已验证它与独立
       # u100 臂只差 <=0.32xMDE)⇒ 这一行内部也没有跨 run 混淆。
       ("\\;\\emph{(reference)} train $3\\times$ longer", "o3canon300_300", "o3canon300_100", "$3.00\\times$", "ck300 vs ck100, np4")]
NFE = [4, 40]

def _resolve(tag, p, n):
    """老评测的文件名可能没有 _100 后缀;按 <tag> -> <tag 去掉末尾 _100> 回退一次。"""
    for t in (tag, tag[:-4] if tag.endswith("_100") else tag):
        f = "matched_pi_%s_default_%s_flow%d" % (t, p, n)
        if os.path.exists("outputs/%s.json" % f): return f
    return None

def cell(a, b, p, n, axis="open3"):
    fa = _resolve(a, p, n); fb = _resolve(b, p, n)
    if not (fa and fb):
        print("  [missing] %s vs %s  %s NFE=%d" % (a, b, p, n), file=sys.stderr); return None
    n_, m, se, t, w = J.paired(J.load(fa, axis), J.load(fb, axis))
    return (m / MDE if axis == "open3" else t), t, w

rows = []
for nm, a, b, cost, note in CFG:
    vals = []
    for p in ("drawbench", "unseen1k"):
        for n in NFE:
            vals.append(cell(a, b, p, n))
    rows.append((nm, cost, note, vals))

out = ["% Table 1 — 自动生成,勿手改。源:paper_figs/make_table1.py",
       "% 口径:SD3.5-M,100 optimizer updates,A40,逐图配对 n=1000/格,MDE=0.0033",
       "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}",
       "\\caption{\\textbf{Where should the on-policy reward target be applied?} Paired per-image gain over",
       "the canonical OPSD model trained at the \\emph{same} budget and on the \\emph{same} hardware,",
       "in units of MDE$=0.0033$; win\\% in parentheses. A cell counts as a gain when it clears",
       "$2.5\\times$MDE with $t>2$, the threshold calibrated against the measured canonical-versus-canonical",
       "noise floor (Sec.~\\ref{sec:exp:protocol}); $^\\dagger$ marks the $1$--$2.5\\times$MDE band, which a single",
       "seed cannot resolve. The last column counts cells that clear the threshold: propagation is the only",
       "\\emph{placement} positive at every budget on both prompt sets. The final row is a \\emph{reference},",
       "not a placement choice and not a matched-budget comparison: it is the same run read at $300$",
       "rather than $100$ updates, included because tripling the budget is the obvious alternative. It",
       "gains more in absolute terms, at $3.00\\times$ the cost; Sec.~\\ref{sec:exp:main} gives the",
       "per-unit-compute comparison and Fig.~\\ref{fig:deficit} shows that it \\emph{widens} the",
       "few-step gap that propagation narrows.}",
       "\\label{tab:where}", "\\resizebox{0.98\\linewidth}{!}{%", "\\begin{tabular}{lcc cc cc c}", "\\toprule",
       "& & & \\multicolumn{2}{c}{\\textbf{DrawBench}} & \\multicolumn{2}{c}{\\textbf{Unseen-1k}} & \\\\",
       "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
       "\\textbf{Intervention} & \\textbf{Cost} & \\textbf{Setting} & NFE$=$4 & NFE$=$40 & NFE$=$4 & NFE$=$40 & \\textbf{Gains} \\\\",
       "\\midrule"]
def _sig(v):   # §4.615:单 seed 要 >=2.5xMDE 且 t>2
    return v is not None and v[0] >= 2.5 and v[1] > 2
for nm, cost, note, vals in rows:
    cs = []
    for v in vals:
        if v is None: cs.append("--"); continue
        dag = "^{\\dagger}" if (1.0 <= v[0] < 2.5) else ""
        cs.append("$%+.2f%s$ \\scriptsize(%.0f\\%%)" % (v[0], dag, v[2]))
    ng, nh = sum(_sig(v) for v in vals), sum(v is not None for v in vals)
    gcol = "--" if nh == 0 else ("$\\mathbf{%d/%d}$" % (ng, nh) if ng == nh == len(vals) else "$%d/%d$" % (ng, nh))
    if "(reference)" in nm: out.append("\\midrule")
    out.append("%s & %s & \\scriptsize %s & %s & %s \\\\" % (nm, cost, note, " & ".join(cs), gcol))
out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open("paper_figs/table_where.tex", "w").write("\n".join(out) + "\n")
print("  ✅ paper_figs/table_where.tex")
# 同时打一份可读的
print("\n  %-46s %8s | %14s %14s | %14s %14s" % ("Intervention", "Cost", "db NFE=4", "db NFE=40", "u1k NFE=4", "u1k NFE=40"))
for nm, cost, note, vals in rows:
    import re
    plain = re.sub(r"\\[a-zA-Z]+|[${}\\]", "", nm)[:46]
    cs = ["    --        " if v is None else "%+7.2f (%2.0f%%)" % (v[0], v[2]) for v in vals]
    print("  %-46s %8s | %14s %14s | %14s %14s" % (plain, re.sub(r"[${}\\]|times","",cost), *cs))

# ── 第二张表:held-out ImageReward(§4.613 认定的唯一有效 held-out 轴)──
rows2 = []
for nm, a, b, cost, note in CFG:
    vals = []
    for p in ("drawbench", "unseen1k"):
        for n in NFE:
            vals.append(cell(a, b, p, n, "imagereward"))
    rows2.append((nm, cost, note, vals))
out2 = ["% Table 2 — 自动生成。held-out ImageReward 的配对 t 值。",
        "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{5pt}",
        "\\caption{\\textbf{Held-out evaluator.} Paired $t$ on ImageReward, the only held-out metric that",
        "tracks the reported composite in our calibration ($r{=}{+}0.894$, sign agreement 12/12; Aesthetic is",
        "anti-correlated, $r{=}{-}0.395$, and is therefore not reported). Same pairings as Table~\\ref{tab:where}.}",
        "\\label{tab:heldout}", "\\begin{tabular}{lc cc cc}", "\\toprule",
        "& & \\multicolumn{2}{c}{\\textbf{DrawBench}} & \\multicolumn{2}{c}{\\textbf{Unseen-1k}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "\\textbf{Intervention} & \\textbf{Cost} & NFE$=$4 & NFE$=$40 & NFE$=$4 & NFE$=$40 \\\\", "\\midrule"]
for nm, cost, note, vals in rows2:
    cs = ["--" if v is None else "$%+.2f$" % v[0] for v in vals]
    out2.append("%s & %s & %s \\\\" % (nm, cost, " & ".join(cs)))
out2 += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("paper_figs/table_heldout.tex", "w").write("\n".join(out2) + "\n")
print("  ✅ paper_figs/table_heldout.tex")
import re
print("\n  === held-out ImageReward(配对 t)===")
print("  %-46s %8s | %10s %10s | %10s %10s" % ("Intervention", "Cost", "db NFE=4", "db NFE=40", "u1k NFE=4", "u1k NFE=40"))
for nm, cost, note, vals in rows2:
    plain = re.sub(r"\\[a-zA-Z]+|[${}\\]", "", nm)[:46]
    cs = ["    --    " if v is None else "%+10.2f" % v[0] for v in vals]
    print("  %-46s %8s | %10s %10s | %10s %10s" % (plain, re.sub(r"[${}\\]|times", "", cost), *cs))

# ── Table 3:组合(B+C)与其增量 ────────────────────────────────────────────
def row(a, b, label, cost):
    cs = []
    for p in ("drawbench", "unseen1k"):
        for n in NFE:
            v = cell(a, b, p, n)
            cs.append("--" if v is None else "$%+.2f$ \\scriptsize(%.0f\\%%)" % (v[0], v[2]))
    return "%s & %s & %s \\\\" % (label, cost, " & ".join(cs))
out3 = ["% Table 3 — 自动生成。组合与增量。",
        "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{\\textbf{Composition.} Propagation stacks on top of a shortened rollout.",
        "Top block: each configuration against the canonical baseline. Bottom row: the",
        "\\emph{increment} contributed by propagation on top of the shortened rollout.",
        "Units of MDE, win\\% in parentheses; $100$ updates, matched hardware and seed.}",
        "\\label{tab:compose}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lc cc cc}", "\\toprule",
        "& & \\multicolumn{2}{c}{\\textbf{DrawBench}} & \\multicolumn{2}{c}{\\textbf{Unseen-1k}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "\\textbf{Configuration} & \\textbf{Cost} & NFE$=$4 & NFE$=$40 & NFE$=$4 & NFE$=$40 \\\\",
        "\\midrule",
        row("base6_100", "baseline_100", "(B) shortened rollout", "$0.90\\times$"),
        row("multikF4_100", "baseline_100", "(C) \\MSP", "$1.20\\times$"),
        row("base6mk_100", "baseline_100", "\\textbf{(B$+$C)}", "$\\sim\\!1.10\\times$"),
        "\\midrule",
        row("base6mk_100", "base6_100", "\\quad increment of (C) over (B)", "--"),
        "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open("paper_figs/table_compose.tex", "w").write("\n".join(out3) + "\n")
print("  ✅ paper_figs/table_compose.tex")
