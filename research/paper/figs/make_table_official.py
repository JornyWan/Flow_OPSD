#!/usr/bin/env python
"""主表:官方口径的 SD3.5-M 块。
上半 = **继承自 \citep{opsd2026} 的报告值**(含其全部 baseline),逐个标 source;
下半 = **我们自己在同协议下跑出来的**。
协议:DrawBench · 40 步 flow · guidance 1.0 · 512 · n=1000 · checkpoint EMA(§4.692 逐项核过)。
继承值抄自 reference_papers/arXiv-2608.24646v1/tables/main_results.tex,逐行核对过。"""
import json, os
def st(tag, adapter="default"):
    f = "outputs/matched_%s%s_%s_drawbench_flow40.json" % ("" if tag.startswith("dflt") else "pi_", tag, adapter)
    if not os.path.exists(f): return None
    d = json.load(open(f)); e = d["results"][0] if "results" in d else d
    return {k: v["mean"] for k, v in e["score_stats"].items()}
COLS = [("pickscore","Pick",2), ("clipscore","CLIP",3), ("hpsv2","HPSv2.1",3),
        ("aesthetic","Aes",2), ("imagereward","ImgR",2)]
def R(p,c,h,a,i): return dict(pickscore=p, clipscore=c, hpsv2=h, aesthetic=a, imagereward=i)
# 继承块:name, reward-specific 标记, updates, 值
INH = [
 ("\\emph{Reference backbones}", None, None, None),
 ("\\quad SDXL",                 "\\xmark", "--",   R(22.42,0.287,0.280,5.60,0.76)),
 ("\\quad SD3.5-L",              "\\xmark", "--",   R(22.91,0.289,0.288,5.50,0.96)),
 ("\\quad FLUX.1-dev",           "\\xmark", "--",   R(22.84,0.295,0.274,5.71,0.96)),
 ("MIDRULE", None, None, None),
 ("\\emph{SD3.5-M, one checkpoint across all columns}", None, None, None),
 ("\\quad SD3.5-M w/o CFG",      "\\xmark", "--",   R(20.51,0.237,0.204,5.13,-0.58)),
 ("\\quad $+$ CFG",              "\\xmark", "--",   R(22.34,0.285,0.279,5.36,0.85)),
 ("\\quad $+$ FlowGRPO",         "\\xmark", "4k",   R(23.50,0.280,0.316,5.90,1.29)),
 ("\\quad $+$ DiffusionNFT",     "\\xmark", "300",  R(23.62,0.294,0.340,6.02,1.45)),
 ("\\quad $+$ DanceOPD",         "\\xmark", "300",  R(23.06,0.272,0.322,5.98,1.23)),
 ("\\quad $+$ DiffusionOPD",     "\\xmark", "300",  R(23.05,0.269,0.322,6.04,1.23)),
 ("\\quad $+$ FlowOPD",          "\\xmark", "300",  R(22.62,0.263,0.300,5.98,0.92)),
 ("\\quad $+$ \\opsd",           "\\xmark", "300",  R(25.51,0.333,0.389,6.03,1.51)),
 ("MIDRULE", None, None, None),
 ("\\emph{SD3.5-M, one checkpoint per evaluator}", None, None, None),
 ("\\quad $+$ ReFL",             "\\cmark", "100",  R(23.92,0.308,0.358,12.09,1.28)),
 ("\\quad $+$ DiffusionNFT",     "\\cmark", "100",  R(23.43,0.298,0.336,9.11,1.46)),
 ("\\quad $+$ \\opsd",           "\\cmark", "100",  R(24.94,0.340,0.390,12.08,1.76)),
]
OURS = [
 ("\\quad SD3.5-M w/o CFG",      "\\xmark", "--",   "dflt_basemodel", "none"),
 ("\\quad $+$ \\opsd (repro)",   "\\xmark", "100",  "baseline_100",   "default"),
 ("\\quad $+$ \\opsd (repro)",   "\\xmark", "300",  "o3canon300_300", "default"),
 ("\\quad $+$ \\MSP (ours)",     "\\xmark", "100",  "multikF4_100",   "default"),
 ("\\quad $+$ \\MSP (ours)",     "\\xmark", "300",  "o3msp300_300",   "default"),
 ("MIDRULE", None, None, None, None),
 ("\\emph{SD3.5-M, one checkpoint per evaluator}", None, None, None, None),
 ("\\quad $+$ \\opsd (repro)",   "\\cmark", "100",  {"pickscore":"rspickscorecanon_100",
    "clipscore":"rsclipscorecanon_100", "hpsv2":"rshpsv2canon_100",
    "aesthetic":"rsaestheticcanon_100", "imagereward":"rsimagerewardcanon_100"}, "default"),
 ("\\quad $+$ \\MSP (ours)",     "\\cmark", "100",  {"pickscore":"rspickscoremsp_100",
    "clipscore":"rsclipscoremsp_100", "hpsv2":"rshpsv2msp_100",
    "aesthetic":"rsaestheticmsp_100", "imagereward":"rsimagerewardmsp_100"}, "default"),
]
out = ["% 自动生成,勿手改。源:paper_figs/make_table_official.py",
       "\\begin{table}[t]", "\\centering", "\\setlength{\\tabcolsep}{4pt}",
       "\\caption{\\textbf{SD3.5-M under the published protocol.} DrawBench, $40$-step flow,",
       "guidance $1.0$, $512^2$, $n{=}1000$, checkpoint-averaged parameters. The upper block is",
       "reproduced verbatim from \\citet{opsd2026}; we did not rerun those baselines. The lower",
       "block is ours under the same protocol. \\cmark: one checkpoint per evaluator column;",
       "\\xmark: a single checkpoint evaluated across all columns. Higher is better.}",
       "\\label{tab:official}", "\\resizebox{0.98\\linewidth}{!}{%",
       "\\begin{tabular}{lcc %s}" % ("r"*len(COLS)), "\\toprule",
       "\\textbf{Model} & \\textbf{R.-spec.} & \\textbf{Upd.} & " +
       " & ".join("\\textbf{%s}"%c[1] for c in COLS) + "\\\\", "\\midrule",
       "\\multicolumn{%d}{l}{\\textbf{Reported by \\citet{opsd2026}}}\\\\" % (3+len(COLS)), "\\midrule"]
def fmt(v):
    return " & ".join("--" if (v is None or v.get(k) is None) else "$%.*f$"%(d,v[k]) for k,_,d in COLS)
def percol(tagmap, ad="default"):
    """reward-specific 行:每列取**在该列 reward 上训练**的那个臂。缺的留空,不跨 reward 借数。"""
    v = {}
    for k,_,_ in COLS:
        t = tagmap.get(k)
        if not t: v[k] = None; continue
        d = st(t, ad); v[k] = None if d is None else d.get(k)
    return v
print("  %-50s %-8s %5s %s" % ("row","r-spec","upd"," ".join("%9s"%c[1] for c in COLS)))
for nm, rs, upd, v in INH:
    if nm=="MIDRULE": out.append("\\midrule"); continue
    if v is None: out.append("\\multicolumn{%d}{l}{%s}\\\\"%(3+len(COLS), nm)); print("  %s"%nm); continue
    out.append("%s & %s & %s & %s\\\\" % (nm, rs, upd, fmt(v)))
    print("  %-50s %-8s %5s %s" % (nm.replace("\\quad ","  ")[:50], rs.replace("\\",""), upd,
          " ".join("%9.*f"%(d,v[k]) for k,_,d in COLS)))
out += ["\\midrule", "\\multicolumn{%d}{l}{\\textbf{Ours, same protocol}}\\\\"%(3+len(COLS)), "\\midrule"]
print("  ── Ours ──")
for nm, rs, upd, tag, ad in OURS:
    if nm=="MIDRULE": out.append("\\midrule"); continue
    if rs is None: out.append("\\multicolumn{%d}{l}{%s}\\\\"%(3+len(COLS), nm)); print("  %s"%nm); continue
    v = percol(tag, ad) if isinstance(tag, dict) else st(tag, ad)
    out.append("%s & %s & %s & %s\\\\" % (nm, rs, upd, fmt(v)))
    print("  %-50s %-8s %5s %s" % (nm.replace("\\quad ","  ")[:50], rs.replace("\\",""), upd,
          " ".join(("%9s"%"--") if (v is None or v.get(k) is None) else "%9.*f"%(d,v[k]) for k,_,d in COLS)))
out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
open("paper_figs/table_official.tex","w").write("\n".join(out)+"\n")
print("\n  ✅ paper_figs/table_official.tex")
