#!/usr/bin/env python
"""Reward-specific 块:复刻官方 tab:main_results 的 \cmark 行口径 ——
**每一列由"在该 reward 上训练的 checkpoint"填**(官方 app:main_result_details 明确)。
协议:DrawBench · 40 步 flow · guidance 1.0 · 512 · n=1000 · 100 updates · β=1.0。
缺的臂留 --,不编数、不跨 reward 借数。"""
import json, os
SEED = "20260830"
REWARDS = [("hpsv2","HPSv2.1",3), ("pickscore","Pick",2), ("clipscore","CLIP",3)]
def score(arm_reward, which_arm, evaluator):
    """arm_reward 上训练的 which_arm 臂,在 evaluator 这一列上的分数。
    tag 必须与 watch_matched_ckpt.sh 挂的 tag 一致:rs{reward}{canon|msp}_{ck}。"""
    tag = "rs%s%s_100" % (arm_reward, "canon" if which_arm=="baseline" else "msp")
    for cand in ("matched_pi_%s_default_drawbench_flow40.json" % tag,):
        f = "outputs/" + cand
        if os.path.exists(f):
            d = json.load(open(f)); e = d["results"][0] if "results" in d else d
            return e["score_stats"].get(evaluator, {}).get("mean")
    return None
ROWS = [("SD3.5-M w/o CFG", None),
        ("$+$ \\opsd (reward-specific)", "baseline"),
        ("$+$ \\MSP (reward-specific)",  "multikF4")]
OFF = {"SD3.5-M w/o CFG": dict(hpsv2=0.204, pickscore=20.51, clipscore=0.237),
       "$+$ \\opsd (reward-specific)": dict(hpsv2=0.390, pickscore=24.94, clipscore=0.340)}
def base(evaluator):
    f = "outputs/matched_dflt_basemodel_none_drawbench_flow40.json"
    if not os.path.exists(f): return None
    d = json.load(open(f)); e = d["results"][0] if "results" in d else d
    return e["score_stats"].get(evaluator, {}).get("mean")
out = ["% 自动生成,勿手改。源:paper_figs/make_table_rewardspec.py",
       "\\begin{center}", "\\begin{tabular}{l %s}" % ("r"*len(REWARDS)), "\\toprule",
       "\\textbf{Model} & " + " & ".join("\\textbf{%s}"%r[1] for r in REWARDS) + "\\\\",
       "\\multicolumn{%d}{l}{\\footnotesize each column is filled by the checkpoint trained on that evaluator}\\\\" % (1+len(REWARDS)),
       "\\midrule"]
print("  %-32s %s" % ("row", " ".join("%10s"%r[1] for r in REWARDS)))
for nm, arm in ROWS:
    cs, raw = [], []
    for r, lab, dec in REWARDS:
        v = base(r) if arm is None else score(r, arm, r)
        cs.append("--" if v is None else "$%.*f$" % (dec, v))
        raw.append("    --    " if v is None else "%10.*f" % (dec, v))
    out.append("%s & %s\\\\" % (nm, " & ".join(cs)))
    print("  %-32s %s" % (nm.replace("\\opsd","OPSD").replace("\\MSP","MSP")[:32], " ".join(raw)))
out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
open("paper_figs/table_rewardspec.tex","w").write("\n".join(out)+"\n")
print("\n  官方报告(参考): OPSD reward-specific @100 = HPSv2 0.390 / Pick 24.94 / CLIP 0.340")
print("  ✅ paper_figs/table_rewardspec.tex")
