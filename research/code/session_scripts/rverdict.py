import json, math, sys, os
O="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs"
MDE=0.0033
def load(tag,p,n):
    f=f"{O}/matched_pi_{tag}_default_{p}_flow{n}.json"
    if not os.path.exists(f): return None
    return json.load(open(f))
def key_of(d):
    # 取每图的主指标(训练目标轴)
    return d
def paired(a,b,metric):
    A={r["prompt"]:r for r in a["rows"]} if "rows" in a else None
    B={r["prompt"]:r for r in b["rows"]} if "rows" in b else None
    return A,B
# 先看结构
d=load("rsimagerewardmsp_100","unseen1k",4)
print("keys:", list(d.keys())[:12])
for k in list(d.keys())[:12]:
    v=d[k]
    print(" ", k, type(v).__name__, (len(v) if hasattr(v,'__len__') else v) if not isinstance(v,str) else v[:60])
