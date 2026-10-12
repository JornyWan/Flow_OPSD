"""Render all collection categories and every accepted/harmful case, without sampling."""
from collections import Counter
from pathlib import Path
import html
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opdlab.io import ROOT, read_json
from review_prospective import visual

campaign=ROOT/"experiments/local_expert_v20"
result=read_json(campaign/"results.json")
cases=result["cases"]
counts=Counter(c["comparison"]["status"] for c in cases)
fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
labels=["Safe repair","Known harm","Unknown","No encoded repair"]
values=[counts[k] for k in ("safe_repair","unsafe","unknown","non_repair")]
bars=axes[0].barh(labels,values,color=["#278a68","#c4534c","#ab8a43","#87929f"])
axes[0].bar_label(bars,padding=4)
axes[0].invert_yaxis();axes[0].set_xlim(0,29)
axes[0].set_xlabel("States / 48; categories are disjoint")
axes[0].set_title("One GenEval action at each new state")
byid=result["coverage_gate"]["safe_by_prompt"]
ids=sorted(byid)
complete=[sum(c["safe_repair"] and c["complete_scene"] for c in cases if c["id"]==pid) for pid in ids]
partial=[byid[pid]-n for pid,n in zip(ids,complete)]
names=[i.removeprefix("jp1-train-") for i in ids]
axes[1].barh(names,complete,label="Safe and complete",color="#278a68")
axes[1].barh(names,partial,left=complete,label="Safe, still incomplete",color="#91c5aa")
axes[1].axvline(2,color="#777777",ls="--",lw=1)
axes[1].set_xlim(0,8);axes[1].set_xlabel("Accepted states / 8 per prompt")
axes[1].set_title("Coverage gate: 9 safe; 3 prompts >=2; 8 complete")
axes[1].legend(loc="lower right",fontsize=8)
fig.suptitle("Phase20 collection | frozen assistant labels; no student training result",fontsize=13)
for ext in ("png","pdf"):
    fig.savefig(campaign/f"collection_coverage.{ext}",dpi=180)
plt.close(fig)

lookup={(e["case"],e["row"]["mode"]):e for e in result["joined_review"]}
run=ROOT/"runs/le20_collection_001"
parts=['<!doctype html><meta charset="utf-8"><title>Phase20 complete repair/harm inventory</title>',
       '<style>body{font:16px system-ui;max-width:1060px;margin:auto}article{border-top:1px solid #ccc;padding:20px 0}.pair{display:flex;gap:12px}.pair div{width:50%}img{max-width:100%}pre{white-space:pre-wrap}</style>',
       '<h1>All 9 accepted actions and all 4 known harms</h1><p>Complete inventory of these two categories. The other 35 states remain in the frozen review/results. Provisional assistant labels; these are teacher endpoints, not learned student outputs.</p>']
for category in ("safe_repair","unsafe"):
    parts.append(f'<h2>{html.escape(category)}</h2>')
    for c in cases:
        if c["comparison"]["status"]!=category: continue
        parts.append(f'<article><h3>Case {c["case"]}: {html.escape(c["id"])} / seed {c["seed"]}</h3><div class="pair">')
        for mode in ("noop","geneval@1"):
            e=lookup[c["case"],mode];o=e["observation"]
            path=os.path.relpath(run/e["row"]["image"],campaign)
            parts.append(f'<div><b>{html.escape(mode)}</b><img src="{html.escape(path)}"><pre>{html.escape(str(o))}</pre></div>')
        parts.append('</div></article>')
(campaign/"collection_cases.html").write_text("\n".join(parts)+"\n")
print('Wrote coverage plot and complete 9-safe/4-harm image inventory.')
