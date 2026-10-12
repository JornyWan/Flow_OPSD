"""Export full/local generalization, fitting and preservation figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import read_json


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--campaign",required=True)
    args=parser.parse_args()
    out=Path(args.campaign)
    r=read_json(out/"results.json")
    models=["joint128","single","multi","random_matched"]
    names=["Initial","Single state","Multiple states","Random matched"]
    colors=["#7d8597","#457b9d","#2a9d8f","#e9c46a"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]
    cohorts=["positive_fresh_seed","preservation_fresh_seed","unfitted_prompt_fresh_seed"]
    for i,m in enumerate(models):
        vals=[r["scores"][c]["full"][m]["complete_scene"]["sum"] for c in cohorts]
        bars=ax.bar(np.arange(3)+(i-1.5)*.2,vals,.19,label=names[i],color=colors[i])
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=range(3),xticklabels=["Positive (12)","Preservation (12)","Unfitted (8)"],ylim=(0,13),title="Complete full generations on fresh noise")
    ax.legend(fontsize=8)
    ax=axes[0,1]
    for i,path in enumerate(["full","single"]):
        vals=[r["scores"]["positive_fresh_seed"][path][m]["complete_scene"]["sum"] for m in models]
        bars=ax.bar(np.arange(4)+(i-.5)*.35,vals,.33,label="Full model" if path=="full" else "One learned action",color=["#457b9d","#a8dadc"][i])
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=range(4),xticklabels=names,ylim=(0,13),title="Full versus shared-prefix single action (12)")
    ax.tick_params(axis="x",labelrotation=15); ax.legend(fontsize=8)
    ax=axes[1,0]
    for i,subset in enumerate(["original","added"]):
        vals=[r["alignment"][m][subset]["median"] for m in models[1:]]
        bars=ax.bar(np.arange(3)+(i-.5)*.35,vals,.33,label="Original six targets" if subset=="original" else "Added three targets",color=["#457b9d","#2a9d8f"][i])
        ax.bar_label(bars,labels=[f"{v:.3f}" for v in vals],padding=2,fontsize=8)
    ax.axhline(.5,color="gray",linestyle="--",linewidth=1)
    ax.set(xticks=range(3),xticklabels=names[1:],title="Median relative target MSE at128updates",ylabel="MSE / initial target distance")
    ax.legend(fontsize=8)
    ax=axes[1,1]
    for i,(metric,label,color) in enumerate([("known_harm","Known harm","#e76f51"),("comparison_unknown","Pair unknown","#bbbbbb")]):
        vals=[r["scores"]["all_fresh"]["full"][m][metric]["sum"] for m in models]
        bars=ax.bar(np.arange(4)+(i-.5)*.35,vals,.33,label=label,color=color)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=range(4),xticklabels=names,ylim=(0,33),title="Full-generation harm and uncertainty (32fresh)")
    ax.tick_params(axis="x",labelrotation=15); ax.legend(fontsize=8)
    fig.suptitle("Fixed-index multistate supervision | Equal128updates and1536backward queries per arm\nProvisional assistant labels; only two prompts gain training states; fixed-query mechanism test.",fontsize=12)
    for ext in ("png","pdf"):
        fig.savefig(out/f"supervision_study.{ext}",dpi=180)
    plt.close(fig)


if __name__=="__main__":
    main()
