"""Standalone development comparison; no held-out or training-gain claims."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import ROOT,read_json,sha256,write_json


def main():
    campaign = ROOT / "experiments/local_expert_v13"
    result = read_json(campaign / "results.json")
    policies = ["v1","v1_discrete_event","v2"]
    labels = ["Original v1","Discrete event\non v1 labels","Evidence v2"]
    colors = ["#438767","#909090","#bb5656","#c0a646"]
    fig,axes = plt.subplots(1,2,figsize=(11,5))
    bottom = np.zeros(3)
    for status,label,color in zip(("safe_repair","non_repair","unsafe","unknown"),("Safe repair","No repair","Known harm","Unresolved"),colors):
        counts = np.array([result["policies"][p]["accepted_classes"].get(status,0) for p in policies])
        bars = axes[0].bar(labels,counts,bottom=bottom,label=label,color=color)
        axes[0].bar_label(bars,labels=[str(n) if n else "" for n in counts],label_type="center",fontsize=9)
        bottom += counts
    axes[0].set_ylabel("Accepted queries / 64")
    axes[0].set_ylim(0,max(bottom.max()+4,8))
    axes[0].legend(fontsize=8,loc="upper right")
    full = ["noop",*policies]
    values = [result["policies"][p]["endpoints"]["complete_scene"]["sum"] for p in full]
    bars = axes[1].bar(["Noop",*labels],values,color=["#999999","#5681ad","#ab8251","#438767"])
    axes[1].bar_label(bars,labels=[f"{int(v)}/64" for v in values],padding=3)
    axes[1].set_ylim(0,max(values)+6)
    axes[1].set_ylabel("Strict complete-scene endpoints")
    fig.suptitle("Verifier development on 192 previously reviewed images")
    fig.text(.5,.035,"Frozen phase12 assistant labels; no fresh-seed confirmation or student training.\n"
             "Three candidates per query share a prompt/state. Unknown is not certified safe.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.98,bottom=.20,top=.88,wspace=.28)
    fig.savefig(campaign / "evidence_audit.png",dpi=160)
    plt.close(fig)
    write_json(campaign / "figure_sources.json",{"results_sha256":sha256(campaign / "results.json"),
               "script_sha256":sha256(__file__),"figure_sha256":sha256(campaign / "evidence_audit.png")})


if __name__ == "__main__":
    main()
