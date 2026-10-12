"""Plot prospective endpoint headroom and frozen-selector precision after review."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    campaign = ROOT / "experiments/local_expert_v12"
    result = read_json(campaign / "results.json")
    fig,axes = plt.subplots(1,2,figsize=(11,4.8))
    policies = ["noop","ocr@1","geneval@1","selected"]
    values = [result["policy_scores"][p]["complete_scene"]["sum"] for p in policies]
    bars = axes[0].bar(policies,values,color=["#888888","#e09740","#397fad","#398362"])
    axes[0].bar_label(bars,labels=[f"{v:g}/64" for v in values],padding=3)
    axes[0].set_ylim(0,max(values)+6)
    axes[0].set_ylabel("Strict complete-scene endpoints")
    axes[0].set_title("Fixed-state interventions; no student fitting")
    accepted = [q for q in result["queries"] if q["accepted"]]
    states = ["safe_repair","non_repair","unsafe","unknown"]
    counts = [sum(q["selected_comparison"]["status"] == s for q in accepted) for s in states]
    bars = axes[1].bar(["Safe repair","No repair","Known harm","Unresolved"],counts,
                       color=["#398362","#999999","#be5755","#bda753"])
    axes[1].bar_label(bars,padding=3)
    axes[1].set_ylim(0,max(counts,default=0)+5)
    axes[1].set_title(f"Frozen selector: {len(accepted)}/64 accepted")
    axes[1].set_ylabel("Accepted queries")
    fig.suptitle("Prospective external-expert candidate audit: all64 training prompts")
    fig.text(.5,.035,"Action-hidden implementing-assistant review; unknown is not certified safe.\n"
             "One fresh seed/prompt, index3, alpha1, flow10. Training-set diagnosis, not held-out policy efficacy.",ha="center",fontsize=9)
    fig.subplots_adjust(left=.065,right=.98,bottom=.19,top=.83,wspace=.28)
    fig.savefig(campaign / "selection_audit.png",dpi=160)
    plt.close(fig)
    write_json(campaign / "figure_sources.json",{"script_sha256":sha256(__file__),"results_sha256":sha256(campaign / "results.json"),
               "figure_sha256":sha256(campaign / "selection_audit.png")})


if __name__ == "__main__":
    main()
