"""Post-review figures for the fixed sparse/dense comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    campaign, run = ROOT / "experiments/local_expert_v10", ROOT / "runs/le10_fit_001"
    results, plan = read_json(campaign / "results.json"), read_json(run / "probe_plan.json")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for arm in ("sparse", "dense"):
        values = [r for r in results["fitting"] if r["arm"] == arm]
        x = [r["optimizer_step"] * plan["arms"][arm]["queries_per_update"] for r in values]
        for ax, y, title in ((axes[0], [r["apple_query"]["relative_target_mse"] for r in values], "Apple query: residual / initial MSE"),
                             (axes[1], [r["all50_loss_over_initial"] for r in values], "All50 fixed points: loss / initial loss")):
            ax.plot(x, y, "o-", label=arm)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Backward queries (audits excluded)")
            ax.set_yscale("log")
            ax.grid(alpha=.2)
            ax.legend()
    fig.suptitle("Same joint128 start; reset AdamW; fixed audit checkpoints", fontsize=12)
    fig.tight_layout()
    fig.savefig(campaign / "fitting.png", dpi=160)
    plt.close(fig)

    apple = "jp1-train-relation-13"
    old = read_json(ROOT / "experiments/local_expert_v9/results.json")
    reference = {e["row"]["candidate"]: e for e in old["joined_review"] if e["row"]["id"] == apple}
    endpoints = {(e["row"]["arm"], e["row"]["optimizer_step"], e["row"]["mode"]): e
                 for e in results["joined_review"] if e["row"]["id"] == apple}
    fig, axes = plt.subplots(2, 4, figsize=(15, 9.1))
    sources = []
    for row_index, mode in enumerate(("local", "full")):
        ref_name = "base" if mode == "local" else "geneval@1"
        panels = [(ROOT / "runs/le9_requery_001", reference[ref_name], "Phase9 baseline" if mode == "local" else "Phase9 verified target")]
        panels += [(run, endpoints[(arm, step, mode)], f"{arm}{step}: {mode}") for arm, step in (("sparse", 32), ("sparse", 320), ("dense", 32))]
        for ax, (folder, entry, title) in zip(axes[row_index], panels):
            path = folder / entry["row"]["image"]
            assert sha256(path) == entry["row"]["image_sha256"]
            obs = entry["observation"]
            fmt = lambda x: "yes" if x is True else "no" if x is False else "unknown"
            ax.imshow(Image.open(path))
            ax.set_title(title, fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"RIGHT: {fmt(obs['clauses']['relation'])} | CLOUD: {fmt(obs['clauses']['text'])}\nFull visibility: {fmt(obs['full_visibility'])}", fontsize=10)
            sources.append({"title": title, "source": str(path.relative_to(ROOT)), "sha256": sha256(path), "observation": obs})
    fig.suptitle("One selected training case: can the fresh repair survive fitting?", fontsize=15)
    fig.text(.5, .025, "Same seed, flow10/512. Local = fixed phase9 state + learned action + frozen joint128 suffix.\n"
             "Full = learned student throughout. Implementing-assistant blinded labels; not held-out evidence.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.93, bottom=.12, hspace=.28, wspace=.045)
    fig.savefig(campaign / "apple_endpoints.png", dpi=150)
    plt.close(fig)
    write_json(campaign / "figure_sources.json", {"script_sha256": sha256(__file__), "panels": sources,
               "fitting_sha256": sha256(campaign / "fitting.png"), "apple_endpoints_sha256": sha256(campaign / "apple_endpoints.png")})


if __name__ == "__main__":
    main()
