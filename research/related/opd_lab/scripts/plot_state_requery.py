"""Post-unblinding illustration of the apple diagnostic; never a selection input."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from opdlab.io import ROOT, read_json, sha256, write_json


def main():
    ident = "jp1-train-relation-13"
    campaign = ROOT / "experiments/local_expert_v9"
    previous = read_json(ROOT / "experiments/local_expert_v8/results.json")
    current = read_json(campaign / "results.json")
    old = {e["row"]["mode"]: e for e in previous["joined_review"] if e["row"]["id"] == ident}
    new = {e["row"]["candidate"]: e for e in current["joined_review"] if e["row"]["id"] == ident}
    panels = [
        (8, old["local"], "Old prefix / student at k / old suffix"),
        (8, old["suffix"], "Old prefix / student from k onward"),
        (8, old["prefix"], "Student through k / old suffix"),
        (8, old["full"], "Student throughout"),
        (9, new["base"], "Current student baseline"),
        (9, new["ocr@1"], "Fresh OCR, alpha = 1"),
        (9, new["geneval@0.5"], "Fresh GenEval, alpha = 0.5"),
        (9, new["geneval@1"], "Fresh GenEval, alpha = 1"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.3))
    provenance = []
    for ax, (phase, entry, title) in zip(axes.flat, panels):
        row, obs = entry["row"], entry["observation"]
        run = ROOT / ("runs/le8_trajectory_001" if phase == 8 else "runs/le9_requery_001")
        path = run / row["image"]
        assert sha256(path) == row["image_sha256"]
        right = obs["clauses"]["relation"]
        text = obs["clauses"]["text"]
        visible = obs["full_visibility"]
        assert type(right) is bool and text is True and type(visible) is bool
        ax.imshow(Image.open(path))
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("RIGHT: " + ("yes" if right else "no") + " | CLOUD: yes | visible: "
                      + ("yes" if visible else "no"),
                      color="#186642" if right and visible else "#9e2727", fontsize=10)
        provenance.append({"title": title, "source": str(path.relative_to(ROOT)),
                           "image_sha256": row["image_sha256"], "observation": obs})
    fig.suptitle("One selected training case: apple RIGHT of bowl; sign says only CLOUD", fontsize=15)
    fig.text(.5, .505, "Fresh teacher intervention at the current student's state; same current-student continuation",
             ha="center", fontsize=11)
    fig.text(.5, .016, "Same seed, 512px, flow10; k = 3 (zero-based). Frozen weights, no new training.\n"
             "Labels: implementing-assistant blinded review, frozen before key join; this is a mechanism example, not generalization evidence.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.015, right=.985, top=.92, bottom=.10, hspace=.32, wspace=.035)
    fig.savefig(campaign / "diagnostic_chain.png", dpi=150)
    plt.close(fig)
    write_json(campaign / "diagnostic_chain_sources.json", {
        "purpose": "post-unblinding explanatory figure, not a selection input",
        "script_sha256": sha256(__file__), "panels": provenance,
        "figure_sha256": sha256(campaign / "diagnostic_chain.png")})


if __name__ == "__main__":
    main()
