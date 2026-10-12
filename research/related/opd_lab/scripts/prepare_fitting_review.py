"""Render preselected phase-3 TRAIN images without revealing identities or scores."""
import argparse
import json
from pathlib import Path
import textwrap

from opdlab.io import digest, read_json, sha256, write_json
from opdlab.fitting_probe import ARMS


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if read_json(run / "result.json")["status"] != "complete":
        raise ValueError("Wait for all predeclared checkpoints")
    plan = read_json(run / "probe_plan.json")
    records = {r["id"]: r for r in map(json.loads, (run / "data.jsonl").read_text().splitlines())}
    queries = {r["id"]: r for r in map(json.loads, (run / "queries.jsonl").read_text().splitlines())}
    post = {(r["arm"], r["id"]): r for r in map(json.loads, (run / "postfit.jsonl").read_text().splitlines())}
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    keys, blanks = [], []
    for number, identity in enumerate(plan["review_ids"], 1):
        if records[identity]["split"] != "train":
            raise ValueError("Coverage audit must use training images")
        arms = sorted(["base", *ARMS], key=lambda a: digest(["phase3_review_arm", identity, a]))
        fig, axes = plt.subplots(1, len(arms), figsize=(20, 5.4), dpi=128)
        mapping = {}
        for letter, ax, arm in zip("ABCDE", axes, arms):
            row = queries[identity]["candidates"][0] if arm == "base" else post[arm, identity]["full"]
            path = run / row["image"]
            if sha256(path) != row["image_sha256"]:
                raise ValueError("Image changed")
            ax.imshow(Image.open(path)); ax.set_title(letter); ax.axis("off")
            mapping[letter] = {"arm": arm, "row": row}
            blanks.append({"case": number, "label": letter, "text_exact_no_extra": None,
                           "scene_explicit": None, "full_visibility": None, "notes": ""})
        fig.suptitle("\n".join(textwrap.wrap(f"Case {number:02d}: {records[identity]['prompt']}", 175)), fontsize=12)
        fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.80, wspace=.01)
        fig.savefig(out / f"case-{number:02d}.png")
        plt.close(fig)
        keys.append({"case": number, "id": identity, "mapping": mapping})
    write_json(out / "key.json", keys)
    write_json(out / "observations_template.json", blanks)
    write_json(out / "provenance.json", {"plan_sha256": sha256(run / "probe_plan.json"),
               "cases": len(keys), "images": len(blanks), "split": "train",
               "scope": "Implementing-assistant descriptive coverage audit, not independent truth. Scores and identities hidden during observation."})


if __name__ == "__main__":
    main()
