"""Render all planned methods for a frozen validation subset, hiding method identities."""
import argparse
import json
from pathlib import Path
import textwrap

from opdlab.evaluate import read_evaluation
from opdlab.io import digest, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--evaluation", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root, out = Path(args.evaluation).resolve(), Path(args.out).resolve()
    plan = read_json(args.plan)
    records = {r["id"]: r for r in map(json.loads, Path(args.dataset).read_text().splitlines())}
    tables = {}
    for arm in plan["arms"]:
        protocol, rows = read_evaluation(root / arm)
        if [protocol["sampler"], protocol["steps"]] != plan["cell"] or protocol["split"] != "validation":
            raise ValueError("Review protocol differs from the frozen plan")
        tables[arm] = {(r["id"], r["replicate"]): r for r in rows.values()}
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    keys = []
    for number, identity in enumerate(plan["ids"], 1):
        arms = sorted(plan["arms"], key=lambda arm: digest(["le2-val-review-arm", identity, arm]))
        fig, axes = plt.subplots(1, len(arms), figsize=(20, 5.4), dpi=128)
        mapping = {}
        for letter, ax, arm in zip("ABCDE", axes, arms):
            row = tables[arm][identity, plan["replicate"]]
            path = root / arm / row["image"]
            if sha256(path) != row["image_sha256"]:
                raise ValueError("Review image changed")
            ax.imshow(Image.open(path)); ax.set_title(letter); ax.axis("off")
            mapping[letter] = {"arm": arm, "row": row}
        fig.suptitle("\n".join(textwrap.wrap(f"Case {number:02d}: {records[identity]['prompt']}", 175)), fontsize=12)
        fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.80, wspace=.01)
        fig.savefig(out / f"case-{number:02d}.png")
        plt.close(fig)
        keys.append({"case": number, "id": identity, "mapping": mapping})
    write_json(out / "key.json", keys)
    write_json(out / "provenance.json", {"plan_sha256": sha256(args.plan), "cases": len(keys),
               "images": len(keys) * len(plan["arms"]), "scope": plan["scope"]})


if __name__ == "__main__":
    main()
