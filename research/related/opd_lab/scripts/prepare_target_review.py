"""Hide noop/target identity for every predeclared accepted training query."""
import argparse
import json
from pathlib import Path
import textwrap

from opdlab.io import digest, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    plan = read_json(run / "action_plan.json")
    parent = Path(plan["parent_run"])
    records = {r["id"]: r for r in map(json.loads, (parent / "data.jsonl").read_text().splitlines())}
    queries = {r["id"]: r for r in map(json.loads, (parent / "queries.jsonl").read_text().splitlines())}
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    keys, template = [], []
    for case, identity in enumerate(plan["review_ids"], 1):
        q, record = queries[identity], records[identity]
        if not q["accepted"] or record["split"] != "train":
            raise ValueError("Unexpected review query")
        points = {"noop": q["candidates"][0], "target": next(c for c in q["candidates"] if c["name"] == q["selected"])}
        order = sorted(points, key=lambda n: digest(["phase4-review-label", identity, n]))
        fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), dpi=128)
        mapping = {}
        blank = {"case": case, "observations": {}, "better": "uncertain", "comparison_notes": ""}
        for label, ax, name in zip("AB", axes, order):
            row = points[name]
            path = parent / row["image"]
            if sha256(path) != row["image_sha256"] or sha256(path) != plan["parent_files"][row["image"]]:
                raise ValueError("Reference image changed")
            ax.imshow(Image.open(path)); ax.set_title(label); ax.axis("off")
            mapping[label] = {"action": name, "row": row}
            blank["observations"][label] = {"clauses": {c["id"]: None for c in record["constraints"]},
                                             "full_visibility": None, "visible_text": None, "notes": ""}
        header = f"Case {case:02d}: {record['prompt']}\nClauses: " + ", ".join(c["id"] for c in record["constraints"])
        fig.suptitle("\n".join(textwrap.wrap(header, 110)), fontsize=11)
        fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.76, wspace=.01)
        fig.savefig(out / f"case-{case:02d}.png"); plt.close(fig)
        keys.append({"case": case, "id": identity, "record": record, "mapping": mapping})
        template.append(blank)
    write_json(out / "key.json", keys)
    write_json(out / "observations_template.json", template)
    write_json(out / "provenance.json", {"plan_sha256": sha256(run / "action_plan.json"),
               "cases": len(keys), "images": len(keys) * 2,
               "scope": "All phase3 accepted training pairs; implementing-assistant audit. Prior exposure limits blinding. No independent ground truth."})


if __name__ == "__main__":
    main()
