"""Outcome-independent qualitative review sampling and research contact sheets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import textwrap

from opdlab.io import digest, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--render-to")
    args = p.parse_args()
    run, plan_path = Path(args.run).resolve(), Path(args.plan).resolve()
    manifest = read_json(run / "diagnostic_queries.json")
    records = {r["id"]: r for r in (json.loads(s) for s in (run / "data.jsonl").read_text().splitlines())}
    if not plan_path.exists():
        groups = defaultdict(list)
        for query in manifest["queries"]:
            groups[(records[query["id"]]["group"], query["query_index"])].append(query)
        selected = [min(groups[key], key=lambda q: digest(["qualitative-review-v1", q["query_id"]]))
                    for key in sorted(groups)]
        write_json(plan_path, {"run": str(run), "query_manifest_sha256": sha256(run / "diagnostic_queries.json"),
                   "selection": "minimum fixed hash within each predefined data group x query position; no outcomes consulted",
                   "cases": [{"case": i + 1, "query": q} for i, q in enumerate(selected)]})
    plan = read_json(plan_path)
    if plan["run"] != str(run) or plan["query_manifest_sha256"] != sha256(run / "diagnostic_queries.json"):
        raise ValueError("Review plan input mismatch")
    print(json.dumps({"plan": str(plan_path), "cases": len(plan["cases"])}))
    if not args.render_to:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    out = Path(args.render_to).resolve()
    out.mkdir(parents=True, exist_ok=False)
    keys = []
    for spec in plan["cases"]:
        query = spec["query"]
        qid = query["query_id"]
        row = read_json(run / "query_results" / qid / "result.json")
        candidates = sorted(row["candidates"], key=lambda c: digest(["review-v1", qid, c["name"]]))
        fig, axes = plt.subplots(1, len(candidates), figsize=(20, 5.4), dpi=128)
        mapping = {}
        for letter, ax, candidate in zip("ABCDE", axes, candidates):
            path = run / candidate["image"]
            if sha256(path) != candidate["image_sha256"]:
                raise ValueError("Review image changed")
            ax.imshow(Image.open(path))
            ax.set_title(letter)
            ax.axis("off")
            mapping[letter] = candidate["name"]
        title = f"Case {spec['case']:02d}: " + records[query["id"]]["prompt"]
        fig.suptitle("\n".join(textwrap.wrap(title, width=180)), fontsize=12)
        fig.subplots_adjust(left=.005, right=.995, bottom=.005, top=.83, wspace=.01)
        fig.savefig(out / f"case-{spec['case']:02d}.png")
        plt.close(fig)
        keys.append({**spec, "mapping": mapping, "selected": row["selected"]})
    write_json(out / "key.json", keys)


if __name__ == "__main__":
    main()
