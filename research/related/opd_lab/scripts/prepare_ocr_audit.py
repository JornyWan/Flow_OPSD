"""Sample unique development prompts without outcomes; render target-hidden audit sheets."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from opdlab.io import digest, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    records = {r["id"]: r for r in map(json.loads, (run / "data.jsonl").read_text().splitlines())}
    queries = read_json(run / "diagnostic_queries.json")["queries"]
    by_group, by_id = defaultdict(set), defaultdict(list)
    for q in queries:
        by_group[records[q["id"]]["group"]].add(q["id"])
        by_id[q["id"]].append(q)
    cases, keys = [], []
    for group in sorted(by_group):
        ids = sorted(by_group[group], key=lambda i: digest(["ocr-audit-v2-id", i]))[:8]
        for identity in ids:
            q = min(by_id[identity], key=lambda q: digest(["ocr-audit-v2-query", q["query_id"]]))
            row = read_json(run / "query_results" / q["query_id"] / "result.json")
            c = min(row["candidates"], key=lambda c: digest(["ocr-audit-v2-arm", q["query_id"], c["name"]]))
            path = run / c["image"]
            if sha256(path) != c["image_sha256"]:
                raise ValueError("Candidate image changed")
            case = {"case": len(cases) + 1, "image": str(path), "image_sha256": c["image_sha256"]}
            cases.append(case)
            keys.append({**case, "record": records[identity], "query": q, "candidate": c})
    write_json(out / "cases.json", {"selection": "8 unique prompt IDs per group, then one query and action, all by fixed hashes without outcomes", "cases": cases})
    write_json(out / "key.json", keys)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    for start in range(0, len(cases), 8):
        fig, axes = plt.subplots(4, 2, figsize=(9, 18), dpi=128)
        for ax, case in zip(axes.flat, cases[start:start + 8]):
            ax.imshow(Image.open(case["image"]))
            ax.set_title(f"Case {case['case']:02d}")
            ax.axis("off")
        fig.tight_layout(pad=.4)
        fig.savefig(out / f"sheet-{start // 8 + 1}.png")
        plt.close(fig)
    print({"cases": len(cases), "out": str(out)})


if __name__ == "__main__":
    main()
