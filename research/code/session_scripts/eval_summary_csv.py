# Writes research/results/eval_summary.csv from opd_related/outputs/*.json (run from opd_related).
import json, glob, os, re, csv, sys
rows = []
AX = ["pickscore", "clipscore", "hpsv2", "imagereward", "aesthetic"]
for p in sorted(glob.glob("outputs/*.json")):
    try: d = json.load(open(p))
    except Exception: continue
    if not isinstance(d, dict) or "scores" not in d: continue
    n = os.path.basename(p)[:-5]
    m = re.match(r"matched_pi_(.+)_(\d+)_(default|[a-z0-9]+)_(drawbench|unseen1k|[a-z0-9]+)_flow(\d+)$", n)
    tag, ck = (m.group(1), int(m.group(2))) if m else (n, "")
    ss = d.get("score_stats", {})
    ck_path = d.get("ckpt", "") or ""
    run = ck_path.split("/outputs/")[1].split("/")[0] if "/outputs/" in ck_path else os.path.basename(os.path.dirname(os.path.dirname(ck_path.rstrip("/"))))
    r = dict(file=n, tag=tag, ck=ck, run_dir=run, config=d.get("config", ""), train_reward=d.get("train_reward", ""),
             prompt_set=d.get("prompt_set", ""), num_steps=d.get("num_steps", ""), n=d.get("num_prompts", ""))
    for a in AX:
        r[a] = d["scores"].get(a, ""); r[a + "_se"] = ss.get(a, {}).get("se", "")
    rows.append(r)
with open(sys.argv[1], "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "rows")
