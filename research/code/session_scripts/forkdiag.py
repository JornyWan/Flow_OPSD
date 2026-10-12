import sys, json
from collections import defaultdict
def load(d):
    out = {}
    for l in open(f"outputs/{d}/metrics.jsonl"):
        r = json.loads(l)
        if "od_urms" in r: out[r["step"]] = r
    return out
def health(d, e):
    try: rows = [json.loads(l) for l in open(f"outputs/{d}/train_rewards/epoch_{e:04d}.jsonl")]
    except FileNotFoundError: return None
    R = sorted(r["rewards_raw"]["avg"] for r in rows)
    return R[len(R)//2], sum(x < 1.3 for x in R)
runs = sys.argv[1].split(",")
D = {r: load(r) for r in runs}
steps = sorted(set().union(*[set(v) for v in D.values()]))
print("step | " + " | ".join(f"{r[:16]:>40s}" for r in runs))
for s in steps:
    cells = []
    for r in runs:
        x = D[r].get(s)
        if not x: cells.append(" " * 40); continue
        gx, b, g = x["od_gn_x"], x.get("od_gn_xb", float("nan")), x.get("od_gn_xg", float("nan"))
        c = (gx*gx - b*b - g*g) / (2*b*g) if b == b and b > 0 else float("nan")
        h = health(r, s + 1) or (0, -1)
        cells.append("u%.2f q%.4f x%.4f c%+.2f qq%+.2f n%3d" % (x["od_urms"], x["od_gn_q"], gx, c, x.get("od_cos_qq", float("nan")), h[1]))
    print(f"{s:4d} | " + " | ".join(cells))
