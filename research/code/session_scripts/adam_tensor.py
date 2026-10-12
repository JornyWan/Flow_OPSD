import sys, json, math, torch
sys.path.insert(0, "probes")
from p11_abcd_dirs import param_order
keys = param_order("stabilityai/stable-diffusion-3.5-medium")
run, cks = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
out = {}
for ck in cks:
    o = torch.load(f"outputs/{run}/checkpoints/checkpoint-{ck}/optimizer.pt", map_location="cpu", weights_only=False)
    pg = o["param_groups"][0]; b1, b2 = pg["betas"]; eps = pg["eps"]
    t = o["state"][0]["step"]; t = int(t.item() if torch.is_tensor(t) else t)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    rows = []
    for i, k in enumerate(keys):
        m = o["state"][i]["exp_avg"].double() / bc1; v = o["state"][i]["exp_avg_sq"].double() / bc2
        u = m / (v.sqrt() + eps)
        rows.append((k, float(u.pow(2).mean().sqrt()), float(v.mean().sqrt()), float(m.pow(2).mean().sqrt()), u.numel(), float(u.pow(2).sum())))
    out[ck] = rows
    tot = sum(r[5] for r in rows); n = sum(r[4] for r in rows)
    srt = sorted(rows, key=lambda r: -r[1])
    sv = sorted(r[2] for r in rows); medv = sv[len(sv) // 2]
    top = srt[:8]
    share = sum(r[5] for r in srt[:20]) / tot
    print(f"ck{ck} t{t} global rms(u)={math.sqrt(tot/n):.3f}  top20/{len(rows)} 张量占 ‖u‖² {share:.0%}  #张量 rms(u)>0.5: {sum(r[1]>0.5 for r in rows)}")
    for r in top:
        print(f"    {r[0][len('base_model.model.'):][:58]:58s} rms(u) {r[1]:.2f}  rms√v/中位 {r[2]/medv:.2f}")
json.dump({str(k): v for k, v in out.items()}, open(f"{sys.argv[3]}", "w"))
