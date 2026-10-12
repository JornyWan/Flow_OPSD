import sys, math, torch
run = sys.argv[1]; cks = range(int(sys.argv[2]), int(sys.argv[3]) + 1, 10)
res = []
for ck in cks:
    try: o = torch.load(f"outputs/{run}/checkpoints/checkpoint-{ck}/optimizer.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError: continue
    pg = o["param_groups"][0]; b1, b2 = pg["betas"]; eps = pg["eps"]
    t = o["state"][0]["step"]; t = int(t.item() if torch.is_tensor(t) else t)
    s2 = n = 0; m2 = v2 = 0.0
    for st in o["state"].values():
        m = st["exp_avg"].double() / (1 - b1 ** t); v = st["exp_avg_sq"].double() / (1 - b2 ** t)
        s2 += float((m / (v.sqrt() + eps)).pow(2).sum()); n += m.numel(); m2 += float(m.pow(2).sum()); v2 += float(v.sum())
    res.append(f"{ck}:{math.sqrt(s2/n):.3f}")
print(run[13:], " ".join(res), flush=True)
