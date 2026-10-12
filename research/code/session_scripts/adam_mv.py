import sys, math, torch
run = sys.argv[1]; cks = range(int(sys.argv[2]), int(sys.argv[3]) + 1, int(sys.argv[4]))
out = []
for ck in cks:
    try: o = torch.load(f"outputs/{run}/checkpoints/checkpoint-{ck}/optimizer.pt", map_location="cpu", weights_only=False)
    except FileNotFoundError: continue
    pg = o["param_groups"][0]; b1, b2 = pg["betas"]
    t = o["state"][0]["step"]; t = int(t.item() if torch.is_tensor(t) else t)
    m2 = v1 = n = 0.0
    for st in o["state"].values():
        m = st["exp_avg"].double() / (1 - b1 ** t); v = st["exp_avg_sq"].double() / (1 - b2 ** t)
        m2 += float(m.pow(2).sum()); v1 += float(v.sum()); n += m.numel()
    out.append(f"{ck}:{1e4*math.sqrt(m2/n):.3f}/{1e4*math.sqrt(v1/n):.3f}")
print(run[13:], " ".join(out), flush=True)
