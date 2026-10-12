import re, math, torch
src = open("CA-OPD/code/DiffusionOPSD-rpi/scripts/train_opsd_ri_sd3.py").read()
fn = re.search(r"def _mk_split_step\(.*?\n(?=\n\ndef )", src, re.S).group(0)
g = {"torch": torch, "math": math, "_OD": {}, "_SA": {"m": None, "v": None, "t": 0}, "_MK_SPLIT": 0.5}
exec(fn, g)
torch.manual_seed(0)
P = [torch.randn(5, 3, dtype=torch.float32), torch.randn(7, dtype=torch.float32)]
Q = [p.clone().requires_grad_(True) for p in P]
ref = torch.optim.Adam(Q, lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
class O: param_groups = [{"betas": (0.9, 0.999), "eps": 1e-8, "lr": 3e-4}]
mx = 0.0
for it in range(20):
    G = [torch.randn_like(p) * 1e-3 for p in P]
    g["_OD"]["xs"] = [x.clone() for x in G]
    before = [p.clone() for p in P]
    out = g["_mk_split_step"](P, O, 1.0, False)
    for q, x in zip(Q, G): q.grad = x.clone()
    qb = [q.detach().clone() for q in Q]; ref.step()
    for p, b, q, c in zip(P, before, Q, qb):
        mx = max(mx, float(((p - b) - 0.5 * (q.detach() - c)).abs().max()))
print("max |Δsplit − 0.5·Δadam| =", mx, "| last", out)
# skip 与截断
g["_OD"]["xs"] = [torch.ones_like(p) for p in P]; b = [p.clone() for p in P]
o = g["_mk_split_step"](P, O, 1.0, True); print("skip:", o, "unchanged:", all(torch.equal(p, q) for p, q in zip(P, b)))
g["_OD"]["xs"] = [torch.ones_like(p) * 10 for p in P]; t0 = g["_SA"]["t"]
o = g["_mk_split_step"](P, O, 1.0, False); print("clip big grad gn=%.2f t %d->%d" % (o["sa_gn"], t0, g["_SA"]["t"]))
print("no xs:", g["_mk_split_step"](P, O, 1.0, False))
