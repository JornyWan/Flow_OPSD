import json, sys
def load(d):
    E = {}
    for l in open(f"outputs/{d}/metrics.jsonl"):
        r = json.loads(l)
        if "reward_avg" in r: E.setdefault(r["epoch"], {})["R"] = r["reward_avg"]
        if "reward_std_mean" in r: E.setdefault(r["_step"], {})["sd"] = r["reward_std_mean"]
        if "opa_delta_res" in r: E.setdefault(r["_step"], {})["dres"] = r["opa_delta_res"]
        if "opa_policy_loss" in r: E.setdefault(r["epoch"], {})["L"] = r["opa_policy_loss"]; E[r["epoch"]]["pn"] = r["opa_posneg_absdiff"]
    return E
runs = sys.argv[1].split(",")
lo, hi, st = map(int, sys.argv[2:5])
Es = [load(d) for d in runs]
print("ep  " + " | ".join(f"{d[13:40]:>36s}" for d in runs))
for e in range(lo, hi, st):
    print(f"{e:3d} " + " | ".join("R%.3f L%.3f d%.4f sd%.3f" % (E.get(e,{}).get("R",0), E.get(e,{}).get("L",0), E.get(e,{}).get("dres",0), E.get(e,{}).get("sd",0)) for E in Es))
