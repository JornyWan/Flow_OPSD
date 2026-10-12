"""Blind all prospective candidates, then audit the already-frozen selector."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from opdlab.config import Experiment
from opdlab.io import ROOT, digest, read_json, sha256, write_json
from opdlab.method import Candidate, Verdict, select
from opdlab.report import prompt_summary


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def validate(run):
    plan, result = read_json(run / "probe_plan.json"), read_json(run / "result.json")
    assert result["status"] == "complete" and sha256(run / "candidates.jsonl") == result["candidates_sha256"]
    for path, h in read_json(run / "probe_attestation.json")["inputs"].items():
        assert sha256(path) == h, path
    rows, config = lines(run / "candidates.jsonl"), Experiment.load(run / "config.json")
    assert len(rows) == 64 and [r["query"] for r in rows] == plan["queries"]
    for row in rows:
        assert [c["name"] for c in row["candidates"]] == plan["candidate_order"]
        assert sha256(run / row["saved_query"]) == row["saved_query_sha256"]
        assert row["target_sha256"] == row["action_hashes"][row["selected"]]
        candidates = []
        for c in row["candidates"]:
            assert c["valid"] and sha256(run / c["image"]) == c["image_sha256"]
            candidates.append(Candidate(c["name"], c["teacher"], c["alpha"], torch.empty(0), c["distance"],
                              Verdict(tuple(c["satisfied"]), tuple(c["progress"]), c["valid"], c["quality"])))
        winner = select(candidates, config, row["seed"], row["query"]["ordinal"])
        assert row["selected"] == candidates[winner].name and row["accepted"] == (winner != 0)
    return rows


def prepare(run, review):
    rows = validate(run)
    records = {r["id"]: r for r in lines(run / "data.jsonl")}
    review.mkdir(parents=True, exist_ok=False)
    key, items = [], []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    for case, row in enumerate(sorted(rows, key=lambda r: digest(["phase12_blind_query", r["query"]["query_id"]])), 1):
        candidates = sorted(row["candidates"], key=lambda c: digest(["phase12_blind_action", row["query"]["query_id"], c["name"]]))
        record = records[row["id"]]
        sheet = Image.new("RGB", (1024, 1200), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((6,4), f"Case {case:02d}\n" + "\n".join(textwrap.wrap(record["prompt"],125)), font=font, fill="black", spacing=3)
        mapping, images = {}, {}
        for i, (letter, candidate) in enumerate(zip("ABC", candidates)):
            image = f"image-{case:02d}-{letter}.png"
            shutil.copyfile(run / candidate["image"], review / image)
            x, y = (i % 2) * 512, 128 + (i // 2) * 536
            draw.text((x+8,y),letter,font=font,fill="black")
            sheet.paste(Image.open(review / image),(x,y+24))
            mapping[letter], images[letter] = candidate, image
        sheet.save(review / f"case-{case:02d}.png")
        key.append({"case":case,"row":row,"mapping":mapping})
        items.append({"case":case,"prompt":record["prompt"],"constraints":record["constraints"],"images":images})
    write_json(review / "items.json",items)
    write_json(review / "key.json",key)
    write_json(review / "inputs.json",{"run":str(run),"cases":64,"images":192,"result_sha256":sha256(run / "result.json"),
               "key_sha256":sha256(review / "key.json"),"items_sha256":sha256(review / "items.json")})
    print("Prepared64 blinded three-view sheets; all192 candidates retained.")


def check_labels(items, observations):
    bycase = {r["case"]:r for r in observations}
    if len(bycase) != len(observations) or set(bycase) != {i["case"] for i in items}:
        raise ValueError("Missing/duplicate reviewed case")
    for item in items:
        views = bycase[item["case"]]["views"]
        assert set(views) == set("ABC")
        for obs in views.values():
            assert set(obs["clauses"]) == {c["id"] for c in item["constraints"]}
            assert all(type(v) is bool or v is None for v in [*obs["clauses"].values(),obs["full_visibility"]])
            assert isinstance(obs["notes"],str) and isinstance(obs["visible_text"],str)
    return bycase


def freeze(run, review):
    if (review / "observations.json").exists() or (review / "observations_frozen.json").exists():
        raise FileExistsError("Already frozen")
    items = read_json(review / "items.json")
    observations = sorted(lines(review / "observations_draft.jsonl"),key=lambda r:r["case"])
    bycase = check_labels(items,observations)
    seen = {}
    for item in items:
        for letter,path in item["images"].items():
            obs = bycase[item["case"]]["views"][letter]
            label,h = (obs["clauses"],obs["full_visibility"]),sha256(review / path)
            if h in seen and seen[h] != label:
                raise ValueError("Identical pixels have inconsistent labels")
            seen[h] = label
    write_json(review / "observations.json",observations)
    write_json(review / "observations_frozen.json",{"frozen_at_utc":datetime.now(timezone.utc).isoformat(),
               "frozen_before_key_join":True,"cases":64,"views":192,
               "reviewer":"implementing assistant; action/model/score/selection hidden; not independent human ground truth",
               **{f"{n}_sha256":sha256(review / f"{n}.json") for n in ("observations","key","items")},
               "rubric_sha256":sha256(run / "REVIEW_RUBRIC.md")})
    print("All192 observations frozen before key join.")


def compare(base, candidate):
    a,b = base["clauses"],candidate["clauses"]
    assert set(a) == set(b)
    repairs = [k for k in a if a[k] is False and b[k] is True]
    harms = [k for k in a if a[k] is True and b[k] is False]
    visibility_harm = base["full_visibility"] is True and candidate["full_visibility"] is False
    unknown = any(v is None for v in [*a.values(),*b.values(),base["full_visibility"],candidate["full_visibility"]])
    known_harm = bool(harms) or visibility_harm
    safe = bool(repairs) and not known_harm and not unknown
    status = "unsafe" if known_harm else "unknown" if unknown else "safe_repair" if safe else "non_repair"
    return {"status":status,"safe_repair":safe,"known_harm":known_harm,"has_unknown":unknown,
            "repaired_clauses":repairs,"harmed_clauses":harms,"visibility_harm":visibility_harm,
            "safe_visibility_gain":base["full_visibility"] is False and candidate["full_visibility"] is True and not known_harm and not unknown}


def visual(obs):
    vals = list(obs["clauses"].values())
    return {"complete_scene":all(v is True for v in vals) and obs["full_visibility"] is True,
            "joint_clauses":all(v is True for v in vals),"full_visibility":obs["full_visibility"] is True,
            "has_unknown":any(v is None for v in [*vals,obs["full_visibility"]])}


def ratio(numerator, denominator):
    a,b = np.asarray(numerator,dtype=float),np.asarray(denominator,dtype=float)
    assert len(a) == len(b) and len(a) and np.all(a <= b)
    rng = np.random.default_rng(0)
    indices = rng.integers(0,len(a),size=(2000,len(a)))
    an,bn = a[indices].sum(axis=1),b[indices].sum(axis=1)
    values = an[bn > 0] / bn[bn > 0]
    return {"numerator":int(a.sum()),"denominator":int(b.sum()),
            "mean":float(a.sum()/b.sum()) if b.sum() else None,
            "ci95":np.quantile(values,[.025,.975]).tolist() if len(values) else None,
            "bootstrap_nonzero_denominator":len(values),"prompts":len(a)}


def summarize(run, review):
    validate(run)
    frozen, inputs = read_json(review / "observations_frozen.json"),read_json(review / "inputs.json")
    assert frozen["frozen_before_key_join"] is True
    for n in ("observations","key","items"):
        assert sha256(review / f"{n}.json") == frozen[f"{n}_sha256"]
    assert sha256(run / "REVIEW_RUBRIC.md") == frozen["rubric_sha256"] and sha256(run / "result.json") == inputs["result_sha256"]
    labels = check_labels(read_json(review / "items.json"),read_json(review / "observations.json"))
    queries, joined = [], []
    for entry in read_json(review / "key.json"):
        row, byname = entry["row"], {}
        for letter,c in entry["mapping"].items():
            obs = labels[entry["case"]]["views"][letter]
            byname[c["name"]] = obs
            joined.append({"case":entry["case"],"view":letter,"id":row["id"],"candidate":c,"observation":obs})
        comparisons = {n:compare(byname["noop"],byname[n]) for n in ("ocr@1","geneval@1")}
        chosen = comparisons[row["selected"]] if row["accepted"] else None
        queries.append({"case":entry["case"],"id":row["id"],"group":row["query"]["group"],"seed":row["seed"],
                        "selected":row["selected"],"accepted":row["accepted"],"observations":byname,
                        "proxy":{c["name"]:{"utility":c["utility"],"joint":c["joint_success"]} for c in row["candidates"]},
                        "comparisons":comparisons,"selected_comparison":chosen,
                        "any_safe_repair":any(c["safe_repair"] for c in comparisons.values())})
    accepted = [q["accepted"] for q in queries]
    safe = [q["accepted"] and q["selected_comparison"]["safe_repair"] for q in queries]
    harm = [q["accepted"] and q["selected_comparison"]["known_harm"] for q in queries]
    unclear = [q["accepted"] and q["selected_comparison"]["status"] == "unknown" for q in queries]
    determinate = [a and not u for a,u in zip(accepted,unclear)]
    calibration = {"acceptance":ratio(accepted,[True]*64),"conservative_precision":ratio(safe,accepted),
                   "determinate_precision":ratio(safe,determinate),"accepted_harm":ratio(harm,accepted),
                   "accepted_unresolved":ratio(unclear,accepted),
                   "recall_including_rejections":ratio(safe,[q["any_safe_repair"] for q in queries]),
                   "candidate_headroom":ratio([q["any_safe_repair"] for q in queries],[True]*64)}
    policy_scores, candidate_scores, transitions = {}, {}, {}
    for policy in ("noop","ocr@1","geneval@1","selected"):
        values, changes = {}, {}
        for q in queries:
            name = q["selected"] if policy == "selected" else policy
            obs,base = q["observations"][name],q["observations"]["noop"]
            vm,bm = visual(obs),visual(base)
            for k,v in {**vm,**{f"clause_{k}":v is True for k,v in obs["clauses"].items()},
                        "proxy_utility":q["proxy"][name]["utility"],"proxy_joint":q["proxy"][name]["joint"]}.items():
                values.setdefault(k,[]).append((q["id"],v))
            for k in ("complete_scene","joint_clauses","full_visibility"):
                values.setdefault(f"{k}_delta",[]).append((q["id"],int(vm[k])-int(bm[k])))
            for k,before in {**base["clauses"],"visibility":base["full_visibility"]}.items():
                if before is True:
                    after = obs["full_visibility"] if k == "visibility" else obs["clauses"][k]
                    bucket = changes.setdefault(k,{"denominator":0,"false":0,"unknown":0})
                    bucket["denominator"] += 1
                    bucket["false"] += after is False
                    bucket["unknown"] += after is None
        policy_scores[policy] = {k:{**prompt_summary(v),"sum":sum(float(x) for _,x in v)} for k,v in values.items()}
        transitions[policy] = changes
    for name in ("ocr@1","geneval@1"):
        candidate_scores[name] = {k:ratio([q["comparisons"][name][k] for q in queries],[True]*64)
                                  for k in ("safe_repair","known_harm","has_unknown","safe_visibility_gain")}
    gate = {"at_least8_safe_accepted":sum(safe)>=8,"conservative_precision_at_least80pct":bool(sum(accepted)) and sum(safe)/sum(accepted)>=.8,
            "zero_known_harms_accepted":sum(harm)==0,"unresolved_accepted_at_most10pct":bool(sum(accepted)) and sum(unclear)/sum(accepted)<=.1}
    campaign = ROOT / "experiments/local_expert_v12"
    write_json(campaign / "results.json",{"scope":"prospective train-only frozen-state candidate/selector audit; no trained-policy gain claim",
               "calibration":calibration,"candidate_scores":candidate_scores,"policy_scores":policy_scores,"protected_transitions":transitions,
               "learning_gate":{"checks":gate,"passed":all(gate.values())},"queries":queries,"joined_review":joined,
               "gpu_result":read_json(run / "result.json"),"review_inputs":inputs})
    for n in ("observations","observations_frozen"):
        shutil.copyfile(review / f"{n}.json",campaign / f"{n}.json")
    print(json.dumps({"calibration":calibration,"learning_gate":gate},indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("stage",choices=["prepare","freeze","summarize"])
    p.add_argument("--run",required=True)
    p.add_argument("--review",required=True)
    args = p.parse_args()
    globals()[args.stage](Path(args.run).resolve(),Path(args.review).resolve())
