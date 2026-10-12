"""Freeze the phase12 image pool for phase13 verifier development."""
import argparse
import json
import shutil

from opdlab.config import Experiment
from opdlab.evidence_verifier import SPEC
from opdlab.io import ROOT,read_json,sha256,write_json
from opdlab.runtime import prepare


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--attempt",default="001")
    parser.add_argument("--site",default="site.local.json")
    args = parser.parse_args()
    old = ROOT / "runs/le12_collect_002"
    campaign = ROOT / "experiments/local_expert_v13"
    rows = [json.loads(s) for s in (old / "candidates.jsonl").read_text().splitlines()]
    config = Experiment.load(old / "config.json")
    run = prepare(config,ROOT / f"runs/le13_measure_{args.attempt}",read_json(args.site))
    queries = []
    for row in rows:
        candidates = [{"name":c["name"],"image":str(old / c["image"]),"image_sha256":c["image_sha256"],
                       "distance":c["distance"],"v1":c} for c in row["candidates"]]
        queries.append({"id":row["id"],"seed":row["seed"],"v1_selected":row["selected"],"candidates":candidates})
    write_json(run / "audit_plan.json",{"version":1,"scope":"development on previously reviewed phase12 images; no new validation",
               "verifier_spec":SPEC,"source":str(old),"queries":queries})
    shutil.copyfile(campaign / "PROTOCOL.md",run / "PROTOCOL.md")
    inputs = [old / n for n in ("result.json","candidates.jsonl","probe_plan.json")]
    inputs += [ROOT / "experiments/local_expert_v12" / n for n in ("report_index.json","observations.json","observations_frozen.json","results.json")]
    inputs += [run / "audit_plan.json",run / "PROTOCOL.md"]
    write_json(run / "audit_attestation.json",{"inputs":{str(p):sha256(p) for p in inputs}})
    write_json(campaign / f"attempt_{args.attempt}.json",{"run":str(run),"audit_plan_sha256":sha256(run / "audit_plan.json"),
               "protocol_sha256":sha256(run / "PROTOCOL.md"),"input_images":192,"generated_images":0})
    print(run)


if __name__ == "__main__":
    main()
