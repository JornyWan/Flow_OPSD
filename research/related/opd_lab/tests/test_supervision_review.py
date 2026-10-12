import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_supervision as review
from opdlab.io import sha256, write_json


def test_review_keeps_paths_separate_and_rebuilds_paired_metrics(tmp_path, monkeypatch):
    run, out = tmp_path / "run", tmp_path / "review"
    run.mkdir(); out.mkdir()
    (run / "REVIEW_RUBRIC.md").write_text("fixture")
    write_json(run / "result.json", {})
    points = [{"key": str(i), "original": i < 6} for i in range(9)]
    write_json(run / "probe_plan.json", {"points": points, "updates": 128})
    metrics = [{"arm": a, "step":128, "key":str(i), "alignment":{"relative_target_mse":.1}}
               for a in ("single","multi","random_matched") for i in range(9)]
    (run / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics)+"\n")
    items, keys, observations = [], [], []
    cohorts = {1:["positive_fresh_seed","all_fresh","added_prompts_fresh"],
               2:["positive_fresh_seed","all_fresh","other_positive_prompts_fresh"],
               3:["preservation_fresh_seed","all_fresh"],4:["unfitted_prompt_fresh_seed","all_fresh"]}
    for case, (pid,path) in enumerate([(1,"full"),(2,"full"),(3,"full"),(4,"full"),(1,"single"),(2,"single")],1):
        views, mapping, images = {}, {}, {}
        for letter, model in zip("ABCD", ("joint128","single","multi","random_matched")):
            text = model == "multi" or pid >= 3
            if path == "single" and model == "multi" and pid == 1:
                text = False
            obs = {"clauses":{"text":text,"count":True},"full_visibility":True,"visible_text":"fixture","notes":"synthetic"}
            name=f"{case}-{letter}.png"
            (out/name).write_text(f"{pid}-{model}-{text}")
            images[letter]=name; views[letter]=obs
            mapping[letter]={"model":model,"id":str(pid),"pair_id":str(pid),"path":"full" if model=="joint128" else path,
                             "cohorts":cohorts[pid],"seed":pid,"image_sha256":sha256(out/name)}
        items.append({"case":case,"images":images,"constraints":[{"id":"text"},{"id":"count"}]})
        keys.append({"case":case,"pair_id":str(pid),"path":path,"mapping":mapping})
        observations.append({"case":case,"views":views})
    for n,v in (("items",items),("key",keys),("inherited",{})):
        write_json(out/f"{n}.json",v)
    write_json(out/"inputs.json",{"cases":6,"views":24,"result_sha256":sha256(run/"result.json"),"review_script_sha256":sha256(review.__file__)})
    (out/"observations_draft.jsonl").write_text("\n".join(json.dumps(o) for o in observations)+"\n")
    review.freeze(run,out,tmp_path)
    monkeypatch.setattr(review,"validate",lambda _: [])
    r=review.build_report(run,out)
    assert r["scores"]["positive_fresh_seed"]["full"]["multi"]["complete_scene"]["sum"]==2
    assert r["scores"]["positive_fresh_seed"]["single"]["multi"]["complete_scene"]["sum"]==1
    assert r["multi_vs_other_complete"]["positive_fresh_seed"]["full"]["single"]["mean"]==1
    assert r["multi_vs_other_complete"]["positive_fresh_seed"]["full"]["single"]["prompts"]==2
    assert r["alignment"]["multi"]["added"]=={"states":3,"median":.1}
    assert r["promising_pilot"]["passed"]
