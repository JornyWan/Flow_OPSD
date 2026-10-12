from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import finish_supervision_run as finish
from opdlab.io import read_json,write_json


def test_success_callback_only_prepares_review_and_preserves_existing_labels(tmp_path,monkeypatch):
    run,review,campaign=(tmp_path/n for n in ("run","review","campaign"))
    run.mkdir();campaign.mkdir()
    write_json(run/"probe_plan.json",{"points":[{}],"evaluation_images":4,"reference_images":1,"expected_counts":{"backward_queries":12}})
    write_json(run/"result.json",{"arms":{"single":{"historical_replay_exact":True,"optimizer_updates":1}},"counts":{"backward_queries":12},"evaluation_images":4})
    (run/"references.jsonl").write_text('{"exact":true}\n')
    monkeypatch.setattr(finish,"validate",lambda _: [])
    def prepare(r,v,c):
        v.mkdir()
        write_json(v/"inputs.json",{"cases":1,"result_sha256":finish.sha256(r/"result.json")})
    monkeypatch.setattr(finish,"prepare",prepare)
    finish.finish(run,review,campaign)
    status=read_json(campaign/"progress.json")
    assert status["status"]=="ready_for_image_review" and not status["results_available"] and not status["scientific_analysis_complete"]
    assert not (review/"observations.json").exists()
    (review/"observations_draft.jsonl").write_text('existing manual draft\n')
    finish.finish(run,review,campaign)
    assert (review/"observations_draft.jsonl").read_text()=='existing manual draft\n'
