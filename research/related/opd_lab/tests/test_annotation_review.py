import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_annotation_probe as review
from opdlab.io import read_json, sha256, write_json


def fixture(tmp_path):
    run, out = tmp_path / "run", tmp_path / "review"
    run.mkdir()
    out.mkdir()
    (run / "REVIEW_RUBRIC.md").write_text("fixture rubric")
    items, labels, key = [], [], []
    cohorts = ["positive_fresh_seed"] * 12 + ["preservation_fresh_seed"] * 12 + ["unfitted_prompt_fresh_seed"] * 8 + ["fitting_seed_diagnostic"] * 6
    for case, cohort in enumerate(cohorts, 1):
        images, views, mapping = {}, {}, {}
        for letter, model in zip("ABCD", ("joint128", *review.ARMS)):
            name = f"{case}-{letter}.png"
            (out / name).write_bytes(name.encode())
            images[letter] = name
            obs = {"clauses": {"text": True, "objects": True}, "full_visibility": True, "visible_text": "WORD", "notes": "synthetic"}
            if cohort == "positive_fresh_seed" and model != "label_selected":
                obs["clauses"]["text"] = False
            views[letter] = obs
            mapping[letter] = {"model": model, "cohort": cohort, "pair_id": str(case), "seed": case,
                               "id": f"{cohort}-{(case - 1) // 2 if case <= 24 else case}", "utility": 0.0}
        items.append({"case": case, "images": images, "constraints": [{"id": "text"}, {"id": "objects"}]})
        labels.append({"case": case, "views": views})
        key.append({"case": case, "mapping": mapping})
    write_json(out / "items.json", items)
    write_json(out / "key.json", key)
    write_json(out / "inherited.json", {"1": {"A": copy.deepcopy(labels[0]["views"]["A"])}})
    write_json(out / "inputs.json", {})
    write_json(run / "result.json", {})
    (run / "metrics.jsonl").write_text("\n".join(json.dumps({"arm": "label_selected", "step": 128, "label_target_alignment": {"relative_target_mse": .25}}) for _ in range(6)))
    return run, out, labels


def draft(out, labels):
    (out / "observations_draft.jsonl").write_text("\n".join(json.dumps(r) for r in labels) + "\n")


def test_freeze_rejects_changed_inherited_or_identical_pixel_labels(tmp_path):
    run, out, labels = fixture(tmp_path)
    labels[0]["views"]["A"]["clauses"]["text"] = True
    draft(out, labels)
    with pytest.raises(AssertionError, match="old pixels"):
        review.freeze(run, out)
    labels[0]["views"]["A"]["clauses"]["text"] = False
    (out / "1-B.png").write_bytes((out / "1-A.png").read_bytes())
    draft(out, labels)
    with pytest.raises(AssertionError, match="identical pixels"):
        review.freeze(run, out)
    assert not (out / "observations_frozen.json").exists()


def test_paired_scoring_keeps_unknown_and_clusters_replicate_seeds(tmp_path, monkeypatch):
    run, out, labels = fixture(tmp_path)
    # Baseline objects are known true; selected becomes unknown, not a known harm.
    labels[12]["views"]["B"]["clauses"]["objects"] = None
    draft(out, labels)
    review.freeze(run, out)
    monkeypatch.setattr(review, "validate", lambda run: None)
    result = review.build_report(run, out)
    primary = result["label_vs_other_complete"]["positive_fresh_seed"]["random_matched"]
    assert primary["sum"] == 12 and primary["prompts"] == 6 and primary["observations"] == 12
    preserve = result["scores"]["preservation_fresh_seed"]["label_selected"]
    assert preserve["has_unknown"]["sum"] == preserve["objects_true_to_unknown"]["sum"] == 1
    assert preserve["known_harm"]["sum"] == 0
    assert preserve["complete_scene"]["sum"] == 11
    assert result["promising_pilot"]["checks"]["primary_at_least2_more_complete_than_random"]
    assert not result["promising_pilot"]["passed"]
    assert not result["promising_pilot"]["checks"]["preservation_complete_not_worse"]
    # Key edits after freezing must fail before score recomputation.
    (out / "key.json").write_text("[]")
    with pytest.raises(AssertionError):
        review.build_report(run, out)
