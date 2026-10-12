import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import review_action_transfer as review
from opdlab.io import sha256, write_json


def fixture(tmp_path):
    run, out = tmp_path / "run", tmp_path / "review"
    run.mkdir()
    out.mkdir()
    (run / "REVIEW_RUBRIC.md").write_text("synthetic rubric")
    write_json(run / "result.json", {})
    write_json(run / "probe_plan.json", {"scope": "synthetic"})
    items, labels, key, refs, queries = [], [], [], [], []
    for case in range(1, 19):
        fresh = case <= 12
        modes = ("student_single", "ocr@1", "geneval@1") if fresh else ("student_single",)
        images, views, mapping = {}, {}, {}
        for letter, mode in zip("ABC", modes):
            name = f"{case}-{letter}.png"
            (out / name).write_bytes(name.encode())
            images[letter] = name
            obs = {"clauses": {"text": True, "count": True}, "full_visibility": True, "visible_text": "WORD", "notes": "synthetic"}
            if mode == "student_single":
                obs["clauses"]["count"] = None
            views[letter] = obs
            mapping[letter] = {"mode": mode}
        items.append({"case": case, "images": images, "constraints": [{"id": "text"}, {"id": "count"}]})
        labels.append({"case": case, "views": views})
        key.append({"case": case, "pair_id": str(case), "mapping": mapping})
        queries.append({"pair_id": str(case), "id": str((case - 1) // 2 if fresh else case),
                        "cohort": review.FRESH if fresh else review.OLD})
        for mode in ("joint128", "label_selected"):
            refs.append({"pair_id": str(case), "mode": mode, "image_sha256": "base"})
    for n, values in (("references", refs), ("queries", queries)):
        (run / f"{n}.jsonl").write_text("\n".join(json.dumps(v) for v in values) + "\n")
    for n, value in (("items", items), ("key", key), ("inherited", {})):
        write_json(out / f"{n}.json", value)
    write_json(out / "inputs.json", {"result_sha256": sha256(run / "result.json"),
               **{f"{n}_sha256": sha256(out / f"{n}.json") for n in ("items", "key", "inherited")}})
    (out / "observations_draft.jsonl").write_text("\n".join(json.dumps(o) for o in labels) + "\n")
    return run, out, labels


def test_inherited_pixels_and_duplicates_cannot_change(tmp_path):
    _, out, labels = fixture(tmp_path)
    old = copy.deepcopy(labels[0]["views"]["A"])
    old["clauses"]["count"] = True
    write_json(out / "inherited.json", {"1": {"A": old}})
    with pytest.raises(AssertionError, match="inherited"):
        review.check_labels(out)
    write_json(out / "inherited.json", {})
    (out / "1-A.png").write_bytes((out / "1-B.png").read_bytes())
    with pytest.raises(AssertionError, match="duplicate"):
        review.check_labels(out)


def test_unknown_not_promoted_to_mechanistic_failure(tmp_path, monkeypatch):
    run, out, _ = fixture(tmp_path)
    review.freeze(run, out)
    base = {"clauses": {"text": False, "count": True}, "full_visibility": True, "visible_text": "WOR", "notes": "synthetic"}
    monkeypatch.setattr(review, "validate", lambda run: None)
    monkeypatch.setattr(review, "inherited_labels", lambda: {"base": base})
    result = review.build_report(run, out)
    signals = result["diagnostic_signals"][review.FRESH]
    assert signals["any_teacher_safe"]["count"] == 12
    assert signals["teacher_safe_local_not_safe"]["count"] == 12
    assert signals["teacher_safe_local_determinate_not_safe"]["count"] == 0
    score = result["scores"][review.FRESH]["student_single"]
    assert score["has_unknown"]["sum"] == 12
    assert score["known_harm"]["sum"] == score["complete_scene"]["sum"] == 0
    assert score["has_unknown"]["prompts"] == 6
    (out / "key.json").write_text("[]")
    with pytest.raises(AssertionError):
        review.build_report(run, out)
