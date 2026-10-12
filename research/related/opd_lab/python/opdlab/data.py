"""Explicit per-image constraints and split manifests; demo data is not a benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from .io import digest


def load_records(path, split):
    records = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids, prompts, groups = set(), {}, {}
    for row in records:
        if set(row) - {"id", "prompt", "split", "group", "constraints"}:
            raise ValueError(f"Unknown data fields in {row.get('id')}")
        for key in ("id", "prompt", "split", "group"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Missing string field {key}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        for mapping, key in ((prompts, row["prompt"].strip().casefold()), (groups, row["group"])):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError(f"Cross-split leakage in prompt/group: {key}")
            mapping[key] = row["split"]
        constraints = row.get("constraints")
        if not isinstance(constraints, list) or not constraints:
            raise ValueError("Every image needs at least one explicit constraint")
        constraint_ids = set()
        for c in constraints:
            if not c.get("id") or c["id"] in constraint_ids:
                raise ValueError("Constraint ids must be unique within an image")
            constraint_ids.add(c["id"])
            if c.get("kind") == "ocr":
                if set(c) != {"id", "kind", "text"} or not isinstance(c["text"], str) or not c["text"].strip():
                    raise ValueError("OCR constraint needs nonempty text")
            elif c.get("kind") == "geneval":
                if set(c) != {"id", "kind", "metadata"}:
                    raise ValueError("GenEval clause needs explicit metadata")
                meta = c["metadata"]
                if set(meta) - {"tag", "include"}:
                    raise ValueError("v1 GenEval clauses support tag/include only; upstream strict reward ignores exclude")
                if meta.get("tag") not in {"single_object", "two_object", "counting", "colors", "position", "color_attr"}:
                    raise ValueError("Unknown GenEval tag")
                if not meta.get("include") and not meta.get("exclude"):
                    raise ValueError("Empty GenEval clause")
                # A clause can include reference objects needed by a relation.
                for i, obj in enumerate(meta.get("include", [])):
                    if set(obj) - {"class", "count", "color", "position"}:
                        raise ValueError("Unknown object constraint field")
                    if type(obj.get("count")) is not int or obj["count"] < 1 or not obj.get("class"):
                        raise ValueError("Included objects need positive integer count and class")
                    if "color" in obj and obj["color"] not in {
                        "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white"}:
                        raise ValueError("Unsupported GenEval color")
                    if "position" in obj:
                        rel, reference = obj["position"]
                        if rel not in {"left of", "right of", "above", "below"} or type(reference) is not int or not 0 <= reference < i:
                            raise ValueError("Relation needs an earlier reference object and supported relation")
            elif c.get("kind") == "toy":
                if set(c) != {"id", "kind", "axis", "target", "tolerance"} or c["axis"] not in {0, 1} or c["tolerance"] <= 0:
                    raise ValueError("Invalid toy clause")
            else:
                raise ValueError(f"Unknown constraint kind: {c.get('kind')}")
    selected = [row for row in records if row["split"] == split]
    if not selected:
        raise ValueError(f"Empty split: {split}")
    return selected


def records_digest(records):
    return digest(records)
