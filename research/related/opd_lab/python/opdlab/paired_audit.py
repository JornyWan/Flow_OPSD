"""Frozen phase14 paired evidence interface; diagnostic only, no training imports."""
import json
import re
import unicodedata

from .evidence_dominance import admissible_repair
from .vlm_audit import OBSERVATION_PROMPT, _object


SPEC = {
    "version": "paired_evidence_dev1", "seed": 14014,
    "min_pixels": 3136, "max_pixels": 262144,
    "observation_max_new_tokens": 256, "pair_max_new_tokens": 640,
    "max_input_tokens_per_call": 12288, "do_sample": False,
    "observation_calls": 192, "pair_calls": 320, "self_pairs": 64,
    "selection": "order_consensus_literal_guard_preserve_possible_baseline_success",
}

PAIR_INSTRUCTION = '''Compare image A (first image) and image B (second image) using ONLY the explicit requirements below. Neither image is assumed better. For each requirement describe concrete evidence in BOTH images, then judge A and B separately and the A-to-B change. A repair means a requirement was false in A and becomes true in B; damage means true becomes false; same means both true or both false; unknown means either value cannot be determined. Better legibility alone is not a repair when the requested word was already readable. Copy letters literally, including extra/repeated letters and small labels; do not correct spelling using the expected word. Count individual objects, including partially visible objects, without inventing hidden ones. A group is not an extra object. Exact count, color and relation are distinct from full visibility. For full_visibility, missing, clearly cropped or substantially occluded required objects fail; inspect image edges and object parts, including chair legs and bowl rims. Unknown is allowed PER FIELD: do not discard known fields because another is uncertain. Do not replace evidence with a restatement of the request. Prior target-free observations are fallible observations, not ground truth; use the images to check them.
Return only JSON: {"clauses": [{"id": "requested id", "evidence": "A: visible facts; B: visible facts", "A": true, "B": false, "change": "damage"}]}.
Include every requested id exactly once, including full_visibility. A and B must be true, false or null. evidence must be a string or null. change must be repair, damage, same or unknown. No overall preference or extra fields.
Requirements: '''


def requirements(record):
    """Compile the existing encoded clauses, never the unchecked full prompt."""
    out, required = [], set()
    for c in record["constraints"]:
        if c["kind"] == "ocr":
            sentence = f'The only visible text anywhere must be exactly {json.dumps(c["text"])} (case-insensitive); no extra letters, words, digits or punctuation.'
        elif c["kind"] == "geneval":
            meta = c["metadata"]
            if set(meta) - {"tag", "include"}:
                raise ValueError("Unsupported constraint metadata")
            included, parts = meta["include"], []
            for i, obj in enumerate(included):
                if set(obj) - {"class", "count", "color", "position"}:
                    raise ValueError("Unsupported object requirement")
                required.add(obj["class"])
                parts.append(f'Exactly {obj["count"]} object(s) of class {obj["class"]} must be present' +
                             (f', all {obj["color"]}' if "color" in obj else "") + ".")
                if "position" in obj:
                    direction, ref = obj["position"]
                    if (direction not in ("left of", "right of") or type(ref) is not int or
                        ref < 0 or ref >= len(included) or ref == i):
                        raise ValueError("Unsupported relation")
                    parts.append(f'The {obj["class"]} must be to the {direction.upper()} the {included[ref]["class"]}, as seen by the viewer.')
            sentence = " ".join(parts)
        else:
            raise ValueError("Unsupported constraint kind")
        out.append({"id": c["id"], "requirement": sentence})
    out.append({"id": "full_visibility", "requirement":
                "All required objects (" + ", ".join(sorted(required)) +
                ") must be present, complete inside the image and without substantial occlusion. Missing objects fail this field."})
    if len({c["id"] for c in out}) != len(out):
        raise ValueError("Duplicate/reserved clause id")
    return out


def pair_request(clauses, observations):
    return PAIR_INSTRUCTION + json.dumps(clauses, sort_keys=True) + \
        "\nTarget-free observations, in the same A/B order: " + json.dumps(dict(zip(("A", "B"), observations)), sort_keys=True)


def json_object(raw):
    value = raw.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.S)
        if not match:
            raise ValueError("Malformed JSON fence")
        value = match.group(1)
    obj = json.loads(value, object_pairs_hook=_object,
                     parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if not isinstance(obj, dict):
        raise ValueError("Expected object")
    return obj


def parse_observation(raw):
    out = {"text_lines": None, "text_uncertain": None, "scene_description": None}
    errors = []
    try:
        obj = json_object(raw)
    except (ValueError, TypeError) as exc:
        return out, [str(exc)]
    for key in out:
        v = obj.get(key)
        valid = (isinstance(v, list) and all(isinstance(x, str) for x in v)) if key == "text_lines" else \
                type(v) is bool if key == "text_uncertain" else isinstance(v, str)
        if valid or (key in obj and v is None):
            out[key] = v
        else:
            errors.append(f"Invalid/missing {key}")
    if set(obj) - set(out):
        errors.append("Unexpected observation keys")
    return out, errors


def change(a, b):
    if a is None or b is None:
        return "unknown"
    return "same" if a == b else "repair" if b else "damage"


def parse_pair(raw, ids):
    empty = lambda: {"A": None, "B": None, "change": "unknown", "evidence": None}
    out, errors = {k: empty() for k in ids}, []
    try:
        obj = json_object(raw)
        if not isinstance(obj.get("clauses"), list):
            raise ValueError("Missing clause list")
    except (ValueError, TypeError) as exc:
        return out, [str(exc)]
    if set(obj) != {"clauses"}:
        errors.append("Unexpected top-level keys")
    for k in ids:
        rows = [c for c in obj["clauses"] if isinstance(c, dict) and c.get("id") == k]
        if len(rows) != 1:
            errors.append(f"{k}: missing/duplicate clause")
            continue
        c = rows[0]
        if set(c) != {"id", "A", "B", "change", "evidence"} or not (isinstance(c["evidence"], str) or c["evidence"] is None):
            errors.append(f"{k}: invalid fields/evidence")
            continue
        out[k]["evidence"] = c["evidence"]
        for side in ("A", "B"):
            if type(c[side]) is bool or c[side] is None:
                out[k][side] = c[side]
            else:
                errors.append(f"{k}: invalid {side}")
        if c["change"] != change(out[k]["A"], out[k]["B"]):
            out[k] = empty()
            errors.append(f"{k}: inconsistent change")
        elif any(out[k][s] is not None for s in ("A", "B")) and not (c["evidence"] and c["evidence"].strip()):
            out[k] = empty()
            errors.append(f"{k}: determinate claim without evidence")
        else:
            out[k]["change"] = c["change"]
    if any(not isinstance(c, dict) or c.get("id") not in out for c in obj["clauses"]):
        errors.append("Unexpected/invalid clause")
    return out, errors


def literal_label(observation, expected):
    if observation["text_uncertain"] is not False or observation["text_lines"] is None:
        return None
    normalize = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()
    return normalize("\n".join(observation["text_lines"])) == normalize(expected)


def endpoints(parsed, reverse=False):
    sides = ("B", "A") if reverse else ("A", "B")
    return tuple({"clauses": {k: v[s] for k, v in parsed.items() if k != "full_visibility"},
                  "full_visibility": parsed["full_visibility"][s]} for s in sides)


def agreed_endpoints(first, second):
    a, b = endpoints(first), endpoints(second, reverse=True)
    return tuple({"clauses": {k: x["clauses"][k] if x["clauses"][k] == y["clauses"][k] else None for k in x["clauses"]},
                  "full_visibility": x["full_visibility"] if x["full_visibility"] == y["full_visibility"] else None}
                 for x, y in zip(a, b))


def guarded_endpoints(pair, literal):
    return tuple({"clauses": {**e["clauses"], "text": e["clauses"]["text"] if e["clauses"]["text"] == t else None},
                  "full_visibility": e["full_visibility"]} for e, t in zip(pair, literal))


def choose_pairs(pairs, distances):
    eligible = []
    for ordinal, name in enumerate(("ocr@1", "geneval@1")):
        a, b = pairs[name]
        if admissible_repair(a, b):
            repairs = sum(v is False and b["clauses"][k] is True for k, v in a["clauses"].items())
            eligible.append(((repairs, -distances[name], -ordinal), name))
    return max(eligible)[1] if eligible else "noop"
