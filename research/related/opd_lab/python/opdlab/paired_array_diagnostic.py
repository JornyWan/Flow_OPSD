"""Posthoc container-only compatibility diagnostic; frozen dev1 is unchanged."""
import json
import re

from .paired_audit import parse_pair
from .vlm_audit import _object


VERSION = "paired_array_container_posthoc1"


def parse_array_compatible(raw, ids):
    value = raw.strip()
    if value.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.S)
        if not match:
            return parse_pair(raw, ids)
        value = match.group(1)
    try:
        obj = json.loads(value, object_pairs_hook=_object,
                         parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (ValueError, TypeError):
        return parse_pair(raw, ids)
    # Exactly one lossless structural mapping; no values/ids/strings are inferred.
    if isinstance(obj, list):
        return parse_pair(json.dumps({"clauses": obj}, allow_nan=False), ids)
    return parse_pair(raw, ids)
