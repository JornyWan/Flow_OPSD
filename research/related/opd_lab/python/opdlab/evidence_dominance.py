"""Posthoc development ablation: preserve possibly correct baseline clauses.

This is separate from frozen endpoint_evidence_v2_dev1 and is not wired to training.
"""


VERSION = "evidence_dominance_posthoc1"


def admissible_repair(base, candidate):
    if set(base["clauses"]) != set(candidate["clauses"]):
        raise ValueError("Mismatched clauses")
    a = {**base["clauses"],"visibility":base["full_visibility"]}
    b = {**candidate["clauses"],"visibility":candidate["full_visibility"]}
    repair = any(a[k] is False and b[k] is True for k in base["clauses"])
    # None represents {False, True}; a known-true candidate is no worse under
    # either baseline interpretation. Unknown->true does not earn repair credit.
    return repair and all(v is not None for v in b.values()) and all(a[k] is False or b[k] is True for k in a)


def choose_dominance(evidence, distances):
    base = evidence["noop"]
    eligible = []
    for ordinal,name in enumerate(("ocr@1","geneval@1")):
        candidate = evidence[name]
        if admissible_repair(base,candidate):
            repairs = sum(v is False and candidate["clauses"][k] is True for k,v in base["clauses"].items())
            eligible.append(((repairs,-distances[name],-ordinal),name))
    return max(eligible)[1] if eligible else "noop"
