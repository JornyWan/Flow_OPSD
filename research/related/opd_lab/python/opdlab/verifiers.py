"""Non-differentiable, per-clause terminal verification with explicit validity."""
from __future__ import annotations

from collections import Counter
import re
import unicodedata

import numpy as np
import torch

from .method import Verdict


def normalize_text(value):
    # This defines our OCR exact-match metric; upstream substring reward is different.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()


def text_progress(actual, expected):
    a, b = normalize_text(actual), normalize_text(expected)
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return a == b, 1 - row[-1] / max(len(a), len(b), 1)


class ToyVerifier:
    def __init__(self, config, site=None, device=None, records=None):
        self.counters = Counter()

    def __call__(self, image, record):
        self.counters["verifier_images"] += 1
        sat, progress = [], []
        for c in record["constraints"]:
            if c["kind"] != "toy":
                raise ValueError("Toy backend requires toy constraints")
            error = abs(float(image[0, c["axis"]]) - c["target"])
            sat.append(error <= c["tolerance"])
            progress.append(max(0.0, 1 - error))
        return Verdict(tuple(sat), tuple(progress), quality=1.0)


class JointVerifier:
    version = "joint_constraints_v1"

    def __init__(self, config, site, device, records):
        kinds = {c["kind"] for r in records for c in r["constraints"]}
        if kinds - {"ocr", "geneval"}:
            raise ValueError("SD3 verifier supports OCR and explicit GenEval clauses")
        self.counters = Counter()
        self.ocr, self.geneval, self.quality = None, None, None
        # Fail at setup for missing weights/dependencies, rather than silently reject all training targets.
        if "ocr" in kinds:
            from flow_grpo.ocr import OcrScorer
            self.ocr = OcrScorer(use_gpu=False).ocr
        if "geneval" in kinds:
            from flow_grpo.gen_eval import load_geneval
            self.geneval = load_geneval(str(device))
        if config.quality_floor is not None or config.quality_drop_tolerance is not None:
            from flow_grpo.aesthetic_scorer import AestheticScorer
            self.quality = AestheticScorer(torch.float32, str(device))

    @torch.no_grad()
    def __call__(self, image, record):
        self.counters["verifier_images"] += 1
        n = len(record["constraints"])
        try:
            sat, progress, details = [], [], {"clauses": []}
            recognized = None
            for constraint in record["constraints"]:
                if constraint["kind"] == "ocr":
                    if recognized is None:
                        result = self.ocr.ocr(np.asarray(image), cls=False)
                        lines = [] if not result or result[0] is None else result[0]
                        recognized = " ".join(item[1][0] for item in lines)
                        details["ocr"] = {"recognized_text": recognized, "lines": [
                            {"text": item[1][0], "confidence": float(item[1][1]),
                             "box": [[float(x) for x in point] for point in item[0]]} for item in lines]}
                        self.counters["ocr_images"] += 1
                    exact, score = text_progress(recognized, constraint["text"])
                    sat.append(exact)
                    progress.append(score)
                    details["clauses"].append({"id": constraint["id"], "expected_text": constraint["text"],
                                               "normalized_recognized": normalize_text(recognized),
                                               "exact": exact, "progress": score})
                else:
                    metadata = {**constraint["metadata"], "prompt": record["prompt"]}
                    scores, _, strict, _, _ = self.geneval([image], [metadata], only_strict=True)
                    self.counters["geneval_clause_images"] += 1
                    sat.append(bool(strict[0]))
                    # Upstream progress may be negative on overcounting; our named metric clips it.
                    progress.append(float(np.clip(scores[0], 0, 1)))
                    details["clauses"].append({"id": constraint["id"], "metadata": metadata,
                                               "strict": bool(strict[0]), "upstream_progress": float(scores[0])})
            quality = None
            if self.quality:
                quality = float(self.quality([image])[0])
                self.counters["aesthetic_images"] += 1
            return Verdict(tuple(sat), tuple(progress), quality=quality, details=details)
        except Exception as exc:
            self.counters["verifier_errors"] += 1
            # A model/data runtime error is unknown, never a negative training label.
            return Verdict((False,) * n, (0.0,) * n, valid=False,
                           error=f"{type(exc).__name__}: {exc}")


def build_verifier(config, site, device, records):
    cls = ToyVerifier if config.backend == "toy" else JointVerifier
    return cls(config, site, device, records)
