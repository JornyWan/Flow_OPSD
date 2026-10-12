"""Versioned, abstaining endpoint evidence. Does not replace joint_constraints_v1."""
from collections import Counter
import inspect
import json

import numpy as np
from PIL import Image, ImageOps
import torch

from .verifiers import normalize_text


SPEC = {
    "version": "endpoint_evidence_v2_dev1",
    "ocr_scale": 2, "crop_padding": 8, "ocr_min_confidence": .8,
    "row_overlap": .5, "max_text_rows": 32,
    "object_padding": 16, "boundary_margin": 3,
    "selection": "confirmed_repair_all_labels_known_no_true_to_false",
}


def consensus(values):
    return values[0] if values and values[0] is not None and all(v == values[0] for v in values) else None


def text_rows(lines):
    """Assemble boxes on the same text row; retain every row and character."""
    groups = []
    for line in sorted(lines, key=lambda x: (min(p[1] for p in x["box"]), min(p[0] for p in x["box"]))):
        xs, ys = zip(*line["box"])
        box = [min(xs), min(ys), max(xs), max(ys)]
        match = None
        for group in groups:
            b = group["box"]
            overlap = max(0, min(box[3], b[3]) - max(box[1], b[1]))
            if overlap / max(1, min(box[3]-box[1], b[3]-b[1])) >= SPEC["row_overlap"]:
                match = group
                break
        if match is None:
            match = {"box": box.copy(), "lines": []}
            groups.append(match)
        match["lines"].append(line)
        b = match["box"]
        match["box"] = [min(b[0],box[0]),min(b[1],box[1]),max(b[2],box[2]),max(b[3],box[3])]
    for group in groups:
        ordered = sorted(group.pop("lines"), key=lambda x: min(p[0] for p in x["box"]))
        # Join separate boxes, not spaces within recognized words. Never drop glyphs.
        group["text"] = "".join(normalize_text(x["text"]) for x in ordered)
        group["confidence"] = min(x["confidence"] for x in ordered)
    return groups


def text_label(readings, expected):
    """Agreement must include the full image: cropped sign alone cannot certify."""
    texts = []
    for rows in readings:
        if not rows or any(r["confidence"] < SPEC["ocr_min_confidence"] for r in rows):
            return None
        texts.append("\n".join(normalize_text(r["text"]) for r in rows))
    # Different erroneous strings may agree that the exact requested word is
    # absent. Agreement is on the clause, not on a chosen OCR transcription.
    return consensus([text == normalize_text(expected) for text in texts])


def visibility_support(details, constraints, size, offset=0):
    """A declared bbox heuristic, to be audited; not human visibility ground truth."""
    outcomes = []
    width, height = size
    margin = SPEC["boundary_margin"]
    for detail, constraint in zip(details, constraints):
        for req in constraint["metadata"]["include"]:
            boxes = detail["boxes"].get(req["class"], [])
            if len(boxes) < req["count"]:
                outcomes.append(False)
                continue
            outcomes.append(all(b[0]-offset > margin and b[1]-offset > margin
                                and b[2]-offset < width-margin and b[3]-offset < height-margin for b in boxes))
    return all(outcomes) if outcomes else None


def choose(evidence, distances):
    """Require a discrete repair and complete knowledge; stable ties prefer less change."""
    names = ["noop", *[n for n in ("ocr@1", "geneval@1") if n in evidence]]
    if set(names) != set(evidence):
        raise ValueError("Expected noop and the declared expert candidates")
    base = evidence["noop"]
    decisions, choices = {}, []
    for ordinal, name in enumerate(names):
        candidate = evidence[name]
        if name == "noop":
            continue
        if set(base["clauses"]) != set(candidate["clauses"]):
            raise ValueError("Mismatched evidence clauses")
        repaired = [k for k,v in base["clauses"].items() if v is False and candidate["clauses"][k] is True]
        harmed = [k for k,v in base["clauses"].items() if v is True and candidate["clauses"][k] is False]
        vis_harm = base["full_visibility"] is True and candidate["full_visibility"] is False
        unknown = any(v is None for e in (base,candidate) for v in [*e["clauses"].values(),e["full_visibility"]])
        reasons = (["unknown"] if unknown else []) + (["protected_harm"] if harmed or vis_harm else [])
        if not repaired:
            reasons.append("no_confirmed_repair")
        decisions[name] = {"accepted":not reasons,"repaired":repaired,"harmed":harmed,"visibility_harm":vis_harm,"reasons":reasons}
        if not reasons:
            choices.append(((len(repaired), -distances[name], -ordinal), name))
    return {"selected":max(choices)[1] if choices else "noop","candidates":decisions}


class EvidenceVerifier:
    def __init__(self, ocr, geneval):
        self.ocr, self.counters = ocr, Counter()
        # Access the pinned upstream evaluator's existing detail output, without
        # patching its code, thresholds, detector or color classifier.
        self.evaluate_image = inspect.getclosurevars(inspect.unwrap(geneval)).nonlocals["evaluate_image"]

    def detect_text(self, image, scale):
        scaled = image.resize((image.width*scale,image.height*scale), Image.Resampling.LANCZOS) if scale != 1 else image
        result = self.ocr.ocr(np.asarray(scaled), cls=False)
        self.counters["ocr_detection_views"] += 1
        raw = [] if not result or result[0] is None else result[0]
        return [{"box":[[float(x)/scale,float(y)/scale] for x,y in row[0]],
                 "text":row[1][0],"confidence":float(row[1][1])} for row in raw]

    def reread(self, image, rows):
        if len(rows) > SPEC["max_text_rows"]:
            return []
        reads = []
        for row in rows:
            x1,y1,x2,y2 = row["box"]
            p = SPEC["crop_padding"]
            box = [max(0,int(x1)-p),max(0,int(y1)-p),min(image.width,int(np.ceil(x2))+p),min(image.height,int(np.ceil(y2))+p)]
            crop = image.crop(box)
            crop = crop.resize((crop.width*SPEC["ocr_scale"],crop.height*SPEC["ocr_scale"]), Image.Resampling.LANCZOS)
            result = self.ocr.ocr(np.asarray(crop), det=False, cls=False)
            self.counters["ocr_recognition_crops"] += 1
            if not result or not result[0]:
                reads.append({"box":box,"text":"","confidence":0.0})
            else:
                text, confidence = result[0][0]
                reads.append({"box":box,"text":text,"confidence":float(confidence)})
        return reads

    @torch.no_grad()
    def __call__(self, image, record):
        self.counters["evidence_images"] += 1
        raw, scaled = self.detect_text(image,1), self.detect_text(image,SPEC["ocr_scale"])
        rows, scaled_rows = text_rows(raw), text_rows(scaled)
        reread, scaled_reread = self.reread(image,rows), self.reread(image,scaled_rows)
        expected = next(c["text"] for c in record["constraints"] if c["kind"] == "ocr")
        ocr_label = text_label([rows,scaled_rows,reread,scaled_reread],expected)
        constraints = [c for c in record["constraints"] if c["kind"] == "geneval"]
        p = SPEC["object_padding"]
        views = [image,ImageOps.expand(image,border=p,fill="white")]
        object_details = []
        for view in views:
            details = []
            for constraint in constraints:
                metadata = {**constraint["metadata"],"prompt":record["prompt"]}
                result = self.evaluate_image([view],[metadata],only_strict=True)[0]
                self.counters["geneval_clause_views"] += 1
                details.append({"id":constraint["id"],"strict":bool(result["strict_correct"]),"score":float(result["score"]),
                                "reason":result["reason"],"boxes":json.loads(result["details"])})
            object_details.append(details)
        clauses = {"text":ocr_label}
        for i,c in enumerate(constraints):
            clauses[c["id"]] = consensus([bool(d[i]["strict"]) for d in object_details])
        visibility = [visibility_support(details,constraints,image.size,offset) for details,offset in zip(object_details,(0,p))]
        return {"version":SPEC["version"],"clauses":clauses,"full_visibility":consensus(visibility),
                "evidence":{"ocr":{"raw":raw,"scaled":scaled,"rows":rows,"scaled_rows":scaled_rows,
                                    "reread":reread,"scaled_reread":scaled_reread},
                            "objects":object_details,"visibility_views":visibility}}
