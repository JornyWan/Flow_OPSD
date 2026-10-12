"""Compare two readers with frozen provisional, target-hidden assistant transcriptions."""
import argparse
from collections import Counter
from pathlib import Path

from opdlab.audit_ocr import AuditOCR
from opdlab.io import read_json, sha256, write_json
from opdlab.verifiers import text_progress


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--audit", required=True)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    audit = Path(args.audit).resolve()
    if (audit / "result.json").exists():
        raise FileExistsError("Do not overwrite audit results")
    labels = {r["case"]: r for r in read_json(audit / "assistant_transcriptions.json")["cases"]}
    reader = AuditOCR(args.config)
    results, counts = [], {name: Counter() for name in ("paddle", "tesseract")}
    for row in read_json(audit / "key.json"):
        label = labels[row["case"]]
        expected = next(c["text"] for c in row["record"]["constraints"] if c["kind"] == "ocr")
        secondary = reader(row["image"], row["image_sha256"])
        primary = row["candidate"]["details"]["ocr"]["recognized_text"]
        judged = label["transcript"] is not None and label["additional_text"] is not None
        correct = (text_progress(label["transcript"], expected)[0] and not label["additional_text"]) if judged else None
        predictions = {"paddle": text_progress(primary, expected)[0],
                       "tesseract": text_progress(secondary["text"], expected)[0]}
        for name, value in predictions.items():
            if judged:
                counts[name]["assessed"] += 1
                counts[name]["agreement"] += int(value == correct)
                counts[name]["false_positive"] += int(value and not correct)
                counts[name]["false_negative"] += int(not value and correct)
        results.append({"case": row["case"], "id": row["record"]["id"], "image_sha256": row["image_sha256"],
                        "expected": expected, "assistant": label, "assistant_exact": correct,
                        "paddle_text": primary, "tesseract": secondary, "exact": predictions})
    value = {"reader_version": reader.version, "config_sha256": sha256(args.config),
             "labels_sha256": sha256(audit / "assistant_transcriptions.json"), "cases": len(results),
             "summary": {k: dict(v) for k, v in counts.items()}, "rows": results,
             "interpretation": "Small implementing-assistant audit, uncertain readings excluded; neither reader is ground truth. No independent-human or population-accuracy claim."}
    write_json(audit / "result.json", value)
    print({k: value[k] for k in ("cases", "summary", "interpretation")})


if __name__ == "__main__":
    main()
