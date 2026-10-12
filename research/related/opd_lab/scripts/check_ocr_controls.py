"""Known-answer OCR fixtures; exercise the actual joint verifier, not mocked scores."""
import argparse
from pathlib import Path
import shutil
import time

from PIL import Image, ImageDraw, ImageFont

from opdlab.backends import activate_upstream
from opdlab.config import Experiment
from opdlab.io import environment, sha256, tree_fingerprint, write_json
from opdlab.verifiers import JointVerifier


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--font", default="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "check_ocr_controls.py")
    activate_upstream(args.runtime)
    fixtures = [("exact", "MINT", "MINT", True), ("blank", "", "MINT", False),
                ("extra_text", "MINT EXTRA", "MINT", False), ("wrong_word", "PINE", "MINT", False),
                ("case_normalized", "mint", "MINT", True)]
    records = [{"id": name, "prompt": f'A sign saying "{expected}"',
                "constraints": [{"kind": "ocr", "id": "text", "text": expected}]}
               for name, text, expected, correct in fixtures]
    font = ImageFont.truetype(args.font, 64)
    verifier = JointVerifier(Experiment(backend="sd3"), None, "cpu", records)
    started, results = time.perf_counter(), []
    for record, (name, text, expected, correct) in zip(records, fixtures):
        image = Image.new("RGB", (640, 256), "white")
        ImageDraw.Draw(image).text((320, 128), text, font=font, fill="black", anchor="mm")
        path = out / f"{name}.png"
        image.save(path)
        verdict = verifier(image, record)
        results.append({"case": name, "rendered_text": text, "expected_text": expected,
                        "expected_success": correct, "pass": verdict.valid and verdict.satisfied == (correct,),
                        "image_sha256": sha256(path), **verdict.as_dict(0.5)})
    report = {"status": "passed" if all(r["pass"] for r in results) else "failed",
              "results": results, "environment": environment(), "font_sha256": sha256(args.font),
              "runtime": tree_fingerprint(Path(args.runtime) / "flow_grpo/ocr.py", full=True),
              "elapsed_seconds": time.perf_counter() - started,
              "interpretation": "Synthetic high-contrast known-answer OCR controls; not a validation of generated-image or GenEval accuracy."}
    write_json(out / "result.json", report)
    print(report["status"], [(r["case"], r["pass"]) for r in results], flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
