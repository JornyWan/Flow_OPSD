"""Prepare score/arm-hidden review of EVERY phase7 update128 endpoint."""
import argparse
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont

from opdlab.io import digest, read_json, sha256, write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run, out = Path(args.run).resolve(), Path(args.out).resolve()
    if read_json(run / "result.json")["status"] != "complete":
        raise ValueError("Review requires completed fitting")
    rows = [json.loads(x) for x in (run / "endpoints.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["optimizer_step"] == 128]
    if len(rows) != 20 or len({(r["arm"], r["id"], r["mode"]) for r in rows}) != 20:
        raise ValueError("Expected the complete 5 cases × 2 fitting arms × 2 inference modes")
    records = {r["id"]: r for r in [json.loads(x) for x in (run / "data.jsonl").read_text().splitlines()]}
    rows.sort(key=lambda r: digest(["phase7_blind", r["id"], r["arm"], r["mode"]]))
    out.mkdir(parents=True, exist_ok=False)
    public, key = [], []
    for number, row in enumerate(rows, 1):
        image = run / row["image"]
        if sha256(image) != row["image_sha256"]:
            raise ValueError("Endpoint image changed")
        destination = out / f"image-{number:02d}.png"
        shutil.copyfile(image, destination)
        public.append({"case": number, "prompt": records[row["id"]]["prompt"],
                       "image": destination.name, "image_sha256": sha256(destination)})
        key.append({"case": number, "row": row, "record": records[row["id"]]})
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    for start in range(0, len(public), 4):
        sheet = Image.new("RGB", (1024, 1216), "white")
        draw = ImageDraw.Draw(sheet)
        for position, case in enumerate(public[start:start + 4]):
            x, y = (position % 2) * 512, (position // 2) * 608
            text = f"Image {case['case']:02d}\n" + "\n".join(textwrap.wrap(case["prompt"], 68))
            draw.text((x + 6, y + 3), text, fill="black", font=font, spacing=2)
            sheet.paste(Image.open(out / case["image"]), (x, y + 96))
        sheet.save(out / f"sheet-{start // 4 + 1:02d}.png")
    write_json(out / "items.json", public)
    write_json(out / "key.json", key)
    write_json(out / "inputs.json", {"run": str(run), "result_sha256": sha256(run / "result.json"),
               "endpoints_sha256": sha256(run / "endpoints.jsonl"), "key_sha256": sha256(out / "key.json"),
               "scope": "all final endpoints, arm-blinded implementing-assistant review; not independent"})
    print(f"Prepared {len(public)} images in five sheets at {out}; freeze observations before opening key")


if __name__ == "__main__":
    main()
