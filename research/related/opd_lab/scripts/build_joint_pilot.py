"""Deterministic constructed pilot, frozen before looking at model outputs."""
import json
from pathlib import Path

from opdlab.data import load_records
from opdlab.io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/joint_pilot_v1"
WORDS = {
    "train": "MINT PINE DAWN LUNA GLOW SODA LOFT WAVE FRESH NORTH MAPLE PEACH BLOOM CLOUD CLEAR SUNNY".split(),
    "validation": "EMBER CEDAR CORAL HAVEN RIVER STONE MANGO OLIVE".split(),
    "test": "BRIGHT MEADOW OCEAN SILVER FOREST HONEY AMBER VIOLET RAVEN PEARL SPRUCE IVORY STAR RAIN SNOW LEAF".split(),
}
OBJECTS = [("cup", "cups"), ("bottle", "bottles"), ("apple", "apples"), ("chair", "chairs"),
           ("bowl", "bowls"), ("vase", "vases"), ("book", "books"), ("banana", "bananas")]
PAIRS = [("cup", "bottle"), ("apple", "bowl"), ("chair", "potted plant"), ("bicycle", "car")]
COLORS = ["red", "blue", "green", "yellow"]
NUMBERS = {1: "one", 2: "two", 3: "three"}
FAMILIES = ["count", "color", "relation", "count_color"]


def clause(name, tag, objects):
    return {"id": name, "kind": "geneval", "metadata": {"tag": tag, "include": objects}}


def build():
    if OUT.exists():
        raise FileExistsError("Pilot v1 is immutable; create a new version instead of replacing it")
    OUT.mkdir(parents=True)
    records = []
    for split, words in WORDS.items():
        for family in FAMILIES:
            for i, word in enumerate(words):
                noun, plural = OBJECTS[i % len(OBJECTS)]
                count = 1 + (i // 2) % 3
                color = COLORS[(i // 2 + i % 2) % len(COLORS)]
                constraints = [{"id": "text", "kind": "ocr", "text": word}]
                if family == "count":
                    scene = f"exactly {NUMBERS[count]} {noun if count == 1 else plural}, all fully visible and separated"
                    constraints.append(clause("count", "counting", [{"class": noun, "count": count}]))
                elif family == "color":
                    scene = f"exactly one {color} {noun}, fully visible"
                    constraints.append(clause("color", "colors", [{"class": noun, "count": 1, "color": color}]))
                elif family == "relation":
                    subject, reference = PAIRS[i % len(PAIRS)]
                    relation = "left of" if (i // len(PAIRS)) % 2 == 0 else "right of"
                    scene = f"exactly one {subject} clearly to the {relation} exactly one {reference}, both fully visible"
                    # 'to the left of' and 'to the right of' are rendered by the same template.
                    constraints.append(clause("relation", "position", [
                        {"class": reference, "count": 1},
                        {"class": subject, "count": 1, "position": [relation, 0]}]))
                else:
                    scene = f"exactly {NUMBERS[count]} {color} {noun if count == 1 else plural}, all fully visible and separated"
                    constraints.extend([clause("count", "counting", [{"class": noun, "count": count}]),
                        clause("color", "colors", [{"class": noun, "count": count, "color": color}])])
                if split == "train":
                    prompt = f'A sharp, realistic studio photograph of {scene}. Above the objects, a large white sign displays only the word "{word}" in bold black letters.'
                elif split == "validation":
                    prompt = f'A well-lit catalog photograph shows {scene} against a plain background. A prominent placard above them reads "{word}" in clear black capital letters on white.'
                else:
                    prompt = f'A realistic shop-window display contains {scene}. A large sign at the top of the display reads "{word}" in legible black capital letters on a white background.'
                prompt += " No other letters or words are visible."
                records.append({"id": f"jp1-{split}-{family}-{i:02d}", "prompt": prompt, "split": split,
                                "group": f"jp1-{split}-{family}", "constraints": constraints})
    path = OUT / "records.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    counts = {split: len(load_records(path, split)) for split in WORDS}
    classes = set((ROOT / "upstream/diffusionopd/flow_grpo/assets/object_names.txt").read_text().splitlines())
    for row in records:
        for c in row["constraints"]:
            if c["kind"] == "geneval":
                assert all(o["class"] in classes for o in c["metadata"]["include"])
    assert not (set(WORDS["train"]) & set(WORDS["validation"]) | set(WORDS["train"]) & set(WORDS["test"])
                | set(WORDS["validation"]) & set(WORDS["test"]))
    write_json(OUT / "provenance.json", {"version": 1, "kind": "constructed_development_pilot_not_benchmark",
               "generator_sha256": sha256(__file__), "records_sha256": sha256(path), "counts": counts,
               "families": FAMILIES, "words": WORDS,
               "split_policy": "distinct text vocabulary and prompt template per split; object classes intentionally shared",
               "frozen_before_gpu_results": True})
    print(json.dumps({"file": str(path), "counts": counts}))


if __name__ == "__main__":
    build()
