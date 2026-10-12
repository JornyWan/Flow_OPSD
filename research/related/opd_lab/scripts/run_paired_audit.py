"""Frozen independent Qwen worker; target-free observations precede all pairs."""
import argparse
from pathlib import Path
import time

from opdlab.io import append_jsonl, read_json, sha256, write_json
from opdlab.paired_audit import OBSERVATION_PROMPT, SPEC, pair_request, parse_observation, parse_pair
from run_vlm_audit import environment


def main(run):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    run = Path(run).resolve()
    start = time.perf_counter()
    plan, a = read_json(run / "paired_plan.json"), read_json(run / "attestation.json")
    assert plan["spec"] == SPEC
    assert sha256(run / "paired_plan.json") == a["plan_sha256"]
    assert sha256(run / "PROTOCOL.md") == a["protocol_sha256"]
    for p, h in plan["input_files"].items():
        assert sha256(p) == h, p
    for p, h in plan["framework_files"].items():
        assert sha256(run / p) == h, p
    assert environment() == plan["environment"], "Independent VLM environment changed"
    mp = Path(plan["assets"]["snapshot"])
    for f, meta in plan["assets"]["files"].items():
        assert sha256(mp / f) == meta["sha256"], f
    for r in plan["images"]:
        assert sha256(r["image"]) == r["image_sha256"]
    for f in ("observations.jsonl", "pairs.jsonl", "result.json"):
        if (run / f).exists():
            raise FileExistsError("Use a fresh attempt")
    torch.manual_seed(SPEC["seed"])
    torch.cuda.manual_seed_all(SPEC["seed"])
    processor = AutoProcessor.from_pretrained(mp, local_files_only=True, min_pixels=SPEC["min_pixels"], max_pixels=SPEC["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mp, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    model.requires_grad_(False)
    model.generation_config.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, k, None)
    write_json(run / "effective_processor.json", {"class": type(processor.image_processor).__name__,
               "configuration": processor.image_processor.to_dict()})
    counts = {"calls": 0, "observation_calls": 0, "pair_calls": 0, "image_presentations": 0,
              "input_tokens": 0, "output_tokens": 0, "responses_with_errors": 0, "truncated_outputs": 0}
    images = {r["uid"]: Image.open(r["image"]).convert("RGB") for r in plan["images"]}

    def call(image_ids, request, stage, ids=()):
        content = []
        for i in range(len(image_ids)):
            if len(image_ids) == 2:
                content.append({"type": "text", "text": "Image " + ("A" if i == 0 else "B") + ":"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": request})
        text = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[images[i] for i in image_ids], return_tensors="pt").to("cuda:0")
        assert inputs.input_ids.shape[1] <= SPEC["max_input_tokens_per_call"]
        cap = SPEC[stage + "_max_new_tokens"]
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=cap, do_sample=False)
        suffix = output[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        truncated = suffix.shape[1] >= cap
        parsed, errors = parse_observation(raw) if stage == "observation" else parse_pair(raw, ids)
        if truncated:
            parsed, _ = parse_observation("null") if stage == "observation" else parse_pair("null", ids)
            errors.append("Generation reached token cap")
        counts["calls"] += 1
        counts[stage + "_calls"] += 1
        counts["image_presentations"] += len(image_ids)
        counts["input_tokens"] += inputs.input_ids.shape[1]
        counts["output_tokens"] += suffix.shape[1]
        counts["responses_with_errors"] += bool(errors)
        counts["truncated_outputs"] += truncated
        return {"request": request, "raw": raw, "parsed": parsed, "errors": errors, "truncated": truncated,
                "input_tokens": inputs.input_ids.shape[1], "output_tokens": suffix.shape[1],
                "image_grid_thw": inputs.image_grid_thw.cpu().tolist()}

    observations = {}
    for r in plan["images"]:
        response = call([r["uid"]], OBSERVATION_PROMPT, "observation")
        observations[r["uid"]] = response["parsed"]
        append_jsonl(run / "observations.jsonl", {"uid": r["uid"], "response": response})
        print({"stage": "observation", "complete": counts["observation_calls"]}, flush=True)
    write_json(run / "observations_frozen.json", {"sha256": sha256(run / "observations.jsonl"),
               "calls": counts["calls"], "before_any_pair_requests": True})
    pairs = {p["uid"]: p for p in plan["pairs"]}
    for task in plan["tasks"]:
        pair = pairs[task["pair_uid"]]
        request = pair_request(pair["clauses"], [observations[i] for i in task["images"]])
        response = call(task["images"], request, "pair", [c["id"] for c in pair["clauses"]])
        append_jsonl(run / "pairs.jsonl", {**task, "response": response})
        print({"stage": "pair", "complete": counts["pair_calls"]}, flush=True)
    assert counts["calls"] == 512 and counts["observation_calls"] == 192 and counts["pair_calls"] == 320
    write_json(run / "result.json", {"status": "complete", "counts": counts, "elapsed_seconds": time.perf_counter() - start,
               "plan_sha256": a["plan_sha256"], "observations_sha256": sha256(run / "observations.jsonl"),
               "pairs_sha256": sha256(run / "pairs.jsonl"), "environment": environment(),
               "gpu": torch.cuda.get_device_name(0), "peak_memory_bytes": torch.cuda.max_memory_allocated(),
               "generated_images": 0, "teacher_actions": 0, "optimizer_updates": 0})


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--run", required=True)
    main(p.parse_args().run)
