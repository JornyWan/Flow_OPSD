"""Fresh explicit-direction requests; original phase6 scores remain unchanged."""
import argparse
from pathlib import Path
import time

from opdlab.io import append_jsonl, read_json, sha256, write_json
from opdlab.relation_audit import parse_relation, relation_request
from run_vlm_audit import environment


def main(run):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    run = Path(run).resolve(); plan = read_json(run/'control_plan.json')
    a = read_json(run/'attestation.json')
    if sha256(run/'control_plan.json') != a['plan_sha256'] or sha256(run/'PROTOCOL.md') != a['protocol_sha256']:
        raise ValueError('Plan changed')
    for p, h in plan['inputs'].items():
        if sha256(p) != h: raise ValueError('Source image/report changed')
    for p, h in plan['framework_files'].items():
        if sha256(run/p) != h: raise ValueError('Frozen code changed')
    if environment() != plan['environment']: raise ValueError('Environment changed')
    mp = Path(plan['assets']['snapshot'])
    for f, meta in plan['assets']['files'].items():
        if sha256(mp/f) != meta['sha256']: raise ValueError('Model assets changed')
    if (run/'responses.jsonl').exists() or (run/'result.json').exists(): raise FileExistsError('Use fresh attempt')
    start = time.perf_counter(); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    processor = AutoProcessor.from_pretrained(mp, local_files_only=True, min_pixels=3136, max_pixels=262144)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(mp, local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False); model.generation_config.do_sample = False
    for key in ('temperature', 'top_p', 'top_k'): setattr(model.generation_config, key, None)
    counts = dict(images=0, calls=0, input_tokens=0, output_tokens=0, invalid_outputs=0, truncated_outputs=0)
    for r in plan['rows']:
        responses = {}; image = Image.open(r['image']).convert('RGB')
        for condition in r['conditions']:
            request = relation_request(r['record'], inverse=condition == 'inverse')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': request}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors='pt').to('cuda:0')
            with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
            suffix = output[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed, error = None, None
            try: parsed = parse_relation(raw)
            except (ValueError, TypeError) as exc: error = f'{type(exc).__name__}: {exc}'
            truncated = suffix.shape[1] >= 192
            if truncated: parsed, error = None, 'Generation reached maximum token limit'
            responses[condition] = dict(request=request, raw=raw, parsed=parsed, parse_error=error,
                truncated=truncated, input_tokens=inputs.input_ids.shape[1], output_tokens=suffix.shape[1],
                image_grid_thw=inputs.image_grid_thw.cpu().tolist())
            counts['calls'] += 1; counts['invalid_outputs'] += error is not None; counts['truncated_outputs'] += truncated
            counts['input_tokens'] += inputs.input_ids.shape[1]; counts['output_tokens'] += suffix.shape[1]
        append_jsonl(run/'responses.jsonl', {'uid': r['uid'], 'image_sha256': r['image_sha256'], 'responses': responses})
        counts['images'] += 1
        print({'event': 'relation_control_complete', 'uid': r['uid'], 'images': counts['images']}, flush=True)
    write_json(run/'result.json', {'status': 'complete', 'counts': counts, 'elapsed_seconds': time.perf_counter()-start,
        'environment': environment(), 'plan_sha256': a['plan_sha256'], 'responses_sha256': sha256(run/'responses.jsonl'),
        'gpu': torch.cuda.get_device_name(0), 'peak_memory_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run', required=True); main(p.parse_args().run)
