"""Compare exposed-image direction interfaces without inventing inverse labels."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from opdlab.audit_metrics import confusion
from opdlab.io import read_json, sha256, write_json
from opdlab.relation_audit import parse_relation, relation_request
from summarize_vlm_audit import lines


def main(run, out):
    plan, a, result = [read_json(run/n) for n in ('control_plan.json', 'attestation.json', 'result.json')]
    if (sha256(run/'control_plan.json') != a['plan_sha256'] or sha256(run/'PROTOCOL.md') != a['protocol_sha256'] or
        result['plan_sha256'] != a['plan_sha256'] or result['responses_sha256'] != sha256(run/'responses.jsonl') or
        result['status'] != 'complete' or result['environment'] != plan['environment']):
        raise ValueError('Incomplete/changed control')
    for p, h in plan['inputs'].items():
        if sha256(p) != h: raise ValueError('Input changed')
    for p, h in plan['framework_files'].items():
        if sha256(run/p) != h: raise ValueError('Frozen source changed')
    if sha256(Path(__file__).resolve().parents[1]/'python/opdlab/relation_audit.py') != plan['framework_files']['framework/python/opdlab/relation_audit.py']:
        raise ValueError('Control semantics changed')
    output = lines(run/'responses.jsonl')
    if len(output) != 20 or [r['uid'] for r in output] != [r['uid'] for r in plan['rows']]:
        raise ValueError('Coverage changed')
    counts = dict(images=0, calls=0, input_tokens=0, output_tokens=0, invalid_outputs=0, truncated_outputs=0)
    pairs = defaultdict(lambda: defaultdict(list)); consistency = defaultdict(Counter); details = []
    for r, s in zip(plan['rows'], output):
        if r['image_sha256'] != s['image_sha256'] or set(s['responses']) != {'given', 'inverse'}:
            raise ValueError('Image/condition changed')
        values = {}
        for condition, v in s['responses'].items():
            if v['request'] != relation_request(r['record'], inverse=condition == 'inverse'):
                raise ValueError('Request changed')
            parsed, error = None, None
            try: parsed = parse_relation(v['raw'])
            except (TypeError, ValueError) as exc: error = f'{type(exc).__name__}: {exc}'
            if v['truncated']: parsed, error = None, 'Generation reached maximum token limit'
            if parsed != v['parsed'] or error != v['parse_error']: raise ValueError('Parser replay differs')
            values[condition] = parsed['satisfied'] if parsed else None
            counts['calls'] += 1; counts['invalid_outputs'] += error is not None; counts['truncated_outputs'] += v['truncated']
            for k in ('input_tokens', 'output_tokens'): counts[k] += v[k]
        ref = r['reference']['clauses']['relation']; cohort = r['cohort']
        for name, pred in [('v1', r['predictions']['v1']['relation']),
                           ('qwen_original_metadata', r['predictions']['qwen_exact']['relation']),
                           ('qwen_explicit_given', values['given'])]:
            pairs[cohort][name].append((pred, ref))
        x, y = values['given'], values['inverse']
        kind = 'unknown' if x is None or y is None else 'both_true' if x and y else 'both_false' if not x and not y else 'complementary'
        consistency[cohort][kind] += 1
        details.append({'uid': r['uid'], 'cohort': cohort, 'reference_relation': ref,
                        'original': r['predictions']['qwen_exact']['relation'], 'explicit': values,
                        'consistency': kind, 'image_sha256': r['image_sha256'], 'responses': s['responses']})
        counts['images'] += 1
    if counts != result['counts']: raise ValueError('Counters differ')
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'per_image.json', details)
    summary = {'status': 'complete', 'meaning': 'Post-audit exposed-image interface diagnostic, not independent calibration; original results unchanged.',
        'metrics': {c: {n: confusion(v) for n, v in d.items()} for c, d in pairs.items()},
        'direction_consistency': {c: {k: v[k] for k in ('both_true', 'both_false', 'complementary', 'unknown')} for c, v in consistency.items()},
        'counts': counts, 'elapsed_seconds_excluding_asset_checks': result['elapsed_seconds'],
        'inputs_sha256': {str(run/n): sha256(run/n) for n in ('control_plan.json', 'attestation.json', 'result.json', 'responses.jsonl')},
        'source_sha256': sha256(__file__), 'per_image_sha256': sha256(out/'per_image.json')}
    write_json(out/'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(Path(a.run).resolve(), Path(a.out).resolve())
