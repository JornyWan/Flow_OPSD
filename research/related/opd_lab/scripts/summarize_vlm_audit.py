"""Verify all phase6 outputs and compare to frozen provisional observations."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from opdlab.audit_metrics import confusion, conjunction
from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.verifiers import normalize_text
from opdlab.vlm_audit import OBSERVATION_PROMPT, parse_response, scene_request


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def load_run(run):
    run = Path(run).resolve()
    plan, a, result = [read_json(run/f) for f in ('vlm_plan.json', 'attestation.json', 'result.json')]
    if (sha256(run/'vlm_plan.json') != a['plan_sha256'] or
        sha256(run/'PROTOCOL.md') != a['protocol_sha256'] or
        result['plan_sha256'] != a['plan_sha256'] or result['status'] != 'complete' or
        sha256(run/'responses.jsonl') != result['responses_sha256'] or result['environment'] != plan['environment']):
        raise ValueError('Changed/incomplete VLM run')
    for p, h in plan['input_files'].items():
        if sha256(p) != h: raise ValueError(f'Input changed: {p}')
    for p, h in plan['framework_files'].items():
        if sha256(run/p) != h: raise ValueError('Frozen source changed')
    # Enforce that this report uses exactly the worker's frozen prompt/parser.
    if sha256(ROOT/'python/opdlab/vlm_audit.py') != plan['framework_files']['framework/python/opdlab/vlm_audit.py']:
        raise ValueError('Audit semantics changed since inference')
    output = lines(run/'responses.jsonl')
    if [r['uid'] for r in output] != [r['uid'] for r in plan['rows']]:
        raise ValueError('Missing, reordered or duplicated responses')
    counts = Counter(images=0, calls=0, input_tokens=0, output_tokens=0, invalid_outputs=0, truncated_outputs=0)
    merged = []
    for row, out in zip(plan['rows'], output):
        if (row['image_sha256'] != out['image_sha256'] or row['cohort'] != out['cohort'] or
            sha256(row['image']) != row['image_sha256'] or set(out['responses']) != {'observation', 'scene'}):
            raise ValueError('Image or call coverage changed')
        ids = [c['id'] for c in row['record']['constraints'] if c['kind'] == 'geneval']
        for stage, r in out['responses'].items():
            request = OBSERVATION_PROMPT if stage == 'observation' else scene_request(row['record'])
            if r['request'] != request: raise ValueError('Unexpected request')
            parsed, error = None, None
            try: parsed = parse_response(r['raw'], stage, ids)
            except (ValueError, TypeError) as exc: error = f'{type(exc).__name__}: {exc}'
            if r['truncated']:
                parsed, error = None, 'Generation reached maximum token limit'
            if parsed != r['parsed'] or error != r['parse_error']:
                raise ValueError('Parser replay differs')
            counts['calls'] += 1
            counts['invalid_outputs'] += error is not None
            counts['truncated_outputs'] += r['truncated']
            for k in ('input_tokens', 'output_tokens'): counts[k] += r[k]
        counts['images'] += 1
        merged.append({**row, 'responses': out['responses'], 'vlm_run': str(run)})
    if dict(counts) != result['counts']: raise ValueError('Response counters differ')
    return merged, result, plan


def references():
    refs, frozen, files = {}, {}, {}
    dev = ROOT/'runs/le4_target_review_001'
    for cohort, path in [('development', dev), ('exposed_calibration', ROOT/'runs/le5_calibration_review_001'),
                         ('new_calibration', ROOT/'runs/le6_calibration_review_001')]:
        f, obs, key = [read_json(path/n) for n in ('observations_frozen.json', 'observations.json', 'key.json')]
        if sha256(path/'observations.json') != f['observations_sha256']:
            raise ValueError('Frozen references changed')
        if cohort != 'development' and f['scores_revealed']:
            raise ValueError('Calibration references were not frozen blind')
        frozen[cohort] = f
        for n in ('observations_frozen.json', 'observations.json', 'key.json'):
            files[str(path/n)] = sha256(path/n)
        if cohort == 'development':
            if [r['case'] for r in obs] != list(range(1, 23)): raise ValueError('Development reference coverage')
            omap = {r['case']: r for r in obs}
            for k in key:
                for side, m in k['mapping'].items():
                    refs[f"development:{k['case']:02d}:{m['action']}"] = {
                        'observation': omap[k['case']]['observations'][side], 'image_sha256': m['row']['image_sha256']}
        else:
            if [r['case'] for r in obs] != list(range(1, 17)) or [r['case'] for r in key] != list(range(1, 17)):
                raise ValueError('Calibration reference coverage')
            for r, k in zip(obs, key):
                refs[f"{cohort}:{r['case']:02d}"] = {'observation': r, 'image_sha256': k['image_sha256']}
    rubric = ROOT/'experiments/local_expert_v6/REVIEW_RUBRIC.md'
    if sha256(rubric) != frozen['new_calibration']['rubric_sha256']: raise ValueError('Rubric changed')
    files[str(rubric)] = sha256(rubric)
    return refs, frozen, files


def summarize(smoke, main, out):
    start = time.perf_counter()
    rows, sr, sp = load_run(smoke)
    more, mr, mp = load_run(main); rows += more
    if sp['assets'] != mp['assets'] or sp['environment'] != mp['environment'] or sp['framework_files'] != mp['framework_files']:
        raise ValueError('Smoke and main implementation differ')
    if (len(rows) != 76 or len({r['uid'] for r in rows}) != 76 or
        dict(Counter(r['cohort'] for r in rows)) != {'development': 44, 'exposed_calibration': 16, 'new_calibration': 16}):
        raise ValueError('Campaign image coverage differs')
    refs, frozen, files = references()
    if set(refs) != {r['uid'] for r in rows}: raise ValueError('Unmatched image/reference')
    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    details = []
    for r in rows:
        ref = refs[r['uid']]
        if ref['image_sha256'] != r['image_sha256']: raise ValueError('Reference image changed')
        label = ref['observation']; constraints = r['record']['constraints']
        if set(label['clauses']) != {c['id'] for c in constraints}: raise ValueError('Reference clauses changed')
        if len(r['v1']['satisfied']) != len(constraints): raise ValueError('v1 clause coverage')
        obs, scene = [r['responses'][s]['parsed'] for s in ('observation', 'scene')]
        scene_map = {c['id']: c['satisfied'] for c in scene['clauses']} if scene else {}
        v1 = {c['id']: r['v1']['satisfied'][i] if r['v1']['valid'] else None for i, c in enumerate(constraints)}
        qwen, compact = {}, {}
        for c in constraints:
            if c['kind'] == 'ocr':
                if obs is None or obs['text_uncertain']:
                    qwen[c['id']] = compact[c['id']] = None
                else:
                    actual, expected = normalize_text(' '.join(obs['text_lines'])), normalize_text(c['text'])
                    qwen[c['id']] = actual == expected
                    compact[c['id']] = ''.join(actual.split()) == ''.join(expected.split())
            else:
                qwen[c['id']] = compact[c['id']] = scene_map.get(c['id'])
        predictions = {'v1': v1, 'qwen_exact': qwen, 'qwen_compact_sensitivity': compact}
        joint = {}
        for name, pred in predictions.items():
            for c in constraints: metrics[r['cohort']][name][c['id']].append((pred[c['id']], label['clauses'][c['id']]))
            joint[name] = conjunction(pred.values())
            metrics[r['cohort']][name]['joint_encoded'].append((joint[name], conjunction(label['clauses'].values())))
        visibility = scene['full_visibility'] if scene else None
        metrics[r['cohort']]['qwen_exact']['full_visibility'].append((visibility, label['full_visibility']))
        details.append({**r, 'reference': label, 'predictions': predictions, 'joint_encoded': joint, 'qwen_visibility': visibility,
                        'reference_joint_encoded': conjunction(label['clauses'].values())})
    summary_metrics = {cohort: {name: {clause: confusion(pairs) for clause, pairs in kinds.items()}
                               for name, kinds in methods.items()} for cohort, methods in metrics.items()}
    dev = {(r['case'], r['action']): r for r in details if r['cohort'] == 'development'}
    pairs = {}
    for name in ('v1', 'qwen_exact', 'qwen_compact_sensitivity', 'reference'):
        counts = Counter(gained=0, lost=0, tied=0, unknown=0)
        for case in range(1, 23):
            a, b = dev[case, 'noop'], dev[case, 'target']
            x, y = (a['reference_joint_encoded'], b['reference_joint_encoded']) if name == 'reference' else (
                a['joint_encoded'][name], b['joint_encoded'][name])
            counts['unknown' if x is None or y is None else 'gained' if y > x else 'lost' if y < x else 'tied'] += 1
        pairs[name] = dict(counts)
    counts = dict(Counter(sr['counts']) + Counter(mr['counts']))
    # Counter addition omits zero keys; retain explicit zero failures.
    counts = {k: sr['counts'][k]+mr['counts'][k] for k in sr['counts']}
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'per_image.json', details)
    summary = {'status': 'complete', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'images': dict(Counter(r['cohort'] for r in rows)), 'calls': counts,
        'metrics': summary_metrics, 'development_joint_encoded_pairs': pairs, 'observations_frozen': frozen,
        'meaning': 'Audit only, provisional implementing-assistant references; no human ground truth or policy efficacy claim. Joint encoded excludes style and full visibility.',
        'gpu_process_seconds_excluding_asset_checks': sr['elapsed_seconds']+mr['elapsed_seconds'],
        'cpu_analysis_seconds': time.perf_counter()-start,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
            [Path(__file__), ROOT/'python/opdlab/audit_metrics.py', ROOT/'python/opdlab/vlm_audit.py', ROOT/'python/opdlab/verifiers.py']},
        'inputs_sha256': {**files, **{str(Path(run)/n): sha256(Path(run)/n) for run in (smoke, main)
                         for n in ('vlm_plan.json', 'attestation.json', 'result.json', 'responses.jsonl')}},
        'per_image_sha256': sha256(out/'per_image.json')}
    write_json(out/'summary.json', summary)
    text = ['# Phase6: independent model audit', '', summary['meaning'], '',
            '| Cohort | Reader | Clause | TP | FP | TN | FN | Abstain +/- | Unknown reference | Coverage | Recall incl. abstain |',
            '|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
    for cohort, methods in summary_metrics.items():
        for name, kinds in methods.items():
            for clause, m in kinds.items():
                if name == 'qwen_compact_sensitivity' and clause not in ('text', 'joint_encoded'): continue
                def fmt(v): return 'NA' if v is None else f'{v:.3f}'
                text.append(f"| {cohort} | {name} | {clause} | {m['TP']} | {m['FP']} | {m['TN']} | {m['FN']} | {m['abstained_positive']}/{m['abstained_negative']} | {m['reference_unknown']} | {fmt(m['coverage_on_reference_known'])} | {fmt(m['recall_including_abstentions'])} |")
    (out/'report.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'images': summary['images'], 'calls': counts, 'out': str(out)}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--smoke', required=True); p.add_argument('--main', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); summarize(Path(a.smoke).resolve(), Path(a.main).resolve(), Path(a.out).resolve())
