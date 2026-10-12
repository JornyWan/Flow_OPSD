"""Export all observed phase6 false positives and schema failures, post hoc."""
import argparse
import json
from pathlib import Path

from opdlab.io import read_json, sha256, write_json


def main(run, out):
    summary = read_json(run/'summary.json')
    if sha256(run/'per_image.json') != summary['per_image_sha256']:
        raise ValueError('Report changed')
    rows = read_json(run/'per_image.json'); errors, fps = [], []
    for r in rows:
        for stage, s in r['responses'].items():
            if s['parse_error']:
                raw = s['raw'].strip()
                raw = raw[8:-4] if raw.startswith('```json\n') and raw.endswith('\n```') else raw
                decoded = json.loads(raw)
                errors.append({'uid': r['uid'], 'stage': stage, 'error': s['parse_error'], 'raw': s['raw'],
                    'visibility_evidence_is_null': decoded.get('visibility_evidence') is None,
                    'full_visibility_is_null': decoded.get('full_visibility') is None})
        for c, v in r['reference']['clauses'].items():
            if v is False and r['predictions']['qwen_exact'][c] is True:
                fps.append({'uid': r['uid'], 'clause': c, 'reference': r['reference'],
                            'qwen_responses': r['responses'], 'image': r['image'], 'image_sha256': r['image_sha256']})
    value = {'meaning': 'Post-hoc failure description only; original strict parsing and scores remain unchanged. No salvage of clauses from invalid responses.',
             'invalid_outputs': errors, 'false_positives': fps,
             'text_uncertain_true': sum(r['responses']['observation']['parsed']['text_uncertain'] for r in rows),
             'image_grids': sorted({str(s['image_grid_thw']) for r in rows for s in r['responses'].values()})}
    if out.exists(): raise FileExistsError('Use fresh output')
    write_json(out, value)
    write_json(out.with_suffix('.provenance.json'), {'summary_sha256': sha256(run/'summary.json'),
        'per_image_sha256': sha256(run/'per_image.json'), 'source_sha256': sha256(__file__), 'output_sha256': sha256(out)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(Path(a.run), Path(a.out))
