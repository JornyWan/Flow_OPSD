"""Show all frozen phase6 calibration images/prompts, hiding automatic scores."""
import argparse
import json
from pathlib import Path
import textwrap

from opdlab.io import read_json, sha256, write_json


def main(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    plan, result = read_json(run/'generation_plan.json'), read_json(run/'result.json')
    attestation = read_json(run/'attestation.json')
    if (result['status'] != 'complete' or result['images'] != 16 or
        sha256(run/'calibration.jsonl') != result['rows_sha256'] or
        sha256(run/'generation_plan.json') != attestation['plan_sha256'] or
        sha256(run/'PROTOCOL.md') != attestation['protocol_sha256']):
        raise ValueError('Incomplete or changed generation')
    rows = [json.loads(s) for s in (run/'calibration.jsonl').read_text().splitlines()]
    if [(r['case'], r['id'], r['noise_seed']) for r in rows] != [
        (r['case'], r['id'], r['noise_seed']) for r in plan['calibration']]:
        raise ValueError('Calibration selection changed')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out.mkdir(parents=True, exist_ok=False)
    template = []
    for r in rows:
        if sha256(r['image']) != r['image_sha256']:
            raise ValueError('Image changed')
        fig, ax = plt.subplots(figsize=(7, 7.5), dpi=110)
        ax.imshow(Image.open(r['image'])); ax.axis('off')
        fig.suptitle('\n'.join(textwrap.wrap(
            f"Case {r['case']:02d}: {r['record']['prompt']}", 87)), fontsize=10)
        fig.subplots_adjust(left=.01, right=.99, bottom=.01, top=.82)
        fig.savefig(out/f"case-{r['case']:02d}.png"); plt.close(fig)
        template.append({'case': r['case'], 'clauses': {c['id']: None for c in r['record']['constraints']},
                         'full_visibility': None, 'visible_text': None, 'notes': ''})
    write_json(out/'template.json', template)
    write_json(out/'key.json', rows)
    write_json(out/'provenance.json', {'run': str(run), 'images': len(rows),
               'plan_sha256': sha256(run/'generation_plan.json'),
               'calibration_sha256': sha256(run/'calibration.jsonl'), 'source_sha256': sha256(__file__)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); main(a.run, a.out)
