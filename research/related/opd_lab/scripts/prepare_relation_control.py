"""Freeze an exposed-image direction-interface diagnostic, after phase6 audit."""
import argparse
from pathlib import Path
import shutil

from opdlab.io import ROOT, digest, read_json, sha256, write_json


def main(attempt):
    campaign = ROOT/'experiments/local_expert_v6'
    source = ROOT/'runs/le6_report_001'
    if sha256(source/'per_image.json') != read_json(source/'summary.json')['per_image_sha256']:
        raise ValueError('Phase6 report changed')
    rows = [r for r in read_json(source/'per_image.json') if 'relation' in r['reference']['clauses']]
    if len(rows) != 20: raise ValueError('Expected all20 relation images')
    for r in rows: r['conditions'] = sorted(['given', 'inverse'], key=lambda c: digest([r['uid'], c]))
    run = ROOT/f'runs/le6_relation_control_{attempt}'; run.mkdir(exist_ok=False)
    framework = run/'framework/python/opdlab'; framework.mkdir(parents=True)
    for f in ('__init__.py', 'io.py', 'vlm_audit.py', 'relation_audit.py'):
        shutil.copyfile(ROOT/'python/opdlab'/f, framework/f)
    for f in ('run_relation_control.py', 'run_vlm_audit.py'):
        shutil.copyfile(ROOT/'scripts'/f, run/f)
    shutil.copyfile(campaign/'RELATION_CONTROL_PROTOCOL.md', run/'PROTOCOL.md')
    plan = {'rows': rows, 'assets': read_json(campaign/'vlm_assets.json'),
            'environment': read_json(campaign/'vlm_environment.json'),
            'inputs': {str(p): sha256(p) for p in [source/'summary.json', source/'per_image.json', *[Path(r['image']) for r in rows]]},
            'framework_files': {str(f.relative_to(run)): sha256(f) for f in [*run.glob('*.py'), *framework.glob('*.py')]}}
    write_json(run/'control_plan.json', plan)
    write_json(run/'attestation.json', {'plan_sha256': sha256(run/'control_plan.json'), 'protocol_sha256': sha256(run/'PROTOCOL.md')})
    write_json(run/'site.json', {'python': plan['environment']['executable'], 'environment': {'HF_HUB_OFFLINE': '1', 'HF_HOME': '/midtier/minilab/scratch/chw4026/.cache/huggingface'}})
    write_json(campaign/f'relation_control_attempt_{attempt}.json', {'run': str(run), 'plan_sha256': sha256(run/'control_plan.json')})
    print({'run': str(run), 'images': len(rows), 'calls': 2*len(rows)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--attempt', default='001'); main(p.parse_args().attempt)
