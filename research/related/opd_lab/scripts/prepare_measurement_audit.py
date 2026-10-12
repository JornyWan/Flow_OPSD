"""Freeze development evidence and a balanced, fresh-noise TRAIN calibration set."""
import argparse
from collections import defaultdict
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT, digest, read_json, sha256, stable_seed, write_json
from opdlab.measurement_audit import RULES
from opdlab.runtime import prepare, verify_run


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--attempt', default='001')
    args = p.parse_args()
    parent, action = ROOT/'runs/le3_probe_001', ROOT/'runs/le4_action_001'
    review = ROOT/'runs/le4_target_review_001'
    campaign = ROOT/'experiments/local_expert_v5'
    verify_run(parent); verify_run(action)
    excluded = set(read_json(parent/'probe_plan.json')['review_ids'] + read_json(action/'action_plan.json')['query_ids'])
    config = Experiment.load(parent/'config.json')
    families = defaultdict(list)
    for r in load_records(config.dataset, 'train'):
        if r['id'] not in excluded:
            families[r['group']].append(r)
    if len(families) != 4 or any(len(v) < 4 for v in families.values()):
        raise ValueError('Insufficient calibration pool')
    selected = [r for group in sorted(families) for r in sorted(families[group],key=lambda r:digest(['phase5_calibration',r['id']]))[:4]]
    selected.sort(key=lambda r:digest(['phase5_review',r['id']]))
    calibration = [{'case':i,'id':r['id'],'noise_seed':stable_seed(314159,'phase5_calibration',r['id'])} for i,r in enumerate(selected,1)]
    development, files = [], [parent/'frozen_behavior.pt', parent/'data.jsonl', parent/'config.json', parent/'probe_plan.json',
        action/'actions.jsonl', action/'result.json', action/'action_plan.json', review/'key.json',
        review/'observations.json', review/'observations_frozen.json']
    for k in read_json(review/'key.json'):
        for m in k['mapping'].values():
            row = m['row']; image = parent/row['image']
            if sha256(image) != row['image_sha256']:
                raise ValueError('Development image changed')
            files.append(image)
            development.append({'case':k['case'],'action':m['action'],'record':k['record'], 'image':str(image),
                                'image_sha256':row['image_sha256'], 'v1':row,'cohort':'development'})
    audit_config = ROOT/'experiments/local_expert_v2/ocr_audit_config.json'
    files.append(audit_config)
    plan = {'version':1,'rules':list(RULES),'calibration':calibration,'development':development,
            'excluded_ids':sorted(excluded),'parent_run':str(parent),'action_run':str(action),
            'review_run':str(review),'behavior_file':str(parent/'frozen_behavior.pt'),
            'audit_ocr_config':str(audit_config),'input_files':{str(f):sha256(f) for f in files}}
    path=campaign/'measurement_plan.json'
    if path.exists() and read_json(path)!=plan:
        raise ValueError('Plan changed; do not overwrite')
    if not path.exists():
        write_json(path,plan)
    run=prepare(config,ROOT/f'runs/le5_measurement_{args.attempt}',read_json(ROOT/'site.local.json'))
    for src,name in ((path,'measurement_plan.json'),(campaign/'PROTOCOL.md','PROTOCOL.md')):
        shutil.copyfile(src,run/name)
    a={'plan_sha256':sha256(run/'measurement_plan.json'),'protocol_sha256':sha256(run/'PROTOCOL.md')}
    write_json(run/'measurement_attestation.json',a)
    write_json(campaign/f'attempt_{args.attempt}.json',{'run':str(run),**a})
    print({'run':str(run),'calibration_ids':[r['id'] for r in calibration],**a})
