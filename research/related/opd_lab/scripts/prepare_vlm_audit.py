"""Prepare frozen generation, smoke and remaining-image VLM audit attempts."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.data import load_records
from opdlab.io import ROOT,digest,read_json,sha256,stable_seed,write_json
from opdlab.runtime import prepare,verify_run


CAMPAIGN=ROOT/'experiments/local_expert_v6'


def prepare_generation(attempt):
    parent=ROOT/'runs/le3_probe_001';verify_run(parent)
    previous=read_json(ROOT/'experiments/local_expert_v5/measurement_plan.json')
    excluded=set(previous['excluded_ids']+[r['id'] for r in previous['calibration']])
    config=Experiment.load(parent/'config.json');groups=defaultdict(list)
    for r in load_records(config.dataset,'train'):
        if r['id'] not in excluded:groups[r['group']].append(r)
    if len(groups)!=4 or any(len(v)<4 for v in groups.values()):raise ValueError('Insufficient new prompt pool')
    chosen=[r for g in sorted(groups) for r in sorted(groups[g],key=lambda r:digest(['phase6_calibration',r['id']]))[:4]]
    chosen.sort(key=lambda r:digest(['phase6_review',r['id']]))
    calibration=[{'case':i,'id':r['id'],'noise_seed':stable_seed(271828,'phase6_calibration',r['id'])} for i,r in enumerate(chosen,1)]
    files=[parent/'frozen_behavior.pt',parent/'data.jsonl',ROOT/'experiments/local_expert_v5/measurement_plan.json',
           ROOT/'runs/le5_measurement_001/calibration.jsonl',ROOT/'runs/le5_measurement_001/result.json',
           ROOT/'runs/le5_calibration_review_001/observations.json',ROOT/'runs/le5_calibration_review_001/observations_frozen.json',
           ROOT/'runs/le4_target_review_001/key.json',ROOT/'runs/le4_target_review_001/observations.json',ROOT/'runs/le4_target_review_001/observations_frozen.json']
    plan={'calibration':calibration,'excluded_ids':sorted(excluded),'behavior_file':str(parent/'frozen_behavior.pt'),
          'inputs':{str(f):sha256(f) for f in files}}
    run=prepare(config,ROOT/f'runs/le6_generation_{attempt}',read_json(ROOT/'site.local.json'))
    write_json(run/'generation_plan.json',plan);shutil.copyfile(CAMPAIGN/'PROTOCOL.md',run/'PROTOCOL.md')
    write_json(run/'attestation.json',{'plan_sha256':sha256(run/'generation_plan.json'),'protocol_sha256':sha256(run/'PROTOCOL.md')})
    rows=[]
    for r in previous['development']:
        rows.append({**r,'uid':f"development:{r['case']:02d}:{r['action']}"})
    old=ROOT/'runs/le5_measurement_001'
    if sha256(old/'calibration.jsonl')!=read_json(old/'result.json')['artifacts_sha256']['calibration.jsonl']:raise ValueError('Old calibration changed')
    for r in map(json.loads,(old/'calibration.jsonl').read_text().splitlines()):
        rows.append({**r,'cohort':'exposed_calibration','uid':f"exposed_calibration:{r['case']:02d}"})
    for r in rows:
        if sha256(r['image'])!=r['image_sha256']:raise ValueError('Source image changed')
    master={'generation_run':str(run),'generation_plan_sha256':sha256(run/'generation_plan.json'),
            'existing_rows':rows,'new_calibration':calibration,'smoke_uids':[r['uid'] for r in rows[:2]],
            'protocol_sha256':sha256(CAMPAIGN/'PROTOCOL.md')}
    path=CAMPAIGN/f'campaign_{attempt}.json'
    if path.exists():raise FileExistsError('Campaign attempt exists')
    write_json(path,master)
    print({'generation_run':str(run),'master':str(path),'calibration_ids':[r['id'] for r in calibration]})


def prepare_vlm(mode,attempt,campaign_attempt):
    master_path=CAMPAIGN/f'campaign_{campaign_attempt}.json';master=read_json(master_path)
    if sha256(CAMPAIGN/'PROTOCOL.md')!=master['protocol_sha256']:raise ValueError('Protocol changed')
    rows=master['existing_rows'];inputs={str(master_path):sha256(master_path)}
    if mode=='smoke':rows=[r for r in rows if r['uid'] in master['smoke_uids']]
    else:
        smoke=ROOT/f'runs/le6_vlm_smoke_{campaign_attempt}';s=read_json(smoke/'result.json')
        if s['status']!='complete' or s['counts']['calls']!=4 or s['counts']['invalid_outputs']:
            raise ValueError('Smoke must load and return all four valid outputs')
        if sha256(smoke/'responses.jsonl')!=s['responses_sha256']:raise ValueError('Smoke changed')
        generation=Path(master['generation_run']);g=read_json(generation/'result.json')
        if g['status']!='complete' or sha256(generation/'calibration.jsonl')!=g['rows_sha256']:raise ValueError('Generation incomplete/changed')
        new=[json.loads(v) for v in (generation/'calibration.jsonl').read_text().splitlines()]
        if [(r['case'],r['id'],r['noise_seed']) for r in new]!=[(r['case'],r['id'],r['noise_seed']) for r in master['new_calibration']]:raise ValueError('Calibration differs')
        rows=[r for r in rows if r['uid'] not in master['smoke_uids']]+[{**r,'uid':f"new_calibration:{r['case']:02d}"} for r in new]
        inputs.update({str(f):sha256(f) for f in (smoke/'result.json',smoke/'responses.jsonl',generation/'result.json',generation/'calibration.jsonl')})
    if len(rows)!=(2 if mode=='smoke' else 74):raise ValueError('Incorrect VLM coverage')
    for r in rows:inputs[r['image']]=r['image_sha256']
    run=ROOT/f'runs/le6_vlm_{mode}_{attempt}';run.mkdir(exist_ok=False)
    framework=run/'framework/python/opdlab';framework.mkdir(parents=True)
    for name in ('__init__.py','io.py','vlm_audit.py'):shutil.copyfile(ROOT/'python/opdlab'/name,framework/name)
    shutil.copyfile(ROOT/'scripts/run_vlm_audit.py',run/'worker.py')
    shutil.copyfile(CAMPAIGN/'PROTOCOL.md',run/'PROTOCOL.md')
    plan={'mode':mode,'rows':rows,'assets':read_json(CAMPAIGN/'vlm_assets.json'),
          'environment':read_json(CAMPAIGN/'vlm_environment.json'),'input_files':inputs,
          'framework_files':{str(f.relative_to(run)):sha256(f) for f in [run/'worker.py',*framework.glob('*.py')]}}
    write_json(run/'vlm_plan.json',plan)
    write_json(run/'attestation.json',{'plan_sha256':sha256(run/'vlm_plan.json'),'protocol_sha256':sha256(run/'PROTOCOL.md')})
    site={'python':plan['environment']['executable'],'environment':{'HF_HUB_OFFLINE':'1','HF_HOME':'/midtier/minilab/scratch/chw4026/.cache/huggingface'}}
    write_json(run/'site.json',site)
    write_json(CAMPAIGN/f'{mode}_attempt_{attempt}.json',{'run':str(run),'campaign':str(master_path),'plan_sha256':sha256(run/'vlm_plan.json')})
    print({'run':str(run),'rows':len(rows),'python':site['python']})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--stage',choices=('generation','smoke','main'),required=True)
    p.add_argument('--attempt',default='001');p.add_argument('--campaign-attempt',default='001')
    a=p.parse_args()
    if a.stage=='generation':prepare_generation(a.attempt)
    else:prepare_vlm(a.stage,a.attempt,a.campaign_attempt)
