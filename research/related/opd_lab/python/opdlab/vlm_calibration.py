"""Generate the predeclared fresh images for a separate VLM measurement check."""
import argparse
from pathlib import Path
import time
import torch

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import seed_all
from .fitting_probe import tensor_digest
from .io import append_jsonl,assert_environment,read_json,sha256,write_json
from .runtime import verify_run
from .verifiers import build_verifier


@torch.no_grad()
def run_generation(run):
    run=Path(run).resolve();verify_run(run)
    a=read_json(run/'attestation.json');plan=read_json(run/'generation_plan.json')
    if sha256(run/'generation_plan.json')!=a['plan_sha256'] or sha256(run/'PROTOCOL.md')!=a['protocol_sha256']:raise ValueError('Plan changed')
    for p,h in plan['inputs'].items():
        if sha256(p)!=h:raise ValueError('Input changed')
    if (run/'calibration.jsonl').exists():raise FileExistsError('Use a fresh attempt')
    c=Experiment.load(run/'config.json');site=read_json(run/'site.json')
    assert_environment(read_json(run/'manifest.json')['environment'])
    records={r['id']:r for r in load_records(c.dataset,'train')}
    seed_all(c.seed);start=time.perf_counter()
    backend=build_backend(c,site,run/'upstream',torch.device('cuda:0'),training=False)
    frozen=torch.load(plan['behavior_file'],map_location='cpu',weights_only=True);backend.load_student(frozen)
    verifier=build_verifier(c,site,torch.device('cuda:0'),list(records.values()))
    (run/'images').mkdir(exist_ok=False)
    for spec in plan['calibration']:
        record=records[spec['id']];image=backend.generate(record,spec['noise_seed'],c.steps,'flow',adapter='behavior')
        v=verifier(image,record)
        if not v.valid:raise RuntimeError('Invalid verifier output')
        path=run/'images'/f"{spec['case']:02d}.png";image.save(path)
        append_jsonl(run/'calibration.jsonl',{**spec,'cohort':'new_calibration','record':record,'image':str(path),
                     'image_sha256':sha256(path),'v1':v.as_dict(c.utility_joint_weight)})
        print({'event':'calibration_image','case':spec['case']},flush=True)
    if tensor_digest(backend.behavior_state())!=tensor_digest(frozen):raise ValueError('Behavior changed')
    elapsed=time.perf_counter()-start
    write_json(run/'result.json',{'status':'complete','images':16,'optimizer_updates':0,'behavior_sha256':tensor_digest(frozen),
               'elapsed_seconds':elapsed,'allocated_gpu_hours':elapsed/3600,'counts':dict(backend.counters+verifier.counters),
               'plan_sha256':a['plan_sha256'],'rows_sha256':sha256(run/'calibration.jsonl')})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True)
    run_generation(p.parse_args().run)
