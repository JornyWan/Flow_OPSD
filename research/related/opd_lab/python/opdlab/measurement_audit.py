"""Audit-only measurement rules and detector traces; never imported by training."""
import argparse
import inspect
import json
from pathlib import Path
import re
import time

import numpy as np
import torch
from PIL import Image

from .backends import build_backend
from .config import Experiment
from .data import load_records
from .engine import seed_all
from .fitting_probe import tensor_digest
from .io import append_jsonl, assert_environment, read_json, sha256, write_json
from .runtime import verify_run
from .verifiers import build_verifier, normalize_text, text_progress

RULES = ('v1', 'paddle_compact', 'tesseract_compact', 'both_compact')


def ocr_checks(primary, secondary, expected):
    expected = normalize_text(expected)
    if not re.fullmatch('[a-z]+', expected):
        raise ValueError('Compact audit is restricted to one alphabetic target word')
    p, t = [text_progress(re.sub(r'\s+', '', normalize_text(s)), expected) for s in (primary, secondary)]
    return dict(zip(RULES, (text_progress(primary, expected), p, t, (p[0] and t[0], min(p[1], t[1])))))


def detector_components(scorer):
    evaluate_image = inspect.getclosurevars(inspect.unwrap(scorer)).nonlocals['evaluate_image']
    closure = inspect.getclosurevars(evaluate_image)
    c = closure.nonlocals
    constants = {k: c[k] for k in ('THRESHOLD', 'COUNTING_THRESHOLD', 'MAX_OBJECTS', 'NMS_THRESHOLD')}
    if constants != dict(THRESHOLD=.3, COUNTING_THRESHOLD=.9, MAX_OBJECTS=16, NMS_THRESHOLD=1.):
        raise ValueError('Unsupported detector filtering contract')
    return c['object_detector'], c['classnames'], closure.globals['inference_detector'], constants


def trace_boxes(boxes, classnames, record, size, constants):
    names = sorted({o['class'] for c in record['constraints'] if c['kind'] == 'geneval' for o in c['metadata']['include']})
    raw = {n: np.asarray(boxes[classnames.index(n)], dtype=np.float64).reshape(-1, 5) for n in names}
    if any(not np.isfinite(b).all() for b in raw.values()):
        raise ValueError('Nonfinite detector boxes')
    clauses = []
    for c in record['constraints']:
        if c['kind'] != 'geneval':
            continue
        threshold = constants['COUNTING_THRESHOLD'] if c['metadata']['tag'] == 'counting' else constants['THRESHOLD']
        retained = {}
        for o in c['metadata']['include']:
            n, b = o['class'], raw[o['class']]
            order = np.argsort(b[:, 4])[::-1]
            kept = b[order[b[order, 4] > threshold][:constants['MAX_OBJECTS']]]
            retained[n] = {'expected_count': o['count'], 'count': len(kept), 'boxes': kept.tolist(),
                           'touches_frame': any(x[0] <= 1 or x[1] <= 1 or x[2] >= size[0]-1 or x[3] >= size[1]-1 for x in kept)}
        clauses.append({'id': c['id'], 'tag': c['metadata']['tag'], 'threshold': threshold, 'objects': retained,
                        'count_component_satisfied': all(v['count'] == v['expected_count'] for v in retained.values())})
    return {'raw_requested_class_boxes': {n:b.tolist() for n,b in raw.items()}, 'clauses': clauses,
            'frame_flag': any(v['touches_frame'] for c in clauses for v in c['objects'].values())}


def verify_inputs(run):
    run = Path(run)
    plan = read_json(run / 'measurement_plan.json')
    a = read_json(run / 'measurement_attestation.json')
    if sha256(run / 'measurement_plan.json') != a['plan_sha256'] or sha256(run / 'PROTOCOL.md') != a['protocol_sha256']:
        raise ValueError('Measurement plan changed')
    for path, expected in plan['input_files'].items():
        if sha256(path) != expected:
            raise ValueError(f'Changed input: {path}')
    if plan['rules'] != list(RULES) or len(plan['calibration']) != 16 or len(plan['development']) != 44:
        raise ValueError('Unexpected cohort/rule coverage')
    return plan


@torch.no_grad()
def run_gpu(run):
    run = Path(run).resolve()
    verify_run(run)
    plan = verify_inputs(run)
    if (run / 'result.json').exists() or (run / 'calibration.jsonl').exists():
        raise FileExistsError('Use a fresh attempt')
    config, site = Experiment.load(run / 'config.json'), read_json(run / 'site.json')
    assert_environment(read_json(run / 'manifest.json')['environment'])
    records = {r['id']: r for r in load_records(config.dataset, 'train')}
    started = time.perf_counter()
    seed_all(config.seed)
    backend = build_backend(config, site, run / 'upstream', torch.device('cuda:0'), training=False)
    frozen = torch.load(plan['behavior_file'], map_location='cpu', weights_only=True)
    backend.load_student(frozen)
    behavior_hash = tensor_digest(frozen)
    verifier = build_verifier(config, site, torch.device('cuda:0'), list(records.values()))
    (run / 'images').mkdir(exist_ok=False)
    images = list(plan['development'])
    for spec in plan['calibration']:
        record = records[spec['id']]
        image = backend.generate(record, spec['noise_seed'], config.steps, 'flow', adapter='behavior')
        verdict = verifier(image, record)
        if not verdict.valid:
            raise RuntimeError('Invalid calibration verdict')
        path = run / 'images' / f"{spec['case']:02d}.png"
        image.save(path)
        row = {**spec, 'image':str(path), 'image_sha256':sha256(path), 'record':record,
               'v1':verdict.as_dict(config.utility_joint_weight), 'cohort':'calibration'}
        append_jsonl(run / 'calibration.jsonl', row)
        images.append(row)
        print({'event':'calibration_image_complete','case':spec['case']}, flush=True)
    detector, classes, inference, constants = detector_components(verifier.geneval)
    for row in images:
        if sha256(row['image']) != row['image_sha256']:
            raise ValueError('Trace image changed')
        image = Image.open(row['image']).convert('RGB')
        output = inference(detector, [np.asarray(image)])[0]
        boxes = output[0] if isinstance(output, tuple) else output
        trace = trace_boxes(boxes, classes, row['record'], image.size, constants)
        cids = [c['id'] for c in row['record']['constraints']]
        count_checks = {c['id']: c['count_component_satisfied'] == row['v1']['satisfied'][cids.index(c['id'])]
                        for c in trace['clauses'] if c['tag'] == 'counting'}
        append_jsonl(run / 'scene_traces.jsonl', {'cohort':row['cohort'], 'id':row['record']['id'],
                     'image_sha256':row['image_sha256'], 'counting_reproduction':count_checks, **trace})
    if tensor_digest(backend.behavior_state()) != behavior_hash:
        raise ValueError('Behavior changed')
    elapsed = time.perf_counter()-started
    write_json(run / 'result.json', {'status':'complete','new_images':16,'scene_traces':len(images),
               'optimizer_updates':0,'behavior_sha256':behavior_hash,'detector_constants':constants,
               'counts':dict(backend.counters+verifier.counters), 'additional_detector_image_calls':len(images),
               'elapsed_seconds':elapsed, 'allocated_gpu_hours':elapsed/3600,
               'plan_sha256':sha256(run/'measurement_plan.json'),
               'artifacts_sha256':{n:sha256(run/n) for n in ('calibration.jsonl','scene_traces.jsonl')}})


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run', required=True)
    run_gpu(p.parse_args().run)
