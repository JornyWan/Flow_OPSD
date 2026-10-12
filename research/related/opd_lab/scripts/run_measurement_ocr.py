"""Read fixed image cohorts with the independent OCR; save every rule separately."""
import argparse
import json
from pathlib import Path
import time

from opdlab.audit_ocr import AuditOCR
from opdlab.io import append_jsonl, read_json, sha256, write_json
from opdlab.measurement_audit import ocr_checks, verify_inputs


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',required=True); p.add_argument('--out',required=True)
    p.add_argument('--cohort',choices=('matrix','calibration'),required=True)
    args=p.parse_args(); run,out=Path(args.run).resolve(),Path(args.out).resolve()
    plan=verify_inputs(run)
    if args.cohort=='matrix':
        action=Path(plan['action_run'])
        records={r['id']:r for r in map(json.loads,(Path(plan['parent_run'])/'data.jsonl').read_text().splitlines())}
        rows=[{'id':r['id'],'action':r['action'],'image':str(action/r['image']),'image_sha256':r['image_sha256'],
               'record':records[r['id']],'v1':r} for r in map(json.loads,(action/'actions.jsonl').read_text().splitlines())]
        if len(rows)!=286 or len({(r['id'],r['action']) for r in rows})!=286:
            raise ValueError('Incomplete matrix')
    else:
        result=read_json(run/'result.json')
        if result['status']!='complete' or sha256(run/'calibration.jsonl')!=result['artifacts_sha256']['calibration.jsonl']:
            raise ValueError('Incomplete/changed calibration')
        rows=[{**r,'action':'base'} for r in map(json.loads,(run/'calibration.jsonl').read_text().splitlines())]
        if [(r['case'],r['id'],r['noise_seed']) for r in rows]!=[(r['case'],r['id'],r['noise_seed']) for r in plan['calibration']]:
            raise ValueError('Calibration coverage differs')
    out.mkdir(parents=True,exist_ok=False)
    reader=AuditOCR(plan['audit_ocr_config']); started=time.perf_counter(); cache={}
    for i,r in enumerate(rows):
        if sha256(r['image'])!=r['image_sha256']:
            raise ValueError('Input image changed')
        h=r['image_sha256']
        if h not in cache:
            cache[h]=reader(r['image'],h)
        text_index=next(i for i,c in enumerate(r['record']['constraints']) if c['kind']=='ocr')
        expected=r['record']['constraints'][text_index]['text']
        primary=r['v1']['details']['ocr']['recognized_text']
        rules=ocr_checks(primary,cache[h]['text'],expected)
        if list(rules['v1'])!=[r['v1']['satisfied'][text_index],r['v1']['progress'][text_index]]:
            raise ValueError('Original OCR score cannot be reconstructed')
        append_jsonl(out/'ocr.jsonl',{**r,'text_index':text_index,'expected':expected,'secondary':cache[h],
                     'rules':{k:{'exact':v[0],'progress':v[1]} for k,v in rules.items()}})
        if (i+1)%50==0: print({'event':'ocr_progress','images':i+1},flush=True)
    write_json(out/'result.json',{'status':'complete','cohort':args.cohort,'images':len(rows),'unique_ocr_calls':len(cache),
               'elapsed_seconds':time.perf_counter()-started,'gpu_hours':0,'reader_version':reader.version,
               'config_sha256':sha256(plan['audit_ocr_config']),'plan_sha256':sha256(run/'measurement_plan.json'),
               'rows_sha256':sha256(out/'ocr.jsonl'),'source_sha256':sha256(__file__)})
    print({'status':'complete','cohort':args.cohort,'images':len(rows),'unique_ocr_calls':len(cache)})
