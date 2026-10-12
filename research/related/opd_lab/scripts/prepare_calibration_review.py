"""Prepare all new calibration images without showing scores to the observer."""
import argparse
import json
from pathlib import Path
import textwrap

from opdlab.io import read_json, sha256, write_json
from opdlab.measurement_audit import verify_inputs


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();run,out=Path(args.run).resolve(),Path(args.out).resolve()
    plan=verify_inputs(run);result=read_json(run/'result.json')
    if result['status']!='complete' or sha256(run/'calibration.jsonl')!=result['artifacts_sha256']['calibration.jsonl']:
        raise ValueError('Incomplete calibration')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    rows=[json.loads(s) for s in (run/'calibration.jsonl').read_text().splitlines()]
    if [r['id'] for r in rows]!=[r['id'] for r in plan['calibration']]:
        raise ValueError('Calibration order changed')
    out.mkdir(parents=True,exist_ok=False);template=[]
    for r in rows:
        if sha256(r['image'])!=r['image_sha256']:raise ValueError('Image changed')
        fig,ax=plt.subplots(figsize=(7,7.5),dpi=110)
        ax.imshow(Image.open(r['image']));ax.axis('off')
        fig.suptitle('\n'.join(textwrap.wrap(f"Case {r['case']:02d}: {r['record']['prompt']}",87)),fontsize=10)
        fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.82)
        fig.savefig(out/f"case-{r['case']:02d}.png");plt.close(fig)
        template.append({'case':r['case'],'clauses':{c['id']:None for c in r['record']['constraints']},
                         'full_visibility':None,'visible_text':None,'notes':''})
    write_json(out/'template.json',template)
    write_json(out/'key.json',rows)
    write_json(out/'provenance.json',{'run':str(run),'images':16,'plan_sha256':sha256(run/'measurement_plan.json'),
               'calibration_sha256':sha256(run/'calibration.jsonl'),'source_sha256':sha256(__file__)})
