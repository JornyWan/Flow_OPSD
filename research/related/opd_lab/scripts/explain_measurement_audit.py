"""Post-hoc explanatory examples and exact decomposition; no tuning or new scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from opdlab.io import read_json, sha256, write_json
from opdlab.report import prompt_summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('run','report','matrix','calibration','out'):p.add_argument('--'+key,required=True)
    args=p.parse_args();run,report,matrix,calibration,out=[Path(getattr(args,k)).resolve() for k in ('run','report','matrix','calibration','out')]
    summary=read_json(report/'summary.json')
    if summary['plan_sha256']!=sha256(run/'measurement_plan.json'):raise ValueError('Plan mismatch')
    cohorts={}
    for name,path in (('development',matrix),('calibration',calibration)):
        if sha256(path/'ocr.jsonl')!=read_json(path/'result.json')['rows_sha256']:raise ValueError('Changed rows')
        cohorts[name]=[json.loads(s) for s in (path/'ocr.jsonl').read_text().splitlines()]
    rows={(r['id'],r['action']):r for r in cohorts['development']}
    ids=sorted({i for i,a in rows});weight=read_json(run/'config.json')['utility_joint_weight']
    decomposition={}
    for beta in (0,.25,.5,.75,1):
        parts={'joint_bonus':[],'mean_progress':[],'total':[]}
        for i in ids:
            x,b=rows[i,f'ray@{beta:g}']['v1'],rows[i,'ray@0']['v1']
            joint=weight*(int(x['joint_success'])-int(b['joint_success']))
            progress=(1-weight)*(np.mean(x['progress'])-np.mean(b['progress']))
            if not np.isclose(joint+progress,x['utility']-b['utility'],atol=1e-12,rtol=0):raise ValueError('Decomposition mismatch')
            for k,v in (('joint_bonus',joint),('mean_progress',progress),('total',joint+progress)):parts[k].append((i,v))
        decomposition[str(beta)]={k:prompt_summary(v) for k,v in parts.items()}
    out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    fig,axes=plt.subplots(1,3,figsize=(12,4.6),constrained_layout=True)
    specifications=[('development',6,'ray@1','cup'),('development',16,'ray@1','apple'),('calibration',12,'base','vase')]
    examples=[]
    for ax,(cohort,case,action,name) in zip(axes,specifications):
        entry=next(r for r in summary['scene'][cohort]['disagreements'] if r['case']==case and r['action']==action)
        row=next(r for r in cohorts[cohort] if r['id']==entry['id'] and r['action']==action)
        if sha256(row['image'])!=entry['image_sha256']:raise ValueError('Example image changed')
        threshold=entry['trace']['threshold'];required=entry['trace']['objects'][name]['expected_count']
        ax.imshow(Image.open(row['image']));ax.axis('off')
        boxes=sorted(entry['raw_requested_class_boxes'][name],key=lambda b:b[4],reverse=True)
        for box in boxes:
            if box[4]<=.3:continue
            x1,y1,x2,y2,confidence=box
            kept=confidence>threshold;color='#00885a' if kept else '#c35a00'
            ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1.8,linestyle='-' if kept else '--'))
            ax.text(max(1,x1),max(10,y1),f'{confidence:.3f}',fontsize=8,color='white',backgroundcolor=color)
        ax.set_title(f'{cohort} case {case:02d}: {required} {name}\nthreshold {threshold:g}; kept {entry["trace"]["objects"][name]["count"]}',fontsize=10)
        examples.append({'cohort':cohort,'case':case,'action':action,'image_sha256':row['image_sha256'],'required':required,
                         'threshold':threshold,'top_boxes':boxes[:6],'original_v1':entry['v1'],'provisional_visual':entry['visual']})
    fig.suptitle('Illustrative post-hoc cases; boxes shown above 0.3; green retained, dashed orange filtered',fontsize=10)
    for ext in ('png','pdf'):fig.savefig(out/f'detector_examples.{ext}',dpi=160)
    plt.close(fig)
    write_json(out/'explanation.json',{'scope':'Post-hoc arithmetic decomposition and three illustrative error cases selected after phase5 outcomes. No parameter tuning or efficacy test.',
               'source_sha256':sha256(__file__),'summary_sha256':sha256(report/'summary.json'),'decomposition':decomposition,'examples':examples})
    print({'joint_bonus_fraction_of_full_target_mean_gain':decomposition['1']['joint_bonus']['mean']/decomposition['1']['total']['mean']})
