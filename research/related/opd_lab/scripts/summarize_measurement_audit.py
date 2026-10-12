"""Report fixed measurement rules, provisional labels, and all action sensitivities."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from opdlab.action_probe import ARMS, BETAS
from opdlab.io import read_json, sha256, write_json
from opdlab.measurement_audit import RULES, verify_inputs
from opdlab.method import Verdict
from opdlab.report import prompt_summary
from opdlab.runtime import verify_run


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def read_ocr(path, run):
    result=read_json(path/'result.json')
    if result['status']!='complete' or result['rows_sha256']!=sha256(path/'ocr.jsonl') or result['plan_sha256']!=sha256(run/'measurement_plan.json'):
        raise ValueError('Incomplete/changed OCR audit')
    rows=lines(path/'ocr.jsonl')
    if len(rows)!=result['images']:raise ValueError('OCR coverage mismatch')
    for r in rows:
        if sha256(r['image'])!=r['image_sha256']:raise ValueError('Changed audit image')
    return rows,result


def confusion(pairs):
    c=Counter(TP=0,FP=0,TN=0,FN=0,unknown=0,predicted_positive_unknown=0)
    for pred,label in pairs:
        if label is None:
            c['unknown']+=1;c['predicted_positive_unknown']+=int(pred)
        else:
            c[('T' if pred==label else 'F')+('P' if pred else 'N')]+=1
    return {**c,'precision_known':c['TP']/(c['TP']+c['FP']) if c['TP']+c['FP'] else None,
            'recall_known':c['TP']/(c['TP']+c['FN']) if c['TP']+c['FN'] else None}


def summarize(run, matrix_path, calibration_path, review):
    verify_run(run);plan=verify_inputs(run);gpu=read_json(run/'result.json')
    if gpu['status']!='complete' or gpu['optimizer_updates']!=0:raise ValueError('Incomplete diagnostic')
    for f,h in gpu['artifacts_sha256'].items():
        if sha256(run/f)!=h:raise ValueError('Changed GPU artifact')
    matrix,matrix_result=read_ocr(matrix_path,run)
    calibration,calibration_result=read_ocr(calibration_path,run)
    frozen=read_json(review/'observations_frozen.json')
    if frozen['observations_sha256']!=sha256(review/'observations.json') or frozen['scores_revealed']:
        raise ValueError('Calibration observations were not frozen before score reveal')
    labels=read_json(review/'observations.json')
    if [r['case'] for r in labels]!=list(range(1,17)):raise ValueError('Incomplete calibration review')
    old_review=Path(plan['review_run']);old_obs={r['case']:r for r in read_json(old_review/'observations.json')}
    old_frozen=read_json(old_review/'observations_frozen.json')
    if old_frozen['observations_sha256']!=sha256(old_review/'observations.json'):raise ValueError('Development labels changed')
    mmap={(r['id'],r['action']):r for r in matrix}
    ids=read_json(Path(plan['action_run'])/'action_plan.json')['query_ids']
    names=[f'ray@{b:g}' for b in BETAS]+[f'{a}/{k}' for a in ARMS for k in ('actual','projected')]
    if len(matrix)!=286 or set(mmap)!={(i,n) for i in ids for n in names}:raise ValueError('Incomplete action matrix')
    if [(r['case'],r['id'],r['noise_seed']) for r in calibration]!=[(r['case'],r['id'],r['noise_seed']) for r in plan['calibration']]:
        raise ValueError('Calibration coverage differs')
    development=[]
    for key in read_json(old_review/'key.json'):
        for side,m in key['mapping'].items():
            row=mmap[key['id'],'ray@0' if m['action']=='noop' else 'ray@1']
            if row['image_sha256']!=m['row']['image_sha256']:raise ValueError('Development endpoint mismatch')
            development.append((row,old_obs[key['case']]['observations'][side],key['case']))
    cohorts={'development':development,'calibration':[(r,l,r['case']) for r,l in zip(calibration,labels)]}
    confusions={name:{rule:confusion([(r['rules'][rule]['exact'],label['clauses']['text']) for r,label,_ in pairs])
                      for rule in RULES} for name,pairs in cohorts.items()}
    traces=lines(run/'scene_traces.jsonl')
    tmap={(t['cohort'],t['id'],t['image_sha256']):t for t in traces}
    expected={(cohort,r['id'],r['image_sha256']) for cohort,pairs in cohorts.items() for r,_,_ in pairs}
    if len(traces)!=60 or len(tmap)!=60 or set(tmap)!=expected:raise ValueError('Incomplete scene traces')
    scene={}; counting_checks=[]
    for cohort,pairs in cohorts.items():
        disagreements=[];visibility=[];class_pairs=[]
        for r,label,case in pairs:
            t=tmap[cohort,r['id'],r['image_sha256']]
            counting_checks += [{'cohort':cohort,'case':case,'image_sha256':r['image_sha256'],'clause':k,'same':v}
                                for k,v in t['counting_reproduction'].items()]
            visibility.append((t['frame_flag'],None if label['full_visibility'] is None else not label['full_visibility']))
            for j,c in enumerate(r['record']['constraints']):
                if c['kind']=='ocr':continue
                seen=label['clauses'][c['id']];pred=r['v1']['satisfied'][j]
                class_pairs.append((pred,seen))
                if seen is not None and pred!=seen:
                    disagreements.append({'case':case,'id':r['id'],'action':r['action'],'image_sha256':r['image_sha256'],
                        'clause':c['id'],'v1':pred,'visual':seen,'notes':label['notes'],
                        'trace':next(x for x in t['clauses'] if x['id']==c['id']),
                        'raw_requested_class_boxes':t['raw_requested_class_boxes']})
        scene[cohort]={'explicit_clause_confusion':confusion(class_pairs),'disagreements':disagreements,
                       'frame_flag_vs_visual_visibility_failure':confusion(visibility)}
    weight=read_json(run/'config.json')['utility_joint_weight']

    def value(row,rule):
        sat,prog=list(row['v1']['satisfied']),list(row['v1']['progress'])
        j=row['text_index'];sat[j]=row['rules'][rule]['exact'];prog[j]=row['rules'][rule]['progress']
        return Verdict(tuple(sat),tuple(prog)).utility(weight),sat

    for row in matrix+calibration:
        if value(row,'v1')[0]!=row['v1']['utility']:raise ValueError('Original utility changed')

    def contrast(candidate,baseline,rule):
        values=[];protected=[];damaged=[]
        for i in ids:
            v,s=value(mmap[i,candidate],rule);b,bs=value(mmap[i,baseline],rule)
            harm=any(x and not y for x,y in zip(bs,s));values.append((i,v-b))
            if harm:damaged.append(i)
            if v>b and not harm:protected.append(i)
        return {'utility':prompt_summary(values),'protected_positive_ids':protected,'damaged_ids':damaged,
                'positive':sum(v>0 for _,v in values),'negative':sum(v<0 for _,v in values)}

    sensitivity={rule:{'ray_vs_noop':{f'{b:g}':contrast(f'ray@{b:g}','ray@0',rule) for b in BETAS},
                       'projected_minus_actual':{a:contrast(f'{a}/projected',f'{a}/actual',rule) for a in ARMS}}
                 for rule in RULES}
    return {'scope':'Fixed measurement checks; provisional implementing-assistant labels; no independent human truth or deployed-policy efficacy claim. Matrix shares GenEval; calibration uses previously trained prompts with fresh noise.',
            'ocr_confusion':confusions,'scene':scene,'counting_reproduction':counting_checks,
            'sensitivity':sensitivity,'gpu':gpu,'cpu':[matrix_result,calibration_result],
            'calibration_observations_frozen':frozen,'images':{'development':44,'matrix_including_development':286,'new_calibration':16},
            'source_sha256':sha256(__file__),'plan_sha256':sha256(run/'measurement_plan.json')}


def render(s,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for rule in RULES:
        rows=list(s['sensitivity'][rule]['ray_vs_noop'].values());y=[r['utility']['mean'] for r in rows]
        axes[0].plot(BETAS,y,'o-',label=rule)
    axes[0].set(xlabel='Fraction of selected target correction',ylabel='Utility delta vs noop',title='Same 22 selected TRAIN prompts')
    axes[0].legend(fontsize=8);axes[0].axhline(0,color='gray',lw=.8)
    for offset,metric in ((-.16,'FP'),(.16,'FN')):
        axes[1].bar(np.arange(4)+offset,[s['ocr_confusion']['calibration'][r][metric] for r in RULES],.3,label=metric)
    axes[1].set(xticks=np.arange(4),xticklabels=RULES,title='16 fresh calibration images: provisional labels',ylabel='Number of errors')
    axes[1].tick_params(axis='x',rotation=20);axes[1].legend()
    for ax in axes:ax.grid(axis='y',alpha=.2)
    for ext in ('png','pdf'):fig.savefig(out/f'measurement.{ext}',dpi=160)
    plt.close(fig)
    report=['# Measurement audit','',s['scope'],'', '| Cohort | Rule | TP | FP | TN | FN | Unknown | Precision | Recall |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort,values in s['ocr_confusion'].items():
        for rule,v in values.items():
            report.append(f"| {cohort} | {rule} | {v['TP']} | {v['FP']} | {v['TN']} | {v['FN']} | {v['unknown']} | {v['precision_known']} | {v['recall_known']} |")
    report += ['', 'Unknown labels are not silently counted as correct or incorrect. Rule agreement is not truth. Border flags do not cover occlusion or all clipping. Full counts/traces/intervals are in summary.json.']
    (out/'report.md').write_text('\n'.join(report)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for key in ('run','matrix','calibration','review','out'):p.add_argument('--'+key,required=True)
    args=p.parse_args();paths=[Path(getattr(args,k)).resolve() for k in ('run','matrix','calibration','review','out')]
    s=summarize(*paths[:4]);out=paths[4];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'summary.json',s);render(s,out)
    print(json.dumps({'ocr_confusion':s['ocr_confusion'],'counting_reproduction_checks':len(s['counting_reproduction']),
                      'counting_mismatches':sum(not r['same'] for r in s['counting_reproduction'])}))
