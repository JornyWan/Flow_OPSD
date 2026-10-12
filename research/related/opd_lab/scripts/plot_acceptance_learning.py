"""Plot the frozen primary outcome, full-path stress check, harm and training fit."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from opdlab.io import ROOT,read_json

c=ROOT/'experiments/local_expert_v20'
r=read_json(c/'learning_results.json')
models=['joint128','all_teacher','certified','random_matched']
labels=['Initial','All teacher','Certified','Random matched']
fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
x=np.arange(4)
for offset,path,color,label in [(-.18,'single','#318977','Single action (primary)'),(.18,'full','#a4b4c6','Full generation')]:
    for ax,cohort,metric,n in [(axes[0],'positive_fresh_seed','complete_scene',12),(axes[1],'all_fresh','known_harm',32)]:
        vals=[r['scores'][cohort][path][m][metric]['sum'] for m in models]
        bars=ax.bar(x+offset,vals,width=.35,color=color,label=label)
        ax.bar_label(bars,fmt='%g',padding=3,fontsize=8)
        ax.set_xticks(x,labels,rotation=20,ha='right')
        ax.set_ylim(0,max(1,max(vals))+3)
        ax.set_ylabel(f'Cases / {n}')
axes[0].set_title('Complete scenes: fresh positive seeds')
axes[0].set_ylim(0,13);axes[0].legend(fontsize=8,loc='upper left')
axes[1].set_title('Known harms: all fresh seeds')
axes[1].set_ylim(0,max(r['scores']['all_fresh'][path][m]['known_harm']['sum'] for path in ('single','full') for m in models)+3)
for model,color in zip(models[1:],['#57799b','#318977','#a77b58']):
    steps=[0,32,128]
    vals=[r['accepted_alignment'][model][str(s)]['median'] for s in steps]
    axes[2].plot(steps,vals,'o-',label=labels[models.index(model)],color=color)
axes[2].set_title('Fit of the 9 accepted training targets')
axes[2].set_xlabel('Optimizer updates');axes[2].set_ylabel('Median relative target MSE')
axes[2].set_ylim(bottom=0);axes[2].legend(fontsize=8)
fig.suptitle('Phase20 | fixed-query OPD-style study; provisional assistant labels',fontsize=13)
for ext in ('png','pdf'):fig.savefig(c/f'learning_outcomes.{ext}',dpi=180)
plt.close(fig)
print('Wrote complete primary/full outcomes, harm and training-fit plot.')
