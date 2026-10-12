#!/usr/bin/env python
"""未筛面板(附录):按评测文件的**文件序**取前 N 条,不做任何挑选。
与正文的按规则挑选面板并置,读者可自行判断挑选的影响。"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
# CFG=1.5,与正文 Fig.6 同一显示工作点 —— 两图之间**只差"选不选 prompt"**一个变量。
ROWS=[("OPSD","paper_figs/qualcfg/canonUnsel_cfg15_n4","#1f77b4"),
      ("+MSP (ours)","paper_figs/qualcfg/mspUnsel_cfg15_n4","#d62728")]
PICK=list(range(8))
fig,axes=plt.subplots(len(ROWS),len(PICK),figsize=(1.55*len(PICK),1.63*len(ROWS)))
for r,(rlab,d,c) in enumerate(ROWS):
    for i,p in enumerate(PICK):
        ax=axes[r][i]; ax.set_xticks([]); ax.set_yticks([])
        f="%s/%04d.png"%(d,p)
        if os.path.exists(f): ax.imshow(np.asarray(Image.open(f)))
        else: ax.text(.5,.5,"missing",ha="center",va="center",fontsize=6)
        for s in ax.spines.values(): s.set_linewidth(.5); s.set_color("#bbb")
        if i==0:
            ax.text(.04,.95,rlab,transform=ax.transAxes,fontsize=7.5,color="white",va="top",
                    weight="bold",bbox=dict(boxstyle="round,pad=0.22",fc=c,ec="none",alpha=.9))
fig.suptitle("Unselected: first eight prompts in file order, NFE$=$4 (CFG $=1.5$, same display setting as the main qualitative figure)",fontsize=8.5,y=1.0)
fig.tight_layout(rect=[0,0,1,0.965])
fig.savefig("paper_figs/fig8_qual_unselected.pdf",bbox_inches="tight")
fig.savefig("paper_figs/fig8_qual_unselected.png",dpi=140,bbox_inches="tight")
print("  ✅ paper_figs/fig8_qual_unselected.{pdf,png}")
