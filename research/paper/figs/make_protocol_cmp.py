#!/usr/bin/env python
"""协议对比:同一批模型,现行协议(512/CFG=1/NFE=4)vs CFG=4.5/NFE=40。"""
import os, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
def grid(f,per=8):
    im=np.asarray(Image.open(f).convert('L'),dtype=np.float32); im-=im.mean(); N=im.shape[0]
    r=[]
    for spec in (np.abs(np.fft.rfft(im,axis=1)).mean(0), np.abs(np.fft.rfft(im,axis=0)).mean(1)):
        k=int(round(N/per)); lo,hi=max(1,k-14),min(len(spec)-1,k+15)
        b=np.median(np.r_[spec[lo:k-2], spec[k+3:hi]]); r.append(spec[k]/(b+1e-9))
    return float(np.mean(r))
COLS=[("base",            "paper_figs/qualdiag/r512_cfg1_n4",      "paper_figs/qualdiag/r512_cfg45_n40",  "#666666"),
      ("OPSD (ours repro)","paper_figs/qual3/canonical_nfe4",      "paper_figs/qualdiag2/canon_cfg45_n40","#1f77b4"),
      ("+MSP (ours)",      "paper_figs/qual3/oursC_nfe4",          "paper_figs/qualdiag2/msp_cfg45_n40",  "#d62728"),
      ("official OPSD hpsv2","paper_figs/qualofc/opsdOfficialHps_nfe4","paper_figs/qualdiag2/ofc_cfg45_n40","#2ca02c")]
PROMPTS=[4,3,9]
G={}
for lab,a,b,c in COLS:
    for d in (a,b):
        fs=sorted(glob.glob(d+"/*.png")); G[d]=np.mean([grid(f) for f in fs]) if fs else float("nan")
fig,axes=plt.subplots(2*len(PROMPTS), len(COLS), figsize=(2.6*len(COLS), 2.85*len(PROMPTS)))
for pi,p in enumerate(PROMPTS):
    for ci,(lab,dcur,dnew,c) in enumerate(COLS):
        for ri,d in enumerate((dcur,dnew)):
            ax=axes[2*pi+ri][ci]; ax.set_xticks([]); ax.set_yticks([])
            f="%s/%04d.png"%(d,p)
            if os.path.exists(f): ax.imshow(np.asarray(Image.open(f)))
            else: ax.text(.5,.5,"missing",ha="center",va="center",fontsize=7)
            if pi==0 and ri==0: ax.set_title(lab, fontsize=9, color=c, pad=5)
            if ci==0:
                ax.set_ylabel(("current protocol\nCFG=1, NFE=4" if ri==0 else "CFG=4.5, NFE=40"),
                              fontsize=7.5, color=("#b00" if ri==0 else "#070"))
            ax.set_xlabel("8px grid %.2f"%G[d], fontsize=6.5, labelpad=1.5,
                          color=("#b00" if ri==0 else "#070"))
fig.suptitle("Same model, same prompt, same seed — only the inference protocol changes (all at 512).",
             fontsize=10, y=0.998)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig("paper_figs/protocol_cmp.png", dpi=150, bbox_inches="tight")
fig.savefig("paper_figs/protocol_cmp.pdf", bbox_inches="tight")
print("  ✅ paper_figs/protocol_cmp.png")
