#!/usr/bin/env python
"""校验论文正文里的手写数字:穷举所有可计算的配对量,查正文的每个数能否对上。
对不上的不一定错(可能来自别处),但必须人工确认 —— 这是防"臂标错/对照错"的网。"""
import sys, os, re, glob, itertools, importlib.util
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
PSETS=("drawbench","unseen1k"); NFES=(2,4,6,10,20,40)
# 只取论文里真正出现的臂(穷举全部会慢到超时)
tags=["baseline_20","baseline_30","baseline_50","baseline_80","baseline_100","baserep_20","baserep_30",
      "baserep_50","baserep_80","baserep_100","bases2_100","basenp3_100","dflt_basemodel",
      "multikF4_20","multikF4_30","multikF4_50","multikF4_80","multikF4_100","mkF4seed2_100",
      "multik4_20","multik4_30","multik4_50","multik4_80","multik4_100","multikHI_50",
      "base6_100","base6mk_100","qstrain7_100","qs7mk_50","qs7mk_100","lr8x_100","lr8x_50",
      # 官方口径对齐批次(§4.692 起)
      "o3canon300_100","o3canon300_300","o3msp300_100","o3msp300_300",
      "rshpsv2canon_50","rshpsv2canon_100","rshpsv2msp_50","rshpsv2msp_100",
      "rspickscorecanon_100","rspickscoremsp_100","rsclipscorecanon_100","rsclipscoremsp_100",
      "beta10_50","beta10_100","beta03_50","beta03_100",
      "multikHI_50","multikHI_100","multiklr_50","multiklr_100",
      "anneal200_250","anneal200_300","resumectl200_250","resumectl200_300",
      "canonresume200_250","canonresume200_300","mspsoup3_300","canonsoup3_300","rblposeuler_50","rblposeuler_100"]
tags=[t for t in tags if glob.glob("outputs/matched_pi_%s_default_*.json"%t)]
_C={}
def load(t,p,n,ax):
    k=(t,p,n,ax)
    if k not in _C:
        f="matched_pi_%s_default_%s_flow%d"%(t,p,n)
        _C[k]=J.load(f,ax) if os.path.exists("outputs/%s.json"%f) else None
    return _C[k]
cat={}   # 值 -> [描述]
for a,b in itertools.permutations(tags,2):
    for p in PSETS:
        for n in NFES:
            for ax,lab in (("open3","MDE"),("imagereward","imgR_t")):
                A=load(a,p,n,ax); B=load(b,p,n,ax)
                if A is None or B is None: continue
                r=J.paired(A,B)
                v = r[1]/J.MDE if ax=="open3" else r[3]
                key="%.2f"%v
                cat.setdefault(key,[]).append("%s vs %s %s/%d %s"%(a,b,p[:2],n,lab))
                cat.setdefault("%.1f"%v,[]).append("%s vs %s %s/%d %s"%(a,b,p[:2],n,lab))
# ── 绝对量:各臂各格各列的均值(论文里的绝对分表、复现对照表用的是这些)──
import json as _json
for t in tags:
    for p in PSETS:
        for n in NFES:
            f="outputs/matched_pi_%s_default_%s_flow%d.json"%(t,p,n)
            if not os.path.exists(f): continue
            d=_json.load(open(f)); e=d["results"][0] if "results" in d else d
            for k,st in (e.get("score_stats") or {}).items():
                v=st.get("mean")
                if v is None: continue
                for fmt in ("%.1f","%.2f","%.3f","%.4f"):
                    cat.setdefault(fmt%v,[]).append("abs %s %s/%d %s"%(t,p[:2],n,k))
            # 复合分绝对值(论文多处引用)
            A=load(t,p,n,"open3")
            if A:
                import statistics as _stat
                m=_stat.mean(A.values())
                for fmt in ("%.3f","%.4f"): cat.setdefault(fmt%m,[]).append("abs-composite %s %s/%d"%(t,p[:2],n))
# ── 官方发表值:从主表生成器的源码提取,**不另抄一份**(单一真相源)──
_src=open("paper_figs/make_table_official.py",encoding="utf-8").read()
for _m in re.finditer(r'R\(([-\d.,\s]+)\)', _src):
    for _x in _m.group(1).split(","):
        _x=_x.strip()
        if not _x: continue
        try: _v=float(_x)
        except ValueError: continue
        for fmt in ("%.1f","%.2f","%.3f"): cat.setdefault(fmt%_v,[]).append("published(opsd2026)")
# ── 同一模型的**少步缺口**(自己的 flow40 − 自己的 flow4),§4.728 用 ──
import statistics as _stt
for t in tags:
    for p in PSETS:
        for ax in ("open3","imagereward","pickscore","clipscore","hpsv2"):
            a4,a40=load(t,p,4,ax),load(t,p,40,ax)
            if not(a4 and a40): continue
            ks=sorted(set(a4)&set(a40))
            if len(ks)<2: continue
            dd=[a40[k]-a4[k] for k in ks]
            m=_stt.mean(dd); se=_stt.stdev(dd)/len(dd)**.5
            vals=[m/J.MDE, m/se, m] if ax=="open3" else [m, m/se]
            for v in vals:
                for fmt in ("%.1f","%.2f","%.4f"):
                    cat.setdefault(fmt%v,[]).append("deficit %s %s/%s"%(t,p[:2],ax))
# ── §4.723 的容差比值 |ours-official|/tol(复现对照表那一行)──
_OFF=dict(pickscore=25.51,clipscore=0.333,hpsv2=0.389,aesthetic=6.03,imagereward=1.51)
_REL=0.016; _ABSF=dict(imagereward=0.032)
for t in ("o3canon300_300","o3msp300_300","baseline_100","multikF4_100"):
    f="outputs/matched_pi_%s_default_drawbench_flow40.json"%t
    if not os.path.exists(f): continue
    d=_json.load(open(f)); e=d["results"][0] if "results" in d else d
    for k,o in _OFF.items():
        g=(e.get("score_stats") or {}).get(k,{}).get("mean")
        if g is None: continue
        tol=max(_REL*abs(o), _ABSF.get(k,0.0))
        cat.setdefault("%.2f"%(abs(g-o)/tol),[]).append("tol-ratio %s %s"%(t,k))
# 退火占可部署增益的百分比(§4.824)。验证器的匹配是数值碰撞,不校验来源 ⇒
# 论文里引的这四个百分数必须有**真正的**计算来源,否则只是碰巧撞上别的量。
for _ck in (250, 300):
    for _p in ("drawbench", "unseen1k"):
        def _mde(_a, _b):
            _A = load(_a, _p, 4, "open3"); _B = load(_b, _p, 4, "open3")
            return None if (_A is None or _B is None) else J.paired(_A, _B)[1] / J.MDE
        _ar  = _mde("anneal200_%d"%_ck, "resumectl200_%d"%_ck)
        _dep = _mde("anneal200_%d"%_ck, "canonresume200_%d"%_ck)
        if _ar is None or _dep is None or abs(_dep) < 1e-9: continue
        _v = 100.0 * _ar / _dep
        for fmt in ("%.1f", "%.2f"):
            cat.setdefault(fmt % _v, []).append("anneal-share %d %s" % (_ck, _p[:2]))

print("  可计算的值: %d 个(配对量 + 绝对量 + 官方发表值 + 容差比值)" % len(cat))
pat=re.compile(r'\$([+-]?\d+\.\d+)\$')
miss=[]
for f in sorted(glob.glob("paper_msp/sections/*.tex")):
    for i,l in enumerate(open(f,encoding='utf-8',errors='replace'),1):
        if l.strip().startswith('%'): continue
        for m in pat.finditer(l):
            s=m.group(1); v=float(s)
            if abs(v)<1.0 and '.' in s and len(s.split('.')[1])>2: continue   # 绝对分数/σ 之类
            k=("%+.2f"%v).lstrip('+') if '.' in s and len(s.split('.')[1])==2 else None
            hit = any(abs(float(c)-v)<0.005 for c in cat)
            if not hit: miss.append((os.path.basename(f),i,s,l.strip()[:72]))
print("\n  ═══ 对不上任何配对计算的数字(需人工确认来源)═══")
seen=set()
for f,i,s,l in miss:
    if (f,s) in seen: continue
    seen.add((f,s))
    print("  %-18s L%-4d %-9s %s" % (f,i,s,l))
print("\n  共 %d 处(去重后 %d)" % (len(miss), len(seen)))
