import json, math, os
O="/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs"
MDE=0.0033
def load(tag,p,n):
    f=f"{O}/matched_pi_{tag}_default_{p}_flow{n}.json"
    return json.load(open(f)) if os.path.exists(f) else None
def col(d,m):
    out={}
    for r in d["per_image"]:
        v=r.get(m) if m in r else (r.get("scores") or {}).get(m)
        if v is not None: out[r["index"]]=float(v)
    return out
def stat(a,b,m):
    A,B=col(a,m),col(b,m)
    ks=sorted(set(A)&set(B))
    if not ks: return None
    d=[A[k]-B[k] for k in ks]
    n=len(d); mu=sum(d)/n
    sd=math.sqrt(sum((x-mu)**2 for x in d)/(n-1))
    se=sd/math.sqrt(n)
    win=sum(1 for x in d if x>0)/n
    return dict(n=n,mu=mu,t=mu/se if se else float('nan'),mde=mu/MDE,win=win,
                A=sum(A[k] for k in ks)/n, B=sum(B[k] for k in ks)/n)
print(f"{'reward':<12}{'set':<10}{'nfe':<5}{'MSP':>9}{'canon':>9}{'Δ':>10}{'×MDE':>8}{'t':>8}{'win%':>7}")
print("-"*80)
for rw,tagm,tagc,metric in [
  ("imagereward","rsimagerewardmsp_100","rsimagerewardcanon_100","imagereward"),
]:
  for p in ("drawbench","unseen1k"):
    for n in (40,4):
      a,b=load(tagm,p,n),load(tagc,p,n)
      if not a or not b: print(f"{rw:<12}{p:<10}{n:<5}  ⏳ 缺"); continue
      s=stat(a,b,metric)
      print(f"{rw:<12}{p:<10}{n:<5}{s['A']:>9.4f}{s['B']:>9.4f}{s['mu']:>+10.4f}{s['mde']:>+8.2f}{s['t']:>+8.2f}{s['win']*100:>7.1f}")

print()
print("═══ 少步缺口(NFE40 − NFE4)与比值 ═══")
for p in ("drawbench","unseen1k"):
    r={}
    for lab,tag in (("MSP","rsimagerewardmsp_100"),("canon","rsimagerewardcanon_100")):
        a40,a4=load(tag,p,40),load(tag,p,4)
        c40,c4=col(a40,"imagereward"),col(a4,"imagereward")
        ks=sorted(set(c40)&set(c4))
        r[lab]=sum(c40[k]-c4[k] for k in ks)/len(ks)
    print(f"  {p:<10} canon={r['canon']:+.4f}  MSP={r['MSP']:+.4f}  比值={r['MSP']/r['canon']:.3f}  (收窄 {100*(1-r['MSP']/r['canon']):.0f}%)")
