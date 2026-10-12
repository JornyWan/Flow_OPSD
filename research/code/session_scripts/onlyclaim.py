import sys, importlib.util, os
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
NFE=[2,4,6,10,20,40]
SER=[("A 抬查询态 σ0.278→0.465","qstrain7_100","basenp3_100"),
     ("B 缩短 rollout 10→6","base6_100","baseline_100"),
     ("C 多态传播 (ours)","multikF4_100","baseline_100")]
def g(t,p,n):
    f=f"outputs/matched_pi_{t}_default_{p}_flow{n}.json"
    return f[8:-5] if os.path.exists(f) else None
print("判据:×MDE ≥ 2.5 且 t > 2(§4.615)。12 格 = 6 NFE × 2 prompt set")
for lab,a,b in SER:
    bad=[]; miss=[]; ok=0
    for p in ("drawbench","unseen1k"):
        for n in NFE:
            fa,fb=g(a,p,n),g(b,p,n)
            if not fa or not fb: miss.append(f"{p[:2]}/{n}"); continue
            r=J.paired(J.load(fa,"open3"),J.load(fb,"open3"))
            v=r[1]/J.MDE
            if v>=2.5 and r[3]>2: ok+=1
            else: bad.append(f"{p[:2]}/{n}={v:+.2f}(t={r[3]:+.2f})")
    print(f"  {lab:<28} 过阈 {ok}/{ok+len(bad)}" + (f"  ✗ 未过: {', '.join(bad)}" if bad else "  ✓ 全过")
          + (f"  [缺格 {', '.join(miss)}]" if miss else ""))
