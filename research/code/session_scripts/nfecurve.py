import sys, importlib.util
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

NFES=(2,4,6,10,20,40)
print("MSP(multikF4_100) − canonical(baseline_100),open3 复合,逐图配对 MDE=0.0033")
print(f"{'NFE':>4} | {'drawbench ×MDE':>15}{'t':>8} | {'unseen1k ×MDE':>15}{'t':>8}")
print("-"*62)
rows={}
for n in NFES:
    out=[]
    for p in ("drawbench","unseen1k"):
        A=J.load(f"matched_pi_multikF4_100_default_{p}_flow{n}","open3")
        B=J.load(f"matched_pi_baseline_100_default_{p}_flow{n}","open3")
        r=J.paired(A,B)
        out.append((r[1]/J.MDE, r[3]))
    rows[n]=out
    print(f"{n:>4} | {out[0][0]:>+15.2f}{out[0][1]:>+8.2f} | {out[1][0]:>+15.2f}{out[1][1]:>+8.2f}")
print()
print("绝对分(看少步缺口本身):")
print(f"{'NFE':>4} | {'canon db':>9}{'MSP db':>9} | {'canon u1k':>10}{'MSP u1k':>9}")
import statistics as st
for n in NFES:
    v=[]
    for p in ("drawbench","unseen1k"):
        for tag in ("baseline_100","multikF4_100"):
            v.append(st.mean(J.load(f"matched_pi_{tag}_default_{p}_flow{n}","open3").values()))
    print(f"{n:>4} | {v[0]:>9.4f}{v[1]:>9.4f} | {v[2]:>10.4f}{v[3]:>9.4f}")
