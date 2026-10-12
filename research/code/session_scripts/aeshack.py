import sys, importlib.util, statistics as st
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
def f(t,p,n): return f"matched_pi_{t}_default_{p}_flow{n}"
AX=["aesthetic","open3","imagereward","pickscore","clipscore","hpsv2"]
print("Aesthetic 训练臂 @ck50 —— 六个轴并看(aesthetic 是训练目标,其余是代价)")
for p in ("drawbench","unseen1k"):
  for n in (4,40):
    print(f"\n── {p} flow-{n} ──")
    print(f"  {'轴':<13}{'MSP':>10}{'canon':>10}{'Δ(MSP−canon)':>15}{'t':>9}")
    for ax in AX:
        A=J.load(f("rsaestheticmsp_50",p,n),ax); B=J.load(f("rsaestheticcanon_50",p,n),ax)
        if not A or not B: print(f"  {ax:<13} 缺"); continue
        r=J.paired(A,B)
        print(f"  {ax:<13}{st.mean(A.values()):>10.4f}{st.mean(B.values()):>10.4f}{r[1]:>+15.4f}{r[3]:>+9.2f}")
print()
print("═══ 关键对照:未训练底模在这些轴上是多少(看 canon 偏离了多远)═══")
for p in ("drawbench",):
  for n in (4,40):
    print(f"  ── {p} flow-{n} ──")
    for ax in ("aesthetic","open3","imagereward"):
        try:
            D=J.load(f("dflt_basemodel",p,n),ax)
            print(f"    {ax:<13} 底模={st.mean(D.values()):>9.4f}")
        except Exception as e: print(f"    {ax:<13} 缺底模格")
