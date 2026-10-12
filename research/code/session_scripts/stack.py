import sys, importlib.util, statistics as st
SP="/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec=importlib.util.spec_from_file_location("judge",SP+"/judge.py"); J=importlib.util.module_from_spec(spec)
sys.argv=[sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
def f(t,p,n): return f"matched_pi_{t}_default_{p}_flow{n}"
CELL={"A 基线 (10步rollout, 无MSP)":"baseline_100","B 缩短rollout (6步)":"base6_100",
      "C MSP (ours)":"multikF4_100","D 两者 (6步+MSP)":"base6mk_100"}
for n in (4,40):
  print(f"══════ flow-{n} ══════")
  print(f"{'臂':<30}{'db 绝对':>10}{'u1k 绝对':>10}   {'db −A ×MDE':>12}{'t':>7}  {'u1k −A ×MDE':>13}{'t':>7}")
  base={}
  for p in ("drawbench","unseen1k"): base[p]=J.load(f("baseline_100",p,n),"open3")
  for lab,t in CELL.items():
    row=[]; absv=[]
    for p in ("drawbench","unseen1k"):
      A=J.load(f(t,p,n),"open3"); absv.append(st.mean(A.values()))
      r=J.paired(A,base[p]); row.append((r[1]/J.MDE, r[3]))
    print(f"{lab:<30}{absv[0]:>10.4f}{absv[1]:>10.4f}   {row[0][0]:>+12.2f}{row[0][1]:>+7.2f}  {row[1][0]:>+13.2f}{row[1][1]:>+7.2f}")
  # 交互:(D−B) − (C−A) = MSP 在 6 步 rollout 上的增益 − 在 10 步上的增益
  for p in ("drawbench","unseen1k"):
    A=base[p]; B=J.load(f("base6_100",p,n),"open3"); C=J.load(f("multikF4_100",p,n),"open3"); D=J.load(f("base6mk_100",p,n),"open3")
    ca=J.paired(C,A)[1]/J.MDE; db=J.paired(D,B)[1]/J.MDE
    dc=J.paired(D,C); dbA=J.paired(D,A)
    print(f"  [{p}] MSP增益: 在10步rollout上 {ca:+.2f}  在6步rollout上 {db:+.2f}  ⇒ 交互 {db-ca:+.2f}×MDE")
    print(f"         D vs C(最好单项): {dc[1]/J.MDE:+.2f}×MDE t={dc[3]:+.2f} | D vs A: {dbA[1]/J.MDE:+.2f}×MDE t={dbA[3]:+.2f}")
  print()
