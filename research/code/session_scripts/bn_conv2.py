import torch,math
torch.manual_seed(1)
exec(open("bn_e2e.py").read().split('print(f"  逐样本')[0])
print("  小 B 偏差是否随 reps 收敛(=方差驱动,非公式错):")
for accum,bs,t in [(12,4,2500),(11,4,2500)]:
    r=[run(accum,bs,t,d=1500,reps=n) for n in (300,1500,6000)]
    print(f"    accum={accum} bs={bs} B={accum*bs} 真值={t}: "
          + " -> ".join(f"{x:.0f}" for x in r))
