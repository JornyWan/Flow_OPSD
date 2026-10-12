import importlib.util,sys
spec=importlib.util.spec_from_file_location("m","bn_e2e.py")
# 直接复用 run(),但避免执行 __main__ 打印:改为内联
import torch,math
torch.manual_seed(1)
exec(open("bn_e2e.py").read().split('print(f"  逐样本')[0])
print("  小 B 失败行是不是'重复次数不够'?随 reps 增大应收敛到真值:")
print(f"  {'配置':>14} {'真值':>6} | {'reps=300':>9} {'1200':>9} {'4800':>9}")
for accum,bs,t in [(12,4,2500),(11,4,2500),(7,6,1152)]:
    row=""
    for r in (300,1200,4800):
        row+=f"{run(accum,bs,t,reps=r):>10.0f}"
    print(f"  accum={accum:>2},bs={bs:>2}  B={accum*bs:>4} {t:>6} |{row}")
print("\n  实际训练配置(3卡: bs_per_rank=8 ×3 = 24/microbatch, accum=48, B=1152):")
for t in (400,835,1152,2500,4131):
    e=run(48,24,t,reps=1200)
    print(f"    真值 {t:>5} -> 估计 {e:>6.0f}  ({e/t-1:+.1%})")
