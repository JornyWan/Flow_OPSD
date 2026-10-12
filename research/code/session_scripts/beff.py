import sys
sys.path.insert(0, "."); sys.path.insert(0, "../shim_rpi")
from config import public
c = public.get_config("sd35_open3")
bs = int(c.sample.train_batch_size); acc = int(c.train.gradient_accumulation_steps)
keff = 942
print("  == 每个 optimizer step 实际用到多少样本 ==")
for np_ in (3, 4):
    B = bs * acc * np_
    print("    bs=%d x accum=%d x nproc=%d  =>  B_eff = %d   B_eff/k_eff = %.2f" % (bs, acc, np_, B, B / keff))
print()
print("  实测梯度有效维度 k_eff = %d (canonical, 全部 epoch)" % keff)
print()
print("  => B_eff 与 k_eff 同量级,正好在转折点上:")
print("     B << k : 批梯度被抽样噪声主导")
print("     B >> k : 才收敛到真实平均梯度")
