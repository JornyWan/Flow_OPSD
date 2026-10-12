"""端到端自检:用真实 torch .grad 累积复现训练循环,验证插入 train_opsd_ri_sd3.py 的公式。
故意包含奇数 accum,以检验 (accum/h)*(accum/(accum-h)) 这两个因子。"""
import torch, math
torch.manual_seed(0)

def run(accum, bs, Bn_true, d=4000, reps=300, broken=False):
    """L_i = w·a_i => dL_i/dw = a_i。‖G‖=1,trΣ=Bn_true。
    完全照抄训练循环:每 microbatch 反传 mean_j(L)/accum;累积到 half 快照;累积满算数。"""
    G = torch.zeros(d); G[0] = 1.0
    sd = math.sqrt(Bn_true / d)
    num_s = den_s = 0.0
    h = accum // 2
    for _ in range(reps):
        w = torch.zeros(d, requires_grad=True)
        half = None
        for m in range(accum):
            a = G + torch.randn(bs, d) * sd          # 逐样本梯度
            loss = (w * a.mean(0)).sum()             # microbatch 平均
            (loss / accum).backward()                # ← 与 :2002 同
            k = (m + 1) % accum
            if k == h:
                half = w.grad.detach().clone()
            elif k == 0:
                gf = w.grad.detach()
                dot = float((half.double() * (gf - half).double()).sum())
                f2 = float((gf.double() * gf.double()).sum())
                if broken:
                    ab = dot * 4.0                    # 错误版本:硬编码 2×2
                else:
                    ab = dot * (accum / h) * (accum / (accum - h))
                num_s += f2 - ab; den_s += ab
    B = accum * bs
    return B * num_s / den_s

print(f"  逐样本梯度总数 B = accum×bs;真值 = trΣ/‖G‖²")
print(f"  {'accum':>6} {'bs':>4} {'B':>6} | {'真值':>7} {'估计':>9} {'相对误差':>9}")
ok = True
for accum, bs in [(48, 24), (48, 8), (12, 4), (11, 4), (7, 6)]:
    for t in (400, 1152, 2500):
        e = run(accum, bs, t)
        rel = e / t - 1
        flag = "" if abs(rel) < 0.06 else "  ❌"
        if abs(rel) >= 0.06: ok = False
        print(f"  {accum:>6} {bs:>4} {accum*bs:>6} | {t:>7} {e:>9.0f} {rel:>+8.1%}{flag}")
print(f"\n  正向判据(全部 |相对误差|<6%): {'✅ 通过' if ok else '❌ 未通过'}")

print("\n  === 反向检验:奇数 accum 下硬编码因子 4(即假设 accum 为偶)应当**测出偏差** ===")
for accum, bs, t in [(11, 4, 1152), (7, 6, 1152)]:
    good = run(accum, bs, t); bad = run(accum, bs, t, broken=True)
    print(f"    accum={accum:>3} 真值={t}: 正确公式={good:>7.0f}  硬编码4={bad:>7.0f}"
          f"  (偏差 {bad/t-1:+.1%})  {'✅ 反向检出' if abs(bad/t-1)>0.06 else '❌ 反向未检出'}")
