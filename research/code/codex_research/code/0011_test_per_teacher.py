"""CODEX-0011 逐 teacher 预算的语义测试。

验证三件事:
  1. gather 取到的确实是"本样本路由到的那个 teacher"那一列;
  2. 逐 teacher 预算下,到 **其他** teacher(尤其 PickScore)的 KL 变化不再影响该样本的违约;
  3. 0002 的聚合口径在同一构造下会被其他 teacher 的 KL 污染(即复现出问题本身)。
"""
import torch

B, T = 6, 3            # 6 样本, 3 teacher (0=pickscore, 1=ocr, 2=geneval)
REL, ABS = 0.02, 1e-8
RET_SLOTS = (1, 2)

torch.manual_seed(0)
tidx = torch.tensor([0, 1, 2, 1, 2, 0])
ref_pt = torch.rand(B, T) * 1e-3 + 1e-3          # 冻结参考,逐 teacher
cur_pt = ref_pt.clone()

# 只让 **pickscore(slot 0)** 那一列显著变大 —— 模拟 reward 优化推高对 pickscore teacher 的 KL
cur_pt[:, 0] = ref_pt[:, 0] * 5.0

own = tidx.clamp(min=0).unsqueeze(1)
ref_own = ref_pt.gather(1, own).squeeze(1)
cur_own = cur_pt.gather(1, own).squeeze(1)

# 1) gather 正确性
for i in range(B):
    assert torch.isclose(ref_own[i], ref_pt[i, tidx[i]]), f"ref gather 错 @{i}"
    assert torch.isclose(cur_own[i], cur_pt[i, tidx[i]]), f"cur gather 错 @{i}"

# 2) 逐 teacher 口径:retention 样本(slot 1/2)完全没变过 -> 违约必须 <= 0
viol_pt = cur_own - ((1.0 + REL) * ref_own + ABS)
ret_mask = torch.zeros(B)
for s in RET_SLOTS:
    ret_mask += (tidx == s).float()
act_pt = ((viol_pt > 0).float() * ret_mask).sum() / ret_mask.sum().clamp(min=1)
assert float(act_pt) == 0.0, f"逐 teacher 口径不该激活,实得 {float(act_pt)}"

# 3) 0002 的聚合口径:同一构造下会被 pickscore 列污染 -> 违约转正、hinge 激活
cur_sum, ref_sum = cur_pt.sum(1), ref_pt.sum(1)
viol_agg = cur_sum - ((1.0 + REL) * ref_sum + ABS)
act_agg = ((viol_agg > 0).float() * ret_mask).sum() / ret_mask.sum().clamp(min=1)
assert float(act_agg) == 1.0, f"聚合口径应全激活,实得 {float(act_agg)}"

print("CODEX-0011 per-teacher budget: PASS")
print(f"  同一构造下 active_frac: 逐teacher={float(act_pt):.3f}  聚合(0002)={float(act_agg):.3f}")
print("  -> 证实 0002 的违约可以完全由 **非 retention** teacher(PickScore)的 KL 增长造成,")
print("     而逐 teacher 口径不受其影响。")
