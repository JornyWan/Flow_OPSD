#!/usr/bin/env python3
"""0010 配对实验状态:拉两臂 wandb 历史,做同 epoch 配对比较。

只在**共同 epoch 集**上比较 —— 公共集变化会造成选择效应(τ 上栽过一次)。
"""
import os, sys, math
os.environ.setdefault("WANDB_SILENT", "true")
import wandb

ARMS = {"gdirect(w=0)": "krimof9p", "gguard3(w=3)": "q2y5fo09", "gguard(w=10)": "l5xf3icz"}
# gguard3 的 run id 启动后填入;None 表示先跳过。
CR = ["cr_active_frac", "cr_violation", "cr_retention_loss", "cr_direct_loss",
      "cr_current_teacher_kl", "cr_reference_teacher_kl",
      "cr_direct_param_grad_norm", "cr_retention_param_grad_norm"]

api = wandb.Api(timeout=60)
hist = {}
for label, rid in list(ARMS.items()):
    if rid is None:
        print(f"{label:15s} (run id 未填,跳过)"); del ARMS[label]; continue
    r = api.run(f"jorny-cornell-university/flow_grpo/{rid}")
    rows = [x for x in r.scan_history() if "epoch" in x]
    hist[label] = {int(x["epoch"]): x for x in rows}
    print(f"{label:15s} state={r.state:9s} epochs={len(rows)}")

# 逐对各取公共 epoch 集(不用全局交集):否则一个刚启动的臂会把已有的两臂对比也挡掉。
# 每个对比各自报自己的 n —— 不同对比之间不可混用步集(公共集变化会造成选择效应)。
BASE = "gdirect(w=0)"
pair_common = {a: sorted(set(hist[BASE]) & set(hist[a])) for a in hist if a != BASE}
common = sorted(set.intersection(*[set(h) for h in hist.values()])) if hist else []
print(f"\n三臂公共 epoch 数 n={len(common)}"
      + (f"  范围 {common[0]}..{common[-1]}" if common else "  (剂量表暂不可用)"))
for a, c in pair_common.items():
    print(f"  {BASE} vs {a}: n={len(c)}" + (f"  范围 {c[0]}..{c[-1]}" if c else ""))
if not any(pair_common.values()):
    sys.exit()

# 前置条件检查
g = hist["gguard(w=10)"]
_ge = sorted(g)
act = [g[e].get("cr_active_frac", 0) or 0 for e in _ge]
grn = [g[e].get("cr_retention_param_grad_norm", 0) or 0 for e in _ge]
print(f"\n[前置] gguard cr_active_frac: 均值={sum(act)/len(act):.4f} 非零 epoch={sum(1 for a in act if a>0)}/{len(act)}")
print(f"[前置] gguard retention 梯度探针非零次数={sum(1 for v in grn if v>0)}/{len(grn)} (探针每 epoch 只打一个 batch,0 不代表无梯度)")
e0 = _ge[0]
print(f"[前置] epoch{e0} 冻结参考自洽: cur_kl={g[e0].get('cr_current_teacher_kl')} ref_kl={g[e0].get('cr_reference_teacher_kl')}")

def paired(key, arm):
    c = pair_common[arm]
    a = [hist[BASE][e].get(key) for e in c]
    b = [hist[arm][e].get(key) for e in c]
    d = [y-x for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 3: return None
    n = len(d)
    m = sum(d)/n
    sd = math.sqrt(sum((v-m)**2 for v in d)/(n-1)) if n > 1 else 0.0
    t = m/(sd/math.sqrt(n)) if sd > 0 else float('nan')
    # 逐 epoch 的训练指标高度自相关,直接用 n 会高估显著性。
    # 用 lag-1 自相关做有效样本量校正 n_eff = n*(1-r1)/(1+r1),并给出校正后的 t。
    num = sum((d[i]-m)*(d[i+1]-m) for i in range(n-1))
    den = sum((v-m)**2 for v in d)
    r1 = num/den if den > 0 else 0.0
    r1 = max(-0.99, min(0.99, r1))
    # 负自相关会让 n*(1-r1)/(1+r1) > n,凭空造出样本量并抬高 t(实测 loss 一项 n=7 被算成 n_eff=14,
    # t 从 2.05 抬到 2.90 越过阈值)。有效样本量不可能超过实际样本量,必须封顶。
    n_eff = min(float(n), max(2.0, n*(1-r1)/(1+r1)))
    t_eff = m/(sd/math.sqrt(n_eff)) if sd > 0 else float('nan')
    pos = sum(1 for v in d if v > 0)
    return m, sd, t, pos, n, r1, n_eff, t_eff

for arm in [a for a in ARMS if a != BASE]:
    if not pair_common.get(arm): 
        print(f"\n=== {arm} − gdirect: 尚无公共 epoch,跳过 ==="); continue
    print(f"\n=== 同 epoch 配对 ({arm} − gdirect), n={len(pair_common[arm])} ===")
    print(f"{'metric':22s} {'mean diff':>12s} {'t_naive':>8s} {'r1':>6s} {'n_eff':>6s} {'t_eff':>7s} {'+/n':>7s}")
    for k in ["reward_mean", "loss", "train_reverse_kl", "rtc_R0", "rtc_acc_p", "cr_direct_loss", "grad_norm"]:
        r = paired(k, arm)
        if r is None: continue
        m, sd, t, pos, n, r1, n_eff, t_eff = r
        star = "  <-- |t_eff|>2.5" if abs(t_eff) > 2.5 else ""
        print(f"{k:22s} {m:12.6f} {t:8.2f} {r1:6.2f} {n_eff:6.1f} {t_eff:7.2f} {pos:3d}/{n:<3d}{star}")
    print("  t_eff 用 lag-1 自相关校正有效样本量(n_eff=n(1-r1)/(1+r1));判显著以 t_eff 为准。")

# 剂量-反应:效应应随 retention weight 单调变化。若不随剂量变化,任何"差异"更可能是偏置而非机制。
print("\n=== 剂量-反应 (末个共同 epoch) ===")
if common:
    e = common[-1]
    print(f"{'arm':16s} {'cur_teacher_kl':>15s} {'violation':>12s} {'active':>8s} {'direct_loss':>12s}")
    for a in ARMS:
        x = hist[a][e]
        print(f"{a:16s} {x.get('cr_current_teacher_kl', float('nan')):>15.5e} "
              f"{x.get('cr_violation', float('nan')):>12.3e} {x.get('cr_active_frac', float('nan')):>8.4f} "
              f"{x.get('cr_direct_loss', float('nan')):>12.5f}")
print("\n注:训练指标只说明优化行为,方法结论必须看 checkpoint 的 OCR/GenEval/PickScore 评测。")
