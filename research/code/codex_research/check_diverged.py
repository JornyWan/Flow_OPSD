#!/usr/bin/env python
"""训练发散的**运行期守卫**(不是判定判据)。§4.779 之前它只活在手工读图里,
结果一条发散臂跑满 100 步还排了 4 个评测(约 9.6 GPU·小时 + 6.5 A40·小时)。

两个指标:
  (A) §4.285 登记的判据:连续 ≥N 个 epoch 的 reward_avg 低于起点 ep0。
      ⚠️ 加 2% 相对余量 —— 不加会把从 ep100 续起的短探针(kfit 系,降幅 0.43%)
      全判成发散(实测 3 条假阳性)。
  (B) **峰值回撤**(新增,仅作运行期早报警,不参与任何实验判定):
      max over t of (峰值_到t − 值_t)/峰值_到t。aesthetic 臂在 ep50 就有 16% 回撤,
      而 (A) 要到 ep56 才触发 ⇒ (B) 能更早省算力。

⚠️ 只报告、不删不停:判负必须人看过逐点序列再定([[dont-kill-ideas-before-running]])。
用法: check_diverged.py [N] [margin%] [drawdown%]   默认 4 / 2 / 15
"""
import json, sys, glob, os
N   = int(sys.argv[1]) if len(sys.argv) > 1 else 4
MAR = (float(sys.argv[2]) if len(sys.argv) > 2 else 2.0) / 100
DD  = (float(sys.argv[3]) if len(sys.argv) > 3 else 15.0) / 100
rows = []
for f in sorted(glob.glob("outputs/*/metrics.jsonl")):
    arm = os.path.basename(os.path.dirname(f))
    xs = []
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        ep, r = d.get("epoch"), d.get("reward_avg")
        if isinstance(ep, int) and r is not None: xs.append((ep, float(r)))
    if len(xs) < N + 1: continue
    xs.sort(); start = xs[0][1]
    thr = start * (1 - MAR) if start > 0 else start * (1 + MAR)
    run = best = 0; at = None
    peak = xs[0][1]; dd = 0.0; dd_at = None
    for ep, r in xs[1:]:
        run = run + 1 if r < thr else 0
        if run > best: best, at = run, ep
        if r > peak: peak = r
        d_ = (peak - r) / peak if peak > 0 else 0.0
        if d_ > dd: dd, dd_at = d_, ep
    rows.append((best, dd, arm, len(xs), start, peak, xs[-1][1], at, dd_at))
rows.sort(key=lambda r: (-r[0], -r[1]))
print(f"守卫:(A) 连续 ≥{N} 个 epoch 低于 ep0 的 {100*(1-MAR):.0f}%;(B) 峰值回撤 ≥{DD*100:.0f}%。"
      f"共 {len(rows)} 条臂。")
print(f"{'A':>4}{'B回撤':>8} {'臂':<44}{'步':>4}{'ep0':>9}{'峰值':>9}{'末值':>9}  A@   B@")
na = nb = 0
for best, dd, arm, n, start, peak, last, at, dd_at in rows:
    fa = "A" if best >= N else " "; fb = "B" if dd >= DD else " "
    na += best >= N; nb += dd >= DD
    if not (best >= N or dd >= DD): continue
    print(f"{fa}{fb}{best:>2}{dd*100:>7.1f}% {arm:<44}{n:>4}{start:>9.4f}{peak:>9.4f}{last:>9.4f}"
          f"  {('ep'+str(at)) if at else '-':>6}{('ep'+str(dd_at)) if dd_at else '-':>7}")
print(f"\n⇒ 触发 (A): {na} 条   触发 (B): {nb} 条")
