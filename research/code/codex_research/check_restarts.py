#!/usr/bin/env python
"""续训/重启审计。§4.782 查明:`save_ckpt` 写进 `lora/` 的是 **EMA 平均权重**
(`ema.copy_ema_to(..., store_temp=True)` 后 `save_pretrained`),而 resume 把它载为**活权重**
⇒ **任何训练中途的续跑都等于一次 EMA 重启**,是比"RNG 不恢复"大得多的扰动。
⇒ 连续训练的臂与中途续跑过的臂**不可直接配对**。

判别(§4.784 定标,靠的是 slurm 里守卫 requeue 在 L51/81/88、写 run_attempts 在 L417):
  - 被守卫拒掉的尝试**不写** run_attempts ⇒
    `restart>0` + 全部 `resumed=0` + **单行**  ⇒ 守卫重排,训练前就退了,**良性**
  - 任一行 `resumed=1`                        ⇒ **中途 EMA 重启**,标红
  - 多行且全 `resumed=0`                      ⇒ 多次独立提交、每次都从零开始,良性
                                                (前一次必然死在首个存点之前)
用法: check_restarts.py            扫全部臂
      check_restarts.py <子串>...  只看名字含该子串的臂
"""
import json, sys, glob, os
pats = sys.argv[1:]
rows = []
for f in sorted(glob.glob("outputs/*/run_attempts.jsonl")):
    arm = os.path.basename(os.path.dirname(f))
    if pats and not any(p in arm for p in pats): continue
    rs = []
    for l in open(f):
        try: rs.append(json.loads(l))
        except Exception: pass
    if not rs: continue
    mx = max(int(r.get("slurm_restart_count") or 0) for r in rs)
    res = [int(r.get("resumed") or 0) for r in rs]
    if any(res):      verdict, flag = "❌ 中途 EMA 重启(不可与连续臂配对)", 2
    elif len(rs) > 1: verdict, flag = "✓ 多次独立提交,均从零开始", 0
    elif mx > 0:      verdict, flag = "✓ 守卫重排(训练前退出),良性", 0
    else:             verdict, flag = "✓ 一次跑完", 0
    rows.append((flag, arm, mx, sum(res), len(rs), verdict))
rows.sort(key=lambda r: (-r[0], r[1]))
print(f"{'臂':<48}{'restart':>8}{'resumed':>8}{'行数':>5}  判")
nbad = 0
for flag, arm, mx, res, n, v in rows:
    if flag: nbad += 1
    print(f"{arm:<48}{mx:>8}{res:>8}{n:>5}  {v}")
print(f"\n⇒ 共 {len(rows)} 条;**中途 EMA 重启** 的 {nbad} 条")
