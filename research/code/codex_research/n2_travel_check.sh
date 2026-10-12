#!/bin/bash
# 0049 §⑤ 的"无行程"自检:N2 一开跑就判断这个杠杆到底有没有行程。
# 若 median(cv)<0.05 且 |median(drift)-1|<0.05,则 N2 数值上 ≈ baseline,
# 继续跑满 100 updates 只会得到一个不可解释的打平(A1 刚演示过这种浪费)。
set -uo pipefail
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
D=$O/outputs/ab_open3_n2_s20260830_u100
echo "[n2check] 等 N2 产出前 40 步指标 $(date)"
while true; do
  ls "$D" >/dev/null 2>&1                      # 刷 NFS 属性缓存
  # ⚠️ 不能写 `grep -c ... || echo 0`:文件存在但零匹配时 grep 既打印 "0" 又返回 1,
  # 于是 || 分支也触发,得到两行 "0\n0",后面 [ ] 直接报 integer expression expected。
  n=$(grep -c 'n2_wf_cv_p' "$D/metrics.jsonl" 2>/dev/null); n=${n:-0}
  [ "$n" -ge 40 ] && break
  sleep 60
done
ls "$D" >/dev/null 2>&1
"$O/../conda_envs/envs/diffusionopd/bin/python" - "$D/metrics.jsonl" <<'PY'
import sys, json, io, statistics as st
rows=[json.loads(l) for l in io.open(sys.argv[1]) if l.strip() and 'n2_wf_cv_p' in l]
cv=[r['n2_wf_cv_p'] for r in rows]; dr=[r['n2_wf_drift_p'] for r in rows]
mcv, mdr = st.median(cv), st.median(dr)
print(f"[n2check] n={len(rows)}  median(cv)={mcv:.4f}  median(drift)={mdr:.4f}")
print(f"[n2check]   cv 范围 [{min(cv):.4f},{max(cv):.4f}]  drift 范围 [{min(dr):.4f},{max(dr):.4f}]")
dead = mcv < 0.05 and abs(mdr-1.0) < 0.05
print("[n2check] 判定: " + ("❌ 无行程 —— N2 数值上≈baseline,打平不可解释为'机制无效'(0049 §⑤)"
                            if dead else "✅ 有行程 —— N2 确实施加了干预,可按 0049 §④ 判读结果"))
PY
echo "[n2check] 完 $(date)"
