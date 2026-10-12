#!/bin/bash
# 重生成所有表 → 同步到 paper_msp/tables/ → 编译。
# 加这个脚本的原因:生成器写 paper_figs/,论文读 paper_msp/tables/,此前靠手工 cp,
# 2026-10-01 发现 table_rewardspec 因此挂着过期的 "--"(§4.720)。
set -eu
O=/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related
PY=$O/../conda_envs/envs/diffusionopd/bin/python
TECTONIC=/midtier/minilab/scratch/chw4026/.cache/texenv/bin/tectonic
cd "$O"
for g in paper_figs/make_table*.py; do "$PY" "$g" >/dev/null; done
for g in paper_figs/make_figs.py paper_figs/make_fig7.py paper_figs/make_fig10.py paper_figs/make_fig11.py; do "$PY" "$g" >/dev/null; done
for f in paper_figs/*.tex; do
  b=$(basename "$f")
  [ -f "paper_msp/tables/$b" ] && cp "$f" "paper_msp/tables/$b"
done
# 图同样是手工副本 —— 只同步论文里已引用的那些,不引入新文件
for f in paper_figs/*.pdf; do
  b=$(basename "$f")
  [ -f "paper_msp/figures/$b" ] && cp "$f" "paper_msp/figures/$b"
done
"$PY" paper_figs/check_consistency.py || true
cd paper_msp
rm -f main.log                       # 必须先删:否则失败时会读到上一次的成功 log
"$TECTONIC" -X compile main.tex --keep-intermediates --keep-logs >/dev/null 2>&1 || true
[ -f main.log ] || { echo "❌ 编译没产出 main.log(tectonic 本身失败)"; exit 1; }
nerr=$(grep -c "^error:" main.log || true)
if [ "${nerr:-0}" -gt 0 ]; then echo "❌ $nerr errors"; grep "^error:" main.log | head; exit 1; fi
echo "✓ pages=$(grep -oP 'main.xdv \(\K[0-9]+' main.log || echo '?')" \
     "overfull=$(grep -c 'Overfull' main.log || true)" \
     "undefined=$(grep -c 'undefined' main.log || true)"
