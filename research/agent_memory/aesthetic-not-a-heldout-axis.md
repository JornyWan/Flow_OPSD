---
name: aesthetic-not-a-heldout-axis
description: "aesthetic 与训练复合分反相关(r=−0.395,符号一致 4/12)不能当 held-out 验证;imagereward 才是(r=+0.894,12/12)"
metadata:
  type: reference
---

2026-09-29,在 12 个臂上量出来(全部 vs `baserep`,A100/np=3/seed20260830/A40/flow-40,n=1000,
复合分跨 −35.46 到 +17.95×MDE):

```
r(报告复合分, imagereward) = +0.894   符号一致 12/12
r(报告复合分, aesthetic)   = −0.395   符号一致  4/12
```

**⇒ aesthetic 在本项目不能用作 held-out 验证轴**,它与复合分**反相关**、符号一致率比抛硬币还差。
最极端:`qstrain5` 复合分 −14.19×MDE 而 aesthetic **+27.70 t**;`inner2lrhalf` −15.42 而 aesthetic +11.62。
**⇒ imagereward 是 held-out 主轴。**

**Why 这条重要:** 本项目多次拿 aesthetic 支撑/反驳结论。
EXPERIMENT_LOG §4.595 把"`lr8x` vs canonical 的 aesthetic 显著更差(t=−14.8/−19.7)"当作加速线的
保留意见 —— 在 r=−0.395 下那是**弱一致于复合分更好**,那条保留意见被高估了。
凡靠 aesthetic 下的结论都要按本条重新加权。

**曾经的错解(已作废):** 我先以为"aesthetic 是欠训代理(欠训会升)",依据是 `lrhalf@ck50`(+9.00)
和 `baserep` 早停(+4.23)。但 `lr8x`(训练最充分、复合分 +17.95)的 aesthetic 也是 +3.28 **正的**。
真相不是"随欠训升",而是**几乎不携带复合分方向的信息**。

**How to apply:** `judge.py` 的每轴输出已自动打标(aesthetic ⚠️ / imagereward ✅)。
判 held-out 时只看 imagereward;要第二条 held-out 证据就另找轴,别用 aesthetic 凑。
另见 [[multik-multistate-distill]](它的 aesthetic +8.79 因此作废,但 imagereward +5.87 保留 ——
且两个欠训参照的 imagereward 都是**负**的,符号相反,所以那条增益不是欠训假象)、
[[train-metric-not-heldout-proxy]]。
