---
name: opsd-mirror-degeneracy-bug
description: 我的 Q-OPSD 一步镜像构造让 pos_loss ≡ neg_loss,advantage 权重 r1 对梯度贡献严格为零;查法是看两个 loss 的**差**而非量级
metadata:
  type: feedback
---

**2026-09-02,codex 审计 0069 §5 发现,我已独立验证成立。**

DiffusionOPSD 的策略侧本来就是镜像的:
```
v_pos = v_old + β·δv      v_neg = v_old − β·δv      (δv = v_θ − v_old)
⇒ y_pos − y0 = −(y_neg − y0)
```
我的 Q-OPSD 一步构造又把**目标**也做成严格镜像:`y± = y0 ± α·budget·unit(d)`。两个镜像叠加后:
```
y_pos − y_plus = −(y_neg − y_minus)   逐元素成立
⇒ 平方项相同、|·| 归一化分母相同、且 acc_p ≡ acc_m ≡ 1
⇒ pos_loss ≡ neg_loss
⇒ r1·pos_loss + (1−r1)·neg_loss ≡ pos_loss
```
**group advantage `r1` 对梯度的贡献严格为零。** 方法退化成"对所有采到的 state 做无条件
reward-ascent 方向回归",已经不是 OPSD 的 self-distillation。

实测证据(`metrics.jsonl`,`mean|L_pos−L_neg|`):

| arm | mean\|差\| | 相对 L_pos≈0.2 |
|---|---|---|
| baseline_np3 | 1.44e−3 | 7e−3 |
| qopsd | **3.50e−7** | **1.75e−6**(fp32 舍入量级) |
| qbt | 2.48e−7 | 1.4e−6 |

baseline 不退化,因为它是**两步**构造:第二步分别在 `y0±h·g` 处**重算**梯度,
reward 的曲率打破了严格镜像。

**Why 我没早发现:** 我在 §4.137 只比较了 `opa_pos_loss` 与 `opa_neg_loss` 的**量级**,
看到"两者接近"就当作正常(它们本来就该接近)。**没查过差值是否退化为零。**

**How to apply:**
1. **凡是引入对称/镜像结构的改动,先问:下游有没有依赖不对称性的加权项?**
   加权项遇到对称输入会静默失效,不报错、不发散、指标照常下降。
2. 诊断要看**差**,不看量级:`mean/max |L_pos − L_neg|` 必须显著大于数值误差。
3. 更强的检查:**把 `r1` 随机打乱,总梯度必须改变**。不变就说明 advantage 没进训练。
4. 由此,"qbt vs baseline 的差 ⇒ 回溯有害"这个归因**不成立**:
   qbt 相对 baseline 同时变了三件事(两步→一步、固定步长→回溯、非镜像→镜像),
   一步镜像才是共因嫌疑最大的那个。**多因子同时变 = 不能归因单因子**。

相关: [[train-metric-not-heldout-proxy]] [[print-the-series-first]] [[opsd-flag-chain-completeness]]
