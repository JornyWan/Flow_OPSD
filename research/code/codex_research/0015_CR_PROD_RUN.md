# 0015 — GuardedOPSD 的**正式**配对实验(真实 per-slot 数据集)

日期: 2026-08-29 13:04 | 作业: `crdirect=1036359`(ai-gpu11) / `crguard=1036358`(ai-gpu09),各 2×A100

## 1. 为什么要重做

0010 的 200-epoch 跑误用了 `codex_cr_smoke()`,三个 task slot 全部指向
`testdata/pickscore`(256 条),OCR/GenEval 约束**从未在各自任务分布上被测量**
(codex 0014 复核指出,已独立核实)。该次实验的 GenEval 相关推论全部作废。

用户 2026-08-29 明确要求:**实验一律正式配置 + 完整数据;smoke 只用于验证代码正确性。**

## 2. 本次的正式配置 `codex_cr_prod`

继承 `mopd()`,只叠加 constrained-retention 开关,不改动任何数据/任务设置:

| slot | dataset | prompt_fn | proxy_reward |
|---|---|---|---|
| pickscore | `dataset/pickscore`(25432 条) | general_ocr | pickscore |
| ocr | `dataset/ocr` | general_ocr | ocr |
| geneval | `dataset/geneval` | **geneval** | geneval |

## 3. 三层防护(0010 的错误在每一层都会被拦住)

1. **配置内断言**:三个 slot 的 dataset 必须互不相同,否则 `get_config` 直接抛错;
2. **slurm 运行期不变量**:数据集互异、不含 `testdata`、slot[1] 以 `/ocr` 结尾、
   slot[2] 以 `/geneval` 结尾且 `prompt_fn=geneval`,并逐行打印实际路径;
3. **proxy_reward 断言反转**:原 smoke 不变量要求 proxy_reward 为空,
   正式口径改为**每个 slot 必须有与自己同名的 proxy reward** —— 同样能拦住 slot 错配。

日志实测输出:
```
[CODEX-0015] slot dataset: .../dataset/pickscore
[CODEX-0015] slot dataset: .../dataset/ocr
[CODEX-0015] slot dataset: .../dataset/geneval
[CODEX-0015] per-slot proxy_reward OK: pickscore->pickscore, ocr->ocr, geneval->geneval
[CODEX-0003] effective: bs=3 nproc=2 nbpe=12 gas=12 samples_per_epoch=72 epochs=200
[CODEX-CR] direct_slot=0 retention_slots=(1, 2) weights=(1.0,0.0) / (1.0,10.0)
```

## 4. 新增:逐 slot 诊断(`code/0015_per_slot_cr_logging.patch`)

0010 的日志把 slot 1(ocr) 与 slot 2(geneval) 平均到一起,无法回答
"哪个约束在激活、哪个主导梯度"(codex 0014 §2.3)。本补丁只**读**已有张量,
按 slot 分别记 `cr_cur_kl_* / cr_ref_kl_* / cr_violation_* / cr_active_* / cr_n_*`。

### 首个结果(epoch 1,`crguard` w=10)—— 旧日志无法给出的信息

| slot | cur_kl | ref_kl | violation | active |
|---|---|---|---|---|
| **ocr** | 1.110e−3 | 1.004e−3 | **+8.55e−5** | **0.537** |
| **geneval** | 1.379e−3 | 1.342e−3 | +9.83e−6 | **0.224** |

**OCR 约束在 54% 样本上激活,GenEval 仅 22%;OCR 违约幅度约为 GenEval 的 9 倍。**
⇒ **hinge 预算主要被 OCR 消耗,是它在主导梯度。**

(`n=1` 与配置自洽:`teacher_idx` 按 batch 轮转,bs=3、3 个 teacher,
故每 slot 平均每 micro-batch 摊到 1 个样本。)

## 5. 本实验仍**不是** 0006 的主方法

实现是 scalarized hinge penalty(`loss = w_d·L_direct + w_r·mean(relu(D−thr))`),
**不是** 0006 的 update-space QP projection(无 Gram 矩阵/active-set QP/certificate)。
本条目检验的是:**在正确的 per-slot 状态分布下,fixed-penalty 的行为与 0010 有何不同**。
0006 主方法仍待实现,codex 0014 §4 建议先做 gradient-geometry probe 判定可行方向是否存在。

## 6. 前 10 epoch:**修正 prompt 后,两个约束双双被守住** —— 0010 的推论确被证伪

`crdirect`(w=0,无约束):两个约束都持续被违反

| ep | OCR viol / active | GenEval viol / active |
|---|---|---|
| 3 | +4.26e−4 / 0.574 | +3.44e−4 / 0.528 |
| 5 | +4.03e−4 / 0.604 | +4.07e−4 / 0.484 |
| 9 | +1.75e−4 / 0.463 | +1.04e−5 / 0.169 |

`crguard`(w=10):两个约束都被拉回预算内

| ep | OCR viol / active | GenEval viol / active |
|---|---|---|
| 3 | **−4.53e−4** / 0.009 | **−5.69e−4** / 0.000 |
| 5 | **−4.74e−4** / 0.026 | **−7.08e−4** / 0.000 |
| 9 | **−4.51e−4** / 0.012 | **−6.20e−4** / 0.002 |

**关键**:GenEval 的违约甚至比 OCR 更负(−7.1e−4 vs −4.7e−4),
即**它被守得更紧**。而 0010 的结论是"GenEval 没被守住,说明 hinge 做不到逐 teacher 保持" ——
**该推论现已被直接证伪**:那是错误 prompt 分布造成的假象,不是方法性质。

两臂 epoch 0 逐位相同、epoch 1 几乎相同、epoch 2 分岔 —— 与 0010 相同的干净因果链。

### 与 0010 的对照(说明配置修正带来的实际差别)

| | 0010(错误 prompt) | 0015(正确 per-slot) |
|---|---|---|
| 能否分辨两个约束 | ✗ 只有合并平均 | ✓ 逐 slot |
| OCR 是否被守住 | 是(但状态分布错误) | **是** |
| GenEval 是否被守住 | 结论"否"(**已作废**) | **是,且更紧** |

**仍待观察**:约束守住两者的**代价**(pickscore/aes/hps)在正确配置下是否仍如 0010 那样巨大。
这要等 checkpoint 评测,不能从训练指标外推。

## 7. 与 0010 的直接对照(相同 step {4,8},只差 prompt 配置)——**代价几乎不变**

评测步集刻意取与 0010 相同的点,使"修正配置"成为唯一变量。
`guard(w=10) − direct(w=0)` 的差值,单位 ×2σ,两点均值:

| 轴 | 0010(错 prompt) | 0015(正确 prompt) | 差别 |
|---|---|---|---|
| **pick** | −9.1 | **−9.1** | +0.0 |
| **aes** | −15.2 | **−13.7** | +1.4 |
| **hps** | −11.6 | **−12.4** | −0.7 |
| ocr | +0.5 | +0.2 | −0.3 |
| geneval | −0.5 | −0.1 | +0.4 |
| clip | +1.0 | −0.1 | −1.1 |

step 8 绝对值(guard 臂):pick **0.8759 → 0.8755**、aes 5.3563 → 5.3419、hps **0.2742 → 0.2742**。

### 必须分开陈述的两件事

1. **codex 的批评成立且必要**:配置确实错了。基于它做的**GenEval 专属推论**
   ("GenEval 没被守住 ⇒ hinge 做不到逐 teacher 保持")**作废** ——
   本次正式跑的逐 slot 日志已直接证伪(GenEval violation −6.3e−4,比 OCR 守得更紧)。
2. **但"代价"这一核心发现在正确配置下依然成立**:fixed-penalty 对 pickscore/aes/hps
   的巨大损害**与 prompt 配置无关**,两种配置下量级一致。

### n=4 复核(steps {4,8,12,16},与 0010 同口径,自相关校正)

| 轴 | 0010(错 prompt) ×2σ / t_eff | 0015(正确 prompt) ×2σ / t_eff |
|---|---|---|
| **pick** | −14.2 / **−2.99✱** | **−13.9 / −2.93✱** |
| **aes** | −21.4 / **−3.15✱** | **−19.9 / −2.96✱** |
| **hps** | −17.7 / **−3.09✱** | **−17.5 / −2.93✱** |
| ocr | +1.5 / 1.78 | +1.0 / 1.45 |
| geneval | −0.5 / −2.31 | −0.3 / −1.55 |
| clip | +1.4 / **3.02✱** | +0.9 / 1.03 |

**三条代价轴在两种配置下都显著,且量级几乎相同**(pick −14.2 vs −13.9,
aes −21.4 vs −19.9,hps −17.7 vs −17.5)。

⇒ **代价是 scalarized hinge 这个机制本身的性质,与状态分布是否正确无关。**

这与 codex 0014 的判断一致:该继续检验的是 **0006 的 QP projection**,
而不是在 fixed penalty 上继续调权重或换 prompt。

### n=5 复核(加入 ep25)—— 结论不变

| 轴 | 0010(错 prompt) | 0015(正确 prompt) |
|---|---|---|
| **pick** | −15.1 / −3.99✱ | **−14.8 / −3.91✱** |
| **aes** | −21.7 / −4.11✱ | **−20.7 / −3.93✱** |
| **hps** | −17.3 / −3.87✱ | **−17.3 / −3.73✱** |
| ocr | +1.9 / 1.92 | +1.3 / 1.72 |
| geneval | −0.4 / −1.66 | −0.3 / −1.95 |
| clip | +1.8 / 3.31✱ | +1.1 / 1.27 |

### 正确配置下 OCR 的逐点轨迹 —— 约束确实在守住能力

| step | direct(w=0) | guard(w=10) | diff |
|---|---|---|---|
| 4 | 0.9474 | 0.9340 | −0.0134 |
| 8 | 0.9208 | 0.9475 | +0.0267 |
| 12 | 0.8923 | 0.9543 | +0.0620 |
| 16 | 0.8699 | 0.9493 | +0.0795 |
| 25 | 0.8546 | 0.9550 | **+0.1004** |

**无约束臂的 OCR 单调退化(0.9474 → 0.8546),约束臂稳定在 ~0.95,差距单调拉大。**
这是目前最清楚的"约束按设计工作"的证据 —— 且是在**正确的 prompt 分布**上,
不再有 0010 那个状态分布错配的问题。

### 本条目当前的完整判读

1. **约束有效**:OCR 被切实守住(diff 单调增至 +0.10),GenEval 亦被守住
   (逐 slot violation −6.3e−4,比 OCR 更紧);
2. **代价真实且与配置无关**:pick −14.8、aes −20.7、hps −17.3(×2σ),
   与错误配置下的量级几乎相同;
3. ⇒ **fixed-penalty 判负成立,且现在建立在正确配置上**。
   但这**仍不是** 0006 的 QP projection —— 该方法尚未被检验。

**仍待补**:更晚的点(0010 的判负基于 n=9/10)。代价轴在 n=4/n=5 上均稳定,不预期翻转。

## 8. 状态

训练中(200 epoch)。判读时:两臂同 step 配对、自相关校正、
并**分别**看 OCR 与 GenEval 的 `cr_active_*` 以确定各自约束是否真的在起作用。
