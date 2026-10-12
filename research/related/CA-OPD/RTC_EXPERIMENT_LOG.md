# RTC 主线实验清单（2026-08-26 夜）

所有诊断的 job id、配置、判据与结论。便于论文写作时回溯与复现。

## 一、G_construct 探针（核心正向结果）

| job | 配置 | n | 结论 |
|---|---|---|---|
| 1034406 | `sopd_gcprobe` | 3 | grad +0.0946 / teacher +0.0162 / rand −0.0032 |
| **1034410** | `sopd_gcprobe` | **24** | **grad +0.09275 / teacher +0.01430 / rand −0.00411** |

配对检验（n=24）：grad−teacher t=106.7，teacher−rand t=36.1，grad−rand t=85.5。
**grad/teacher = 6.48×，teacher 只吃到 15.4%。**
σ*=0.278（sigmas[8]），固定后缀 F_q，ρ=0.10 严格半径匹配。单 seed，论文需扩多 seed。

## 二、RTC 原始形式（有限差分估 JJ^T g）—— 已撤回

| job | 发现 |
|---|---|
| 1034405/1034407 | cos(JJ^T g,g)=0.0153/0.0162；`eps_stab=0.34`（应≈1）⇒ 噪声主导 |
| 1034409 | 关 TF32 后 `max\|v1−v0\|` **恰好为 0** |
| 1034411 | 精确双反向 JVP 失败：flash-attn 无二阶导 |
| 1034412 | 拷贝落地=2.28（正确）、1000× 扰动 dv=5.45（参数在前向路径）、1× 扰动 dv=0 |
| 1034413 | eps 扫描 1×~1000×：`\|dir\|∝1/scale`，相邻 cos 仅 0.2~0.65，**无线性区** |
| 1034414 | 前向**完全确定**（max\|vA−vB\|=0，无 dropout）⇒ 排除随机性 |

**同一次运行内 scale=1 报 dv=1.09 而 FD 报 0 —— 真实矛盾，真因未查明，如实记录。**
已停止追查（该路径被 RTC-K 替换）。撤回"65× 提升"说法。

## 三、RTC-K（可实现目标）

| job | 配置 | 结论 |
|---|---|---|
| 1034415 | `rtc_k` K=4 | **acc_p=0.3056（失败）** |
| 1034416 | `rtc_kcurve` | 上升第 2 步达峰 +0.0237，第 4 步 −0.0038（跌回起点下）；reward-位移曲线 |
| 1034418 | `rtc_kbest` K=6 | **acc_p=1.0000**，kbest_gain=+0.031，kbest_idx≈2.1 |
| 1034422 | `rtc_kb_train` | acc_p=1.0000，kbest_gain=+0.033，loss/grad_norm 非零，_step 递增 |
| 1034426 | `rtc_raw_train` | acc_p=0.97~1.00，loss 0.31~0.34，grad_norm 0.0005 |
| 1034429 | `rtc_kcurve` n=12 | 运行中：曲线加误差棒 |

**半径≈3% 处**：grad +0.0579 / K步实际点 +0.0237（41%）/ 指向终点的直线 +0.0071（12%）
⇒ **上升路径是弯的，线性化丢掉 3.3×**，反对线性化预条件。

**两臂 smoke 对比**：raw 的 loss 大 4× 但 grad_norm 小 7× ⇒ 残差落在参数改变不了的方向上。

## 四、RTP（reward 梯度投影到 teacher 子空间）

| job | 结论 |
|---|---|
| 1034417 | **无效**：`gc_ntproj=1`，子空间退化 |
| 1034423 | **无效**：计算节点读到旧 config（`K_max=1`） |
| 1034424 | `K_max=3` 确认，门控有效，运行中 |

根因：`train_sd3_opd.py:726` 的"每 slot 算全部 teacher"只在 `cagrad`/`ste` 时执行。
已加 `rtp` / `gc_all_teachers`。
**定位修正**：baseline 无"等权求和"，RTP 的对手是 conflict-averse 组合（已证明有害）。

## 五、修掉的 bug（都不会让程序崩溃）

1. 固定 K 越过极值 → 按实测 reward 逐样本取最优迭代点
2. 信赖域被当强制值 → `budget = min(budget, |dir|)`
3. **`continue` 绕过 `sync_gradients`** → `global_step` 不前进 ⇒
   checkpoint 全覆盖到 `checkpoint-0`、EMA 节奏错乱、**diag2 样本量塌成 n=1**
4. 预检漏查 config 内容 → 新增 `RUN_REQUIRE_CFG_MARKER`
5. auto_eval 步号列表未覆盖 RTC 的 `global_step` 偏移（0,30,60 vs 1,31,61）

## 六、不可信、拒绝下结论的指标

`rtc_realized ≈ 0`：`dy_norm/|y0| = 5e-4` 低于 16 位前向的 4e-3 分辨率，
与毁掉 FD 的是同一陷阱。已在训练配置关闭。要测 G_realized 须跨多步窗口。
