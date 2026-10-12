---
name: opsd-flag-chain-completeness
description: 给 DiffusionOPSD fork 加/放宽 env 开关时,必须追完整条"生产→消费"链;只改用到的那一处已栽三次
metadata:
  type: feedback
---

**在 `train_opsd_ri_sd3.py` 里新增或放宽一个 env 开关时,被该开关门控的往往不止一处。
只改"报错的那一处"或"我用到的那一处",一定会在运行时崩 —— 已栽三次,每次都烧掉 GPU 时间。**

| # | 开关 | 只改了 | 漏掉的 | 代价 |
|---|---|---|---|---|
| 1 | `RPI_A1` | `_GATEA_LAST` 的**读取**条件 | **写入**条件(`_reward_scores_grad` 内) | job 1037521 跑 2:39 崩 |
| 2 | `RPI_GATEP_PROBE` | 读 `_GATEA_LAST` | 它在 `y_plus` 之后已被上升途中的打分**覆盖** | launch 前抓到;否则是**静默错数据** |
| 3 | `RPI_CERT_PROBE` | `prev_x0` 的**计算**条件(L1378) | `sample_entry["opa_rollout_z_prev"]` 的**生产**条件(L1170) | job 1038153 跑 2:20 崩 |

**Why:** 变量的"定义行号早于使用行"**不等于**它有值。第 3 次我确实逐个核对了
`prev_x0/xi/opa_margin/...` 的定义行号都早于探针行,但 `prev_x0` 依赖一个
**在采样阶段、由另一个守卫决定是否写入**的 dict 键 —— 行号检查看不出这种跨阶段依赖。

| 4 | `RPI_Q2A_PROBE` | 探针块条件 | `prev_x0` 的守卫(生产端与消费端) | job 1038453 起跑即 `UnboundLocalError` |

**⇒ 栽到第 4 次后已做结构性修复(2026-09-01),不要再手写 `A or B or C`:**

```python
# train_opsd_ri_sd3.py 顶部,**单一真相源**
_NEEDS_SUFFIX      = _CERTP_ON or _GATEP_ON or _QGRAD_ON or _Q2A_ON   # 需 rollout 后缀机制
_NEEDS_AXIS_SCORES = _GATEA_ON or _A1_ON or _GATEP_ON or _QGRAD_ON or _Q2A_ON  # 需逐轴 reward 分量
```

全部守卫改用这两个派生标志(`if ri_opa_cert or _NEEDS_SUFFIX:` 等)。
**加新探针开关时只改这两行**,不必再逐处追链 —— 这才是根治,前三次都只是补单点。

**How to apply(改开关前的固定动作):**
1. `grep -n '<被门控的原开关>' file` **列出全部出现**,逐个判断新开关是否也需要
   (第 3 次这么做才发现除生产端外还有一条 `opa_state_mode='forward'` 的兼容性校验也该覆盖);
2. 对每个跨阶段传递的量(dict 键、模块级全局、缓存),**分别确认"谁写"和"谁读"**,
   两端的条件必须一致;
3. 模块级可变全局(如 `_GATEA_LAST`)还要确认**读取时机** —— 中间有没有别的调用把它覆盖;
4. 语法检查(`ast.parse`)与"变量定义行号"都**发现不了**这类问题,别拿它们当通过标准。

相关: [[no-smoke-configs]] [[opd-caopd-project]]

**核验旋钮是否真的生效,去训练日志的 config dump 找,不是 run_config.json**(2026-09-08 实测):
`outputs/ab_open3_*/run_config.json` **只记** `sample.*` / `reward_fn.*` / `opsd.*`,
**不记** `train.learning_rate` / `train.num_inner_epochs` —— 拿它核 `train.*` 旋钮会一无所获。
正确位置是训练 stdout:解析后的 config dump(`  learning_rate: 0.0012` / `  num_inner_epochs: 2`)
以及 `train_opsd_ri_sd3.py:995/996] Number of inner epochs = N`。
后者是**被消费**的证据(代码从解析后的 config 打印),强于 `[ab] extracfg=` 那行的回显
—— 回显只证明 flag 被传进去,§4.324 那次正是"传了但被 `EXTRA_CFG=""` 覆盖"。

**2026-09-29(第 5 次,代价:hybrid 白跑 3 分钟 + 一个 4×A100 时段)**:
加 `RPI_SELECT_M` 时我确实按本条去追了"生产→消费"两端,但**漏了一层**:
这条链的两端条件并不是各自手写的,而是都挂在一个**派生标志**上 ——
```python
_NEEDS_SUFFIX = _CERTP_ON or _GATEP_ON or _QGRAD_ON or _Q2A_ON or _QOPSD_ON or _QCORE_ON
```
生产端 `sample_entry["opa_rollout_z_prev"]`(L1392)与消费端 `prev_x0`(L1631)都写
`if ri_opa_cert or _NEEDS_SUFFIX:`。我没把 `_SEL_M` 加进那一行 ⇒ 两端**同时**不生效 ⇒
`UnboundLocalError: prev_x0`。而那一行上面的注释**就写着**"加新开关时只改这一行、
本项目已因只改一端栽过 4 次" —— 注释在我正在编辑的同一个文件里。

**⇒ 追链时先找"这条链有没有派生标志/单一真相源"。** 本仓库有两条:
`_NEEDS_SUFFIX`(需要 rollout 后缀 / prev_x0 / opa_rollout_z_prev)与
`_NEEDS_AXIS_SCORES`(需要逐轴 reward 分量 / `_GATEA_LAST`)。
新开关若用到这两类资源,**必须加进对应那一行**,而不是在两端各写条件。
注意 `_SEL_M` 原来定义在 L~416、在 `_NEEDS_SUFFIX`(L165)之后 ⇒ 要把读取上移,否则 NameError。

同一轮还查出一个语义 bug:单状态选择分支只判 `if _SEL_M > 0:`,没判混合门
⇒ 混合臂主状态(σ=0.278,门 0.465 之下)本该走梯度却进了选择分支。
**教训:新开关既要追"资源链",也要追"哪些已有分支的条件需要收紧"。**
起训前用 code-correctness smoke(独立 OUTPUT_DIR、1 卡、UPDATES=1)验一遍,
比直接烧 35h 便宜三个数量级 —— 这类 UnboundLocalError 只有运行时才暴露。

**2026-09-30 补(第 6 次:改一个旋钮**隐式**改了另一个)**:
`base4` 臂只加了 `--config.sample.num_steps=4`,结果 `opa_query_sigma=0.278` 的 **argmin 选态**
落到了 4 步 σ 表的退化末态 **0.00893**(表 = 1.000, 0.858, 0.602, 0.00893, 0;
|0.602−0.278|=0.324 > |0.00893−0.278|=0.269),目标构造退化,reward 从 ep15 的 1.2342 跌到 ep29 的 0.8845。
`opa_delta_res` 与 canonical 一致(0.0713)⇒ 代码没问题,是**查询态选错**。
**⇒ 改 `sample.num_steps` 的臂,起跑后必须核 `opa_kq` 与 `opa_sigma_q`,不能只核被改的那个旋钮。**
修法:显式钉 `--config.opsd.opa_query_sigma=<该步数表的倒数第二个非退化态>`
(4 步=0.60215;6 步的 argmin 落在 0.43223,**无需修**)。
真 σ 表可用 `FlowMatchEulerDiscreteScheduler.from_pretrained(<model>, subfolder="scheduler")`
+ `set_timesteps(num_inference_steps=n)` 复现(shift=3.0,SD3 走 `sigmas=None` 的默认表)。
