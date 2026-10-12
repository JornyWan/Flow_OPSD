# 0037 "我们的 baseline 是自己训的"——问题确认、代价核算与两条路线(2026-08-30)

用户问:"我们验证他们不成功的 baseline 也是我们自己训的吗?然后在其基础上 addon 了我们的
东西然后发现并没有超过?" **基本属实,且比这更具体。** 本文把事实钉死,并给出可选路线与代价。

## ① 事实:`crd` 不是官方 OPSD

`codex_cr_prod`(我们的 guard,`crg`)的定义是 `rtc=True, rtc_precond="raw", constrained_retention=True`;
`crd` 就是同配置去掉 retention。注释原文:**"GuardedOPSD = RTC-raw 直推 reward + retention 约束"**。
即 **guard 叠在我们自己写的 RTC-raw 之上,再和同一个 RTC-raw 比。这条链里没有官方 OPSD 代码。**

08-27 已逐项核对过(`PLAN_2026-08-27.md` §一),今日又从官方运行时日志独立复核了右列:

| 项 | 我们 `rtc_raw` | 官方 OPSD | |
|---|---|---|---|
| query_sigma | 0.278 | 0.278 | ✅ |
| rho | 0.10 | 0.10 | ✅ |
| n_ascent | 2 | 2 | ✅ |
| dual | True | True | ✅ |
| margin | **0.0** | **0.005** | ❌ |
| 缩放 | beta=0.5 | eta=1.0 | ❌ 不同参数化,对应关系**未验证** |
| opa_mb | **无** | 6 | ❌ 缺失 |
| samples/epoch | 72 | 1152 | 训练设置差 16× |
| guidance_scale | 4.5 | 1.0 | 差 |

当时的结论仍然正确:**`rtc_raw` = "OPSD 的构造思路"放进"DiffusionOPD 的训练设置"。
作内部受控对比有效;当作"OPSD 本身"下判断不成立。**

## ② 但不是全部悬空:两个真外部锚点

1. **发布版 DiffusionOPD Student LoRA** —— 官方固定权重,非我们所训,八列全部在 ±1×2σ 内
   (三重验证)。0022 全景排名里 ratio=1.0000 的参照就是它。
   ⇒ **"没有 idea 全轴超过 Student"这句话的参照物是可靠的。**
2. **`opsd_repro_sd35_*` 用的是官方 DiffusionOPSD 代码**,且今日 base 锚点五列对上论文
   (Aes −0.10%,见 0033 §⑫)。这条线干净,但**起点是 base、口径是 gs=1.0**,与我们的续训线不同。

## ③ 分级影响

| 结论 | 是否受影响 |
|---|---|
| "没有 idea 全轴超过官方 Student" | **不受影响**(参照是外部官方权重) |
| "guard 六轴严格支配朴素 KL" | **不受影响**(两臂同代码、同预算、同起点,干净的内部受控对比) |
| **"guard 把损害从 −10.9% 压到 −2.0%"** | **受影响** —— 那个 −10.9% 是我们重写版的损害,不是官方 OPSD 的 |

## ④ 路线 A:把 guard 移植进官方 OPSD —— 代价比预期高

今日读码确认的可行性:

- ✅ **起点可换**:`config.train.lora_path` 走 `PeftModel.from_pretrained`(L750),
  且 L1002-1005 在训练前**显式全量拷贝 `old ← default`**,
  ⇒ 设 lora_path=Student 即可让 policy 与 behavior policy 都从 Student 起,**无需改代码**。
- ✅ **插入点单一**:`loss = opa_policy`(L1775),加项即可。
- ✅ **双 adapter 前向机制现成**(`set_adapter("old")/("default")`,见 L1745-1760 的 mq 块)。
- ❌ **核心障碍:官方 OPSD 里没有 teacher。**
  我们的 retention 是 `relu(cur_kl − (1+rel_tol)·ref_kl − abs_tol)`,
  `cur_kl` = 当前 student 对 **OCR/GenEval teacher** 的 KL,`ref_kl` = 冻结初始 Student 对同一 teacher 的 KL。
  官方是纯 reward,没有 teacher;去掉 teacher 后 **`ref_kl ≡ 0`,hinge 退化成阈值化 KL**,
  那已经接近 `crkl5`,而我们实测 **hinge 六轴严格支配 crkl5** ⇒ **退化版会丢掉方法的核心结构**。
  忠实移植必须把 task teacher 机制一并搬过去,那是大改。

**代价:大。收益:让 §③ 第三条结论成立。**

## ⑤ 路线 B:把 `rtc_raw` 校准到官方目标函数 —— 便宜得多

只改三处已知偏离:`margin 0.0 → 0.005`、核对 `beta` 与 `opa_eta` 的对应关系、补 `opa_mb=6`。
训练设置差异(72 vs 1152 samples/epoch、gs 4.5 vs 1.0)**不改** ——
那是 DiffusionOPD 的续训设置,是本项目研究问题的正确设置,不是缺陷。

**代价:小。收益:让 `crd` 成为"忠实的 OPSD 目标函数放在续训设置里",
归因从"我们的近似 vs 我们的近似+addon"变成"OPSD 目标 vs OPSD 目标+addon"。**

⚠️ 但要诚实:即使做完 B,**也不能说"复现了 OPSD"** —— 起点、数据量、gs 全不同。
能说的是"OPSD 的目标函数在同一续训设置下的对照臂"。

## ⑥ 建议

**先做 B,不做 A。** 理由:
1. B 直接消除"目标函数不忠实"这一条,而这正是审稿人会问的;
2. A 的收益依赖把 teacher 搬进官方仓库,工作量与风险都大,且**做完之后的实验起点仍是 Student**,
   与官方 OPSD 的 from-base 主表依然不可比 —— 也就是说 A 并不能换来"与论文数字对齐";
3. 真正与论文对齐的那条线(`opsd_repro_*`,官方代码 + from base)**已经在跑**,
   今晚会拿到 Open3 与两个教师臂,那才是与论文对话的接口。

**⚠️ 本文只做判断与代价核算,未修改任何代码,未提交任何任务。** 等用户裁决再动。
