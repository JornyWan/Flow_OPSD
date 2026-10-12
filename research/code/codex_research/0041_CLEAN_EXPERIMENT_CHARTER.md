# 0041 实验口径章程:把"保持一致"变成会报错的机制(2026-08-30)

用户要求:"方法基于 baseline 的实现,baseline 已确定复现成功,同一套数据/base 模型/基础代码设施,
每次换 idea 都要保持干净。" 本文把它落成**可执行的检查**,而不是承诺。

## ① 先报一个本该早发现的问题:参考仓库被我们改过

`CA-OPD/reference_papers/DiffusionOPSD-main/` **不是原版**:

| 文件 | 改动时间 |
|---|---|
| `config/opsd_defaults.py` | 2026-08-28 21:44 (+22 行) |
| `scripts/train_opsd_ri_sd3.py` | 2026-08-28 21:48 (**+201 行**) |

加进去的是我们自己的消融扩展:`opa_topk`(RAFT 式 best-of-N)、`opa_ls`(沿上升射线线搜索)、
`opa_mq_sigmas`(多查询点)、`OPSD_WF_MODE=frozen`(冻结归一化器)。

**更要命的是时间线:**`pickscore` 复现跑是 **08-27 05:28(改动之前)**,
而 open3/clipscore/hpsv2 是 **08-30(改动之后)**。
即**我们唯一验证过的那一列,与现在跑的臂用的不是同一版代码。**

**这个问题是 `opsd_manifest_diff.py` 第一次运行就抓出来的** ——
pickscore 的 manifest 缺 6 个 `opsd.*` 字段。靠人记是记不住的。

## ② 已完成的验证:默认值下与官方原版**行为完全一致**

从 GitHub 拉取原版并逐块比对(`git clone https://github.com/worldbench/DiffusionOPSD`,
commit **`9c4c48cdfe5c3fa712d32957345d75fae9295696`**):

- 仅 3 个文件不同:`config/opsd_defaults.py`、`scripts/train_opsd_ri_sd3.py`、`README.md`;
- 训练脚本 12 个 hunk,逐个核对:

| hunk 类型 | 数量 | 结论 |
|---|---|---|
| 新增代码块,首行即 `if <开关> > 0` / `if <开关>:` 守卫 | 7 | 开关默认关 ⇒ 惰性 |
| 等长修改 | 3 | ①加默认 `n_ls=0` 的可选参数;②`if ri_opa_cert:` → `or (ri_opa_ls>0 and ...)`(ls=0 时恒假);③wandb.log 多一个诊断字段 |
| 模块级常量 | 1 | 两个 `None` 全局,惰性 |
| env 守卫块 | 1 | `if os.environ.get("OPSD_WF_MODE","adaptive")=="frozen"`,默认跳过 |

**⇒ 所有开关取默认、`OPSD_WF_MODE` 未设时,与官方 `9c4c48c` 行为完全一致。已验证,非假设。**

正在跑的三个臂全部满足:manifest 显示 `opa_ls=0 / opa_topk=0 / opa_mq_sigmas="" / dir_mode=grad`,
日志中 `[WF-FROZEN]` 出现 **0 次**,`opsd_repro.slurm` 未设该 env,登录 shell 无残留。
**⇒ open3 / clipscore / hpsv2 三个臂是 canonical 的。**

## ③ 已落地的三项机制

1. **原版固化**:`reference_papers/DiffusionOPSD-pristine-9c4c48c/`,
   `config/` `scripts/` `*.md` 置为只读,`PINNED_COMMIT.txt` 记录 SHA。
   今后任何"官方 baseline"声明都以它为准,不以被改过的那份为准。
2. **manifest 差异检查器** `opsd_manifest_diff.py`:
   ```
   opsd_manifest_diff.py RUN_A RUN_B --allow <本次有意变动的字段>
   ```
   任何**未声明**的口径差异 ⇒ 打印并非零退出。IGNORE 只放
   `run_name/run_random_nonce/wall_clock_start/git_commit` 四个必然不同的字段。
3. **env 中和**:`opsd_repro.slurm` 显式 `export OPSD_WF_MODE=adaptive`。
   理由:该变量**会改变目标函数,却不进 run_config.json** —— 只看 manifest 发现不了,
   必须在启动侧堵死。

## ④ 种子政策:复现跑与 A/B 跑必须分开,且不可混用

官方代码支持 `config.seed`(非 None 即 `set_seed`),但**默认 None**,
`config/base.py` 注释写明"DiffusionOPSD policy: no fixed seed"。

| 用途 | 种子 | 理由 |
|---|---|---|
| **复现跑**(pickscore/clipscore/hpsv2/open3) | **不固定**(现状) | 与论文政策一致,才能与论文数字对照 |
| **A/B 对照**(baseline vs A1 vs A2) | **同一固定种子,配对** | 否则臂间差异里混着 RNG 抽样,归因不干净 |

**⚠️ 两类跑不可混进同一张表。** 复现跑证明"实现忠实",A/B 跑证明"改动有效",是两个问题。
A/B 完成后再补 2~3 个独立种子给置信区间(用户已同意暂缓 seed,此处只是先写定政策)。

## ⑤ 冻结的公共口径(A/B 全臂必须逐位相同)

```text
代码        DiffusionOPSD @ 9c4c48c(默认开关;我们的改动仅以显式 flag 开启)
base 模型   stabilityai/stable-diffusion-3.5-medium
LoRA        r=32, alpha=64, init="gaussian"
训练数据    pickapic 25415 条(prepare_pickapic_prompts.py,SHA-256 与论文 recipe 一致)
分辨率      512²
rollout     CFG-free, 10 步, dpm2, deterministic
预算        300 optimizer updates(Open3 口径), 1152 samples/epoch
beta        0.1(Open3)
world_size  必须相同(manifest 已记录;3 卡与 4 卡梯度等价但仍须声明)
评测        DrawBench 1000, flow 40 步, gs=1.0, 512², seed 42, pickscore_scale=raw
            scorer 集: hpsv2,clipscore,aesthetic,pickscore,imagereward
种子        A/B 全臂同一固定值
```

**每次提交 A/B 臂之前,先跑 `opsd_manifest_diff.py` 并把干预字段写进 `--allow`;
未声明的差异一律视为实验无效,不是"小问题"。**

## ⑥ 尚未解决的一项

`run_config.json` 的 `git_commit` 字段恒为 `"unknown"`(官方代码没实现)。
目前靠 `PINNED_COMMIT.txt` + 本文记录来补。
若后续在 fork 上开发,应让 fork 把自己的 commit 写进 manifest,否则代码版本仍不可追溯。

## ⑦ 一致性纪律当天就抓到的第二个 bug(2026-08-30 15:10,结果落地前)

`compare_opsd_repro.py` 靠 eval JSON 里的 `train_reward` 判"**哪一列可比**"
(`cmp_ok = "✅可比" if k == train_rw`)。而该字段来自 `cross_eval.py:349`:

```python
train_reward = ",".join(config.reward_fn.keys())     # 来自 --config
```

**`opsd_eval.slurm` 把 `--config sd35_pickscore` 写死了。**

后果:
- 已产出的 `base` 与 `Student` 两份 JSON 被错标成 `train_reward='pickscore'` ——
  base 根本没训练、Student 是多教师蒸馏,这个标签无意义(**仅元数据错,分数不受影响**);
- 更严重的是**今晚**:clipscore 与 hpsv2 臂跑完后监视进程会自动评测,
  同样会被标成 `pickscore` ⇒ 对表时把 **Pick 那一列**标成"可比"、
  而真正该比的 **CLIP 0.340 / HPS 0.390** 被标成"不可比"。**基线确认会被判错。**

**修法(改标签,不动评测):**
先核对四个 config 的**评测相关**字段:

| config | resolution | gs | noise | eval_steps | model | dataset |
|---|---|---|---|---|---|---|
| sd35_pickscore / clipscore / hpsv2 / open3 | 512 | 1.0 | 0.7 | 40 | SD3.5-M | pickapic |

**四者逐项相同** ⇒ 换 `--config` 只改 `train_reward` 元数据,评测逐位不变。
于是 `opsd_eval.slurm` 加 `EVALCFG`(默认 `sd35_pickscore` 保持向后兼容),
两个监视脚本按 `$ARM` 选 `sd35_$ARM`,base/Student 这类无训练 reward 的回落到默认。

**⚠️ 改监视脚本前先 kill 了正在运行的三个实例。**
编辑运行中的 bash 脚本会让解释器读到错乱内容(本项目栽过一次 `unexpected EOF`);
改完再重启,监视只是 sleep 轮询,重启无损失。

**这条的意义:** bug 在**结果落地之前**被抓到,不是事后发现数据判错。
`train_reward` 这种"看起来只是标签"的字段,恰恰是判据的输入 —— 它错了,结论就错。

## ⑧ 种子政策的计划后果:正在跑的 Open3 **不能**直接当 A/B baseline

§④ 定了:复现跑不固定种子(与论文政策一致),A/B 对照臂同一固定种子配对。
**这意味着 1037080(正在跑的 open3)有两个用途,但都不是 A/B baseline:**

| 用途 | 说明 |
|---|---|
| ✅ 复现/表征论文的 joint 目标 | 不固定种子 = 论文政策,可与论文对话 |
| ✅ 给中期 Gate A 探针提供 checkpoint | ck100/ck200 可 resume 做中期 VFR(见 0040 §⑨ 边界 2) |
| ❌ **A/B 的 baseline** | 它是**未播种**的;拿它与播种的 A1 比,差异里混着 RNG |

**⇒ A/B 实验需要自己的 baseline 跑(播种),与 A1/A2 用同一个种子。**

### 代价估算(按实测节奏 3.76 min/epoch,3×H200)

| 臂 | 预算 | 墙钟 | GPU-小时 |
|---|---|---|---|
| Open3-baseline(seed=S) | 300 upd | ~19h | ~57 |
| A1(seed=S) | 300 upd | ~19h | ~57 |
| A2(seed=S) | 300 upd | ~19h | ~57 |

**最小可判方案 = 2 臂**(baseline + A1,同种子配对) ≈ 114 GPU-小时 ≈ 38h 墙钟(串行 3 卡)。
A1 赢了再加 A2;两者都过再补 2~3 个独立种子给置信区间。

**⚠️ 不要为了省算力去拿未播种的 1037080 当 baseline。**
那正是"每次换 idea 都要保持干净"要防的事:差异里会同时含有干预和 RNG,
而 A1 的效应量若与 RNG 波动同量级,就完全无法归因。

### 一个可以省一半的替代方案(待用户裁决)

若算力紧张,可把 A/B 预算从 300 降到 **100 updates**(与三个 reward-specific 专家同预算):
- 优点:每臂 ~6.3h,两臂 ~13h,当天可出结论;
- 代价:偏离论文 Open3 的 300-update 设定,**不能与论文 Open3 曲线并列**,
  只能作内部 A/B(但内部 A/B 本来就是自比,这个损失可接受);
- **前提**:baseline 与 A1 必须同为 100,不可一个 300 一个 100。

**本文只列方案与代价,未提交任何 A/B 任务。**
