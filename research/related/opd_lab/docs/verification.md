# 本次搭建的验证记录

日期：2026-09-10。本文下面的“初建记录”保留首次 CPU 验证时的状态；后续真实实验见 [campaign 状态](../experiments/local_expert_v1/STATUS.md)。

## 2026-09-11 第二阶段

- 43 项测试通过；评测逐约束统计的 8 项相关测试另通过。均衡查询清单、固定动作验证、关闭约束保护、真实续训的契约已覆盖。
- 四个 SD3.5-M 训练臂各完成 192 查询、12 次更新；共同清单/首批候选相同，操作次数匹配。3,840 张候选图片的 hash 与评分有效性通过。
- 最终 raw checkpoint 的 flow10 / flow40 / DPM2-40 统一验证完成，共 960 张图。模型身份、冻结源码/协议、逐图统计重算及图片 hash 通过，test 未使用。
- 第二 OCR 的 CPU 分离执行完成，主评分/图片未变化；预声明 80 张验证图的身份隐藏实施者复核完成。第二 OCR 漏读严重，不能当真值。
- **本轮没有显示稳定增益**；全部 vs-base 效用比较的 95% prompt 区间包含 0。选择目标的局部收益不能代表训练后完整策略收益。
- 含三个生成前失败作业，共 3.353889 L40S GPU-hours。失败保留，完整结论、区间、图、源记录见 [phase 2](../experiments/local_expert_v2/STATUS.md)。

## 第一阶段后续验证（历史）

- 当前 40 项测试通过，包括环境漂移、缓存负标记与实际权重变动的区分、scheduler 元数据规范化、固定查询证据、两节点 rank/总 world size 解析。
- 完成 H100 CUDA/MMCV 检查、5 个实际 OCR 已知答案测试、H200 上 12 查询/60 图冻结诊断；noop 跨位置一致。
- 原版 MOPD 和 OPSD 都完成单卡两轮训练与 checkpoint 保存；本地 round_robin/verified 各完成两轮训练和真实 checkpoint 续训。
- 统一评测已完成：4 个 validation prompts × base/两方法 raw+EMA × flow/DPM2 10 步，共 40 张图，严格协议比较通过；两次更新的模型均未显示增益，不作有效性结论。原版 MOPD OCR 和 OPSD DrawBench/open3 的 checkpoint 重载评测也已完成，各 4 条。
- 完整 64-prompt 冻结诊断已完成：384 查询、1920 候选、0 invalid、128 条轨迹 noop 跨位置一致，图片/记录哈希全部通过。12 查询/60 图经过实施者定性复核；发现 OCR 分歧，不能替代独立人工评价。
- 三档学习率的一次更新检查完成，初始候选图片/分数/选择完全相同；接受目标误差的 after/before 为 3e-4: 3.379、1e-4: 0.672、3e-5: 0.426。该结果不等于训练后生成质量增益。尚无正式训练有效性或论文数值复现结论。
- 原版 8-GPU/2-node 启动封装和完整数据计划已准备，Slurm test-only 请求通过；实际跨节点 GPU 执行尚未验证。

## 初建记录（历史）

以下是提交首批 GPU 作业前的记录。

## 已验证

- 两份新 clone 的 origin、HEAD 与 `upstream.lock.json` 一致，工作树干净。
- 原 OPD 训练入口在原 OPD 环境成功导入；新 OPSD 训练入口在 `.venv-opsd` 成功导入。两个环境分别保留 diffusers 0.33.1/0.36.0、PEFT 0.10.0/0.17.1。
- 最终 `python -m pytest -q`：**27 passed**。包含从上游 AST 提取 loss 的数值比较、生命周期补丁 AST 等价性、教师路径审计、约束保护、invalid/quality/tie 处理、split/协议拒绝、完整续训恢复和真实小型 SD3+PEFT 的梯度归属测试。
- 小型 SD3+PEFT 测试也在 OPSD 覆盖环境单独通过。
- CLI 单进程 CPU toy：训练到 epoch 1、从完整 checkpoint 续训到 epoch 3、base/训练后 adapter 的 held-out 评测、配对比较，全部完成。
- CLI 双进程 CPU/gloo toy：完成一个 epoch，保存 `rng-rank-0.pt`、`rng-rank-1.pt`，没有将两个 rank 的更新数误相加。
- OPD 的独立 Accelerate 配置成功解析为单机 8 进程、fp16；不读取用户以前的默认 launcher 配置。
- OPSD 复用 train prompts 的 SHA256 和 25415 行与新上游 recipe 一致，test 从新上游 DrawBench 复制。
- 已生成两种 native evaluation 的计划；未生成真实图像。

## 可直接查看的本机产物

| 路径 | 状态 |
|---|---|
| `runs/ready-native-opd/config.audit.json` | 原 MOPD defaults/resolved/diff；包含三条教师路径绑定和明确的 300 outer-epoch 预算 |
| `runs/ready-native-opd/runtime_patches.json` | OCR 路径、循环上限和 checkpoint hook 的完整补丁 |
| `runs/ready-native-opsd/config.audit.json` | 原 public `sd35_open3` 配置，只绑定模型和输出路径 |
| `runs/ready-native-opd-eval/plan.json` | 原版 OPD OCR 评测计划 |
| `runs/ready-native-opsd-eval/plan.json` | 原版 OPSD DrawBench/open3 评测计划 |
| `runs/ready-verified-demo/` | 新方法 SD3 配置和资产快照，**仅使用 schema demo 数据，未训练** |
| `runs/verified-toy-smoke/` | 单进程训练/续训/逐候选记录和三个 epoch 的 checkpoint |
| `runs/toy-distributed-smoke/` | 双进程 CPU smoke |
| `runs/smoke-eval-base/`、`runs/smoke-eval-trained/` | 统一协议的 CPU toy 逐样本结果 |
| `docs/toy-comparison.json` | toy 配对比较；仅验证程序链路，无图像方法有效性含义 |
| `docs/doctor.opd.local.json`、`docs/doctor.opsd.local.json` | 本机环境与主要资产检查 |

## 初建时尚未验证（历史状态，后续进展见顶部）

- SD3.5 全尺寸模型的真实 GPU forward/backward、显存占用、PaddleOCR/GenEval 图像路径及长时间多卡稳定性。
- 任一论文的数值复现；保留 source/config 并不等于已复现 reward 曲线。
- 新方法是否优于 MOPD/OPSD、固定教师、随机筛选等，以及跨 sampler/steps 是否稳定增益。
- 正式同图多约束 benchmark。当前只有 schema demo，需要准备足量且严格分开的训练/验证/测试 prompts。

旧项目训练代码、旧实验记录、模型和教师权重未改写。新 OPSD 覆盖包仅安装于本目录；运行时复用的模型/数据/奖励资产路径与指纹均显式记录。
