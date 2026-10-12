# 第三阶段：冻结查询的拟合与保持诊断

本阶段使用独立的 `opdlab.fitting_probe`，不改变原版 native 入口或第二阶段的
on-policy engine。固定 64 个训练查询，先执行一次共同更新，再从完全相同的
student/Adam/GradScaler/RNG 分叉。每个分叉保存第 2、4 次更新的 raw LoRA。

`load_student(..., refresh_behavior=False)` 是此处必要的恢复方式。一般训练
仍保留原来的默认刷新行为。probe 的行为模型从收集到所有后验审查保持不变。
保持项是拒绝点的冻结 noop 回归；接受点仍使用原 transition-mean loss。

## 固定输入与执行

在 lab 根目录设置 `OPDLAB_PY` 为 `site.local.json` 中的解释器，并设置
`PYTHONPATH=python`。不要 source 旧实验的 env.sh。

```bash
"$OPDLAB_PY" scripts/prepare_fitting_probe.py --site site.local.json --attempt 001
```

该命令拒绝覆盖已有 attempt。另一个 attempt 仍须符合原先冻结的 query plan；
若要改实验设计，应新建 campaign，而不是修改已完成运行的输入。

准备结果包含两个目录：

- `runs/le3_probe_001`：框架、上游、config/site/data/protocol/query plan 快照；
  使用 `scripts/submit_slurm.py` 启动 `-m opdlab.fitting_probe --run <绝对路径>`。
- `runs/le3_eval_001`：已固定 hash 的 `worker.py` 和 `inputs.json`；在训练
  `afterok` 依赖满足后执行 `worker.py --inputs <绝对路径>`，逐一评价 base 和
  四个 raw 检查点的 flow10/flow40、32 validation prompts、seed 42。

已执行命令、资源、job id 与依赖完整保存在 campaign 的 `ledger.jsonl` 及各
attempt 的 `submission.json`，无需猜测运行参数。只用
`refresh_campaign.py --campaign experiments/local_expert_v3` 更新本阶段记账。

## 验证与报告

```bash
"$OPDLAB_PY" scripts/summarize_fitting_probe.py \
  --run runs/le3_probe_001 --evaluation runs/le3_eval_001 --out runs/le3_report_001

"$OPDLAB_PY" scripts/prepare_fitting_review.py \
  --run runs/le3_probe_001 --out runs/le3_review_001
```

汇总器检查 1152 张保存图片、448 次 backward query、共同首步、behavior hash、
四份 Adam 更新次数，以及 safetensors 导出与完整训练状态的一致性。任一输入、
图像、更新预算或评价协议不匹配都拒绝生成有效报告。统计保留全部预声明对照，
train query 局部收益、train 完整生成、validation 普通生成分开报告。

图像审查遵循 campaign 的 `REVIEW_RUBRIC.md`，只看八个预选训练 prompt 的
40 张图。先看匿名 A–E 图片，将 true/false/null 观察写入 `observations.json`，
再保存其 SHA256 与时间到 `observations_frozen.json`；此后才读取 `key.json`，
运行 `summarize_fitting_review.py --review runs/le3_review_001 --dataset runs/le3_probe_001/data.jsonl`。观察者不是独立
人工评价者；该审查不修改 v1 分数，也不构成新的方法有效性指标。

`training.pt` 保存的是此冻结 probe 的分叉状态，包含随机状态但没有 EMA。
不能把它当作 `engine.train_loop` 的普通 on-policy resume checkpoint。
将来若进入 on-policy 实验，应另立方案和运行目录，重新定义共同预算与对照。
