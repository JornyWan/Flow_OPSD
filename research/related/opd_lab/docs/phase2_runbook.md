# 第二阶段的执行与复核

所有命令在 `opd_lab/` 执行。实验定义在 [冻结方案](../experiments/local_expert_v2/PROTOCOL.md)，当前 run 和 job 的对应关系在 [runs.json](../experiments/local_expert_v2/runs.json) 与 [evaluation_jobs.json](../experiments/local_expert_v2/evaluation_jobs.json)。

```bash
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH="$PWD/python"
```

## 数据到训练

`scripts/prepare_controlled_pilot.py --site site.local.json` 是本轮已经执行过的准备入口。它冻结 192 条均衡查询，并给四臂各建独立源码/config/data 快照；设计已存在时拒绝覆盖。实际提交命令及参数完整保存在 `ledger.jsonl`，每个 run 另存 `job.sh` 和 `submission.json`。

所有训练臂实际执行 12 次更新。每个 epoch：复制 behavior → 按共同清单采集 16 个查询 → 每个查询保存 5 个候选终点 → 选择/拒绝 → 一次累积更新 → 查询状态上的更新后诊断。不同臂更新后的状态允许不同。

新实验应新建 campaign、配置、查询清单和 run 名称。改变 seed、数据、步数或总查询预算时，必须重新生成与配置严格匹配的查询清单；不能通过修改已有 run 的 manifest 续跑。

## 调度与失败记录

```bash
python scripts/refresh_campaign.py --campaign experiments/local_expert_v2
```

该命令只查询本 campaign 的 Slurm 作业并追加结算记录，不提交、取消或判定科学有效性。GPU-hours 按所有成功和失败作业的已分配 GPU 数乘实际时长累计。

首批三个评价作业在生成图像前因计算节点缺少 OCR 共享库而退出。attempt 002 保留相同 GPU 生成和主评分，将第二 OCR 延后到已校准的 CPU 主机。准备/提交脚本为 `prepare_controlled_evaluation.py`；旧映射保存在 `evaluation_jobs_before_attempt_002.json`。这一修复没有重新训练，也没有替换评分器。

## 评价与审查

GPU 每格依次生成 base 和四臂的全部图片，输出状态 `awaiting_secondary_ocr`。在原 CPU 校准主机执行：

```bash
"$OPDLAB_PY" scripts/finalize_controlled_evaluation.py \
  --campaign experiments/local_expert_v2 --refresh-directory-cache
```

它只处理已经生成完毕但尚未完成第二 OCR 的格；调用 run 中冻结的 worker，对相同 PNG 评分，不加载扩散模型。完成后检查主评分产物完全未变，保存 CPU 用时、命令、主机和哈希。第二 reader 的假阴性问题继续保留在解释中。

本机 NFS 的目录缓存为 1800 秒，计算节点已经写完的新文件可能暂时在登录节点显示不存在。上述可选参数只刷新本轮输出目录的 mtime，不改文件内容、不改全局挂载设置；之后仍严格校验所有评分与图片哈希。

预定的 flow40 图片复核为 16 prompts × 5 arms，replicate 42。使用 `prepare_validation_review.py`，输入 `qualitative_validation_plan.json` 与本轮 flow40 的 `evaluations/`。先看 `case-*.png`、保存观察，再读取 `key.json` 和评分。实施者审查不能声称是独立人工盲测。

保存 `assistant_observations.json` 后，用以下命令揭示身份并合并两套分数，保留所有不确定读数：

```bash
"$OPDLAB_PY" scripts/summarize_validation_review.py \
  --review runs/le2_validation_review_001 \
  --evaluation runs/le2_eval_flow40_job_002/evaluations
```

## 训练诊断与最终报告

对每个完成的训练 run 执行一次：

```bash
"$OPDLAB_PY" scripts/report_training_smoke.py \
  --run runs/le2_train_verified_multi_001 \
  --out runs/le2_train_verified_multi_001/postfit_audit.json
```

最后先更新 Slurm 结算，再运行：

```bash
"$OPDLAB_PY" scripts/summarize_controlled_pilot.py \
  --campaign experiments/local_expert_v2 --out runs/le2_report_001 \
  --review runs/le2_validation_review_001
```

报告拒绝修改过的方案、缺失查询、候选 invalid、损坏图片、缺失评价格或不匹配的协议。结果目录不能覆盖。报告包括全部预定对比、逐约束回退、第二 OCR 敏感性及训练中目标→局部拟合→完整采样的诊断；`STATUS.md` 记录解释与下一决策。

64 条 test 始终未使用；本轮仅支持开发阶段判断。完整论文复现、多个训练 seed、更大数据和计算匹配的 native baseline 需要单独的实验阶段。
