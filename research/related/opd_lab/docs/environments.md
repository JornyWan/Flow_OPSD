# 环境与依赖

## 两份上游不能共用一套核心版本

新拉取的 OPD `setup.py` 固定 torch 2.6.0、diffusers 0.33.1、transformers 4.40.0、PEFT 0.10.0、accelerate 1.4.0。本机现有 `conda_envs/envs/diffusionopd` 满足这些核心版本；旧训练源码不会随之被使用，因为 launcher 将 run 内源码置于 PYTHONPATH 最前。

新 OPSD `pyproject.toml` 要求 diffusers >=0.36、transformers >=4.51,<4.52、Hub >=0.30,<1.0 等。diffusers 0.36 自身另外要求 PEFT >=0.17。不能直接拿原 OPD 环境启动，然后声称两者环境等价。

当前目录创建了 `.venv-opsd`，通过 Python venv `--system-site-packages` 复用原环境的 torch/CUDA、检测器和奖励依赖，仅在新目录覆盖：

```text
diffusers==0.36.0
transformers==4.51.3
huggingface-hub==0.34.4
tokenizers==0.21.4
peft==0.17.1
timm==1.0.19
```

这些是满足公开实现版本范围的一组显式选择，不是作者未提供的完整历史 lockfile。该覆盖环境不是完全隔离的容器：底层环境若变动也会影响它。`environment` 审计记录实际版本，训练启动时检查与准备时一致；`scripts/create_opsd_overlay.sh` 可重新建立相同覆盖层。真实 OPSD smoke 暴露了共享 timm 0.6.13 缺少 Transformers 4.51 需要的 ImageNetInfo，因此仅在覆盖层升级 timm；共享 OPD 环境未改。

本机 xformers 显示其二进制编译 CUDA 版本与 torch wheel 不匹配，bitsandbytes 显示未启用 GPU 编译。当前普通 AdamW/SDPA 路径已完成真实 GPU 训练 smoke，PaddleOCR/MMDetection 也完成 60 图诊断检查；这些未使用的加速扩展仍未验证。原 OPD 的 Accelerate unwrap 会导入已安装 DeepSpeed，site 需显式绑定现有 CUDA 12.6 工具链的 CUDA_HOME。Slurm 为每个作业使用节点本地临时目录，避免 NFS 多进程清理错误。

## 从另一台机器重建

1. 根据每份固定上游的安装文档分别准备环境。OPD 的 MMDetection/GenEval、PaddleOCR 等奖励依赖和权重也需完整安装；不要只安装本项目轻量 `pyproject.toml` 就认为图像训练环境齐全。
2. `python -m opdlab fetch` 获取 `upstream.lock.json` 中的两个 commit。
3. 从 `site.example.json` 创建自己的 site 文件，绑定明确的 Python、base model、教师、reward_ckpts、Pick-a-Pic 数据。
4. 执行 `doctor`、pytest、native-prepare 检查；在 GPU 作业中做一个明确标记为 smoke 的训练/评测。少步、小 batch、小数据 smoke 不能替代原 recipe 的复现。

原始上游安装要求保留在 `upstream/diffusionopd/setup.py` 和 `upstream/diffusionopsd/pyproject.toml`，不将两个冲突的依赖列表合并。`site.local.json` 用原 OPD 环境；`site.opsd.local.json` 用覆盖环境。资产全部只读复用，现有教师的 adapter_config（包括兼容性处理）连同权重均进入指纹，未被本项目改写。

## 缓存和计量

HF cache/torch cache 通过 site 中显式的缓存路径访问；不存在自动复制数十 GB 模型的操作。当前将 CLIP/PickScore/processor 绑定到具体 HF snapshot 目录，上游 scorer 中模型 ID 的本地路径替换有独立 diff，数学实现不变；GenEval OpenCLIP 的本地 ViT-L-14.pt 也单独纳入资产指纹。

早期 run 指纹覆盖了整个 HF 缓存；新版 Transformers 添加 .no_exist/chat_template.json/.jinja 标记，触发过保护。审计确认所有原文件未变后，旧 run 只允许这两种新增负缓存标记，并显式记录；任何原文件改变、删除或新增权重仍拒绝。新 run 直接冻结 payload snapshot，避免这类可变缓存元数据。所有旧 manifest 和失败记录保留。

首次完整权重 SHA256 会读取较大文件；使用 `--hash-weights` 显式执行。默认小文件 hash + 大文件 stat 仅用于本机快速变更检查。
