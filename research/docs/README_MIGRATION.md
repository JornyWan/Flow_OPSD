# CA-OPD / DiffusionOPD —— 从 colligo(Pluto) 迁移到 minilab SLURM 集群

> 迁移日期 2026-08-23。原项目全量存档在 HF 私有仓库 `Jorny/CA-OPD` (94.17 GB / 8597 文件)。

## 0. 原服务器 vs 本集群

| | 原 (colligo / Pluto) | 本集群 (minilab SLURM) |
|---|---|---|
| 调度 | Pluto job (AIP MCP 提交), P0/P2 抢占等级 | SLURM `sbatch`, sablab-gpu / minilab-gpu |
| GPU | 8×H200 **单节点** (p5en.48xlarge) | 单节点最多 4 卡 → **2 节点 × 4 卡 = 8 进程** (NCCL over InfiniBand,已实测) |
| 存储 | `/mnt/localssd` (作业结束即擦) + S3 同步 | `/midtier/minilab/scratch/` NFS,**持久**,不需要同步 |
| 环境 | uv venv (`/mnt/localssd/opd/venv{,_mopd}`),打包成 S3 tar 分发 | conda env `conda_envs/envs/diffusionopd`,**单一环境含全部 reward 依赖** |
| 分发 | 每个 pod 从 S3 拉 bundle 解包到相同绝对路径 | 无需分发,所有节点共享 NFS |
| 联网 | pod 有公网 | **计算节点离线**,登录节点有网 → 权重/模型必须预热 |
| 断点 | autorecovery-v2 自动重启(从头) | `--requeue`;⚠️ 训练脚本本身**没有 resume** |

## 1. 目录布局

```
opd_related/
├── CA-OPD/                     # HF 仓库全量镜像
│   ├── code/DiffusionOPD-main/ # 代码(已含全部补丁) ← 训练/评测在这里跑
│   │   └── reward_ckpts/       # aesthetic / mask2former / open_clip / HPSv2.1 / paddleocr
│   ├── paper/                  # CA-OPD 论文 (LaTeX + 图 + 数据表)
│   ├── results/eval/           # 每个 ckpt 每个 axis 的评测 JSON
│   ├── conversation/           # 原 Claude 会话记录 + opd_project_notes.md (交接文档,必读)
│   ├── reference_papers/       # DiffusionOPD / Flow-OPD / DiffusionNFT / flow-grpo 的 arXiv 源
│   ├── html/                   # 结果 dashboard
│   └── checkpoints/<run>/      # 每个 run 的 LoRA checkpoint (每 30 步一存)
├── models/
│   ├── stable-diffusion-3.5-medium/   # gated,需 HF token
│   └── DiffusionOPD/{Aes,OCR,GenEval}Teacher/lora, Student/lora, Multi_Task_RL/*
├── src/{mmcv,mmdetection}/     # editable 安装的源码 (不要在这两个目录下跑 python!)
├── build/0{1..4}_*.sh          # 环境构建脚本 + 日志 (可重跑)
├── slurm/opd_{train,eval}.slurm
├── outputs/                    # SLURM 日志 + 训练 logs + eval 结果
└── env.sh                      # 统一环境段,所有脚本 source 它
```

## 2. 跑起来

```bash
source opd_related/env.sh          # 设好 env、模型路径、离线开关、缓存位置

# 训练 (2 节点 × 4 卡 = 8 进程,复刻原 8×H200 的 72 samples/epoch)
RUN_CFG=mopd        sbatch opd_related/slurm/opd_train.slurm
RUN_CFG=mopd_caopd  sbatch opd_related/slurm/opd_train.slurm
RUN_CFG=mopd_poe_l20 sbatch opd_related/slurm/opd_train.slurm

# 单节点 4 卡冒烟 (effective batch 减半,只验证跑通,别用来出数)
RUN_CFG=mopd sbatch --nodes=1 opd_related/slurm/opd_train.slurm

# 评测 (student 全数据集用 gs=4.5;GenEval-teacher 用 1.0)
CKPT=.../checkpoints/checkpoint-721 DATASETS=geneval,ocr,drawbench OUTNAME=mopd721 \
  sbatch opd_related/slurm/opd_eval.slurm
CKPT=base GS=1.0 OUTNAME=base_gs1.0 sbatch opd_related/slurm/opd_eval.slurm
```

## 3. 迁移时要注意的坑 (原服务器踩过 + 本集群新增)

**沿用的(已写进构建脚本/代码)**
1. `setuptools<81` —— ≥81 移除了 `pkg_resources`,mmcv1.x/openmim 会 `ModuleNotFoundError`。
2. mmcv/mmdet 必须 `pip install -e . --no-build-isolation` (setup.py 在 build 期 import torch)。
3. `import mmcv` 必须在 `import torch` 之后 (mmcv._ext 需要先加载 libc10.so)。
4. **不要在 `src/mmcv/` 目录下跑 python** —— 会把 mmcv 包 shadow 成 namespace package。
5. peft==0.10.0 不认 teacher `adapter_config.json` 里的 `corda_config` 等新字段 → 构建脚本已剥离(留 `.orig` 备份)。
6. paddle + forked DataLoader worker 不兼容 → 所有 `num_workers=0` (代码里已改)。
7. `train_sd3_opd.py` 原 repo 忘了调 `save_ckpt()` → 已补。
8. `config_flags` 的 `--config x.py:name` 是把 name 当**参数**传给 `get_config(name)`,不是直接调用 `name()` → 两个 config 文件都已加分发器。

**本集群新增**
9. 单节点最多 4 卡 → 用 `--nodes=2`;`mopd` 的 `8×bs3×3 = 72 samples/epoch` 必须用 8 进程才忠实,4 卡会减半。
10. **计算节点没有外网** → open_clip ViT-L-14、PaddleOCR 模型、HF 权重都必须在登录节点预热 (`build/04_warm_and_check.sh`),运行时 `HF_HUB_OFFLINE=1`。
11. 配置里多处用 `os.getcwd()` 拼 dataset 路径 → **必须 cd 到 repo 根目录再启动**。
12. `evaluation.py` 是按路径跑的,需要 `PYTHONPATH=$OPD_REPO` 才能 import `flow_grpo`。
13. **resume 已补上**(上游只存不读)。`save_ckpt` 除 `lora/` 外多写一份 `train_state.pt`(~300MB):
    **RAW(非 EMA)权重** + AdamW 动量 + EMA 影子 + epoch/global_step + CA-OPD 的 `teacher_kl_ema`/`teacher_kl_floor`;
    `os.replace` 原子写入,只保留最新一份。用 `OPD_RESUME=auto`(扫 save_dir)或指定 checkpoint 目录;
    不设/设 `none` 就是上游原行为。
    ⚠️ **`lora/` 里存的是 EMA 权重(供 eval)**,直接拿它续训等于每次 resume 都把 EMA 再叠一层,所以必须另存 RAW 权重。
    验证脚本:`slurm/resume_test.slurm`(两阶段:训 2 epoch → resume 续到 4 epoch)。
14. ⚠️ **`run_name` 里的空格和方括号是反复出现的地雷**。`config.save_dir` 形如
    `logs/[MOPD-CAOPD + SD3.5-M + PickScore+OCR+GenEval]`,已经咬过两次:
    - **glob 字符类**:`glob.glob(f"{save_dir}/checkpoints/*")` 静默返回空(`[...]` 被当成字符集)
      → 必须 `glob.escape(save_dir)`。(auto-resume 找不到 checkpoint、旧 train_state 不被清理都源于此)
    - **shell word-split**:`CKARG="--checkpoint_path $CKPT"` 再不加引号展开,路径按空格被拆成多个参数
      → argparse 退出码 2。必须用数组:`CKARG=(--checkpoint_path "$CKPT")` + `"${CKARG[@]}"`。
    写任何遍历/传递 checkpoint 路径的代码前,先想一遍这两条。
15. 评测脚本也必须加 **GPU 数守卫**。SLURM 降级到 3 卡时,`--max_samples 512` 不能被 rank 数整除,
    分布式采样器会补齐出重复样本(512 → 513),破坏与原记录的可比性(影响 ~0.2%,小于噪声但应避免)。
16. SLURM 有时会**降级 gres 请求**(要 4 卡给 3 卡) → 脚本里已加 `nvidia-smi -L | wc -l` 守卫 + requeue。
