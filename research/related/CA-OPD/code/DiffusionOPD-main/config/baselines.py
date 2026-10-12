"""Config functions for the DiffusionOPD paper's Multi-Task RL baselines (Table 1):
  - mt_grpoguard          -> Multi-Task GRPO-Guard  (scripts/baselines/train_sd3_GRPO_Guard_multireward.py, accelerate)
  - mt_nft                -> Multi-Task NFT          (scripts/baselines/train_nft_sd3_multireward.py, torchrun)
  - cascade_nft_s{1,2,3}  -> Cascade NFT stages      (same NFT script, one task per stage, LoRA-chained)

All: SD3.5-Medium @512, LoRA r=32/alpha=64 (hardcoded in the trainers), same 3 reward tasks as OPD's
`mopd` (Aes = 1:1:1 PickScore+ClipScore+HPSv2.1 / OCR / GenEval) alternating datasets (round-robin).
Guidance: GRPO-Guard gs=4.5 (CFG), NFT gs=1.0 + dpm2 deterministic (CFG-free). Launch: `--config config/baselines.py:<fn>`.
"""
import importlib.util
import os

import ml_collections


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))
_MODELS = os.environ.get("OPD_MODELS_DIR", "/mnt/localssd/opd/models")
SD35_PATH = os.environ.get("SD35_PATH", os.path.join(_MODELS, "stable-diffusion-3.5-medium"))


def _scale_to_nproc(config):
    """本集群单节点最多 4 卡,而上游配置是按 8 卡写的(bs3 x 8proc x 3batch = 72 samples/epoch)。

    OPD_NPROC=4 时把 num_batches_per_epoch 翻倍 -> 4proc x bs3 x 6batch = 72,与 8 卡**梯度严格等价**:
      - 每 rank 收集 total_batch_size = bs * nbpe 个样本,reshape 成 nbpe 个 micro-batch,
        每个 micro-batch 大小恒为 bs(=3),两种配置一致;
      - accelerate 的 gradient_accumulation_steps = nbpe * num_train_timesteps,每次 backward 除以它,
        DDP 再对 rank 取平均 -> 两种配置都等于"对同样 72 个样本的 micro-loss 求均值";
      - 每个 teacher 每 epoch 拿到的 prompt 数也相同(8卡 1batch x 24prompt = 4卡 2batch x 12prompt = 24)。
    差异仅在于每 epoch 的串行 micro-step 变多(墙钟更慢),以及 prompt 采样的随机顺序不同。
    """
    # 在线 eval 是纯观测的(train_sd3_opd.py 里 eval_metrics_per_slot 赋值后从未被使用),
    # 却占 ~60% 墙钟。OPD_EVAL_FREQ 可统一放大间隔;save_freq 不变,checkpoint 粒度不受影响,
    # 论文数字本来就来自 scripts/evaluation.py 对 checkpoint 的离线评测。
    _ef = os.environ.get("OPD_EVAL_FREQ")
    if _ef:
        config.eval_freq = int(_ef)
    # 1 epoch = 1 个 optimizer step(gas = nbpe x num_train_timesteps = 内层 backward 次数)。
    # 参照组 caopd(tau=0.5) 的离线评测在 541->721->781 已走平,mopd 最佳点就是 721,
    # 所以 800 步足够覆盖全部对比点并留余量。用户 2026-08-24 定为 800。
    _ne = os.environ.get("OPD_NUM_EPOCHS")
    if _ne:
        config.num_epochs = int(_ne)
    nproc = int(os.environ.get("OPD_NPROC", "8"))
    if nproc == 8:
        return config
    assert 8 % nproc == 0, f"OPD_NPROC={nproc} 必须能整除 8"
    factor = 8 // nproc
    config.sample.num_batches_per_epoch *= factor
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    n_t = len(config.train.teachers) if "teachers" in config.train else 1
    assert config.sample.num_batches_per_epoch % n_t == 0, (
        f"缩放后 num_batches_per_epoch={config.sample.num_batches_per_epoch} 必须能被 teacher 数 {n_t} 整除")
    return config


def get_config(name):
    # config_flags calls get_config(<the :suffix>), e.g. --config config/baselines.py:mt_nft -> get_config("mt_nft")
    config = _scale_to_nproc(globals()[name]())
    if name.startswith("eph_"):
        # 必须在 _scale_to_nproc 之后:后者按 8->4 卡把 nbpe 翻倍,
        # 而 ephemeral 给的是每卡绝对值,最后生效才是准的。
        config = _ephemeral_override(config, name[len("eph_"):])
    return config


def _aes():
    # paper's Aesthetics objective: equal-weight (1:1:1) mixture
    return {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}


def _tasks():
    cwd = os.getcwd()
    return [
        {"name": "aes",     "dataset": os.path.join(cwd, "dataset/pickscore"), "prompt_fn": "general_ocr", "reward_fn": _aes()},
        {"name": "ocr",     "dataset": os.path.join(cwd, "dataset/ocr"),       "prompt_fn": "general_ocr", "reward_fn": {"ocr": 1.0}},
        {"name": "geneval", "dataset": os.path.join(cwd, "dataset/geneval"),   "prompt_fn": "geneval",     "reward_fn": {"geneval": 1.0}},
    ]


def _common(config):
    config.pretrained.model = SD35_PATH
    config.resolution = 512
    config.use_lora = True
    config.mixed_precision = "fp16"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")  # single-task fallback (unused in multi-task)
    # group-relative RL: many images per prompt. 8 GPUs * tbs 9 = 72; 72 % nipp 24 == 0; nbpe 6 % 3 tasks == 0.
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = 6
    config.sample.test_batch_size = 16
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.7
    config.train.batch_size = config.sample.train_batch_size
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.ema = True
    config.train.adv_clip_max = 5
    config.train.learning_rate = 1e-4
    config.train.lora_path = None
    config.per_prompt_stat_tracking = True
    config.save_freq = 30
    config.eval_freq = 30
    config.num_epochs = 100000  # stop manually at best-performing checkpoint (paper protocol)
    return config


def mt_grpoguard():
    config = _common(base.get_config())
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5
    config.train.gradient_accumulation_steps = 3
    config.train.cfg = True
    config.train.beta = 0.0
    config.train.clip_range = 4e-6
    # GRPO-Guard-specific (absent from base.py)
    config.rationorm = True
    config.proxy_advantage_ratio = 0.0
    config.proxy_reward_fn = {}
    # multi-task tasks live under train.datasets for the GRPO-Guard trainer
    config.train.datasets = _tasks()
    config.run_name = "MT-GRPOGuard_SD35M_Aes+OCR+GenEval"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


def _nft_base():
    config = _common(base.get_config())
    config.sample.guidance_scale = 1.0   # DiffusionNFT is CFG-free
    config.sample.eval_guidance_scale = 1.0
    config.sample.solver = "dpm2"
    config.sample.deterministic = True
    config.train.gradient_accumulation_steps = 6
    config.train.beta = 0.0001
    config.train.adv_mode = "all"
    # NFT-specific top-level (absent from base.py)
    config.beta = 1.0
    config.decay_type = 1
    config.resume_from = None
    return config


def mt_nft():
    config = _nft_base()
    config.datasets = _tasks()           # multi-task tasks at TOP-LEVEL for the NFT trainer
    config.run_name = "MT-NFT_SD35M_Aes+OCR+GenEval"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


def _ephemeral_override(config, name):
    """BootstrapOPD: 把一次 NFT run 压成"短程微专家"。

    DistributedKRepeatSampler 有硬约束 (nproc * train_batch_size) % num_image_per_prompt == 0,
    且上游默认按 8 卡写(8*9=72, k=24 -> 每 batch 3 个 prompt x 24 张)。本集群 4 卡时
    4*9=36 不被 24 整除会直接 assert 失败,所以这里显式给出一组自洽的 4 卡参数:
        k=12, bs=12 -> 每 batch 4*12=48 张 = 4 个 prompt x 12 张; nbpe=6 -> 288 张/epoch
    (基线 432 张/epoch 的 0.67x, 单 epoch 算力与一个 OPD epoch 相当)。
    专家只需指出改进方向、不需要收敛,因此 EPH_EPOCHS 默认 25。
    BOOT_TAG 让每一轮的 save_dir 互不覆盖。
    """
    if os.environ.get("EPH_ON") != "1":
        return config
    nproc = int(os.environ.get("OPD_NPROC", "8"))
    k = int(os.environ.get("EPH_GROUP", 12))
    bs = int(os.environ.get("EPH_BS", 6))
    nbpe = int(os.environ.get("EPH_NBPE", 12))
    if (nproc * bs) % k != 0:
        raise ValueError(
            f"ephemeral expert: (nproc={nproc} * bs={bs}) = {nproc * bs} 不能被 "
            f"num_image_per_prompt={k} 整除,DistributedKRepeatSampler 会 assert 失败。"
            f"请调整 EPH_BS / EPH_GROUP。"
        )
    config.sample.num_image_per_prompt = k
    config.sample.train_batch_size = bs
    config.train.batch_size = bs
    config.sample.num_batches_per_epoch = nbpe
    config.train.gradient_accumulation_steps = nbpe    # 1 epoch = 1 个 optimizer step
    config.num_epochs = int(os.environ.get("EPH_EPOCHS", 25))
    config.save_freq = config.num_epochs               # 只存最后一版(专家用完即弃)
    config.eval_freq = 10 ** 9                         # 在线 eval 纯观测,短程 run 不需要
    tag = os.environ.get("BOOT_TAG", "r0")
    config.run_name = f"Bootstrap-{tag}_{name}"
    config.save_dir = os.path.join("logs", config.run_name)
    if os.environ.get("OPD_SMOKE") == "1":
        # 与 config/opd.py 的 smoke 分支对齐:微型数据集 + 单一 reward + 1 个 epoch。
        # 只验证「专家能训出来并落盘、OPD 能把它当 teacher 加载」这条链路,不产出研究数据。
        _tiny = ("/midtier/minilab/scratch/chw4026/projects/side_project/generative/"
                 "opd_related/testdata/pickscore")
        config.dataset = _tiny
        config.prompt_fn = "general_ocr"
        config.reward_fn = {"pickscore": 1.0}
        config.num_epochs = int(os.environ.get("EPH_EPOCHS", 1))
        config.save_freq = config.num_epochs
        config.run_name = "[SMOKE] " + config.run_name
        config.save_dir = os.path.join("logs", config.run_name)
    return config


def _nft_stage(name, dataset, prompt_fn, reward_fn, beta, decay_type, num_steps, lora_path):
    config = _nft_base()                  # no config.datasets -> single-dataset back-compat path
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.prompt_fn = prompt_fn
    config.reward_fn = reward_fn
    config.beta = beta
    config.decay_type = decay_type
    config.sample.num_steps = num_steps
    config.train.lora_path = lora_path    # CASCADE CHAIN: weights-only init from previous stage
    config.run_name = f"Cascade-NFT_{name}"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


# ============ Ephemeral Expert (BootstrapOPD) 的三个单任务微专家 ============
# CFG 一致性是这里的关键正确性约束:
#   OPD 训练时 student 的速度用 CFG(train.cfg=True, gs=4.5)算,teacher 目标按 per-teacher
#   guidance_scale 算 -> aes/ocr 是 4.5, geneval 是 1.0(见 config/opd.py 的 *_teacher_gs)。
#   所以微专家必须在**它将被当作 teacher 评估时的同一个 CFG** 下训练,否则 teacher 目标落在
#   训练分布之外,会把一个本可以成立的想法测成假阴性。
#   -> aes / ocr 用 GRPO-Guard(CFG 4.5, train.cfg=True);geneval 用 DiffusionNFT(CFG-free, 1.0)。
# 三者都从 CASCADE_PREV_LORA(= 当前 student)初始化,这正是"ephemeral"的含义。
def _grpo_stage(name, dataset, prompt_fn, reward_fn):
    config = _common(base.get_config())
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5
    config.train.gradient_accumulation_steps = 3
    config.train.cfg = True
    config.train.beta = 0.0
    # mt_grpoguard 用 4e-6,比 base.py 的默认 1e-4 小 25 倍。实测该配置下 pickscore
    # 在 240 步里完全不动(mtgrpo720/780/960 = 0.8821/0.8814/0.8821),而我们的 ephemeral
    # 专家在 25 epoch 里 reward 反而下降(1.5809 -> 1.5337, ~8 SEM)。
    # BootstrapOPD 的成立与否取决于短程 RL 能否造出**更好**的专家,所以必须先排除
    # "是配置压制了学习"这个混淆项 —— EPH_CLIP 用来做这个对照。
    config.train.clip_range = float(os.environ.get("EPH_CLIP", 4e-6))
    config.rationorm = True
    config.proxy_advantage_ratio = 0.0
    config.proxy_reward_fn = {}
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")   # 单数据集回退路径
    config.prompt_fn = prompt_fn
    config.reward_fn = reward_fn
    config.train.lora_path = os.environ.get("CASCADE_PREV_LORA")
    config.run_name = f"Eph-GRPO_{name}"
    config.save_dir = os.path.join("logs", config.run_name)
    return config


def eph_aes():
    return _grpo_stage("aes", "pickscore", "general_ocr", _aes())


def eph_ocr():
    return _grpo_stage("ocr", "ocr", "general_ocr", {"ocr": 1.0})


def eph_aes_nft():
    """用 NFT 而非 GRPO 训练 aes 微专家。

    动机: 归档里 mt_nft 的训练曲线显示本仓库的 NFT **确实有效**
    (geneval reward 0.21->0.87 @630 步, aes 1.22->1.48),之后才崩塌;
    而 GRPO-Guard 的多任务基线 240 步 pickscore 不动。
    所以"ephemeral 专家没变好"到底是 trainer 选错还是步数太少,需要用 NFT 做对照。
    注意 NFT 是 CFG-free(gs=1.0),所以它产出的专家只能在 gs=1.0 下当 teacher 用。
    """
    return _nft_stage("aes", "pickscore", "general_ocr", _aes(), beta=0.1, decay_type=1,
                      num_steps=25, lora_path=os.environ.get("CASCADE_PREV_LORA"))


def eph_geneval():
    # GenEval teacher 在 OPD 里用 guidance_scale=1.0 -> 用 CFG-free 的 NFT 训练才对齐
    return _nft_stage("geneval", "geneval", "geneval", {"geneval": 1.0}, beta=1.0, decay_type=1,
                      num_steps=10, lora_path=os.environ.get("CASCADE_PREV_LORA"))


# Cascade NFT: sequential stages, each a single-task DiffusionNFT run initialized from the previous stage's LoRA.
# Per-task hyperparameters mirror upstream DiffusionNFT-main/config/nft.py. Set CASCADE_PREV_LORA between stages.
def cascade_nft_s1_aes():
    return _nft_stage("s1_aes", "pickscore", "general_ocr", _aes(), beta=0.1, decay_type=1, num_steps=25,
                      lora_path=os.environ.get("CASCADE_PREV_LORA"))


def cascade_nft_s2_ocr():
    return _nft_stage("s2_ocr", "ocr", "general_ocr", {"ocr": 1.0}, beta=0.1, decay_type=2, num_steps=10,
                      lora_path=os.environ.get("CASCADE_PREV_LORA"))


def cascade_nft_s3_geneval():
    return _nft_stage("s3_geneval", "geneval", "geneval", {"geneval": 1.0}, beta=1.0, decay_type=1, num_steps=10,
                      lora_path=os.environ.get("CASCADE_PREV_LORA"))
