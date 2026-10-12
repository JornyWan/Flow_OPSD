import importlib.util
import os

import ml_collections

def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base = load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

# --- reproduction path overrides (env-var with local defaults) ---
# Same layout on the dev node and the training pod: /mnt/localssd/opd/models/...
# Override any of these with env vars at launch time if the layout differs.
_MODELS = os.environ.get("OPD_MODELS_DIR", "/mnt/localssd/opd/models")
SD35_PATH = os.environ.get("SD35_PATH", os.path.join(_MODELS, "stable-diffusion-3.5-medium"))
AES_TEACHER_PATH = os.environ.get("AES_TEACHER_PATH", os.path.join(_MODELS, "DiffusionOPD/AesTeacher/lora"))
OCR_TEACHER_PATH = os.environ.get("OCR_TEACHER_PATH", os.path.join(_MODELS, "DiffusionOPD/OCRTeacher/lora"))
GENEVAL_TEACHER_PATH = os.environ.get("GENEVAL_TEACHER_PATH", os.path.join(_MODELS, "DiffusionOPD/GenEvalTeacher/lora"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = SD35_PATH
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    config.prompt_fn = "general_ocr"

    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def mopd():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.pretrained.model = SD35_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.teacher_guidance_scale = 4.5

    config.resolution = 512
    config.run_project = "flow_grpo"

    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 3
    assert config.sample.num_batches_per_epoch % 3 == 0, (
        "Please set config.sample.num_batches_per_epoch to a multiple of "
        "len(config.train.teachers) (=3 here)."
    )
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"

    config.sample.noise_level = 0.0

    config.save_freq = 30
    config.eval_freq = 30

    config.reward_fn = ml_collections.ConfigDict()
    config.proxy_reward_fn = {}
    config.prompt_fn = "general_ocr"

    pickscore_lora = AES_TEACHER_PATH
    pickscore_teacher_gs = 4.5
    ocr_lora = OCR_TEACHER_PATH
    ocr_teacher_gs = 4.5
    geneval_lora = GENEVAL_TEACHER_PATH
    geneval_teacher_gs = 1.0
    config.train.teachers = [
        {
            "name": "pickscore",
            "dataset": os.path.join(os.getcwd(), "dataset/pickscore"),
            "lora_path": pickscore_lora,
            "proxy_reward": {"pickscore": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": pickscore_teacher_gs,
        },
        {
            "name": "ocr",
            "dataset": os.path.join(os.getcwd(), "dataset/ocr"),
            "lora_path": ocr_lora,
            "proxy_reward": {"ocr": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": ocr_teacher_gs,
        },
        {
            "name": "geneval",
            "dataset": os.path.join(os.getcwd(), "dataset/geneval"),
            "lora_path": geneval_lora,
            "proxy_reward": {"geneval": 1.0},
            "prompt_fn": "geneval",
            "guidance_scale": geneval_teacher_gs,
        },
    ]
    config.train.beta = 0.0

    config.run_name = "[MOPD + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def mopd_caopd():
    """CA-OPD: conflict-averse / Pareto multi-teacher OPD.

    Identical to mopd() except the uniform per-task loss average is replaced by an
    adaptive Chebyshev (min-max) reweighting over teachers. Linear scalarization
    (uniform mean) drives the student to the per-step teacher-MEAN velocity, which
    only reaches the convex hull of the per-task Pareto front and is Pareto-dominated
    when teacher objectives conflict. The min-max reweighting up-weights the
    currently worst-matched teacher, provably reaching the full (incl. non-convex)
    Pareto set. Enabled via config.train.ca_opd; tau->inf recovers mopd exactly."""
    config = mopd()
    config.train.ca_opd = True
    config.train.ca_opd_tau = 0.5   # Chebyshev temperature: ->0 pure min-max (worst teacher); large -> uniform (=mopd)
    config.train.ca_opd_ema = 0.9   # EMA over per-teacher KL for a stable difficulty estimate
    config.run_name = "[MOPD-CAOPD + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def _caopd_tau(tau, tag):
    """Original absolute-min-max CA-OPD at a given temperature tau (tau-sweep to test whether the
    non-SOTA result was just under-tuned; tau->inf recovers uniform mopd, tau->0 pure min-max)."""
    config = mopd_caopd()
    config.train.ca_opd_tau = tau
    config.num_epochs = 1000
    config.run_name = f"[MOPD-CAOPD-TAU{tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_caopd_tau02():
    return _caopd_tau(0.2, "0.2")


def mopd_caopd_tau10():
    return _caopd_tau(1.0, "1.0")


def mopd_caopd_tau20():
    return _caopd_tau(2.0, "2.0")


def _poe(lam, tag):
    """Product-of-Experts OPD: distill to v_base + lam*(v_teacher - v_base) per round-robin teacher, so the
    student converges to v_base + lam*mean(residual). lam=1 == uniform mopd (average-reward optimum);
    lam=M(=3) == sum-of-residuals == product-of-experts p* ~ p_ref*exp(sum_m R_m/beta), the true JOINT
    multi-reward optimum that uniform OPD under-applies by a factor M. Sweeps the average->joint frontier."""
    config = mopd()
    config.train.poe = True
    config.train.poe_lambda = lam
    config.num_epochs = 1000
    config.run_name = f"[MOPD-POE-L{tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_poe_l15():
    return _poe(1.5, "1.5")


def mopd_poe_l20():
    return _poe(2.0, "2.0")


def mopd_poe_l30():
    return _poe(3.0, "3.0")


def mopd_poe_l125():
    return _poe(1.25, "1.25")


def mopd_poe_protect():
    """Weighted-POE: amplify the ROBUST model-based rewards (PickScore, OCR: lambda=2) while keeping the
    BRITTLE rule-based GenEval at lambda=1 (protect composition). Targets the weighted product-of-experts
    p* ~ p_ref*exp((2 R_pick + 2 R_ocr + R_geneval)/beta). Teacher order = [pickscore, ocr, geneval]."""
    config = mopd()
    config.train.poe = True
    config.train.poe_lambda = [2.0, 2.0, 1.0]
    config.num_epochs = 1000
    config.run_name = "[MOPD-POE-PROTECT + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_poe_protect2():
    """Weighted-POE, gentler on PickScore/Clip: PickScore lambda=1.5, OCR lambda=2, GenEval lambda=1.
    Aims to keep the Aes/HPS/OCR gains while limiting the PickScore/Clip dip and protecting GenEval."""
    config = mopd()
    config.train.poe = True
    config.train.poe_lambda = [1.5, 2.0, 1.0]
    config.num_epochs = 1000
    config.run_name = "[MOPD-POE-PROTECT2 + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_caopd_regret():
    """CA-OPD with REGRET weighting r_m = ema_m - floor_m (reducible gap), floor = running-min KL.
    Absolute min-max chases intrinsically-hard-but-irreducible teachers (the diagnosed cause of the
    ceiling loss in plain CA-OPD); regret weights only the gap the student can still close, so it
    should reach a joint point that dominates uniform mopd rather than trading ceiling for speed."""
    config = mopd_caopd()
    config.train.ca_opd_regret = True
    config.num_epochs = 1000
    config.run_name = "[MOPD-CAOPD-REGRET + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_caopd_conflict():
    """Regret-weighted CA-OPD on a HIGHER-CONFLICT teacher set: the OCR teacher is distilled on the
    SAME (pickscore) prompt pool as the PickScore/aesthetic teacher, so the two objectives directly
    oppose on identical prompts (text-rendering vs. preference/aesthetics). This is the regime where
    min-max should yield a genuine Pareto improvement, not just faster convergence."""
    config = mopd_caopd_regret()
    # Share the OCR (quoted-text) prompt pool: PickScore teacher -> dataset/ocr. Both the PickScore
    # (any prompt) and OCR (needs `"..."`) rewards parse it, so aesthetics vs text-rendering genuinely
    # conflict on the SAME text-heavy prompts. (Sharing the pickscore pool instead breaks the OCR reward.)
    config.train.teachers[0]["dataset"] = os.path.join(os.getcwd(), "dataset/ocr")
    config.run_name = "[MOPD-CAOPD-CONFLICT + SD3.5-M + PickScore(on-OCR)+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_conflict():
    """Uniform (mopd) baseline on the SAME higher-conflict teacher set as mopd_caopd_conflict, so the
    conflict variant has a matched apples-to-apples uniform reference for per-axis comparison."""
    config = mopd()
    config.train.teachers[0]["dataset"] = os.path.join(os.getcwd(), "dataset/ocr")  # PickScore teacher -> OCR (quoted-text) prompts
    config.num_epochs = 1000
    config.run_name = "[MOPD-CONFLICT + SD3.5-M + PickScore(on-OCR)+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_robust():
    """Idea: robust / trust-region OPD. Huber on the per-step velocity residual so pathological
    RL-teacher targets (reward-hacking) don't dominate the match; a PPO-clip analog in velocity space."""
    config = mopd()
    config.train.opd_huber = True
    config.train.huber_delta = 1.0
    config.num_epochs = 780
    config.run_name = "[MOPD-ROBUST + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_snr():
    """Idea: Min-SNR-gamma per-step weighting. The ODE-L2 default weights denoising steps by a
    discretization artifact; reweight by min(SNR, gamma) to balance the learning signal across noise levels."""
    config = mopd()
    config.train.snr_weight = True
    config.train.snr_gamma = 5.0
    config.num_epochs = 780
    config.run_name = "[MOPD-SNR + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_anneal():
    """Idea: SDE->ODE noise annealing. Start with SDE stochasticity (noise_level=0.7) for exploration,
    anneal to the deterministic ODE regime (0.0) for sharp convergence; extends the paper's noise-level ablation."""
    config = mopd()
    config.train.anneal_noise = True
    config.train.noise_start = 0.7
    config.train.noise_end = 0.0
    config.train.anneal_epochs = 400
    config.sample.noise_level = 0.7  # start in the SDE regime
    config.num_epochs = 780
    config.run_name = "[MOPD-ANNEAL + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_base():
    """Shared single-teacher OPD defaults for SD3.5-M.

    Mirrors `mopd()` but without the multi-teacher-specific assertion;
    each `sopd_*` wrapper only needs to set the dataset, prompt_fn,
    a single `config.train.teachers` entry, and the run/save name.
    """
    config = compressibility()

    config.pretrained.model = SD35_PATH
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.teacher_guidance_scale = 4.5

    config.resolution = 512
    config.run_project = "flow_grpo"

    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 1
    config.sample.num_batches_per_epoch = 3
    config.sample.test_batch_size = 16

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99

    config.sample.global_std = False
    config.sample.same_latent = False
    config.train.ema = True
    config.mixed_precision = "fp16"

    config.sample.noise_level = 0.0

    config.save_freq = 30
    config.eval_freq = 30

    config.reward_fn = ml_collections.ConfigDict()
    config.proxy_reward_fn = {}
    config.train.beta = 0.0

    return config


def sopd_pickscore():
    config = sopd_base()

    pickscore_lora = AES_TEACHER_PATH
    pickscore_teacher_gs = 4.5

    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.prompt_fn = "general_ocr"
    config.train.teachers = [
        {
            "name": "pickscore",
            "dataset": os.path.join(os.getcwd(), "dataset/pickscore"),
            "lora_path": pickscore_lora,
            "proxy_reward": {"pickscore": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": pickscore_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def sopd_vteacher_pick():
    """MVP-1: VALUE-GRADIENT VIRTUAL TEACHER (no RL expert).

    Tests the mentor's hypothesis that Flow-OPD's pre-trained RL experts are unnecessary: their only
    role is to turn a sparse terminal reward into a dense per-step velocity. Here we get that dense
    velocity DIRECTLY from the FROZEN PickScore reward model's gradient -- v_target = v_base -
    (alpha/sigma_t) * grad_{xhat0} R(decode(xhat0)) -- the terminal limit of the value gradient
    (VGG-Flow / Doob h-transform). No AesTeacher LoRA signal is used. Head-to-head vs sopd_pickscore.

    The reward-gradient target is computed only on the last `vteacher_nsteps` (least-noisy) trajectory
    steps to bound the decode+CLIP backward memory; other steps carry no loss (target = student's own
    detached mean)."""
    config = sopd_pickscore()
    config.train.vteacher = True
    config.train.vteacher_reward = "pickscore"
    config.train.vteacher_alpha = 0.5     # fraction of |v_base| to move along the reward-improvement direction
    config.train.vteacher_nsteps = 4      # trajectory steps/batch with the decode+reward backward (memory bound)
    config.num_epochs = 1000
    config.run_name = "[SOPD-VTEACHER-PICK + SD3.5-M + PickScore(reward-grad, no expert)]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_vvalue_pick():
    """MVP-2: LEARNED-VALUE-GRADIENT virtual teacher. A conv critic V_psi(x_t,t,c) is trained by regression
    to the scalar PickScore (Monte-Carlo return); the target becomes v_base + alpha*grad_x V -- a smooth,
    non-myopic improvement direction that fixes MVP-1's myopic/CLIP-adversarial reward gradient. No RL
    expert; head-to-head vs sopd_pickscore. Uses only the scalar reward, so it extends to non-diff rewards."""
    config = sopd_vteacher_pick()
    config.train.vteacher_mode = "value"
    config.train.vteacher_alpha = 0.5
    config.train.vteacher_sign = 1.0
    config.train.value_lr = 1e-4
    config.train.value_steps = 16
    config.train.value_bs = 64
    config.run_name = "[SOPD-VVALUE-PICK + SD3.5-M + PickScore(value-grad, no expert)]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_ocr():
    config = sopd_base()

    ocr_lora = OCR_TEACHER_PATH
    ocr_teacher_gs = 4.5

    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.prompt_fn = "general_ocr"
    config.train.teachers = [
        {
            "name": "ocr",
            "dataset": os.path.join(os.getcwd(), "dataset/ocr"),
            "lora_path": ocr_lora,
            "proxy_reward": {"ocr": 1.0},
            "prompt_fn": "general_ocr",
            "guidance_scale": ocr_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + OCR]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def sopd_geneval():
    config = sopd_base()

    geneval_lora = GENEVAL_TEACHER_PATH
    geneval_teacher_gs = 1.0

    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.prompt_fn = "geneval"
    config.train.teachers = [
        {
            "name": "geneval",
            "dataset": os.path.join(os.getcwd(), "dataset/geneval"),
            "lora_path": geneval_lora,
            "proxy_reward": {"geneval": 1.0},
            "prompt_fn": "geneval",
            "guidance_scale": geneval_teacher_gs,
        },
    ]

    config.run_name = "[SOPD + SD3.5-M + GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"

    return config


def mopd_cagrad():
    """CAGrad multi-teacher OPD: resolve inter-teacher conflict via an output-space min-norm (MGDA) weighted
    combination of all M teacher targets, blended toward uniform by cagrad_c. Targets stay at the REAL teacher
    velocities (no amplification => structurally cannot reward-hack); aims to reach every teacher peak jointly."""
    config = mopd()
    config.train.cagrad = True
    config.train.cagrad_c = 0.5
    config.num_epochs = 1000
    config.run_name = "[MOPD-CAGRAD + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_cagrad_c10():
    config = mopd_cagrad()
    config.train.cagrad_c = 1.0
    config.run_name = "[MOPD-CAGRAD-C1.0 + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_cagrad_c025():
    config = mopd_cagrad()
    config.train.cagrad_c = 0.25
    config.run_name = "[MOPD-CAGRAD-C0.25 + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_cagrad_c075():
    config = mopd_cagrad()
    config.train.cagrad_c = 0.75
    config.run_name = "[MOPD-CAGRAD-C0.75 + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_pcgrad():
    """PCGrad multi-teacher OPD: per-sample remove pairwise-conflicting components of the teacher descent
    directions, then AVERAGE (bounded like uniform => no reward-hacking). Genuine conflict resolution."""
    config = mopd_cagrad()
    config.train.cagrad_mode = "pcgrad"
    config.run_name = "[MOPD-PCGRAD + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_resumetest():
    """仅用于验证 resume 机制:单 teacher + 微型数据集(train 256 / test 32),
    save_freq=1 让 checkpoint 立刻落盘。不产出任何研究数据。"""
    config = sopd_pickscore()
    tiny = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/testdata/pickscore"
    config.dataset = tiny
    for tc in config.train.teachers:
        tc["dataset"] = tiny
    config.save_freq = 1
    config.eval_freq = 10 ** 9      # epoch 0 仍会 eval(0%N==0),但测试集只有 32 条 = 1 个 batch
    config.num_epochs = int(os.environ.get("RESUMETEST_EPOCHS", 2))
    config.run_name = "[RESUMETEST + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


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
    if _ef not in (None, ""):
        config.eval_freq = int(_ef)   # 0 = 完全关闭在线 eval("0" 是假值,不能用 `if _ef:`)
    # 1 epoch = 1 个 optimizer step(gas = nbpe x num_train_timesteps = 内层 backward 次数)。
    # 参照组 caopd(tau=0.5) 的离线评测在 541->721->781 已走平,mopd 最佳点就是 721,
    # 所以 800 步足够覆盖全部对比点并留余量。用户 2026-08-24 定为 800。
    _ne = os.environ.get("OPD_NUM_EPOCHS")
    if _ne:
        config.num_epochs = int(_ne)
    # OPD_SMOKE=1: 换成微型数据集 + 立刻存档 + 只跑几个 epoch,用于在排队前验证任意配置能跑通。
    # 不产出研究数据。
    if os.environ.get("OPD_SMOKE") == "1":
        _tiny = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/testdata/pickscore"
        config.dataset = _tiny
        for _tc in config.train.get("teachers", []):
            _tc["dataset"] = _tiny
            _tc["prompt_fn"] = "general_ocr"
            _tc["proxy_reward"] = {"pickscore": 1.0}
        config.save_freq = 1
        config.eval_freq = 0          # 0 = 完全跳过在线 eval(smoke 只验证能跑通)
        config.num_epochs = int(os.environ.get("OPD_SMOKE_EPOCHS", 2))
        config.run_name = "[SMOKE] " + config.run_name
        config.save_dir = f"logs/{config.run_name}"
        config.save_eval_dir = f"log_save_images/{config.run_name}"
    nproc = int(os.environ.get("OPD_NPROC", "8"))
    if nproc == 8:
        return config
    # 真正的要求是"缩放后 nbpe 为整数",而非 8 % nproc == 0(后者过严)。
    # samples/epoch = bs x nproc x nbpe 必须恒为 72,micro-batch 总数 nproc x nbpe 必须恒为 24:
    #   4 卡 -> nbpe=6 (4x6=24);  3 卡 -> nbpe=8 (3x8=24);  2 卡 -> nbpe=12。
    # 三者 micro-batch 数、每 micro-batch 样本数、总样本数完全一致 ⇒ DDP 平均后梯度相同。
    # 放宽此断言是为了在 SLURM 只给到 3 卡时仍能跑出与 4 卡可比的结果(集群持续饱和,
    # --exclusive 的预计启动时间为 3 天后)。
    _tot_mb = config.sample.num_batches_per_epoch * 8      # 8 卡基准下的 micro-batch 总数
    assert _tot_mb % nproc == 0, (
        f"OPD_NPROC={nproc} 无法保持 micro-batch 总数 {_tot_mb} 不变")
    config.sample.num_batches_per_epoch = _tot_mb // nproc
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    n_t = len(config.train.teachers) if "teachers" in config.train else 1
    assert config.sample.num_batches_per_epoch % n_t == 0, (
        f"缩放后 num_batches_per_epoch={config.sample.num_batches_per_epoch} 必须能被 teacher 数 {n_t} 整除")
    return config


def _itopd(mode, tag, **kw):
    """IT-OPD: Implicit-Teacher OPD —— 不训练 expert,在 student 访问的每个 state 上就地解出 teacher。

    基于 sopd_pickscore 以保证与"真 Aes teacher 蒸馏"严格 apples-to-apples(同数据集/同 rollout 设置)。
    teacher LoRA 仍然加载:IT-OPD 不用它算 loss,但用它做 **对齐诊断**
    (log itopd_cos_real_teacher = cos(virtual teacher 方向, 真 RL teacher 方向)) —— 这是论文
    "expensive RL expert 只是在摊销一个可以在线合成的 local policy-improvement field" 的核心证据。

    对照组: sopd_pickscore(真 teacher, pickscore 0.924) / sopd_vteacher_pick(MVP-1, 0.859)
            / base_gs4.5(0.863)。MVP-1 锚在 v_base 所以退化到 base;IT-OPD 锚在 μ_θ。
    """
    config = sopd_pickscore()
    config.train.itopd = True
    config.train.itopd_mode = mode
    for k, v in kw.items():
        setattr(config.train, "itopd_" + k, v)
    config.num_epochs = 1000
    config.run_name = f"[IT-OPD-{tag} + SD3.5-M + PickScore(no expert)]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_itopd_search():
    """Version B(黑盒): K 个候选 transition -> reward 加权 -> improved mean。可用于不可微 reward。"""
    return _itopd("search", "SEARCH", k=4, tau=0.05, explore=0.05, step=0.05, nsteps=2)


def sopd_itopd_grad():
    """Version A(可微): reward 对候选 latent 的梯度方向 + 信赖域。与 MVP-1 的差别只在锚点与步长归一。"""
    return _itopd("grad", "GRAD", step=0.05, nsteps=2)


def sopd_itopd_search_k8():
    """消融: 候选数 K=4 -> 8(search 质量 vs compute)。"""
    return _itopd("search", "SEARCH-K8", k=8, tau=0.05, explore=0.05, step=0.05, nsteps=2)


def sopd_itopd_search_nogate():
    """消融: 关掉 advantage 门控 —— 验证"只在局部确有改进时才学"是否必要。"""
    return _itopd("search", "SEARCH-NOGATE", k=4, tau=0.05, explore=0.05, step=0.05,
                  nsteps=2, gate=False)


def sopd_itopd_align_aes():
    """**同 reward 对照**:Aes teacher(用 aesthetic reward 训出) ↔ aesthetic 梯度。
    专门用来判定 IT-OPD 的立论基础 —— itopd_cos_real_teacher 若仍 ≈0,
    说明真 RL teacher 方向并不是"局部 reward 上升方向",核心论点不成立。"""
    config = _itopd("grad", "ALIGN-AES", step=0.05, nsteps=2)
    config.train.itopd_reward_fn = "aesthetic"
    config.run_name = "[IT-OPD-ALIGN-AES + SD3.5-M + Aesthetic(matched)]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_itopd_diag():
    """**纯诊断**:逐 timestep 测量真 RL teacher 的增量幅度与其同 reward 梯度方向的对齐度。
    不修改训练目标(照常用真 teacher),只产出 diag_cos_j* / diag_tdelta_j*。
    回答:teacher 在哪些噪声层发力?那些层的对齐度是多少?(上一轮只测最低噪声的 2 步,可能选错了位置)"""
    config = _itopd("grad", "DIAG", step=0.05, nsteps=99)
    config.train.itopd_reward_fn = "aesthetic"      # 与 Aes teacher 同 reward
    config.train.itopd_diag_only = True
    config.run_name = "[IT-OPD-DIAG + SD3.5-M + Aesthetic]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ 新方向 A: Realizability-Aware OPD (student 锚定的 alpha 插值) ============
# 核心问题:完整 teacher target 是否超出 student 当前可实现范围? alpha<1 更好即为证据。
# 与 poe 的区别:poe 锚 base(v_base + lam*(v_T-v_base)),这里锚 student(mu_S + alpha*(mu_T-mu_S))。
def _realiz(alpha, tag):
    config = mopd()
    config.train.opd_alpha = alpha
    config.num_epochs = 800
    config.run_name = f"[REALIZ-A{tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_realiz_a050():
    return _realiz(0.50, "0.50")


def mopd_realiz_a075():
    return _realiz(0.75, "0.75")


def mopd_realiz_a125():
    return _realiz(1.25, "1.25")


# ============ 新方向 B: On-Manifold Search OPD (teacher-free) ============
# 诊断结论:像素空间 reward 梯度与 teacher 方向基本正交(只解释 0.0004% 能量)。
# 候选改为同 prompt 的 K 条真实 rollout —— 天然落在模型可实现的流形上。
def _onm(K, tag, **kw):
    config = sopd_pickscore()
    config.train.onmanifold = True
    # ⚠️ 结构性约束:on-manifold 搜索需要"从同一状态分叉出多个候选"。
    #   same_latent=True  -> 同 prompt 的 K 个副本共享初始 latent(create_generator 按 prompt 播种)
    #   noise_level>0     -> SDE 采样,K 条轨迹才会分叉;若保持 ODE(noise_level=0)则 K 条完全相同,零候选
    # 代价:论文自己的消融与我们重跑的 anneal arm 都显示 ODE > SDE。所以本 arm 实际检验的是
    #      "搜索带来的收益能否盖过改用 SDE 的损失",必须与同 noise_level 的对照组比,而非与 ODE 基线比。
    config.sample.same_latent = True
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = K
    # 分组采样器要求 train_batch_size % K == 0
    config.sample.train_batch_size = K * 2
    config.train.batch_size = config.sample.train_batch_size
    for k, v in kw.items():
        setattr(config.train, "onm_" + k, v)
    config.num_epochs = 800
    config.run_name = f"[ONMANIFOLD-K{tag} + SD3.5-M + PickScore(no expert)]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_onmanifold_k4():
    return _onm(4, "4", tau=1.0, step=0.10)


def sopd_onmanifold_k8():
    return _onm(8, "8", tau=1.0, step=0.10)


def sopd_sde_control():
    """on-manifold 的**同条件对照组**:一样用 SDE(noise_level=0.7)+same_latent+K=4 的 batch 设置,
    但不开 onmanifold。用来把"SDE 的损失"与"搜索的收益"分开 —— 否则 on-manifold 对比 ODE 基线
    的差异里混了两个变量,无法归因。"""
    config = sopd_pickscore()
    config.sample.same_latent = True
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = 4
    config.sample.train_batch_size = 8
    config.train.batch_size = 8
    config.num_epochs = 800
    config.run_name = "[SDE-CONTROL + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_diag2():
    """Diagnostic 2: teacher correction 的 alpha 探针(纯测量)。
    判据:R(alpha) 峰值在 alpha<1 -> 完整 teacher 步局部过量,支持 Realizability-Aware OPD;
         单调上升 -> teacher target 不过量,该方向证据不足。"""
    config = sopd_pickscore()
    config.train.diag2_alpha = True
    # 每个 (alpha, j) 组合都要走完剩余轨迹并解码打分,显存/时间随组合数线性增长。
    # A40-48G 实测 OOM;缩到 5 个 alpha x 2 个噪声层,并投大显存卡。
    config.train.diag2_alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    config.train.diag2_js = [5, 8]
    config.num_epochs = 6
    config.eval_freq = 10 ** 9
    config.run_name = "[DIAG2-ALPHA + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def get_config(name):
    return _scale_to_nproc(globals()[name]())


# ============ 新方向 D: Spatio-Temporal Expert OPD (STE-OPD) ============
# 动机: 现行 loss 对 H*W 个 token 取均值, 一个"只关心小区域"的 teacher(OCR 只关心文字区)
# 其信号被 ~1/N 稀释。STE 在每个 (timestep, token) 上对 M 个 teacher 残差做 softmax 路由,
# 再塌缩回 K=1(与 round-robin mopd 的 loss 尺度完全一致, 因此可与基线直接比较)。
# ste_tau -> inf 退化为均匀平均(= uniform mopd); -> 0 为逐位置 winner-take-all。
def _ste(mode, tau, norm, tag):
    config = mopd()
    config.train.ste = True
    config.train.ste_mode = mode
    config.train.ste_tau = tau
    config.train.ste_norm = norm
    config.run_name = f"[MOPD-STE-{tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_ste_spatial():
    """主臂: 逐 token 路由(z-score 归一, tau=1.0 => 软路由)。"""
    return _ste("spatial", 1.0, "zscore", "spatial")


def mopd_ste_hard():
    """消融: tau=0.3, 接近逐位置 winner-take-all。"""
    return _ste("spatial", 0.3, "zscore", "hard")


def mopd_ste_temporal():
    """消融: 只按 timestep 路由(空间上均匀), 用于区分"时间专精"与"空间专精"各贡献多少。"""
    return _ste("temporal", 1.0, "zscore", "temporal")


def mopd_ste_smoke():
    config = mopd_ste_spatial()
    config.num_epochs = 2
    config.eval_freq = 10 ** 9
    config.save_freq = 10 ** 9
    return config


# ============ 新方向 E: Ephemeral Expert / BootstrapOPD ============
# 动机(与我们已有的正向结果自洽): 固定 teacher 的残差随 student 逼近而衰减,学习信号自然枯竭,
# student 原理上无法超过 teacher。POE(把残差放大 lambda=1.5)之所以是目前唯一真正推动 Pareto 前沿
# 的做法,正是因为它粗暴地"比 teacher 走得更远"。BootstrapOPD 用正确的方式做同一件事:
# 每一轮都从**当前 student**出发重新跑一小段单任务 RL,得到"用完即弃"的微专家作为新 teacher,
# 于是改进方向始终是 on-policy 且可达的,过程可以持续爬升而不被原 teacher 的水平封顶。
#
# 可证伪的观测量: realiz_dT_rel = ||mu_T - mu_S|| / ||mu_S||。
#   固定 teacher  -> 随训练单调衰减(信号枯竭);
#   ephemeral 专家 -> 每轮刷新后回升、跨轮大致持平。
# 这里把 opd_alpha 设为 1.0(数学恒等: mu_S + 1.0*(mu_T - mu_S) = mu_T, 不改变目标),
# 仅仅为了打开该指标的日志。
def mopd_bootstrap():
    config = mopd()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    config.train.opd_alpha = 1.0          # identity;只为开启 realiz_dT_rel 日志
    _tag = os.environ.get("BOOT_TAG", "r0")
    config.run_name = f"[MOPD-BOOTSTRAP-{_tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ 新方向 F: Multi-Step / Horizon OPD ============
# 动机: OPD 的 KL 是逐步的,只保证"这一步速度对齐",而我们真正在乎的是终点。
# 让 teacher 从 student 状态自己滚 H 步,把 H 步位移按 sigma 比例折算成当前这一步的目标,
# 等价于一阶匹配 teacher 的 H 步 flow map:方向被 H 步平均、抑制了单步噪声。
# H=1 恒等于基线,因此 H 是一个干净的连续消融轴。代价: teacher 前向 x H。
def _horizon(h):
    config = mopd()
    config.train.horizon = h
    config.run_name = f"[MOPD-HORIZON-H{h} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_horizon_h2():
    return _horizon(2)


def mopd_horizon_h3():
    return _horizon(3)


# ============ 新方向 G: Continual Capability OPD ============
# 动机: teacher 是 round-robin 轮转的,轮到 OCR 时没有任何力量维持刚学会的 GenEval 能力,
# 于是能力在访问间隔里被挤掉(多任务蒸馏的灾难性遗忘)。
# 做法: 每 retain_every 个 epoch 把 student 自己存一份慢照,在 loss 里加
#   retain_w * ||mu_S - mu_snapshot||^2
# 把已获得的能力钉住(LwF 式自蒸馏锚)。retain_w=0 时零开销、基线逐位不变。
def _retain(w, every, tag):
    config = mopd()
    config.train.retain_w = w
    config.train.retain_every = every
    config.run_name = f"[MOPD-RETAIN-{tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_retain_w02():
    return _retain(0.2, 100, "w0.2")


def mopd_retain_w05():
    return _retain(0.5, 100, "w0.5")


def mopd_retain_w02_fast():
    """更短的锚滞后(30 epoch),锚更接近当下 -> 约束更弱但更贴合最新能力。"""
    return _retain(0.2, 30, "w0.2f")


# ============ 诊断: Bootstrap 前提 (teacher 信号是否随训练枯竭) ============
# BootstrapOPD 的前提是"固定 teacher 的残差随 student 逼近而衰减,于是学习信号枯竭、
# student 被 teacher 水平封顶"。这个前提可以**直接测**,不必先花 26 小时训练:
# 在同一批 prompt 上,对不同训练进度的 student checkpoint 测
#     realiz_dT_rel = ||mu_T - mu_S|| / ||mu_S||
# 单调衰减 -> 前提成立;基本持平 -> 前提错,Bootstrap 大概率无效。
# opd_alpha=1.0 是恒等变换(mu_S + 1.0*(mu_T - mu_S) = mu_T),只为打开该指标的日志;
# learning_rate=0 让它成为纯测量,不会改变被测的 checkpoint。
def mopd_gapprobe():
    config = mopd()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    config.train.opd_alpha = 1.0
    config.train.learning_rate = 0.0
    config.save_freq = 10 ** 9
    _tag = os.environ.get("PROBE_TAG", "x")
    config.run_name = f"[GAPPROBE-{_tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ 诊断: ephemeral 专家的信号强度 ============
# gapprobe 已经量出:固定 teacher 的相对信号从 4.48%(step1) 衰减到 1.34%(step601),
# 且逐段衰减率快速收敛(-54%/-27%/-12%),平台约 1.2-1.3% —— 即 teacher 从未失效。
# 于是 BootstrapOPD 能否成立,就归结为一个可直接测的问题:
#   从**同一个已收敛的 student** 出发跑一小段 RL 得到的 ephemeral 专家,
#   它的 gap 能否超过固定 teacher 剩下的 1.34%?
# 超不过 => 没有提供更多信号,那 26 小时的 arm 大概率白跑。
# 单 teacher(pickscore)以隔离测量:mopd 是三 teacher 轮转,logged 值会被平均掉。
def sopd_gapprobe():
    config = sopd_pickscore()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    config.train.opd_alpha = 1.0          # 恒等,只为打开 realiz_dT_rel 日志
    config.train.learning_rate = 0.0      # 纯测量
    config.save_freq = 10 ** 9
    _tag = os.environ.get("PROBE_TAG", "x")
    config.run_name = f"[EPHGAP-{_tag} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ 诊断: CAGrad 为什么会全轴崩坏 ============
# 实测 cagrad 相对基线整体分数 -7.4%(12 倍噪声带),且六个轴无一幸免 —— 不是权衡,是破坏。
# 假设: MGDA 的 min-norm 解退化,把权重塌到某个 teacher(或给出近零的合成方向),
# 于是 student 收到的信号既偏又弱。cagrad_w_max 直接反映这一点:
#   ~1/M(=0.33) 表示接近均匀(退化为 mopd);  ~1.0 表示塌到单个 teacher。
# lr=0 使其成为纯测量;PROBE_BS 把 batch 降到 1,让 41GB 的探针能塞进 44GB 的 A40。
def mopd_cagrad_probe():
    config = mopd_cagrad()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    config.train.opd_alpha = 1.0
    config.train.learning_rate = 0.0
    config.save_freq = 10 ** 9
    _bs = int(os.environ.get("PROBE_BS", "0"))
    if _bs > 0:
        config.sample.train_batch_size = _bs
        config.train.batch_size = _bs
    _tag = os.environ.get("PROBE_TAG", "x")
    config.run_name = f"[CAGRADPROBE-{_tag} + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ 诊断: G_construct (重做 teacher-free,用正确判据) ============
# 背景: 我们此前用 cos(reward 梯度, teacher delta) ≈ 0.003 ≈ 随机,判定 teacher-free 不成立。
# 但 DiffusionOPSD(arXiv 2608.24646)在他们的低噪声查询点上测 NFT 终点方向的同一个量得到 -0.0002,
# 同样接近零 —— 说明"与终点/teacher 方向正交"是终点式目标的性质,不构成对 reward 梯度的否定。
# 正确判据是 fixed-suffix 构造增益:
#     G_construct = F_q(y_+) - F_q(y_0)
#     F_q(y) = R(decode(把干净图预测 y 插进查询点、其余步用冻结策略跑完))
# 该探针在**单个低噪声查询点**(sigma≈0.278,OPSD 的 SD3.5-M 取值)上同时测四个方向,
# 共享同一后缀因此可直接相减: noop / reward 梯度 / 半径匹配随机 / teacher 方向。
# 最后一条是 OPSD 没有的对照 —— 直接回答"teacher 的方向本身是否 reward-improving"。
# lr=0 使其成为纯测量。
def sopd_gcprobe():
    config = sopd_pickscore()
    config.train.gc_probe = True
    config.train.gc_sigma = 0.278
    config.train.gc_rho = 0.10
    config.train.gc_nascent = 2
    # reward 的选择会影响 teacher 那一列的可解释性,而我们并不确知发布的 AesTeacher 用的是什么:
    # baselines 里的 _aes() 是 {pickscore, clipscore, hpsv2} 等权混合(不是 LAION aesthetic),
    # 而 OPD 配置给该 teacher 槽位指定的 proxy_reward 是 pickscore。两个都跑,避免猜。
    config.train.itopd_reward_fn = os.environ.get("GC_REWARD", "pickscore")
    config.train.learning_rate = 0.0
    config.save_freq = 10 ** 9
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    _tag = os.environ.get("PROBE_TAG", "x")
    config.run_name = f"[GCPROBE-{_tag} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def mopd_gcprobe():
    """多 teacher 版 G_construct 探针。单 teacher 时 tproj 退化为 teacher 方向本身,
    必须用多 teacher 才能检验"reward 梯度在 teacher 子空间内的投影"是否优于简单求和。"""
    config = mopd()
    config.train.gc_probe = True
    config.train.gc_sigma = 0.278
    config.train.gc_rho = 0.10
    config.train.gc_nascent = 2
    # 必须开: 否则每个 batch 只有一个 teacher(按数据集轮转),tproj 退化为 teacher 方向本身,
    # 门控什么都测不到(1034417 就栽在这里,gc_ntproj=1)。
    config.train.gc_all_teachers = True
    config.train.gc_ksteps = int(os.environ.get("GC_KSTEPS", "4"))
    config.train.gc_klr = float(os.environ.get("GC_KLR", "0.02"))
    config.train.itopd_reward_fn = os.environ.get("GC_REWARD", "pickscore")
    config.train.learning_rate = 0.0
    config.save_freq = 10 ** 9
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    if _init:
        config.train.lora_path = _init
    # GC_TAG: 并发跑多个 gcprobe(不同 OPD_STUDENT_INIT)时必须给各自独立的 save_dir。
    # 否则它们共享 `logs/[SMOKE] [GCPROBE-M ...]`,而 smoke 脚本启动时会清理该目录 ——
    # 实测 gcraw121/gcraw151 并发时互删对方的 train_state.pt.tmp,导致原子替换 FileNotFoundError。
    # 默认空串 => 与既有行为完全一致(向后兼容)。
    _gtag = os.environ.get("GC_TAG", "").strip()
    config.run_name = f"[GCPROBE-M{('-' + _gtag) if _gtag else ''} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_gcprobe_k():
    """G_construct 探针 + RTC-K 可实现方向。与 grad/rand/teacher 严格半径匹配,
    直接回答: 限制在模型可达集内还能构造出多少 reward 增益?"""
    config = sopd_gcprobe()
    config.train.gc_ksteps = 4
    config.train.gc_klr = 0.02
    config.run_name = "[GCPROBE-K + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def sopd_gcprobe_aes():
    """同上,但用 LAION aesthetic 作为 reward(GC_REWARD 的另一个取值)。"""
    os.environ.setdefault("GC_REWARD", "aesthetic")
    config = sopd_gcprobe()
    config.train.itopd_reward_fn = "aesthetic"
    _tag = os.environ.get("PROBE_TAG", "x")
    config.run_name = f"[GCPROBE-AES-{_tag} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


# ============ RTC: Realizable Target Construction ============
# 出发点是 DiffusionOPSD(arXiv 2608.24646) 自己量化并列为开放问题的那个洞:
#     G_realized = G_construct - G_fit
#     实测 G_construct = +0.03511 而 G_realized = +0.00046  ->  ~98.7% 的增益在拟合阶段丢失。
# 诊断: 目标 y+ = y0 + h*ghat 在**图像空间**构造,不考虑网络能否产生它。rank-32 LoRA 在 y 空间
#   只能触及 range(J) (J = dy_theta/dtheta);ghat 落在 range(J)^perp 的分量无法实现,只会让
#   实际位移旋转(OPSD 实测 off-direction drift = 0.371)。
# 方法: 方向换成 Gauss-Newton 预条件方向 d = J J^T g,天然落在可实现子空间内。
# 一阶保证: A = JJ^T (PSD),单步实现增益 ∝ <g, A·dir>
#     dir = g/||g||      ->  g^T A g / ||g||
#     dir = Ag/||Ag||    ->  ||Ag||
#   由 Cauchy-Schwarz  ||Ag||·||g|| >= <g, Ag>,故预条件方向的实现增益**严格不小于**原方向,
#   等号仅当 g 是 A 的特征向量。提升倍数 = ||Ag||·||g||/<g,Ag> = g 与模型响应算子的失配度。
# 代价: 一次反向(u = J^T g) + 一次扰动前向(d = J u,有限差分)。OPSD 每步 16 次 diffusion 反向,
#   加这两次仍远低于 DiffusionNFT 的 144 次。
def _rtc(precond, tag, **kw):
    config = sopd_pickscore()          # 单 teacher 配置只为复用其数据/打分管线;RTC 不用 teacher
    config.train.rtc = True
    config.train.rtc_precond = precond
    # 显式写出,便于在日志的配置 dump 里核对(训练脚本侧也有同样的默认值)
    config.train.rtc_sigma = 0.278      # OPSD 的 SD3.5-M 低噪声查询点
    config.train.rtc_rho = 0.10         # 信赖域 = rho*||y0||
    config.train.rtc_nascent = 2
    config.train.rtc_beta = 0.5
    config.train.rtc_dual = True
    config.train.rtc_margin = 0.0
    config.train.rtc_eps = 1e-3
    config.train.itopd_reward_fn = os.environ.get("RTC_REWARD", "pickscore")
    for k, v in kw.items():
        setattr(config.train, "rtc_" + k, v)
    config.run_name = f"[RTC-{tag} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_gn():
    """主臂: Gauss-Newton 预条件方向(本方法)。"""
    return _rtc("gn", "GN")


def rtc_k():
    """主臂(修订版): 可实现目标 = 沿 reward 梯度真实走 K 步后的模型输出。
    有限差分估 JJ^T g 在 bf16 前向下不可行(输出分辨率 ~4e-3,0.1% 扰动被舍入抹掉),
    K 步前瞻不做相近量相减,且目标是真实模型输出,天然可实现。"""
    return _rtc("ksteps", "K", ksteps=4, klr=0.02)


def rtc_kbest_nocap():
    """验证臂: 与 rtc_kbest 相同,但**取消信赖域上限**(位移用满 rho=0.10)。
    依据: curve_n12(n=8) 实测沿最优迭代点方向线性外推到 r=0.10 得 +0.0355,
    明显高于停在最优迭代点自身半径的 +0.0218 ⇒ 上限可能是次优的。
    判据: acc_p 与 kbest_gain 是否优于带上限的版本。"""
    config = _rtc("ksteps", "KBNOCAP", ksteps=6, klr=0.02)
    config.train.rtc_kbest = True
    config.train.rtc_kcap = False        # 关闭 budget=min(budget,|dir|)
    config.train.rtc_curve = True
    return config


def rtc_kbest():
    """RTC-K 修订版: K=6 给线搜索留余量,按**实测 reward** 逐样本取最优迭代点,
    信赖域改为上限(不再把位移强行拉伸到 rho*|y0|)。
    判据: acc_p 必须显著 > 0.5(此前固定 K=4 时为 0.31,因为第 4 步已越过极值)。"""
    config = _rtc("ksteps", "KBEST", ksteps=6, klr=0.02)
    config.train.rtc_kbest = True
    config.train.rtc_curve = True
    return config


def rtc_kb_train():
    """RTC-K 训练臂(主臂)。smoke 判据已过: acc_p=1.000(修复前 0.31)、
    kbest_gain=+0.031、rtc_loss/grad_norm 非零、global_step 正常前进。
    训练时关掉 rtc_curve(每 batch 多 10 次解码,纯诊断开销)。"""
    config = _rtc("ksteps", "KB", ksteps=6, klr=0.02)
    config.train.rtc_kbest = True
    config.train.rtc_curve = False
    config.train.rtc_measure_realized = False   # 单步 dy 落在 16 位量化噪声以下,测不出真值
    config.num_epochs = 800
    return config


def rtc_kb_dtheta():
    """机制直测臂: 记录 cos(MSE梯度, dtheta) 与 |梯度|/|dtheta|。
    dtheta = K 步上升的总参数位移(该位移**确实**产生了目标 y*,故目标完全可实现)。
    理论: MSE 梯度 = -J^T J dtheta ⇒ 更新是 dtheta 被 J^T J 预条件后的结果,而非 dtheta 本身。
    判据: 若 cos 显著 <1 且 |梯度|/|dtheta| 很小,则"y 空间可实现 != 拟合能到达"被直接证实。"""
    config = rtc_kb_train()
    config.train.rtc_measure_dtheta = True
    config.run_name = "[RTC-KB-DTHETA + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_kb_fixed2():
    """判别臂: 关掉 kbest、固定 K=2(实测最优步),使**全批共用同一个 dtheta**。

    动机: mech_drift 显示 y_theta 与 y0 漂移恰为 0,但 cos(更新,dtheta) 仍散布。
    查明真因是: kbest **逐样本**选最优迭代点(k_i 各不相同),而 K 步上升更新的是
    **全局共享的 theta** ⇒ 一个 batch 的目标对应**不同的参数设置**,
    不存在单一 dtheta 能同时产生它们。"可实现"只对单样本成立、对 batch 不成立。

    本臂让全批共用同一个 dtheta,则"目标 = y_{theta+dtheta}"精确成立。
    判据: 若 cos(更新, dtheta) 显著为正且接近 1 ⇒ 理论成立,
          批内目标冲突就是 RTC-K 失败的真因;
          若 cos 仍≈0 ⇒ 另有原因。"""
    config = rtc_kb_train()
    config.train.rtc_kbest = False       # 不逐样本选,全批用第 K 步
    config.train.rtc_ksteps = 2          # 固定 2 步(asc 曲线实测峰值位置)
    config.train.rtc_measure_dtheta = True
    config.run_name = "[RTC-KB-FIXED2 + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_kb_lr1e4():
    """诊断臂: kb 但把 lr 从 3e-4 降到 1e-4。
    依据: 实测 kb 的 grad_norm 是 raw 的约 3 倍(0.0020 vs 0.0007,均未触及 max_grad_norm=1.0),
    同 lr 下 kb 每步走得远约 3 倍 —— 与"快速下探→在更差点震荡→平台更低"的形态一致。
    判据: 若 kb@lr1e-4 追平或超过 raw,则病因是步长而非目标构造本身。"""
    config = rtc_kb_train()
    config.train.learning_rate = 1e-4
    config.run_name = "[RTC-KB-LR1E4 + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_raw_train():
    """对照臂: 等价 OPSD 的原始 reward 梯度方向构造。与 rtc_kb_train 只差方向是否可实现,
    差值可直接归因到"目标是否落在模型可达集内"。"""
    config = _rtc("raw", "RAWB")
    config.train.rtc_curve = False
    config.train.rtc_measure_realized = False
    config.num_epochs = 800
    # 换种子必须换 save_dir,否则与种子42 的 run 共用目录互删 checkpoint
    # (rtc_raw_warm 已有同样处理;这里补上,使 rtcwarm43 的对照也能是同种子)。
    # 默认种子 42 时 run_name 与原来逐字相同 -> 不影响已有的 [RTC-RAWB ...] 结果。
    _sd = os.environ.get("OPD_SEED", "42")
    if _sd != "42":
        config.run_name = f"[RTC-RAWB-s{_sd} + SD3.5-M + PickScore]"
        config.save_dir = f"logs/{config.run_name}"
        config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_gn_train():
    """★主臂训练版: Gauss-Newton 预条件方向 JJ^T g。与 rtc_raw_train 逐项相同,
    只差 precond("gn" vs "raw") -> 差值可直接归因到"是否把目标投到模型可达子空间"。
    注意: gn 方向用有限差分估 JVP,其数值有效性今晚**未被证实**(eps 扫描出现自相矛盾),
    仅确认默认精度下 rtc_cos_gn_g 稳定在 ~0.015 且 acc_p/acc_m 有实测值。"""
    config = _rtc("gn", "GNB")
    config.train.rtc_curve = False
    config.train.rtc_measure_realized = False
    # 必须用**精确 JVP** 作为方向,不能用有限差分:
    #   - fd + rtc_fd_fp32=True(默认): 关掉 TF32 后 max|v1-v0| 恒为 0 -> 方向是零向量(实测 1034599);
    #   - fd + rtc_fd_fp32=False:      被舍入噪声主导(cos(d_eps,d_10eps)=0.34,本该 ~1)。
    # 精确 JVP 原先因 flash-attention 无二阶导而失败,已通过强制 SDPA math 后端修复。
    config.train.rtc_jvp = "exact"
    config.num_epochs = 800
    return config


def rtc_gn_fwd():
    """★主臂(forward-mode 版): 与 rtc_gn_train 逐项相同,只把 J·u 的算法从
    二阶反向换成 **forward-mode AD**。

    二阶反向在本模型上恒产出 NaN(anomaly 定位到 NativeLayerNormBackwardBackward0 /
    BmmBackward0 / SafeSoftmaxBackward0 三处),fp32、unwrapped DDP、手写 LayerNorm
    三种修法均无效。forward-mode 一次前向直接得 J·u,不经过二阶反向。
    离线验证(code/0012_test_forward_mode_jvp.py, job 1036024):在 SD3.5-M 上成功,
    且与有限差分对拍 cos=+0.999728。
    训练中每 epoch 记 `rtc_fwd_vs_fd`,持续监控这一致性。

    与 `rtc_raw_train`(已有 10 点评测的对照)仍然只差 precond -> 差值可直接归因。
    """
    config = rtc_gn_train()
    config.train.rtc_jvp = "fwd"
    config.run_name = "[RTC-GNFWD + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def codex_mopd_continue():
    """**论文自己的方法**作为同预算基线:从发布的 Student 继续做 MOPD 多 teacher 蒸馏。

    codex 0023 §5 指出的缺口:目前只有
      (1) 发布 Student(锚点/初始化)
      (2) Student -> DiffusionOPSD-direct(`crd`,机制 control)
      (3) Student -> **MOPD/DiffusionOPD-continue** —— **一直缺**
      (4) Student -> 朴素 frozen-Student KL(`crkl5`)
      (5) 当前 fixed one-sided same-state hinge(`crg`)
    没有 (3),"守卫有效"就只能和自己的消融比,审稿人第一个问题就是
    "那和论文自己的方法比呢"。

    与 `codex_cr_prod` **逐项相同**,只差方法本身:
      - 同一个 `mopd()` 基础(三个真实 per-slot 数据集、同 teacher、同 guidance);
      - 同 `get_config` 出口的 `_scale_to_nproc` ⇒ **算力预算逐位一致**
        (bs x nproc x nbpe = 72,gas = nbpe);
      - 同起点(发布 Student)、同 epoch 数、同 save_freq、同 eval_freq=0;
      - **不开** rtc,**不开** constrained_retention ⇒ 就是论文方法本身。
    """
    config = mopd()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    if not _init:
        raise ValueError("codex_mopd_continue requires OPD_STUDENT_INIT (released Student LoRA)")
    _run_dir = os.environ.get("CODEX_CR_RUN_DIR", "").strip()
    if not _run_dir:
        raise ValueError("codex_mopd_continue requires CODEX_CR_RUN_DIR (持久 artifact 目录)")
    config.train.lora_path = _init
    # 显式关掉两条我们自己的机制,确保这一臂就是论文方法(默认即关,写出来是为了可核查)
    config.train.rtc = False
    config.train.constrained_retention = False
    config.run_name = "[MOPD-CONTINUE + SD3.5-M + PickScore+OCR+GenEval]"
    config.save_dir = _run_dir
    config.save_eval_dir = os.path.join(_run_dir, "eval_images")
    _ds = [t.get("dataset", "") for t in config.train.teachers]
    if len(set(_ds)) != len(_ds):
        raise ValueError(f"task slot 数据集重复,配置无效: {_ds}")
    return config


def codex_cr_prod():
    """GuardedOPSD 的**正式**配置(取代被误用的 codex_cr_smoke)。

    2026-08-29:0010 的 200-epoch 跑误用了 `codex_cr_smoke()`,该配置把三个 task slot
    全部指向 `testdata/pickscore`(256 条),导致 OCR/GenEval 约束从未在各自任务分布上
    被测量(codex 0014 复核指出)。本配置继承 `mopd()` 的**真实 per-slot 数据集**:
        pickscore -> dataset/pickscore (25432 条)
        ocr       -> dataset/ocr
        geneval   -> dataset/geneval  (prompt_fn=geneval)
    只叠加 constrained-retention 所需的开关,不改动任何数据/任务设置。
    """
    config = mopd()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    if not _init:
        raise ValueError("codex_cr_prod requires OPD_STUDENT_INIT (released Student LoRA)")
    config.train.lora_path = _init
    # CODEX-0018: 这套 RTC 开关原先只写在 codex_cr_smoke 里。GuardedOPSD = RTC-raw 直推 reward
    # + retention 约束,少了它 trainer 会直接抛 "constrained_retention requires rtc target
    # construction"。此前启动行写死 smoke,prod 从未真正被执行过,这个缺失一直没暴露。
    config.train.rtc = True
    config.train.rtc_precond = "raw"
    config.train.rtc_sigma = 0.278
    config.train.rtc_rho = 0.10
    config.train.rtc_nascent = 2
    config.train.rtc_beta = 0.5
    config.train.rtc_dual = True
    config.train.rtc_margin = 0.0
    config.train.itopd_reward_fn = "pickscore"
    config.train.constrained_retention = True
    config.train.cr_reward_teacher_idx = 0          # pickscore = 被推的 reward
    config.train.cr_retention_teacher_indices = [1, 2]   # ocr, geneval = 能力约束
    config.train.cr_rel_tol = float(os.environ.get("CODEX_CR_REL_TOL", "0.02"))
    config.train.cr_abs_tol = float(os.environ.get("CODEX_CR_ABS_TOL", "1e-8"))
    config.train.cr_direct_weight = float(os.environ.get("CODEX_CR_DIRECT_WEIGHT", "1.0"))
    config.train.cr_retention_weight = float(os.environ.get("CODEX_CR_RETENTION_WEIGHT", "1.0"))
    _w = config.train.cr_retention_weight
    config.run_name = f"[CR-PROD-w{_w:g} + SD3.5-M + PickScore+OCR+GenEval]"
    # CODEX-0018: 必须落到持久的 artifact 目录。相对路径会解析到 $WORK_DIR(mktemp 临时目录),
    # checkpoint 随作业结束一起消失。smoke 用的就是 CODEX_CR_RUN_DIR,prod 沿用。
    _run_dir = os.environ.get("CODEX_CR_RUN_DIR", "").strip()
    if not _run_dir:
        raise ValueError("codex_cr_prod requires CODEX_CR_RUN_DIR (持久 artifact 目录)")
    config.save_dir = _run_dir
    config.save_eval_dir = os.path.join(_run_dir, "eval_images")
    # 三个 slot 的数据集必须互不相同 —— 这正是 0010 出错之处,在配置层就断言死
    _ds = [t.get("dataset", "") for t in config.train.teachers]
    if len(set(_ds)) != len(_ds):
        raise ValueError(f"task slot 数据集重复,配置无效: {_ds}")
    return config


def rtc_rand_diag():
    """随机方向诊断对照:与 rtc_gn_fwd_diag 同 rho/nasc,只把方向换成随机。
    用途:检验"小 rho 下 GN 的 acc_p=0.92 是不是仅因步子太小"。
    若随机方向在同 rho 下只有 ~0.5,说明 GN 方向确实在上升(一阶保证成立),
    只是在可用步长下被曲率淹没 —— 这正是本项目要区分的"方向有信息"与"步长可用"。
    """
    config = rtc_raw_train()
    config.train.rtc_precond = "rand"
    _rho = float(os.environ.get("RHO", "0.10"))
    config.train.rtc_rho = _rho
    config.run_name = f"[RTC-RAND-DIAG-r{_rho:g} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_raw_diag():
    """raw 方向的诊断对照:与 rtc_raw_train 逐项相同,只换 run_name + 支持 RHO/NASC。
    **必须换 run_name**:rtc_raw_train 的目录已训到 781,直接跑会 OPD_RESUME=auto
    接上 checkpoint-781,拿不到 epoch 0~2(今晚已在 GN 主跑上踩过同一个坑)。
    用途:给 GN 的 acc_p 对比提供**同种子(42)、同 epoch**的对照 ——
    此前用的 RAWB-s43 是跨种子的。
    """
    config = rtc_raw_train()
    _rho = float(os.environ.get("RHO", "0.10"))
    config.train.rtc_rho = _rho
    _nasc = int(os.environ.get("NASC", str(getattr(config.train, "rtc_nascent", 2))))
    config.train.rtc_nascent = _nasc
    config.run_name = f"[RTC-RAW-DIAG-r{_rho:g}-n{_nasc} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_gn_fwd_diag():
    """诊断专用:与 rtc_gn_fwd 完全相同,只把 run_name 换掉。
    目的是**不与主跑共用 save_dir** —— 同名会让主跑在重启时 resume 到诊断跑留下的
    checkpoint(2026-08-29 已实测踩过一次,日志毫无异常)。
    """
    config = rtc_gn_fwd()
    # RHO: 信赖域半径,用于检验"acc_p 低是否因步长过大被曲率主导"。
    # run_name **必须随 rho 变** —— 并发实例共享 save_dir 会互删 checkpoint
    # (本文件上方 rho 扫描处的注释记载已栽过两次)。
    _rho = float(os.environ.get("RHO", "0.10"))
    config.train.rtc_rho = _rho
    # NASC: 上升步数。读码可知 _dir 在 n_ascent 循环内**不重算**,
    # 且 _step*n_ascent = _budget,故终点与单步相同 -> 该旋钮对"多步小位移"无效。
    # 这里加进来是为了**实测验证**这一点(同 rho 下 n=1 与 n=5 应给出相同 acc_p),
    # 而不是只凭读码断言。
    _nasc = int(os.environ.get("NASC", str(getattr(config.train, "rtc_nascent", 2))))
    config.train.rtc_nascent = _nasc
    config.run_name = f"[RTC-GNFWD-DIAG-r{_rho:g}-n{_nasc} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_raw_meas():
    """★ 直接验证 OPSD 立论的核心量: G_realized / G_construct。
    OPSD 报 G_construct=+0.03511 而 G_realized=+0.00046(98.7% 丢在拟合阶段),
    整条 RTC 主线都建立在这个洞上 —— 但**我们从未独立测过它**:
    `rtc_measure_realized` 的实现一直在,却被 rtc_raw_train/rtc_gn_train 显式设为 False,
    今晚所有运行都没开过。
    本臂用 OPSD 的原始构造(raw)同时打开 realized 测量与 reward-位移曲线,
    直接读出比值。判据: 若比值远大于 1.3%,则"98.7% 丢失"这个前提在本复现下不成立,
    RTC 的立论基础需要重新评估。"""
    config = rtc_raw_train()
    config.train.rtc_measure_realized = True
    config.train.rtc_curve = True
    # RHO: 扫描信赖域半径。核心未知量是 **G_realized 随 rho 如何变化** ——
    # 目前只有 rho=0.10 一个点(G_con=0.04975, G_real=0.000202, 仅实现 0.41%)。
    #   若 realized 随 rho **饱和** -> rho=0.10 是浪费,小 rho + 多步拟合能实现更多(便宜的赢法);
    #   若 realized 随 rho **线性增长** -> 瓶颈是方向而非幅度,必须做 Gauss-Newton 拟合步。
    # run_name 必须随 rho 变:并发跑多个实例若共享 save_dir 会互删 checkpoint(gcprobe 上栽过)。
    _rho = float(os.environ.get("RHO", "0.10"))
    config.train.rtc_rho = _rho
    # run_name 必须同时含 rho 与 WF 模式:否则 frozen 与 adaptive 在同一 rho 下共享 save_dir,
    # smoke 脚本启动时的清理会互删 checkpoint(今晚已在 gcprobe / rho 扫描上各栽一次)。
    _wf = os.environ.get("RTC_WF_MODE", "adaptive")
    _wfs = "" if _wf == "adaptive" else f"-{_wf}"
    # WFC_TAG: 同 rho 同 WF 模式再跑一次(如换硬件复现)时用于区分 save_dir,同 GC_TAG 的用法。
    _tag = os.environ.get("WFC_TAG", "").strip()
    _tags = f"-{_tag}" if _tag else ""
    config.run_name = f"[RTC-RAW-MEAS-r{int(round(_rho*1000)):03d}{_wfs}{_tags} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_raw_warm():
    """★ 检验「teacher 项应被调度」这一 observation 的可操作版本。

    依据: G_construct 探针实测 teacher 方向相对随机的优势在训练早期即变号 ——
    在**训练良好**的 rtcraw 臂上 step 61 已到零/略负(-0.00058), 241 时 -0.00883(15/15 同号);
    而 grad-rand 全程保持强正。=> teacher 只在训练最初携带上升信息。

    可操作推论: **先用 teacher 蒸馏做短暂 warmup, 再切到 reward 梯度**。
    本臂用 `OPD_STUDENT_INIT` 从 teacher 臂的早期 checkpoint 初始化, 其余与 `rtc_raw_train`
    **逐项相同** -> 与已完成 10 点评测的 rtcraw3 直接配对比较, 差值可归因到 warmup。

    重点看 OCR: rtcraw 从零训练时 OCR 由 0.438 退化到 0.198;
    若 warmup 版在同等 pickscore 下保住 OCR, 即为该 idea 的直接证据。
    """
    config = rtc_raw_train()
    _init = os.environ.get("OPD_STUDENT_INIT", "").strip()
    config.train.lora_path = _init if _init else None
    _tag = os.environ.get("WARM_TAG", "w61")
    _sd = os.environ.get("OPD_SEED", "42")
    if _sd != "42": _tag = f"{_tag}s{_sd}"   # 换种子必须换 save_dir,否则互删 checkpoint
    config.run_name = f"[RTC-RAW-WARM-{_tag} + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_kcurve():
    """诊断臂: 测 reward 随位移的曲线(可实现方向 vs 原始梯度方向)。
    acc_p=0.31 提示 rho=0.10 过大,本臂直接读出各方向的最优半径与增益。"""
    config = _rtc("ksteps", "KCURVE", ksteps=4, klr=0.02)
    config.train.rtc_curve = True
    return config


def rtc_k8():
    """更长前瞻,用于看 K 的剂量-响应。"""
    return _rtc("ksteps", "K8", ksteps=8, klr=0.02)


def rtc_raw():
    """消融/对照: 原始 reward 梯度方向 —— 这一臂等价于 OPSD 的目标构造。
    与 rtc_gn 的唯一差别就是方向是否被预条件,因此两者之差直接归因于本方法。"""
    return _rtc("raw", "RAW")


def rtc_rand():
    """归因对照: 同信赖域的随机方向,无 reward 信息。"""
    return _rtc("rand", "RAND")


def rtc_rand_meas():
    """归因对照 + 测量: 随机方向,开 realized/curve。
    rtc_curve 同批记两条曲线 —— `rtc_dR_k_*` 用实际方向(此处=随机),`rtc_dR_g_*` 用 reward 梯度,
    故单个 run 内就有"随机 vs 梯度"的构造增益对照。"""
    config = rtc_rand()
    config.train.rtc_measure_realized = True
    config.train.rtc_curve = True
    config.run_name = "[RTC-RAND-MEAS + SD3.5-M + PickScore]"
    config.save_dir = f"logs/{config.run_name}"
    config.save_eval_dir = f"log_save_images/{config.run_name}"
    return config


def rtc_gn_nodual():
    """消融: 关掉负分支。"""
    return _rtc("gn", "GN-NODUAL", dual=False)
