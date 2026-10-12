"""Downloadable-weight public configurations for DiffusionOPSD.

Examples:
    --config config/public.py:sd35_hpsv2
    --config config/public.py:zimage_clipscore

Run ``python scripts/prepare_pickapic_prompts.py`` once before loading a
configuration.  The public launchers do this automatically.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_PATH = Path(__file__).with_name("api.py")
_SPEC = importlib.util.spec_from_file_location("_diffusionopsd_public_api", API_PATH)
API = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(API)

PUBLIC_REWARDS = {
    "hpsv2", "clipscore", "pickscore", "aesthetic", "imagereward", "hpsv3", "deqa"
}
HEAVY_ZIMAGE_REWARDS = {"hpsv3", "deqa"}


def get_config(name: str):
    """Return ``<backbone>_<reward>`` or the SD3.5 joint ``sd35_open3`` preset."""

    try:
        backbone, reward = name.split("_", 1)
    except ValueError as exc:
        raise ValueError(
            f"Expected <backbone>_<reward>, e.g. sd35_hpsv2; got {name!r}"
        ) from exc

    heavy_zimage = backbone == "zimage" and reward in HEAVY_ZIMAGE_REWARDS
    default_policy_world = "6" if heavy_zimage else "8"
    n_gpus = int(os.environ.get("PUBLIC_POLICY_WORLD_SIZE", os.environ.get("PUBLIC_N_GPUS", default_policy_world)))
    if n_gpus < 1:
        raise ValueError(f"PUBLIC_POLICY_WORLD_SIZE must be positive, got {n_gpus}")
    if heavy_zimage:
        if os.environ.get("ZIMAGE_HEAVY_DIFF_BRIDGE", "0") != "1":
            raise ValueError(
                f"zimage/{reward} requires ZIMAGE_HEAVY_DIFF_BRIDGE=1; "
                "use scripts/train_public.sh or follow README.md#reward-model-setup"
            )
        launch_world = int(os.environ.get("PUBLIC_LAUNCH_WORLD_SIZE", str(n_gpus + 1)))
        if launch_world != n_gpus + 1:
            raise ValueError("heavy Z-Image needs one reward-server rank in addition to policy ranks")

    if backbone == "sd35":
        if reward == "open3":
            base_reward = "pickscore"
        elif reward in PUBLIC_REWARDS:
            base_reward = reward
        else:
            raise ValueError(f"Unsupported public SD3.5 reward: {reward}")

        config = API.sd3_reward_config(base_reward, dataset="pickscore", n_gpus=n_gpus)
        # ── 4× batch 臂(EXPERIMENT_LOG §4.107 第 3 条):把每 epoch 的 prompt 数
        # 从 48 提到 PUBLIC_NUM_GROUPS。`num_groups` 在 nft.py 里是硬编码常量,
        # 但它**只经 num_batches_per_epoch 起作用**,故在此(已声明的 fork 文件)覆盖批参数即可,
        # 不必再 fork nft.py。约束与 nft.py:73-83 完全相同:
        #   total % (n_gpus*bs) == 0  且  bs*n_gpus % num_image_per_prompt == 0
        # 选加 prompt 而非加每 prompt 图数:§4.81 实测 4× 算力下 192×24 给 85.9% 信号占比、
        # 48×96 只给 78.7%;且 96 图/prompt 在上述约束下需 bs=32,超过求解器上限 9,无解。
        _ng = int(os.environ.get("PUBLIC_NUM_GROUPS", "0"))
        if _ng:
            _nimg = int(config.sample.num_image_per_prompt)
            _total = _ng * _nimg
            for _bs in range(9, 0, -1):
                if _total % (n_gpus * _bs) == 0 and (_bs * n_gpus) % _nimg == 0:
                    _nb = _total // (n_gpus * _bs)
                    config.sample.train_batch_size = _bs
                    config.sample.num_batches_per_epoch = _nb
                    config.train.batch_size = _bs
                    config.train.gradient_accumulation_steps = _nb
                    break
            else:
                raise ValueError(
                    f"PUBLIC_NUM_GROUPS={_ng} x {_nimg} 在 {n_gpus} 卡下无可行 batch_size(<=9)")
        params = API.sd35m_opsd_default_params()
        if reward == "open3":
            config.reward_fn = {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0}
            # PickScore is normalized by /26 in its scorer, matching the paper's
            # PickScore/26 + CLIPScore + HPSv2.1 objective.
            config.beta = 0.1
            params["opa_reward_kind"] = "open3"
        elif reward == "imagereward":
            # Paper recipe for the memory-heavy SD3.5-M ImageReward target pass.
            params["opa_mb"] = 1
        elif reward == "deqa":
            # DeQA is an 8B mPLUG-Owl2 scorer; the paper uses one target per
            # differentiable reward micro-batch on SD3.5-M.
            params["opa_mb"] = 1
        API.attach_opsd(config, params)

    elif backbone == "zimage":
        if reward not in PUBLIC_REWARDS:
            raise ValueError(f"Unsupported public Z-Image reward: {reward}")
        config = API.zimage_config("opsd", reward=reward, n_gpus=n_gpus)

    else:
        raise ValueError(f"Unsupported backbone {backbone!r}; choose sd35 or zimage")

    config.dataset = str(ROOT / "data" / "pickapic")
    default_updates = 300 if name == "sd35_open3" else 100
    config.num_epochs = int(os.environ.get("PUBLIC_NUM_UPDATES", str(default_updates)))
    config.save_freq = int(os.environ.get("PUBLIC_SAVE_FREQ", "10"))
    config.eval_freq = 0
    config.run_name = f"public_{name}"
    config.save_dir = str(ROOT / "outputs" / name)
    config.logdir = str(ROOT / "outputs" / "wandb")
    # ── A/B 对照用的固定种子(RPI_SEED 未设时与 upstream 行为完全一致)──────────
    # ⚠️ 不能用 `--config.seed=N` 覆盖:`config.seed` 默认是 None,而 ml_collections 的
    # config_flags **不支持覆盖 NoneType 字段**(实测 FATAL:"Type <class 'NoneType'>
    # of field config.seed is not supported for overriding")。只能在构造时设。
    _rpi_seed = os.environ.get("RPI_SEED", "").strip()
    if _rpi_seed:
        config.seed = int(_rpi_seed)

    return config
