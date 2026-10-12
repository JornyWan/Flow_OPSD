import ast
import copy
from pathlib import Path

import pytest
import torch

from opdlab.io import ROOT
from opdlab.method import transition_mean_loss
from opdlab.native import clean_env, load_config_file, patch_opd_runtime, set_config, snapshot_upstream, verify_upstream


def test_fresh_upstream_pins_are_clean():
    for name in ("diffusionopd", "diffusionopsd"):
        verify_upstream(name)


def test_native_opd_bindings_do_not_modify_algorithm(tmp_path, monkeypatch):
    runtime = tmp_path / "source"
    snapshot_upstream("diffusionopd", runtime)
    monkeypatch.chdir(runtime)
    config = load_config_file(runtime / "config/opd.py", "mopd")
    before = copy.deepcopy(config.to_dict())
    set_config(config, "pretrained.model", "/test/model")
    set_config(config, "train.teachers.0.lora_path", "/test/teacher")
    after = config.to_dict()
    before["pretrained"]["model"] = "/test/model"
    before["train"]["teachers"][0]["lora_path"] = "/test/teacher"
    assert before == after
    assert config.sample.num_steps == 10 and config.sample.eval_num_steps == 40
    assert config.sample.train_batch_size == 3
    assert config.train.gradient_accumulation_steps == 3
    with pytest.raises(KeyError):
        set_config(config, "train.lr_typo", 1)


def test_lifecycle_patch_changes_only_loop_bound_and_checkpoint_hook(tmp_path):
    runtime = tmp_path / "source"
    snapshot_upstream("diffusionopd", runtime)
    path = runtime / "scripts/train_sd3_opd.py"
    before = ast.parse(path.read_text())
    patches = patch_opd_runtime(runtime, tmp_path / "rewards")
    after = ast.parse(path.read_text())
    main_before = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    main_after = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    loops_before = [n for n in main_before.body if isinstance(n, ast.While)]
    loops_after = [n for n in main_after.body if isinstance(n, ast.While)]
    assert len(loops_before) == len(loops_after) == 1
    loop_before, loop_after = loops_before[0], loops_after[0]
    loop_after.test = loop_before.test
    loop_after.body = loop_after.body[:-2]
    assert ast.dump(before) == ast.dump(after)
    assert len(patches) == 2


def test_local_loss_matches_extracted_upstream_kernel():
    tree = ast.parse((ROOT / "upstream/diffusionopd/scripts/train_sd3_opd.py").read_text())
    # Execute the actual upstream delta, loss branch, and reductions on tensors.
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.With) and any(
        isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "delta" for t in x.targets)
        for x in n.body))
    body = loop.body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "delta" for t in n.targets))
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "distill_loss" for t in n.targets))
    kernel = ast.Module(body=body[start:end + 1], type_ignores=[])
    from types import SimpleNamespace
    generator = torch.Generator().manual_seed(10)
    z = torch.randn((2, 4, 3, 3), generator=generator)
    pred, target = torch.randn(z.shape, generator=generator), torch.randn(z.shape, generator=generator)
    dt = -0.13
    scope = {"config": SimpleNamespace(sample=SimpleNamespace(noise_level=0)),
             "prev_sample_mean": z + dt * pred, "teacher_prev_mean": (z + dt * target).unsqueeze(1)}
    exec(compile(kernel, "upstream_loss", "exec"), scope)
    torch.testing.assert_close(scope["distill_loss"], transition_mean_loss(pred, target, dt))


def test_clean_environment_drops_old_experimental_settings(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/old/modified/fork")
    monkeypatch.setenv("RPI_ENABLE", "1")
    monkeypatch.setenv("OPD_OLD_MODE", "1")
    monkeypatch.setenv("PUBLIC_NUM_UPDATES", "1")
    env = clean_env(["/new/snapshot"])
    assert env["PYTHONPATH"] == "/new/snapshot"
    assert not {"RPI_ENABLE", "OPD_OLD_MODE", "PUBLIC_NUM_UPDATES"} & env.keys()


def test_site_keeps_venv_python_symlink(tmp_path):
    from opdlab.io import site_config, write_json
    base = tmp_path / "base_python"
    base.touch()
    venv = tmp_path / "venv/bin/python"
    venv.parent.mkdir(parents=True)
    venv.symlink_to(base)
    path = tmp_path / "site.json"
    write_json(path, {"python": str(venv), "model": str(tmp_path), "teachers": {},
                     "reward_ckpts": str(tmp_path), "pickapic": str(tmp_path)})
    assert site_config(path)["python"] == str(venv)


def test_teacher_bindings_are_visible_in_native_audit(tmp_path, monkeypatch):
    from opdlab.io import read_json, write_json
    from opdlab.native import resolved_config
    runtime = tmp_path / "source"
    snapshot_upstream("diffusionopd", runtime)
    monkeypatch.chdir(runtime)
    plan = tmp_path / "plan.json"
    write_json(plan, {"upstream_config": str(runtime / "config/opd.py"), "preset": "mopd",
                      "bindings": {"train.teachers.0.lora_path": "/new/aesthetic/teacher"}, "overrides": {}})
    resolved_config(plan, audit_only=True)
    change = read_json(tmp_path / "config.audit.json")["changes"]["train.teachers.0.lora_path"]
    assert change["before"] == "YOUR_AES_TEACHER_PATH"
    assert change["after"] == "/new/aesthetic/teacher"


def test_native_smoke_subset_is_explicit_and_preserves_upstream(tmp_path):
    from opdlab.native import subset_native_smoke
    runtime = tmp_path / "source"
    snapshot_upstream("diffusionopd", runtime)
    original = (runtime / "dataset/ocr/train.txt").read_text().splitlines()
    report = subset_native_smoke(runtime, "diffusionopd", 4)
    assert report["scope"] == "prefix_subset_for_engineering_smoke_only"
    assert len(report["files"]) == 6
    assert (runtime / "dataset/ocr/train.txt").read_text().splitlines() == original[:4]
    verify_upstream("diffusionopd")


def test_smoke_recipe_bootstrap_cannot_silently_change_formal_baseline(tmp_path):
    from opdlab.native import prepare_native
    with pytest.raises(ValueError, match="limited"):
        prepare_native("diffusionopsd", "sd35_open3", tmp_path / "run", {}, {},
                       world_size=1, smoke_recipe_world_size=8)
    with pytest.raises(ValueError, match="every batch"):
        prepare_native("diffusionopsd", "sd35_open3", tmp_path / "run", {}, {},
                       world_size=1, smoke_examples=4, smoke_recipe_world_size=8)


def test_recorded_package_drift_is_rejected():
    from opdlab.io import assert_environment, environment
    expected = environment()
    assert_environment(expected)
    expected["packages"]["torch"] = "different-build"
    with pytest.raises(RuntimeError, match="environment changed"):
        assert_environment(expected)


def test_legacy_negative_cache_audit_never_allows_weight_changes(tmp_path):
    from opdlab.io import tree_fingerprint, verify_asset_fingerprint
    cache = tmp_path / "models--openai--clip-vit-large-patch14"
    cache.mkdir()
    (cache / "weights.bin").write_bytes(b"original weights")
    expected = tree_fingerprint(cache, full=True)
    negative = cache / ".no_exist/commit/chat_template.jinja"
    negative.parent.mkdir(parents=True)
    negative.touch()
    assert verify_asset_fingerprint(expected)["all_prepared_files_unchanged"]
    (cache / "weights.bin").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="Asset changed"):
        verify_asset_fingerprint(expected)


def test_hf_snapshot_binding_preserves_nonliteral_upstream_ast(tmp_path):
    from opdlab.native import bind_hf_assets
    runtime = tmp_path / "source"
    snapshot_upstream("diffusionopsd", runtime)
    files = list((runtime / "src").rglob("*.py"))
    before = {p: ast.parse(p.read_text()) for p in files}
    snapshots = {"clip": "/cache/snapshots/clip", "pickscore": "/cache/snapshots/pick",
                 "pickscore_processor": "/cache/snapshots/processor"}
    changes = bind_hf_assets(runtime, "diffusionopsd", {"auxiliary_models": snapshots})
    assert changes
    allowed = {"openai/clip-vit-large-patch14", "yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
               *snapshots.values()}
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str) and node.value in allowed:
                node.value = "PINNED_MODEL_IDENTIFIER"
            return node
    for p in files:
        assert ast.dump(Normalize().visit(before[p])) == ast.dump(Normalize().visit(ast.parse(p.read_text())))


def test_two_node_launch_keeps_total_world_and_distinct_node_ranks():
    from opdlab.native import resolve_node_launch
    plan = {"nodes": 2, "world_size": 8, "command": ["python", "--nnodes=2", "--nproc_per_node=4",
            "--node_rank={node_rank}", "--master_addr={master_addr}", "--master_port={master_port}"]}
    env = {"SLURM_JOB_NUM_NODES": "2", "SLURM_NTASKS": "2", "SLURM_LOCALID": "0", "SLURM_JOB_ID": "123"}
    first, r0 = resolve_node_launch(plan, {**env, "SLURM_PROCID": "0"}, ["gpu-a", "gpu-b"])
    second, r1 = resolve_node_launch(plan, {**env, "SLURM_PROCID": "1"}, ["gpu-a", "gpu-b"])
    assert (r0, r1) == (0, 1)
    assert "--node_rank=0" in first and "--node_rank=1" in second
    assert first[-2:] == second[-2:] == ["--master_addr=gpu-a", "--master_port=15123"]
    with pytest.raises(ValueError, match="one srun task"):
        resolve_node_launch(plan, {**env, "SLURM_NTASKS": "8", "SLURM_PROCID": "0"}, ["gpu-a", "gpu-b"])
