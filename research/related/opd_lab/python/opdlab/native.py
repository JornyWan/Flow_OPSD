"""Run pinned upstream programs in isolated, audited source snapshots.

No reimplementation of either baseline's training algorithm lives here.
"""
from __future__ import annotations

import difflib
import copy
from contextlib import nullcontext
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from .io import ROOT, environment, git, read_json, sha256, tree_fingerprint, write_json, verify_asset_fingerprint


def clean_env(pythonpath, extra=None):
    # Do not inherit experimental algorithm flags or an old fork's PYTHONPATH.
    exact = {"HOME", "USER", "LOGNAME", "PATH", "LD_LIBRARY_PATH", "LIBRARY_PATH",
             "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "LANG", "LC_ALL", "TMPDIR", "TRITON_CACHE_DIR",
             "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "CONDA_PREFIX",
             "MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"}
    prefixes = ("SLURM_", "NCCL_", "TORCH_", "OMP_", "MKL_", "HF_", "HUGGINGFACE_",
                "TRANSFORMERS_", "XDG_")
    env = {k: v for k, v in os.environ.items() if k in exact or k.startswith(prefixes)}
    env.update(PYTHONPATH=os.pathsep.join(map(str, pythonpath)), PYTHONNOUSERSITE="1",
               WANDB_MODE="offline", TOKENIZERS_PARALLELISM="false")
    env.update(extra or {})
    return env


def verify_upstream(name, root=ROOT):
    lock = read_json(root / "upstream.lock.json")[name]
    repo = root / "upstream" / name
    if git(repo, "rev-parse", "HEAD") != lock["commit"]:
        raise ValueError(f"{name}: HEAD does not match upstream.lock.json")
    if git(repo, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError(f"{name}: upstream checkout is modified; refuse to call it pristine")
    if git(repo, "remote", "get-url", "origin") != lock["url"]:
        raise ValueError(f"{name}: unexpected origin")
    return repo, lock


def snapshot_upstream(name, dest, root=ROOT):
    repo, lock = verify_upstream(name, root)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "-C", str(repo), "archive", lock["commit"]])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
                raise ValueError("Unsafe upstream archive path")
        tar.extractall(dest)
    if name == "diffusionopsd":
        # Equivalent to setuptools' package-dir mapping; no installed old fork.
        (dest / "diffusionopsd").symlink_to("src", target_is_directory=True)
    return lock


def record_patch(path, old, new, patches, reason):
    if old == new:
        raise ValueError(f"Empty patch: {path}")
    patches.append({"file": str(path), "reason": reason,
                    "diff": "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                                       fromfile=str(path), tofile=str(path)))})
    path.write_text(new)


def patch_opd_runtime(runtime, reward_dir, managed=True):
    patches = []
    p = runtime / "flow_grpo/ocr.py"
    text = p.read_text()
    anchor = 'path = "YOUR_PDDLEOCR_PATH"'
    if text.count(anchor) != 1:
        raise ValueError("Upstream OCR binding changed; review adapter")
    record_patch(p, text, text.replace(anchor, "path = " + repr(str(Path(reward_dir) / "paddleocr"))),
                 patches, "Bind upstream's mandatory local PaddleOCR weight placeholder")
    if managed:
        p = runtime / "scripts/train_sd3_opd.py"
        old = p.read_text()
        anchor = "    while True:\n\n        pipeline.transformer.eval()"
        if old.count(anchor) != 1 or old.count("        epoch+=1\n") != 1:
            raise ValueError("Upstream loop changed; review lifecycle adapter")
        new = old.replace(anchor, "    while epoch < config.num_epochs:\n\n        pipeline.transformer.eval()")
        new = new.replace("        epoch+=1\n", """        epoch+=1
        from opdlab.native import opd_epoch_end
        opd_epoch_end(epoch, global_step, config, transformer, accelerator,
                      ema if config.train.ema else None,
                      transformer_trainable_parameters, optimizer, save_ckpt)
""")
        record_patch(p, old, new, patches,
                     "Honor explicit outer-epoch budget and invoke checkpoint saving; loss/rollout/optimizer unchanged")
    return patches


def bind_hf_assets(runtime, name, site):
    """Bind model identifier literals to pinned local payload snapshots; no scorer math edits."""
    identifiers = {"clip": "openai/clip-vit-large-patch14", "pickscore": "yuvalkirstain/PickScore_v1",
                   "pickscore_processor": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"}
    patches = []
    package = runtime / ("flow_grpo" if name == "diffusionopd" else "src")
    for path in sorted(package.rglob("*.py")):
        before = after = path.read_text()
        for key, identifier in identifiers.items():
            snapshot = site.get("auxiliary_models", {}).get(key)
            if not snapshot or "snapshots" not in Path(snapshot).parts:
                continue
            for quote in ('"', "'"):
                after = after.replace(quote + identifier + quote, repr(snapshot))
        if after != before:
            record_patch(path, before, after, patches, "Bind cached model IDs to explicit local snapshot revisions; scorer math unchanged")
    return patches


def opd_epoch_end(epoch, step, config, transformer, accelerator, ema, params, save_optimizer, save_ckpt):
    import random
    import numpy as np
    import torch
    from .io import append_jsonl
    if accelerator.is_main_process:
        append_jsonl(Path(config.save_dir) / "progress.jsonl", {"outer_epoch": epoch, "optimizer_step": step})
    if epoch != config.num_epochs and (config.save_freq <= 0 or epoch % config.save_freq):
        return
    accelerator.wait_for_everyone()
    save_ckpt(config.save_dir, transformer, step, accelerator, ema, params, config)
    target = Path(config.save_dir) / "checkpoints" / f"checkpoint-{step}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(transformer).save_pretrained(target / "raw_lora", selected_adapters=["default"])
        torch.save(save_optimizer.state_dict(), target / "optimizer.pt")
        if ema is not None:
            torch.save(ema.state_dict(), target / "ema_state.pt")
        if accelerator.scaler is not None:
            torch.save(accelerator.scaler.state_dict(), target / "scaler.pt")
        write_json(target / "metadata.json", {"outer_epoch": epoch, "optimizer_step": step,
                   "lora": "EMA" if ema is not None else "raw", "raw_lora": "raw",
                   "resume_supported_by_native_entrypoint": False})
    torch.save({"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
               target / f"rng-rank-{accelerator.process_index}.pt")
    accelerator.wait_for_everyone()


def load_config_file(path, preset):
    spec = importlib.util.spec_from_file_location("_opdlab_native_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config(preset)


def set_config(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    part = int(parts[-1]) if isinstance(node, list) else parts[-1]
    if (isinstance(node, list) and part >= len(node)) or (not isinstance(node, list) and part not in node):
        raise KeyError(f"Unknown upstream config key: {key}")
    node[part] = value


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {k: v for key, item in value.items() for k, v in flatten(item, f"{prefix}.{key}".strip(".")).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value) for k, v in flatten(item, f"{prefix}.{i}").items()}
    return {prefix: value}


def resolved_config(plan_path, audit_only=False):
    plan = read_json(plan_path)
    config = load_config_file(plan["upstream_config"], plan["preset"])
    # ConfigDict.to_dict can retain mutable dictionaries inside list values.
    # Copy before bindings, otherwise teacher-path edits disappear from audit.
    before = copy.deepcopy(config.to_dict())
    for k, v in {**plan["bindings"], **plan["overrides"]}.items():
        set_config(config, k, v)
    after = config.to_dict()
    flat_before, flat_after = flatten(before), flatten(after)
    changes = {k: {"before": flat_before.get(k), "after": v,
                    "kind": "explicit_override" if k in plan["overrides"] else "asset_or_output_binding"}
               for k, v in flat_after.items() if flat_before.get(k) != v}
    audit = {"upstream_defaults": before, "resolved": after, "changes": changes}
    audit_path = Path(plan_path).parent / "config.audit.json"
    if audit_only:
        write_json(audit_path, audit)
    elif read_json(audit_path) != json.loads(json.dumps(audit)):
        raise RuntimeError("Native config changed after preparation (including environment defaults)")
    return config


def materialize_pickapic(site, runtime):
    recipe = read_json(runtime / "data/pickapic_recipe.json")
    source = Path(site["pickapic"])
    if sha256(source / "train.txt") != recipe["paper_train_sha256"]:
        raise ValueError("Existing Pick-a-Pic train data does not match the fresh upstream recipe")
    if len((source / "train.txt").read_text().splitlines()) != recipe["paper_train_lines"]:
        raise ValueError("Pick-a-Pic line count mismatch")
    target = runtime / "data/pickapic"
    target.mkdir(exist_ok=True)
    shutil.copyfile(source / "train.txt", target / "train.txt")
    # Use fresh upstream DrawBench bytes, not an unverified legacy test.txt.
    shutil.copyfile(runtime / "data/drawbench/test.txt", target / "test.txt")


def subset_native_smoke(runtime, name, limit):
    if type(limit) is not int or limit < 1:
        raise ValueError("Smoke example count must be positive")
    paths = ([runtime / "dataset" / task / filename for task in ("pickscore", "ocr", "geneval")
              for filename in (("train_metadata.jsonl", "test_metadata.jsonl") if task == "geneval"
                               else ("train.txt", "test.txt"))] if name == "diffusionopd" else
             [runtime / "data/pickapic" / filename for filename in ("train.txt", "test.txt")])
    changes = []
    for path in paths:
        old_hash = sha256(path)
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(lines[:limit]))
        changes.append({"path": str(path.relative_to(runtime)), "before_sha256": old_hash,
                        "before_lines": len(lines), "after_sha256": sha256(path),
                        "after_lines": min(limit, len(lines))})
    return {"scope": "prefix_subset_for_engineering_smoke_only", "examples_per_split": limit,
            "selection": "first N upstream rows, no reward-based selection", "files": changes}


def prepare_native(name, preset, run, site, overrides, world_size=8, literal=False, full_hash=False,
                   smoke_examples=None, smoke_recipe_world_size=None, nodes=1):
    if world_size < 1 or nodes < 1 or world_size % nodes:
        raise ValueError("Positive world_size must be divisible by positive node count")
    processes_per_node = world_size // nodes
    if smoke_recipe_world_size is not None:
        if name != "diffusionopsd" or smoke_examples is None or smoke_recipe_world_size < 1:
            raise ValueError("Recipe world-size override is limited to explicitly labelled OPSD data-subset smoke runs")
        required = {"sample.train_batch_size", "sample.num_image_per_prompt", "sample.num_batches_per_epoch",
                    "train.batch_size", "train.gradient_accumulation_steps"}
        if required - overrides.keys():
            raise ValueError("Smoke recipe bootstrapping requires explicit overrides for every batch/group setting")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    runtime = run / "upstream"
    lock = snapshot_upstream(name, runtime)
    framework = run / "framework/python"
    shutil.copytree(ROOT / "python/opdlab", framework / "opdlab", ignore=shutil.ignore_patterns("__pycache__"))
    (runtime / "reward_ckpts").symlink_to(site["reward_ckpts"], target_is_directory=True)
    patches = []
    bindings = {"pretrained.model": site["model"], "save_dir": str(run / "artifacts"),
                "logdir": str(run / "wandb")}
    (run / "artifacts").mkdir()
    if name == "diffusionopd":
        patches = patch_opd_runtime(runtime, site["reward_ckpts"], not literal)
        # This module is dependency-light and only resolves ml_collections config.
        previous = Path.cwd()
        try:
            os.chdir(runtime)
            config = load_config_file(runtime / "config/opd.py", preset)
        finally:
            os.chdir(previous)
        bindings["save_eval_dir"] = str(run / "eval_images")
        for i, teacher in enumerate(config.train.teachers):
            bindings[f"train.teachers.{i}.lora_path"] = site["teachers"][teacher["name"]]
        config_file = runtime / "config/opd.py"
        script = runtime / "scripts/train_sd3_opd.py"
        write_json(run / "accelerate.json", {
            "compute_environment": "LOCAL_MACHINE", "distributed_type": "MULTI_GPU" if world_size > 1 else "NO",
            "mixed_precision": overrides.get("mixed_precision", config.mixed_precision),
            "use_cpu": False, "debug": False, "num_processes": world_size, "num_machines": nodes, "machine_rank": 0,
            "main_training_function": "main", "rdzv_backend": "static", "same_network": True,
            "dynamo_config": {"dynamo_backend": "NO"}})
        launcher = [site["python"], "-m", "accelerate.commands.launch", "--config_file", str(run / "accelerate.json"),
                    "--num_processes", str(world_size)]
        if world_size > 1:
            launcher.append("--multi_gpu")
        if nodes > 1:
            launcher += ["--num_machines", str(nodes), "--machine_rank", "{node_rank}",
                         "--main_process_ip", "{master_addr}", "--main_process_port", "{master_port}"]
        extra_env = {}
    else:
        materialize_pickapic(site, runtime)
        config_file = runtime / "config/public.py"
        script = runtime / "scripts/train_opsd_ri_sd3.py"
        launcher = [site["python"], "-m", "torch.distributed.run", f"--nnodes={nodes}",
                    f"--nproc_per_node={processes_per_node}"]
        if nodes == 1:
            launcher.append("--standalone")
        else:
            launcher += ["--node_rank={node_rank}", "--master_addr={master_addr}", "--master_port={master_port}"]
        extra_env = {"PUBLIC_N_GPUS": str(world_size), "PUBLIC_POLICY_WORLD_SIZE": str(smoke_recipe_world_size or world_size),
                     "PUBLIC_LAUNCH_WORLD_SIZE": str(world_size), "REWARD_CKPT_PATH": site["reward_ckpts"]}
    patches.extend(bind_hf_assets(runtime, name, site))
    if "resume_from" in overrides and overrides["resume_from"] and name == "diffusionopd":
        raise ValueError("Fresh native OPD does not implement resume_from; do not silently restart")
    extra_env.update(site.get("environment", {}))
    (run / "wandb").mkdir(exist_ok=True)
    extra_env["WANDB_DIR"] = str(run / "wandb")
    data_transform = subset_native_smoke(runtime, name, smoke_examples) if smoke_examples is not None else None
    write_json(run / "data_transform.json", data_transform)
    plan = {"name": name, "preset": preset, "upstream": lock, "upstream_config": str(config_file),
            "smoke_recipe_world_size": smoke_recipe_world_size,
            "bindings": bindings, "overrides": overrides, "world_size": world_size,
            "nodes": nodes, "processes_per_node": processes_per_node,
            "runtime_mode": "literal_upstream" if literal else "managed_lifecycle",
            "cwd": str(runtime), "pythonpath": [str(framework), str(runtime)], "extra_env": extra_env}
    write_json(run / "plan.json", plan)
    wrapper = run / "config.py"
    wrapper.write_text("from opdlab.native import resolved_config\n"
                       f"def get_config(_unused=None):\n    return resolved_config({str(run / 'plan.json')!r})\n")
    env = clean_env(plan["pythonpath"], extra_env)
    subprocess.run([site["python"], "-c", "from opdlab.native import resolved_config; "
                    "from opdlab.io import environment, write_json; "
                    f"resolved_config({str(run / 'plan.json')!r}, audit_only=True); "
                    f"write_json({str(run / 'environment.json')!r}, environment())"],
                   cwd=runtime, env=env, check=True)
    if name == "diffusionopsd":
        resolved = read_json(run / "config.audit.json")["resolved"]
        k = resolved["sample"]["num_image_per_prompt"]
        if k < 2 or resolved["sample"]["train_batch_size"] * world_size % k:
            raise ValueError("Native OPSD requires >=2 images per prompt and a divisible global sampling batch")
    plan["command"] = launcher + [str(script), f"--config={wrapper}"]
    write_json(run / "plan.json", plan)
    write_json(run / "runtime_patches.json", patches)
    assets = {"model": tree_fingerprint(site["model"], full=full_hash),
              "reward_ckpts": tree_fingerprint(site["reward_ckpts"], full=full_hash)}
    assets["auxiliary_models"] = {k: tree_fingerprint(v, full=full_hash) for k, v in site.get("auxiliary_models", {}).items()}
    if name == "diffusionopd":
        assets["teachers"] = {k: tree_fingerprint(v, full=True) for k, v in site["teachers"].items()}
    resolved = read_json(run / "config.audit.json")["resolved"]
    datasets = ({teacher["name"]: teacher["dataset"] for teacher in resolved["train"]["teachers"]}
                if name == "diffusionopd" else {"train_and_test": resolved["dataset"]})
    # Explicit dataset overrides may point outside the source snapshot. Audit those bytes too.
    assets["configured_datasets"] = {key: tree_fingerprint(value, full=True) for key, value in datasets.items()}
    write_json(run / "manifest.json", {"status": "prepared_not_trained", "source": lock,
               "data_transform": data_transform,
               "assets": assets, "data": tree_fingerprint(runtime / ("dataset" if name == "diffusionopd" else "data/pickapic"), full=True),
               "framework": tree_fingerprint(framework, full=True),
               "runtime": tree_fingerprint(runtime, full=False), "environment": read_json(run / "environment.json"),
               "launch_inputs": {p: tree_fingerprint(run / p, full=True)
                                 for p in ["plan.json", "config.py", "config.audit.json", "accelerate.json", "data_transform.json"]
                                 if (run / p).exists()},
               "native_algorithm_reimplemented": False})
    return run


def resolve_node_launch(plan, env=None, hosts=None):
    env = os.environ if env is None else env
    nodes = plan.get("nodes", 1)
    if nodes == 1:
        return list(plan["command"]), 0
    if (int(env.get("SLURM_JOB_NUM_NODES", env.get("SLURM_NNODES", "0"))) != nodes
            or int(env.get("SLURM_NTASKS", "0")) != nodes):
        raise ValueError("Multi-node native launch needs one srun task per allocated node")
    rank = int(env["SLURM_PROCID"])
    if not 0 <= rank < nodes or int(env.get("SLURM_LOCALID", "0")) != 0:
        raise ValueError("Expected one launcher per node, with contiguous node ranks")
    if hosts is None:
        hosts = subprocess.check_output(["scontrol", "show", "hostnames", env["SLURM_JOB_NODELIST"]], text=True).splitlines()
    if len(hosts) != nodes:
        raise ValueError("Allocated hostname count does not match plan")
    substitutions = {"{node_rank}": str(rank), "{master_addr}": hosts[0],
                     "{master_port}": str(15000 + int(env["SLURM_JOB_ID"]) % 40000)}
    command = []
    for argument in plan["command"]:
        for token, value in substitutions.items():
            argument = argument.replace(token, value)
        command.append(argument)
    return command, rank


def run_native(run):
    run = Path(run).resolve()
    plan = read_json(run / "plan.json")
    command, node_rank = resolve_node_launch(plan)
    manifest = read_json(run / "manifest.json")
    for name, expected in manifest.get("launch_inputs", {}).items():
        if tree_fingerprint(run / name, full=True) != expected:
            raise ValueError(f"Native launch input was modified: {name}")
    if tree_fingerprint(run / "framework/python", full=True) != manifest["framework"]:
        raise ValueError("Native framework snapshot was modified")
    if tree_fingerprint(run / "upstream", full=False) != read_json(run / "manifest.json")["runtime"]:
        raise ValueError("Native source snapshot changed after preparation")
    def verify_asset(value):
        if "inventory_digest" in value:
            note = verify_asset_fingerprint(value)
            if note:
                from .io import append_jsonl
                append_jsonl(run / "asset_validation.jsonl", note)
        else:
            for nested in value.values():
                verify_asset(nested)
    for value in read_json(run / "manifest.json")["assets"].values():
        verify_asset(value)
    subprocess.run([plan["command"][0], "-c", "from opdlab.io import read_json, assert_environment; "
                    f"assert_environment(read_json({str(run / 'manifest.json')!r})['environment'])"],
                   env=clean_env(plan["pythonpath"], plan["extra_env"]), check=True)
    marker = run / (f"STARTED-node-{node_rank}" if plan.get("nodes", 1) > 1 else "STARTED")
    with marker.open("x") as f:
        f.write("Do not reuse a started run directory.\n")
    start = time.perf_counter()
    local_tmp = (tempfile.TemporaryDirectory(prefix=f"opdlab-node{node_rank}-", dir="/tmp")
                 if plan.get("nodes", 1) > 1 else nullcontext(None))
    with local_tmp as temporary:
        env = clean_env(plan["pythonpath"], plan["extra_env"])
        if temporary:
            env.update(TMPDIR=temporary, TRITON_CACHE_DIR=str(Path(temporary) / "triton"))
        write_json(run / f"launch-node-{node_rank}.json", {"command": command, "node_rank": node_rank,
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "nodes": plan.get("nodes", 1)})
        result = subprocess.run(command, cwd=plan["cwd"], env=env)
    seconds = time.perf_counter() - start
    exit_record = {"returncode": result.returncode, "wall_seconds": seconds,
                   "configured_gpu_hours": seconds * plan["world_size"] / 3600,
                   "node_rank": node_rank, "cost_note": "Use Slurm allocation accounting for total job cost"}
    write_json(run / f"exit-node-{node_rank}.json", exit_record)
    if node_rank == 0:
        write_json(run / "exit.json", exit_record)
    if result.returncode:
        raise SystemExit(result.returncode)
    if plan["name"] == "diffusionopsd" and node_rank == 0:
        config = read_json(run / "config.audit.json")["resolved"]
        for checkpoint in (Path(config["save_dir"]) / "checkpoints").glob("checkpoint-*"):
            if (checkpoint / "lora/adapter_config.json").is_file():
                write_json(checkpoint / "opdlab_checkpoint.json",
                           {"lora": "EMA" if config["train"]["ema"] else "raw",
                            "source": plan["upstream"], "basis": "pinned native save_ckpt and resolved config"})


def native_eval_plan(run, output, checkpoint, dataset, steps=40, guidance=None, rewards="pickscore,clipscore,hpsv2"):
    """Expose the upstream evaluation protocol separately from unified evaluation."""
    run, output = Path(run).resolve(), Path(output).resolve()
    plan = read_json(run / "plan.json")
    output.mkdir(parents=True, exist_ok=False)
    model = plan["bindings"]["pretrained.model"]
    python = plan["command"][0]
    checkpoint = str(Path(checkpoint).resolve()) if checkpoint else ""
    if plan["name"] == "diffusionopd":
        # New eval-only copy makes model path binding reviewable, without changing
        # the prepared training snapshot or native evaluation math.
        source = Path(plan["cwd"]) / "scripts/evaluation.py"
        text = source.read_text()
        anchor = '"stabilityai/stable-diffusion-3.5-medium"'
        if text.count(anchor) != 1:
            raise ValueError("Upstream evaluation model binding changed")
        bound = text.replace(anchor, repr(model))
        script = output / "evaluation.py"
        script.write_text(bound)
        (output / "model_binding.patch").write_text("".join(difflib.unified_diff(text.splitlines(True), bound.splitlines(True))))
        command = [python, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(script),
                   "--model_type=sd3", f"--dataset={dataset}", f"--output_dir={output / 'results'}",
                   f"--num_inference_steps={steps}", f"--guidance_scale={4.5 if guidance is None else guidance}",
                   "--mixed_precision=fp16", "--resolution=512"]
        if checkpoint:
            # Native checkpoint_path expects a root containing lora/.
            command.append(f"--checkpoint_path={checkpoint}")
    else:
        if dataset != "drawbench":
            raise ValueError("Native OPSD public evaluation uses DrawBench; use unified eval for custom constraints")
        command = [python, str(Path(plan["cwd"]) / "scripts/cross_eval.py"),
                   "--config_file", str(run / "config.py"), "--config", plan["preset"],
                   "--model", model, "--ckpt", checkpoint, "--out", str(output / "results.json"),
                   "--prompts", str(Path(plan["cwd"]) / "data/pickapic/test.txt"),
                   "--num_steps", str(steps), "--sampler", "flow", "--rewards", rewards,
                   "--pickscore_scale", "training", "--dump_per_image"]
        if guidance is not None:
            command.extend(["--guidance_scale", str(guidance)])
    evaluation = {"command": command, "cwd": plan["cwd"], "pythonpath": plan["pythonpath"],
                  "extra_env": plan["extra_env"], "protocol": "native_upstream_not_unified",
                  "checkpoint": checkpoint, "upstream": plan["upstream"]}
    write_json(output / "plan.json", evaluation)
    return evaluation
