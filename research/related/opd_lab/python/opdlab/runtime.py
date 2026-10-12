"""Prepare immutable run inputs, then execute the snapshotted framework."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .config import Experiment
from .data import load_records
from .io import ROOT, environment, read_json, sha256, tree_fingerprint, write_json, verify_asset_fingerprint
from .native import bind_hf_assets, clean_env, patch_opd_runtime, snapshot_upstream


def prepare(config, run, site, world_size=1, full_hash=False):
    config.validate()
    if config.initial_adapter:
        config.initial_adapter = str(Path(config.initial_adapter).expanduser().resolve())
    load_records(config.dataset, config.split)  # Validate the FULL split manifest before copying.
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if config.selection_protocol in {"semantic_batch_v1", "semantic_veto_v1"} and world_size != 1:
        raise ValueError("Semantic batch v1 supports one process only")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=False)
    framework = run / "framework/python"
    shutil.copytree(ROOT / "python/opdlab", framework / "opdlab", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(config.dataset, run / "data.jsonl")
    config.dataset = str(run / "data.jsonl")
    if config.training_manifest:
        from .training_queries import validate_training_manifest
        validate_training_manifest(read_json(config.training_manifest), config,
                                   load_records(config.dataset, config.split), world_size)
        shutil.copyfile(config.training_manifest, run / "training_queries.json")
        config.training_manifest = str(run / "training_queries.json")
    if config.diagnostic_manifest:
        from .diagnostic import validate_manifest
        validate_manifest(read_json(config.diagnostic_manifest), config, load_records(config.dataset, config.split))
        shutil.copyfile(config.diagnostic_manifest, run / "diagnostic_queries.json")
        config.diagnostic_manifest = str(run / "diagnostic_queries.json")
    if config.selection_reference:
        shutil.copyfile(config.selection_reference, run / "count_reference.json")
        config.selection_reference = str(run / "count_reference.json")
    assets = {}
    lock, patches = None, []
    if config.backend == "sd3":
        if site is None:
            raise ValueError("SD3 requires --site")
        lock = snapshot_upstream("diffusionopd", run / "upstream")
        (run / "upstream/reward_ckpts").symlink_to(site["reward_ckpts"], target_is_directory=True)
        patches = patch_opd_runtime(run / "upstream", site["reward_ckpts"], managed=False)
        patches.extend(bind_hf_assets(run / "upstream", "diffusionopd", site))
        assets = {"model": tree_fingerprint(site["model"], full=full_hash),
                  "reward_ckpts": tree_fingerprint(site["reward_ckpts"], full=full_hash),
                  "teachers": {k: tree_fingerprint(site["teachers"][k], full=True) for k in config.teachers}}
        assets["auxiliary_models"] = {k: tree_fingerprint(v, full=full_hash)
                                       for k, v in site.get("auxiliary_models", {}).items()}
        if config.initial_adapter:
            assets["initial_adapter"] = tree_fingerprint(config.initial_adapter, full=True)
    elif config.initial_adapter:
        raise ValueError("Toy initial_adapter is unsupported; use explicit checkpoint resume")
    write_json(run / "config.json", config.as_dict())
    write_json(run / "site.json", site)
    write_json(run / "runtime_patches.json", patches)
    interpreter = site["python"] if site else sys.executable
    subprocess.run([interpreter, "-c", "from opdlab.io import environment, write_json; "
                    f"write_json({str(run / 'environment.json')!r}, environment())"],
                   env=clean_env([framework], site.get("environment", {}) if site else {}), check=True)
    source = tree_fingerprint(framework, full=True)
    manifest = {"format": 1, "status": "prepared", "algorithm": (
                "local_intervention_semantic_veto_v1" if config.selection_protocol == "semantic_veto_v1" else
                "local_intervention_semantic_batch_v1" if config.selection_protocol == "semantic_batch_v1"
                else "local_intervention_v1"),
                "native_baseline": False, "world_size": world_size, "upstream": lock,
                "framework": source, "data": tree_fingerprint(run / "data.jsonl", full=True),
                "assets": assets, "environment": read_json(run / "environment.json"), "full_weight_hashes": full_hash}
    manifest["resolved_config"] = tree_fingerprint(run / "config.json", full=True)
    manifest["site"] = tree_fingerprint(run / "site.json", full=True)
    if config.diagnostic_manifest:
        manifest["diagnostic_queries"] = tree_fingerprint(config.diagnostic_manifest, full=True)
    if config.training_manifest:
        manifest["training_queries"] = tree_fingerprint(config.training_manifest, full=True)
    if config.selection_reference:
        manifest["selection_reference"] = tree_fingerprint(config.selection_reference, full=True)
    if lock:
        manifest["runtime_source"] = tree_fingerprint(run / "upstream", full=False)
    if config.selection_protocol in {"semantic_batch_v1", "semantic_veto_v1"}:
        if config.selection_protocol == "semantic_veto_v1":
            from .veto_selection import BatchSelection
        else:
            from .online_selection import BatchSelection
        BatchSelection(config, load_records(config.dataset, config.split), manifest, world_size)
    write_json(run / "manifest.json", manifest)
    return run


def verify_run(run, check_assets=True):
    run = Path(run)
    manifest = read_json(run / "manifest.json")
    if tree_fingerprint(run / "framework/python", full=True) != manifest["framework"]:
        raise ValueError("Snapshotted framework was modified")
    if tree_fingerprint(run / "data.jsonl", full=True) != manifest["data"]:
        raise ValueError("Run data was modified")
    if tree_fingerprint(run / "config.json", full=True) != manifest["resolved_config"]:
        raise ValueError("Resolved run config was modified")
    if "site" in manifest and tree_fingerprint(run / "site.json", full=True) != manifest["site"]:
        raise ValueError("Run site/asset bindings were modified")
    if "diagnostic_queries" in manifest and tree_fingerprint(run / "diagnostic_queries.json", full=True) != manifest["diagnostic_queries"]:
        raise ValueError("Diagnostic query manifest was modified")
    if "training_queries" in manifest and tree_fingerprint(run / "training_queries.json", full=True) != manifest["training_queries"]:
        raise ValueError("Training query manifest was modified")
    if "selection_reference" in manifest and tree_fingerprint(run / "count_reference.json", full=True) != manifest["selection_reference"]:
        raise ValueError("Selection reference was modified")
    if "runtime_source" in manifest and tree_fingerprint(run / "upstream", full=False) != manifest["runtime_source"]:
        raise ValueError("Upstream runtime source was modified")
    audit = []
    if check_assets:
        def visit(value):
            if "inventory_digest" in value:
                note = verify_asset_fingerprint(value)
                if note:
                    audit.append(note)
            else:
                for item in value.values():
                    visit(item)
        visit(manifest["assets"])
    return audit


def prepare_matched_random(verified_run, run):
    """Prepare fresh random training from verified INPUTS; no git on compute nodes.

    Clone exact frozen source/environment/assets, not the trained checkpoint.
    The completed verified run supplies counts only. Never overwrite either run.
    """
    verified_run, run = Path(verified_run).resolve(), Path(run).resolve()
    source_config = Experiment.load(verified_run / 'config.json')
    if ((source_config.selection_protocol, source_config.method) not in {
            ('semantic_batch_v1', 'verified_fixed'), ('semantic_veto_v1', 'harm_veto')}
            or read_json(verified_run / 'result.json').get('status') != 'complete'):
        raise ValueError('Need a complete semantic verified source')
    if source_config.selection_protocol == 'semantic_veto_v1':
        from .veto_selection import BatchSelection
    else:
        from .online_selection import BatchSelection
    audit = verify_run(verified_run)
    source_manifest = read_json(verified_run / 'manifest.json')
    config = copy.deepcopy(source_config)
    config.method = 'random'
    config.selection_reference = str(verified_run / 'selection_reference.json')
    config.validate()
    BatchSelection(config, load_records(config.dataset, config.split), source_manifest, 1)
    run.mkdir(parents=True, exist_ok=False)
    for folder in ('framework', 'upstream'):
        if (verified_run / folder).is_dir():
            shutil.copytree(verified_run / folder, run / folder, symlinks=True,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('data.jsonl', 'site.json', 'environment.json', 'runtime_patches.json'):
        shutil.copyfile(verified_run / name, run / name)
    config.dataset = str(run / 'data.jsonl')
    if config.training_manifest:
        shutil.copyfile(config.training_manifest, run / 'training_queries.json')
        config.training_manifest = str(run / 'training_queries.json')
    shutil.copyfile(config.selection_reference, run / 'count_reference.json')
    config.selection_reference = str(run / 'count_reference.json')
    write_json(run / 'config.json', config.as_dict())
    manifest = copy.deepcopy(source_manifest)
    manifest['status'] = 'prepared'
    manifest['prepared_from'] = {'run':str(verified_run),
        'manifest_sha256':sha256(verified_run / 'manifest.json'),
        'scope':'Frozen inputs only; original initial_adapter and fresh Adam, no trained weights copied',
        'git_or_subprocess_required':False}
    for key, filename in {'framework':'framework/python','data':'data.jsonl',
                          'resolved_config':'config.json','site':'site.json',
                          'selection_reference':'count_reference.json'}.items():
        manifest[key] = tree_fingerprint(run / filename, full=True)
    if config.training_manifest:
        manifest['training_queries'] = tree_fingerprint(config.training_manifest, full=True)
    if 'runtime_source' in manifest:
        manifest['runtime_source'] = tree_fingerprint(run / 'upstream', full=False)
    BatchSelection(config, load_records(config.dataset, config.split), manifest, 1)
    write_json(run / 'manifest.json', manifest)
    if audit:
        write_json(run / 'asset_validation.json', audit)
    verify_run(run, check_assets=False)
    return run


def launch(run, resume=None, stop_after=None, diagnostic=False):
    run = Path(run).resolve()
    asset_audit = verify_run(run)
    if asset_audit:
        write_json(run / "asset_validation.json", asset_audit)
    site = read_json(run / "site.json")
    manifest = read_json(run / "manifest.json")
    python = site["python"] if site else sys.executable
    args = ["-m", "opdlab.worker", "--run", str(run)]
    if resume:
        args.extend(["--resume", str(Path(resume).resolve())])
    if stop_after is not None:
        args.extend(["--stop-after", str(stop_after)])
    if diagnostic:
        args.append("--diagnostic")
    world = manifest["world_size"]
    command = [python] + args if world == 1 else [python, "-m", "torch.distributed.run", "--standalone",
                            "--nnodes=1", f"--nproc_per_node={world}", "--module", "--", *args[1:]]
    # Exclusive execution lock; a crashed process leaves an explicit lock to inspect.
    lock = run / "RUNNING"
    with lock.open("x") as f:
        f.write(f"launcher_pid={os.getpid()}\n")
    try:
        if not resume and (run / "result.json").exists():
            raise ValueError("Run already has results; do not restart into the same directory")
        start = time.perf_counter()
        result = subprocess.run(command, cwd=run,
                 env=clean_env([run / "framework/python", run / "upstream"],
                               {"REWARD_CKPT_PATH": site["reward_ckpts"], **site.get("environment", {})} if site else {}))
        from .io import append_jsonl
        seconds = time.perf_counter() - start
        append_jsonl(run / "launches.jsonl", {"command": command, "resume": resume,
                     "wall_seconds": seconds, "returncode": result.returncode,
                     "allocated_gpu_hours": seconds * world / 3600
                     if read_json(run / "config.json")["backend"] == "sd3" else 0.0})
        if result.returncode:
            raise RuntimeError(f"Worker exited with code {result.returncode}; inspect run logs")
    finally:
        lock.unlink()


def launch_eval(run, output, adapter, kind, sampler, steps, seeds, split, save_images=False, python=None):
    run = Path(run).resolve()
    asset_audit = verify_run(run)
    if asset_audit:
        write_json(Path(output).parent / (Path(output).name + "_source_asset_validation.json"), asset_audit)
    site = read_json(run / "site.json")
    executable = str(Path(python).expanduser().absolute()) if python else (site["python"] if site else sys.executable)
    args = [executable, "-m", "opdlab.worker", "--run", str(run),
            "--evaluate", "--output", str(Path(output).resolve()), "--adapter-kind", kind,
            "--sampler", sampler, "--steps", str(steps), "--split", split, "--seeds", *map(str, seeds)]
    if adapter:
        args.extend(["--adapter", str(Path(adapter).resolve())])
    if save_images:
        args.append("--save-images")
    subprocess.run(args, cwd=run, env=clean_env([run / "framework/python", run / "upstream"],
                                              site.get("environment", {}) if site else {}), check=True)
