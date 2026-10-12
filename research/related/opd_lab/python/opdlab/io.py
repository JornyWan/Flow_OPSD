"""Small, explicit provenance and atomic I/O helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def tree_fingerprint(path, full=False):
    """Small files always hashed; large weights optionally fully hashed.

    Metadata inventories must never be described as content hashes.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file()
                                                and "__pycache__" not in p.parts and p.suffix != ".pyc")
    rows = []
    for p in files:
        s = p.stat()
        row = {"name": str(p.relative_to(path)) if path.is_dir() else p.name,
               "bytes": s.st_size}
        if full or s.st_size < 2_000_000:
            row["sha256"] = sha256(p)
        else:
            row["mtime_ns"] = s.st_mtime_ns
        rows.append(row)
    return {"path": str(path), "mode": "sha256" if full else "small_sha256_large_stat",
            "inventory_digest": digest(rows), "files": rows}


def verify_asset_fingerprint(expected):
    actual = tree_fingerprint(expected["path"], full=expected["mode"] == "sha256")
    if actual == expected:
        return None
    # Legacy runs inventoried whole HF caches. Transformers can add these
    # negative chat-template lookup markers while all CLIP inputs stay identical.
    before = {r["name"]: r for r in expected["files"]}
    after = {r["name"]: r for r in actual["files"]}
    added = set(after) - set(before)
    metadata_only = (Path(expected["path"]).name.startswith("models--") and added
                     and all(name.startswith(".no_exist/") and name.rsplit("/", 1)[-1]
                             in {"chat_template.json", "chat_template.jinja"} for name in added)
                     and all(after.get(name) == value for name, value in before.items()))
    if metadata_only:
        return {"path": expected["path"], "status": "legacy_chat_negative_cache_additions_only",
                "added": sorted(added), "all_prepared_files_unchanged": True}
    raise ValueError(f"Asset changed after preparation: {expected['path']}")


def environment():
    names = ["torch", "diffusers", "transformers", "peft", "accelerate", "numpy",
             "ml-collections", "paddleocr", "paddlepaddle", "paddlepaddle-gpu", "mmdet", "mmcv", "open-clip-torch",
             "deepspeed", "xformers", "timm"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def stable_seed(seed, *parts):
    return int(digest([seed, *parts])[:15], 16) % (2**31 - 1)


def assert_environment(expected):
    actual = environment()
    differences = {name: {"prepared": version, "current": actual["packages"].get(name)}
                   for name, version in expected["packages"].items()
                   if actual["packages"].get(name) != version}
    if actual["python"] != expected["python"] or differences:
        raise RuntimeError(f"Training environment changed after preparation: {differences}")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def site_config(path):
    site = read_json(path)
    required = {"python", "model", "teachers", "reward_ckpts", "pickapic"}
    if required - site.keys():
        raise ValueError(f"Missing site keys: {required - site.keys()}")
    for key in required - {"teachers"}:
        # Resolving a venv's python symlink would silently launch the base
        # interpreter and bypass the selected environment.
        p = Path(site[key]).expanduser()
        site[key] = str(p.absolute() if key == "python" else p.resolve())
    site["teachers"] = {k: str(Path(v).expanduser().resolve()) for k, v in site["teachers"].items()}
    site["auxiliary_models"] = {k: str(Path(v).expanduser().resolve())
                                for k, v in site.get("auxiliary_models", {}).items()}
    if set(site.get("environment", {})) - {"HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "HF_HUB_OFFLINE", "CUDA_HOME"}:
        raise ValueError("Site environment accepts cache/offline/CUDA toolkit paths only, not algorithm flags")
    return site
