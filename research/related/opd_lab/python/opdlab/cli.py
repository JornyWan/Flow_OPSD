from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from .io import ROOT, environment, read_json, site_config, tree_fingerprint, write_json


def overrides(values):
    result = {}
    for value in values:
        key, raw = value.split("=", 1)
        if key in result:
            raise ValueError(f"Duplicate override: {key}")
        result[key] = json.loads(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description="Pinned native OPD/OPSD and verified local expert corrections")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("fetch", help="Fetch exact lockfile commits into NEW upstream checkouts")
    p = sub.add_parser("doctor", help="Read-only upstream, assets, and dependency audit")
    p.add_argument("--site")
    p.add_argument("--hash-weights", action="store_true")
    p.add_argument("--out")
    p = sub.add_parser("native-prepare", help="Resolve native config and prepare an isolated run, without training")
    p.add_argument("--upstream", choices=["diffusionopd", "diffusionopsd"], required=True)
    p.add_argument("--preset", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--nodes", type=int, default=1, help="Native Slurm node count; world-size is the TOTAL GPU count")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    p.add_argument("--literal", action="store_true", help="OPD original infinite loop, without lifecycle patch")
    p.add_argument("--smoke-examples", type=int, help="Use first N rows per native train/test split; explicitly NOT a full reproduction")
    p.add_argument("--smoke-recipe-world-size", type=int, help="OPSD smoke only: construct a valid upstream recipe at this world size before explicit batch overrides")
    p.add_argument("--hash-weights", action="store_true")
    p = sub.add_parser("native-run")
    p.add_argument("--run", required=True)
    p = sub.add_parser("native-eval", help="Prepare (and optionally execute) the upstream native evaluation")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--dataset", default="drawbench")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--guidance", type=float)
    p.add_argument("--rewards", default="pickscore,clipscore,hpsv2")
    p.add_argument("--execute", action="store_true")
    p = sub.add_parser("prepare", help="Prepare the local intervention engine; does not allocate GPUs")
    p.add_argument("--config", required=True)
    p.add_argument("--site")
    p.add_argument("--run", required=True)
    p.add_argument("--world-size", type=int, default=1)
    p.add_argument("--hash-weights", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    p = sub.add_parser("run")
    p.add_argument("--run", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--diagnostic", action="store_true")
    p = sub.add_parser("make-queries", help="Freeze prompt/seed/time identities for counterfactual diagnosis")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    p.add_argument("--indices", nargs="+", type=int, default=[1, 4, 7])
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--out", required=True)
    p = sub.add_parser("eval", help="Unified held-out evaluation of base, raw LoRA, or EMA LoRA")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--adapter")
    p.add_argument("--adapter-kind", choices=["base", "raw", "ema", "upstream_unknown"], default="base")
    p.add_argument("--sampler", choices=["flow", "dpm1", "dpm2", "ddim", "dance"], default="flow")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--save-images", action="store_true")
    p.add_argument("--python", help="Explicit common evaluation environment; recorded in evaluation metadata")
    p = sub.add_parser("compare")
    p.add_argument("baseline")
    p.add_argument("method")
    p.add_argument("--out")
    p = sub.add_parser("analyze")
    p.add_argument("--run", required=True)
    p.add_argument("--out")
    p = sub.add_parser("report-diagnostic", help="Validate complete image evidence and build prompt-clustered report/review gallery")
    p.add_argument("--run", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        from .native import verify_upstream
        for name, lock in read_json(ROOT / "upstream.lock.json").items():
            target = ROOT / "upstream" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "init", str(target)], check=True)
                subprocess.run(["git", "-C", str(target), "remote", "add", "origin", lock["url"]], check=True)
                subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", "origin", lock["commit"]], check=True)
                subprocess.run(["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True)
            verify_upstream(name)
            print(f"{name}: verified {lock['commit']}")
    elif args.command == "doctor":
        import torch
        from .native import verify_upstream
        report = {"environment": environment(), "cuda_available": torch.cuda.is_available(), "upstreams": {}}
        for name in read_json(ROOT / "upstream.lock.json"):
            try:
                _, lock = verify_upstream(name)
                report["upstreams"][name] = {"clean": True, **lock}
            except Exception as exc:
                report["upstreams"][name] = {"clean": False, "error": str(exc)}
        if args.site:
            site = site_config(args.site)
            report["assets"] = {}
            assets = {"model": site["model"], "reward_ckpts": site["reward_ckpts"],
                      "pickapic": site["pickapic"], **site["teachers"]}
            for key, value in assets.items():
                try:
                    report["assets"][key] = tree_fingerprint(value, full=args.hash_weights)
                except Exception as exc:
                    report["assets"][key] = {"error": str(exc)}
        if args.out:
            write_json(args.out, report)
            print(f"Wrote {args.out}; CUDA available: {report['cuda_available']}")
        else:
            print(json.dumps(report, indent=2))
    elif args.command == "native-prepare":
        from .native import prepare_native
        run = prepare_native(args.upstream, args.preset, args.run, site_config(args.site),
                             overrides(args.set), args.world_size, args.literal, args.hash_weights,
                             args.smoke_examples, args.smoke_recipe_world_size, args.nodes)
        print(f"Prepared {run}; inspect config.audit.json and runtime_patches.json before execution")
    elif args.command == "native-run":
        from .native import run_native
        run_native(args.run)
    elif args.command == "native-eval":
        from .native import clean_env, native_eval_plan
        plan = native_eval_plan(args.run, args.out, args.checkpoint, args.dataset, args.steps, args.guidance, args.rewards)
        if args.execute:
            subprocess.run(plan["command"], cwd=plan["cwd"], env=clean_env(plan["pythonpath"], plan["extra_env"]), check=True)
        else:
            print(json.dumps(plan, indent=2))
    elif args.command == "prepare":
        from .config import Experiment
        from .runtime import prepare
        config = Experiment.load(args.config)
        for key, value in overrides(args.set).items():
            if not hasattr(config, key):
                raise ValueError(f"Unknown experiment key: {key}")
            setattr(config, key, value)
        run = prepare(config, args.run, site_config(args.site) if args.site else None, args.world_size, args.hash_weights)
        print(f"Prepared {run}; no training started")
    elif args.command == "make-queries":
        from .diagnostic import make_manifest
        result = make_manifest(args.dataset, args.split, args.seeds, args.indices, args.steps, args.out, args.ids)
        print(json.dumps({"queries": len(result["queries"]), "records": len(result["record_ids"]), "path": args.out}))
    elif args.command == "run":
        from .runtime import launch
        launch(args.run, args.resume, args.stop_after, args.diagnostic)
    elif args.command == "eval":
        from .runtime import launch_eval
        launch_eval(args.run, args.out, args.adapter, args.adapter_kind, args.sampler, args.steps, args.seeds,
                    args.split, args.save_images, args.python)
    elif args.command == "report-diagnostic":
        from .report import diagnostic_report
        result = diagnostic_report(args.run, args.out)
        print(json.dumps({"report": args.out, "queries": result["queries"], "prompts": result["prompts"]}))
    elif args.command in {"compare", "analyze"}:
        from .evaluate import analyze_queries, compare
        result = compare(args.baseline, args.method) if args.command == "compare" else analyze_queries(args.run)
        if args.out:
            write_json(args.out, result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
