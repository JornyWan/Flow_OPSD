import argparse
from pathlib import Path

from .config import Experiment
from .io import read_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--resume")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--output")
    p.add_argument("--adapter")
    p.add_argument("--adapter-kind", default="base")
    p.add_argument("--sampler", default="flow")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split", default="test")
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()
    run = Path(args.run)
    config, site = Experiment.load(run / "config.json"), read_json(run / "site.json")
    if not args.evaluate:
        from .io import assert_environment
        assert_environment(read_json(run / "manifest.json")["environment"])
    if args.evaluate:
        from .evaluate import evaluate
        evaluate(config, site, run / "upstream", args.output, args.adapter, args.adapter_kind,
                 args.sampler, args.steps, args.seeds, args.split, args.save_images)
    elif args.diagnostic and config.diagnostic_manifest:
        from .diagnostic import run_diagnostic
        run_diagnostic(config, site, run / "upstream", run)
    else:
        if config.diagnostic_manifest:
            raise ValueError("This run has frozen diagnostic queries; invoke it with --diagnostic")
        from .engine import train_loop
        train_loop(config, site, run / "upstream", run, args.resume, args.stop_after, args.diagnostic)


if __name__ == "__main__":
    main()
