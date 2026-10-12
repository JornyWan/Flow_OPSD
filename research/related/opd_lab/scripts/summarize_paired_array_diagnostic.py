"""Separate posthoc array-container replay; never replaces the frozen main result."""
from pathlib import Path

from opdlab.io import ROOT, read_json, sha256, write_json
from opdlab.paired_array_diagnostic import VERSION, parse_array_compatible
from summarize_paired_audit import build_report


def main():
    campaign = ROOT / "experiments/local_expert_v14"
    primary = read_json(campaign / "results.json")
    report = build_report(Path(primary["run"]), diagnostic_pair_parser=parse_array_compatible)
    report.update(analysis_version=VERSION, primary_results_sha256=sha256(campaign / "results.json"),
                  parser_sha256=sha256(ROOT / "python/opdlab/paired_array_diagnostic.py"),
                  diagnostic_script_sha256=sha256(__file__), additional_gpu_calls=0)
    write_json(campaign / "array_diagnostic.json", report)
    print({p: {"accepted": m["acceptance"]["numerator"], "classes": m["accepted_classes"]}
           for p, m in report["policies"].items() if p in ("forward", "reverse", "pair_consensus", "primary")})
    print(report["development_readiness"])


if __name__ == "__main__":
    main()
