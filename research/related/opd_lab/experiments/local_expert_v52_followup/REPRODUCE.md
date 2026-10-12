# Reproduction and archive scope

This follow-up is a post-hoc, CPU-only interpretation of the immutable Phase52 run. It does not change the frozen primary comparison, thresholds, sampling, training, or evaluator. No new GPU job or direct agent image read was used.

From opd_lab, using the pinned diffusionopd Python with `PYTHONPATH=python:scripts`:

```bash
python scripts/analyze_temporal_fit_diagnostic.py --run runs/le52_temporal_fit_001 --out /tmp/phase52-fitting-diagnostic.json
```

The output must not already exist. This checks every consumed record against the completed GPU run output hashes. Active/skipped weighted means and direct-selection win/loss identities were additionally checked against the frozen analysis.

The preview was generated using `scripts/build_stage_preview.py` with Phase51 and Phase52 evaluation endpoint records and the pre-outcome `visual_preview_plan.json`. `scripts/check_stage_preview.cjs` checked the image hashes, exact score provenance, decoding, navigation and responsive layout without screenshots. Its relative dependency paths expect scripts to run from the original project, not this source copy.

`visual_preview.html` embeds all 80 images and works offline. `phase52_complete.zip` contains the immutable Phase52 report archive and this follow-up under their original sibling folder names; start with `local_expert_v52_followup/SUMMARY.md` or `visual_preview.html`. The report also includes CSV, SVG and PDF. Original training checkpoints, full image cohorts and trajectory tensors remain in `runs/le52_temporal_fit_001` with their recorded hashes; they are not embedded in the portable package. Linked Phase51 source provenance is separately preserved on the server.

No training-state fitted endpoint images exist yet. NEXT.md describes the proposed measurement, not an executed experiment.
