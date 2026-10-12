# Phase47 quality-only continuation

Frozen before reading deployment effect statistics. The original job 1062382
timed out at 2710 allocated seconds (2700-second limit; scheduler overrun 10s).
All training, 512 generation endpoints and task scores are complete. 690/1120
quality values are saved. Preserve the original protocol and TIMEOUT record.

This separate, explicitly recorded exception completes only 430 missing quality
cells for the same 224 deployment images. Reuse all cached scores verbatim;
recompute the original eight-image batch at its partial boundary and check its
two cached overlaps within 1e-6. Recreate token audits and ImageReward API parity.
Use the original scorer snapshot, weights and cache fingerprint; record the
continuation code separately. No training, images, task rescores, sample changes,
threshold changes or endpoint changes. Preserve the original frozen analyzer;
its sole adaptation is the assembled run directory, frozen again before scoring.

One new L40 job capped at 120 seconds, charged to the existing 178 seconds left
in the shared 10800-second pool. No further GPU retry; if incomplete, record
missing cells and stop. The denied earlier time-limit change was not applied.
