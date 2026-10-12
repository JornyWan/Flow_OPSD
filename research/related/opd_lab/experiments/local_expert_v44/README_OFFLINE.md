# Phase44 offline report

Open report.html directly. Tables are self-contained; no images or network calls.
Open human_review.html for the separate 24-pair targeted blind diagnostic:
48 original PNGs are embedded and work offline. No human labels are prefilled.
Exported answers belong in human_rating/ on the server. The private method
mapping stays in runs/le44_human_audit_001/; its SHA is in human_audit_manifest.json.
This archive contains source/provenance snapshots, raw numerical scorer evidence,
statistics and decisions. It is not a standalone model execution environment.
PNG/trajectory/model payloads remain on the research server with recorded hashes.
Adapter weights are content-hashed; the large backbone retains the unchanged
source inventory size/mtime checks, not a new full-content SHA claim.
96 noop endpoints are reused with their original cache fingerprints; 384 OCR
comparators come from Phase43. Only 384 GenEval endpoints are newly generated.
No student learning, confirmation generation or independent human labels here.
