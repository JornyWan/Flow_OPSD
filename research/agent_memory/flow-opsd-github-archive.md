---
name: flow-opsd-github-archive
description: JornyWan/Flow_OPSD (PUBLIC) holds the clean MSP release at root + full OPSD research archive under research/; how to refresh it
metadata:
  node_type: memory
  type: reference
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-12T03:04:11.900Z
---

`github.com/JornyWan/Flow_OPSD` — **public** (user confirmed 2026-10-11 to push everything there anyway).
- Root = clean DiffusionOPSD 9c4c48c + MSP only (collaborators use it); old version tagged `v0.1-msp-release` (= 1a186af). Never change root code without asking.
- `research/` = full archive (commit c3e4efd, 2026-10-11 23:05): entry `research/README.md` for a fresh agent on another server; docs, agent_memory (OPD-only memories), fork as 3-file overlay, scripts, results tarballs, paper draft, CA-OPD + opd_lab.
- Refresh: local clone `<session scratchpad>/Flow_OPSD`, builder `<scratchpad>/build_research_archive.sh` (rebuilds research/ from opd_related + memory dir; then hand-edit README snapshot/status, re-run credential scan, commit, push with `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes"`). Scratchpad is session-scoped — if gone, re-clone and rebuild per EXPERIMENT_LOG §4.898.
- Exclude from memory export: jit/tts/project3/latex/paper-github/phase0 notes. Keep each file <50 MB (eval json tar.xz is 44 MB).
- Related: [[opd-ideas-tracker]], [[paper-github-overleaf-sync]] (same SSH auth).
