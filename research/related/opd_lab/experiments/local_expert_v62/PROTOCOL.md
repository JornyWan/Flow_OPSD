# Phase62: versioned semantic selection in the existing online engine

User authorization: continue the Phase61 next step. Initial stage is CPU code and
contract acceptance; no GPU budget is reserved by this file. Any GPU acceptance
must have a separate prospective scope before submission. Zero agent images,
screenshots or remote vision. Completed archives remain immutable.

Research purpose: move the recent A selector into the existing on-policy engine
without silently changing unknown/error semantics or matched-random supervision.
This is implementation acceptance, not evidence of efficacy. Native OPD remains
the independent pinned upstream baseline. No new retention loss, CFG search,
verifier threshold, B selection or quality-based checkpoint selection.

Single-expert/full-amplitude semantic protocol initially; single-process only.
Reuse the same collect_query, fitting/Adam/scaler/EMA/checkpoint and full-student
postfit machinery. Legacy defaults remain unchanged. Candidate A labels are
bool/None; unknown abstains, scorer errors fail before any batch update. The
existing semantic_accept rule remains the only verified selection criterion.

Match random per optimizer accumulation group and family/domain, not merely
per epoch. A random online arm must consume the verified arm's frozen count
schedule, since its own state-dependent acceptance count may differ later.
Require identical query ids/prompts/strata/noise seeds/timesteps and sampling+
optimizer configuration, original weights/assets and source. Allow only method,
run-local data/reference paths to differ. Random does not read its own A labels
when choosing; it samples the fixed teacher action from all slots, permitting
natural overlap with verified. States may diverge after learning; log that scope.

CPU acceptance: unknown/error/known harm handling, true/false selector parity
with historical saved A decisions (without reading images), deterministic counts
and identity contracts, batch-level rather than epoch-only matching, differing
online-state counts, empty selection skipping Adam/EMA, frozen behavior refresh,
no hidden anchor route, detached target/suffix recomputation, complete resume
equality, and unchanged legacy tests. Manifest pins include the count reference.

Success licenses preparing a small separate GPU execution check on train/exposed
development data with joint and all five quality metrics. No inference of stable
gains from CPU toy results; confirmation/final remain sealed. If count identity,
label parity or resume checks fail, fix the implementation before any GPU work.
