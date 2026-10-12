# Phase8 complete: the apple/bowl failure starts before the trained query

One L40S job1048955 completed in94 allocated seconds (.026111 GPU-hours), no
training and no failed attempt.35 images,350 forwards/700 CFG examples. All15
phase7 reference images and all4 index-zero duplicate checks were bit-exact.
Full CPU suite:62 passed. New16 images reviewed mode/score-blind;14 mode rows
inherit pixel-identical frozen phase7 observations.

| Execution rule | Definite retained repairs /5 | Uncertain cases | Clean two-case subset retained |
|---|---:|---:|---:|
|One updated action, old prefix/suffix|4|0|2/2|
|Old prefix, updated query and suffix|3|1|1/2|
|Updated prefix and query, old suffix|2|1|1/2|
|Full updated student|3|0|2/2|
|Old prefix,2 updated steps|3|0|1/2|
|Old prefix,4 updated steps|3|1|1/2|

**Apple/bowl:** local succeeds; old-prefix/new-suffix succeeds; new-prefix/old-
suffix fails; full fails. Its new query state has MSE .0039243 versus the old
training state. The failure therefore follows the changed approach to the query
in these counterfactuals. Later learned actions alone do not destroy this repair
when the old query state is preserved. MSE is descriptive, not a universal danger
threshold or proof of the exact internal cause.

Window2/window4 restore the apple relation on the old prefix, but give no clear
aggregate advantage: window2 clearly breaks the banana word; window4's N is
uncertain. Prefix/suffix banana variants also have an uncertain N or extra mark,
while full and local are clearly correct. Uncertain is not counted as a success
or asserted to be a definite failure. Thus a universal temporal mask is not yet
supported. The bicycle text defect and remaining visibility failures persist.

**Next decision:** before more fitting, query the original OCR/GenEval experts
at the frozen joint128 student's ACTUAL states, with a frozen joint128 suffix.
This checks whether fresh on-policy teacher supervision still has repair value
at these times. It distinguishes stale training states from lack of teacher
headroom at the new state; it does not introduce a new method or claim novelty
for on-policy refresh, which the training engine already supports.

Results are on five deliberately chosen training cases, with provisional
implementing-assistant observations. There is no held-out or sampler claim.
See PROTOCOL.md, results.json, observations.json, ledger.jsonl and the next
phase's separate preregistered protocol.
