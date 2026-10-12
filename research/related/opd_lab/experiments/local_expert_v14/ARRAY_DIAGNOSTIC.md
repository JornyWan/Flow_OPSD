# Separate posthoc container-only parsing diagnostic

During job1049253, the first34 completed pair responses all failed the frozen parser
with `Expected object`. Inspection of the first two showed valid fenced JSON arrays
of clause objects, preserving the requested ids, booleans, change and evidence, but
omitting the requested outer `{"clauses": ...}` object. This is an interface contract
failure, not evidence that Qwen could not see those images.

The immutable worker, requests, protocol, image order and observations remain exactly
unchanged. Complete all512 planned requests for the full128-pair/64-self-control audit.
The primary results.json retains every original invalid/unknown response and its gates.
Do not call this a successful implementation of the original contract.

A separately named **posthoc** parser `paired_array_container_posthoc1` adds exactly
one structural mapping: a valid JSON array becomes the value of `clauses`. Standard
JSON parsing must still reject duplicate keys, nonfinite values and non-JSON prose.
No missing id, boolean, string, evidence or change is invented or corrected. All
original per-field validation, order consensus, literal guard, selection, labels and
metrics remain the same. Truncation still makes the response unknown.

Write array_diagnostic.json separately from the primary result, and reference both
hashes. This compatibility analysis uses the same512 responses and adds zero GPU
requests. Its metrics/gates are descriptive results under a posthoc interface change;
they never promote a failed frozen primary result or silently authorize training.
Report container failure separately from actual visual repair/damage errors. Before
any future prospective audit, a compatible schema must be frozen as a new version.
