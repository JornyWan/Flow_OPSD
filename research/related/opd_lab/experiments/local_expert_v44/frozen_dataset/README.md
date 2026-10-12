# Joint correction v1: frozen prospective tasks

48 development prompts (validation) and 192 confirmation prompts (test), each
with two independently fixed seeds. Four equal task families: OCR+count,
OCR+color, OCR+relation, OCR+count+color. Development has four template styles;
confirmation has eight distinct styles. Words are 5–8 uppercase letters from
an explicit curated vocabulary, excluding every OCR target found in the 70
existing run data manifests at preparation. Whole-image exact text is intended:
the quoted word is the only lettering. No full-visibility/separation requirement
is silently added to scoring. Counts/colors/relations have explicit native
GenEval metadata checked against its class vocabulary.

Words and templates do not overlap between stages. Count/color specifications,
and whole unordered object pairs for relation/multi tasks, are held out within
each task family. Individual objects and semantic components can recur across
families/splits; do not claim all components unseen. This is a custom task suite,
not the original public GenEval benchmark. Caption placement/art style diversity
is still finite and largely controlled composition, not unrestricted real-world
prompt coverage. Publish per-family and per-template outcomes.

Development chooses only the predeclared best fixed intervention T* from starts
{1,3} × lengths{1,2}; confirmation does not alter windows/thresholds/checkpoints.
Bindings include exact prompt IDs, stage, template, word, combination and two
seeds. Blind-review identities (24 prompts, both seeds) and left/right assignment
are fixed now, not selected by outcomes. No human answers exist yet. No images
from this manifest were generated when it was frozen.

Read-only manifest: any change requires a new named version and explicit reason.
