# Phase10 blinded endpoint review

Review every unique endpoint image with prompt, but without arm, update count,
inference mode or reward score. Exact phase9 matches inherit frozen labels.
Record every encoded clause as true/false/null, full_visibility, literal visible
text and concrete notes. Unknown never counts as success or definite failure.

Text is the requested word with no missing/extra characters or other visible
letters/words; plate text counts. Clear stylized letters pass; ambiguous glyphs
remain unknown. RIGHT means the subject is clearly right of the object, not
mainly below with a slight positive center offset. Count complete recognizable
objects; note missing/cropped objects and prompt requirements outside encoded
clauses, such as a separate sign above. Do not revise historical labels.

Apple repair requires relation=true, text=true and full_visibility=true, as the
phase9 accepted target has all three. For each control, preserve all phase9-base
true clauses; preserve visibility when the baseline is fully visible. Report
remaining baseline defects separately. Do not call a control with a known old
text/cropping defect fully correct merely because it avoids a new harm.

Freeze all observations, rubric/key/inherited hashes before unblinding. Reviewer
is the implementing assistant with old-image familiarity, not independent human
ground truth. Report complete fixed endpoint matrix; no per-image model selection.
