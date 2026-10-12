# Phase12 candidate-hidden review of every prospective endpoint

Record each encoded clause, full_visibility, literal visible_text and notes for
all A/B/C images on all64 sheets. Values true/false/null. Freeze all192 views
before reading the action/score/selection key. The reviewer is the implementing
assistant familiar with the project, not independent human ground truth.

Text: requested word case-insensitively, no missing/extra characters or other
visible letters/words. Plate and object-label lettering count. Identifiable
stylized glyphs pass; unclear merged letters/possible extra print stay null.
Decorative non-letter shapes/fasteners alone are not extra text. Record missing
separate placard, poor contrast and nonblack letters as additional scene defects.

Count: exactly the requested recognizable objects. Identifiable cropped objects
contribute to count, but cropping separately fails full_visibility. Uncertain
object identity or multiplicity stays null; count definite missing/extra objects
as false. Color requires the specified number of target objects in the requested
color, following the encoded include/count clause. Count mismatch can fail color
too; note the observed colors independently. Shading/highlights allowed; unclear
pale color is null. Do not replace blue apples/red bananas by their natural color.

Relation: specified object multiplicities and subject clearly on requested side
of reference. Mainly below is not clearly right solely from center coordinates.
Ambiguous additional apple/bowl/cup-like forms stay null; note the ambiguity.

Full visibility: all requested objects complete within the frame. Cropped limbs,
wheels, rims, pages fail. A definitely missing requested object also fails. If
identity or multiplicity prevents determining completeness, use null. Extra
objects do not automatically fail visibility if requested identifiable objects
are complete; enforce multiplicity in the encoded clause. Note deformation,
unrelated objects, overlap and low contrast even when not fully encoded.

Strict complete success requires all clauses=true AND full_visibility=true.
Null is non-success, not a definite error/harm. Positive repair certification
requires determinate baseline and candidate labels; definite true->false harms
are reported even when another label is unknown. Freeze observations before
joining action identity, never change old/current labels to improve a result.
