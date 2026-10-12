# Paper joint-task v2: frozen candidate data, no generated results

1920 distinct prompts and literal targets: train512, development128,
confirmation512, final_test768. Four equally sized task families: text+count,
text+color, text+relation and text+multiple object attributes. Two fixed noise
seeds per prompt. Historical declared OCR targets are excluded. Words, template
IDs and within-family object specifications do not cross stages; unordered
object pairs are held out jointly across relation and multi families.

Each stage is a separate JSONL. Do not concatenate confirmation and final_test
just because both use the existing engine's `test` split value. Use
paper_joint_data.resolve_stage with an explicit stage and purpose. It rejects
confirmation/final for training or development and checks the exact file hash.

Photographs and illustrations occur in both training and evaluation. Final has
an additional192 3D miniature/sculpture prompts, an explicitly shifted subgroup.
Report it separately as well as the fixed overall population; detector errors
can change by style. These are programmatically checked tasks, not manually
validated visual truth or an already established benchmark.

Word source is the pinned first20hours/google-10000-english medium-length list;
source commit/hash and license are included. No image/model/reward queries have
been made. A complete formal model-comparison protocol, cost budget and training
seed plan must accompany any later confirmation/final generation. Text-only
construction is not a result and does not change previous experiment conclusions.
