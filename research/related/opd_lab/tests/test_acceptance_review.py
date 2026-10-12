from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from review_acceptance_learning import pilot_checks


def test_pilot_requires_both_controls_and_preservation_not_only_baseline_gain():
    scores={"all_fresh":{"single":{"certified":{"known_harm":{"sum":0}}}}}
    contrasts={"positive_fresh_seed":{"single":{m:{"sum":v} for m,v in (("joint128",2),("all_teacher",1),("random_matched",1))}},
               "preservation_fresh_seed":{"single":{"joint128":{"sum":0}}},
               "unfitted_prompt_fresh_seed":{"single":{"joint128":{"sum":0}}}}
    assert all(pilot_checks(scores,contrasts).values())
    for control in ("all_teacher","random_matched"):
        changed=deepcopy(contrasts)
        changed["positive_fresh_seed"]["single"][control]["sum"]=0
        assert not all(pilot_checks(scores,changed).values())
    for cohort in ("preservation_fresh_seed","unfitted_prompt_fresh_seed"):
        changed=deepcopy(contrasts)
        changed[cohort]["single"]["joint128"]["sum"]=-1
        assert not all(pilot_checks(scores,changed).values())
    scores["all_fresh"]["single"]["certified"]["known_harm"]["sum"]=1
    assert not all(pilot_checks(scores,contrasts).values())
