"""Prepare unchanged evaluation using the audited fit, counting both CFG text inputs."""
import argparse
from pathlib import Path
import shutil

from opdlab.config import Experiment
from opdlab.io import ROOT,read_json,sha256,write_json
from opdlab.runtime import prepare
from recover_acceptance_fit import encoded_prompt_examples
from review_prospective import lines
from verify_acceptance_learning import verify_fit


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--fit-run',default='runs/le20_fit_recovered_001')
    p.add_argument('--run',default='runs/le20_eval_001')
    a=p.parse_args()
    fit=Path(a.fit_run).resolve()
    verify_fit(fit)
    original=read_json(fit/'probe_plan.json')
    references=[{**q,'target':'noop','action_names':{'noop':'noop'},'action_hashes':{'noop':q['action_hashes']['noop']}}
                for q in original['points'] if q['role']=='positive' and q['replicate']==0]
    assert len(references)==6
    records={r['id']:r for r in lines(fit/'data.jsonl')}
    text_count=encoded_prompt_examples([records[q['id']]['prompt'] for q in original['evaluation']+references])
    assert text_count==40
    run=prepare(Experiment.load(fit/'config.json'),Path(a.run).resolve(),read_json(ROOT/'site.local.json'))
    plan={**original,'stage':'evaluate','references':references,'reference_images':6,'evaluation_images':224,'fit_run':str(fit),
          'expected_counts':{'forward_student':1436,'forward_behavior':582,'forward_examples':4036,'backward_queries':0,
                             'forward_geneval':0,'forward_ocr':0,'decode_images':230,'verifier_images':230,'text_encode_prompts':text_count}}
    (run/'checkpoints').symlink_to(fit/'checkpoints',target_is_directory=True)
    for name in ('PROTOCOL.md','REVIEW_RUBRIC.md'):shutil.copyfile(fit/name,run/name)
    write_json(run/'probe_plan.json',plan)
    inputs={Path(p) for p in read_json(fit/'probe_attestation.json')['inputs']}
    inputs.update([fit/'result.json',fit/'recovery_verification.json',fit/'fit_reports.json',fit/'metrics.jsonl',run/'probe_plan.json',run/'PROTOCOL.md',run/'REVIEW_RUBRIC.md'])
    inputs.update(fit/'checkpoints'/arm/'step128.pt' for arm in plan['arms'])
    inputs.update(ROOT/'scripts'/n for n in ('prepare_acceptance_evaluation.py','recover_acceptance_fit.py','verify_acceptance_learning.py','review_acceptance_learning.py','review_supervision.py'))
    write_json(run/'probe_attestation.json',{'inputs':{str(p):sha256(p) for p in sorted(inputs)}})
    write_json(ROOT/'experiments/local_expert_v20/evaluation_attempt_001.json',{'run':str(run),'fit_run':str(fit),'expected_counts':plan['expected_counts'],'plan_sha256':sha256(run/'probe_plan.json'),'new_training_updates':0})
    print({'run':str(run),'evaluation_images':224,'reference_images':6,'text_examples':text_count})


if __name__=='__main__':main()
