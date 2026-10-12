"""Audit completed fit artifacts after the terminal text-counter assertion failed.

The original failed allocation is immutable. A separate, explicitly recovered
artifact view supplies its completed checkpoints to evaluation, without training.
"""
import argparse
from collections import Counter
from pathlib import Path
import shutil
import subprocess

from opdlab.acceptance_learning import ARMS
from opdlab.config import Experiment
from opdlab.io import ROOT, assert_environment, read_json, sha256, write_json
from opdlab.runtime import prepare, verify_run
from review_prospective import lines
from verify_acceptance_learning import verify_fit


def encoded_prompt_examples(prompts):
    # SD3Backend.encode batches [empty prompt, conditional prompt] on each cache miss.
    assert len(set(prompts)) <= 64, "This accounting assumes the cache does not evict"
    return 2 * len(set(prompts))


def recover(source, destination):
    verify_run(source)
    assert_environment(read_json(source/"manifest.json")["environment"])
    for p,h in read_json(source/"probe_attestation.json")["inputs"].items():
        assert sha256(p)==h,p
    assert not any((source/n).exists() for n in ("RUNNING","result.json","endpoints.jsonl","eval_queries"))
    failure=read_json(source/"failure.json")
    assert failure["error"] == "AssertionError(('text_encode_prompts', 24, 12))"
    job=read_json(source/"submission.json")["job_id"]
    accounting=subprocess.check_output(["sacct","-X","-j",job,"--noheader","--parsable2","--format=JobIDRaw,State,ExitCode,ElapsedRaw"],text=True)
    fields=next(row.split('|') for row in accounting.splitlines() if row.split('|')[0].strip()==job)
    assert fields[1:3]==["FAILED","1:0"]
    plan, reports=read_json(source/"probe_plan.json"),read_json(source/"fit_reports.json")
    assert plan["stage"]=="fit" and set(reports)==set(ARMS)
    records={r["id"]:r for r in lines(source/"data.jsonl")}
    expected_text=encoded_prompt_examples([records[p["id"]]["prompt"] for p in plan["points"]])
    assert expected_text==24 and plan["expected_counts"]["text_encode_prompts"]==12
    references=lines(source/"references.jsonl")
    assert len(references)==63 and all(r["exact"] for r in references)
    for r in references:
        assert sha256(source/r["image"])==r["image_sha256"]
    # Totals are reconstructed, not presented as a recovered in-memory counter dump.
    counts=Counter()
    for arm in ARMS:
        assert reports[arm]["optimizer_updates"]==128
        assert reports[arm]["counts"]=={"backward_queries":1536,"forward_examples":3396,"forward_student":1698}
        counts.update(reports[arm]["counts"])
    reference_counts={"forward_student":54*10,"forward_behavior":54+9*6,
                      "forward_examples":2*(54*10+54+9*6),"decode_images":63,
                      "verifier_images":63,"text_encode_prompts":expected_text}
    counts.update(reference_counts)
    counts.update({"forward_geneval":0,"forward_ocr":0})
    corrected={**plan["expected_counts"],"text_encode_prompts":expected_text}
    assert dict(counts)==corrected
    original_files={str(p.relative_to(source)):sha256(p) for p in source.rglob('*')
                    if p.is_file() and not p.is_symlink() and 'upstream' not in p.parts and '__pycache__' not in p.parts}
    source_outputs={str(p.relative_to(source)):sha256(p) for p in (source/"checkpoints").rglob('*.pt')}
    source_outputs.update({p.name:sha256(p) for p in source.glob('*.jsonl')})
    run=prepare(Experiment.load(source/"config.json"),destination,read_json(ROOT/"site.local.json"))
    for name in ("checkpoints","query_results"):
        if (source/name).exists():(run/name).symlink_to(source/name,target_is_directory=True)
    for p in source.glob('*.jsonl'):
        if p.name!='data.jsonl':shutil.copyfile(p,run/p.name)
    assert sha256(run/'data.jsonl')==sha256(source/'data.jsonl')
    for name in ("fit_reports.json","training_execution.json","PROTOCOL.md","REVIEW_RUBRIC.md"):
        shutil.copyfile(source/name,run/name)
    plan={**plan,"expected_counts":corrected,"recovery_source":str(source)}
    write_json(run/'probe_plan.json',plan)
    recovery={"source":str(source),"source_job_id":job,"source_allocation_state":"FAILED","source_exit_code":"1:0",
              "source_allocated_seconds":int(fields[3]),"source_failure":failure,
              "source_artifacts_sha256":original_files,"original_expected_text_examples":12,"corrected_text_examples":24,
              "counts_provenance":{"per_arm":"actual persisted fit_reports counters","reference_phase":"reconstructed from the frozen load_queries loop and all63 exact reference records","text_examples":"24 observed in terminal AssertionError; independently explained by12 unique conditional prompts each batched with an empty prompt"},
              "new_optimizer_updates":0,"new_images":0,"new_gpu_allocations":0,
              "scope":"Separate audited view of completed fit artifacts. The original failed run, counters, plans and logs are not rewritten."}
    write_json(run/'recovery.json',recovery)
    inputs={Path(p) for p in read_json(source/'probe_attestation.json')['inputs']}
    inputs.update(source/p for p in original_files)
    inputs.update([Path(__file__).resolve(),run/'probe_plan.json',run/'recovery.json'])
    write_json(run/'probe_attestation.json',{'inputs':{str(p):sha256(p) for p in sorted(inputs)}})
    execution=read_json(source/'training_execution.json')['execution']
    write_json(run/'result.json',{"status":"complete","stage":"fit","arms":reports,"training_execution":execution,
               "optimizer_updates":384,"evaluation_images":0,"counts":dict(counts),"new_teacher_queries":0,
               "reference_images":63,"elapsed_seconds":failure['elapsed_seconds'],"output_hashes":source_outputs,
               "artifact_recovery":recovery,"counts_are_reconstructed":True})
    audit=verify_fit(run)
    assert all(sha256(source/p)==h for p,h in original_files.items())
    write_json(run/'recovery_verification.json',audit)
    print({'recovered_fit':str(run),'updates_preserved':384,'new_updates':0,'text_examples':24,'original_allocation':'FAILED, retained'})
    return run


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--source',default='runs/le20_fit_002')
    p.add_argument('--run',default='runs/le20_fit_recovered_001')
    a=p.parse_args()
    recover(Path(a.source).resolve(),Path(a.run).resolve())
