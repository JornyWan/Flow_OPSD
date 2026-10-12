import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from continue_acceptance_learning import completed_result


def test_failed_or_incomplete_job_cannot_release_next_gpu_stage(tmp_path):
    assert not completed_result(tmp_path,"RUNNING","0:0")
    for state in ("TIMEOUT","FAILED","CANCELLED by 123"):
        with pytest.raises(RuntimeError):completed_result(tmp_path,state,"0:0")
    with pytest.raises(FileNotFoundError):completed_result(tmp_path,"COMPLETED","0:0")
    (tmp_path/"result.json").write_text(json.dumps({"status":"complete"}))
    assert completed_result(tmp_path,"COMPLETED","0:0")
    (tmp_path/"failure.json").write_text('{}')
    with pytest.raises(RuntimeError):completed_result(tmp_path,"COMPLETED","0:0")
