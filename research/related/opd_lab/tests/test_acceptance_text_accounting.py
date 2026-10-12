from collections import Counter,OrderedDict
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from opdlab.backends import SD3Backend
from recover_acceptance_fit import encoded_prompt_examples


def test_accounting_matches_real_backend_cfg_batch_and_cache(monkeypatch):
    fake=ModuleType('flow_grpo.diffusers_patch.train_dreambooth_lora_sd3')
    calls=[]
    def encode(encoders,tokenizers,prompts,length):
        calls.append(prompts)
        return torch.zeros(2,1,1),torch.zeros(2,1)
    fake.encode_prompt=encode
    monkeypatch.setitem(sys.modules,fake.__name__,fake)
    b=object.__new__(SD3Backend)
    b.cache=OrderedDict();b.counters=Counter();b.autocast=nullcontext
    b.config=SimpleNamespace(max_sequence_length=256)
    b.pipeline=SimpleNamespace(**{name:None for name in ('text_encoder','text_encoder_2','text_encoder_3','tokenizer','tokenizer_2','tokenizer_3')})
    prompts=['first','first','second','first']
    for prompt in prompts:b.encode(prompt)
    assert calls==[['','first'],['','second']]
    assert b.counters['text_encode_prompts']==encoded_prompt_examples(prompts)==4
    assert encoded_prompt_examples([str(i) for i in range(12)]*8)==24
    assert encoded_prompt_examples([str(i) for i in range(20)]*4)==40
    with pytest.raises(AssertionError):encoded_prompt_examples([str(i) for i in range(65)])
