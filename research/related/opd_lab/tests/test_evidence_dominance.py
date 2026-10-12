from itertools import product

from opdlab.evidence_dominance import admissible_repair


def test_unknown_baseline_can_be_preserved_but_cannot_supply_repair_credit():
    base={'clauses':{'text':None,'relation':False},'full_visibility':True}
    good={'clauses':{'text':True,'relation':True},'full_visibility':True}
    assert admissible_repair(base,good)
    assert not admissible_repair(base,{**good,'clauses':{'text':False,'relation':True}})
    assert not admissible_repair({**base,'clauses':{'text':None,'relation':True}},good)


def test_admitted_repair_preserves_every_possible_baseline_completion():
    for a in product((False,True,None),repeat=3):
        base={'clauses':{'text':a[0],'relation':a[1]},'full_visibility':a[2]}
        for b in product((False,True,None),repeat=3):
            candidate={'clauses':{'text':b[0],'relation':b[1]},'full_visibility':b[2]}
            if not admissible_repair(base,candidate):
                continue
            assert None not in b
            for completion in product(*[(False,True) if v is None else (v,) for v in a]):
                assert all(not old or new for old,new in zip(completion,b))
                assert any(not old and new for old,new in zip(completion[:2],b[:2]))
