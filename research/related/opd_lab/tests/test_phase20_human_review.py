from analyze_phase20_human_review import MODELS, outcome, preferred_letters, summarize


def test_missing_and_unclear_are_unresolved_not_failures():
    assert outcome({'text': 'yes', 'visibility': 'missing'}) == 'unresolved'
    assert outcome({'text': 'unclear', 'visibility': 'yes'}) == 'unresolved'
    assert outcome({'text': 'no', 'visibility': 'missing'}) == 'known_failure'
    assert outcome({'text': 'yes', 'visibility': 'yes'}) == 'confirmed_success'


def test_common_denominator_excludes_group_with_one_missing_field():
    rows = [{'group': g, 'model': m, 'human': {'text': 'yes', 'visibility': 'yes'},
             'assistant': {'text': 'yes', 'visibility': 'yes'}}
            for g in ('H01', 'H02') for m in MODELS]
    rows[0]['human']['text'] = 'unclear'
    rows[-1]['human']['visibility'] = 'missing'
    result = summarize(rows)
    assert result['common_fully_categorical_groups'] == ['H01']
    assert result['common_group_scores']['joint128']['human'] == {'unresolved': 1}
    assert result['common_group_scores']['all_teacher']['human'] == {'confirmed_success': 1}
    assert result['field_counts']['missing'] == result['field_counts']['unclear'] == 1


def test_preference_ties_and_explanation_letters_are_not_extra_votes():
    assert preferred_letters('A/B并列, C,D的字不完整或者不清楚') == ['A', 'B']
    assert preferred_letters('C, 碗比A的碗更正常size') == ['C']
    assert preferred_letters('B') == ['B']
    assert preferred_letters('都还可以') == []
    assert preferred_letters('都不好') == []
