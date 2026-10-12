"""Analyze the partial human review without imputing blanks or replacing Phase20 labels."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import re

from opdlab.io import ROOT, read_json, sha256, write_json


MODELS = ('joint128', 'all_teacher', 'certified', 'random_matched')
VALUES = {True: 'yes', False: 'no', None: 'unclear'}


def outcome(fields):
    # A known violation rules out joint success even if another field is missing.
    if 'no' in fields.values():
        return 'known_failure'
    return 'confirmed_success' if all(v == 'yes' for v in fields.values()) else 'unresolved'


def preferred_letters(text):
    # This export uses leading A, B/C并列, etc. Do not mine letters from explanations.
    match = re.match(r'^\s*([A-D](?:/[A-D])*)(?=$|[，,]|\s*并列)', text)
    return match.group(1).split('/') if match else []


def summarize(rows):
    groups = sorted({r['group'] for r in rows})
    common = [g for g in groups if all('missing' not in r['human'].values()
                                     for r in rows if r['group'] == g)]
    fields = Counter(v for r in rows for v in r['human'].values())
    return {
        'groups': len(groups), 'images': len(rows), 'field_counts': dict(fields),
        'groups_with_categorical_answers': sum(any(v != 'missing' for r in rows
            if r['group'] == g for v in r['human'].values()) for g in groups),
        'images_with_all_categorical_answers': sum('missing' not in r['human'].values() for r in rows),
        'common_fully_categorical_groups': common,
        'common_group_scores': {
            m: {reviewer: dict(Counter(outcome(r[reviewer]) for r in rows
                                      if r['model'] == m and r['group'] in common))
                for reviewer in ('human', 'assistant')}
            for m in MODELS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--answers', default='human_rating/phase20-human-primary-v1-answers.json')
    parser.add_argument('--output', default='experiments/local_expert_v20_human_review_v1/analysis_001')
    args = parser.parse_args()
    source, output = ROOT / args.answers, ROOT / args.output
    if output.exists():
        raise FileExistsError('Keep previous analyses intact; choose a new output directory.')
    review = ROOT / 'experiments/local_expert_v20_human_review_v1'
    private = ROOT / 'runs/le20_human_review_001/key.json'
    frozen = ROOT / 'runs/le20_eval_review_001/observations_frozen.json'
    old_results = ROOT / 'experiments/local_expert_v20/learning_results.json'
    manifest = read_json(review / 'manifest.json')
    expected = {review / 'items.json': manifest['input_sha256'],
                review / 'review.html': manifest['html_sha256'],
                private: manifest['private_key_sha256'], frozen: manifest['source_frozen_sha256']}
    index_path = ROOT / 'experiments/local_expert_v20/report_index.json'
    archive = read_json(index_path)['artifacts_sha256']
    expected.update({ROOT / p: h for p, h in archive.items()})
    for path, digest in expected.items():
        if sha256(path) != digest:
            raise ValueError(f'Input changed: {path}')
    raw = source.read_bytes()
    answers = json.loads(raw)
    if (answers['review_id'], answers['input_sha256']) != (manifest['review_id'], manifest['input_sha256']):
        raise ValueError('Answers belong to a different review or image set.')
    items, key = read_json(review / 'items.json'), read_json(private)
    mapping = {r['review_image_id']: r for r in key}
    ids = {im['id'] for item in items for im in item['images']}
    if set(mapping) != ids or len(mapping) != len(key) or not set(answers['answers']) <= ids:
        raise ValueError('Mismatched or duplicate image IDs.')
    if not set(answers['preferences']) <= {it['id'] for it in items}:
        raise ValueError('Unknown preference group.')
    old = {(r['case'], r['row']['model']): r for r in read_json(old_results)['joined_review']}
    rows, differences, preferences = [], [], []
    for item in items:
        required = [c['id'] for c in item['criteria']] + ['visibility']
        for image in item['images']:
            identity = mapping[image['id']]
            previous = old[identity['source_case'], identity['model']]
            image_hash = sha256((review / image['src']).resolve())
            if not image_hash == image['sha256'] == identity['image_sha256'] == previous['row']['image_sha256']:
                raise ValueError(f'Image identity mismatch: {image["id"]}')
            response = answers['answers'].get(image['id'], {})
            if set(response) - set(required) - {'visible_text', 'notes'}:
                raise ValueError(f'Unexpected field: {image["id"]}')
            human = {k: 'missing' if response.get(k) in ('', None) else response[k] for k in required}
            if not set(human.values()) <= {'yes', 'no', 'unclear', 'missing'}:
                raise ValueError(f'Invalid category: {image["id"]}')
            obs = previous['observation']
            assistant = {**{k: VALUES[v] for k, v in obs['clauses'].items()},
                         'visibility': VALUES[obs['full_visibility']]}
            if set(assistant) != set(human):
                raise ValueError('Rubric fields do not match.')
            row = {**identity, 'group': item['id'], 'human': human, 'assistant': assistant,
                   'human_outcome': outcome(human), 'assistant_outcome': outcome(assistant),
                   'visible_text': response.get('visible_text'), 'notes': response.get('notes')}
            rows.append(row)
            for field, value in human.items():
                if value != 'missing' and value != assistant[field]:
                    differences.append({'review_image_id': image['id'], 'model': identity['model'],
                        'field': field, 'human': value, 'assistant': assistant[field],
                        'opposite_definite': {value, assistant[field]} == {'yes', 'no'}})
        text = answers['preferences'].get(item['id'], '')
        letters = preferred_letters(text)
        preferences.append({'group': item['id'], 'verbatim': text, 'explicit_letters': letters,
            'selected_models': [mapping[f'{item["id"]}-{letter}']['model'] for letter in letters]})
    summary = summarize(rows)
    summary.update({'exported_complete_groups': answers['complete_groups'],
        'groups_with_overall_feedback': sum(bool(p['verbatim'].strip()) for p in preferences),
        'field_disagreements': len(differences),
        'opposite_definite_disagreements': sum(d['opposite_definite'] for d in differences),
        'explicit_preference_groups': sum(bool(p['explicit_letters']) for p in preferences)})
    output.mkdir(parents=True)
    (output / 'answers_source.json').write_bytes(raw)
    if sha256(source) != sha256(output / 'answers_source.json'):
        raise ValueError('Source answers changed during analysis.')
    write_json(output / 'results.json', {'summary': summary, 'rows': rows,
        'field_disagreements': differences, 'overall_preferences': preferences,
        'limitations': ['One human; 12 seeds across six familiar prompts; single-action path only.',
            'Common-group analysis uses seven groups selected by answer coverage, not a new primary cohort.',
            'Missing fields stay missing; unclear is not a known failure; comments do not fill checkboxes.',
            'Overall preference is separate from constraint success; no method ranking or all-32 harm estimate.',
            'Assistant labels are provisional and remain unchanged in the original archive.']})
    write_json(output / 'receipt.json', {
        'analyzed_at_utc': datetime.now(timezone.utc).isoformat(), 'human_observations_received': True,
        'reported_exported_at': answers['exported_at'], 'source_path': str(source.relative_to(ROOT)),
        'source_sha256': sha256(source), 'script_sha256': sha256(__file__),
        'items_sha256': sha256(review / 'items.json'), 'key_sha256': sha256(private),
        'original_results_sha256': sha256(old_results), 'archive_index_sha256': sha256(index_path),
        'original_archive_files_verified': len(archive), 'review_images_verified': len(rows),
        'results_sha256': sha256(output / 'results.json'),
        'note': 'Analysis receipt after analyst identity join; not a claim of pre-join analyst freezing. '
                'The preparation manifest remains a historical snapshot.'})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
