"""Expose every frozen Phase20 evaluation case and diagnostic transition, without reranking."""
import html
import os
from collections import defaultdict

from opdlab.io import ROOT, read_json, sha256, write_json
from review_prospective import visual


campaign = ROOT / 'experiments/local_expert_v20'
run = ROOT / 'runs/le20_eval_001'
report = read_json(campaign / 'learning_results.json')
assert report['review_frozen']['frozen_before_key_join']
groups = defaultdict(dict)
for item in report['joined_review']:
    row = item['row']
    groups[row['pair_id']][item['evaluated_path'], row['model']] = item
assert len(groups) == 32
models = ['joint128', 'all_teacher', 'certified', 'random_matched']
order = [('single', 'joint128')] + [(p, m) for p in ('single', 'full') for m in models[1:]]
pages, summaries = [], []
for pair_id, items in sorted(groups.items(), key=lambda x: (x[1]['single', 'joint128']['row']['id'], x[1]['single', 'joint128']['row']['seed'])):
    base = items['single', 'joint128']
    assert len(items) == 8
    b = base['row']
    pages.append(f'<section id="{pair_id}"><h2>{html.escape(b["id"])} / seed {b["seed"]}</h2><p>{html.escape(b["cohort"])}</p><div class="row">')
    pair = {'pair_id': pair_id, 'id': b['id'], 'seed': b['seed'], 'cohort': b['cohort'], 'models': {}}
    for path, model in order:
        item = items[path, model]
        row, obs = item['row'], item['observation']
        source = run / row['image']
        assert sha256(source) == row['image_sha256']
        label = 'Initial' if model == 'joint128' else f'{model} / {path}'
        metrics = visual(obs)
        pair['models'][path + '/' + model] = {
            'case': item['case'], 'observation': obs, 'comparison': item['comparison'],
            'complete': metrics['complete_scene'], 'unknown': metrics['has_unknown'],
            'image': str(source.relative_to(ROOT)), 'image_sha256': row['image_sha256']}
        src = html.escape(os.path.relpath(source, campaign), quote=True)
        flags = html.escape(str({**obs['clauses'], 'visibility': obs['full_visibility']}))
        pages.append(f'<article><h3>{label}</h3><a href="{src}"><img loading="lazy" src="{src}" width="256" height="256"></a><p>complete={metrics["complete_scene"]}; harm={item["comparison"]["known_harm"]}; unknown={metrics["has_unknown"]}</p><p>{flags}</p><p>{html.escape(obs["visible_text"])}</p><p>{html.escape(obs["notes"])}</p></article>')
    pages.append('</div></section>')
    summaries.append(pair)
assert sum(len(x['models']) for x in summaries) == 224
write_json(campaign / 'learning_cases.json', {
    'source_sha256': sha256(campaign / 'learning_results.json'),
    'script_sha256': sha256(__file__), 'all_pairs': summaries,
    'posthoc_display_only': True, 'new_generation_or_training': False})
(campaign / 'learning_cases.html').write_text('''<!doctype html><meta charset="utf-8"><title>Phase20: all evaluation cases</title>
<style>body{font:15px system-ui;margin:24px;color:#222}section{border-top:1px solid #bbb;padding:16px 0}.row{display:flex;gap:16px;overflow:auto}article{flex:0 0 256px}img{object-fit:contain}h3{font-size:15px}p{overflow-wrap:anywhere}</style>
<h1>Phase20: all 32 pairs / 224 generated endpoints</h1><p>Frozen provisional implementing-assistant observations. Initial plus all three models on both paths. Click images for native resolution. Unknown stays unknown. Display prepared after identity join; no labels or primary statistics changed.</p>
''' + '\n'.join(pages), encoding='utf-8')
print('Wrote all 32 pairs / 224 images, with frozen labels and provenance.')
