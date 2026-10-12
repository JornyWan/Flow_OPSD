"""Build an offline research report from frozen experiments; no new evaluation or labels."""
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from opdlab.io import ROOT, read_json, sha256, write_json


MODELS = ['joint128', 'all_teacher', 'certified', 'random_matched']
PHASES = [
    ('工程与初诊', '工程验证与局部机会', '固定上游版本，跑通OPD/OPSD训练与重载；完成384个查询、1920张候选的冻结诊断。', '同评分器选出的正收益不等于独立质量提升；工程smoke不等于论文数值复现。'),
    ('工程与初诊', '四臂受控训练', '四臂各12次更新，960张验证图；暂未显示跨采样配置的稳定增益。', '64个训练提示词、32个验证提示词、一个训练种子，属于短程开发实验。'),
    ('工程与初诊', '拟合与拒绝点保持', '固定64个查询、22个接受点；保持项减小预测漂移，更多拟合未单调改善终点。', '局部预测稳定不自动保证完整生成质量。'),
    ('工程与初诊', '动作幅度与方向', '在22个接受状态上生成286张图；投影回教师方向未恢复平均代理收益。', '小幅目标或简单方向投影均没有自动有效性保证。'),
    ('测量审查', 'OCR与检测器审计', '定位额外文字漏读、检测阈值和整组大框等误判；保留60份原始检测记录。', '第二OCR或两套OCR一致仍不能充当真值。'),
    ('测量审查', '独立视觉语言模型审查', '检查76张图并补做关系接口对照，发现错字纠正、小字漏检及裁切遗漏。', '视觉语言模型保留为诊断工具，未替换训练奖励。'),
    ('学习与迁移', '局部目标可学习性', '充分拟合5个选定目标：联合拟合保留4/5局部修复、3/5完整生成修复。', '这是有意选择的训练案例，不是新提示词泛化。'),
    ('学习与迁移', '前缀与后缀隔离', '隔离旧/新前缀与后缀，发现苹果案例在到达原监督状态之前已发生漂移。', '一个轨迹干扰案例不能解释所有新状态失败。'),
    ('学习与迁移', '在漂移状态重新查询专家', '在当前学生的新状态上，GenEval完整动作修复苹果/碗关系并保留文字。', '当前状态的新修正不等于旧目标仍然适用。'),
    ('学习与迁移', '稀疏重拟合', '新采修正在32及320次稀疏更新后均保留于苹果案例的普通完整生成。', '五个选定训练提示词/seed；推理无教师，但还不能称稳定迁移。'),
    ('学习与迁移', '部分跨seed迁移', '拟合提示词的新seed完整图2/10→4/10；32例验证图净增2–3张，同时有原正确约束回归。', '区间跨零，不挑最佳检查点，不宣称稳定增益。'),
    ('测量审查', '自动接受目标审计', '64个训练提示词、192张候选；自动接受20个：6安全修复、3损伤、6无确认修复、5未知。', '自动监督可靠性未过门槛，因此未从该批自动目标启动训练。'),
    ('测量审查', '多视图验证', '冻结v2接受6个：3安全、0已知损伤，但安全状态召回仅3/6。', '降低误接受同时丢失监督，仍不足以自动训练。'),
    ('测量审查', '成对证据审查', '512次Qwen请求；主协议遇输出容器问题，事后兼容诊断接受3个，2安全、1未知。', '主协议失败与事后诊断分别保留；未过门槛，未启动训练。'),
    ('学习与迁移', '标签辅助筛选训练', '6个安全目标加6个保留状态；目标相对MSE降至0.05075，拟合seed完整0/6→5/6。', '新seed仍2/12，随机3/12；已拟合不等于泛化。'),
    ('学习与迁移', '单步动作迁移', '单步隔离仍未改变新seed的2/12；新状态教师动作误差明显高于训练状态。', '轨迹累积干扰不是已观测泛化失败的充分解释。'),
    ('学习与迁移', '扩展候选覆盖', '12个新状态中，固定候选安全修复机会1/12；扩展时间/幅度后3/12。', '候选机会增加尚非学生收益；标签辅助完整7/12也非部署成绩。'),
    ('学习与迁移', '新增状态监督', '多状态组改善新增状态拟合；主要新seed仍5/12，与初始相同。', '全部32个新seed有4例已知损伤，未过预设门槛。'),
    ('选择价值', '教师机会与学生缺口', '同12个状态学生5/12；按冻结标签事后选择安全教师或原图可达7/12。', '事后oracle不是可部署选择器；局部方向误差也不能只靠放大消除。'),
    ('选择价值', '单专家验证筛选', '48个状态选出9个标签辅助目标；三组各128更新，筛选主要新seed6/12，与初始相同。', '原全部教师9/12含人工未确认成功；筛选价值尚未建立。'),
]


class Assets:
    def __init__(self):
        self.sources = {}
        self.images = {}

    def read(self, relative):
        path = ROOT / relative
        raw = path.read_bytes()
        self.sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return raw

    def json(self, relative):
        return json.loads(self.read(relative))

    def image(self, relative, expected=None):
        raw = self.read(relative)
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and expected != digest:
            raise ValueError(f'Image differs from frozen identity: {relative}')
        self.images.setdefault(digest, 'data:image/png;base64,' + base64.b64encode(raw).decode())
        return digest


def gallery(assets, directory, results, kind):
    items = {r['case']: r for r in assets.json(f'{directory}/items.json')}
    key = assets.json(f'{directory}/key.json')
    mode_field = 'model' if kind == 'evaluation' else 'mode'
    observations = {(r['case'], r['row'][mode_field]): r for r in results['joined_review']}
    frozen = results['review_frozen']
    for name in ('items', 'key'):
        if assets.sources[f'{directory}/{name}.json'] != frozen[f'{name}_sha256']:
            raise ValueError(f'Frozen {name} mismatch: {directory}')
    groups = []
    for entry in key:
        item = items[entry['case']]
        cards = []
        for letter, identity in entry['mapping'].items():
            mode = identity[mode_field]
            joined = observations[entry['case'], mode]
            if joined['row']['image_sha256'] != identity['image_sha256']:
                raise ValueError('Joined label/image mismatch')
            cards.append({'model': mode, 'image': assets.image(f'{directory}/{item["images"][letter]}', identity['image_sha256']),
                'observation': joined['observation'], 'comparison': joined['comparison']})
        groups.append({'case': entry['case'], 'pair_id': entry['pair_id'], 'prompt': item['prompt'],
            'constraints': item['constraints'], 'seed': identity['seed'], 'id': identity['id'],
            'cohort': identity['cohort'], 'path': entry.get('path', 'single'), 'cards': cards})
    if len(observations) != sum(len(g['cards']) for g in groups):
        raise ValueError('Not all observations are represented')
    return sorted(groups, key=lambda x: x['case'])


def metrics(groups):
    output = {}
    for path in ('single', 'full'):
        output[path] = {}
        for cohort in ('positive_fresh_seed', 'preservation_fresh_seed', 'unfitted_prompt_fresh_seed', 'all_fresh'):
            subset = [g for g in groups if g['path'] == path and (cohort == 'all_fresh' or g['cohort'] == cohort)]
            output[path][cohort] = {}
            for model in MODELS:
                cards = [c for g in subset for c in g['cards'] if c['model'] == model]
                fields = [list(c['observation']['clauses'].values()) + [c['observation']['full_visibility']] for c in cards]
                output[path][cohort][model] = {'n': len(cards), 'complete': sum(all(v is True for v in f) for f in fields),
                    'unknown': sum(None in f for f in fields), 'harm': sum(c['comparison']['known_harm'] for c in cards)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/local_expert_report_20260918/report.html')
    parser.add_argument('--manifest', default='experiments/local_expert_report_20260918/manifest.json')
    args = parser.parse_args()
    output, manifest_path = ROOT / args.out, ROOT / args.manifest
    assets = Assets()
    assets.read('scripts/build_project_report.py')
    template = assets.read('docs/project_report_template.html').decode()
    learning = assets.json('experiments/local_expert_v20/learning_results.json')
    collection = assets.json('experiments/local_expert_v20/results.json')
    teacher = assets.json('experiments/local_expert_v19/results.json')
    human_base = 'experiments/local_expert_v20_human_review_v1/analysis_001'
    human = assets.json(f'{human_base}/results.json')
    receipt = assets.json(f'{human_base}/receipt.json')
    raw_human = assets.json(f'{human_base}/answers_source.json')
    if assets.sources[f'{human_base}/results.json'] != receipt['results_sha256']:
        raise ValueError('Human analysis hash mismatch')
    if assets.sources[f'{human_base}/answers_source.json'] != receipt['source_sha256']:
        raise ValueError('Human source hash mismatch')
    data = {'date': '2026-09-18', 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'evaluation': gallery(assets, 'runs/le20_eval_review_001', learning, 'evaluation'),
        'collection': gallery(assets, 'runs/le20_review_001', collection, 'collection'),
        'teacher': gallery(assets, 'runs/le19_review_001', teacher, 'teacher'),
        'human': human, 'human_raw': raw_human, 'alignment': learning['accepted_alignment'],
        'drift': learning['noop_target_drift'], 'upstream': assets.json('upstream.lock.json')}
    human_by_image = {(r['source_case'], r['model']): r for r in human['rows']}
    for group in data['evaluation']:
        for card in group['cards']:
            review = human_by_image.get((group['case'], card['model']))
            if review:
                if review['image_sha256'] != card['image']:
                    raise ValueError('Human feedback attached to wrong image')
                card['human'] = review
                group['human_group'] = review['group']
    targets = {r['case']: r for r in collection['targets']}
    for group in data['collection']:
        group['selection'] = {k: targets[group['case']][k] for k in ('accepted', 'reason')}
    data['collection_reasons'] = dict(Counter(t['reason'] for t in collection['targets']))
    data['metrics'] = metrics(data['evaluation'])
    for path, cohorts in data['metrics'].items():
        for cohort, models in cohorts.items():
            for model, values in models.items():
                source = learning['scores'][cohort][path][model]
                for metric, source_key in [('complete', 'complete_scene'), ('unknown', 'has_unknown'), ('harm', 'known_harm')]:
                    if values[metric] != source[source_key]['sum'] or values['n'] != source[source_key]['observations']:
                        raise ValueError(f'Recomputed metric mismatch: {path}/{cohort}/{model}/{metric}')
    data['phases'] = []
    for n, (category, title, summary, limit) in enumerate(PHASES, 1):
        folder = Path(f'experiments/local_expert_v{n}')
        phase = {'number': n, 'category': category, 'title': title, 'summary': summary, 'limit': limit,
                 'source': str(folder / 'STATUS.md'), 'status': assets.read(folder / 'STATUS.md').decode(), 'figures': []}
        figures = sorted((ROOT / folder).glob('*.png'))
        if n == 1:
            figures += [ROOT / 'runs/le1_diagnostic_001/reports/v1/figures/fixed_actions.png']
        for figure in figures:
            relative = str(figure.relative_to(ROOT))
            phase['figures'].append({'name': figure.stem, 'source': relative, 'image': assets.image(relative)})
        data['phases'].append(phase)
    data['documents'] = {name: assets.read(name).decode() for name in (
        'docs/method.md', 'docs/pipeline.md', 'docs/research_reassessment_20260912.md',
        'docs/advisor_update_20260918.md', 'experiments/local_expert_v20/NEXT.md',
        'experiments/local_expert_v20_human_review_v1/analysis_001/REPORT.md')}
    # Verify included frozen Phase20 artifacts, without rewriting its historical index.
    archive_path = 'experiments/local_expert_v20/report_index.json'
    archive = assets.json(archive_path)['artifacts_sha256']
    covered = set(archive) & set(assets.sources)
    for source in covered:
        if assets.sources[source] != archive[source]:
            raise ValueError(f'Included archived source changed: {source}')
    data['provenance'] = {'sources_sha256': assets.sources, 'included_phase20_archive_hashes_verified': len(covered),
        'evaluation_groups': len(data['evaluation']), 'evaluation_observations': sum(len(g['cards']) for g in data['evaluation']),
        'collection_groups': len(data['collection']), 'collection_observations': sum(len(g['cards']) for g in data['collection']),
        'teacher_groups': len(data['teacher']), 'teacher_observations': sum(len(g['cards']) for g in data['teacher']),
        'phases': len(data['phases']), 'historical_figures': sum(len(p['figures']) for p in data['phases']),
        'unique_images': len(assets.images), 'images_byte_preserved': True, 'labels_changed': False,
        'scope': 'opd_lab phases 1–20 and human review; all 400 observation slots in the three embedded case galleries; '
                 'historical phase summaries and report figures, not every historical raw generated image.'}
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', r'\u003c')
    images = json.dumps(assets.images, separators=(',', ':'))
    if template.count('__REPORT_DATA__') != 1 or template.count('__REPORT_IMAGES__') != 1:
        raise ValueError('Template must contain one data and one image placeholder')
    html = template.replace('__REPORT_DATA__', payload).replace('__REPORT_IMAGES__', images)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    write_json(manifest_path, {**data['provenance'], 'source_commit': data['source_commit'],
        'report_path': str(output.relative_to(ROOT)), 'report_sha256': sha256(output), 'bytes': output.stat().st_size,
        'rebuild': 'PYTHONPATH=python ../../conda_envs/envs/diffusionopd/bin/python scripts/build_project_report.py'})
    print(json.dumps({k: v for k, v in read_json(manifest_path).items() if k != 'sources_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
