"""Prepare an outcome-independent, identity-hidden human review of all 12 primary pairs."""
import json
import random
import shutil
from pathlib import Path

from opdlab.io import ROOT, read_json, sha256, write_json


TEMPLATE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>图像独立复核 · 12组</title>
<style>
body{font:16px/1.6 system-ui,sans-serif;color:#20252b;background:#f5f6f8;margin:0}main{max-width:1100px;margin:auto;padding:24px}
header,article,.panel{background:white;padding:20px;border:1px solid #ddd;border-radius:10px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}img{width:100%;max-width:512px;height:auto;display:block}
label{display:block;margin:12px 0}select,input,textarea,button{font:inherit;padding:8px;box-sizing:border-box}input,textarea{width:100%}
select{margin-left:8px;max-width:100%}button{cursor:pointer;margin-right:10px}nav{position:sticky;top:0;background:#f5f6f8;padding:12px 0;z-index:1}
small{color:#555}.warn{color:#9c3e00}h1{font-size:24px}h2{font-size:20px}.prompt{white-space:pre-wrap}a{color:#146a9e}
@media(max-width:700px){.grid{grid-template-columns:1fr}main{padding:12px}}
</style><main>
<header><h1>独立复核：12组，每组4张</h1>
<p>请先逐张判断是否满足要求，再选填整体偏好。A–D在每组重新随机排列，字母不固定对应某个模型。这里不显示模型身份、分数或已有判断。</p>
<details><summary>判定说明（请先读）</summary><ul>
<li>文字：整张图只出现指定词，大小写不限。额外文字、漏字、多字或标点记“不满足”；不确定的小字/字形记“看不清或不确定”。正常字距不扣分，明显的词内断开请记“不满足”并备注；如果你不同意这个规则，也请说明。</li>
<li>数量、颜色、位置：按下面的明确要求，数量必须准确；位置按观看者左右。不要根据提示词补全图里看不清的物体。</li>
<li>完整可见：要求的物体都存在，主体/杯底/椅腿等没有被画面裁切或明显遮挡。不要因为其他条件已错而跳过此项。</li>
<li>每项允许“不确定”；所有控件初始未标注。请抄实际看到的文字，不要自动纠正成目标词。整体偏好另记，不代替上述条件。</li>
</ul></details><p><small>点击图片在新标签页查看原始512像素图。页面尝试在当前浏览器自动保存；离开前请下载JSON。图和记录都在本地，不会发送到服务器。</small></p>
</header>
<nav><button id="prev">上一组</button><button id="next">下一组</button><button id="download">下载当前记录 JSON</button><strong id="progress"></strong><div id="storage" class="warn"></div></nav>
<div id="task"></div></main>
<script id="review-data" type="application/json">__DATA__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('review-data').textContent);
const storageKey=data.review_id+':'+data.input_sha256;
let state={answers:{},preferences:{},index:0};
try{const saved=localStorage.getItem(storageKey);if(saved)state=JSON.parse(saved);}catch(e){document.getElementById('storage').textContent='浏览器未允许本地保存，请及时下载JSON。';}
let current=Math.max(0,Math.min(data.items.length-1,Number(state.index)||0));
const escapeHTML=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function answer(id){return state.answers[id]||(state.answers[id]={});}
function required(item){return [...item.criteria.map(x=>x.id),'visibility'];}
function complete(item){return item.images.every(im=>required(item).every(k=>['yes','no','unclear'].includes(answer(im.id)[k]))&&String(answer(im.id).visible_text||'').trim());}
function status(){document.getElementById('progress').textContent=`第${current+1}/12组 · 已标注完整 ${data.items.filter(complete).length}/12组`;}
function save(){state.index=current;try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(e){document.getElementById('storage').textContent='自动保存不可用，请下载JSON保留记录。';}status();}
function choice(id,key,title){return `<label>${escapeHTML(title)} <select data-id="${id}" data-field="${key}"><option value="">未标注</option><option value="yes">满足</option><option value="no">不满足</option><option value="unclear">看不清或不确定</option></select></label>`;}
function render(){
 const item=data.items[current];
 let content=`<div class="panel"><h2>${item.id}</h2><p class="prompt">${escapeHTML(item.prompt)}</p><ul>${item.criteria.map(x=>`<li>${escapeHTML(x.label)}</li>`).join('')}<li>所有要求的物体完整可见。</li></ul></div><div class="grid">`;
 for(const im of item.images){content+=`<article><h2>${im.letter}</h2><a href="${im.src}" target="_blank" rel="noopener"><img src="${im.src}" alt="${item.id} ${im.letter}" width="512" height="512"></a>`;
  for(const c of item.criteria)content+=choice(im.id,c.id,c.label);
  content+=choice(im.id,'visibility','要求的物体是否完整可见？');
  content+=`<label>实际看到的文字（看不清可写“看不清”）<input data-id="${im.id}" data-field="visible_text" autocomplete="off"></label><label>补充说明（选填：小字、字间空隙、裁切、为什么不确定等）<textarea data-id="${im.id}" data-field="notes" rows="2"></textarea></label></article>`;
 }
 content+=`</div><div class="panel"><label>整体更喜欢哪张？（选填，可写 A、A/B并列、都不好或无法判断）<input id="preference" autocomplete="off"></label><small>此项按画面自然、清晰、完整等整体印象填写；满足约束与美观程度可能不同。</small></div>`;
 document.getElementById('task').innerHTML=content;
 for(const el of document.querySelectorAll('[data-field]')){el.value=answer(el.dataset.id)[el.dataset.field]||'';el.addEventListener('input',()=>{answer(el.dataset.id)[el.dataset.field]=el.value;save();});}
 const pref=document.getElementById('preference');pref.value=state.preferences[item.id]||'';pref.addEventListener('input',()=>{state.preferences[item.id]=pref.value;save();});
 document.getElementById('prev').disabled=current===0;document.getElementById('next').disabled=current===data.items.length-1;status();
}
for(const [id,delta] of [['prev',-1],['next',1]])document.getElementById(id).addEventListener('click',()=>{current+=delta;save();render();window.scrollTo(0,0);});
document.getElementById('download').addEventListener('click',()=>{
 const payload={review_id:data.review_id,input_sha256:data.input_sha256,exported_at:new Date().toISOString(),complete_groups:data.items.filter(complete).length,total_groups:12,answers:state.answers,preferences:state.preferences};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=data.review_id+'-answers.json';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
render();
</script></html>'''


def main():
    review = ROOT / 'runs/le20_eval_review_001'
    private = ROOT / 'runs/le20_human_review_001'
    output = ROOT / 'experiments/local_expert_v20_human_review_v1'
    assert not output.exists() and not private.exists(), 'Keep previous review attempts intact.'
    frozen = read_json(review / 'observations_frozen.json')
    for name in ('key', 'items'):
        assert sha256(review / f'{name}.json') == frozen[f'{name}_sha256']
    items = {x['case']: x for x in read_json(review / 'items.json')}
    # Complete predeclared primary cohort, selected using cohort/path only, never outcomes.
    chosen = [x for x in read_json(review / 'key.json') if x['path'] == 'single'
              and {r['cohort'] for r in x['mapping'].values()} == {'positive_fresh_seed'}]
    assert len(chosen) == 12 and len({x['pair_id'] for x in chosen}) == 12
    rng = random.Random(2026091501)
    rng.shuffle(chosen)
    (private / 'images').mkdir(parents=True)
    output.mkdir(parents=True)
    public, mapping = [], []
    for n, key in enumerate(chosen, 1):
        old = items[key['case']]
        task = {'id': f'H{n:02d}', 'prompt': old['prompt'], 'criteria': [], 'images': []}
        for clause in old['constraints']:
            if clause['id'] == 'text':
                label = f'文字仅为 {clause["text"]}，没有其他文字或标点'
            else:
                terms = []
                classes = {'cup': '杯子', 'bowl': '碗', 'chair': '椅子', 'vase': '花瓶', 'apple': '苹果', 'bottle': '瓶子'}
                colors = {'green': '绿色', 'yellow': '黄色', 'red': '红色', 'blue': '蓝色'}
                objects = clause['metadata']['include']
                for obj in objects:
                    terms.append(f'恰好{obj["count"]}个{colors.get(obj.get("color"), "")}{classes[obj["class"]]}')
                    if 'position' in obj:
                        direction, index = obj['position']
                        terms.append(f'{classes[obj["class"]]}在{classes[objects[index]["class"]]}的' + {'left of': '左侧', 'right of': '右侧'}[direction])
                label = '；'.join(terms)
            task['criteria'].append({'id': clause['id'], 'label': label})
        letters = sorted(key['mapping'])
        rng.shuffle(letters)
        for new_letter, old_letter in zip('ABCD', letters):
            row = key['mapping'][old_letter]
            name = f'{task["id"]}-{new_letter}'
            target = private / 'images' / f'{name}.png'
            shutil.copyfile(review / old['images'][old_letter], target)
            assert sha256(target) == row['image_sha256']
            task['images'].append({'id': name, 'letter': new_letter,
                                   'src': f'../../runs/le20_human_review_001/images/{name}.png',
                                   'sha256': row['image_sha256']})
            mapping.append({'review_image_id': name, 'source_case': key['case'], 'source_letter': old_letter,
                            'pair_id': key['pair_id'], **{k: row[k] for k in ('model', 'id', 'seed', 'image_sha256')}})
        public.append(task)
    write_json(private / 'key.json', mapping)
    write_json(output / 'items.json', public)
    payload = {'review_id': 'phase20-human-primary-v1', 'input_sha256': sha256(output / 'items.json'), 'items': public}
    (output / 'review.html').write_text(TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', r'\u003c')), encoding='utf-8')
    write_json(output / 'manifest.json', {'review_id': payload['review_id'], 'source_review': str(review),
        'scope': 'All 12 predefined positive_fresh_seed / single pairs. No outcome-based selection.',
        'images': 48, 'randomization_seed': 2026091501, 'input_sha256': payload['input_sha256'],
        'html_sha256': sha256(output / 'review.html'), 'script_sha256': sha256(__file__),
        'source_frozen_sha256': sha256(review / 'observations_frozen.json'),
        'private_key_sha256': sha256(private / 'key.json'), 'human_observations_received': False,
        'notes': 'Human review is a new annotation version. The user has seen aggregate results; images hide identities and prior labels. No previous labels overwritten. This subset cannot re-estimate all-32 harm.'})
    print(f'Prepared {output / "review.html"}: 12 groups, 48 images, no prefilled labels.')


if __name__ == '__main__':
    main()
