"""Pinned local Qwen audit worker: literal text first, separate scene questions."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

from opdlab.io import append_jsonl,read_json,sha256,write_json
from opdlab.vlm_audit import OBSERVATION_PROMPT,parse_response,scene_request


def environment():
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl,processing_qwen2_5_vl
    from transformers.models.qwen2_vl import image_processing_qwen2_vl
    return {'python':sys.version,'executable':str(Path(sys.executable).absolute()),
            'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate','safetensors','Pillow','numpy')},
            'source_sha256':{str(Path(m.__file__).resolve()):sha256(m.__file__) for m in
                             (modeling_qwen2_5_vl,processing_qwen2_5_vl,image_processing_qwen2_vl)}}


def main(run):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    run=Path(run).resolve();plan=read_json(run/'vlm_plan.json');a=read_json(run/'attestation.json')
    if sha256(run/'vlm_plan.json')!=a['plan_sha256'] or sha256(run/'PROTOCOL.md')!=a['protocol_sha256']:raise ValueError('Plan changed')
    for p,h in plan['input_files'].items():
        if sha256(p)!=h:raise ValueError(f'Input changed: {p}')
    for f,h in plan['framework_files'].items():
        if sha256(run/f)!=h:raise ValueError('Frozen worker/framework changed')
    if environment()!=plan['environment']:raise ValueError('VLM environment changed')
    assets=plan['assets'];model_path=Path(assets['snapshot'])
    for f,meta in assets['files'].items():
        if sha256(model_path/f)!=meta['sha256']:raise ValueError('Model/processor/tokenizer changed')
    if (run/'responses.jsonl').exists() or (run/'result.json').exists():raise FileExistsError('Use fresh attempt')
    start=time.perf_counter();torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=3136,max_pixels=262144)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
              torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
    model.requires_grad_(False)
    # Remove inherited sampling-only generation defaults from the model config.
    model.generation_config.do_sample=False
    for key in ('temperature','top_p','top_k'):setattr(model.generation_config,key,None)
    counts={'images':0,'calls':0,'input_tokens':0,'output_tokens':0,'invalid_outputs':0,'truncated_outputs':0}
    for row in plan['rows']:
        if sha256(row['image'])!=row['image_sha256']:raise ValueError('Image changed')
        image=Image.open(row['image']).convert('RGB');responses={}
        ids=[c['id'] for c in row['record']['constraints'] if c['kind']=='geneval']
        for stage,request in (('observation',OBSERVATION_PROMPT),('scene',scene_request(row['record']))):
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':request}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            inputs=processor(text=[text],images=[image],return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=384,do_sample=False)
            suffix=output[:,inputs.input_ids.shape[1]:]
            raw=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
            parsed,error=None,None
            try:parsed=parse_response(raw,stage,ids)
            except (ValueError,TypeError) as exc:error=f'{type(exc).__name__}: {exc}'
            truncated=suffix.shape[1]>=384
            if truncated:parsed=None;error='Generation reached maximum token limit'
            responses[stage]={'request':request,'raw':raw,'parsed':parsed,'parse_error':error,'truncated':truncated,
                              'input_tokens':inputs.input_ids.shape[1],'output_tokens':suffix.shape[1],
                              'image_grid_thw':inputs.image_grid_thw.cpu().tolist()}
            counts['calls']+=1;counts['input_tokens']+=inputs.input_ids.shape[1];counts['output_tokens']+=suffix.shape[1]
            counts['invalid_outputs']+=int(error is not None);counts['truncated_outputs']+=int(truncated)
        append_jsonl(run/'responses.jsonl',{'uid':row['uid'],'cohort':row['cohort'],'image_sha256':row['image_sha256'],
                     'responses':responses})
        counts['images']+=1
        print({'event':'vlm_image_complete','number':counts['images'],'uid':row['uid']},flush=True)
    elapsed=time.perf_counter()-start
    write_json(run/'result.json',{'status':'complete','counts':counts,'elapsed_seconds':elapsed,'allocated_gpu_hours':elapsed/3600,
               'gpu':torch.cuda.get_device_name(0),'environment':environment(),'plan_sha256':a['plan_sha256'],
               'responses_sha256':sha256(run/'responses.jsonl'),'peak_memory_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run');p.add_argument('--write-environment')
    a=p.parse_args()
    if a.write_environment:write_json(a.write_environment,environment())
    else:main(a.run)
