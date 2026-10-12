import csv, importlib.util, json, os, random, sys
G="/midtier/minilab/scratch/chw4026/projects/side_project/generative/project3/tts/psp/geneval"
GEN="/midtier/minilab/scratch/chw4026/projects/side_project/generative"
sys.path.insert(0,G); os.chdir(G)
spec=importlib.util.spec_from_file_location("gv", os.path.join(G,"evaluation/precompute_geneval_scores_sd35_collect_rewards.py"))
gv=importlib.util.module_from_spec(spec); sys.modules["gv"]=gv; spec.loader.exec_module(gv)
class A: pass
args=A(); args.options={}
args.model_config=f"{GEN}/opd_related/src/mmdetection/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
args.model_path=f"{G}/objdet"
# load_models 是 return 的,那几个名字只在脚本的 __main__ 块里成为模块全局;
# import 时那块不执行 ⇒ 必须手动绑回模块命名空间,否则 evaluate_image_with_breakdown NameError。
_od, (_cm, _tf, _tk), _cn = gv.load_models(args)
gv.object_detector, gv.clip_model, gv.transform, gv.tokenizer, gv.classnames = _od, _cm, _tf, _tk, _cn
gv.THRESHOLD, gv.COUNTING_THRESHOLD = 0.3, 0.9
gv.MAX_OBJECTS, gv.NMS_THRESHOLD, gv.POSITION_THRESHOLD = 16, 1.0, 0.1
meta=[json.loads(l) for l in open(f"{GEN}/project3/tts/psp/Fk-Diffusion-Steering/text_to_image/prompt_files/geneval_metadata.jsonl")]
rows=list(csv.DictReader(open(f"{GEN}/project3/tts/analysis/out/geneval_full.csv")))
rows=[r for r in rows if os.path.exists(r["filename"])]
random.seed(0)
sel=[r for r in rows if r["tag"] in ("position","color_attr")]
sel=random.sample(sel,min(24,len(sel)))
ok=0; n=0
for r in sel:
    m=meta[int(r["prompt_id"])]
    if m["tag"]!=r["tag"]:
        print(f"  ⚠️ tag 不符 pid={r['prompt_id']} meta={m['tag']} csv={r['tag']}"); continue
    c,_,_,_=gv.evaluate_image_with_breakdown(r["filename"], m, args)
    n+=1; ok+= (int(bool(c))==int(r["correct"]))
print(f"\n  复现缓存标签: {ok}/{n}  {'✅ 完全一致' if ok==n else '❌ 有不一致'}")
