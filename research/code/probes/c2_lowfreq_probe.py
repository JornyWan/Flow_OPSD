"""C2 天花板探针:在高噪 query 状态,把"穿过冻结后缀的 reward 梯度"g_Q 投到低频子空间,
端点增益是否**高于**不投影的 g_Q?(C2 剩下的唯一新颖点 = 这个投影;若它不比 g_Q 好,C2 退化为 LeapAlign/FlowBP)

复用训练脚本的 _q_continue_from_grad / _q_continue_from / _reward_scores_grad / _decode01,不重写后缀。
所有方向单位化后走同一步长 0.05·‖y0‖(= η·ρ/n_ascent),端点增益 = R(D(F_old(s,y0+step·d̂))) − R(D(F_old(s,y0)))。
"""
import argparse, importlib.util, json, os, sys
import torch
from torch.cuda.amp import autocast as torch_autocast

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True); ap.add_argument("--lora", required=True)
ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--n-prompts", type=int, default=32); ap.add_argument("--chunk", type=int, default=4)
ap.add_argument("--kqs", default="2,5,7,8"); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.repo, "src")); sys.path.insert(0, a.repo)
spec = importlib.util.spec_from_file_location("tr", os.path.join(a.repo, "scripts/train_opsd_ri_sd3.py"))
tr = importlib.util.module_from_spec(spec); sys.modules["tr"] = tr; spec.loader.exec_module(tr)
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

dev = torch.device("cuda")
pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
for m in (pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3, pipe.transformer):
    m.requires_grad_(False)
pipe.vae.to(dev, dtype=torch.float32)
for m in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3): m.to(dev, dtype=torch.float16)
t = PeftModel.from_pretrained(pipe.transformer.to(dev), a.lora)
try: t.set_adapter("default")
except Exception: t.set_adapter(list(t.peft_config.keys())[0])
pipe.transformer = t; pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
scorer = tr._load_reward_scorer("open3", dev)
prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
KQS = [int(x) for x in a.kqs.split(",")]
STEP = 0.05
proj = lambda g, s: g if s == 1 else torch.nn.functional.interpolate(
    torch.nn.functional.avg_pool2d(g, s), scale_factor=s, mode="nearest")
unit = lambda d: d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)

def make_v(pe, ppe, grad):
    def v(z, sigma):
        tt = torch.full([z.shape[0]], float(sigma) * 1000, device=dev, dtype=torch.long)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx, torch_autocast(enabled=True, dtype=torch.float16):
            return pipe.transformer(hidden_states=z.to(pe.dtype), timestep=tt, encoder_hidden_states=pe,
                                    pooled_projections=ppe, return_dict=False)[0].float()
    return v

rows = []; gen = torch.Generator(device=dev).manual_seed(a.seed)
for i0 in range(0, len(prompts), a.chunk):
    prm = prompts[i0:i0 + a.chunk]; B = len(prm)
    with torch.no_grad():
        pe, ppe = encode_prompt(tes, tks, prm, 128); ne1, npe1 = encode_prompt(tes, tks, [""], 128)
    pe, ppe = pe.to(dev), ppe.to(dev)
    with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
        _, lat, _ = pipeline_with_logprob(pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
            negative_prompt_embeds=ne1.to(dev).repeat(B, 1, 1), negative_pooled_prompt_embeds=npe1.to(dev).repeat(B, 1),
            num_inference_steps=10, guidance_scale=1.0, output_type="pt", height=512, width=512,
            noise_level=0.7, deterministic=True, solver="dpm2", model_type="sd3")
    lat = torch.stack(lat, dim=1); sig = pipe.scheduler.sigmas.float().to(dev)
    for b in range(B):
        p1 = [prm[b]]; e, pp = pe[b:b+1], ppe[b:b+1]
        v_ng, v_g = make_v(e, pp, False), make_v(e, pp, True)
        R_end = lambda y, zq, kq, px: tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe,
                    tr._q_continue_from(zq, kq, y, px, sig, v_ng)), p1).float()
        rec = dict(prompt_i=i0 + b, kq={})
        for kq in KQS:
            zq = lat[b:b+1, kq].float(); zp = lat[b:b+1, kq - 1].float()
            with torch.no_grad():
                y0 = zq - float(sig[kq]) * v_ng(zq, sig[kq])
                px = zp - float(sig[kq - 1]) * v_ng(zp, sig[kq - 1])
            yq = y0.detach().clone().requires_grad_(True)
            ep = tr._q_continue_from_grad(zq, kq, yq, px, sig, v_g)
            r = tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, ep), p1)
            (gQ,) = torch.autograd.grad(r.sum(), yq); gQ = gQ.float().detach()
            del yq, ep, r; torch.cuda.empty_cache()
            rnd = torch.randn(y0.shape, generator=gen, device=dev)
            dirs = {"gQ": gQ, "gQ_p4": proj(gQ, 4), "gQ_p8": proj(gQ, 8), "gQ_p16": proj(gQ, 16),
                    "rand": rnd, "rand_p8": proj(rnd, 8)}
            step = STEP * float(y0.flatten().norm())
            with torch.no_grad():
                base = float(R_end(y0, zq, kq, px))
                gains = {k: float(R_end(y0 + step * unit(d), zq, kq, px)) - base for k, d in dirs.items()}
            spec_ = {s: float(proj(gQ, s).flatten().norm() / (gQ.flatten().norm() + 1e-12)) for s in (2, 4, 8, 16)}
            rec["kq"][str(kq)] = dict(sigma=float(sig[kq]), base=base, gains=gains, spec=spec_)
        rows.append(rec)
    print(f"  [c2] {i0+B}/{len(prompts)}  " + "  ".join(
        f"kq{k}: gQ={sum(r['kq'][str(k)]['gains']['gQ'] for r in rows)/len(rows):+.4f} "
        f"p8={sum(r['kq'][str(k)]['gains']['gQ_p8'] for r in rows)/len(rows):+.4f}" for k in KQS), flush=True)
json.dump(dict(meta=dict(lora=a.lora, kqs=KQS, step=STEP, n=len(rows)), rows=rows), open(a.out, "w"))
print("  [c2] wrote", a.out, flush=True)
