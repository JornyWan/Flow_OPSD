"""P5:信任域内 best-of-N(纯比较器)vs P4 的零阶平均,**等 query 预算**;外加 grad/no-grad 成本实测。

为什么值得单独测:不可微 reward 能直接给的是**比较**,不是梯度。best-of-N 只用比较,
且取 max 而不是加权平均 —— 在 cos≈0.01 的情形下,取 max 很可能优于取平均。
预算口径:best-of-N 用 N 次 reward 前向;对偶零阶 k 对用 2k 次 ⇒ N=2k 才是等预算。

成本实测:同一 batch 上分别计时「带梯度的 reward 求值」与「no-grad reward 求值」,
给出真实倍率,替代"backward≈2×forward"的口算。
"""
import argparse, importlib.util, json, os, sys, time
import torch
from torch.cuda.amp import autocast as torch_autocast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=64); ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--kq", type=int, default=8); ap.add_argument("--max-n", type=int, default=64)
    ap.add_argument("--grad-mb", type=int, default=2); ap.add_argument("--seed", type=int, default=42)
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
    t = pipe.transformer.to(dev); t = PeftModel.from_pretrained(t, a.lora)
    try: t.set_adapter("default")
    except Exception: t.set_adapter(list(t.peft_config.keys())[0])
    pipe.transformer = t; pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
    scorer = tr._load_reward_scorer("open3", dev)

    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    NS = [n for n in (2, 4, 8, 16, 32, 64) if n <= a.max_n]
    STEP_FRAC = 1.0 * 0.10 / 2          # eta*rho/n_ascent,与训练首步同尺度
    timing = []

    @torch.no_grad()
    def R(x, prm):
        return tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, x), list(prm)).float()

    rows = []
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    for i0 in range(0, len(prompts), a.chunk):
        prm = prompts[i0: i0 + a.chunk]; B = len(prm)
        with torch.no_grad():
            pe, ppe = encode_prompt(tes, tks, prm, 128); ne1, npe1 = encode_prompt(tes, tks, [""], 128)
        pe, ppe = pe.to(dev), ppe.to(dev)
        ne, npe = ne1.to(dev).repeat(B, 1, 1), npe1.to(dev).repeat(B, 1)
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            _, lat, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npe,
                num_inference_steps=10, guidance_scale=1.0, output_type="pt",
                height=512, width=512, noise_level=0.7, deterministic=True,
                solver="dpm2", model_type="sd3")
        sig = pipe.scheduler.sigmas.float().to(dev); sq = float(sig[a.kq])
        zq = torch.stack(lat, dim=1)[:, a.kq].float()
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            tt = torch.full([B], sq * 1000, device=dev, dtype=torch.long)
            v = pipe.transformer(hidden_states=zq.to(pe.dtype), timestep=tt,
                                 encoder_hidden_states=pe, pooled_projections=ppe,
                                 return_dict=False)[0].float()
            y0 = zq - sq * v
        n_y0 = y0.flatten(1).norm(dim=1); step = (STEP_FRAC * n_y0).view(-1, 1, 1, 1)
        unit = lambda d: d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)

        # ---- 成本实测:带梯度 vs no-grad,同一 micro-batch 规模 ----
        torch.cuda.synchronize(); t0 = time.perf_counter()
        xg = y0[: a.grad_mb].detach().clone().requires_grad_(True)
        rg = tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, xg), list(prm[: a.grad_mb]))
        (gp,) = torch.autograd.grad(rg.sum(), xg)
        torch.cuda.synchronize(); t_grad = time.perf_counter() - t0
        del xg, rg; torch.cuda.empty_cache()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        _ = R(y0[: a.grad_mb], prm[: a.grad_mb])
        torch.cuda.synchronize(); t_fwd = time.perf_counter() - t0
        timing.append(dict(mb=a.grad_mb, t_grad=t_grad, t_fwd=t_fwd, ratio=t_grad / max(t_fwd, 1e-9)))

        gparts = [gp.float().detach()]
        for j0 in range(a.grad_mb, B, a.grad_mb):
            xg = y0[j0:j0 + a.grad_mb].detach().clone().requires_grad_(True)
            rg = tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, xg), list(prm[j0:j0 + a.grad_mb]))
            (g2,) = torch.autograd.grad(rg.sum(), xg); gparts.append(g2.float().detach())
            del xg, rg, g2; torch.cuda.empty_cache()
        gstar = torch.cat(gparts, dim=0); del gparts, gp; torch.cuda.empty_cache()

        R0 = R(y0, prm)
        gain_true = (R(y0 + step * unit(gstar), prm) - R0)

        res = {}
        for space in ("full", "low8"):
            cand = []
            for _ in range(a.max_n):
                if space == "full":
                    u = torch.randn(y0.shape, generator=gen, device=dev)
                else:
                    lo = torch.randn((B, y0.shape[1], 8, 8), generator=gen, device=dev)
                    u = torch.nn.functional.interpolate(lo, scale_factor=8, mode="nearest")
                cand.append((R(y0 + step * unit(u), prm) - R0))
            C = torch.stack(cand, dim=1)                     # [B, max_n]
            for n in NS:
                res[f"{space}|N{n}"] = dict(best=C[:, :n].max(dim=1).values.tolist(),
                                            worst=C[:, :n].min(dim=1).values.tolist(),
                                            mean=C[:, :n].mean(dim=1).tolist())
            del cand, C; torch.cuda.empty_cache()

        for b in range(B):
            rows.append(dict(prompt_i=i0 + b, sigma_q=sq, R0=float(R0[b]),
                             gain_true=float(gain_true[b]),
                             bon={k: {m: v[m][b] for m in v} for k, v in res.items()}))
        print(f"  [p5] {i0+B}/{len(prompts)}  sigma_q={sq:.4f}  gain_true={float(gain_true.mean()):+.5f}  "
              f"grad/fwd={timing[-1]['ratio']:.2f}", flush=True)

    json.dump(dict(meta=dict(lora=a.lora, kq=a.kq, ns=NS, step_frac=STEP_FRAC, n=len(rows),
                             timing=timing,
                             note="best-of-N 用 N 次 reward 前向;P4 的对偶零阶 k 对用 2k 次 ⇒ N=2k 等预算"),
                   rows=rows), open(a.out, "w"))
    print("  [p5] wrote", a.out, flush=True)

main()
