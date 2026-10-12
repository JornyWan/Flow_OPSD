"""P10(§4.842):suffix-aware 方向的第一阶段诊断 —— 不训练。

在 flow-40 部署 sampler(照抄 cross_eval.py:solver=flow、deterministic、40 步、guidance=1.0)上,
于 query 态 q 注入 clean-output 修改 d(Euler 下等价于 z_s ← z_s⁰ + (h/q)·d),跑完剩余采样,
比较四个单位方向在同一半径下的**真实终点 reward** 增益:
  c   = ∇_y R(D(y0))                                   canonical 局部代理
  L1  = ∇_d R(D(z_s(d) − s·v(z_s(d), s)))|₀           一步前瞻(保留对输入的梯度通路)
  F   = ∇_d R(D(Suffix(z_s(d))))|₀                    完整 flow-40 suffix(逐步 gradient checkpoint)
  rand
复用 tr._load_reward_scorer / _reward_scores_grad / _decode01 / pipeline_with_logprob,不重写。
timestep 照 pipeline 的 v_pred_fn:sigma*1000 取 long。
"""
import argparse, importlib.util, json, os, sys, time
import torch
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast as torch_autocast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="open3")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--qsig", default="0.278,0.465,0.602")
    ap.add_argument("--radii", default="0.05,0.10")
    ap.add_argument("--seed", type=int, default=42)
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
    for m in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3):
        m.to(dev, dtype=torch.float16)
    t = pipe.transformer.to(dev)
    if a.lora.lower() != "none":
        t = PeftModel.from_pretrained(t, a.lora)
        try: t.set_adapter("default")
        except Exception: t.set_adapter(list(t.peft_config.keys())[0])
    pipe.transformer = t; pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
    scorer = tr._load_reward_scorer(a.kind, dev)

    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    qsigs = [float(x) for x in a.qsig.split(",")]; radii = [float(x) for x in a.radii.split(",")]
    gen_r = torch.Generator(device=dev).manual_seed(a.seed + 777)

    def R(x, prm):  # 可微当 x.requires_grad
        return tr._reward_scores_grad(scorer, a.kind, tr._decode01(pipe, x), list(prm)).float()

    rows, checks = [], []
    t0 = time.time()
    for pi, prompt in enumerate(prompts):
        with torch.no_grad():
            pe, ppe = encode_prompt(tes, tks, [prompt], 128)
            ne, npe = encode_prompt(tes, tks, [""], 128)
        pe, ppe, ne, npe = pe.to(dev), ppe.to(dev), ne.to(dev), npe.to(dev)
        g = torch.Generator(device=dev).manual_seed(a.seed + pi)
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            _, lat, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npe,
                num_inference_steps=a.steps, guidance_scale=1.0, output_type="pt",
                height=512, width=512, noise_level=0.7, deterministic=True,
                solver="flow", model_type="sd3", generator=g)
        sig = pipe.scheduler.sigmas.float().to(dev)
        Z = torch.stack(lat, dim=1).float()            # [1, steps+1, C, H, W]

        def v(z, s):  # 与 v_pred_fn 一致:timestep = sigma*1000 取 long
            Bz = z.shape[0]
            with torch_autocast(enabled=True, dtype=torch.float16):
                tt = torch.full([Bz], float(s) * 1000, device=dev, dtype=torch.long)
                out = pipe.transformer(hidden_states=z.to(pe.dtype), timestep=tt,
                                       encoder_hidden_states=pe.expand(Bz, -1, -1),
                                       pooled_projections=ppe.expand(Bz, -1), return_dict=False)[0]
            return out.float()

        def suffix(z, i0, grad=False):
            for i in range(i0, a.steps):
                dt = float(sig[i + 1] - sig[i]); s_i = float(sig[i])
                if grad:
                    z = checkpoint(lambda zz, s_i=s_i, dt=dt: zz + v(zz, s_i) * dt, z, use_reentrant=False)
                else:
                    z = z + v(z, s_i) * dt
            return z

        # 数值自检:我的 Euler 能否复现采样器的下一态
        if pi < 4:
            with torch.no_grad():
                for i in (10, 25, 35):
                    pred = Z[:, i] + v(Z[:, i], float(sig[i])) * float(sig[i + 1] - sig[i])
                    rel = float((pred - Z[:, i + 1]).norm() / (Z[:, i + 1] - Z[:, i]).norm())
                    checks.append(dict(prompt_i=pi, i=i, rel_err_vs_step=rel))
        for qt in qsigs:
            i = int(torch.argmin((sig[: a.steps - 1] - qt).abs()).item())
            q, s = float(sig[i]), float(sig[i + 1]); h = q - s
            zq = Z[:, i]
            with torch.no_grad():
                vq = v(zq, q); y0 = zq - q * vq; zs0 = zq - h * vq
            ny0 = float(y0.flatten(1).norm())
            # c
            y = y0.clone().requires_grad_(True)
            (gc,) = torch.autograd.grad(R(y, [prompt]).sum(), y)
            # L1
            d1 = torch.zeros_like(y0, requires_grad=True)
            zs = zs0 + (h / q) * d1
            Hd = zs - s * v(zs, s)
            (g1,) = torch.autograd.grad(R(Hd, [prompt]).sum(), d1)
            del zs, Hd; torch.cuda.empty_cache()
            # F
            dF = torch.zeros_like(y0, requires_grad=True)
            xend = suffix(zs0 + (h / q) * dF, i + 1, grad=True)
            (gF,) = torch.autograd.grad(R(xend, [prompt]).sum(), dF)
            del xend; torch.cuda.empty_cache()
            gr = torch.randn(y0.shape, generator=gen_r, device=dev)
            unit = lambda u: u / (u.flatten(1).norm() + 1e-12)
            U = {"c": unit(gc.float()), "L1": unit(g1.float()), "F": unit(gF.float()), "rand": unit(gr)}
            cos = lambda p, r: float((U[p] * U[r]).sum())
            # 评测:同一批起点一起跑完 suffix
            keys = [("base", 0.0)] + [(k, r) for r in radii for k in ("c", "L1", "F", "rand")]
            starts = torch.cat([zs0 if k == "base" else zs0 + (h / q) * (r * ny0) * U[k] for k, r in keys], 0)
            with torch.no_grad():
                ends = torch.cat([suffix(starts[j:j + 3], i + 1) for j in range(0, starts.shape[0], 3)], 0)
                Re = torch.cat([R(ends[j:j + 3], [prompt] * ends[j:j + 3].shape[0]) for j in range(0, ends.shape[0], 3)], 0)
                loc = {}
                for k in ("c", "F"):
                    for r in radii:
                        loc[(k, r)] = float(R(y0 + (r * ny0) * U[k], [prompt])[0] - R(y0, [prompt])[0])
            base = float(Re[0])
            row = dict(prompt_i=pi, qt=qt, i=i, sigma_q=q, sigma_s=s, norm_y0=ny0, R_base=base,
                       cos_c_F=cos("c", "F"), cos_L1_F=cos("L1", "F"), cos_c_L1=cos("c", "L1"))
            for j, (k, r) in enumerate(keys[1:], 1):
                row[f"gain_{k}_{r}"] = float(Re[j]) - base
            for (k, r), val in loc.items():
                row[f"local_{k}_{r}"] = val
            rows.append(row)
            print(f"  [p10] p{pi+1}/{len(prompts)} q={q:.4f}(i={i}) cos(c,F)={row['cos_c_F']:+.3f} "
                  f"cos(L1,F)={row['cos_L1_F']:+.3f} | r=0.10 gain c={row.get('gain_c_0.1',float('nan')):+.4f} "
                  f"L1={row.get('gain_L1_0.1',float('nan')):+.4f} F={row.get('gain_F_0.1',float('nan')):+.4f} "
                  f"rand={row.get('gain_rand_0.1',float('nan')):+.4f}  [{time.time()-t0:.0f}s]", flush=True)
            del gc, g1, gF, U, starts, ends; torch.cuda.empty_cache()
    json.dump(dict(rows=rows, checks=checks, cfg=vars(a)), open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
