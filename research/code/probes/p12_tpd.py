"""P12 = TPD 的 S0(§4.866):零训练筛选 —— 在高噪祖先态上,哪种"目标方向"注入后,flow-40 终点 reward 涨得最多。

完全复用 P10(§4.842)的机制:flow-40 部署 sampler(cross_eval 同参:solver=flow、deterministic、40 步、guidance=1.0、512²)、
v(z,s)(timestep = σ·1000 取 long)、Euler 后缀、注入公式 z_s ← z_s0 + (h/q)·d、tr 的 reward 与 decode。
在态 i(σ_i)上,同半径 r·‖y0,i‖ 下比较的方向:
  grad   unit(∇R(y0,i))                                  该态自己的一阶(OPSD 的一步搬到高噪)
  msp    unit(ȳ_q − y0,i),ȳ_q = y0,q + ρ‖y0,q‖·unit(∇R(y0,q))  MSP 的传播目标(q = 离 0.278 最近的态)
  cons   unit(y0,q − y0,i)                               只有一致性(未改进的终点)
  tpd8   unit(Σ_m (w_m − 1/M) x_m),M=8                  TPD:从 z_i 用当前模型的 SDE(eta)走完网格的 M 条续采,w = softmax(标准化 reward)
  tpd4   同上,只用前 4 条
  tpd8f  unit(Σ_m w_m x_m − y0,i)                       不带对照变量
  tpd8h  unit(x_best − x̄)                               硬选择(旧 hybrid 的做法)
  rand
另记后验宽度 ‖x_m − x̄‖/‖x̄‖ 与 ‖x̄ − y0,i‖/‖y0,i‖、方向两两 cos。
"""
import argparse, importlib.util, json, os, sys, time
import torch
from torch.cuda.amp import autocast as torch_autocast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="open3")
    ap.add_argument("--n-prompts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--states", default="0.96,0.858,0.602,0.465")
    ap.add_argument("--qsig", type=float, default=0.278)
    ap.add_argument("--radii", default="0.05,0.10")
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--rho", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--bs", type=int, default=5)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, "src")); sys.path.insert(0, a.repo)
    spec = importlib.util.spec_from_file_location("tr", os.path.join(a.repo, "scripts/train_opsd_ri_sd3.py"))
    tr = importlib.util.module_from_spec(spec); sys.modules["tr"] = tr; spec.loader.exec_module(tr)
    from diffusers import StableDiffusion3Pipeline
    from peft import PeftModel
    from diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
    from diffusers_patch.solver import flow_grpo_step
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
    idx = [i for i in range(len(prompts)) if i % a.nshard == a.shard]
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    states = [float(x) for x in a.states.split(",")]; radii = [float(x) for x in a.radii.split(",")]
    gen_r = torch.Generator(device=dev).manual_seed(a.seed + 777 + a.shard)

    def R(x, prm):
        return tr._reward_scores_grad(scorer, a.kind, tr._decode01(pipe, x), list(prm)).float()

    unit = lambda u: u / (u.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
    rows, checks, t0 = [], [], time.time()
    for pi in idx:
        prompt = prompts[pi]
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
        Z = torch.stack(lat, dim=1).float()

        def v(z, s):  # 与 v_pred_fn 一致
            Bz = z.shape[0]
            with torch_autocast(enabled=True, dtype=torch.float16):
                tt = torch.full([Bz], float(s) * 1000, device=dev, dtype=torch.long)
                out = pipe.transformer(hidden_states=z.to(pe.dtype), timestep=tt,
                                       encoder_hidden_states=pe.expand(Bz, -1, -1),
                                       pooled_projections=ppe.expand(Bz, -1), return_dict=False)[0]
            return out.float()

        def suffix(z, i0):  # 确定性 Euler,与 P10 相同
            for i in range(i0, a.steps):
                z = z + v(z, float(sig[i])) * float(sig[i + 1] - sig[i])
            return z

        def sde_suffix(z, i0, gen):  # 随机续采:flow_grpo_step(eta>0),与训练 rollout 的随机 sampler 同一函数
            for i in range(i0, a.steps):
                z, _, _ = flow_grpo_step(model_output=v(z, float(sig[i])), latents=z, eta=a.eta,
                                         sigmas=sig, index=i, prev_sample=None, generator=gen)
                z = z.float()
            return z

        if pi == idx[0]:
            with torch.no_grad():
                for i in (5, 20, 35):
                    pred = Z[:, i] + v(Z[:, i], float(sig[i])) * float(sig[i + 1] - sig[i])
                    checks.append(dict(prompt_i=pi, i=i, rel_err_vs_step=float(
                        (pred - Z[:, i + 1]).norm() / (Z[:, i + 1] - Z[:, i]).norm())))
        # query 态的改进终点(MSP 的传播目标)
        iq = int(torch.argmin((sig[: a.steps - 1] - a.qsig).abs()).item())
        with torch.no_grad():
            y0q = Z[:, iq] - float(sig[iq]) * v(Z[:, iq], float(sig[iq]))
        yq = y0q.clone().requires_grad_(True)
        (gq,) = torch.autograd.grad(R(yq, [prompt]).sum(), yq)
        ybar_q = (y0q + a.rho * y0q.flatten(1).norm() * unit(gq.float())).detach()
        for st in states:
            i = int(torch.argmin((sig[: a.steps - 1] - st).abs()).item())
            q, s = float(sig[i]), float(sig[i + 1]); h = q - s
            zi = Z[:, i]
            with torch.no_grad():
                vi = v(zi, q); y0i = zi - q * vi; zs0 = zi - h * vi
            ny0 = float(y0i.flatten(1).norm())
            yy = y0i.clone().requires_grad_(True)
            (gi,) = torch.autograd.grad(R(yy, [prompt]).sum(), yy)
            with torch.no_grad():
                gc = torch.Generator(device=dev).manual_seed(a.seed * 1000 + pi * 17 + i)
                X = sde_suffix(zi.expand(a.M, -1, -1, -1).clone(), i, gc)       # [M,C,H,W]
                Rm = torch.cat([R(X[j:j + 4], [prompt] * X[j:j + 4].shape[0]) for j in range(0, a.M, 4)], 0)
                def tilt(Xs, Rs):
                    A = (Rs - Rs.mean()) / (Rs.std() + 1e-8)
                    w = torch.softmax(A, 0).view(-1, 1, 1, 1)
                    return w, Xs.mean(0, keepdim=True)
                w8, xb8 = tilt(X, Rm); w4, xb4 = tilt(X[:4], Rm[:4])
                d = {
                    "grad": gi.float(),
                    "msp": ybar_q - y0i,
                    "cons": y0q - y0i,
                    "tpd8": ((w8 - 1.0 / a.M) * X).sum(0, keepdim=True),
                    "tpd4": ((w4 - 0.25) * X[:4]).sum(0, keepdim=True),
                    "tpd8f": (w8 * X).sum(0, keepdim=True) - y0i,
                    "tpd8h": X[int(Rm.argmax())].unsqueeze(0) - xb8,
                    "rand": torch.randn(y0i.shape, generator=gen_r, device=dev),
                }
                U = {k: unit(val) for k, val in d.items()}
                keys = [("base", 0.0)] + [(k, r) for r in radii for k in d]
                starts = torch.cat([zs0 if k == "base" else zs0 + (h / q) * (r * ny0) * U[k] for k, r in keys], 0)
                ends = torch.cat([suffix(starts[j:j + a.bs], i + 1) for j in range(0, starts.shape[0], a.bs)], 0)
                Re = torch.cat([R(ends[j:j + a.bs], [prompt] * ends[j:j + a.bs].shape[0])
                                for j in range(0, ends.shape[0], a.bs)], 0)
            base = float(Re[0])
            row = dict(prompt_i=pi, state=st, i=i, sigma=q, norm_y0=ny0, R_base=base,
                       R_cont_mean=float(Rm.mean()), R_cont_std=float(Rm.std()), R_cont_max=float(Rm.max()),
                       width=float(((X - xb8).flatten(1).norm(dim=1) / xb8.flatten(1).norm()).mean()),
                       mean_vs_y0=float((xb8 - y0i).flatten(1).norm() / ny0),
                       norm_dir={k: float(val.flatten(1).norm()) for k, val in d.items()})
            row["cos"] = {f"{p}|{r2}": float((U[p] * U[r2]).sum())
                          for p in ("tpd8", "msp", "grad") for r2 in ("msp", "grad", "cons", "tpd8h") if p != r2}
            for j, (k, r) in enumerate(keys[1:], 1):
                row[f"gain_{k}_{r}"] = float(Re[j]) - base
            rows.append(row)
            print(f"[p12] p{pi} σ={q:.3f}(i={i}) width={row['width']:.3f} | r=0.10 gain "
                  + " ".join(f"{k}={row.get(f'gain_{k}_0.1', float('nan')):+.4f}" for k in d)
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
            del X, ends, starts, U, d; torch.cuda.empty_cache()
    tmp = a.out + ".tmp"
    json.dump(dict(rows=rows, checks=checks, cfg=vars(a)), open(tmp, "w")); os.replace(tmp, a.out)
    print("[p12] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
