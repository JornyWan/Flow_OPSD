"""P7:RB-LT 的 target 到底是不是一个"改善"?(§4.829 预登记)

canonical OPSD 的 d 在 y0 处算 ⇒ R(y0+d) > R(y0) 按构造成立。
RB-LT 的 d 在真实终点 x 处算,而 target 是 ŷ_old(z_k)+d ⇒ 没有保证。

逐 σ 测四个量(定义见 §4.829 §三):g_rblt / g_canon / g_end / g_rand。
复用 P6 的加载与 rollout 骨架,一行不重写。
"""
_P6DOC = """P6:RB-LT 的一阶局部性前提是否成立(§4.816 预登记)。

在 ep0 工作点(θ=θ_old,即只挂 LoRA 不更新)上,对 8 个合格 σ 态逐个测 x0 空间残差 RMS:
  r_ref    : 输入 σε+(1−σ)x      目标 x          —— 流形内地板
  r_rbltg  : 输入 z_k+(1−σ)d     目标 ŷ_old(z_k)+d
  r_rand   : 同上但 d→同范数随机方向            —— 分开"方向特异"与"量级"
  r_bridge : 输入 σε+(1−σ)(x+d)  目标 x+d

全部对 r_ref 归一(ρ_X = r_X/r_ref),避免串未核常数(§4.816 §一)。

复用:pipeline_with_logprob / _opa_tr_step / _decode01 / _load_reward_scorer —— 一行不重写。
配置照 rblt 臂抄死:solver=flow, deterministic=True, noise_level=0.7, 10 步,
rho=0.10, n_ascent=2, eta=1.0, dir_mode=grad, kind=pickscore。
"""
import argparse, importlib.util, json, os, sys
import torch
from torch.cuda.amp import autocast as torch_autocast

RHO, NASC, ETA, DIRMODE = 0.10, 2, 1.0, "grad"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="pickscore")
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-sigma", type=float, default=0.278)
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
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    rms = lambda z: (z.double().flatten(1).pow(2).mean(dim=1).sqrt())
    @torch.no_grad()
    def R(z, prm):
        return tr._reward_scores_grad(scorer, a.kind, tr._decode01(pipe, z), list(prm)).float()
    mad = lambda z: (z.double().flatten(1).abs().mean(dim=1))   # = 训练里的 wf_p

    rows = []
    for i0 in range(0, len(prompts), a.chunk):
        prm = prompts[i0: i0 + a.chunk]; B = len(prm)
        with torch.no_grad():
            pe, ppe = encode_prompt(tes, tks, prm, 128)
            ne1, npe1 = encode_prompt(tes, tks, [""], 128)
        pe, ppe = pe.to(dev), ppe.to(dev)
        ne, npe = ne1.to(dev).repeat(B, 1, 1), npe1.to(dev).repeat(B, 1)
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            _, lat, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npe,
                num_inference_steps=10, guidance_scale=1.0, output_type="pt",
                height=512, width=512, noise_level=0.7, deterministic=True,
                solver="flow", model_type="sd3")
        sig = pipe.scheduler.sigmas.float().to(dev)
        Z = torch.stack(lat, dim=1).float()          # [B, nstates, C, H, W]
        eps = Z[:, 0]                                 # = 训练脚本的 rblt_eps
        x   = Z[:, -1]                                # 真实 rollout 终点(= latents_clean)

        # d:复用官方同一个 trust-region step,终点 x 上构造(与 L2071 的 _rb_d 同式)
        xp = tr._opa_tr_step(pipe, scorer, a.kind, x, prm, RHO, NASC, ETA, +1.0, DIRMODE, x)
        d  = (xp.float() - x).detach()
        dr = torch.randn(d.shape, generator=gen, device=dev)
        dr = dr / dr.flatten(1).norm(dim=1).view(-1,1,1,1) * d.flatten(1).norm(dim=1).view(-1,1,1,1)

        def yhat(z, s):
            with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
                tt = torch.full([B], s * 1000, device=dev, dtype=torch.long)
                v = pipe.transformer(hidden_states=z.to(pe.dtype), timestep=tt,
                                     encoder_hidden_states=pe, pooled_projections=ppe,
                                     return_dict=False)[0].float()
            return z - s * v

        # 与生产端 L1414 严格一致:num_states = latents.shape[1] - 1
        nstates = Z.shape[1] - 1
        elig = [k for k in range(1, nstates) if float(sig[k]) >= a.query_sigma]
        if not elig:
            raise RuntimeError(f"无合格态:sig={[float(v) for v in sig[:Z.shape[1]]]} sstar={a.query_sigma}")
        R_x  = R(x, prm)
        R_xd = R(x + d, prm)                       # g_end:d 在自己的锚点上是否上升
        for k in elig:
            s_ = float(sig[k]); zk = Z[:, k]
            y_old = yhat(zk, s_)                   # = ŷ_old(z_k) = y0(z_k)
            # canonical 的同一个量:d_loc 在 y0 处算(复用官方 _opa_tr_step,同 rho/n_ascent/eta)
            yp_loc = tr._opa_tr_step(pipe, scorer, a.kind, y_old, prm,
                                     RHO, NASC, ETA, +1.0, DIRMODE, x)
            dr2 = torch.randn(d.shape, generator=gen, device=dev)
            dr2 = dr2 / dr2.flatten(1).norm(dim=1).view(-1,1,1,1) * d.flatten(1).norm(dim=1).view(-1,1,1,1)
            R_y    = R(y_old, prm)
            R_yd   = R(y_old + d, prm)             # RB-LT 的 target
            R_yloc = R(yp_loc, prm)                # canonical 的 target
            R_yr   = R(y_old + dr2, prm)           # 同范数随机方向
            for b in range(B):
                rows.append(dict(prompt_i=i0+b, k=k, sigma=s_,
                                 g_rblt=float(R_yd[b]-R_y[b]),
                                 g_canon=float(R_yloc[b]-R_y[b]),
                                 g_end=float(R_xd[b]-R_x[b]),
                                 g_rand=float(R_yr[b]-R_y[b]),
                                 R_y=float(R_y[b]), R_x=float(R_x[b]),
                                 norm_d=float(d.flatten(1).norm(dim=1)[b]),
                                 norm_dloc=float((yp_loc-y_old).flatten(1).norm(dim=1)[b])))
            print(f"  [p7] {i0+B}/{len(prompts)} k={k} sig={s_:.4f} "
                  f"g_rblt={float((R_yd-R_y).mean()):+.5f} "
                  f"g_canon={float((R_yloc-R_y).mean()):+.5f} "
                  f"g_rand={float((R_yr-R_y).mean()):+.5f} "
                  f"g_end={float((R_xd-R_x).mean()):+.5f}", flush=True)
            del yp_loc, dr2
        del Z, lat; torch.cuda.empty_cache()

    json.dump(dict(rows=rows, cfg=dict(kind=a.kind, rho=RHO, n_ascent=NASC, eta=ETA,
                                       dir_mode=DIRMODE, solver="flow", lora=a.lora,
                                       n_prompts=len(prompts), seed=a.seed)),
              open(a.out, "w"))
    print("wrote", a.out)

if __name__ == "__main__":
    main()
