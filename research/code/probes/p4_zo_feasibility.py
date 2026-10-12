"""P4:零阶方向可行性 + reward 梯度的频谱分解。

回答两件事(都在 canonical 工作点:baseline ck100、kq=8/σ=0.278、Open3):
  (1) ∇R 有多少能量落在**低频(layout 尺度)子空间**里 —— 这同时是 P1 的频谱版答案;
  (2) k 次 query 的零阶估计与真 ∇R 的 cos,以及**沿该方向真实走一步的 reward 增益**
      (后者才是决策量:OPSD 只用方向,不用幅度)。

与既有量的关系:§4.375 的 gain_clean 是在**exact suffix 终点**上测的;本脚本在 y0 上测,
**是不同的量,不可混用**。

复用:pipeline_with_logprob(与训练同一条采样路径)、_decode01、_load_reward_scorer、
_reward_scores_grad(全部 import 自训练脚本,不重写)。
"""
import argparse, importlib.util, json, math, os, sys, time
import torch
from torch.cuda.amp import autocast as torch_autocast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--kq", type=int, default=8)
    ap.add_argument("--max-pairs", type=int, default=32)   # 对偶采样对数上限(k 取嵌套子集)
    ap.add_argument("--grad-mb", type=int, default=2)   # 真梯度的微批(照训练的 opa_mb=6 思路;VAE 反传激活是大头)
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
    t = PeftModel.from_pretrained(t, a.lora)
    try: t.set_adapter("default")
    except Exception: t.set_adapter(list(t.peft_config.keys())[0])
    pipe.transformer = t; pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
    scorer = tr._load_reward_scorer("open3", dev)

    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]

    BLOCKS = [1, 2, 4, 8, 16, 32, 64]          # 频谱:块常数子空间的边长(latent 是 64x64)
    KS     = [k for k in (1, 2, 4, 8, 16, 32) if k <= a.max_pairs]   # 零阶对偶采样对数
    MUS    = [0.005, 0.02, 0.05]               # 平滑半径 / ||y0||
    RHO, ETA, NASC = 0.10, 1.0, 2
    STEP_FRAC = ETA * RHO / NASC               # = 0.05,与训练里第一步位移同尺度

    def proj_block(g, s):
        """投到'每 s×s 块内常数'的子空间(avgpool 再最近邻上采样 = 正交投影)。"""
        if s == 1: return g
        return torch.nn.functional.interpolate(
            torch.nn.functional.avg_pool2d(g, s), scale_factor=s, mode="nearest")

    @torch.no_grad()
    def R(x, prm):
        return tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, x), list(prm)).float()

    rows = []
    gen = torch.Generator(device=dev).manual_seed(a.seed)
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
                solver="dpm2", model_type="sd3")
        sig = pipe.scheduler.sigmas.float().to(dev)
        zq = torch.stack(lat, dim=1)[:, a.kq].float()
        sq = float(sig[a.kq])
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            tt = torch.full([B], sq * 1000, device=dev, dtype=torch.long)
            v = pipe.transformer(hidden_states=zq.to(pe.dtype), timestep=tt,
                                 encoder_hidden_states=pe, pooled_projections=ppe,
                                 return_dict=False)[0].float()
            y0 = zq - sq * v
        n_y0 = y0.flatten(1).norm(dim=1)
        step = (STEP_FRAC * n_y0).view(-1, 1, 1, 1)

        # 真梯度按微批算:512² 的 VAE 解码器反传要存全部激活,batch 8 会把 80GB 打爆。
        gparts = []
        for j0 in range(0, B, a.grad_mb):
            xg = y0[j0:j0 + a.grad_mb].detach().clone().requires_grad_(True)
            rg = tr._reward_scores_grad(scorer, "open3", tr._decode01(pipe, xg),
                                        list(prm[j0:j0 + a.grad_mb]))
            (gp,) = torch.autograd.grad(rg.sum(), xg)
            gparts.append(gp.float().detach())
            del xg, rg, gp; torch.cuda.empty_cache()
        gstar = torch.cat(gparts, dim=0); del gparts

        R0 = R(y0, prm)
        unit = lambda d: d / (d.flatten(1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)
        gain = lambda d: (R(y0 + step * unit(d), prm) - R0)

        spec = {s: (proj_block(gstar, s).flatten(1).norm(dim=1) /
                    (gstar.flatten(1).norm(dim=1) + 1e-12)).tolist() for s in BLOCKS}
        g_true = gain(gstar)
        d_rand = torch.randn(y0.shape, generator=gen, device=dev)
        g_rand = gain(d_rand)

        zo = {}
        for space in ("full", "low8"):
            us, ds = [], []
            for j in range(a.max_pairs):
                if space == "full":
                    u = torch.randn(y0.shape, generator=gen, device=dev)
                else:                                      # 16x8x8=1024 维低频子空间
                    lo = torch.randn((B, y0.shape[1], 8, 8), generator=gen, device=dev)
                    u = torch.nn.functional.interpolate(lo, scale_factor=8, mode="nearest")
                u = unit(u); us.append(u)
                ds.append({})
            for mu in MUS:
                r = mu * n_y0.view(-1, 1, 1, 1)
                dif = [ (R(y0 + r * u, prm) - R(y0 - r * u, prm)) for u in us ]
                for k in KS:
                    gh = sum(dif[j].view(-1, 1, 1, 1) * us[j] for j in range(k))
                    cs  = torch.nn.functional.cosine_similarity(gh.flatten(1), gstar.flatten(1), dim=1)
                    csl = torch.nn.functional.cosine_similarity(
                        gh.flatten(1), proj_block(gstar, 8).flatten(1), dim=1)
                    zo[f"{space}|mu{mu}|k{k}"] = dict(cos=cs.tolist(), cos_low8=csl.tolist(),
                                                      gain=gain(gh).tolist())
            del us, ds; torch.cuda.empty_cache()

        for b in range(B):
            rows.append(dict(prompt_i=i0 + b, sigma_q=sq, norm_y0=float(n_y0[b]),
                             R0=float(R0[b]), gain_true=float(g_true[b]), gain_rand=float(g_rand[b]),
                             spec={str(s): spec[s][b] for s in BLOCKS},
                             zo={k: dict(cos=v["cos"][b], cos_low8=v["cos_low8"][b], gain=v["gain"][b])
                                 for k, v in zo.items()}))
        print(f"  [p4] {i0+B}/{len(prompts)}  sigma_q={sq:.4f}  gain_true={float(g_true.mean()):+.5f}",
              flush=True)

    json.dump(dict(meta=dict(lora=a.lora, kq=a.kq, n=len(rows), blocks=BLOCKS, ks=KS, mus=MUS,
                             step_frac=STEP_FRAC, rho=RHO, eta=ETA, n_ascent=NASC,
                             note="gain 在 y0 上测,与 §4.375 的 endpoint gain 不是同一个量"),
                   rows=rows), open(a.out, "w"))
    print("  [p4] wrote", a.out, flush=True)

main()
