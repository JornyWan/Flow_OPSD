"""P1:Open3 的「可达奖励范围」随噪声层级的分布 —— reward 到底看不看得见 layout。

做法(**用扩散过程本身定义两类改动**,不用离线滤镜那种 off-manifold 的伪变换):
  对每条 prompt 先生成一张 base 图 x0;再把 x0 **重新加噪到 σ_j** 后用不同随机种子去噪。
  σ_j 越大,被重采样的结构层级越高:σ≈0.28 只换纹理/细节,σ≈0.91 连 layout 一起换。
  于是「在层级 ≤ σ_j 内能达到的奖励散布」就是 OPD 在该 σ 上的**奖励上限的代理**。

判读:sd_j / sd_full 接近 1 ⇒ 奖励的变化几乎全部来自低噪细节 ⇒ 高噪区在这个 reward 上没有空间;
      接近 0 ⇒ 奖励主要由 layout 决定 ⇒ 高噪区才是余量所在。
同时报 best_j − mean_j(best-of-S 优化器真正能拿到的增益)与三个分量。

重加噪用 flow-matching 约定 z = (1−σ)x0 + σ·ε(与 _flow_continue_from 的 v=(z−x0)/σ 自洽);
去噪用与 `_flow_continue_from` 相同的 Euler 更新 z += v(z,σ_i)·(σ_{i+1}−σ_i)。
"""
import argparse, importlib.util, json, os, sys
import torch
from torch.cuda.amp import autocast as torch_autocast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
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
    for m in (pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3): m.to(dev, dtype=torch.float16)
    t = pipe.transformer.to(dev); t = PeftModel.from_pretrained(t, a.lora)
    try: t.set_adapter("default")
    except Exception: t.set_adapter(list(t.peft_config.keys())[0])
    pipe.transformer = t; pipe.transformer.eval(); pipe.set_progress_bar_config(disable=True)
    scorer = tr._load_reward_scorer("open3", dev)
    SUBS = ("pickscore", "clipscore", "hpsv2")

    prompts = [l.strip() for l in open(a.prompts) if l.strip()][: a.n_prompts]
    tes = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
    tks = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]

    pipe.scheduler.set_timesteps(a.steps)
    sig = pipe.scheduler.sigmas.float().to(dev)
    TARGETS = [0.278, 0.465, 0.707, 0.913, 1.000]      # 与训练 10 步网格上的 kq=8/7/5/2/0 对应
    LEVELS = []
    for s in TARGETS:
        j = int(torch.argmin((sig[: a.steps] - s).abs()).item())
        LEVELS.append((j, float(sig[j])))
    print("  [p1] 层级 (idx, sigma, 剩余步数):",
          [(j, round(s, 3), a.steps - j) for j, s in LEVELS], flush=True)

    @torch.no_grad()
    def vel(z, s, pe, ppe):
        tt = torch.full([z.shape[0]], float(s) * 1000, device=dev, dtype=torch.long)
        with torch_autocast(enabled=True, dtype=torch.float16):
            return pipe.transformer(hidden_states=z.to(pe.dtype), timestep=tt,
                                    encoder_hidden_states=pe, pooled_projections=ppe,
                                    return_dict=False)[0].float()

    @torch.no_grad()
    def denoise_from(z, j, pe, ppe):
        for i in range(j, len(sig) - 1):
            z = z + vel(z, sig[i], pe, ppe) * (sig[i + 1] - sig[i])
        return z

    @torch.no_grad()
    def score_all(x, prm):
        img = tr._decode01(pipe, x)
        out = {k: tr._reward_scores_grad(scorer[k], k, img, list(prm)).float() for k in SUBS}
        out["open3"] = sum(out[k] for k in SUBS)
        return out

    rows = []
    for i0 in range(0, len(prompts), a.chunk):
        prm = prompts[i0: i0 + a.chunk]; B = len(prm)
        with torch.no_grad():
            pe, ppe = encode_prompt(tes, tks, prm, 128)
            ne1, npe1 = encode_prompt(tes, tks, [""], 128)
        pe, ppe = pe.to(dev), ppe.to(dev)
        ne, npe = ne1.to(dev).repeat(B, 1, 1), npe1.to(dev).repeat(B, 1)
        g = torch.Generator(device=dev).manual_seed(a.seed + i0)
        with torch.no_grad(), torch_autocast(enabled=True, dtype=torch.float16):
            _, lat, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=ne, negative_pooled_prompt_embeds=npe,
                num_inference_steps=a.steps, guidance_scale=1.0, output_type="pt",
                height=512, width=512, noise_level=0.7, deterministic=True,
                solver="flow", model_type="sd3")
        x0 = torch.stack(lat, dim=1)[:, -1].float()
        base = score_all(x0, prm)

        per = {}   # (j) -> {metric: [B][S]}
        for j, s in LEVELS:
            acc = {k: [] for k in list(SUBS) + ["open3"]}
            for _ in range(a.seeds):
                eps = torch.randn(x0.shape, generator=g, device=dev)
                z = (1.0 - s) * x0 + s * eps
                sc = score_all(denoise_from(z, j, pe, ppe), prm)
                for k in acc: acc[k].append(sc[k])
            per[j] = {k: torch.stack(v, dim=1) for k, v in acc.items()}   # [B,S]
        for b in range(B):
            r = dict(prompt_i=i0 + b, base={k: float(base[k][b]) for k in base}, levels={})
            for j, s in LEVELS:
                r["levels"][str(j)] = dict(sigma=s, remaining=a.steps - j,
                    **{k: per[j][k][b].tolist() for k in per[j]})
            rows.append(r)
        m = per[LEVELS[0][0]]["open3"].std(dim=1).mean()
        M = per[LEVELS[-1][0]]["open3"].std(dim=1).mean()
        print(f"  [p1] {i0+B}/{len(prompts)}  sd(sigma={LEVELS[0][1]:.3f})={m:.4f}  "
              f"sd(full)={M:.4f}  ratio={m/M:.3f}", flush=True)

    json.dump(dict(meta=dict(lora=a.lora, steps=a.steps, seeds=a.seeds,
                             levels=[dict(idx=j, sigma=s) for j, s in LEVELS], n=len(rows)),
                   rows=rows), open(a.out, "w"))
    print("  [p1] wrote", a.out, flush=True)

main()
