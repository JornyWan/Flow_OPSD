"""P11 第二段(§4.850):A→B→C→D 分层 —— 把"本批梯度造成的参数位移"逐步接进 flow-40 部署 sampler。

参数:pre = canonical ck300;work = θ300 + Δ0 + k·δ_f —— k=1 即 fork 的**真实一次更新** θ301_f(第一段从 optimizer 状态精确还原;
Δ0 = 共用的动量+wd 那一步,δ_f = 本批梯度造成的部分);组合 "M" = θ300 + Δ0(只有动量,无本批梯度)。
(§4.850 §八:逐图响应在一步尺度上已非线性,故不做放大/中心差分,直接测真实更新,单侧。)
i* = flow-40 网格上离 σ*=0.278 最近的步(照抄 P10 的取法)。每个组合四格:
  B  只在 i* 这一步用 work,其余全 pre      —— 学生在 query 处是否兑现了 target
  C  i* 及之后用 work,之前 pre               —— 加上 query 之后各步的共享参数变化
  P  i* 之前用 work,之后 pre                 —— query 之前(高噪)各步的共享参数变化
  D  全程 work                               —— 真实 policy 改进
Pre = 全程 pre。

sampler、LoRA 装载、文本编码、打分全部复用 pristine 仓库 cross_eval.py 与 diffusionopsd 的实现:
Pre 轨迹直接由 pipeline_with_logprob 生成;各格的分段续跑用同一个 run_sampling,v_pred 照抄 pipeline 的 v_pred_fn。
--selftest 核三件事(必须逐位相同):分段续跑 == pipeline 终点;我的 decode == pipeline 出图;δ=0 的 work == pre。
"""
import argparse, importlib.util, json, os, sys, time
import torch
from torch.cuda.amp import autocast as torch_autocast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="pristine DiffusionOPSD-main(与正式评测同一份)")
    ap.add_argument("--pre", required=True, help="ck300 的 lora 目录")
    ap.add_argument("--dirs", default="", help="第一段输出的 .pt")
    ap.add_argument("--combos", default="", help="name:k,...;work = θ300 + Δ0 + k·δ_name;name=M 时 work = θ300 + Δ0")
    ap.add_argument("--prompts", default="", help="一行一个 prompt 的 txt")
    ap.add_argument("--trainbatch", default="", help="fork 的 train_rewards/epoch_0301.jsonl(取其中去重后的 prompt)")
    ap.add_argument("--n-noise", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qsig", type=float, default=0.278)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("cross_eval", os.path.join(a.repo, "scripts", "cross_eval.py"))
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
    from diffusionopsd.diffusers_patch.solver import run_sampling
    import diffusionopsd.rewards
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file

    dev = torch.device("cuda")
    config = ce.load_config("config/public.py", "sd35_open3")
    res, gs, nl = int(config.resolution), float(config.sample.guidance_scale), float(config.sample.noise_level)
    assert gs <= 1.0, "本探针按 CFG 关闭(与正式评测一致)写的 v_pred"
    pipe, tes, tks = ce.build_pipeline(config.pretrained.model, a.pre, dev, torch.float16)
    tr = pipe.transformer
    tr.load_adapter(a.pre, adapter_name="work")
    tr.set_adapter("default")
    th0 = load_file(os.path.join(a.pre, "adapter_model.safetensors"), device="cuda")

    D = torch.load(a.dirs, map_location="cpu", weights_only=False) if a.dirs else None
    combos = []
    for c in [x for x in a.combos.split(",") if x]:
        f, k = c.split(":"); combos.append((f, float(k)))
    if D is not None:
        D0 = {k: v.to(dev) for k, v in D["D0"].items()}
        dl = {f: {k: v.to(dev) for k, v in D["delta"][f].items()} for f in {f for f, _ in combos if f != "M"}}

    def set_work(f, kk):  # work = θ300 + Δ0 + kk·δ_f;f="M" 只加 Δ0;f=None 时 work = θ300 原样(selftest)
        if f is None: st = dict(th0)
        elif f == "M": st = {k: th0[k] + D0[k] for k in th0}
        else: st = {k: th0[k] + D0[k] + kk * dl[f][k] for k in th0}
        set_peft_model_state_dict(tr, st, adapter_name="work")

    # ── prompt × noise 列表(按全局下标决定噪声种子,与分片/批大小无关)──
    if a.trainbatch:
        seen, P = set(), []
        for l in open(a.trainbatch):
            p = json.loads(l)["prompt"]
            if p not in seen: seen.add(p); P.append(p)
    else:
        P = ce.load_prompts(a.prompts, 0)
    items = [(pi, p, j) for pi, p in enumerate(P) for j in range(a.n_noise)]
    items = [it for i, it in enumerate(items) if i % a.nshard == a.shard]
    if a.limit: items = items[: a.limit]

    scorers = {n: diffusionopsd.rewards.multi_score(dev, {n: 1.0}) for n in ("pickscore", "clipscore", "hpsv2")}

    def score(img, prm):
        out = {}
        for n, fn in scorers.items():
            d, _ = fn(img, prm, [{} for _ in prm], only_strict=True)
            v = d[n]; out[n] = (v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(v)).reshape(-1).tolist()
        return out

    neg, negp = ce.compute_text_embeddings([""], tes, tks, 128, dev)
    rows, chk, t0 = [], {}, time.time()
    for b0 in range(0, len(items), a.bs):
        bt = items[b0: b0 + a.bs]; prm = [p for _, p, _ in bt]; B = len(bt)
        pe, ppe = ce.compute_text_embeddings(prm, tes, tks, 128, dev)
        gens = [torch.Generator(device=dev).manual_seed(a.seed + pi * 1000 + j) for pi, _, j in bt]
        # cache_enabled=False:autocast 会缓存 requires_grad 叶子权重的 fp16 副本(PEFT 把活动 adapter 设为 requires_grad),
        # 原地改 work 权重后缓存不失效 ⇒ 同一 context 里第二次 set_work 无效(实测 ± 终点逐位相同)。cast 是确定性的,数值不变。
        with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.no_grad():
            tr.set_adapter("default")
            _, Z, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=neg.repeat(B, 1, 1), negative_pooled_prompt_embeds=negp.repeat(B, 1),
                num_inference_steps=40, guidance_scale=gs, output_type="pt", height=res, width=res,
                noise_level=nl, deterministic=True, solver="flow", model_type="sd3", generator=gens, decode=False)
            sig = pipe.scheduler.sigmas.float()
            NS = len(sig) - 1
            ist = int(torch.argmin((sig[: NS - 1] - a.qsig).abs()).item())

            def vfn(z, sigma):  # = pipeline 的 v_pred_fn(非 CFG、sd3 分支)
                tt = torch.full([z.shape[0]], sigma * 1000, device=z.device, dtype=torch.long)
                o = tr(hidden_states=z, timestep=tt, encoder_hidden_states=pe, pooled_projections=ppe,
                       joint_attention_kwargs=None, return_dict=False)[0]
                return o.to(pe.dtype)

            def run(z, i0, i1, ad):  # 从第 i0 步的状态走到第 i1 步,全程用 adapter ad
                if i1 <= i0: return z
                tr.set_adapter(ad)
                z, _, _ = run_sampling(vfn, z, sig[i0:], "flow", True, nl, max_steps=i1 - i0)
                tr.set_adapter("default")
                return z

            def decode(z):  # = pipeline 末尾的 decode
                x = (z / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                img = pipe.vae.decode(x.to(dtype=pipe.vae.dtype), return_dict=False)[0]
                return pipe.image_processor.postprocess(img, output_type="pt")

            if a.selftest and b0 == 0:
                img_ref, _, _ = pipeline_with_logprob(
                    pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                    negative_prompt_embeds=neg.repeat(B, 1, 1), negative_pooled_prompt_embeds=negp.repeat(B, 1),
                    num_inference_steps=40, guidance_scale=gs, output_type="pt", height=res, width=res,
                    noise_level=nl, deterministic=True, solver="flow", model_type="sd3",
                    generator=[torch.Generator(device=dev).manual_seed(a.seed + pi * 1000 + j) for pi, _, j in bt])
                chk["decode_eq_pipeline_image"] = bool(torch.equal(decode(Z[-1]), img_ref))
                chk["tail_from_istar_equal"] = bool(torch.equal(run(Z[ist], ist, NS, "default"), Z[-1]))
                chk["full_resample_equal"] = bool(torch.equal(run(Z[0], 0, NS, "default"), Z[-1]))
                set_work(None, 0.0)
                chk["work_eq_pre_equal"] = bool(torch.equal(run(Z[0], 0, NS, "work"), Z[-1]))
                print(f"[p11] selftest i*={ist} σ*={float(sig[ist]):.4f}: {chk}", flush=True)
                if not all(chk.values()):
                    raise SystemExit("[p11] ❌ selftest 不通过")

            ends = {"Pre": Z[-1]}
            prevD = Z[-1]
            for f, kk in combos:
                set_work(f, kk)
                tag = f"{f}|{kk:g}"
                zB = run(Z[ist], ist, ist + 1, "work")
                ends[tag + "|B"] = run(zB, ist + 1, NS, "default")
                ends[tag + "|C"] = run(Z[ist], ist, NS, "work")
                zP = run(Z[0], 0, ist, "work")
                ends[tag + "|P"] = run(zP, ist, NS, "default")
                ends[tag + "|D"] = run(zP, ist, NS, "work")
                # 守卫:每个组合的 D 终点必须与上一个组合(第一个与 Pre)不同,否则权重没换上(§4.850 §七 ①)
                dprev = float((ends[tag + "|D"].float() - prevD.float()).norm())
                if b0 == 0:
                    print(f"[p11] {tag}: ‖z_D − 上一组合‖={dprev:.4f} ‖z_D − z_pre‖={float((ends[tag + '|D'].float() - Z[-1].float()).norm()):.4f}", flush=True)
                if dprev == 0.0:
                    raise SystemExit(f"[p11] ❌ {tag} 的终点与上一组合完全相同,权重没换上")
                prevD = ends[tag + "|D"]
            imgs = {k: decode(z) for k, z in ends.items()}
        with torch.no_grad():  # 打分在 autocast 之外,与 cross_eval.score_all 一致
            sc = {k: score(im, prm) for k, im in imgs.items()}
        del imgs
        for i, (pi, p, j) in enumerate(bt):
            rows.append(dict(pi=pi, prompt=p, noise=j,
                             cells={k: {n: v[n][i] for n in v} for k, v in sc.items()}))
        print(f"[p11] {b0 + B}/{len(items)} i*={ist} σ*={float(sig[ist]):.4f} [{time.time() - t0:.0f}s]", flush=True)
    meta = dict(cfg=vars(a), istar=ist, sigma_istar=float(sig[ist]), sigmas=sig.tolist(),
                rho50=(float(D["rho50"]) if D is not None else None), selftest=chk, resolution=res, guidance=gs)
    tmp = a.out + ".tmp"
    json.dump(dict(meta=meta, rows=rows), open(tmp, "w")); os.replace(tmp, a.out)
    print("[p11] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
