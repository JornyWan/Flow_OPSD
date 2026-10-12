"""P13 = 策略空间搜索(PS1,§4.869):在学生的小幅 LoRA 变化里找"完整生成确实更好"的版本 —— 零训练。

θ0 = canonical ck300。候选 θ_k = θ0 + s·‖Δ0‖·u,u ∈ {±e1, ±e2, ±e3},外加 θ0 本身(共 1 + 6·|scales| 个):
  e1 = unit(δ_c)  canonical 一个 batch 的本批梯度步(OPSD 自己的更新方向,P11 第一段还原)
  e2 = unit(Δ0)   动量步(训练轨迹正在走的方向)
  e3 = 随机方向(LoRA 参数空间各向同性高斯,固定种子)—— 对照:搜索是否只是在挑噪声
每个候选对同一 prompt、同一初始噪声完整生成 flow-40(与正式评测同 sampler),打 pick/clip/hps(open3)+ imagereward。
另在第一批上量输出变化 D = ‖v_k − v_0‖² / ‖v_0‖²(θ0 轨迹上 σ≈0.858/0.602/0.278 三态)。
实现复用 p11_abcd_eval:pristine cross_eval 的 build_pipeline、work adapter、`autocast(cache_enabled=False)`(§4.860 的教训)、multi_score。
"""
import argparse, importlib.util, json, os, time
import torch
from torch.cuda.amp import autocast as torch_autocast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--pre", required=True); ap.add_argument("--dirs", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--n-prompts", type=int, default=64); ap.add_argument("--noises", default="0,1")
    ap.add_argument("--scales", default="1,3"); ap.add_argument("--e1", default="canon_s20260830")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("cross_eval", os.path.join(a.repo, "scripts", "cross_eval.py"))
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    from diffusionopsd.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
    import diffusionopsd.rewards
    from peft.utils.save_and_load import set_peft_model_state_dict
    from safetensors.torch import load_file

    dev = torch.device("cuda")
    config = ce.load_config("config/public.py", "sd35_open3")
    res, gs, nl = int(config.resolution), float(config.sample.guidance_scale), float(config.sample.noise_level)
    pipe, tes, tks = ce.build_pipeline(config.pretrained.model, a.pre, dev, torch.float16)
    tr = pipe.transformer
    tr.load_adapter(a.pre, adapter_name="work"); tr.set_adapter("default")
    th0 = load_file(os.path.join(a.pre, "adapter_model.safetensors"), device="cuda")
    D = torch.load(a.dirs, map_location="cpu", weights_only=False)
    keys = D["keys"]
    nrm = lambda d: torch.sqrt(sum((d[k].double() ** 2).sum() for k in keys))
    step = float(nrm(D["D0"]))                                           # 一次真实 Adam 更新的范数
    g = torch.Generator(device="cpu").manual_seed(12345)
    rnd = {k: torch.randn(D["D0"][k].shape, generator=g) for k in keys}
    U = {"e1": D["delta"][a.e1], "e2": D["D0"], "e3": rnd}
    U = {n: {k: (u[k].double() / nrm(u)).float().to(dev) for k in keys} for n, u in U.items()}
    scales = [float(x) for x in a.scales.split(",")]

    def set_work(n, c):
        st = {k: (th0[k] if n is None else th0[k] + c * U[n][k]) for k in th0}
        set_peft_model_state_dict(tr, st, adapter_name="work")

    P = ce.load_prompts(a.prompts, 0)[: a.n_prompts]
    noises = [int(x) for x in a.noises.split(",")]
    items = [(pi, p, j) for pi, p in enumerate(P) for j in noises]
    items = [it for i, it in enumerate(items) if i % a.nshard == a.shard]
    scorers = {n: diffusionopsd.rewards.multi_score(dev, {n: 1.0}) for n in ("pickscore", "clipscore", "hpsv2", "imagereward")}

    def score(img, prm):
        out = {}
        for n, fn in scorers.items():
            d, _ = fn(img, prm, [{} for _ in prm], only_strict=True)
            v = d[n]; out[n] = (v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(v)).reshape(-1).tolist()
        return out

    neg, negp = ce.compute_text_embeddings([""], tes, tks, 128, dev)

    def out_change(Zb, pe, ppe, B):
        """D = ‖v_work − v_default‖² / ‖v_default‖²,θ0 轨迹上 σ≈0.858/0.602/0.278 三态的均值。"""
        sig = pipe.scheduler.sigmas.float(); dl = []
        with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.no_grad():
            for stt in (0.858, 0.602, 0.278):
                i = int(torch.argmin((sig[:39] - stt).abs()))
                tt = torch.full([B], float(sig[i]) * 1000, device=dev, dtype=torch.long)
                vs = []
                for ad in ("default", "work"):
                    tr.set_adapter(ad)
                    vs.append(tr(hidden_states=Zb[i], timestep=tt, encoder_hidden_states=pe,
                                 pooled_projections=ppe, return_dict=False)[0].float())
                tr.set_adapter("default")
                dl.append(float(((vs[1] - vs[0]) ** 2).flatten(1).sum(1).div((vs[0] ** 2).flatten(1).sum(1)).mean()))
        return dl

    # ── 按输出变化校准(GPT:候选大小按模型预测的实际变化,而非参数范数)──
    # 固定校准集 = 前 8 条 prompt、噪声 0(各分片相同);D ∝ c²(实测 e1:s=3 / s=1 = 9.1)⇒ 每个方向一个平方根因子,
    # 使各方向在 s=1 时的 D 都等于 e1(canonical 本批梯度步)在 c = ‖Δ0‖ 时的 D。
    cp = ce.load_prompts(a.prompts, 0)[:8]
    cpe, cppe = ce.compute_text_embeddings(cp, tes, tks, 128, dev)
    with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.no_grad():
        tr.set_adapter("default")
        _, Zc, _ = pipeline_with_logprob(
            pipe, prompt_embeds=cpe, pooled_prompt_embeds=cppe,
            negative_prompt_embeds=neg.repeat(len(cp), 1, 1), negative_pooled_prompt_embeds=negp.repeat(len(cp), 1),
            num_inference_steps=40, guidance_scale=gs, output_type="pt", height=res, width=res, noise_level=nl,
            deterministic=True, solver="flow", model_type="sd3",
            generator=[torch.Generator(device=dev).manual_seed(a.seed + pi * 1000) for pi in range(len(cp))], decode=False)
    # 先把每个方向放大到 D ≥ 1e-4 再量(低于此会被 fp16 速度的舍入噪声抬高,smoke 1072303 实测随机方向因此低估 4.6 倍),
    # 再按 D ∝ c² 算目标系数,复测一次并修正一次。
    def Dof(n, c):
        set_work(n, c); return sum(out_change(Zc, cpe, cppe, len(cp))) / 3
    Dunit = {"e1": Dof("e1", step)}
    coef1 = {}
    for n in ("e1", "e2", "e3"):
        c = step; d = Dof(n, c)
        while d < 1e-4:
            c *= 4.0; d = Dof(n, c)
        c = c * (Dunit["e1"] / d) ** 0.5            # 第一次估计
        d2 = Dof(n, c); c = c * (Dunit["e1"] / d2) ** 0.5   # 复测修正
        coef1[n] = c; Dunit[n + "_final"] = Dof(n, c)
    fac = {n: coef1[n] / step for n in coef1}
    print(f"[p13] 校准:目标 D = {Dunit['e1']:.3e};系数因子 {fac};校准后 D = "
          f"{ {k: round(v, 6) for k, v in Dunit.items()} }", flush=True)
    cands = [("base", None, 0.0)] + [(f"{sg}{n}@{s:g}", n, (1 if sg == "+" else -1) * s * step * fac[n])
                                       for s in scales for n in ("e1", "e2", "e3") for sg in ("+", "-")]
    rows, dchk, t0 = [], {}, time.time()
    for b0 in range(0, len(items), a.bs):
        bt = items[b0: b0 + a.bs]; prm = [p for _, p, _ in bt]; B = len(bt)
        pe, ppe = ce.compute_text_embeddings(prm, tes, tks, 128, dev)
        mk = lambda: [torch.Generator(device=dev).manual_seed(a.seed + pi * 1000 + j) for pi, _, j in bt]
        sc, prev = {}, None
        for name, n, c in cands:
            set_work(n, c)
            with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.no_grad():
                tr.set_adapter("work")
                img, Z, _ = pipeline_with_logprob(
                    pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                    negative_prompt_embeds=neg.repeat(B, 1, 1), negative_pooled_prompt_embeds=negp.repeat(B, 1),
                    num_inference_steps=40, guidance_scale=gs, output_type="pt", height=res, width=res,
                    noise_level=nl, deterministic=True, solver="flow", model_type="sd3", generator=mk())
                tr.set_adapter("default")
            # 守卫:每个候选的图必须与前一个不同(§4.860 的 autocast 缓存教训)
            if prev is not None and torch.equal(img, prev):
                raise SystemExit(f"[p13] ❌ 候选 {name} 与上一候选出图逐位相同,权重没换上")
            prev = img
            if b0 == 0 and name == "base":
                Zb = Z
            if b0 == 0:
                dchk[name] = out_change(Zb, pe, ppe, B)
            with torch.no_grad():
                sc[name] = score(img.float(), prm)
        for i, (pi, p, j) in enumerate(bt):
            rows.append(dict(pi=pi, prompt=p, noise=j, cells={k: {m: v[m][i] for m in v} for k, v in sc.items()}))
        print(f"[p13] {b0 + B}/{len(items)} [{time.time() - t0:.0f}s]", flush=True)
    meta = dict(cfg=vars(a), step_norm=step, candidates=[c[0] for c in cands], coef={c[0]: c[2] for c in cands},
                D_unit=Dunit, factor=fac, D_out=dchk)
    tmp = a.out + ".tmp"
    json.dump(dict(meta=meta, rows=rows), open(tmp, "w")); os.replace(tmp, a.out)
    print("[p13] wrote", a.out, "| D =", {k: [round(x, 5) for x in v] for k, v in dchk.items()}, flush=True)


if __name__ == "__main__":
    main()
