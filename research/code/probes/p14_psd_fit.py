"""P14 = PS2(§4.870):学生能不能把策略空间搜索的收益学进去 —— 从 canonical ck300 做"一次更新"的 fork,评完整 40 步。

与 P11 的 fork 口径一致:同一 ck300 权重 + 同一 Adam 状态(ck300 的 optimizer.pt),一次 AdamW 更新(lr 3e-4、wd 1e-4、clip 1.0)。
loss 与 OPSD 同一几何:逐样本·逐态 ‖v_θ(s) − v_tgt(s)‖² / mean|v_θ(s) − v_tgt(s)|(分母 detach ⇒ 只取方向),态间等权。
target = 该样本在 PS1 里的赢家候选 θ_k* = θ0 + c·u 在**原学生 θ0 的轨迹态**上的速度预测(同一样本所有态用同一个赢家)。
赢家 = base 的样本残差为 0、不贡献梯度。训练样本 = PS1 的 unseen1k 前 64 条 × 噪声 A(按 open3 选赢家)。
变体:
  Sm  结构化候选集 S = {±e1, ±e2}@s=1 的赢家,多态 σ≈{0.858, 0.602, 0.278}
  Sq  同一批赢家,只监督 σ≈0.278(query-only,GPT 要的对照)
  Rm  随机候选集 {±e3}@s=1 的赢家,多态(对照:搜索是否只是在挑噪声)
评测格(完整 flow-40,同 PS1 的 sampler 与打分):base(θ0)、M(θ0+Δ0,只有动量)、CANON(θ0+Δ0+δ_c,P11 的 canonical 真实一步)、
Sm / Sq / Rm 的真实一步更新。prompt:u1k 噪声 A(样本内)、u1k 噪声 B(换噪声)、drawbench 64 条(换 prompt)。
"""
import argparse, glob, importlib.util, json, os, time
import torch
from torch.cuda.amp import autocast as torch_autocast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--pre", required=True); ap.add_argument("--optim", required=True)
    ap.add_argument("--dirs", required=True); ap.add_argument("--ps1", required=True, help="PS1 u1k 分片 json 的 glob")
    ap.add_argument("--u1k", required=True); ap.add_argument("--db", required=True)
    ap.add_argument("--n-prompts", type=int, default=64); ap.add_argument("--n-db", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--states", default="0.858,0.602,0.278"); ap.add_argument("--qstate", type=float, default=0.278)
    ap.add_argument("--bs", type=int, default=2); ap.add_argument("--ebs", type=int, default=8)
    ap.add_argument("--variants", default="Sm,Sq,Rm"); ap.add_argument("--out", required=True)
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

    # ── PS1 的赢家与候选系数 ──
    rows, meta = [], None
    for f in sorted(glob.glob(a.ps1)):
        d = json.load(open(f)); rows += d["rows"]; meta = meta or d["meta"]
    coef = meta["coef"]
    nrm = lambda d_: torch.sqrt(sum((d_[k].double() ** 2).sum() for k in keys))
    g = torch.Generator(device="cpu").manual_seed(12345)                      # 与 PS1 同一个随机方向
    rnd = {k: torch.randn(D["D0"][k].shape, generator=g) for k in keys}
    Uraw = {"e1": D["delta"][meta["cfg"]["e1"]], "e2": D["D0"], "e3": rnd}
    U = {n: {k: (u[k].double() / nrm(u)).float().to(dev) for k in keys} for n, u in Uraw.items()}
    o3 = lambda c: c["pickscore"] / 26.0 + c["clipscore"] + c["hpsv2"]
    byA = {r["pi"]: r["cells"] for r in rows if r["noise"] == 0 and r["pi"] < a.n_prompts}
    def winners(C):
        return {pi: max(["base"] + C, key=lambda k: o3(c[k])) for pi, c in byA.items()}
    candS = [c for c in meta["candidates"] if c[1:3] in ("e1", "e2") and c.endswith("@1")]
    candR = [c for c in meta["candidates"] if c[1:3] == "e3" and c.endswith("@1")]
    W = {"S": winners(candS), "R": winners(candR)}

    def weights_of(name):
        if name == "base": return dict(th0)
        n = name[1:3]
        return {k: th0[k] + coef[name] * U[n][k] for k in th0}

    def set_adapter_weights(adapter, st):
        set_peft_model_state_dict(tr, st, adapter_name=adapter)

    # ── 可训练参数(default adapter,模块顺序 = 训练脚本的顺序)与 ck300 的 Adam 状态 ──
    params = []
    for n_, p in tr.named_parameters():
        is_def = ".default." in n_ and "lora_" in n_
        p.requires_grad_(is_def)
        if is_def: params.append(p)
    assert len(params) == len(keys), (len(params), len(keys))
    tr.get_base_model().enable_gradient_checkpointing()   # A40 上 bs=4 的反传 OOM(1072366);只省显存,数值不变
    ost = torch.load(a.optim, map_location="cpu", weights_only=False)
    for i, p in enumerate(params):
        assert tuple(ost["state"][i]["exp_avg"].shape) == tuple(p.shape), i

    U1k = ce.load_prompts(a.u1k, 0)[: a.n_prompts]
    neg, negp = ce.compute_text_embeddings([""], tes, tks, 128, dev)
    sts = [float(x) for x in a.states.split(",")]

    def traj(prompts, seeds, adapter="default"):
        pe, ppe = ce.compute_text_embeddings(prompts, tes, tks, 128, dev)
        B = len(prompts)
        with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.no_grad():
            tr.set_adapter(adapter)
            img, Z, _ = pipeline_with_logprob(
                pipe, prompt_embeds=pe, pooled_prompt_embeds=ppe,
                negative_prompt_embeds=neg.repeat(B, 1, 1), negative_pooled_prompt_embeds=negp.repeat(B, 1),
                num_inference_steps=40, guidance_scale=gs, output_type="pt", height=res, width=res, noise_level=nl,
                deterministic=True, solver="flow", model_type="sd3",
                generator=[torch.Generator(device=dev).manual_seed(s) for s in seeds])
            tr.set_adapter("default")
        return img, Z, pe, ppe

    def vpred(z, s, pe, ppe, adapter, grad=False):
        tt = torch.full([z.shape[0]], float(s) * 1000, device=dev, dtype=torch.long)
        tr.set_adapter(adapter)
        with torch_autocast(enabled=True, dtype=torch.float16, cache_enabled=False), torch.set_grad_enabled(grad):
            v = tr(hidden_states=z, timestep=tt, encoder_hidden_states=pe, pooled_projections=ppe,
                   return_dict=False)[0].float()
        tr.set_adapter("default")
        return v

    # ── 预先算好每个训练样本在 θ0 轨迹上各态的状态与目标(target 在 no_grad 下,按赢家分组设 work 权重)──
    cache = []   # (prompt, seed, {σ: z}, pe, ppe)
    for b0 in range(0, len(U1k), a.bs):
        P = U1k[b0: b0 + a.bs]; seeds = [a.seed + (b0 + i) * 1000 + 0 for i in range(len(P))]
        _, Z, pe, ppe = traj(P, seeds)
        sig = pipe.scheduler.sigmas.float()
        zs = {st: Z[int(torch.argmin((sig[:39] - st).abs()))].detach() for st in sts}
        ss = {st: float(sig[int(torch.argmin((sig[:39] - st).abs()))]) for st in sts}
        cache.append(dict(pis=list(range(b0, b0 + len(P))), zs=zs, ss=ss, pe=pe.detach(), ppe=ppe.detach()))

    def targets(var):
        Wv = W["S"] if var in ("Sm", "Sq") else W["R"]
        out = []
        for c in cache:
            T = {}
            for st in sts:
                v = torch.zeros_like(c["zs"][st], dtype=torch.float32)
                for name in set(Wv[pi] for pi in c["pis"]):
                    idx = [j for j, pi in enumerate(c["pis"]) if Wv[pi] == name]
                    set_adapter_weights("work", weights_of(name))
                    v[idx] = vpred(c["zs"][st], c["ss"][st], c["pe"], c["ppe"], "work")[idx]
                T[st] = v
            out.append(T)
        return out

    def one_update(var):
        set_adapter_weights("default", dict(th0))
        opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
        opt.load_state_dict(ost)
        Ts = targets(var)
        use = [a.qstate] if var == "Sq" else sts
        opt.zero_grad(set_to_none=True)
        nb = len(cache); tot = 0.0
        for c, T in zip(cache, Ts):
            loss = 0.0
            for st in use:
                v = vpred(c["zs"][st], c["ss"][st], c["pe"], c["ppe"], "default", grad=True)
                r = v - T[st]
                wf = r.detach().abs().flatten(1).mean(1).clamp(min=1e-5).view(-1, 1, 1, 1)
                loss = loss + (r ** 2 / wf).flatten(1).mean(1).mean()
            loss = loss / len(use) / nb
            loss.backward(); tot += float(loss)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        new = {k: p.detach().clone() for k, p in zip(keys, params)}
        d = torch.sqrt(sum(((new[k] - th0[k]).double() ** 2).sum() for k in keys))
        set_adapter_weights("default", dict(th0))
        return new, dict(loss=tot, grad_norm=gn, step_norm=float(d),
                         n_nonbase=sum(1 for pi, k in (W["S"] if var != "Rm" else W["R"]).items() if k != "base"))

    upd, info = {}, {}
    for var in a.variants.split(","):
        upd[var], info[var] = one_update(var)
        print(f"[p14] {var}: {info[var]}", flush=True)
    cells = {"base": dict(th0), "M": {k: th0[k] + D["D0"][k].to(dev) for k in keys},
             "CANON": {k: th0[k] + D["D0"][k].to(dev) + D["delta"][meta["cfg"]["e1"]][k].to(dev) for k in keys}}
    cells.update(upd)

    scorers = {n: diffusionopsd.rewards.multi_score(dev, {n: 1.0}) for n in ("pickscore", "clipscore", "hpsv2", "imagereward")}
    def score(img, prm):
        o = {}
        for n, fn in scorers.items():
            d_, _ = fn(img, prm, [{} for _ in prm], only_strict=True)
            v = d_[n]; o[n] = (v.detach().float().cpu() if torch.is_tensor(v) else torch.tensor(v)).reshape(-1).tolist()
        return o
    DB = ce.load_prompts(a.db, 0)[: a.n_db]
    sets = {"u1kA": (U1k, 0), "u1kB": (U1k, 1), "db": (DB, 0)}
    res_rows, t0 = [], time.time()
    for sn, (P, j) in sets.items():
        for b0 in range(0, len(P), a.ebs):
            Pb = P[b0: b0 + a.ebs]; seeds = [a.seed + (b0 + i) * 1000 + j for i in range(len(Pb))]
            sc, prev = {}, None
            for cn, st in cells.items():
                set_adapter_weights("work", st)
                img, _, _, _ = traj(Pb, seeds, adapter="work")
                if prev is not None and torch.equal(img, prev):
                    raise SystemExit(f"[p14] ❌ 格 {cn} 与上一格出图逐位相同")
                prev = img
                with torch.no_grad(): sc[cn] = score(img.float(), Pb)
            for i, p in enumerate(Pb):
                res_rows.append(dict(set=sn, pi=b0 + i, prompt=p, cells={k: {m: v[m][i] for m in v} for k, v in sc.items()}))
            print(f"[p14] eval {sn} {b0 + len(Pb)}/{len(P)} [{time.time() - t0:.0f}s]", flush=True)
    tmp = a.out + ".tmp"
    json.dump(dict(info=info, winners={k: {str(p): w for p, w in v.items()} for k, v in W.items()}, rows=res_rows,
                   cfg=vars(a)), open(tmp, "w")); os.replace(tmp, a.out)
    print("[p14] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
