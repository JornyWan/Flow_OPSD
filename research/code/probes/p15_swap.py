"""P15 = 去噪前后缀交换(§4.880,用户转来的 GPT 方案第一步):canonical(C)与 MSP(M)在噪声阈值 cut 处分段。
同一 prompt、同一初始噪声、同一 sampler(flow,deterministic),σ ≥ cut 的模型调用用"前段"adapter,σ < cut 用"后段"adapter。
组合:MC(M 前段 + C 后段)、CM(C 前段 + M 后段);CC / MM 直接用已有的 matched_pi 逐图结果做配对基线。

实现:pristine cross_eval 的 build_pipeline / generate_images / score_all 原样复用(评测口径逐位相同),
只在 pipeline.transformer 外包一层:按传入的 timestep(= σ·1000)切 adapter。C 装为 "default",M 用 load_adapter 装为 "M"。
初始噪声:matched 评测按 16 条一批、generator 种子 = 42 + 批号;本脚本只处理整批区间 [b0, b1),并令 seed = 42 + b0 ⇒ 逐图同噪声。
自检(任一不过即中止,不写输出):本分片第一批的 CC 与 MM 两格,逐图四个 reward 与 matched_pi 文件差 < 1e-3。
"""
import argparse, importlib.util, json, os, sys, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--C", required=True); ap.add_argument("--M", required=True)
    ap.add_argument("--ref-C", required=True, help="matched_pi 前缀(不含 _<set>_flow<n>.json),如 outputs/matched_pi_o3canon300_100_default")
    ap.add_argument("--ref-M", required=True)
    ap.add_argument("--prompts", required=True); ap.add_argument("--set", required=True)
    ap.add_argument("--b0", type=int, required=True); ap.add_argument("--b1", type=int, required=True)
    ap.add_argument("--cuts", default="0.9,0.7,0.45"); ap.add_argument("--nfes", default="4,40")
    ap.add_argument("--combos", default="MC,CM"); ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = importlib.util.spec_from_file_location("cross_eval", os.path.join(a.repo, "scripts", "cross_eval.py"))
    ce = importlib.util.module_from_spec(spec); spec.loader.exec_module(ce)
    dev = torch.device("cuda")
    config = ce.load_config("config/public.py", "sd35_open3")
    pipe, tes, tks = ce.build_pipeline(config.pretrained.model, a.C, dev, torch.float16)
    tr = pipe.transformer
    tr.load_adapter(a.M, adapter_name="M"); tr.set_adapter("default")
    names = {"C": "default", "M": "M"}
    st = {"pre": "default", "post": "default", "cut": 2.0, "seen": {}}
    fwd0 = tr.forward

    def fwd(*args, **kw):
        s = float(kw["timestep"].flatten()[0]) / 1000.0
        ad = st["pre"] if s >= st["cut"] else st["post"]
        tr.set_adapter(ad)
        st["seen"].setdefault(round(s, 4), set()).add(ad)
        return fwd0(*args, **kw)
    tr.forward = fwd

    P = ce.load_prompts(a.prompts, 0)
    sub = P[a.b0 * 16: a.b1 * 16]
    rewards = ["pickscore", "clipscore", "hpsv2", "imagereward"]

    def gen(prompts, nfe, pre, post, cut, seed):
        st.update(pre=names[pre], post=names[post], cut=cut, seen={})
        ga = argparse.Namespace(guidance_scale=-1.0, batch_size=16, seed=seed, max_sequence_length=128,
                                num_steps=nfe, sampler="flow")
        imgs = ce.generate_images(pipe, tes, tks, prompts, config, ga, dev, torch.float16)
        return imgs, {k: sorted(v) for k, v in st["seen"].items()}

    def score(imgs, prompts):
        col = {}
        ce.score_all(imgs, prompts, rewards, dev, 16, "raw", per_image_out=col)
        return col

    def ref(prefix, nfe, lo, hi):
        d = json.load(open(f"{prefix}_{a.set}_flow{nfe}.json"))["per_image"]
        return {r: [x[r] for x in d[lo:hi]] for r in rewards}

    # ── 自检:第一批的 CC / MM 必须逐图复现 matched_pi(打分一次装载)──
    first = sub[:16]; ims, keys = [], []
    for nfe in [int(x) for x in a.nfes.split(",")]:
        for comb in ("CC", "MM"):
            im, seen = gen(first, nfe, comb[0], comb[1], 2.0, 42 + a.b0)
            ims.append(im); keys.append((comb, nfe, seen))
    got = score(torch.cat(ims), first * len(ims))
    for i, (comb, nfe, seen) in enumerate(keys):
        want = ref(a.ref_C if comb == "CC" else a.ref_M, nfe, a.b0 * 16, a.b0 * 16 + len(first))
        err = max(abs(got[r][i * len(first) + j] - want[r][j]) for r in rewards for j in range(len(first)))
        print(f"[p15] selftest {comb} nfe={nfe}: max|Δ|={err:.2e} sigmas={list(seen)}", flush=True)
        if err >= 1e-3:
            raise SystemExit(f"[p15] ❌ 自检失败:{comb} nfe={nfe} 与 matched_pi 不一致(max|Δ|={err:.3e})")
    # ── 交换格:先全部生成,最后一次打分;切换守卫 = 两种 adapter 都出现且按 cut 分在两侧 ──
    out = dict(meta=dict(cfg=vars(a), rewards=rewards), cells={})
    t0 = time.time(); ims, keys = [], []
    for nfe in [int(x) for x in a.nfes.split(",")]:
        for cut in [float(x) for x in a.cuts.split(",")]:
            for comb in a.combos.split(","):
                im, seen = gen(sub, nfe, comb[0], comb[1], cut, 42 + a.b0)
                ads = {ad for v in seen.values() for ad in v}
                if len(ads) != 2 or any((s >= cut) != (names[comb[0]] in v) for s, v in seen.items()):
                    raise SystemExit(f"[p15] ❌ 切换守卫失败 {comb} cut={cut} nfe={nfe}: {seen}")
                ims.append(im); keys.append((f"{comb}|{cut:g}|{nfe}", seen))
                print(f"[p15] {comb} cut={cut:g} nfe={nfe} generated [{time.time() - t0:.0f}s]", flush=True)
    got = score(torch.cat(ims), sub * len(ims)); n = len(sub)
    for i, (k, seen) in enumerate(keys):
        out["cells"][k] = dict(seen={str(x): v for x, v in seen.items()},
                               scores={r: got[r][i * n:(i + 1) * n] for r in rewards})
    out["meta"]["index_range"] = [a.b0 * 16, a.b0 * 16 + len(sub)]
    tmp = a.out + ".tmp"; json.dump(out, open(tmp, "w")); os.replace(tmp, a.out)
    print("[p15] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
