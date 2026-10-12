"""P11 第一段(§4.850):从 fork 的 optimizer 状态精确还原"本批梯度"造成的那部分参数位移。

ck300 带着 Adam 动量,一次真实更新 Δ_f = θ301 − θ300 里约 90% 是各 fork 共用的动量。
诊断要看 target 换了以后**本批梯度**让参数怎么动:
    δ_f = Δ_f − Δ_0,    Δ_0 = 同一 optimizer 状态下 g=0 的那一步(动量 + weight decay,各 fork 共用)
g 由 m301 = β1·m300 + (1−β1)·g 反解(fp32)。
自检(任一不过就不写输出):
  ① optimizer 的第 i 个参数与按训练脚本顺序重建的第 i 个 LoRA 张量形状一致
  ② v301 ≈ β2·v300 + (1−β2)·g²
  ③ fork 落盘的 lora 是 EMA:EMA301 = EMA300 + 0.1·(θ301 − EMA300),而 θ300 = EMA300(resume 把 EMA 权重装进活权重)
     ⇒ lora_{t+1} − lora_t 必须等于 (1−d)·Δ_f,d = min((1+t1)/(10+t1), 0.9)(ema.py 的 get_current_decay,t1 = 更新后的 step;
       t1=301 时 1−d=0.1,t1=51 时 0.1475)—— 这同时检验了 ① 的顺序映射
"""
import argparse, json, math, os
import torch
from safetensors.torch import load_file

TARGETS = ["attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
           "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"]


def param_order(model_id):
    """训练脚本 L1084 的顺序:get_peft_model + add_adapter("old") + set_adapter("default") 后 requires_grad 的参数。"""
    from accelerate import init_empty_weights
    from diffusers import SD3Transformer2DModel
    from peft import LoraConfig, get_peft_model
    cfg = SD3Transformer2DModel.load_config(model_id, subfolder="transformer")
    with init_empty_weights():
        t = SD3Transformer2DModel.from_config(cfg)
    lc = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=TARGETS)
    t = get_peft_model(t, lc); t.add_adapter("old", lc); t.set_adapter("default")
    return [n.replace(".default.", ".") for n, p in t.named_parameters() if p.requires_grad]


def adamw_delta(theta, m, v, t, pg):
    lr, (b1, b2), eps, wd = pg["lr"], pg["betas"], pg["eps"], pg["weight_decay"]
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    return -lr * wd * theta - (lr / bc1) * m / (v.sqrt() / math.sqrt(bc2) + eps)


def flat(d, keys):
    return torch.cat([d[k].reshape(-1).double() for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="ck300 目录(含 lora/ 与 optimizer.pt)")
    ap.add_argument("--ref", required=True, help="用来定半径的更早 checkpoint(ck250)目录")
    ap.add_argument("--forks", required=True, help="name=dir,...;dir 指到 fork 的 checkpoint-301")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-3.5-medium")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    names = param_order(a.model)
    th0 = load_file(os.path.join(a.pre, "lora", "adapter_model.safetensors"))
    keys = [n for n in names]
    assert len(keys) == len(th0) and set(keys) == set(th0), (len(keys), len(th0))
    o0 = torch.load(os.path.join(a.pre, "optimizer.pt"), map_location="cpu", weights_only=False)
    pg = {k: v for k, v in o0["param_groups"][0].items() if k != "params"}
    b1, b2 = pg["betas"]
    t0 = int(o0["state"][0]["step"])
    for i, k in enumerate(keys):  # ①
        assert tuple(o0["state"][i]["exp_avg"].shape) == tuple(th0[k].shape), (i, k)
    m0 = {k: o0["state"][i]["exp_avg"].double() for i, k in enumerate(keys)}
    v0 = {k: o0["state"][i]["exp_avg_sq"].double() for i, k in enumerate(keys)}
    T0 = {k: th0[k].double() for k in keys}
    D0 = {k: adamw_delta(T0[k], b1 * m0[k], b2 * v0[k], t0 + 1, pg) for k in keys}  # 动量 + wd 那一步

    thr = load_file(os.path.join(a.ref, "lora", "adapter_model.safetensors"))
    rho50 = float((flat(th0, keys) - flat(thr, keys)).norm())
    stats = dict(t0=t0, pg={k: (list(v) if isinstance(v, tuple) else v) for k, v in pg.items()},
                 rho50=rho50, norm_theta300=float(flat(th0, keys).norm()), norm_D0=float(flat(D0, keys).norm()),
                 forks={})
    out = dict(keys=keys, rho50=rho50, delta={}, D0={k: D0[k].float() for k in keys}, stats=stats)
    for spec in a.forks.split(","):
        nm, d = spec.split("=")
        o1 = torch.load(os.path.join(d, "optimizer.pt"), map_location="cpu", weights_only=False)
        t1 = int(o1["state"][0]["step"]); assert t1 == t0 + 1, (nm, t1, t0)
        m1 = {k: o1["state"][i]["exp_avg"].double() for i, k in enumerate(keys)}
        v1 = {k: o1["state"][i]["exp_avg_sq"].double() for i, k in enumerate(keys)}
        g = {k: (m1[k] - b1 * m0[k]) / (1 - b1) for k in keys}
        vchk = float((flat(v1, keys) - flat({k: b2 * v0[k] + (1 - b2) * g[k] ** 2 for k in keys}, keys)).norm()
                     / flat(v1, keys).norm())                                   # ②
        Df = {k: adamw_delta(T0[k], m1[k], v1[k], t1, pg) for k in keys}
        th1 = load_file(os.path.join(d, "lora", "adapter_model.safetensors"))
        ema_diff = flat(th1, keys) - flat(th0, keys)
        omd = 1 - min((1 + t1) / (10 + t1), 0.9)  # EMAModuleWrapper(decay=0.9) 的当步系数
        echk = float((ema_diff - omd * flat(Df, keys)).norm() / (omd * flat(Df, keys)).norm())   # ③
        dl = {k: (Df[k] - D0[k]).float() for k in keys}
        fd = flat(dl, keys)
        stats["forks"][nm] = dict(dir=d, v_check_rel=vchk, ema_check_rel=echk, norm_g=float(flat(g, keys).norm()),
                                  norm_Delta=float(flat(Df, keys).norm()), norm_delta=float(fd.norm()),
                                  cos_delta_Delta=float(fd @ flat(Df, keys) / (fd.norm() * flat(Df, keys).norm())))
        print(f"[p11-dirs] {nm}: v② rel={vchk:.2e}  ema③ rel={echk:.2e}  ‖g‖={stats['forks'][nm]['norm_g']:.4f} "
              f"‖Δ‖={stats['forks'][nm]['norm_Delta']:.3e} ‖δ‖={float(fd.norm()):.3e}", flush=True)
        if not (vchk < 1e-3 and echk < 1e-3):
            raise SystemExit(f"[p11-dirs] ❌ {nm} 自检不过(②={vchk:.2e} ③={echk:.2e}),不写输出")
        out["delta"][nm] = dl
    fl = {nm: flat(out["delta"][nm], keys) for nm in out["delta"]}
    stats["cos"] = {f"{p}|{q}": float(fl[p] @ fl[q] / (fl[p].norm() * fl[q].norm()))
                    for i, p in enumerate(fl) for q in list(fl)[i + 1:]}
    print(f"[p11-dirs] rho50={rho50:.4e}  ‖θ300‖={stats['norm_theta300']:.3e}  ‖Δ0‖={stats['norm_D0']:.3e}", flush=True)
    for k, v in stats["cos"].items():
        print(f"[p11-dirs] cos({k}) = {v:+.4f}", flush=True)
    torch.save(out, a.out)
    json.dump(stats, open(a.out + ".json", "w"), indent=2)
    print("[p11-dirs] wrote", a.out)


if __name__ == "__main__":
    main()
