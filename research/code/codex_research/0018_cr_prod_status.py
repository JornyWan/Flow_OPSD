#!/usr/bin/env python3
"""0018 正式 CR 对照的状态与预注册检验(读本地 .wandb,不依赖网络)。

两条预注册判据见 codex_research/0018_CR_PROD_PREREG.md 与 0016_GEOM_PROBE.md:
  A. 标量化病理:H1 守卫降低 OCR 违反 且 H2 守卫**抬高** GenEval 违反
  B. 一阶有效性:cumsum(fo_pred_dC_m) 与实测 cr_violation_m 轨迹是否同号且同量级

只在**共同 step 集**上比较(公共集变化会造成选择效应,τ 上栽过一次)。
"""
import os, sys, math, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from read_wandb import read

W = "/midtier/minilab/scratch/chw4026/.cache/wandb/wandb"

def load(run_id):
    """优先读本地 .wandb;为空则回退 wandb API。

    online 模式下本地文件会滞后 flush —— 0019 探针实际在正常记录,
    但本地文件长时间保持 0 字节,只读本地会误判为"无数据"。
    """
    hits = glob.glob(os.path.join(W, f"*-{run_id}", "*.wandb"))
    if hits and os.path.getsize(hits[0]) >= 64:
        try:
            return read(hits[0])
        except Exception as e:
            print(f"  [warn] {run_id} 本地读取失败({type(e).__name__}),转 API")
    try:
        os.environ.setdefault("WANDB_SILENT", "true")
        import wandb
        r = wandb.Api(timeout=60).run(f"jorny-cornell-university/flow_grpo/{run_id}")
        out = {}
        for row in r.scan_history():
            st = row.get("_step")
            if st is None:
                continue
            for k, v in row.items():
                if k.startswith("_") or not isinstance(v, (int, float)):
                    continue
                out.setdefault(k, []).append((int(st), float(v)))
        return out or None
    except Exception as e:
        print(f"  [warn] {run_id} API 读取失败: {type(e).__name__}")
        return None

def series(s, k):
    return dict(s[k]) if k in s else {}

def paired(a, b, key):
    """b - a,同 step 配对 + lag-1 自相关校正。"""
    da, db = series(a, key), series(b, key)
    c = sorted(set(da) & set(db))
    d = [db[s] - da[s] for s in c]
    if len(d) < 3:
        return None
    n = len(d); m = sum(d)/n
    sd = math.sqrt(sum((x-m)**2 for x in d)/(n-1)) if n > 1 else 0.0
    t = m/(sd/math.sqrt(n)) if sd > 0 else float("nan")
    if n > 2 and sd > 0:
        num = sum((d[i]-m)*(d[i+1]-m) for i in range(n-1))
        r1 = num/sum((x-m)**2 for x in d)
    else:
        r1 = 0.0
    # 负自相关会让 n(1-r1)/(1+r1) > n,凭空造样本量 —— 封顶到 n
    n_eff = min(float(n), max(2.0, n*(1-r1)/(1+r1)))
    t_eff = m/(sd/math.sqrt(n_eff)) if sd > 0 else float("nan")
    base = sum(abs(da[s]) for s in c)/len(c)
    return dict(n=n, mean=m, sd=sd, t=t, r1=r1, n_eff=n_eff, t_eff=t_eff,
                rel=(m/base if base else float("nan")), steps=(c[0], c[-1]))

def main():
    ARMS = {"crdirect(w=0)": "8odvdj4x", "crguard(w=10)": "q78m5irv"}
    GEOM = GEOM_FOR_NULL = "5uu5f8de"   # job 1036492,含 0019+0020
    if len(sys.argv) > 1:
        GEOM = sys.argv[1]
    hist = {}
    for lab, rid in ARMS.items():
        s = load(rid)
        if s is None:
            print(f"{lab:16s} 无数据"); continue
        hist[lab] = s
        n = len(series(s, "cr_direct_loss"))
        print(f"{lab:16s} run={rid}  n={n}")
    if len(hist) < 2:
        print("\n两臂数据不全,跳过配对检验"); return

    A, B = hist["crdirect(w=0)"], hist["crguard(w=10)"]
    print("\n=== A. 标量化病理检验(w=10 相对 w=0 的差) ===")
    print(f"{'metric':24s} {'mean diff':>12s} {'rel':>8s} {'r1':>6s} {'n_eff':>6s} {'t_eff':>8s}  判定")
    HYP = {"cr_violation_ocr": ("H1 期望<0(守卫降低违反)", -1),
           "cr_violation_geneval": ("H2 期望>0(守卫抬高违反)", +1)}
    verdicts = {}
    for k in ["cr_violation_ocr", "cr_violation_geneval",
              "cr_active_ocr", "cr_active_geneval", "cr_direct_loss"]:
        r = paired(A, B, k)
        if r is None:
            print(f"{k:24s} 点数不足"); continue
        sig = abs(r["t_eff"]) > 2.5
        note = ""
        if k in HYP:
            desc, want = HYP[k]
            ok = sig and (r["mean"] * want > 0)
            verdicts[k] = ok
            note = f"  {desc} -> {'成立' if ok else ('方向相反' if sig else '不显著')}"
        print(f"{k:24s} {r['mean']:+12.4g} {r['rel']:+8.2f} {r['r1']:+6.2f} "
              f"{r['n_eff']:6.1f} {r['t_eff']:+8.2f}{note}")
    if len(verdicts) == 2:
        h1, h2 = verdicts["cr_violation_ocr"], verdicts["cr_violation_geneval"]
        # 不显著 != 被证否。n_eff 被自相关压得很低时,方向对但过不了门槛,
        # 那是样本不足,不能写成"守卫失效"(那会误导早上读结论的人)。
        d_ocr = paired(A, B, "cr_violation_ocr")
        d_ge = paired(A, B, "cr_violation_geneval")
        dir_ok = (d_ocr and d_ocr["mean"] < 0) and (d_ge and d_ge["mean"] > 0)
        if h1 and h2:
            msg = "**标量化病理确认**(H1&H2 均过门槛):逐约束乘子有动机"
        elif h1 and not h2:
            msg = ("H1 过门槛、H2 未过。方向仍与病理一致,等 n 累积" if dir_ok
                   else "守卫单纯有效,病理不成立:固定权重已够,不必上 QP")
        elif dir_ok:
            msg = ("两条都未过门槛,但**符号方向均与病理一致** —— "
                   "属样本不足(n_eff 被自相关压低),不是证否。继续累积。")
        else:
            msg = "未过门槛且方向不一致:按预注册记为不成立"
        print("\n  ⇒ " + msg)

    print("\n=== A2. 几何量的零模型对照 ===")
    # 高维空间里两个**独立**随机向量的 cos ~ N(0, 1/d)。不给出这个尺度,
    # "cos ~= 0" 既可能是"真的正交(可行方向存在)",也可能是"两个向量都只是噪声"。
    # 原判据把 ~=0 直接读成前者,从未对照过后者 —— 这是闸门的一个漏洞。
    gp = load(GEOM_FOR_NULL) if GEOM_FOR_NULL else None
    if gp:
        import math
        D = 18776064   # student LoRA 可训练参数总数(从 adapter_model.safetensors 数得)
        sd = 1.0/math.sqrt(D)
        print(f"  d={D:,}  零模型 SD = 1/sqrt(d) = {sd:.3e}")
        for k in ["geom_cos_g_ocr", "geom_cos_g_geneval", "geom_cos_ocr_geneval"]:
            if k not in gp: continue
            v = [x[1] for x in sorted(gp[k]) if x[1] != 0]
            if not v: continue
            m = sum(v)/len(v); z = m/sd
            tag = ("与噪声不可区分" if abs(z) < 2 else
                   "略超噪声" if abs(z) < 5 else "显著超出噪声")
            print(f"  {k:22s} n={len(v):2d} 均值={m:+.4g}  {z:+7.2f} SD  {tag}")
        print("  注:零模型只给**尺度**,g 与 a_m 并非独立抽样,不作严格检验用。")

    print("\n=== B. 一阶有效性检验(0019 探针) ===")
    g = load(GEOM)
    if g is None:
        print("  探针无数据"); return
    fo = sorted(k for k in g if k.startswith("fo_"))
    if not fo:
        print(f"  该 run 无 fo_* 指标(可能是 0019 之前的探针)。可传 run id 作参数指定新探针。")
        return
    NMIN = 20   # 0020 预注册:fo_* 结论至少 n>=20,n=4 只用于提出问题
    for k in ["fo_cos_u_negg", "fo_cos_u_negacc", "fo_cos_g_gacc",
              "fo_step_norm", "fo_gacc_norm"]:
        if k in g:
            v = [x[1] for x in sorted(g[k])]
            print(f"  {k:20s} n={len(v):3d} 均值={sum(v)/len(v):+.4g} 末={v[-1]:+.4g}")
    # 三层拆解:cos(u,-g) 低是批噪声/retention 还是 AdamW 预条件?
    if "fo_cos_u_negacc" in g:
        va = [x[1] for x in sorted(g["fo_cos_u_negacc"])]
        n = len(va); mu = sum(va)/n
        print(f"\n  [拆解] cos(u, -累积梯度) 均值={mu:+.3f}  n={n}")
        if n < NMIN:
            print(f"      n<{NMIN},按预注册**不下判词**(仅提出问题)")
        elif mu > 0.8:
            print("      ⇒ AdamW 基本不改方向:低 cos(u,-g) 来自批噪声/retention"
                  " ⇒ QP 改用累积梯度即可,值得实现")
        elif mu >= 0.3:
            print("      ⇒ 预条件有影响但非主导 ⇒ QP 需在累积梯度上做并配步长控制")
        else:
            print("      ⇒ **预条件主导**:作用在原始梯度上的 QP 与实际更新脱节,不实现")
        if "fo_cos_g_gacc" in g:
            vg = [x[1] for x in sorted(g["fo_cos_g_gacc"])]
            print(f"  [拆解] cos(单批 direct g, 累积全梯度) 均值={sum(vg)/len(vg):+.3f}"
                  f"  (低则说明探针的 g 只是全梯度的噪声估计)")
    for slot in ("ocr", "geneval"):
        pk, vk = f"fo_pred_dC_{slot}", f"cr_violation_{slot}"
        if pk not in g or vk not in g:
            continue
        pred = sorted(g[pk]); meas = dict(g[vk])
        cum, acc = [], 0.0
        for st, val in pred:
            acc += val; cum.append((st, acc))
        # 实测轨迹相对首点的累积变化
        c = [s for s, _ in cum if s in meas]
        if len(c) < 3:
            print(f"  {slot}: 点数不足"); continue
        base = meas[c[0]]
        dm = [meas[s] - base for s in c]
        dp = [dict(cum)[s] for s in c]
        same = sum(1 for x, y in zip(dp, dm) if x*y > 0)
        ratio = [abs(x)/abs(y) for x, y in zip(dp, dm) if y != 0]
        med = sorted(ratio)[len(ratio)//2] if ratio else float("nan")
        print(f"  {slot:8s} n={len(c)}  同号 {same}/{len(c)}  "
              f"|预测/实测| 中位数={med:.3g}  末:预测={dp[-1]:+.3g} 实测={dm[-1]:+.3g}")
        if len(c) < NMIN:
            print(f"      n<{NMIN},按预注册**不下判词**")
        elif same < 0.6*len(c):
            print(f"      ⇒ 反号为主:线性化在该步长下无效,QP 与 GN 同因失效")
        elif 1/3 <= med <= 3:
            print(f"      ⇒ 同号且同量级:一阶保证是实的,QP 值得实现")
        else:
            print(f"      ⇒ 同号但差 >=1 个数量级:QP 需配步长控制,单靠投影不够")

if __name__ == "__main__":
    main()
