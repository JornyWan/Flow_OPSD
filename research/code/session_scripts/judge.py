"""统一判定机器:给臂名,自动核三条轴 → 拉对照 → 套**已登记**的判据。

为什么要它(本会话实际栽过的错,每一条都对应下面一道防线):
  §4.531 配错基线(+18.34 → +17.01)          ⇒ 必须 ckpt 反查 + 核训练硬件
  §4.537 文件名里的数字**不是外层 epoch**      ⇒ CK_MULT 由臂名乘积规则算
  §4.577 CK_MULT 漏 inner8 一格,静默落到 1   ⇒ 乘积规则 + 未知 inner/gacc 报警
  §4.579 `lr8x` 前缀在三条臂上重名,索引静默覆盖 ⇒ 键用**完整臂目录名**,撞键**报错**不覆盖
  §4.592 训练口径 {1,1,1} vs 报告口径 pick/26  ⇒ 明确标注用报告口径(MDE 按它标定)
  §4.596 只报复合分,漏两条 held-out 轴        ⇒ 三轴一起报

阈值**硬编码自登记小节,不可覆盖**。
"""
import json, math, os, glob, re, sys, statistics as st

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs"
MDE = 0.0033
AXES = ("open3", "imagereward", "aesthetic")

def ck_mult(arm):
    m = 1
    for k in re.findall(r"(?:inner|gacc)(\d+)", arm): m *= int(k)
    if m == 1 and re.search(r"(?:inner|gacc)(?![0-9])", arm):
        print(f"  ⚠️ {arm}: 臂名含 inner/gacc 但解析不出倍数 —— 人工核对")
    return m

def index_evals(sampler="flow", steps=40, allow_evalhw=False,
                prompt_set="drawbench", adapter="default"):
    """键 = (完整臂目录名, ck)。撞键**报错**,不静默覆盖(§4.579)。

    官方口径由三项共同定义,缺一项就会撞键(实测:重复来自 prompt 集与 adapter,不是评测硬件):
      prompt_set=drawbench、adapter=default(eval_protocol 以 _default 结尾)、sampler=flow-40。
    """
    idx = {}
    for p in sorted(glob.glob(os.path.join(O, "matched_*.json"))):
        b = os.path.basename(p)[:-5]
        if not allow_evalhw and ("_a100" in b or "_l40" in b): continue
        try: d = json.load(open(p))
        except Exception: continue
        if d.get("sampler") != sampler or int(d.get("num_steps") or 0) != steps: continue
        if prompt_set and d.get("prompt_set") != prompt_set: continue
        if adapter and not str(d.get("eval_protocol", "")).endswith("_" + adapter): continue
        mm = re.search(r"ab_open3_([^/]+)/checkpoints/checkpoint-(\d+)", str(d.get("ckpt", "")))
        if not mm: continue
        key = (mm.group(1), int(mm.group(2)))
        if key in idx and idx[key] != b:
            # §4.576:matched 路径的 gres 恒为 a40 ⇒ 同 (arm,ck,口径) 的两份文件应当**一致**。
            # 逐位一致 ⇒ 取其一;不一致 ⇒ 报错(说明口径定义还漏了一维),并**不覆盖**。
            try:
                x, y = load(idx[key], "open3"), load(b, "open3")
                ksx = sorted(set(x) & set(y))
                dmax = max(abs(x[k] - y[k]) for k in ksx) if ksx else float("inf")
            except Exception:
                dmax = float("inf")
            if dmax < 1e-6:
                idx.setdefault("__dupok__", []).append((key, idx[key], b))
            else:
                idx.setdefault("__dup__", []).append((key, idx[key], b, dmax))
                continue          # 不覆盖:保留先取到的,并已记录冲突
        idx[key] = b
    return idx

def prov(armdir):
    p = os.path.join(O, "ab_open3_" + armdir, "arm_provenance.json")
    if not os.path.exists(p): return {}
    try: return json.load(open(p))
    except Exception: return {}

def load(f, axis):
    d = json.load(open(os.path.join(O, f + ".json")))
    pi = d.get("per_image") or []
    if axis == "open3":
        return {r["index"]: r["pickscore"]/26.0 + r["clipscore"] + r["hpsv2"] for r in pi}
    return {r["index"]: r[axis] for r in pi if axis in r}

def paired(a, b):
    ks = sorted(set(a) & set(b)); n = len(ks)
    if n < 2: return 0, float("nan"), float("nan"), float("nan"), float("nan")
    dd = [a[k]-b[k] for k in ks]; m = st.mean(dd); sd = st.stdev(dd); se = sd/math.sqrt(n)
    return n, m, se, (m/se if se > 0 else float("nan")), sum(1 for x in dd if x > 0)/n*100

def resolve(idx, armdir, outer_ep):
    """按**外层 epoch** 取,自动乘 CK_MULT。"""
    ck = outer_ep * ck_mult(armdir)
    f = idx.get((armdir, ck))
    if f is None: return None, f"缺 {armdir}@ck{ck}(外层 ep{outer_ep})"
    j = prov(armdir)
    d = json.load(open(os.path.join(O, f + ".json")))
    return dict(file=f, arm=armdir, ck=ck, outer=outer_ep, n=len(d.get("per_image") or []),
                gpu=str(j.get("gpu", "?")).replace("NVIDIA ", "")[:18], np=j.get("nproc"),
                seed=j.get("RPI_SEED"), sampler=f"{d.get('sampler')}-{d.get('num_steps')}"), None

def report(lab, A, B, thresholds=None):
    """A/B 是 resolve 的结果。核三条轴,报三条 reward 轴。"""
    ok = True
    for k, nm in (("gpu", "训练硬件"), ("np", "nproc"), ("seed", "seed"), ("outer", "外层ep"), ("sampler", "采样")):
        if A[k] != B[k]:
            print(f"      ⚠️ {nm} 不配: {A[k]} vs {B[k]}")
            ok = ok and (k in ("np",))       # np 不配只是 ~0.17×MDE 偏移(§4.560),不作废
    out = {}
    for ax in AXES:
        try: n, m, se, t, w = paired(load(A["file"], ax), load(B["file"], ax))
        except Exception as e: print(f"      {ax}: 读取失败 {e}"); continue
        out[ax] = (n, m, se, t, w)
        sc = m/MDE if ax == "open3" else m
        unit = "×MDE" if ax == "open3" else ""
        flag = "  ⚠️不可作 held-out 验证(§4.613: r=-0.395, 符号一致 4/12)" if ax == "aesthetic" else (
            "  ✅held-out 主轴(§4.613: r=+0.894, 12/12)" if ax == "imagereward" else "")
        print(f"      {ax:<12} Δ={m:+.5f} ±{se:.5f}  t={t:+7.2f}  胜率={w:5.1f}%  "
              f"{sc:+8.2f}{unit}  n={n}{flag}")
    if thresholds and "open3" in out:
        m = out["open3"][1]
        for cell, lo, hi in thresholds:
            if (lo is None or m > lo) and (hi is None or m <= hi):
                print(f"      ⇒ **{cell}**")
                break
    return out

# ── 已登记的判据(阈值硬编码,不可覆盖)────────────────────────────────────
# 三格式:(格名, 下界, 上界);按顺序匹配第一个命中的
T3 = lambda a, b, c: [(a, 0.0033, None), (b, -0.0033, 0.0033), (c, None, -0.0033)]
REG = {
 "qcoreQ7": dict(outer=100, axes=[
    ("G 轴(§4.582):vs basenp3", "baseline_np3_s20260830_u100", T3("G1 正向", "G2 打平", "G3 为负")),
    ("参照:vs qstrain7(clean×同σ)", "qstrain7_s20260830_u100", None),
    ("参照:vs qcore(Q×σ=0.278)", "qcore_s20260830_u100", None)]),
 "multik4": dict(outer=100, axes=[
    ("M 轴(§4.588):vs baseAA", "baseAA100_s20260830_u100", T3("M1 正向", "M2 打平", "M3 为负")),
    ("N 轴(等成本):vs inner4", "inner4_s20260830_u100", T3("N1 多状态更优", "N2 等价", "N3 更差"))]),
 "hybrid": dict(outer=100, axes=[
    ("H 轴(§4.603):vs baseAA", "baseAA100_s20260830_u100", T3("H1 正向", "H2 打平", "H3 为负")),
    ("X1(核心主张):vs hybgrad", "hybgrad_s20260830_u100", T3("X1 增益来自选择", "X1 无", "X1 反向")),
    ("X2:vs multik4", "multik4_s20260830_u100", None),
    ("X3:vs qstrain5(单状态高噪梯度)", "qstrain5_s20260830_u100", None)]),
 "multikF4": dict(outer=100, axes=[
    ("vs basenp3(同 H200)", "baseline_np3_s20260830_u100", T3("正向", "打平", "为负"))]),
 "vadv": dict(outer=100, axes=[
    ("vs basenp3(同 H200)", "baseline_np3_s20260830_u100", T3("正向", "打平", "为负")),
    ("vs a3(cert 二值版)", "a3_s20260830_u100", None)]),
 "sello": dict(outer=100, axes=[
    ("反向门:vs basenp3(应当**输**)", "baseline_np3_s20260830_u100", T3("⚠️赢了=增益与选择无关", "打平", "输(符合预期)"))]),
 "multik2":  dict(outer=100, axes=[("vs baseAA","baseAA100_s20260830_u100",T3("正向","打平","为负")),
                                   ("vs multik4","multik4_s20260830_u100",None)]),
 "multik4s": dict(outer=100, axes=[("vs baseAA","baseAA100_s20260830_u100",T3("正向","打平","为负")),
                                   ("vs multik4","multik4_s20260830_u100",None)]),
 "multiklr": dict(outer=100, axes=[("vs lr8x(叠加基准)","lr8x_s20260830_u100",T3("叠加为正","无叠加","叠加为负")),
                                   ("vs baseAA","baseAA100_s20260830_u100",None)]),
 "hybgrad":  dict(outer=100, axes=[("vs baseAA","baseAA100_s20260830_u100",T3("正向","打平","为负"))]),
}

def judge(name, outer=None):
    if name not in REG: print(f"  未登记的臂: {name}"); return
    cfg = REG[name]; oe = outer or cfg["outer"]
    idx = index_evals()
    for k, a, b, dm in idx.get("__dup__", []):
        print(f"  ❌ 撞键且内容不同 {k}: {a} vs {b}  max|delta|={dm:.2e} —— 口径定义漏了一维,人工核")
    if idx.get("__dupok__"):
        print(f"  (撞键但逐位相同 {len(idx['__dupok__'])} 处,已取其一)")
    armdir = f"{name}_s20260830_u100"
    A, err = resolve(idx, armdir, oe)
    if err: print(f"  ⏳ {name}: {err}"); return
    print(f"\n  ═══ {name} @ 外层ep{oe} (ck{A['ck']}) ═══")
    print(f"      文件={A['file']}  n={A['n']}  {A['gpu']} np={A['np']} seed={A['seed']} {A['sampler']}")
    print(f"      口径:报告复合分 pick/26+clip+hps(MDE={MDE});⚠️ 训练口径是 {{1,1,1}},方向不同(§4.604)")
    for lab, ref, th in cfg["axes"]:
        B, err2 = resolve(idx, ref, oe)
        print(f"\n    ── {lab}")
        if err2: print(f"      ⏳ {err2}"); continue
        print(f"      对照={B['file']}  {B['gpu']} np={B['np']}")
        report(lab, A, B, th)

if __name__ == "__main__":
    args = sys.argv[1:] or list(REG)
    for a in args: judge(a)
