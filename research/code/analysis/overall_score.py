#!/usr/bin/env python
"""把 8 个轴聚合成"整体分数"。

各轴量纲差异极大(geneval 0-1, aesthetic ~6.4, clipscore ~0.29),直接平均没有意义。
这里做两种聚合,并且都带噪声带 —— 没有噪声带的整体分数会诱导过度解读:

  Ratio  = 相对官方 Student LoRA 的比值的均值。可读性好("整体比官方学生高 2.1%"),
           但会被相对波动最大的轴(geneval)主导。
  Z      = 各轴先在模型集合内做 z-score 再取均值。压掉了量纲与离散度差异,
           回答的是"综合排名靠前多少个标准差"。

噪声带不是拍的:用复现研究里"同一 checkpoint 在原服务器与本集群各评一次"的
配对差值,直接估每个轴的重测标准差 sigma_axis,再传播到聚合量:
  SE(Ratio) = sqrt(sum (sigma_a / ref_a)^2) / n_axes
  SE(Z)     = sqrt(sum (sigma_a / sd_a)^2)  / n_axes
"""
import json, os, math, sys, glob, re, statistics as st

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
R, N = f"{O}/CA-OPD/results/eval", f"{O}/outputs/eval"
TRAINED = ["geneval", "ocr", "pickscore", "clipscore", "aesthetic", "hpsv2"]
HELDOUT = ["imagereward", "unifiedreward"]
DS = ("geneval", "ocr", "drawbench")


def _axis_of(d):
    """把一个 eval 目录读成 {axis: value}。IR/UNI 目录只贡献那一个 held-out 轴。"""
    f = os.path.join(d, "average_scores.json")
    if not os.path.exists(f):
        return {}
    try:
        raw = json.load(open(f))
    except Exception:
        return {}
    out = {}
    for k, v in raw.items():
        kl = k.lower()
        if kl in TRAINED:
            out[kl] = v
        elif "unified" in kl:
            out["unifiedreward"] = v
        elif "image" in kl and "reward" in kl:
            out["imagereward"] = v
    return out


def collect(base):
    """base 下所有目录 -> {模型名: {axis: value}}。IR/UNI 后缀并回主模型。"""
    models = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        nm = os.path.basename(d)
        for ds in DS:
            if nm.endswith("_" + ds):
                nm = nm[: -len(ds) - 1]
                break
        else:
            continue
        vals = _axis_of(d)
        if not vals:
            continue
        # mopd931IR / mopd931UNI -> mopd931
        if nm.endswith("UNI"):
            nm = nm[:-3]
        elif nm.endswith("IR"):
            nm = nm[:-2]
        # 同一个轴可能被多个目录各测一次(UnifiedReward 的评测顺带也算 imagereward,
        # 于是 *IR_drawbench 和 *UNI_drawbench 都有 imagereward)。原来用 dict.update 会让
        # 字典序靠后的那个**静默覆盖**前一个 —— 取哪个纯属偶然,而两次测量能差 0.027(>3σ)。
        # 这里改成把所有重复测量收集起来,最后取平均:既消除任意性,又把噪声降到 1/sqrt(k)。
        for k, v in vals.items():
            models.setdefault(nm, {}).setdefault(k, []).append(v)
    return {m: {k: sum(v) / len(v) for k, v in ax.items()} for m, ax in models.items()}


def collect_raw(base):
    """同 collect,但保留每个轴的全部重复测量(用于估噪声)。"""
    models = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        nm = os.path.basename(d)
        for ds in DS:
            if nm.endswith("_" + ds):
                nm = nm[: -len(ds) - 1]
                break
        else:
            continue
        vals = _axis_of(d)
        if not vals:
            continue
        if nm.endswith("UNI"):
            nm = nm[:-3]
        elif nm.endswith("IR"):
            nm = nm[:-2]
        for k, v in vals.items():
            models.setdefault(nm, {}).setdefault(k, []).append(v)
    return models


def retest_sigma(orig, ours):
    """同一模型在两台机器上各评一次 -> 每个轴的重测标准差。"""
    sig = {}
    for a in TRAINED + HELDOUT:
        diffs = [ours[m][a] - orig[m][a]
                 for m in orig if m in ours and a in orig[m] and a in ours[m]]
        if len(diffs) >= 4:
            # 两次独立评测之差的 sd = sqrt(2)*单次 sd
            sig[a] = st.pstdev(diffs) / math.sqrt(2)
    return sig


def within_sigma(raw):
    """用**同一集群内**的重复测量估噪声。

    held-out 两轴(ImageReward/UnifiedReward)原服务器从未测过,没有跨机配对可用。
    但 UnifiedReward 的评测顺带也算 imagereward,于是每个模型有两次独立测量;
    把这些配对差值汇总即可估出单次测量的 sd(除以 sqrt(2))。
    """
    sig = {}
    for a in TRAINED + HELDOUT:
        d = [v[i + 1] - v[i] for ax in raw.values()
             for v in [ax.get(a, [])] for i in range(len(v) - 1)]
        if len(d) >= 4:
            sig[a] = st.pstdev(d) / math.sqrt(2)
    return sig


def repeat_sigma(raw):
    """用"同一 checkpoint 重复评测 k 次"直接估单次测量的 sd。

    nbIR / nbUNI 作业提交的 StudentIRrep1..4 / StudentUNIrep1..4 就是为此准备的:
    它们是同一个官方 Student LoRA 的独立重评,组内离散度即评测噪声本身。
    """
    groups = {}
    for m, ax in raw.items():
        g = re.fullmatch(r"(.*?)(?:IR|UNI)rep\d+", m)
        if not g:
            continue
        for a, vs in ax.items():
            groups.setdefault(a, []).extend(vs)
    return {a: st.pstdev(v) for a, v in groups.items() if len(v) >= 3}


ours = collect(N)
orig = collect(R)
sigma = retest_sigma(orig, ours)
# 跨机配对拿不到的轴,退回用集群内重复测量估计
_raw = collect_raw(N)
for _src in (within_sigma(_raw), repeat_sigma(_raw)):
    for _a, _s in _src.items():
        sigma.setdefault(_a, _s)

REF = "Student"          # 官方发布的 Student LoRA = 论文正文数字
if REF not in ours:
    sys.exit("找不到参照模型 Student")

EXCLUDE = ("Teacher", "base_", "mtnft", "mtgrpo")   # 教师/底模/失败的 RL 基线不是同一比较层级
import re as _re
_REP = _re.compile(r"rep\d+$")      # StudentIRrep1 之类是"同一 checkpoint 的重复评测",用于估噪声,不是独立模型


def agg(axes, label, peers_only=False):
    have = [m for m in ours if all(a in ours[m] for a in axes) and not _REP.search(m)]
    if peers_only:
        have = [m for m in have if not any(e in m for e in EXCLUDE)]
    if REF not in have:
        print(f"\n[{label}] 参照模型缺轴,跳过"); return
    ref = {a: ours[REF][a] for a in axes}
    # z-score 的均值/标准差只在当前比较集合内统计:混入崩掉的基线会撑大 sd、把差异压平
    mu = {a: st.mean(ours[m][a] for m in have) for a in axes}
    sd = {a: (st.pstdev([ours[m][a] for m in have]) or 1e-9) for a in axes}

    n_cov = sum(1 for a in axes if a in sigma)
    if n_cov == len(axes):
        se_ratio = math.sqrt(sum((sigma[a] / ref[a]) ** 2 for a in axes)) / len(axes)
        se_z = math.sqrt(sum((sigma[a] / sd[a]) ** 2 for a in axes)) / len(axes)
        band = (f"重测 1σ(Ratio)={se_ratio*100:.2f}%  判定阈值 2σ={se_ratio*200:.2f}%"
                f"   |   1σ(Z)={se_z:.3f}  2σ={se_z*2:.3f}")
    else:
        se_ratio = se_z = None
        band = (f"⚠ 无噪声估计({n_cov}/{len(axes)} 轴有配对重测数据) —— "
                f"这两列原服务器从未测过,所以下面的排序**不带误差棒,不能当证据**")

    print(f"\n=== {label} ({len(axes)} 轴, {len(have)} 个模型) ===")
    print(band)
    rows = sorted(((st.mean(ours[m][a] / ref[a] for a in axes),
                    st.mean((ours[m][a] - mu[a]) / sd[a] for a in axes), m) for m in have),
                  reverse=True)
    print(f"{'模型':<16}{'Ratio':>9}{'Δ%':>8}{'':3}{'Z':>7}   判定")
    for r, z, m in rows:
        d = (r - 1.0) * 100
        if m == REF:
            tag = "←参照(官方 Student)"
        elif se_ratio is None:
            tag = "无误差棒"
        elif abs(d) < se_ratio * 200:
            tag = "与参照打平(噪声内)"
        else:
            tag = "显著高于参照" if d > 0 else "显著低于参照"
        print(f"{m:<16}{r:>9.4f}{d:>+8.2f}{'':3}{z:>+7.3f}   {tag}")
    return se_ratio


agg(TRAINED, "训练目标内的 6 轴 —— 全部模型", peers_only=False)
agg(TRAINED, "训练目标内的 6 轴 —— 仅蒸馏学生(同层级比较)", peers_only=True)
agg(HELDOUT, "held-out 2 轴 (ImageReward + UnifiedReward)")

print("\n--- 各轴重测 1σ (来自复现配对) ---")
for a in TRAINED + HELDOUT:
    print(f"  {a:<15}{sigma.get(a, float('nan')):.5f}" if a in sigma else f"  {a:<15}(样本不足)")
