#!/usr/bin/env python
"""A/B 对比:baseline(官方 Open3 目标) vs A1(向量可行性 veto)。

用法: ab_compare.py [--step 100]

先核口径(run_config.json 必须除声明字段外全同),再比逐轴。
口径不一致时**拒绝出结论** —— 那样的对比没有意义。
"""
import argparse, json, os, subprocess, sys

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
SEED = "20260830"
# 臂目录/标签的对应。--arm 选比哪一条(baseline 恒为对照)。
ARM_DIR = {"baseline": "baseline", "a1": "a1", "n1": "n1", "n2": "n2",
           "qopsd": "qopsd", "a1v2": "a1v2", "b4x": "b4x", "a3": "a3", "qbt": "qbt",
           # np3 baseline 当作"臂"来比:量 **评测口径上**的 nproc artifact。
           # 训练 reward 上已测得 +0.0017(t=+3.12,§4.111),但评测侧从未测过 ——
           # 而 Q-OPSD 用 np3 对照、历史四臂用 np4,这个数决定两套结果能否互相参照。
           "basenp3": "baseline_np3"}
# 对照 baseline 可选:np4(原四臂用的)| np3(Q-OPSD/b4x/a1v2 用的同 nproc 对照)
BASE_TAG = {"np4": ("ab_base", "ab_open3_baseline_s{seed}_u100"),
            "np3": ("ab_basenp3", "ab_open3_baseline_np3_s{seed}_u100"),
            # qbt 作为对照:用于 Q-OPSD 的消融判读(0067 §⑩ / §4.117)——
            # Δ(qopsd vs qbt) 直接回答"收益来自 d_Q 方向还是仅仅来自在线回溯"。
            # 两臂都走 env 干预,config 应逐字段全同。
            "qbt": ("ab_qbt", "ab_open3_qbt_s{seed}_u100")}
# 各臂**有意变动**的 run_config 字段。干预若走 env(RPI_A1 / RPI_N2)则 config 无差异,
# 允许集为空 —— 任何差异都应判不可比。
ARM_ALLOW = {
    "a1": [],          # A1 走 RPI_A1 env,config 逐字段应全同
    "n2": [],          # N2 走 RPI_N2 env,config 逐字段应全同
    # N1 改权重必然带出 opa_reward_kind: open3 -> mixed。已核实两分支**是同一段代码**,
    # 只是 weights 的来源不同(open3 写死 1:1:1,mixed 取 config.reward_fn),
    # scorers 加载路径也相同 ⇒ mixed@1:1:1 与 open3 数值等价。
    # 唯一残留差异是 dict 迭代顺序造成的浮点求和次序(~1e-7 相对),远低于任何效应量。
    "n1": ["reward_fn.pickscore", "reward_fn.hpsv2", "opsd.opa_reward_kind"],
    "a3": ["opsd.opa_cert"],   # a3 的干预是 --config.opsd.opa_cert=1,该字段必然不同
    "qopsd": [],       # Q-OPSD 走 RPI_QOPSD env,config 逐字段应全同
    "qbt": [],         # 回溯消融同样走 env,config 应全同
    # ⚠️ 2026-09-01:原以为 nproc 不进 run_config —— **错了**,`world_size` 就在里面。
    # 口径闸据此判"不可比"并拒绝出结论(正确行为)。本对比的**全部目的**就是量 world_size
    # 的影响,故把它声明为本次的干预变量。其余臂(qopsd/qbt/a1v2/b4x vs np3)两侧同为
    # world_size=3,不受影响,**不放宽它们的声明**。
    "basenp3": ["world_size"],
    "a1v2": [],        # A1-v2 走 RPI_A1/RPI_A1_MODE env,config 应全同
    # b4x 改的是每 epoch 的 prompt 数 ⇒ 批参数必然不同,这些是**有意变动**的字段
    "b4x": ["sample.num_batches_per_epoch", "train.gradient_accumulation_steps"],
}
def arms_for(arm, base="np4", arm_updates=100):
    _bt, _bd = BASE_TAG[base]
    return [(_bt, f"{O}/outputs/" + _bd.format(seed=SEED)),
            (f"ab_{arm}", f"{O}/outputs/ab_open3_{ARM_DIR[arm]}_s{SEED}_u{arm_updates}")]
COLS = [("pickscore", "Pick"), ("clipscore", "CLIP"), ("hpsv2", "HPSv2.1"),
        ("aesthetic", "Aes"), ("imagereward", "ImgR")]
# 重测 1σ:2026-08-30 实测(n=5 同模型重复评测,0034 §⑦)。imagereward 首次测得。
SIGMA = {"pickscore": 0.00075, "clipscore": 0.00108, "hpsv2": 0.00043,
         "aesthetic": 0.01118, "imagereward": 0.00875}
# ⚠️ pickscore 的 σ 是在 /26 归一口径下测的;这里的 raw 口径要 ×26
SIGMA_RAW = dict(SIGMA, pickscore=SIGMA["pickscore"] * 26)

def load(tag, step):
    p = f"{O}/outputs/opsd_eval_{tag}_ck{step}_raw.json"
    os.listdir(f"{O}/outputs")                       # 刷 NFS
    if not os.path.exists(p):
        return None, p
    return json.load(open(p)), p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--step", default="100")
    ap.add_argument("--arm", default="a1", choices=sorted(k for k in ARM_DIR if k != "baseline"),
                    help="与 baseline 对比的臂(默认 a1)")
    ap.add_argument("--base", default="np4", choices=sorted(BASE_TAG),
                    help="对照 baseline:np4(原四臂)| np3(Q-OPSD/b4x/a1v2 的同 nproc 对照)")
    ap.add_argument("--arm-updates", default="100",
                    help="臂目录名里的 updates 数(b4x 算力配平跑 25)")
    ap.add_argument("--base-step", default=None,
                    help="对照侧的 checkpoint step(默认同 --step)。"
                         "b4x 是**算力配平**:b4x@ck25 的总样本 = baseline@ck100,"
                         "故须 --step 25 --base-step 100(0064 §④c 明令禁止用同 step 的 4×算力口径)")
    a = ap.parse_args()
    ARMS = arms_for(a.arm, base=a.base, arm_updates=a.arm_updates)
    BTAG = ARMS[0][0]   # 对照臂的 tag(np4→ab_base,np3→ab_basenp3,qbt→ab_qbt)
    # ⚠️ 2026-09-02:加 --base 时忘了这一步,下游仍硬编码 "ab_base",
    #    导致所有 --base np3/qbt 的对比都报"两臂结果未齐"(数据其实都在)。
    XTAG, XLAB = f"ab_{a.arm}", a.arm.upper()

    # ── 1) 口径核对 ────────────────────────────────────────────
    print("=" * 78)
    print("① 口径核对(除声明字段外必须全同)")
    dirs = [d for _, d in ARMS if os.path.isdir(d)]
    if len(dirs) == 2:
        _allow = ARM_ALLOW.get(a.arm, [])
        _cmd = [sys.executable, f"{O}/opsd_manifest_diff.py", *dirs]
        if _allow:
            _cmd += ["--allow", *_allow]
        r = subprocess.run(_cmd, capture_output=True, text=True)
        print(r.stdout.rstrip())
        if r.returncode != 0:
            print("\n❌ 口径不一致 —— **拒绝出结论**。先查清差异来源。")
            return 1
    else:
        print(f"  ⚠️ 只有 {len(dirs)}/2 个 run 目录存在,跳过口径核对")

    # ── 2) 逐轴对比 ────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"② 逐轴对比 @ checkpoint-{a.step}")
    D = {}
    _bstep = a.base_step or a.step
    for _i, (tag, _) in enumerate(ARMS):
        d, p = load(tag, _bstep if _i == 0 else a.step)
        if d is None:
            print(f"  ⏳ {tag}: {os.path.basename(p)} 尚未产出"); continue
        D[tag] = d
    base_json, _ = load("base", "base")
    if base_json: D["base"] = base_json

    if len(D) < 2 or BTAG not in D or XTAG not in D:
        print("\n  两臂结果未齐,无法对比。"); return 2

    b, x = D[BTAG]["scores"], D[XTAG]["scores"]
    bs, xs = D[BTAG].get("score_stats", {}), D[XTAG].get("score_stats", {})
    print(f"\n  {'轴':<9s}{'baseline':>11s}{XLAB:>11s}{'差':>10s}{'/2σ':>8s}{'SE_b':>9s}{'SE_a':>9s}  判定")
    wins = losses = 0
    for k, lab in COLS:
        if k not in b or k not in x:
            print(f"  {lab:<9s}{'—':>11s}"); continue
        d = x[k] - b[k]
        two_sig = 2 * SIGMA_RAW[k]
        ratio = d / two_sig
        seb = bs.get(k, {}).get("se", float("nan")); sea = xs.get(k, {}).get("se", float("nan"))
        # 判定同时看效应量(>2σ)与两臂 SE 合成的 t
        se_pool = (seb ** 2 + sea ** 2) ** 0.5 if seb == seb and sea == sea else float("nan")
        t = d / se_pool if se_pool == se_pool and se_pool > 0 else float("nan")
        sig = (abs(ratio) > 1.0) and (abs(t) > 2.5)
        mark = (f"{XLAB} 更好" if d > 0 else f"{XLAB} 更差") if sig else "打平"
        if sig and d > 0: wins += 1
        if sig and d < 0: losses += 1
        print(f"  {lab:<9s}{b[k]:>11.4f}{x[k]:>11.4f}{d:>+10.4f}{ratio:>+8.2f}"
              f"{seb:>9.4f}{sea:>9.4f}  {mark}(t={t:+.2f})")
    print(f"\n  判据:|差| > 2σ(实测重测噪声) **且** |t| > 2.5(两臂 SE 合成)")
    print(f"  结果:{XLAB} 显著更好 {wins}/5 轴,显著更差 {losses}/5 轴,其余打平")

    # ── 3) 相对 base 的位置 ────────────────────────────────────
    if "base" in D:
        bb = D["base"]["scores"]
        print("\n" + "=" * 78)
        print("③ 两臂相对 base SD3.5-M(看是否都还在 base 之上)")
        print(f"  {'轴':<9s}{'base':>11s}{'baseline Δ':>13s}{XLAB+' Δ':>13s}")
        for k, lab in COLS:
            if k in bb and k in b and k in x:
                print(f"  {lab:<9s}{bb[k]:>11.4f}{b[k]-bb[k]:>+13.4f}{x[k]-bb[k]:>+13.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
