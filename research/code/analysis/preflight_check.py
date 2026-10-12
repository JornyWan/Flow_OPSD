#!/usr/bin/env python
"""训练提交前的一致性自检。任何一项不符即非零退出,不让作业起跑。

动机(2026-08-30 用户要求):"每次 run 训练前都检查是不是保持一致的数据等设置"。
本项目已有的 `opsd_manifest_diff.py` 是**事后**比对 run_config.json —— 跑完才知道口径错了,
一夜算力就废了。本脚本在**提交前**查。

用法:
  preflight_check.py --repo <训练用仓库> --config sd35_open3 [--nproc 3] [--seed N]
                     [--expect-updates 100] [--expect-fork-clean]
"""
import argparse, hashlib, json, os, re, subprocess, sys

OPD = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
# ⚠️ 必须用 conda env 的解释器 —— 系统 python3 没有 ml_collections
ENV_PY = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/conda_envs/envs/diffusionopd/bin/python"
PRISTINE = f"{OPD}/CA-OPD/reference_papers/DiffusionOPSD-pristine-9c4c48c"
# 冻结口径(0041 §⑤)。数据 md5 来自 2026-08-30 实测,与论文 recipe 的 SHA-256 校验一致。
PINNED = {
    "data/pickapic/train.txt": ("4741a943d082ab05e812c111ba6d3efd", 25415),
    # ⚠️ test.txt 末行无换行符 => `wc -l` 报 999,实际 **1000** 条
    #    (评测 JSON 的 num_prompts=1000 佐证)。项目文档里的"999"是 wc -l 的假象。
    "data/pickapic/test.txt":  ("43771716e454e8fbbef7d87ec80dc2ff", 1000),
}
# 这些 env 会**改变目标函数**;有的还不进 run_config.json,只能在提交前查
DANGEROUS_ENV = {
    # 未设 == 代码默认 "adaptive",安全;只有显式 =frozen 才危险
    "OPSD_WF_MODE": ({"adaptive"}, "冻结归一化器会改目标函数,且不进 manifest"),
    "RPI_A1": (None, "A1 干预;baseline/探针必须为 0 或未设"),
    "RPI_GATEA_PROBE": (None, "探针会多做前向;A/B 臂应为 0 或未设"),
    "RPI_A1_TOL": (None, "A1 容差"),
    "RPI_N2": (None, "N2 干预(冻结 |r| 归一化分母);baseline/其它臂必须为 0 或未设"),
    "RPI_Q2A_PROBE": (None, "Gate Q2a 探针(逐样本 LoRA 参数梯度,多次 backward);A/B 臂必须为 0"),
    "RPI_BNOISE_PROBE": (None, "Gate B 探针(半批梯度噪声尺度,只读 .grad);A/B 臂必须为 0"),
    "RPI_BNOISE_SPLIT": (None, "Gate B 半批切分方式;prompt 会重排 perm(不改整批梯度,但属训练路径)"),
    "RPI_RTCREACH_PROBE": (None, "RTC 可达性探针(每周期 1+K 次 VJP,只读);A/B 臂必须为 0"),
    "RPI_QOPSD": (None, "Q-OPSD 训练干预:用 d_Q 替换 y± 构造方向 + Q1b 在线回溯;canonical 臂必须为 0"),
    "RPI_QCORE": (None, "Q-OPSD-Core 训练干预:保留官方两步构造与 advantage,只把 oracle 换成 ∇R(D(F_old(s,y)));canonical 臂必须为 0"),
    "RPI_QCORE_SUFFIX": ({"dpm2"}, "Q 后缀 solver;官方=dpm2。设成 flow 会换掉 oracle 所在的 solver 家族(§4.516),必须显式声明"),
    "RPI_QCORE_FROZENREF": ({"0"}, "Q oracle 的去噪器换成冻结底模(§4.847);官方=0"),
    "RPI_QCORE_LOOKAHEAD": ({"0"}, "Q oracle 只走一步取下一态 x̂0(L1,§4.844);官方=0,设 1 会换掉 target 方向,必须显式声明"),
    "RPI_QCORE_FLOW_STEPS": ({"0"}, "flow 后缀细分步数;>0 会改变可微后缀的前向次数与显存(§4.517-P),必须显式声明"),
    "RPI_HYBRID_SIGMA": ({"0"}, "混合模式的 σ 门;canonical 必须为 0"),
    "RPI_SELECT_M": ({"0"}, "提案 H:选择式目标的候选数;canonical 必须为 0"),
    "RPI_SELECT_TR": ({"0"}, "选择方向是否投影回信任域;默认 0"),
    "RPI_SELECT_TAU": ({"0"}, "reward 加权平均的温度(以 reward 标准差为尺度);0=argmax,canonical 必须为 0"),
    "RPI_MULTIK_SAMPLE": ({"0"}, "每步随机取 K 个已存状态;0=全用,canonical 必须为 0"),
    "RPI_MULTIK_SYNTH": ({"0"}, "额外状态用 y0 重加噪代替 rollout 态;canonical 必须为 0"),
    "RPI_VADV": ({"0"}, "提案 B:把 cert 的二值接受权重换成连续的已验证增益;canonical 必须为 0"),
    "RPI_MULTIK_STATES": ({""}, "提案 A:把 y± 传播到额外 rollout 状态;canonical 必须为空"),
    "RPI_RBLT": (None, "RB-LT(借 PCFM 的终点修正→插值回中间时刻):换掉 query 态与正目标;canonical 必须为 0 或未设"),
    "RPI_ROTQ": (None, "ROTQ 单变量对照:canonical 目标 + 轮转 σ_k 的 query 态;canonical 必须为 0 或未设"),
    "RPI_LTVAR": ({""}, "LT 目标变体(pcfm|bridge);canonical 与 ours 必须为空"),
    "RPI_TPD": ({"0"}, "TPD 倾斜后验蒸馏(§4.868):额外态目标换成 y0,k ± Δ_k(续采加权);官方=0"),
    "RPI_TPD_M": (None, "TPD 的续采条数;只在 RPI_TPD=1 时有意义,其它臂必须未设"),
    "RPI_TPD_XDIAG": ({"0"}, "TPD 归因探针(§4.889):同一 z_k 再做一组独立续采,记两组 Δ 的夹角;纯日志、成本 ×2;官方=0"),
    "RPI_MULTIK_TARGET": ({"prop"}, "MSP 额外态目标分解(§4.861):prop=现有 | dir=只留 reward 方向 | cons=只留一致性;官方/现有=prop"),
    "RPI_EMA_RESTART": ({"0"}, "周期性 EMA 重启(§4.860):每 P 次更新 live、old ← EMA;官方=0"),
    "RPI_MULTIK_WFDETACH": ({"0"}, "MSP 额外态自适应归一化 detach(§4.876,对齐官方/主项写法);现有 MSP=0"),
    "RPI_MK_GATE": ({"0"}, "MSP 额外态门控(§4.883):‖g_额外‖>τ·‖g_主‖ 时把额外态梯度缩回;现有 MSP=0"),
    "RPI_MK_SPLIT": ({"0"}, "MSP 额外态独立 Adam(§4.888):主优化器只吃主项梯度,额外态走另一套 m/v、位移乘 λ;现有 MSP=0"),
    "RPI_STEP_CAP": ({"0"}, "每步参数位移上限(§4.875 修复臂 B):‖Δθ‖ 超过 c×近期 EMA 时按比例缩回;官方=0"),
    "RPI_OPT_DIAG": ({"0"}, "优化器诊断日志(§4.875):纯记录 Adam 有效步长与主项/额外态梯度分解,不改训练;官方=0"),
    "RPI_MULTIK_SUFFIX": ({""}, "低噪 suffix 锚(§4.855 (a)):在 query 之下的 σ 上加 MSP 式额外态;canonical 必须为空"),
    "RPI_MULTIK_ADVPOS": (None, "MSP 额外态 r1 下限 0.5(§4.839,去掉坏样本的正反馈项);canonical 必须为 0 或未设"),
    "RPI_LTOPSD": (None, "LT-OPSD 直接迁移版(§4.830):ε–y± 全局插值 + 下一步 state 比较;canonical 必须为 0 或未设"),
    "RPI_MULTIK_WEIGHT": ({"uniform"}, "提案 A 的状态权重;snr 需显式声明"),
    "RPI_KFIT": (None, "κ_fit 探针:测目标位移被实现的比例;**纯诊断,测完复原参数**,A/B 臂应为 0"),
    "RPI_KFIT_PS": (None, "κ_fit 逐样本梯度对照:分离批内稀释与表示上限;同为纯诊断,A/B 臂应为 0"),
    "PUBLIC_NUM_GROUPS": (None, "每 epoch 的 prompt 数(默认 48);改它会改 B_eff,canonical 臂必须未设"),
    "RPI_A1_MODE": (None, "A1 可行性判据:abs(原版)| rel(A1-v2 相对改善);canonical 臂应为 abs 或未设"),
    "RPI_QOPSD_DIR": (None, "Q-OPSD 方向:q(d_Q)| canon(仅回溯消融);canonical 臂应为 q 或未设"),
    "RPI_Q2A_NSAMP": (None, "Gate Q2a 每 chunk 取样数"),
    "RPI_Q1B_PROBE": (None, "Gate Q1b 探针(额外 3 个回溯步长的 exact suffix);A/B 臂必须为 0"),
    "RPI_QGRAD_PROBE": (None, "Gate Q1 探针(穿后缀反传求 action-value 梯度);极慢,A/B 臂必须为 0"),
    "RPI_CERT_PROBE": (None, "cert 探针(每 chunk 多两次 rollout 续跑+打分);A/B 臂必须为 0"),
    "RPI_GATEP_PROBE": (None, "Gate P 探针(每 state 多构造 6 个候选目标);会大幅拖慢,A/B 臂必须为 0"),
    "RPI_N2_MOM": (None, "N2 的 EMA 动量"),
}

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--nproc", type=int, default=3)
    ap.add_argument("--seed", default=None)
    ap.add_argument("--expect-updates", type=int, default=None)
    ap.add_argument("--allow-env", action="append", default=[],
                    help="声明本次**有意**设置的干预 env(如 RPI_A1)。未声明的一律判失败。")
    ap.add_argument("--expect-fork-clean", action="store_true",
                    help="要求仓库与 pristine 只差 scripts/train_opsd_ri_sd3.py")
    a = ap.parse_args()
    fails, warns = [], []

    print(f"[preflight] repo={a.repo}")
    print(f"[preflight] config={a.config} nproc={a.nproc} seed={a.seed}")

    # ── 1) 数据 ────────────────────────────────────────────────
    for rel, (want_md5, want_n) in PINNED.items():
        p = os.path.join(a.repo, rel)
        os.path.isdir(os.path.dirname(p)) and os.listdir(os.path.dirname(p))  # 刷 NFS
        if not os.path.exists(p):
            fails.append(f"数据缺失: {rel}"); continue
        got_md5 = md5(p)
        got_n = sum(1 for _ in open(p, encoding="utf-8", errors="replace"))
        ok = (got_md5 == want_md5) and (got_n == want_n)
        print(f"  {'✅' if ok else '❌'} {rel}: n={got_n}(期望{want_n}) md5={got_md5[:12]}…")
        if not ok:
            fails.append(f"数据不一致: {rel} md5={got_md5} n={got_n}")

    # ── 2) 代码状态 ────────────────────────────────────────────
    if a.expect_fork_clean:
        r = subprocess.run(["diff", "-rq", "-x", ".git", "-x", "__pycache__",
                            "-x", "*COMMIT.txt", "-x", "data", "-x", "reward_ckpts",
                            "-x", "outputs", "-x", "wandb",
                            PRISTINE, a.repo], capture_output=True, text=True)
        diffs = [l for l in r.stdout.splitlines() if l.strip()]
        # fork 允许不同的文件:训练脚本(探针+A1)、config/public.py 与 config/mixed.py(均为 RPI_SEED 钩子)
        _ALLOWED = ("scripts/train_opsd_ri_sd3.py", "config/public.py", "config/mixed.py")

        def _declared(line):
            """按**精确相对路径**判定,不用子串匹配。

            2026-08-31 踩坑:原来写的是 `any(a in line for a in _ALLOWED)`,
            于是仓库里一个 `train_opsd_ri_sd3.py.bak_pre_n2` 备份文件
            被当成"已声明文件"放行 —— 守卫的全部意义就是抓未声明差异,
            这里却把它漏了。凡是文件名**包含**允许名的文件都能蒙混过关。
            """
            cand = []
            # diff -rq 两种句式:
            #   "Files <A>/<rel> and <B>/<rel> differ"
            #   "Only in <dir>: <name>"
            mm = re.match(r"^Files (\S+) and (\S+) differ$", line)
            if mm:
                for side, root in ((mm.group(1), PRISTINE), (mm.group(2), a.repo)):
                    if side.startswith(root.rstrip("/") + "/"):
                        cand.append(side[len(root.rstrip("/")) + 1:])
            else:
                mo = re.match(r"^Only in (.+?): (.+)$", line)
                if mo:
                    d = mo.group(1).rstrip("/")
                    for root in (PRISTINE.rstrip("/"), a.repo.rstrip("/")):
                        if d == root:
                            cand.append(mo.group(2))
                        elif d.startswith(root + "/"):
                            cand.append(f"{d[len(root) + 1:]}/{mo.group(2)}")
            return bool(cand) and all(c in _ALLOWED for c in cand)

        unexpected = [l for l in diffs if not _declared(l)]
        print(f"  {'✅' if not unexpected else '❌'} 与 pristine 的差异: {len(diffs)} 项"
              f"{'(均为已声明文件)' if diffs and not unexpected else ''}")
        for l in unexpected:
            print(f"      ❌ 未预期: {l}")
        if unexpected:
            fails.append(f"仓库有 {len(unexpected)} 处未预期改动")

    # ── 3) 危险 env ────────────────────────────────────────────
    for k, (want, why) in DANGEROUS_ENV.items():
        got = os.environ.get(k)
        if k in a.allow_env:
            print(f"  ✅ env {k}={got!r} (**已声明**为本次干预变量)")
            continue
        if want is None:
            ok = got in (None, "0")
        else:                       # want 是允许值集合;未设 = 用代码默认 = 安全
            ok = (got is None) or (got in want)
        print(f"  {'✅' if ok else '❌'} env {k}={got!r} ({why})")
        if not ok:
            fails.append(f"危险 env: {k}={got!r}")

    # ── 4) config 解析 ─────────────────────────────────────────
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{OPD}/CA-OPD/code/shim_rpi:{a.repo}"
    for v in ("PUBLIC_N_GPUS", "PUBLIC_LAUNCH_WORLD_SIZE", "PUBLIC_POLICY_WORLD_SIZE"):
        env[v] = str(a.nproc)
    code = (
        "import json;from config.public import get_config;"
        f"c=get_config('{a.config}');o=c.opsd;"
        "print(json.dumps({'reward_fn':dict(c.reward_fn),'beta':float(c.beta),"
        "'num_epochs':int(c.num_epochs),'seed':c.seed,'resolution':int(c.resolution),"
        "'gs':float(c.sample.guidance_scale),'noise':float(c.sample.noise_level),"
        "'dataset':c.dataset,'opa':o.get('opa'),'opa_kind':o.get('opa_reward_kind'),"
        "'cert':o.get('opa_cert'),'dir_mode':o.get('opa_dir_mode'),"
        "'opa_ls':o.get('opa_ls'),'opa_topk':o.get('opa_topk'),"
        "'opa_mq':o.get('opa_mq_sigmas'),"
        "'bs':int(c.sample.train_batch_size),'accum':int(c.train.gradient_accumulation_steps)}))"
    )
    r = subprocess.run([ENV_PY, "-c", code], cwd=a.repo, env=env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print("  ❌ config 解析失败:"); print("     " + r.stderr.strip().splitlines()[-1][:160])
        fails.append("config 解析失败")
    else:
        cfg = json.loads(r.stdout.strip().splitlines()[-1])
        total = cfg["bs"] * a.nproc * cfg["accum"]
        print(f"  ✅ config: reward_fn={cfg['reward_fn']} beta={cfg['beta']} "
              f"epochs={cfg['num_epochs']} seed={cfg['seed']}")
        print(f"     协议: {cfg['resolution']}² gs={cfg['gs']} noise={cfg['noise']} "
              f"| samples/epoch={cfg['bs']}x{a.nproc}x{cfg['accum']}={total}")
        print(f"     OPA: opa={cfg['opa']} kind={cfg['opa_kind']} cert={cfg['cert']} "
              f"dir_mode={cfg['dir_mode']}")
        # 我们加进 DiffusionOPSD-main 的消融开关必须全关。
        # 注意:**键不存在(None)是更安全的情况** —— 说明该仓库(如 pristine 派生的 fork)
        # 根本没有这个功能,而不是有但关着。两者都算 canonical,有值且非默认才是问题。
        for k, want in (("opa_ls", 0), ("opa_topk", 0), ("opa_mq", "")):
            v = cfg.get(k)
            if v is None:
                print(f"     ✅ {k}: 该仓库无此功能(pristine 派生)")
            elif v != want:
                fails.append(f"消融开关未关: {k}={v!r}(canonical 应为 {want!r})")
                print(f"     ❌ {k}={v!r} 应为 {want!r}")
            else:
                print(f"     ✅ {k}={v!r}(已关)")
        if total != 1152:
            warns.append(f"samples/epoch={total},与冻结口径 1152 不符")
        if a.expect_updates is not None and cfg["num_epochs"] != a.expect_updates:
            warns.append(f"config 里 num_epochs={cfg['num_epochs']},"
                         f"命令行会覆盖为 {a.expect_updates}(正常)")
        if a.seed is not None and str(cfg["seed"]) != str(a.seed):
            warns.append(f"config 里 seed={cfg['seed']},命令行会覆盖为 {a.seed}(正常)")
        if os.path.realpath(cfg["dataset"]) != os.path.realpath(
                os.path.join(a.repo, "data/pickapic")):
            warns.append(f"dataset 指向 {cfg['dataset']}")

    print()
    for w in warns:
        print(f"  ⚠️  {w}")
    if fails:
        print(f"\n[preflight] ❌ 不通过,{len(fails)} 项:")
        for f in fails:
            print(f"    - {f}")
        return 1
    print("[preflight] ✅ 全部通过")
    return 0

if __name__ == "__main__":
    sys.exit(main())
