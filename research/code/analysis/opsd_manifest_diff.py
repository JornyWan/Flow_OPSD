#!/usr/bin/env python
"""比对多个 OPSD run 的 run_config.json,把"实验口径一致"从承诺变成会报错的检查。

动机:本项目栽过两次同类错误 —— 守卫校验了 A 配置却启动了 B(0010/0015 全部作废);
`rtc_raw` 被当作"等价 OPSD"用了很久,直到逐项审计才发现 margin/缩放/opa_mb 三处偏离。
口径一致不能靠记忆,要靠机器拒绝。

用法:
    opsd_manifest_diff.py RUN_DIR [RUN_DIR ...] [--allow FIELD ...]

`--allow` 声明本次实验**有意变动**的字段(即干预变量)。
任何**未声明**的差异 -> 打印并以非零退出。
"""
import json, os, sys

# 这些字段每次运行必然不同,不构成口径差异
IGNORE = {"run_name", "run_random_nonce", "wall_clock_start", "git_commit"}

def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out

def main():
    args = sys.argv[1:]
    allow, dirs = set(), []
    it = iter(range(len(args)))
    i = 0
    while i < len(args):
        if args[i] == "--allow":
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                allow.add(args[i]); i += 1
        else:
            dirs.append(args[i]); i += 1
    if len(dirs) < 2:
        print("需要至少两个 run 目录"); return 2

    mans = {}
    for d in dirs:
        p = os.path.join(d, "run_config.json")
        os.listdir(d) if os.path.isdir(d) else None   # 刷 NFS 属性缓存
        if not os.path.exists(p):
            print(f"❌ {d} 没有 run_config.json —— 该 run 无法核对口径"); return 2
        mans[os.path.basename(d.rstrip("/"))] = flat(json.load(open(p)))

    keys = set()
    for m in mans.values(): keys |= set(m)
    names = list(mans)

    undeclared, declared, missing = [], [], []
    for k in sorted(keys):
        if k in IGNORE: continue
        vals = {n: mans[n].get(k, "<缺失>") for n in names}
        if len(set(json.dumps(v, sort_keys=True) for v in vals.values())) == 1:
            continue
        if any(v == "<缺失>" for v in vals.values()):
            missing.append((k, vals))
        elif k in allow:
            declared.append((k, vals))
        else:
            undeclared.append((k, vals))

    def show(title, rows):
        if not rows: return
        print(f"\n{title}")
        for k, vals in rows:
            print(f"  {k}")
            for n, v in vals.items():
                print(f"      {n:<34s} {json.dumps(v, ensure_ascii=False)[:100]}")

    print(f"比对 {len(names)} 个 run: {', '.join(names)}")
    show("✅ 已声明的干预变量(--allow):", declared)
    show("⚠️ 字段在部分 run 中缺失(manifest 版本不一致?):", missing)
    show("❌ 未声明的口径差异 —— 这些 run 不能直接对比:", undeclared)

    if undeclared or missing:
        print(f"\n结论: 不可比。未声明差异 {len(undeclared)} 项,缺失字段 {len(missing)} 项。")
        return 1
    print("\n结论: ✅ 除已声明的干预变量外,所有口径字段一致,可以对比。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
