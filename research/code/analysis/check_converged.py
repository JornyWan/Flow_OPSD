#!/usr/bin/env python
"""判断某个 arm 是否已收敛(读离线 wandb 的 per-epoch monitor_avg,不占 GPU)。

判据:末段窗口相对前一窗口的提升 < 阈值,且已跑够最少步数。
用法: check_converged.py <run目录或.wandb> [--min-epoch N] [--tol 0.004]
退出码 0=已收敛, 1=未收敛
"""
import sys, os, glob, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_wandb import read

def main():
    args = sys.argv[1:]
    path = args[0] if args and not args[0].startswith("--") else None
    def opt(name, d):
        return type(d)(args[args.index(name)+1]) if name in args else d
    min_epoch = opt("--min-epoch", 300); tol = opt("--tol", 0.004)
    if path and os.path.isdir(path):
        c = glob.glob(os.path.join(path, "*.wandb")); path = c[0] if c else None
    if not path:
        runs = sorted(glob.glob("/midtier/minilab/scratch/chw4026/.cache/wandb/wandb/offline-run-*/*.wandb"),
                      key=os.path.getmtime)
        path = runs[-1] if runs else None
    if not path: print("找不到 .wandb"); return 1
    s = read(path)
    key = "monitor_avg" if "monitor_avg" in s else next((k for k in s if k.startswith("monitor_")), None)
    if not key: print("没有 monitor_* 指标"); return 1
    pts = sorted(s[key]); n = len(pts); last = pts[-1][0]
    print(f"# {os.path.basename(os.path.dirname(path))}  {key}  epoch≈{last}  ({n} 点)")
    if n < 40: print(f"  数据太少 ({n} 点)"); return 1
    w = max(15, n // 8)
    cur = statistics.mean(v for _, v in pts[-w:])
    prev = statistics.mean(v for _, v in pts[-2*w:-w])
    delta = cur - prev
    # 同时看各轴
    axes = {k: sorted(s[k]) for k in s if k.startswith("monitor_") and k != key}
    print(f"  末窗均值 {cur:.4f}  前窗均值 {prev:.4f}  变化 {delta:+.4f}  (窗口 {w} epoch)")
    for k, p in sorted(axes.items()):
        if len(p) < 2*w: continue
        c2 = statistics.mean(v for _, v in p[-w:]); p2 = statistics.mean(v for _, v in p[-2*w:-w])
        print(f"    {k:<22} {c2:.4f}  ({c2-p2:+.4f})")
    ok = last >= min_epoch and delta < tol
    print(f"  => {'✅ 已收敛' if ok else '⏳ 未收敛'}  (需 epoch≥{min_epoch} 且 变化<{tol})")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
