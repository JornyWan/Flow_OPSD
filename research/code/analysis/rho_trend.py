#!/usr/bin/env python
"""frozen-const 下 rho 扫描的 realized 增益趋势。

按 run_name 定位(不靠时间戳,wandb_by_name.py 的教训),取每 epoch 的 rtc_realized。
该量极噪(单 run 内摆动可达 20 倍),故用中位数 + Mann-Whitney,不用均值/t 检验。
"""
import sys, glob, os, statistics as st, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_wandb import read
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

W = "/midtier/minilab/scratch/chw4026/.cache/wandb/wandb/"

def run_name(d):
    f = glob.glob(d + "/*.wandb")
    if not f: return None
    try:
        ds = DataStore(); ds.open_for_scan(f[0])   # 正在跑的 run 文件头可能还是空的
    except Exception:
        return None
    for _ in range(400):
        try: data = ds.scan_data()
        except Exception: break
        if data is None: break
        rec = pb.Record()
        try: rec.ParseFromString(data)
        except Exception: continue
        if rec.WhichOneof("record_type") == "run": return rec.run.display_name
    return None

def mannwhitney(a, b):
    """正态近似的秩和检验;n 小的时候只做参考。"""
    c = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    vals = [x[0] for x in c]
    r = [0.0] * len(c)
    while i < len(c):
        j = i
        while j + 1 < len(c) and vals[j + 1] == vals[i]: j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1): r[k] = avg
        i = j + 1
    R1 = sum(r[k] for k in range(len(c)) if c[k][1] == 0)
    n1, n2 = len(a), len(b)
    U = R1 - n1 * (n1 + 1) / 2.0
    mu = n1 * n2 / 2.0
    sd = (n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5
    return (U - mu) / sd if sd > 0 else 0.0

if __name__ != "__main__":
    raise SystemExit  # 供 import 复用 mannwhitney/read 时不跑主流程

# run_name 片段 -> 展示标签
WANT = [("RTC-RAW-MEAS-r010-frozen ",      "rho=0.010  H200"),
        ("RTC-RAW-MEAS-r100-frozen ",      "rho=0.100  H200"),
        ("RTC-RAW-MEAS-r100-frozen-h100rep", "rho=0.100  H100 (硬件对照)"),
        ("RTC-RAW-MEAS-r300-frozen ",      "rho=0.300  H100"),
        ("RTC-RAW-MEAS-r1000-frozen ",     "rho=1.000  H100")]

found = {}
for d in sorted(glob.glob(W + "offline-run-*"), key=os.path.getmtime, reverse=True):
    nm = run_name(d)
    if not nm: continue
    for pat, lab in WANT:
        if pat in nm and lab not in found:      # 取最新的一个
            try: s = read(glob.glob(d + "/*.wandb")[0])
            except Exception: continue
            v = [x for _, x in s.get("rtc_realized", [])]
            if len(v) >= int(os.environ.get("MIN_N", "10")):   # 过滤掉短 run(如 3 epoch)
                found[lab] = v

print(f"{'条件':<28}{'n':>4}{'median':>14}{'IQR':>24}")
for _, lab in WANT:
    if lab not in found: print(f"{lab:<28}   (无满足 n>={os.environ.get('MIN_N','10')} 的 run)"); continue
    v = sorted(found[lab]); n = len(v)
    q1, q3 = v[n // 4], v[3 * n // 4]
    print(f"{lab:<28}{n:>4}{st.median(v):>14.3e}   [{q1:.2e}, {q3:.2e}]")

print("\n两两 Mann-Whitney (|z|>1.96 才算显著):")
labs = [l for _, l in WANT if l in found]
for a, b in itertools.combinations(labs, 2):
    z = mannwhitney(found[a], found[b])
    ratio = st.median(found[b]) / st.median(found[a]) if st.median(found[a]) else float("nan")
    mark = "  <-- 显著" if abs(z) > 1.96 else ""
    print(f"  {a:<28} vs {b:<28} z={z:+.2f}  中位数比={ratio:.2f}{mark}")
