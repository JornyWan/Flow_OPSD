#!/usr/bin/env python
"""按 run_name 精确定位 wandb offline run —— 不靠时间戳猜。
时间戳匹配曾把 curve_r30 误认成 d100_nocap,导致一次无效对比。"""
import glob, os, json, sys
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb
W = "/midtier/minilab/scratch/chw4026/.cache/wandb/wandb/"
def run_name(d):
    f = glob.glob(d + "/*.wandb")
    if not f: return None
    try:
        ds = DataStore(); ds.open_for_scan(f[0])
    except Exception:
        return None   # 刚创建的 run 其 .wandb 可能是 0 字节,跳过而非抛异常
    for _ in range(400):
        try: data = ds.scan_data()
        except Exception: break
        if data is None: break
        rec = pb.Record()
        try: rec.ParseFromString(data)
        except Exception: continue
        if rec.WhichOneof("record_type") == "config":
            for it in rec.config.update:
                if it.key == "run_name":
                    try: return json.loads(it.value_json)
                    except Exception: return it.value_json
    return None
def find(pat):
    """返回最新的、run_name 含 pat 的 run 目录"""
    for d in sorted(glob.glob(W + "offline-run-*"), key=os.path.getmtime, reverse=True):
        n = run_name(d)
        if n and pat in n: return d
    return None
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(find(sys.argv[1]) or "(未找到)")
    else:
        for d in sorted(glob.glob(W + "offline-run-2026082*"), key=os.path.getmtime, reverse=True)[:10]:
            print(f"{os.path.basename(d):40s} -> {run_name(d)}")
