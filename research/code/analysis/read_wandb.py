#!/usr/bin/env python
"""读 wandb offline run 的 .wandb 二进制记录,导出每 epoch 的 monitor_* 曲线。

计算节点离线,wandb 只落盘不上传;summary 又要 run 结束才写。所以直接解析 datastore。
用途:不占 H200、不跑昂贵的在线 eval,就能判断某个 arm 是否收敛。
"""
import sys, glob, os, collections
from wandb.sdk.internal.datastore import DataStore
from wandb.proto import wandb_internal_pb2 as pb

def read(path):
    ds = DataStore(); ds.open_for_scan(path)
    series = collections.defaultdict(list)   # key -> [(step, value)]
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None: break
        rec = pb.Record()
        try: rec.ParseFromString(data)
        except Exception: continue
        if rec.WhichOneof("record_type") != "history": continue
        step = rec.history.step.num
        for item in rec.history.item:
            key = item.key or (".".join(item.nested_key) if item.nested_key else "")
            if not key: continue
            try: v = float(item.value_json)
            except Exception: continue
            series[key].append((step, v))
    return series

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path and os.path.isdir(path):
        cand = glob.glob(os.path.join(path, "*.wandb")); path = cand[0] if cand else None
    if not path:
        runs = sorted(glob.glob("/midtier/minilab/scratch/chw4026/.cache/wandb/wandb/offline-run-*/*.wandb"),
                      key=os.path.getmtime)
        path = runs[-1] if runs else None
    if not path: sys.exit("找不到 .wandb 文件")
    print(f"# {path}\n")
    s = read(path)
    # 白名单之外的诊断指标(ste_w_*, retain_kl, realiz_dT_rel ...)默认看不到,
    # 用 --keys a,b 追加,或 --all 打印全部数值型 key。
    _extra = []
    _all = "--all" in sys.argv
    for _a in sys.argv[1:]:
        if _a.startswith("--keys="):
            _extra = [x.strip() for x in _a.split("=", 1)[1].split(",") if x.strip()]
    keys = [k for k in sorted(s) if _all or k.startswith("monitor_") or k in
            ("ca_opd_w_max", "train_reverse_kl", "policy_loss", "grad_norm") or k in _extra]
    for k in keys:
        pts = sorted(s[k])
        if not pts: continue
        idx = sorted(set([0, len(pts)//4, len(pts)//2, 3*len(pts)//4, len(pts)-1]))
        vals = "  ".join(f"@{pts[i][0]}={pts[i][1]:.4f}" for i in idx)
        # 末尾 20% 的均值 vs 前一段,粗判是否走平
        n=len(pts); tail=[v for _,v in pts[int(n*0.8):]]; prev=[v for _,v in pts[int(n*0.6):int(n*0.8)]]
        trend=""
        if len(tail)>2 and len(prev)>2:
            a,b=sum(prev)/len(prev), sum(tail)/len(tail)
            trend=f"   末段变化 {b-a:+.4f}"
        print(f"{k:<20} n={len(pts):<5}{vals}{trend}")
