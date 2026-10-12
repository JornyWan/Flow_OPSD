#!/usr/bin/env python3
"""论文级基线对照表(codex 0023 §5 的比较关系,论文口径 geneval)。

比较链(缺任何一环,"守卫有效"就只能和自己的消融比):
  1. 发布 DiffusionOPD Student —— 初始化/锚点,本集群同代码重评
  2. Student -> DiffusionOPSD-direct (`crd`)      —— 机制 control,证明不加保护会怎样
  3. Student -> **MOPD/DiffusionOPD-continue** (`mopdc`) —— **论文自己的方法**,同算力预算
  4. Student -> 朴素 frozen-Student KL (`crkl5`)  —— 最低成本的 regularization baseline
  5. Student -> fixed one-sided same-state hinge (`crg`, w=10 已冻结)
  6. (待 Phase-1 gate) optimizer-aware projection

用法: 0029_baseline_table.py [step]        默认 181
"""
import json, os, sys, glob

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
E = os.path.join(O, "outputs", "eval")
# 论文口径:geneval 目录里的 `geneval` 字段(Student 0.9615,与原论文 ~0.96 吻合),
# 不是 `accuracy`(0.9376)。两者混用曾让表格与聚合分数不是同一个指标(codex 0023 §3.2)。
AX = [("pickscore", "pick", "drawbench", 0.00148),
      ("ocr",       "ocr",  "ocr",       0.0200),
      ("geneval",   "geneval", "geneval", 0.0297),
      ("clipscore", "clip", "drawbench", 0.00143),
      ("aesthetic", "aes",  "drawbench", 0.0179),
      ("hpsv2",     "hps",  "drawbench", 0.00110)]
ARMS = [("Student",   "Student",  "发布 Student(锚点)"),
        ("mopdc",     None,       "MOPD-continue(**论文方法**)"),
        ("crd",       None,       "OPSD-direct(无保护)"),
        ("crkl5",     None,       "朴素 frozen-Student KL"),
        ("crg",       None,       "one-sided hinge (w=10)")]

def val(tag, ds, key):
    p = os.path.join(E, f"{tag}_{ds}", "average_scores.json")
    try:
        os.listdir(os.path.dirname(p))
    except Exception:
        return None
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p)).get(key)
    except Exception:
        return None

def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "181"
    rows = {}
    for arm, fixed, label in ARMS:
        tag = fixed if fixed else f"{arm}_{step}"
        rows[label] = {sh: val(tag, ds, k) for k, sh, ds, _ in AX}
        rows[label]["_tag"] = tag
    ref = rows["发布 Student(锚点)"]
    print(f"=== 基线对照表(step {step};论文口径 geneval;差值单位 = 该轴重测 2σ) ===")
    hdr = f"{'臂':30s} " + " ".join(f"{sh:>9s}" for _, sh, _, _ in AX)
    print(hdr); print("-" * len(hdr))
    for label, r in rows.items():
        cells = []
        for k, sh, ds, sig in AX:
            v = r.get(sh)
            cells.append("      —  " if v is None else f"{v:9.4f}")
        print(f"{label:30s} " + " ".join(cells))
    print()
    print("--- 相对发布 Student 的差(单位 2σ;'—' = 数据未齐) ---")
    print(hdr); print("-" * len(hdr))
    for label, r in rows.items():
        if label.startswith("发布"):
            continue
        cells = []
        for k, sh, ds, sig in AX:
            v, rv = r.get(sh), ref.get(sh)
            cells.append("      —  " if (v is None or rv is None) else f"{(v-rv)/(2*sig):+9.2f}")
        print(f"{label:30s} " + " ".join(cells))
    print()
    miss = [f"{lab}({r['_tag']})" for lab, r in rows.items()
            if any(r.get(sh) is None for _, sh, _, _ in AX)]
    if miss:
        print("⚠ 数据未齐:", "; ".join(miss))
        print("  在全部臂齐备之前,**不得**据此表下「某方法更好」的结论 ——")
        print("  缺论文方法基线(mopdc)时,任何比较都只是与自己的消融比。")

if __name__ == "__main__":
    main()
