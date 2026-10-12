#!/usr/bin/env python
"""把官方复现的 cross_eval 结果与论文主表逐列对照。

论文数值取自 arXiv-2608.24646v1 `tables/main_results.tex`,SD3.5-M / reward-specific / 100 updates。
只有 5 列可对照(见 PLAN §3.4.0):HPSv3/DeQA 我们没有,AltCLIP/Point/Pair 是论文内部评测器。
"""
import json, sys, glob, os

# 论文 SD3.5-M reward-specific 100-updates 行
PAPER = {
    "OPSD":         dict(pickscore=24.94, clipscore=0.340, hpsv2=0.390, aesthetic=12.08, imagereward=1.76),
    "ReFL":         dict(pickscore=23.92, clipscore=0.308, hpsv2=0.358, aesthetic=12.09, imagereward=1.28),
    "DiffusionNFT": dict(pickscore=23.43, clipscore=0.298, hpsv2=0.336, aesthetic=9.11,  imagereward=1.46),
    "FlowGRPO":     dict(pickscore=22.96, clipscore=0.275, hpsv2=0.305, aesthetic=5.46,  imagereward=1.01),
    "SD3.5-M base": dict(pickscore=20.51, clipscore=0.237, hpsv2=0.204, aesthetic=5.13,  imagereward=-0.58),
}
COLS = [("pickscore","Pick"),("clipscore","CLIP"),("hpsv2","HPSv2.1"),("aesthetic","Aes"),("imagereward","ImgR")]

def load(p):
    with open(p) as f: d = json.load(f)
    return d.get("scores", {}), d.get("score_stats", {}), d

def main():
    pats = sys.argv[1:] or sorted(glob.glob(
        "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/opsd_eval_ck*_*.json"))
    if not pats:
        print("未找到 opsd_eval_*.json —— opsdeval 尚未产出结果"); return
    for p in pats:
        sc, stt, raw = load(p)
        tag = os.path.basename(p)
        print(f"\n{'='*74}\n{tag}  (prompts={raw.get('prompt_set_name','?')} protocol={raw.get('protocol_name','?')})")
        print(f"{'列':>9s} {'我们复现':>18s} {'论文 OPSD':>10s} {'差':>9s} {'':>6s}  参照(ReFL/NFT/FlowGRPO/base)")
        train_rw = raw.get("train_reward", "?")
        _multi = "," in str(train_rw)
        print(f"  ⚠️ 论文 OPSD 行是 **reward-specific**（表 caption: \\cmark = 每列用该列 reward 训练的 checkpoint）。")
        if _multi:
            # multi-reward 臂(如 Open3):论文主表里没有它的对应行 ——
            # 主表每列都来自单 reward 专训 checkpoint,而 Open3 是一个 checkpoint 打全部列。
            # 论文的 Open3 对比只有图(assets/robustness.png),**没有数值表**(见 0033 §①)。
            print(f"     本臂是 **multi-reward**（{train_rw}）⇒ **与论文主表任何一列都不可直接比**：")
            print(f"     主表每列来自该列 reward 的专训 checkpoint，本臂是一个 checkpoint 打全部列。")
            print(f"     且论文的 Open3 对比**只有图、没有数值表** ⇒ 下面的『差』仅供参考，")
            print(f"     **不能据此说复现成功或失败**。可用的对照是右列的 base/ReFL/NFT/FlowGRPO。")
        else:
            print(f"     我们只训练了 `{train_rw}` 一个臂 ⇒ **只有 {train_rw} 那一列可比**，其余列论文用的是别的 checkpoint。")
        for k, disp in COLS:
            if k not in sc:
                print(f"{disp:>9s} {'未评测':>18s}"); continue
            cmp_ok = ("—仅参考" if _multi else ("✅可比" if k == train_rw else "❌不可比"))
            v = sc[k]; s = stt.get(k, {})
            se = s.get("se", float('nan')); n = s.get("n", "?")
            pv = PAPER["OPSD"].get(k)
            d = v - pv if pv is not None else float('nan')
            others = "/".join(f"{PAPER[m][k]:g}" for m in ("ReFL","DiffusionNFT","FlowGRPO","SD3.5-M base"))
            se_s = f"±{se:<7.4f}" if se == se else " " * 8
            print(f"{disp:>9s} {v:10.4f}{se_s} {pv:10.3f} {d:+9.4f} {cmp_ok}  {others}")
        print(f"  (n={n};±为标准误)")
    print("\n注意:\n"
          "  - Pick 有 x26 尺度差 -> 用 pickscore_scale=raw 的那份与论文对照;\n"
          "  - Aes 论文 12.08 vs 我们约 6.28:**已查清,不是评测器版本问题**(两仓库 scorer 逐行相同,\n"
          "    且两篇报告的 base 一致:5.13 vs 5.18)。12.08 是直接用 aes 当 reward 专训的效果\n"
          "    (ReFL 12.09/NFT 9.11);我们只训 pickscore,故停在 6.28。见 0032。\n"
          "  - HPSv3/DeQA/AltCLIP/Point/Pair 不可对照(见 PLAN §3.4.0)。")

if __name__ == "__main__":
    main()
