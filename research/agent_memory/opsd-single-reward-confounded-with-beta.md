---
name: opsd-single-reward-confounded-with-beta
description: "OPSD 官方协议里单 reward 用 β=1.0、联合 open3 用 β=0.1 —— 单 reward 的结果同时差两件事,不能读成'方法需要多 reward'"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-10-01T18:10:44.796Z
---

OPSD 官方协议的两个主表块**同时**差两件事(2026-10-01 从源码核实,三处独立确认):

- `config/public.py:92` —— `config.beta = 0.1` **只在 `if reward == "open3"` 分支内**
- `config/nft.py:103` —— 上游默认 `config.beta = 1.0`
- 单 reward config(`sd35_hpsv2`/`sd35_aesthetic`/`sd35_imagereward`)自己打印 `beta = 1.0`

⇒ **联合 open3 = β 0.1 + 三 reward;单 reward = β 1.0 + 一 reward。**

**Why:** MSP(多状态传播)在联合 open3 上是 +10.70×MDE(drawbench flow-4, t=+23.60),
在单 reward hpsv2 上**恰好为零**(±0.0001, |t|<0.4, 回补 ±0.7%,§4.719 判 R2 无效)。
把这个零读成"方法需要多个冲突 reward"是**错的归因** —— β 也差了 10 倍。
判定格是 2×2 的第四格(open3 + MSP + β=1.0,臂 `beta10msp`,§4.719 预登记 I 轴判据)。

**How to apply:**
- 任何跨"联合 vs 单 reward"的比较,先问 β 是否也变了;两个旋钮不能混着归因。
- 要在 open3 上改 β:`ARM=opacfg`(canonical)或 `ARM=multikF4cfg`(MSP),
  配 `OPA_CFG=--config.beta=X`。已验证该覆盖能压过 `public.py:92`,且不动 `reward_fn`/`opa_rho`。
- `config.beta` 在 OPA 路径的主状态(`train_opsd_ri_sd3.py:2254/2255`)与 multik 额外状态
  (2673/2674)用同一个值,所以改 β 是均匀的。**`/β` 那个除法只在 x0/NFT 路径(2914),我们不走。**
- ⚠️ 不要据 [[identity-is-not-evidence]] 之外再去"证明"β 改变位移需求 ——
  `δ* = F/β` 是代数恒等式。决定性证据只能是 2×2 交互的实测。
- ⚠️ 也不要把论文里"β 缩放**函数空间**位移"这句,按被证伪的"β 导致 10× **权重**位移"
  (实测 ‖ΔW‖ 31.12 vs 29.36)去纠正 —— 两者不等价,见 [[dont-inflate-decisive-experiments]]。
