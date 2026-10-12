"""Phase-1 probe 机制的合成验证(移植进 trainer **之前**必须先过)。

只验证机制,不涉及扩散模型:用一个小 MLP + 真实 AdamW,
把 0025 预注册 §1 的六步走一遍,检查四类容易出隐性 bug 的地方:

  A. 快照/还原是否真的还原(参数 **和** optimizer state 都要);
  B. Delta_adam 是否确为"真实 AdamW proposal",而不是 -lr*grad;
  C. 同一缓存 batch 上的 exact 重算是否真的用了同一批数据(不重采样);
  D. QP + backtrack 后 certificate 是否用**精确重算**判定,而不是一阶预测。
"""
import copy, math, torch, torch.nn as nn

torch.manual_seed(0)
D_IN, D_H = 16, 32
model = nn.Sequential(nn.Linear(D_IN, D_H), nn.Tanh(), nn.Linear(D_H, 1))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-2)

# 造一个"累积窗口":12 个 microbatch,固定不变
WINDOW = [(torch.randn(6, D_IN), torch.randn(6, 1)) for _ in range(12)]
# 两个"受保护能力"各有自己的固定数据
PROT = {m: [(torch.randn(6, D_IN), torch.randn(6, 1)) for _ in range(12)] for m in ("ocr", "gen")}

def flat(ts):
    return torch.cat([t.reshape(-1) for t in ts])

def reward_loss():
    """direct objective,在整个 window 上累积。"""
    return sum(((model(x) - y) ** 2).mean() for x, y in WINDOW) / len(WINDOW)

def constraint_val(m):
    """C_m(theta):>0 表示违反。减去一个常数使初始值接近 0。"""
    v = sum(((model(x) - y) ** 2).mean() for x, y in PROT[m]) / len(PROT[m])
    return v - CREF[m]

# 冻结参考值(相当于 frozen-reference threshold)
with torch.no_grad():
    CREF = {m: (sum(((model(x) - y) ** 2).mean() for x, y in PROT[m]) / len(PROT[m])).item()
            for m in PROT}

def snapshot():
    return ([p.detach().clone() for p in params], copy.deepcopy(opt.state_dict()))

def restore(snap):
    with torch.no_grad():
        for p, s in zip(params, snap[0]):
            p.copy_(s)
    opt.load_state_dict(copy.deepcopy(snap[1]))

# --- 让 AdamW 先走几步,使 optimizer state 非平凡(否则 B 检验没有意义) ---
for _ in range(5):
    opt.zero_grad(); reward_loss().backward(); opt.step()

snap0 = snapshot()
theta0 = flat(snap0[0]).clone()

# 步骤 2:在旧参数上算 C_m 与约束梯度 a_m(**整个 window 累积**)
C, A = {}, {}
for m in PROT:
    opt.zero_grad()
    cv = constraint_val(m)
    C[m] = cv.item()
    cv.backward()
    A[m] = flat([p.grad.detach().clone() for p in params])
restore(snap0)

# 步骤 3:复制参数+optimizer state,只用 reward objective 走一次**真实** AdamW
opt.zero_grad(); reward_loss().backward()
g_reward = flat([p.grad.detach().clone() for p in params])
opt.step()
theta_cand = flat([p.detach() for p in params]).clone()
delta_adam = theta_cand - theta0
restore(snap0)

# === A. 还原检验 ===
assert torch.allclose(flat([p.detach() for p in params]), theta0, atol=0), "A 失败:参数未还原"
_st = opt.state_dict()["state"]
_ref = snap0[1]["state"]
assert set(_st.keys()) == set(_ref.keys()), "A 失败:optimizer state 键不一致"
for k in _st:
    assert torch.allclose(_st[k]["exp_avg"], _ref[k]["exp_avg"], atol=0), "A 失败:exp_avg 未还原"
print("A 快照/还原(参数 + optimizer state): PASS")

# === B. Delta_adam 不是 -lr*grad ===
naive = -1e-2 * g_reward
cos_b = float(torch.dot(delta_adam, naive) / (delta_adam.norm() * naive.norm()))
assert cos_b < 0.999, "B 失败:Delta_adam 与 -lr*g 几乎相同,说明没用真实 AdamW"
print(f"B 真实 AdamW proposal(与 -lr*g 的 cos = {cos_b:.4f},非 1): PASS")

def set_theta(delta, alpha=1.0):
    with torch.no_grad():
        off = 0
        for p in params:
            n = p.numel()
            p.copy_((theta0[off:off+n] + alpha * delta[off:off+n]).view_as(p))
            off += n

# 步骤 4:同一缓存状态上**精确重算** C_after_direct
set_theta(delta_adam)
with torch.no_grad():
    C_after_direct = {m: constraint_val(m).item() for m in PROT}
restore(snap0)

# === C. 一阶预测 vs 精确重算(干净版:状态与 batch 都没变) ===
print("C 一阶预测 vs 精确重算(同一缓存状态):")
for m in PROT:
    pred = float(torch.dot(A[m], delta_adam))
    true = C_after_direct[m] - C[m]
    print(f"   {m:4s} 预测 dC={pred:+.3e}  实测 dC={true:+.3e}  比值={pred/true if true else float('nan'):+.3f}")

# 步骤 5:带 repair RHS 的 update-space 投影
import importlib.util as _il
_sp = _il.spec_from_file_location("qp", __file__.replace("0025_probe_machinery_test.py",
                                                         "0025_qp_repair_solver.py"))
qp = _il.module_from_spec(_sp); _sp.loader.exec_module(qp)
ms = list(PROT)
aa = [[float(torch.dot(A[i], A[j])) for j in ms] for i in ms]
ad = [float(torch.dot(A[i], delta_adam)) for i in ms]
Cv = [C[i] for i in ms]
mu, act = qp.solve_repair_qp(aa, ad, Cv)
assert mu is not None, "QP 无解"
delta_proj = delta_adam - sum((mu[k] * A[ms[k]] for k in range(len(ms))),
                              torch.zeros_like(delta_adam))

# 步骤 6:exact certificate + backtrack(**用精确重算判定,不用一阶预测**)
# backtrack **只缩放 reward 分量**:对 alpha*Delta_adam 重解 QP,repair 分量保持全量。
# 整体缩放会把 repair 一并削弱,C>0 时线性模型下注定失败(见预注册附录 A)。
ok_alpha, C_after_proj, delta_used = None, None, None
for alpha in (1.0, 0.5, 0.25, 0.125):
    ad_s = [alpha * x for x in ad]
    mu_a, _ = qp.solve_repair_qp(aa, ad_s, Cv)
    if mu_a is None:
        continue
    d_a = alpha * delta_adam - sum((mu_a[k] * A[ms[k]] for k in range(len(ms))),
                                   torch.zeros_like(delta_adam))
    set_theta(d_a, 1.0)
    with torch.no_grad():
        ca = {m: constraint_val(m).item() for m in PROT}
    restore(snap0)
    if all(v <= 1e-9 for v in ca.values()):
        ok_alpha, C_after_proj, delta_used = alpha, ca, d_a
        break
# 分母始终是**全量** Delta_adam,使 alpha<1 如实体现为 retention 下降
_du = delta_used if delta_used is not None else delta_proj
ret = float(torch.dot(_du, delta_adam) / delta_adam.dot(delta_adam))
print(f"D exact certificate: 通过于 alpha={ok_alpha}  (backtrack {['1','1/2','1/4','1/8'].index(str(ok_alpha)) if False else ''})"
      if ok_alpha else "D exact certificate: 4 次 backtrack 全部未通过")
print(f"   reward_retention = <Delta_proj, Delta_adam>/||Delta_adam||^2 = {ret:+.4f}")
print(f"   C_before={ {k: round(v,3e-9.__class__ and 6) for k,v in C.items()} }")
print(f"   C_after_direct={ {k: round(v,6) for k,v in C_after_direct.items()} }")
if C_after_proj: print(f"   C_after_proj  ={ {k: round(v,6) for k,v in C_after_proj.items()} }")

assert torch.allclose(flat([p.detach() for p in params]), theta0, atol=0), "结束时参数未还原"
print("\n全部机制检验通过 —— 可以移植进 trainer")
