"""带 repair RHS 的 update-space 投影(codex 0023 §3.4)。

与 0017 的区别 —— 0017 解的是"不再恶化":
    min ||d - g||^2   s.t.  <a_m, d> >= 0
本文件解的是 codex 指出的**真正需要的问题**:

    min_Delta  1/2 ||Delta - Delta_adam||^2
    s.t.       C_m + <a_m, Delta> <= 0     (每个受保护能力 m)

区别有二,都很关键:
1. **基准点是真实的 AdamW proposal `Delta_adam`,不是 -g**。
   实测 cos(实际更新, -累积梯度) = 0.339,在原始梯度空间投影与实际更新脱节。
2. **RHS 非零**。若当前状态已违反(C_m > 0),约束要求 `<a_m,Delta> <= -C_m < 0`,
   即更新必须**主动修复**,而不只是"不再恶化"。0017 拿不到这个解。

只依赖内积(Gram + <a_m, Delta_adam> + C_m),故可在高维上只传标量。
"""
import itertools


def solve_repair_qp(aa, ad, C, tol=1e-12):
    """aa[i][j]=<a_i,a_j>;ad[i]=<a_i, Delta_adam>;C[i]=当前约束值(>0 表示已违反)。

    返回 (mu, active):Delta = Delta_adam - sum_i mu_i a_i,mu_i >= 0。
    显式枚举 active set(m 个约束共 2^m 个子集),取可行且目标最小者。
    """
    m = len(C)
    best = None
    for r in range(m + 1):
        for act in itertools.combinations(range(m), r):
            mu = [0.0] * m
            if act:
                # 对 active 集解 G mu = b,其中 b_i = <a_i,Delta_adam> + C_i
                G = [[aa[i][j] for j in act] for i in act]
                b = [ad[i] + C[i] for i in act]
                sol = _solve(G, b)
                if sol is None:
                    continue
                for k, i in enumerate(act):
                    mu[i] = sol[k]
            if any(x < -1e-9 for x in mu):
                continue
            # 可行性:所有约束都要满足 C_i + <a_i,Delta> <= 0
            ok = True
            for i in range(m):
                adi = ad[i] - sum(mu[j] * aa[i][j] for j in range(m))
                if C[i] + adi > 1e-9:
                    ok = False
                    break
            if not ok:
                continue
            obj = 0.5 * sum(mu[i] * mu[j] * aa[i][j] for i in range(m) for j in range(m))
            if best is None or obj < best[1] - 1e-15:
                best = (mu, obj, act)
    return (best[0], best[2]) if best else (None, None)


def _solve(G, b):
    """小规模稠密线性方程组(高斯消元,带主元)。奇异返回 None。"""
    n = len(b)
    A = [row[:] + [b[i]] for i, row in enumerate(G)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-30:
            return None
        A[c], A[piv] = A[piv], A[c]
        for r in range(n):
            if r == c:
                continue
            f = A[r][c] / A[c][c]
            for k in range(c, n + 1):
                A[r][k] -= f * A[c][k]
    return [A[i][n] / A[i][i] for i in range(n)]


def reward_retention(gg_ad, ad, mu, aa):
    """保留了多少 reward 更新:<Delta, Delta_adam> / ||Delta_adam||^2。

    gg_ad = ||Delta_adam||^2。1.0 = 完全保留;0 = 完全抵消。
    """
    if gg_ad <= 0:
        return float("nan")
    m = len(mu)
    return (gg_ad - sum(mu[i] * ad[i] for i in range(m))) / gg_ad


if __name__ == "__main__":
    import random, math
    random.seed(7)

    def dot(x, y): return sum(a * b for a, b in zip(x, y))

    npass = 0
    n_repair = 0
    for trial in range(300):
        d = 4
        M = random.choice([1, 2, 3])
        Da = [random.gauss(0, 1) for _ in range(d)]
        A = [[random.gauss(0, 1) for _ in range(d)] for _ in range(M)]
        # 一半的算例让约束已经被违反(C>0),以覆盖 repair 分支
        C = [random.gauss(0, 0.5) if trial % 2 else -abs(random.gauss(0, 0.5)) for _ in range(M)]
        aa = [[dot(A[i], A[j]) for j in range(M)] for i in range(M)]
        ad = [dot(A[i], Da) for i in range(M)]
        mu, act = solve_repair_qp(aa, ad, C)
        assert mu is not None, f"trial {trial}: 无解(约束线性无关时应恒有解)"
        D = [Da[k] - sum(mu[i] * A[i][k] for i in range(M)) for k in range(d)]
        # 1) 可行:每个约束都满足
        for i in range(M):
            assert C[i] + dot(A[i], D) <= 1e-7, f"trial {trial}: 约束 {i} 不满足"
        # 2) 乘子非负
        assert all(x >= -1e-9 for x in mu)
        # 3) Delta_adam 本身可行时,不应改动它
        if all(C[i] + ad[i] <= -1e-9 for i in range(M)):
            assert max(abs(D[k] - Da[k]) for k in range(d)) < 1e-8, f"trial {trial}: 可行时不应投影"
        # 4) 目标不劣于随机可行点
        obj = sum((D[k] - Da[k]) ** 2 for k in range(d))
        for _ in range(400):
            R = [Da[k] + random.gauss(0, 1) for k in range(d)]
            if all(C[i] + dot(A[i], R) <= 0 for i in range(M)):
                assert obj <= sum((R[k] - Da[k]) ** 2 for k in range(d)) + 1e-7, f"trial {trial}: 非最优"
        # 5) repair 分支确实被覆盖:C>0 时投影必须使其变为 <=0
        if any(c > 0 for c in C):
            n_repair += 1
        npass += 1
    print(f"repair-QP 合成对拍: {npass}/300 通过(其中 {n_repair} 个算例含已违反约束,走 repair 分支)")
    print("  检验:可行性、KKT 乘子非负、Delta_adam 可行时恒等、目标不劣于随机可行点、repair 分支覆盖")
