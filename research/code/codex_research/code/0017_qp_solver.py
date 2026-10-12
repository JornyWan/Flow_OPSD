"""二维可行更新投影的 QP 求解器 + 合成对拍(codex 0014 §4.5 第 4 条)。

问题:参数更新为 theta <- theta - eta*d。约束 C_m 的变化
    dC_m ≈ <a_m, -eta*d> = -eta * <a_m, d>
要求 dC_m <= 0,即 **<a_m, d> >= 0**。

于是解
    min_d  0.5||d - g||^2   s.t.  <a_m, d> >= 0, m in active set
KKT:  d = g + sum_m mu_m a_m,  mu_m >= 0,  active 时 <a_m, d> = 0。

本文件只依赖 g 与 a_m 的**内积**(Gram),不需要高维向量本身,
故可在训练中用少量标量完成,再按 d = g + sum mu_m a_m 写回梯度。
"""


def solve_qp2(gg, ga, aa):
    """gg=<g,g>; ga=[<a_0,g>, <a_1,g>]; aa=[[<a_0,a_0>,<a_0,a_1>],[...]]

    返回 (mu, active):mu 为长度 2 的乘子(未激活为 0.0)。
    用显式 active-set 枚举(2 个约束共 4 种组合),取**可行且目标最小**的解。
    """
    import itertools
    best = None
    for act in itertools.chain([()], [(0,), (1,)], [(0, 1)]):
        mu = [0.0, 0.0]
        if len(act) == 1:
            i = act[0]
            if aa[i][i] <= 1e-30:
                continue
            mu[i] = -ga[i] / aa[i][i]
        elif len(act) == 2:
            a, b, c = aa[0][0], aa[0][1], aa[1][1]
            det = a * c - b * b
            if abs(det) <= 1e-30:
                continue
            mu[0] = (-ga[0] * c + ga[1] * b) / det
            mu[1] = (-ga[1] * a + ga[0] * b) / det
        # 可行性:mu >= 0 且所有约束 <a_m, d> >= 0
        if any(m < -1e-12 for m in mu):
            continue
        ok = True
        for m in (0, 1):
            adm = ga[m] + mu[0] * aa[m][0] + mu[1] * aa[m][1]
            if adm < -1e-9:
                ok = False
                break
        if not ok:
            continue
        # 目标 0.5||d-g||^2 = 0.5 * ||sum mu_m a_m||^2
        obj = 0.5 * (mu[0] * mu[0] * aa[0][0] + 2 * mu[0] * mu[1] * aa[0][1] + mu[1] * mu[1] * aa[1][1])
        if best is None or obj < best[1] - 1e-15:
            best = (mu, obj, act)
    if best is None:
        return [0.0, 0.0], ()
    return best[0], best[2]


def progress_ratio(gg, ga, aa, mu):
    """<g,d>/||g||^2,d = g + sum mu_m a_m。1.0 表示 reward 进展未受损。"""
    if gg <= 0:
        return float("nan")
    return (gg + mu[0] * ga[0] + mu[1] * ga[1]) / gg


if __name__ == "__main__":
    import random, itertools
    random.seed(0)

    def brute(g, A):
        """低维显式对拍:在 R^3 上用细网格近似求解,验证解析解。"""
        import math
        best = None
        for _ in range(200000):
            d = [g[i] + random.gauss(0, 1) * 0.5 for i in range(3)]
            if all(sum(a[i] * d[i] for i in range(3)) >= -1e-9 for a in A):
                obj = sum((d[i] - g[i]) ** 2 for i in range(3))
                if best is None or obj < best[1]:
                    best = (d, obj)
        return best

    def dot(x, y): return sum(a * b for a, b in zip(x, y))

    npass = 0
    for trial in range(200):
        g = [random.gauss(0, 1) for _ in range(3)]
        A = [[random.gauss(0, 1) for _ in range(3)] for _ in range(2)]
        gg = dot(g, g)
        ga = [dot(a, g) for a in A]
        aa = [[dot(A[i], A[j]) for j in range(2)] for i in range(2)]
        mu, act = solve_qp2(gg, ga, aa)
        d = [g[i] + mu[0] * A[0][i] + mu[1] * A[1][i] for i in range(3)]
        # 1) 可行性
        assert all(dot(a, d) >= -1e-7 for a in A), f"trial {trial}: 解不可行"
        # 2) 乘子非负
        assert all(m >= -1e-9 for m in mu), f"trial {trial}: mu 为负"
        # 3) 若 g 本身可行,必须 d == g
        if all(x >= -1e-9 for x in ga):
            assert max(abs(d[i] - g[i]) for i in range(3)) < 1e-9, f"trial {trial}: g 可行时不应投影"
        # 4) 目标不劣于随机可行点
        obj = sum((d[i] - g[i]) ** 2 for i in range(3))
        for _ in range(500):
            dd = [g[i] + random.gauss(0, 1) * 0.5 for i in range(3)]
            if all(dot(a, dd) >= 0 for a in A):
                assert obj <= sum((dd[i] - g[i]) ** 2 for i in range(3)) + 1e-7, f"trial {trial}: 非最优"
        npass += 1
    print(f"QP 求解器合成对拍: {npass}/200 通过")
    print("  (检验:可行性、KKT 乘子非负、g 可行时恒等、目标不劣于随机可行点)")
