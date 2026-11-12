"""統計工具：信賴區間、Holm、置換檢定、CLQ、Fisher、Mantel–Haenszel。

每一項都用「答案已知」的資料驗證，而不是只檢查不報錯。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dadaocheng.stats import (clq, cluster_bootstrap_ci, fisher_lower, holm, mantel_haenszel,
                              permutation_share, proportion_with_ci, tables_by_stratum, wilson_ci)


def test_wilson_known_values(tmp_root):
    lo, hi = wilson_ci(50, 100)
    assert lo == pytest.approx(0.404, abs=0.01) and hi == pytest.approx(0.596, abs=0.01)
    assert wilson_ci(0, 10)[0] == 0.0
    assert wilson_ci(10, 10)[1] == pytest.approx(1.0)
    assert all(np.isnan(v) for v in wilson_ci(0, 0))


def test_cluster_bootstrap_wider_when_clustered(tmp_root):
    """同一群集內完全相關時，群集自助法的區間必須比 Wilson 寬。"""
    # 20 個街區，每區 20 家店；每區內部完全一致（整條街一起早關或一起晚關）
    vals, cl = [], []
    for b in range(20):
        v = 1.0 if b < 12 else 0.0
        vals += [v] * 20
        cl += [f"B{b:02d}"] * 20
    lo_b, hi_b, ncl = cluster_bootstrap_ci(vals, cl)
    lo_w, hi_w = wilson_ci(int(sum(vals)), len(vals))
    assert ncl == 20
    assert (hi_b - lo_b) > 3 * (hi_w - lo_w)
    assert lo_b < 0.6 < hi_b


def test_cluster_bootstrap_matches_wilson_when_independent(tmp_root):
    """每個群集只有一家店時，群集自助法 ≈ 個體自助法，寬度應與 Wilson 接近。"""
    rng = np.random.default_rng(0)
    v = (rng.random(400) < 0.6).astype(float)
    lo_b, hi_b, _ = cluster_bootstrap_ci(v, [f"B{i}" for i in range(400)])
    lo_w, hi_w = wilson_ci(int(v.sum()), len(v))
    assert abs((hi_b - lo_b) - (hi_w - lo_w)) < 0.03


def test_proportion_ignores_missing(tmp_root):
    pr = proportion_with_ci([True, False, None, True, None], ["A", "A", "B", "B", "C"], "測試", n_boot=200)
    assert pr.n == 3 and pr.k == 2
    assert pr.row()["組別"] == "測試"


def test_holm_family_size_is_conservative(tmp_root):
    """只完成 3 條時仍以 m=4 校正；完成的排在前面＝最保守。"""
    h = holm({"H1": 0.001, "H3": 0.02, "H5": 0.04}).set_index("假說")
    assert h.loc["H1", "Holm 乘數"] == 4 and h.loc["H1", "校正後 p"] == pytest.approx(0.004)
    assert h.loc["H3", "Holm 乘數"] == 3 and h.loc["H3", "校正後 p"] == pytest.approx(0.06)
    assert h.loc["H5", "Holm 乘數"] == 2
    # 單調性：校正後 p 不遞減
    assert h.loc["H5", "校正後 p"] >= h.loc["H3", "校正後 p"]
    assert not h.loc["H3", "α=0.05 下顯著"]


def test_holm_marks_missing(tmp_root):
    h = holm({"H1": 0.01, "H3": 0.02, "H5": 0.03, "H6": float("nan")}).set_index("假說")
    assert np.isnan(h.loc["H6", "校正後 p"]) and not h.loc["H6", "α=0.05 下顯著"]


def test_permutation_detects_and_rejects(tmp_root):
    n = 400
    region = np.zeros(n, dtype=bool); region[:100] = True      # 四分之一的 POI 在該區位
    # 集中：目標業種 40 家全在區位內
    tgt = np.zeros(n, dtype=bool); tgt[:40] = True
    r = permutation_share(region, tgt, n_perm=499, seed_offset=1)
    assert r["observed"] == 1.0 and r["baseline"] == pytest.approx(0.25) and r["p"] <= 0.01
    # 不集中：目標業種隨機分布
    rng = np.random.default_rng(3)
    tgt2 = np.zeros(n, dtype=bool); tgt2[rng.choice(n, 40, replace=False)] = True
    r2 = permutation_share(region, tgt2, n_perm=499, seed_offset=2)
    assert r2["p"] > 0.05
    assert permutation_share(region, np.zeros(n, dtype=bool))["underpowered"]


def test_clq_detects_clustering(tmp_root):
    rng = np.random.default_rng(0)
    bg = rng.uniform(0, 1000, size=(300, 2))
    clump = rng.normal(500, 15, size=(60, 2))
    xy = np.vstack([bg, clump])
    is_a = np.zeros(len(xy), dtype=bool); is_a[300:] = True
    r = clq(xy, is_a, k=10, n_perm=299, seed_offset=4)
    assert r["clq"] > 2 and r["p"] <= 0.01
    # 隨機標籤 → CLQ ≈ 1
    lab = np.zeros(len(xy), dtype=bool); lab[rng.choice(len(xy), 60, replace=False)] = True
    r2 = clq(xy, lab, k=10, n_perm=299, seed_offset=5)
    assert 0.6 < r2["clq"] < 1.6 and r2["p"] > 0.05


def test_fisher_direction_and_power_flag(tmp_root):
    # 歷建內 2/40 熱食，歷建外 200/400 熱食 → 單尾顯著
    r = fisher_lower(2, 38, 200, 200)
    assert r["p_in"] < r["p_out"] and r["p"] < 0.001 and r["odds_ratio"] < 1
    assert not r["underpowered"]
    # 歷建內只有 10 家 → 回報檢定力不足
    assert fisher_lower(1, 9, 200, 200)["underpowered"]
    # 反方向不應該顯著
    assert fisher_lower(30, 10, 100, 300)["p"] > 0.9


def test_mantel_haenszel(tmp_root):
    df = pd.DataFrame({
        "block": sum([[f"B{i}"] * 100 for i in range(4)], []),
        "in_heritage": sum([[True] * 20 + [False] * 80 for _ in range(4)], []),
        "hot": sum([[True] * 2 + [False] * 18 + [True] * 40 + [False] * 40 for _ in range(4)], []),
    })
    r = mantel_haenszel(tables_by_stratum(df, "block", "in_heritage", "hot"))
    assert r["n_strata"] == 4 and r["or_mh"] < 1 and r["p"] < 0.001
    assert mantel_haenszel([np.array([[1, 0], [0, 0]])])["underpowered"]
    assert not r.get("unstable")
    # 歷建內完全沒有熱食 → 共同勝算比為 0，應標為不穩定而不是照報
    zero = [np.array([[0, 20], [40, 40]]) for _ in range(3)]
    z = mantel_haenszel(zero)
    assert z["unstable"] and z["n_strata"] == 3


# ---------------------------------------------------------------------------
# 野自助法與 leave-one-cluster-out
# ---------------------------------------------------------------------------

def _clustered_design(n_per=30, G=17, effect=0.0, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    cl, x, y = [], [], []
    for g in range(G):
        u = rng.normal(0, 1.0)                       # 街區層級的隨機效果
        xi = rng.normal(0, 1, n_per)
        yi = effect * xi + u + rng.normal(0, 1, n_per)
        cl += [f"B{g:02d}"] * n_per
        x.append(xi)
        y.append(yi)
    x = np.concatenate(x)
    X = np.column_stack([np.ones_like(x), x])
    return np.concatenate(y), X, cl


def test_webb_wild_bootstrap_null_is_not_anticonservative():
    """虛無為真時，p 值不應該系統性偏小（這是 G 小的時候群集穩健 SE 的老問題）。"""
    import numpy as np

    from dadaocheng import stats

    ps = []
    for s in range(30):
        y, X, cl = _clustered_design(effect=0.0, seed=s)
        r = stats.webb_wild_bootstrap_p(y, X, cl, coef_index=1, n_boot=299, seed_offset=s)
        assert r["refused"] is None and r["G"] == 17
        ps.append(r["p"])
    ps = np.array(ps)
    assert ((ps >= 0) & (ps <= 1)).all()
    assert (ps < 0.05).mean() <= 0.20          # 30 次只允許少數假陽性
    assert np.median(ps) > 0.20


def test_webb_wild_bootstrap_detects_real_effect():
    from dadaocheng import stats

    y, X, cl = _clustered_design(effect=0.9, seed=7)
    r = stats.webb_wild_bootstrap_p(y, X, cl, coef_index=1, n_boot=599)
    assert r["p"] < 0.01 and r["beta"] > 0.5


def test_webb_wild_bootstrap_refuses_too_few_clusters():
    from dadaocheng import stats

    y, X, cl = _clustered_design(n_per=20, G=3, effect=0.5)
    r = stats.webb_wild_bootstrap_p(y, X, cl)
    assert r["refused"] and "3 個群集" in r["refused"]


def test_leave_one_cluster_out_flags_dominant_cluster():
    """效果只存在於一個很大的街區時，拿掉它估計值必須明顯掉下來。"""
    import numpy as np
    import pandas as pd

    from dadaocheng import stats

    rng = np.random.default_rng(1)
    big = pd.DataFrame({"block": "BIG", "x": rng.normal(0, 1, 400)})
    big["y"] = 2.0 * big["x"] + rng.normal(0, 0.5, 400)
    rest = []
    for g in range(8):
        d = pd.DataFrame({"block": f"B{g}", "x": rng.normal(0, 1, 25)})
        d["y"] = rng.normal(0, 0.5, 25)              # 其他街區沒有效果
        rest.append(d)
    df = pd.concat([big, *rest], ignore_index=True)

    def fit(sub):
        X = np.column_stack([np.ones(len(sub)), sub["x"].to_numpy()])
        return float(np.linalg.lstsq(X, sub["y"].to_numpy(), rcond=None)[0][1])

    out = stats.leave_one_cluster_out(fit, df, "block")
    assert out.loc[0, "拿掉的街區"] == "（全部保留）"
    assert len(out) == 10
    row = out[out["拿掉的街區"] == "BIG"].iloc[0]
    assert row["該街區筆數"] == 400
    assert abs(row["估計值"]) < 0.2
    assert row["變動比例"] < -0.8
    # 拿掉小街區幾乎不影響
    small = out[out["拿掉的街區"].str.startswith("B") & out["拿掉的街區"].ne("BIG")]
    assert small["變動比例"].abs().max() < 0.15


def test_wild_cluster_wald_single_restriction_matches_t_test_logic():
    """一條限制時，W 應等於 t² 的群集穩健版本；虛無為真時 p 不偏小。"""
    import numpy as np

    from dadaocheng import stats

    ps = []
    for s in range(20):
        y, X, cl = _clustered_design(effect=0.0, seed=100 + s)
        r = stats.wild_cluster_wald(y, X, cl, R=np.array([[0.0, 1.0]]), n_boot=199, seed_offset=s)
        ols = stats.cluster_ols(y, X, cl)
        t = ols.beta[1] / ols.se()[1]
        # CR1 校正因子會讓 t² 與未校正的 W 差一個常數倍
        assert r["W"] > 0 and t ** 2 > 0
        ps.append(r["p"])
    ps = np.array(ps)
    assert (ps < 0.05).mean() <= 0.25 and np.median(ps) > 0.15


def test_wild_cluster_wald_joint_equality_detects_heterogeneity():
    """兩組斜率不同時，聯合（斜率相等）檢定要拒絕；相同時不拒絕。"""
    import numpy as np

    from dadaocheng import stats

    rng = np.random.default_rng(3)
    G, m = 17, 40
    cl = np.repeat(np.arange(G), m)
    grp = rng.integers(0, 3, G * m)
    x = rng.normal(0, 1, G * m)
    u = rng.normal(0, 0.5, G)[cl]
    for diff, expect_reject in [(0.8, True), (0.0, False)]:
        slope = np.array([0.2, 0.2 + diff, 0.2])[grp]
        y = slope * x + u + rng.normal(0, 1, G * m)
        X = np.column_stack([np.ones_like(x), (grp == 1), (grp == 2),
                             x * (grp == 0), x * (grp == 1), x * (grp == 2)]).astype(float)
        R = np.array([[0, 0, 0, -1, 1, 0], [0, 0, 0, -1, 0, 1]], float)
        r = stats.wild_cluster_wald(y, X, cl, R, n_boot=499)
        assert r["q"] == 2 and r["G"] == G
        assert (r["p"] < 0.05) == expect_reject, (diff, r)


def test_within_demean_equals_fixed_effects():
    import numpy as np
    import pandas as pd

    from dadaocheng import stats

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"g": np.repeat(list("abcde"), 30), "x": rng.normal(0, 1, 150)})
    df["y"] = 1.5 * df["x"] + df["g"].map({"a": 0, "b": 5, "c": -3, "d": 2, "e": 9}) + rng.normal(0, 0.1, 150)
    dm = stats.within_demean(df, ["y", "x"], "g")
    b = np.linalg.lstsq(dm[["x"]].to_numpy(), dm["y"].to_numpy(), rcond=None)[0][0]
    D = pd.get_dummies(df["g"]).to_numpy(float)
    bfe = np.linalg.lstsq(np.column_stack([df["x"], D]), df["y"], rcond=None)[0][0]
    assert abs(b - bfe) < 1e-8 and abs(b - 1.5) < 0.05
