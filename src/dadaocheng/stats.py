"""假說檢定共用的統計工具（T11–T15）。

全部依 docs/preregistration.md 的規格實作。幾個刻意的選擇：

- **比例的信賴區間同時給兩個版本。** Wilson 區間假設每家店獨立；但同一街區的店家
  打烊時間明顯相關（里長說「整條街七點關」），所以主要版本是「以街區為單位的群集自助法」。
  兩者並列，讀者自己看差多少。
- **Holm 校正以預先登錄的家族大小 m=4 計算**（H1、H3、H5、H6）。H6 還沒跑完時，
  把已完成的檢定排在前面、假設 H6 的 p 值最大——這是最保守的做法，不會因為少一條而放寬。
- **置換檢定固定位置、只洗標籤。** 這樣虛無分布自動吃到 POI 本身的空間不均勻。
- 檢定力不足時回報 `underpowered`，不換檢定（CLAUDE.md）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# 比例與信賴區間
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score 區間（假設觀測獨立）。"""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence, alpha: float = 0.05,
                         n_boot: int = 2000, seed_offset: int = 0,
                         stat: Callable[[np.ndarray], float] | None = None) -> tuple[float, float, int]:
    """以群集（街區）為單位重抽的百分位自助法區間。回傳 (下界, 上界, 有效群集數)。

    重抽群集而不是個體，才會把「同街區的店家彼此相關」算進去。
    """
    v = np.asarray(values, dtype=float)
    cl = np.asarray(clusters, dtype=object)
    ok = ~pd.isna(v) & ~pd.isna(cl)
    v, cl = v[ok], cl[ok]
    if len(v) == 0:
        return (float("nan"), float("nan"), 0)
    stat = stat or (lambda a: float(np.mean(a)))
    groups = {}
    for g, x in zip(cl, v):
        groups.setdefault(g, []).append(x)
    keys = sorted(groups, key=str)
    arrs = [np.asarray(groups[k], dtype=float) for k in keys]
    rng = config.rng(seed_offset)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(arrs), size=len(arrs))
        out[b] = stat(np.concatenate([arrs[i] for i in pick]))
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi), len(arrs))


def binom_test_greater(k: int, n: int, p0: float = 0.5) -> float:
    """單尾精確二項檢定，H₀：p ≤ p0。"""
    if n == 0:
        return float("nan")
    from scipy.stats import binomtest

    return float(binomtest(k, n, p0, alternative="greater").pvalue)


@dataclass
class Proportion:
    label: str
    k: int
    n: int
    wilson: tuple[float, float] = (float("nan"), float("nan"))
    boot: tuple[float, float] = (float("nan"), float("nan"))
    n_clusters: int = 0

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def row(self) -> dict:
        return {"組別": self.label, "n": self.n, "符合": self.k, "比例": self.p,
                "Wilson 下界": self.wilson[0], "Wilson 上界": self.wilson[1],
                "街區自助法 下界": self.boot[0], "街區自助法 上界": self.boot[1],
                "街區數": self.n_clusters}


def proportion_with_ci(flags: Sequence, clusters: Sequence, label: str, alpha: float = 0.05,
                       n_boot: int = 2000, seed_offset: int = 0) -> Proportion:
    f = pd.Series(list(flags))
    c = pd.Series(list(clusters))
    ok = f.notna()
    f, c = f[ok].astype(bool), c[ok]
    k, n = int(f.sum()), int(len(f))
    pr = Proportion(label, k, n, wilson_ci(k, n, alpha))
    lo, hi, ncl = cluster_bootstrap_ci(f.astype(float).values, c.values, alpha, n_boot, seed_offset)
    pr.boot, pr.n_clusters = (lo, hi), ncl
    return pr


# ---------------------------------------------------------------------------
# 多重比較
# ---------------------------------------------------------------------------

PREREG_FAMILY = ["H1", "H3", "H5", "H6"]


def holm(pvals: dict[str, float], family_size: int | None = None) -> pd.DataFrame:
    """Holm 校正。family_size 預設為預先登錄的家族大小（4）。

    只有部分檢定完成時，把完成的排在最前面（等於假設未完成者的 p 值最大），
    這會給出最大的乘數，也就是最保守的校正。
    """
    m = int(family_size or len(PREREG_FAMILY))
    items = [(k, v) for k, v in pvals.items() if v is not None and not (isinstance(v, float) and math.isnan(v))]
    items.sort(key=lambda kv: kv[1])
    rows, running = [], 0.0
    for rank, (k, p) in enumerate(items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running = max(running, adj)  # Holm 的單調性
        rows.append({"假說": k, "原始 p": p, "Holm 乘數": m - rank + 1, "校正後 p": running,
                     "α=0.05 下顯著": running < 0.05})
    missing = [k for k in pvals if k not in dict(items)]
    for k in missing:
        rows.append({"假說": k, "原始 p": float("nan"), "Holm 乘數": float("nan"),
                     "校正後 p": float("nan"), "α=0.05 下顯著": False})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 置換檢定（H3）
# ---------------------------------------------------------------------------

def permutation_share(in_region: Sequence[bool], is_target: Sequence[bool], n_perm: int = 999,
                      seed_offset: int = 0, alternative: str = "greater") -> dict:
    """固定位置、洗牌業種標籤，檢定「目標業種落在某區位的比例」是否高於整體。

    統計量＝目標業種中落在該區位的比例。虛無分布＝把同樣數量的標籤隨機灑在所有 POI 上。
    單尾 p = (1 + #{置換 ≥ 觀測}) / (1 + n_perm)。
    """
    reg = np.asarray(list(in_region), dtype=bool)
    tgt = np.asarray(list(is_target), dtype=bool)
    n, n_t = len(reg), int(tgt.sum())
    obs_k = int((reg & tgt).sum())
    obs = obs_k / n_t if n_t else float("nan")
    base = float(reg.mean()) if n else float("nan")
    if n_t == 0 or n == 0:
        return {"observed": float("nan"), "baseline": base, "p": float("nan"), "n_target": 0,
                "k_in_region": 0, "n_perm": n_perm, "underpowered": True}
    rng = config.rng(seed_offset)
    idx = np.arange(n)
    null = np.empty(n_perm)
    for b in range(n_perm):
        pick = rng.choice(idx, size=n_t, replace=False)
        null[b] = reg[pick].mean()
    if alternative == "greater":
        p = (1 + int((null >= obs - 1e-12).sum())) / (1 + n_perm)
    else:
        p = (1 + int((null <= obs + 1e-12).sum())) / (1 + n_perm)
    return {"observed": obs, "baseline": base, "p": float(p), "n_target": n_t, "k_in_region": obs_k,
            "n_perm": n_perm, "null_mean": float(null.mean()), "underpowered": n_t < 10}


def clq(points_xy: np.ndarray, is_a: Sequence[bool], k: int = 10, n_perm: int = 999,
        seed_offset: int = 0) -> dict:
    """共同區位商數 CLQ(A→A)（Leslie & Kronenfeld 2011）與置換檢定。

    CLQ = 「A 類點的 k 個最近鄰中屬於 A 的平均比例」÷「A 在全體中的比例（扣掉自己）」。
    > 1 代表 A 類比隨機更靠在一起。
    """
    xy = np.asarray(points_xy, dtype=float)
    a = np.asarray(list(is_a), dtype=bool)
    n, n_a = len(xy), int(a.sum())
    if n_a < 3 or n - 1 < k:
        return {"clq": float("nan"), "p": float("nan"), "n_a": n_a, "k": k, "underpowered": True}
    from scipy.spatial import cKDTree

    tree = cKDTree(xy)
    _, nn = tree.query(xy, k=min(k + 1, n))      # 第 0 個是自己
    nn = nn[:, 1:]

    def stat(labels: np.ndarray) -> float:
        share = labels[nn[labels]].mean()
        denom = (labels.sum() - 1) / (n - 1)
        return float(share / denom) if denom > 0 else float("nan")

    obs = stat(a)
    rng = config.rng(seed_offset)
    null = np.empty(n_perm)
    for b in range(n_perm):
        lab = np.zeros(n, dtype=bool)
        lab[rng.choice(n, size=n_a, replace=False)] = True
        null[b] = stat(lab)
    p = (1 + int((null >= obs - 1e-12).sum())) / (1 + n_perm)
    return {"clq": obs, "p": float(p), "n_a": n_a, "k": k, "null_mean": float(np.nanmean(null)),
            "underpowered": n_a < 10}


# ---------------------------------------------------------------------------
# 列聯表（H5）
# ---------------------------------------------------------------------------

def fisher_lower(a: int, b: int, c: int, d: int) -> dict:
    """2×2 表的單尾 Fisher 精確檢定，檢定「第一列的比例較低」。

    表的排法：
                 熱食    非熱食
      歷建內        a        b
      歷建外        c        d
    H₁：a/(a+b) < c/(c+d)（歷建內熱食比例較低）。
    """
    from scipy.stats import fisher_exact

    res = fisher_exact([[a, b], [c, d]], alternative="less")
    odds = getattr(res, "statistic", res[0])
    p = getattr(res, "pvalue", res[1])
    return {"a": a, "b": b, "c": c, "d": d,
            "p_in": a / (a + b) if a + b else float("nan"),
            "p_out": c / (c + d) if c + d else float("nan"),
            "odds_ratio": float(odds), "p": float(p),
            "n_in": a + b, "underpowered": (a + b) < 20}


def mantel_haenszel(tables: list[np.ndarray]) -> dict:
    """以街區分層的 Mantel–Haenszel 共同勝算比與檢定。"""
    from statsmodels.stats.contingency_tables import StratifiedTable

    usable = [t for t in tables if np.asarray(t).sum() > 0 and np.asarray(t).shape == (2, 2)
              and np.asarray(t).sum(axis=1).min() > 0 and np.asarray(t).sum(axis=0).min() > 0]
    if len(usable) < 2:
        return {"or_mh": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p": float("nan"), "n_strata": len(usable), "underpowered": True, "unstable": True}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 空格會讓 log(OR) 變 ±inf，下面自己判斷
        st = StratifiedTable([np.asarray(t, dtype=float) for t in usable])
        test = st.test_null_odds()
        try:
            ci = st.oddsratio_pooled_confint()
        except Exception:  # noqa: BLE001
            ci = (float("nan"), float("nan"))
        or_mh = float(st.oddsratio_pooled)
        p = float(test.pvalue)
        stat = float(test.statistic)
    # 某一格為 0 時共同勝算比會變成 0 或無限大，區間也不可信：標為不穩定而不是照報數字
    unstable = (not np.isfinite(or_mh)) or or_mh <= 0 or not np.isfinite(ci[0]) or not np.isfinite(ci[1])
    return {"or_mh": or_mh, "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": p if np.isfinite(p) else float("nan"), "statistic": stat,
            "n_strata": len(usable), "underpowered": len(usable) < 3, "unstable": bool(unstable)}


def tables_by_stratum(df: pd.DataFrame, stratum: str, row: str, col: str) -> list[np.ndarray]:
    """每個分層產生一張 2×2 表（row/col 皆為布林欄）。"""
    out = []
    for _, g in df.groupby(stratum):
        r, c = g[row].astype(bool), g[col].astype(bool)
        out.append(np.array([[int((r & c).sum()), int((r & ~c).sum())],
                             [int((~r & c).sum()), int((~r & ~c).sum())]]))
    return out


# ---------------------------------------------------------------------------
# 報表輔助
# ---------------------------------------------------------------------------

def fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1%}"


def fmt_p(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return "< 0.001" if x < 0.001 else f"{x:.3f}"


@dataclass
class HypothesisResult:
    """一條假說的檢定結果，統一輸出成表與結論句。"""
    hid: str
    claim: str
    p_for_holm: float = float("nan")
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    verdict: str = ""

    def save(self) -> None:
        d = config.path("tables")
        d.mkdir(parents=True, exist_ok=True)
        for name, t in self.tables.items():
            t.to_csv(d / f"{self.hid.lower()}_{name}.csv", index=False, encoding="utf-8-sig")

    def summary(self) -> str:
        lines = [f"【{self.hid}】{self.claim}", f"結論：{self.verdict}"]
        lines += [f"  · {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 一致度（T10 閘門一）
# ---------------------------------------------------------------------------

def cohen_kappa(a: Sequence, b: Sequence, labels: Sequence | None = None) -> dict:
    """Cohen's κ。缺值（None/NaN）成對剔除。

    「不明」視為一個正常類別：人與 LLM 都說不明時算一致，這才反映真實的可用性。
    """
    sa, sb = pd.Series(list(a)).reset_index(drop=True), pd.Series(list(b)).reset_index(drop=True)
    ok = sa.notna() & sb.notna()
    sa, sb = sa[ok].astype(str), sb[ok].astype(str)
    n = len(sa)
    if n == 0:
        return {"kappa": float("nan"), "accuracy": float("nan"), "n": 0, "labels": []}
    labs = list(labels) if labels is not None else sorted(set(sa) | set(sb))
    idx = {l: i for i, l in enumerate(labs)}
    m = np.zeros((len(labs), len(labs)), dtype=float)
    for x, y in zip(sa, sb):
        if x in idx and y in idx:
            m[idx[x], idx[y]] += 1
    tot = m.sum()
    if tot == 0:
        return {"kappa": float("nan"), "accuracy": float("nan"), "n": 0, "labels": labs}
    po = float(np.trace(m) / tot)
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum() / (tot * tot))
    kappa = 1.0 if abs(1 - pe) < 1e-12 else (po - pe) / (1 - pe)
    return {"kappa": float(kappa), "accuracy": po, "n": int(tot), "labels": labs,
            "matrix": m.astype(int), "p_expected": pe}


def kappa_bootstrap_ci(a: Sequence, b: Sequence, labels: Sequence | None = None, alpha: float = 0.05,
                       n_boot: int = 2000, seed_offset: int = 10) -> tuple[float, float]:
    """κ 的自助法百分位區間（以抽查筆數為重抽單位）。"""
    sa, sb = pd.Series(list(a)).reset_index(drop=True), pd.Series(list(b)).reset_index(drop=True)
    ok = sa.notna() & sb.notna()
    sa, sb = sa[ok].to_numpy(), sb[ok].to_numpy()
    n = len(sa)
    if n < 5:
        return (float("nan"), float("nan"))
    rng = config.rng(seed_offset)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        k = cohen_kappa(sa[pick], sb[pick], labels)["kappa"]
        if not (isinstance(k, float) and math.isnan(k)):
            vals.append(k)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def weighted_accuracy(correct: Sequence[bool], sampling_prob: Sequence[float]) -> float:
    """依抽樣機率加權的準確率（分層抽樣時，每筆代表 1/p 個母體單位）。"""
    c = pd.Series(list(correct)).astype(float)
    p = pd.Series(list(sampling_prob)).astype(float)
    ok = c.notna() & p.notna() & (p > 0)
    if not ok.any():
        return float("nan")
    w = 1.0 / p[ok]
    return float((c[ok] * w).sum() / w.sum())


def confusion_frame(res: dict) -> "pd.DataFrame":
    """把 cohen_kappa 的矩陣轉成可讀的表（列＝人工，欄＝LLM）。"""
    if "matrix" not in res:
        return pd.DataFrame()
    return pd.DataFrame(res["matrix"], index=[f"人工：{l}" for l in res["labels"]],
                        columns=[f"LLM：{l}" for l in res["labels"]]).reset_index(names="")


# ---------------------------------------------------------------------------
# 群集數很少、而且大小極不平均時的推論（H6：17 個街區，最大的一個 438 家）
# ---------------------------------------------------------------------------

WEBB_WEIGHTS = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def webb_wild_bootstrap_p(y: Sequence[float], X: np.ndarray, clusters: Sequence,
                          coef_index: int = 1, n_boot: int = 1999,
                          seed_offset: int = 11) -> dict:
    """OLS 係數的野自助法（wild cluster bootstrap）雙尾 p 值，Webb 六點權重。

    為什麼不用一般的群集穩健標準誤：群集數 G 很小（這裡 17）時，群集穩健 t 統計量
    的分布離 t 分布很遠，p 值會系統性偏小。Webb 權重在 G < 20 時比 Rademacher
    （只有 ±1、最多只有 2^G 種抽法）更穩。

    用的是受限虛無假設下的重抽（restricted WCB）：先在「該係數＝0」的限制下估計，
    把殘差按群集乘上同一個權重，重新產生 y*，再算 t 統計量的分布。

    回傳 beta、群集穩健 t、野自助法 p、群集數，以及 refused 說明（群集太少時）。
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    cl = np.asarray(clusters, dtype=object)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & ~pd.isna(cl)
    y, X, cl = y[ok], X[ok], cl[ok]
    keys = sorted(set(cl.tolist()), key=str)
    G = len(keys)
    idx = {k: np.flatnonzero(cl == k) for k in keys}

    def ols(Xm, yv):
        return np.linalg.lstsq(Xm, yv, rcond=None)[0]

    def cluster_t(Xm, yv, j):
        b = ols(Xm, yv)
        e = yv - Xm @ b
        XtX_inv = np.linalg.pinv(Xm.T @ Xm)
        meat = np.zeros((Xm.shape[1], Xm.shape[1]))
        for k in keys:
            i = idx[k]
            s = Xm[i].T @ e[i]
            meat += np.outer(s, s)
        V = XtX_inv @ meat @ XtX_inv
        se = float(np.sqrt(max(V[j, j], 0.0)))
        return float(b[j]), (float(b[j] / se) if se > 0 else float("nan")), se

    beta, t_obs, se = cluster_t(X, y, coef_index)
    if G < 5:
        return {"beta": beta, "se": se, "t": t_obs, "p": float("nan"), "G": G,
                "refused": f"只有 {G} 個群集，野自助法無法給有意義的 p 值"}

    # 受限模型：拿掉要檢定的那一欄
    keep = [c for c in range(X.shape[1]) if c != coef_index]
    Xr = X[:, keep]
    br = ols(Xr, y)
    fitted_r, e_r = Xr @ br, y - Xr @ br

    rng = config.rng(seed_offset)
    t_star = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.choice(WEBB_WEIGHTS, size=G)
        ys = fitted_r.copy()
        for wi, k in zip(w, keys):
            i = idx[k]
            ys[i] = fitted_r[i] + wi * e_r[i]
        t_star[b] = cluster_t(X, ys, coef_index)[1]
    t_star = t_star[np.isfinite(t_star)]
    if t_star.size == 0 or not np.isfinite(t_obs):
        return {"beta": beta, "se": se, "t": t_obs, "p": float("nan"), "G": G,
                "refused": "自助法的 t 統計量全都無法計算（設計矩陣可能有共線性）"}
    p = (1.0 + np.sum(np.abs(t_star) >= abs(t_obs))) / (1.0 + t_star.size)
    return {"beta": beta, "se": se, "t": t_obs, "p": float(p), "G": G,
            "n_boot_used": int(t_star.size), "refused": None}


def leave_one_cluster_out(fit: Callable[[pd.DataFrame], float], df: pd.DataFrame,
                          cluster_col: str) -> pd.DataFrame:
    """逐一拿掉每個群集重估，看結論是不是由單一街區撐起來的。

    H6 的最大街區有 438 家（佔分析單位 23%）。如果拿掉它效果就消失，那就不是
    「街區內的變異」在說話，而是那一個街區的特殊性，必須在論文裡講清楚。
    """
    rows = [{"拿掉的街區": "（全部保留）", "該街區筆數": 0, "剩餘筆數": len(df),
             "估計值": float(fit(df))}]
    for k, g in df.groupby(cluster_col, dropna=False):
        sub = df[df[cluster_col].ne(k) if pd.notna(k) else df[cluster_col].notna()]
        try:
            est = float(fit(sub))
        except Exception as exc:                       # noqa: BLE001
            est = float("nan")
            rows.append({"拿掉的街區": str(k), "該街區筆數": len(g), "剩餘筆數": len(sub),
                         "估計值": est, "備註": f"估不出來：{type(exc).__name__}"})
            continue
        rows.append({"拿掉的街區": str(k), "該街區筆數": len(g), "剩餘筆數": len(sub),
                     "估計值": est})
    out = pd.DataFrame(rows)
    base = out.loc[0, "估計值"]
    out["與全樣本的差"] = out["估計值"] - base
    out["變動比例"] = np.where(np.isclose(base, 0), np.nan, (out["估計值"] - base) / base)
    return out


# ---------------------------------------------------------------------------
# 一般線性限制的群集穩健 OLS 與野自助法 Wald 檢定（H6 的聯合檢定）
# ---------------------------------------------------------------------------

@dataclass
class ClusterOLS:
    """OLS 加群集穩健共變異（CR1）。X 已含常數或已做組內去平均。"""
    beta: np.ndarray
    V: np.ndarray            # CR1 共變異矩陣
    resid: np.ndarray
    G: int
    n: int
    k: int

    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.V), 0, None))


def _cluster_index(clusters) -> tuple[np.ndarray, int]:
    codes, uniq = pd.factorize(pd.Series(list(clusters)), sort=True)
    return codes.astype(int), len(uniq)


def _cr_meat(X: np.ndarray, e: np.ndarray, codes: np.ndarray, G: int) -> np.ndarray:
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, codes, X * e[:, None])
    return scores.T @ scores


def cluster_ols(y, X, clusters, resid_for_meat: np.ndarray | None = None,
                absorbed: int = 0) -> ClusterOLS:
    """OLS ＋ CR1 群集穩健共變異。

    resid_for_meat：用指定的殘差（例如虛無模型的殘差）計算共變異，給事前 MDE 用。
    absorbed：被組內去平均吸收掉的參數個數（固定效果個數），只影響小樣本校正因子。
    """
    y = np.asarray(y, float); X = np.asarray(X, float)
    codes, G = _cluster_index(clusters)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    e_m = e if resid_for_meat is None else np.asarray(resid_for_meat, float)
    kk = k + absorbed
    c = (G / max(G - 1, 1)) * ((n - 1) / max(n - kk, 1))
    V = c * XtX_inv @ _cr_meat(X, e_m, codes, G) @ XtX_inv
    return ClusterOLS(beta=b, V=V, resid=e, G=G, n=n, k=k)


def wild_cluster_wald(y, X, clusters, R: np.ndarray, n_boot: int = 9999,
                      seed_offset: int = 13, weights: str = "webb") -> dict:
    """檢定線性限制 R β = 0 的野自助法（restricted wild cluster bootstrap）。

    - 統計量：群集穩健 Wald，W = (Rb)'(R V R')⁻¹(Rb)；只有一條限制時等於 t²。
    - 在虛無（R β = 0）下以限制最小平方法估計，殘差按群集乘上同一個權重產生 y*。
    - 權重：Webb 六點（G < 20 時比 Rademacher 穩）。
    回傳 W、自助法 p、限制數 q、群集數 G。
    """
    y = np.asarray(y, float); X = np.asarray(X, float); R = np.atleast_2d(np.asarray(R, float))
    codes, G = _cluster_index(clusters)
    q = R.shape[0]
    XtX_inv = np.linalg.pinv(X.T @ X)
    A = XtX_inv @ X.T                                  # b = A y

    def wald(yv):
        b = A @ yv
        e = yv - X @ b
        V = XtX_inv @ _cr_meat(X, e, codes, G) @ XtX_inv
        rb = R @ b
        M = R @ V @ R.T
        return float(rb @ np.linalg.pinv(M) @ rb)

    W = wald(y)
    if G < 5:
        return {"W": W, "p": float("nan"), "q": q, "G": G, "refused": f"只有 {G} 個群集"}
    # 限制最小平方：b_R = b − (X'X)⁻¹R'[R(X'X)⁻¹R']⁻¹ R b
    b = A @ y
    adj = XtX_inv @ R.T @ np.linalg.pinv(R @ XtX_inv @ R.T) @ (R @ b)
    bR = b - adj
    fit_R = X @ bR
    e_R = y - fit_R
    pool = WEBB_WEIGHTS if weights == "webb" else np.array([-1.0, 1.0])
    rng = config.rng(seed_offset)
    Ws = np.empty(n_boot)
    for i in range(n_boot):
        w = rng.choice(pool, size=G)
        Ws[i] = wald(fit_R + w[codes] * e_R)
    Ws = Ws[np.isfinite(Ws)]
    p = (1.0 + np.sum(Ws >= W)) / (1.0 + Ws.size)
    return {"W": W, "p": float(p), "q": q, "G": G, "n_boot": int(Ws.size), "refused": None}


def within_demean(df: pd.DataFrame, cols: list[str], group: str) -> pd.DataFrame:
    """組內去平均（等同於加入群組固定效果）。"""
    g = df.groupby(group)[cols].transform("mean")
    return df[cols] - g
