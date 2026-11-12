"""T14：H6（簡報稱「假說四」）歷史環境與店家關注度的關聯因業態而異。主問題。

預先登錄（docs/preregistration.md）：
  模型    log(1 + 評論數) = 歷史密度(300 m, z) × 業態大類 + 是否連鎖 + 是否位於主街
                          + 捷運距離 + 價位 + 街區固定效果
          參照業態：熱食餐飲；排除業態「其他」；文化空間不估固定效果交互項，改作描述
  推論    主要 p 值：Webb 權重的 wild cluster bootstrap（預設 9,999 次）
  Holm p  所有「歷史密度 × 業態」交互項的聯合 Wald 檢定（野自助法 p）
  證偽    每個交互項的 95% 信賴區間都落在 ±0.10（log 單位，每 1 SD 歷史密度）之內
  MDE     估計前先算；MDE > 0.10 時事先宣告無法證偽
  附錄    leave-one-cluster-out

實作上的參數化：每個業態各自一條歷史密度斜率（kde × 1[業態 = c]），而不是「共同斜率＋交互項」。
兩者等價；這樣各業態的斜率直接是係數（森林圖用），交互項＝該業態斜率 − 熱食餐飲斜率，
聯合檢定＝「所有業態斜率相等」這組線性限制。

三個規格：
  M0 描述      不含固定效果與控制變數（含常數與業態主效果）
  M1 主規格    預先登錄的完整模型（街區固定效果＋控制變數）← Holm 用這個
  M2 穩健性    街區固定效果，不含控制變數

業態來源：預設 Places 類別（閘門一未通過，提案書 8.8）；`--category-col category` 改用 LLM 類別（敏感性）。
是否連鎖只能來自 LLM（extractions.parquet）；沒有該檔時自動略過這個控制變數並註明。

用法：
  python -m dadaocheng.t14_h6_density --mde-only        # 先算 MDE（不估計業態斜率），寫進 deviations／預先登錄
  python -m dadaocheng.t14_h6_density                   # 完整估計
  python -m dadaocheng.t14_h6_density --kde-col heritage_kde_150 --tag bw150   # 敏感性
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats as sps

from . import config
from .stats import HypothesisResult, cluster_ols, fmt_p, within_demean, wild_cluster_wald
from .t11_h1_hours import load_analysis_units

REF = "熱食餐飲"
MODEL_CATS = [REF, "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "一般零售與品牌"]
DESCRIBE_ONLY = "文化空間"
EXCLUDED = {"其他", "不明"}
FALSIFY_BOUND = 0.10


# ---------------------------------------------------------------------------
# 資料
# ---------------------------------------------------------------------------

def prepare(au: pd.DataFrame, kde_col: str = "heritage_kde_300", category_col: str = "places_category",
            extractions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """回傳 (估計樣本, 註記)。歷史密度在分析單位全體（排除業態前）標準化。"""
    notes = []
    df = au.copy()
    if extractions is not None:
        keep = [c for c in ["poi_id", "is_chain", "category"] if c in extractions.columns]
        df = df.merge(extractions[keep].rename(columns={"category": "llm_category"}), on="poi_id", how="left")
    if category_col == "category":
        category_col = "llm_category"
    df["kde_z"] = (df[kde_col] - df[kde_col].mean()) / df[kde_col].std(ddof=0)
    if "dist_mrt_m" in df:
        df["mrt_z"] = (df["dist_mrt_m"] - df["dist_mrt_m"].mean()) / df["dist_mrt_m"].std(ddof=0)
    df["cat"] = df[category_col]
    df["y"] = np.log1p(pd.to_numeric(df["user_rating_count"], errors="coerce"))

    n0 = len(df)
    miss_y = df["y"].isna()
    notes.append(f"分析單位 {n0} 家；缺評論數 {int(miss_y.sum())} 家排除（評論數是結果變數，缺值不補）")
    df = df[~miss_y & df["kde_z"].notna() & df["cat"].notna()]
    excl = df["cat"].isin(EXCLUDED)
    notes.append(f"依預先登錄排除業態「其他」／「不明」{int(excl.sum())} 家")
    df = df[~excl]
    n_cs = int((df["cat"] == DESCRIBE_ONLY).sum())
    notes.append(f"文化空間 {n_cs} 家不估固定效果交互項，改作描述（見 h6_culture_describe.csv）")
    other = ~df["cat"].isin(MODEL_CATS + [DESCRIBE_ONLY])
    if other.any():
        notes.append(f"不在模型業態清單中的 {int(other.sum())} 家排除：{sorted(df.loc[other, 'cat'].unique())}")
        df = df[~other]
    return df.reset_index(drop=True), notes


def design(df: pd.DataFrame, fe: bool, controls: bool) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """回傳 (y, X, 欄名, 被吸收的參數數)。"""
    cols = {}
    for c in MODEL_CATS:
        cols[f"slope:{c}"] = df["kde_z"] * (df["cat"] == c)
    for c in MODEL_CATS[1:]:
        cols[f"cat:{c}"] = (df["cat"] == c).astype(float)
    if controls:
        cols["on_main_street"] = df["on_main_street"].astype(float)
        if "mrt_z" in df:
            cols["dist_mrt_z"] = df["mrt_z"].fillna(0.0)
        if "price_level_ord" in df:
            p = pd.to_numeric(df["price_level_ord"], errors="coerce")
            cols["price_missing"] = p.isna().astype(float)
            cols["price_level"] = p.fillna(p.median() if p.notna().any() else 0.0)
        if "is_chain" in df and df["is_chain"].notna().any():
            ch = df["is_chain"].fillna("不明")
            cols["chain:是"] = (ch == "是").astype(float)
            cols["chain:不明"] = (ch == "不明").astype(float)
    X = pd.DataFrame(cols).astype(float)
    # 沒有變異的欄位（例如某控制變數全為 0）拿掉，避免奇異矩陣
    X = X.loc[:, X.std(ddof=0) > 0]
    y = df["y"].astype(float)
    if fe:
        tmp = pd.concat([X, y.rename("__y"), df["block_id"]], axis=1)
        dm = within_demean(tmp, list(X.columns) + ["__y"], "block_id")
        X = dm[list(X.columns)]
        X = X.loc[:, X.std(ddof=0) > 1e-12]
        y = dm["__y"]
        absorbed = int(df["block_id"].nunique())
    else:
        X.insert(0, "const", 1.0)
        absorbed = 0
    return y.to_numpy(), X.to_numpy(), list(X.columns), absorbed


def restriction_equal_slopes(names: list[str]) -> tuple[np.ndarray, list[str]]:
    """「各業態斜率都等於熱食餐飲斜率」：每列是 slope:c − slope:熱食。"""
    j0 = names.index(f"slope:{REF}")
    rows, labels = [], []
    for c in MODEL_CATS[1:]:
        k = f"slope:{c}"
        if k in names:
            r = np.zeros(len(names)); r[names.index(k)] = 1.0; r[j0] = -1.0
            rows.append(r); labels.append(c)
    return np.array(rows), labels


def tcrit(G: int, q: float) -> float:
    return float(sps.t.ppf(q, df=max(G - 1, 1)))


# ---------------------------------------------------------------------------
# MDE（估計前）
# ---------------------------------------------------------------------------

def mde(df: pd.DataFrame, power: float = 0.80, alpha: float = 0.05) -> pd.DataFrame:
    """每個交互項的最小可檢測效果。

    只用設計矩陣與「虛無模型」（所有業態共同斜率）的殘差計算標準誤，
    不估計、也不看任何業態別斜率，所以可以在 H6 估計之前寫進預先登錄。
    MDE = (t_{1−α/2, G−1} + t_{power, G−1}) × SE。
    """
    y, X, names, absorbed = design(df, fe=True, controls=True)
    R, labels = restriction_equal_slopes(names)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    bR = b - XtX_inv @ R.T @ np.linalg.pinv(R @ XtX_inv @ R.T) @ (R @ b)
    e_null = y - X @ bR
    fit = cluster_ols(y, X, df["block_id"], resid_for_meat=e_null, absorbed=absorbed)
    G = fit.G
    mult = tcrit(G, 1 - alpha / 2) + tcrit(G, power)
    rows = []
    for r, c in zip(R, labels):
        se = float(np.sqrt(r @ fit.V @ r))
        rows.append({"業態（相對熱食餐飲）": c, "家數": int((df["cat"] == c).sum()),
                     "交互項 SE（虛無模型殘差）": se, "MDE（log 單位）": mult * se,
                     "MDE（評論數倍數）": float(np.exp(mult * se)),
                     "可證偽（MDE ≤ 0.10）": bool(mult * se <= FALSIFY_BOUND)})
    out = pd.DataFrame(rows)
    out.attrs.update({"G": G, "n": len(df), "multiplier": mult, "power": power, "alpha": alpha})
    return out


# ---------------------------------------------------------------------------
# 估計
# ---------------------------------------------------------------------------

def fit_spec(df: pd.DataFrame, fe: bool, controls: bool, n_boot: int, seed_offset: int,
             slope_p: bool) -> dict:
    y, X, names, absorbed = design(df, fe=fe, controls=controls)
    fit = cluster_ols(y, X, df["block_id"], absorbed=absorbed)
    G = fit.G
    tc = tcrit(G, 0.975)
    R, labels = restriction_equal_slopes(names)
    joint = wild_cluster_wald(y, X, df["block_id"], R, n_boot=n_boot, seed_offset=seed_offset)

    slopes = []
    for i, c in enumerate(MODEL_CATS):
        k = f"slope:{c}"
        if k not in names:
            continue
        j = names.index(k)
        est, se = float(fit.beta[j]), float(fit.se()[j])
        row = {"業態": c, "家數": int((df["cat"] == c).sum()),
               "有該業態的街區數": int(df.loc[df["cat"] == c, "block_id"].nunique()),
               "斜率": est, "SE（CR1）": se, "95% 下界": est - tc * se, "95% 上界": est + tc * se}
        if slope_p:
            r1 = np.zeros(len(names)); r1[j] = 1.0
            row["野自助法 p"] = wild_cluster_wald(y, X, df["block_id"], r1[None, :], n_boot=n_boot,
                                                 seed_offset=seed_offset + 1 + i)["p"]
        slopes.append(row)
    inter = []
    for r, c in zip(R, labels):
        est = float(r @ fit.beta); se = float(np.sqrt(r @ fit.V @ r))
        inter.append({"業態（相對熱食餐飲）": c, "交互項": est, "SE（CR1）": se,
                      "95% 下界": est - tc * se, "95% 上界": est + tc * se,
                      "區間落在 ±0.10 內": bool(abs(est - tc * se) <= FALSIFY_BOUND
                                             and abs(est + tc * se) <= FALSIFY_BOUND)})
    coefs = pd.DataFrame({"變數": names, "係數": fit.beta, "SE（CR1）": fit.se()})
    return {"names": names, "fit": fit, "joint": joint, "slopes": pd.DataFrame(slopes),
            "interactions": pd.DataFrame(inter), "coefs": coefs, "G": G, "n": fit.n}


def loco(df: pd.DataFrame) -> pd.DataFrame:
    """逐一剔除街區，重估主規格的各業態斜率與聯合 Wald 統計量（不做自助法）。"""
    rows = []
    for b in [None] + sorted(df["block_id"].unique()):
        sub = df if b is None else df[df["block_id"] != b]
        try:
            y, X, names, absorbed = design(sub, fe=True, controls=True)
            fit = cluster_ols(y, X, sub["block_id"], absorbed=absorbed)
            R, labels = restriction_equal_slopes(names)
            rb = R @ fit.beta
            W = float(rb @ np.linalg.pinv(R @ fit.V @ R.T) @ rb)
            row = {"剔除的街區": "（全部保留）" if b is None else str(b),
                   "該街區家數": 0 if b is None else int((df["block_id"] == b).sum()),
                   "剩餘家數": len(sub), "聯合 Wald（未自助）": W}
            for c in MODEL_CATS:
                k = f"slope:{c}"
                row[f"斜率：{c}"] = float(fit.beta[names.index(k)]) if k in names else np.nan
        except Exception as exc:  # noqa: BLE001
            row = {"剔除的街區": str(b), "備註": f"估不出來：{type(exc).__name__}"}
        rows.append(row)
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    for c in MODEL_CATS:
        k = f"斜率：{c}"
        if k in out:
            out[f"變動：{c}"] = out[k] - base[k]
    return out


def identification(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in [("全部（估計樣本）", df)] + [(c, df[df["cat"] == c]) for c in MODEL_CATS]:
        if len(g) < 2:
            continue
        tot = float(g["kde_z"].std(ddof=0))
        win = float((g["kde_z"] - g.groupby("block_id")["kde_z"].transform("mean")).std(ddof=0))
        rows.append({"樣本": name, "家數": len(g), "總 SD": tot, "街區內 SD": win,
                     "街區內 ÷ 總（標準差）": win / tot if tot > 0 else np.nan,
                     "街區內變異數占比": (win / tot) ** 2 if tot > 0 else np.nan})
    return pd.DataFrame(rows)


def describe_culture(au_df: pd.DataFrame) -> pd.DataFrame:
    g = au_df[au_df["cat"] == DESCRIBE_ONLY]
    if len(g) < 3:
        return pd.DataFrame([{"說明": f"文化空間只有 {len(g)} 家，不做描述"}])
    slope = float(np.polyfit(g["kde_z"], g["y"], 1)[0]) if g["kde_z"].std() > 0 else np.nan
    return pd.DataFrame([{"家數": len(g), "街區數": int(g["block_id"].nunique()),
                          "log(1+評論數) 平均": float(g["y"].mean()),
                          "簡單迴歸斜率（無固定效果、無推論）": slope,
                          "說明": "樣本不足以做固定效果交互項估計；此斜率僅供描述"}])


def verdict(main: dict, mde_tab: pd.DataFrame | None) -> str:
    p = main["joint"]["p"]
    inter = main["interactions"]
    all_narrow = bool(inter["區間落在 ±0.10 內"].all()) if len(inter) else False
    unreachable = mde_tab is not None and not bool(mde_tab["可證偽（MDE ≤ 0.10）"].all())
    if np.isnan(p):
        return "**無法檢定**：" + (main["joint"].get("refused") or "野自助法失敗")
    if p < 0.05:
        return (f"**支持**：各業態的歷史密度斜率不全相同（聯合檢定野自助法 p = {fmt_p(p)}）")
    if all_narrow:
        return ("**證偽**：交互項不顯著，且每個交互項的 95% 區間都落在 ±0.10 之內")
    if unreachable:
        return (f"**無法證偽，也未獲支持**：聯合檢定 p = {fmt_p(p)}；依事前 MDE，"
                "本設計無法分辨 0 與 ±0.10 以內的效果。結論限於「未發現大於 MDE 的業態差異」")
    return (f"**不顯著，但區間太寬無法證偽**：聯合檢定 p = {fmt_p(p)}；"
            "至少一個交互項的 95% 區間超出 ±0.10")


def run(au: pd.DataFrame, kde_col: str = "heritage_kde_300", category_col: str = "places_category",
        n_boot: int = 9999, extractions: pd.DataFrame | None = None,
        slope_p: bool = True) -> tuple[HypothesisResult, dict]:
    res = HypothesisResult("H6", "歷史環境與店家關注度的關聯因業態而異（主問題）")
    df, notes = prepare(au, kde_col, category_col, extractions)
    res.notes += notes
    model_df = df[df["cat"].isin(MODEL_CATS)].reset_index(drop=True)
    res.notes.append(f"估計樣本 {len(model_df)} 家、{model_df['block_id'].nunique()} 個街區；"
                     f"歷史密度欄位 {kde_col}（z 分數）；業態欄位 {category_col}")
    if "is_chain" not in df or not df["is_chain"].notna().any():
        res.notes.append("沒有 extractions.parquet 的 is_chain：控制變數不含是否連鎖（與預先登錄不同，須記偏離）")
    else:
        res.notes.append("是否連鎖來自 LLM 抽取（未經 κ 驗證，僅作控制變數；不明另設一類）")

    mde_tab = mde(model_df)
    specs = {}
    for key, fe, ctl, off in [("M0", False, False, 101), ("M1", True, True, 201), ("M2", True, False, 301)]:
        specs[key] = fit_spec(model_df, fe=fe, controls=ctl, n_boot=n_boot, seed_offset=off,
                              slope_p=slope_p and key == "M1")
    main = specs["M1"]
    res.p_for_holm = main["joint"]["p"]
    res.verdict = verdict(main, mde_tab)

    summary = []
    for key, label in [("M0", "M0 描述（無固定效果、無控制）"), ("M1", "M1 主規格（固定效果＋控制）"),
                       ("M2", "M2 穩健性（固定效果、無控制）")]:
        s = specs[key]
        summary.append({"規格": label, "n": s["n"], "街區數": s["G"],
                        "聯合檢定 Wald": s["joint"]["W"], "限制數": s["joint"]["q"],
                        "野自助法 p": s["joint"]["p"], "自助法次數": s["joint"].get("n_boot")})
    res.tables["specs"] = pd.DataFrame(summary)
    res.tables["slopes_main"] = main["slopes"]
    res.tables["interactions_main"] = main["interactions"]
    res.tables["coefs_main"] = main["coefs"]
    for key in ("M0", "M2"):
        res.tables[f"slopes_{key.lower()}"] = specs[key]["slopes"]
    res.tables["mde"] = mde_tab
    res.tables["loco"] = loco(model_df)
    res.tables["identification"] = identification(model_df)
    res.tables["culture_describe"] = describe_culture(df)

    j = main["joint"]
    res.notes.append(f"主規格聯合檢定（{j['q']} 個交互項）：Wald = {j['W']:.2f}，Webb 野自助法 p = {fmt_p(j['p'])}"
                     f"（{j.get('n_boot')} 次，G = {j['G']}）")
    worst = float(mde_tab["MDE（log 單位）"].max()) if len(mde_tab) else np.nan
    res.notes.append(f"事前 MDE（80% 檢定力）：最大 {worst:.3f} log 單位"
                     + ("，大於 0.10 → 預先登錄的證偽條件不可達" if worst > FALSIFY_BOUND else "，可證偽"))
    lo = res.tables["loco"]
    if "聯合 Wald（未自助）" in lo:
        base_W = lo.iloc[0]["聯合 Wald（未自助）"]
        res.notes.append(f"逐一剔除街區：聯合 Wald 介於 {lo['聯合 Wald（未自助）'].min():.2f}–"
                         f"{lo['聯合 Wald（未自助）'].max():.2f}（全樣本 {base_W:.2f}）")
    return res, specs


# ---------------------------------------------------------------------------
# 圖
# ---------------------------------------------------------------------------

def forest_plot(slopes: pd.DataFrame, tag: str = "") -> object:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import setup_cjk_font

    setup_cjk_font()
    out = config.path("figures"); out.mkdir(parents=True, exist_ok=True)
    s = slopes.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.7 * len(s) + 1.5))
    yy = np.arange(len(s))
    ax.errorbar(s["斜率"], yy, xerr=[s["斜率"] - s["95% 下界"], s["95% 上界"] - s["斜率"]],
                fmt="o", color="#1F2A33", ecolor="#2F5D62", capsize=4)
    ax.axvline(0, color="#888", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{c}（n={n}）" for c, n in zip(s["業態"], s["家數"])])
    ax.set_xlabel("每 1 SD 歷史密度的 log(1+評論數) 變化（95% 區間，CR1，t(G−1)）")
    ax.set_title("H6：各業態的歷史密度斜率（街區固定效果＋控制變數）", fontsize=10)
    fig.tight_layout()
    p = out / f"h6_forest{('_' + tag) if tag else ''}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_extractions() -> pd.DataFrame | None:
    p = config.path("final") / "extractions.parquet"
    return pd.read_parquet(p) if p.exists() else None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mde-only", action="store_true", help="只算事前 MDE，不估計業態斜率")
    ap.add_argument("--kde-col", default="heritage_kde_300")
    ap.add_argument("--category-col", default="places_category",
                    help="places_category（預設）或 category（LLM，敏感性）")
    ap.add_argument("--n-boot", type=int, default=9999)
    ap.add_argument("--tag", default="", help="敏感性分析的輸出檔名後綴；有後綴時不寫 h6_pvalue.csv")
    args = ap.parse_args(argv)

    au = load_analysis_units()
    ex = load_extractions()
    d = config.path("tables"); d.mkdir(parents=True, exist_ok=True)

    if args.mde_only:
        df, notes = prepare(au, args.kde_col, args.category_col, ex)
        m = mde(df[df["cat"].isin(MODEL_CATS)].reset_index(drop=True))
        for n in notes:
            print("·", n)
        print(m.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"\nG = {m.attrs['G']}，n = {m.attrs['n']}，乘數 t(.975)+t(.80) = {m.attrs['multiplier']:.3f}")
        worst = m["MDE（log 單位）"].max()
        print(f"最大 MDE = {worst:.3f}："
              + ("大於 0.10，預先登錄的證偽條件不可達。請在看任何 H6 係數之前把這個數字記進 docs/deviations.md。"
                 if worst > FALSIFY_BOUND else "可證偽。"))
        p = d / f"h6_mde{('_' + args.tag) if args.tag else ''}.csv"
        m.to_csv(p, index=False, encoding="utf-8-sig")
        meta = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.attrs.items()}
        p.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("已存：", p)
        return

    res, specs = run(au, args.kde_col, args.category_col, args.n_boot, ex)
    if args.tag:
        res.hid = f"H6_{args.tag}"
    res.save()
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    print(res.tables["specs"].to_string(index=False, float_format=fmt))
    print()
    print(res.tables["slopes_main"].to_string(index=False, float_format=fmt))
    print()
    print(res.tables["interactions_main"].to_string(index=False, float_format=fmt))
    print()
    print(res.summary())
    print("圖：", forest_plot(res.tables["slopes_main"], args.tag))
    if not args.tag:
        pd.DataFrame([{"hypothesis": "H6", "p_raw": res.p_for_holm, "verdict": res.verdict}]) \
            .to_csv(d / "h6_pvalue.csv", index=False, encoding="utf-8-sig")
        print("Holm 用的 p 值已存：", d / "h6_pvalue.csv", "→ 再跑 python -m dadaocheng.report_rq1")


if __name__ == "__main__":
    main()
