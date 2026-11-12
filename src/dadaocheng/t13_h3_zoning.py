"""T13：H3 業種分區（主街中藥、巷弄文創、民生西路以北南北貨）。

預先登錄：
  三個描述   (a) 中藥集中在主街；(b) 文創集中在巷弄；(c) 南北貨集中在民生西路以北
  檢定       每個描述：該業種落在指定區位的比例，對照所有分析單位 POI 落在該區位的比例；
             置換檢定固定位置、洗牌業種標籤 999 次，單尾
  輔助       中藥、文創、南北貨三組的共同區位商數（CLQ）與同一套置換檢定（描述用）
  Holm p     三個描述 p 值中的第二小者（至少兩個描述成立才算支持）；
             可計算的 p 值少於兩個時，H3 無法檢定、不提供 Holm p（家族大小仍為 4）
  證偽       三個分區描述中有兩個以上不顯著

業種判定：
  中藥、南北貨   由店名關鍵字細分（settings.traditional_subtypes），不依賴 LLM
  文創           預設用 Places 類別的「選物與文創零售」＋「文化空間」；
                 閘門一通過後可改用 LLM 的 wenqing_group（--wenqing-col）
這一條在閘門一之前就能交付（提案書 8.8 備援方案：H3 改用 Places 類別）。

用法：python -m dadaocheng.t13_h3_zoning
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .stats import HypothesisResult, clq, fmt_p, fmt_pct, permutation_share

CULTURAL_CATEGORIES = ["選物與文創零售", "文化空間"]


def define_groups(au: pd.DataFrame, wenqing_col: str | None = None) -> pd.DataFrame:
    """加上三個業種旗標與兩個區位旗標。"""
    df = au.copy()
    df["grp_中藥"] = df["traditional_subtype"] == "中藥"
    df["grp_南北貨"] = df["traditional_subtype"] == "南北貨"
    if wenqing_col and wenqing_col in df:
        df["grp_文創"] = df[wenqing_col].isin(["選物與工藝", "閱讀與展演"])
    else:
        df["grp_文創"] = df["places_category"].isin(CULTURAL_CATEGORIES)
    df["loc_主街"] = df["on_main_street"].astype("boolean").fillna(False).astype(bool)
    df["loc_巷弄"] = ~df["loc_主街"]
    df["loc_民生西路以北"] = df["north_of_minsheng"].astype("boolean").fillna(False).astype(bool)
    return df


CLAIMS = [
    ("a", "中藥集中在主街", "grp_中藥", "loc_主街"),
    ("b", "文創集中在巷弄", "grp_文創", "loc_巷弄"),
    ("c", "南北貨集中在民生西路以北", "grp_南北貨", "loc_民生西路以北"),
]


def run(au: pd.DataFrame, wenqing_col: str | None = None, n_perm: int = 999, k_nn: int = 10) -> HypothesisResult:
    res = HypothesisResult("H3", "業種分區：主街中藥、巷弄文創、民生西路以北南北貨")
    df = define_groups(au, wenqing_col)
    res.notes.append("文創定義：" + ("LLM 的 " + wenqing_col if wenqing_col and wenqing_col in df
                                   else "Places 類別的「選物與文創零售」＋「文化空間」"))
    res.notes.append("中藥與南北貨由店名關鍵字細分（settings.traditional_subtypes）")

    rows = []
    for tag, claim, grp, loc in CLAIMS:
        r = permutation_share(df[loc], df[grp], n_perm=n_perm, seed_offset=13 + ord(tag))
        rows.append({"描述": f"({tag}) {claim}", "業種家數": r["n_target"],
                     "落在該區位": r["k_in_region"], "業種內比例": r["observed"],
                     "全體 POI 在該區位的比例": r["baseline"],
                     "差": r["observed"] - r["baseline"] if r["n_target"] else np.nan,
                     "置換檢定 p": r["p"], "樣本不足": r["underpowered"]})
    tab = pd.DataFrame(rows)
    res.tables["claims"] = tab

    ps = sorted(pd.to_numeric(tab["置換檢定 p"], errors="coerce").dropna().tolist())
    res.p_for_holm = ps[1] if len(ps) >= 2 else float("nan")
    sig = sum(p < 0.05 for p in ps)
    for _, r in tab.iterrows():
        res.notes.append(f"{r['描述']}：{r['業種家數']} 家中有 {r['落在該區位']} 家"
                         f"（{fmt_pct(r['業種內比例'])}，全體為 {fmt_pct(r['全體 POI 在該區位的比例'])}）"
                         f"，p = {fmt_p(r['置換檢定 p'])}"
                         + ("　※樣本不足 10 家" if r["樣本不足"] else ""))

    # CLQ（描述用）
    xy = df[["x", "y"]].to_numpy(dtype=float)
    clq_rows = []
    for name in ["中藥", "南北貨", "文創"]:
        c = clq(xy, df[f"grp_{name}"], k=k_nn, n_perm=n_perm, seed_offset=20 + len(name))
        clq_rows.append({"業種": name, "家數": c["n_a"], f"CLQ(k={k_nn})": c["clq"],
                         "置換檢定 p": c["p"], "樣本不足": c["underpowered"]})
    res.tables["clq"] = pd.DataFrame(clq_rows)
    for r in clq_rows:
        if not np.isnan(r[f"CLQ(k={k_nn})"]):
            res.notes.append(f"CLQ {r['業種']} = {r[f'CLQ(k={k_nn})']:.2f}（p = {fmt_p(r['置換檢定 p'])}）"
                             "，>1 代表同業比隨機更靠在一起")

    if len(ps) < 2:
        res.verdict = f"**無法檢定**：只有 {len(ps)} 個分區描述可計算有效 p 值；不提供 Holm p 值"
    elif sig >= 2:
        res.verdict = f"**支持**：三個分區描述中有 {sig} 個達顯著（預先登錄要求至少 2 個）"
    elif len(ps) - sig >= 2:
        res.verdict = f"**證偽**：三個分區描述中只有 {sig} 個達顯著（有兩個以上不顯著）"
    else:
        res.verdict = "**未能支持或證偽**：可檢定的分區描述中，顯著與不顯著各一"
    return res


def figures(au: pd.DataFrame, res: HypothesisResult, wenqing_col: str | None = None) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import geo
    from .plotting import setup_cjk_font

    setup_cjk_font()
    out = config.path("figures"); out.mkdir(parents=True, exist_ok=True)
    df = define_groups(au, wenqing_col)
    paths = []

    # 分布地圖
    import geopandas as gpd

    main = gpd.GeoSeries([geo.main_polygon_metric()], crs=geo.metric_crs())
    fig, ax = plt.subplots(figsize=(6, 9))
    main.boundary.plot(ax=ax, color="#999", linewidth=1)
    ax.scatter(df["x"], df["y"], s=3, c="#dddddd", label="其他 POI")
    for name, col, colour, marker in [("中藥", "grp_中藥", "#2f855a", "s"), ("南北貨", "grp_南北貨", "#b7791f", "^"),
                                      ("文創", "grp_文創", "#c53030", "o")]:
        g = df[df[col]]
        ax.scatter(g["x"], g["y"], s=18, c=colour, marker=marker, label=f"{name}（{len(g)}）", alpha=0.85)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_title("H3：三個業種的空間分布", fontsize=11)
    fig.tight_layout()
    p = out / "h3_zoning_map.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 觀測比例 vs 全體基準
    t = res.tables["claims"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(t))
    ax.barh(y + 0.19, t["業種內比例"], height=0.36, color="#4a6fa5", label="該業種落在該區位的比例")
    ax.barh(y - 0.19, t["全體 POI 在該區位的比例"], height=0.36, color="#cccccc", label="全體 POI 的基準比例")
    for i, r in t.iterrows():
        ax.text(max(r["業種內比例"], r["全體 POI 在該區位的比例"]) + 0.015, i,
                f"p = {fmt_p(r['置換檢定 p'])}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(t["描述"], fontsize=9)
    ax.set_xlim(0, 1.12); ax.set_xlabel("比例")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("H3：業種在指定區位的集中程度（置換檢定）", fontsize=11)
    fig.tight_layout()
    p = out / "h3_claims.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wenqing-col", default=None, help="閘門一通過後改用 LLM 的 wenqing_group")
    ap.add_argument("--n-perm", type=int, default=999)
    args = ap.parse_args(argv)
    from .t11_h1_hours import load_analysis_units

    au = load_analysis_units()
    res = run(au, wenqing_col=args.wenqing_col, n_perm=args.n_perm)
    res.save()
    print(res.tables["claims"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(res.tables["clq"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(res.summary())
    for p in figures(au, res, args.wenqing_col):
        print("圖：", p)
    pd.DataFrame([{"hypothesis": "H3", "p_raw": res.p_for_holm, "verdict": res.verdict}]) \
        .to_csv(config.path("tables") / "h3_pvalue.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
