"""T12：H5 歷史建築限制熱食與明火。

預先登錄：
  變數      in_heritage；是否為熱食餐飲
  檢定      2×2 列聯表與單尾 Fisher 精確檢定（歷建內熱食比例較低）
            以街區分層的 Mantel–Haenszel 作為穩健性檢查
  Holm p    Fisher 單尾 p 值
  證偽      歷建內熱食比例不低於歷建外
  檢定力     歷建內 POI 少於 20 家時，報告「檢定力不足」，不改用其他檢定

業態來源：預設用 Places 類別（places_category），閘門一通過後可改用 LLM 的 category 並附 PPI 修正版
（--category-col 可切換）。這一條在閘門一之前就能交付。

用法：python -m dadaocheng.t12_h5_heritage
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .stats import HypothesisResult, fisher_lower, fmt_p, fmt_pct, mantel_haenszel, tables_by_stratum
from .t11_h1_hours import load_analysis_units

HOT_FOOD = "熱食餐飲"


def run(au: pd.DataFrame, category_col: str = "places_category") -> HypothesisResult:
    res = HypothesisResult("H5", "歷史建築限制熱食與明火（歷建內的熱食餐飲比例較低）")
    df = au[au["in_heritage"].notna() & au[category_col].notna()].copy()
    df["is_hot"] = df[category_col] == HOT_FOOD
    df["in_h"] = df["in_heritage"].astype(bool)
    res.notes.append(f"業態來源欄位：{category_col}"
                     + ("（Places 類別，不依賴 LLM）" if category_col == "places_category" else "（LLM 抽取）"))

    a = int((df["in_h"] & df["is_hot"]).sum())
    b = int((df["in_h"] & ~df["is_hot"]).sum())
    c = int((~df["in_h"] & df["is_hot"]).sum())
    d = int((~df["in_h"] & ~df["is_hot"]).sum())
    f = fisher_lower(a, b, c, d)

    res.tables["contingency"] = pd.DataFrame([
        {"": "歷建內", "熱食餐飲": a, "非熱食": b, "合計": a + b, "熱食比例": f["p_in"]},
        {"": "歷建外", "熱食餐飲": c, "非熱食": d, "合計": c + d, "熱食比例": f["p_out"]},
    ])
    res.p_for_holm = f["p"]
    res.notes.append(f"歷建內 {a}/{a + b} = {fmt_pct(f['p_in'])} 是熱食；"
                     f"歷建外 {c}/{c + d} = {fmt_pct(f['p_out'])}")
    res.notes.append(f"單尾 Fisher 精確檢定 p = {fmt_p(f['p'])}；勝算比 = {f['odds_ratio']:.3f}")

    mh = mantel_haenszel(tables_by_stratum(df, "block_id", "in_h", "is_hot"))
    res.tables["mantel_haenszel"] = pd.DataFrame([mh])
    if np.isnan(mh["or_mh"]) or mh.get("unstable"):
        res.notes.append(f"Mantel–Haenszel：估計不穩定（可用分層 {mh['n_strata']} 個，"
                         "某些分層的熱食或歷建格為 0），不報共同勝算比；以 Fisher 為主要檢定")
    else:
        res.notes.append(f"以街區分層的 Mantel–Haenszel 共同勝算比 = {mh['or_mh']:.3f}"
                         f" [{mh.get('ci_low', float('nan')):.3f}, {mh.get('ci_high', float('nan')):.3f}]"
                         f"，p = {fmt_p(mh['p'])}（{mh['n_strata']} 個分層）")

    if f["underpowered"]:
        res.verdict = (f"**檢定力不足**：歷建內只有 {a + b} 家（預先登錄門檻為 20 家）。"
                       f"點估計為歷建內 {fmt_pct(f['p_in'])} 對歷建外 {fmt_pct(f['p_out'])}，"
                       "依預先登錄照實報告，不改用其他檢定。")
        res.notes.append("提高樣本的方式：補齊古蹟（classifyCode 1.1）、或把比對距離從 15 公尺放寬"
                         "（放寬要寫進 deviations.md，且會引入誤判）")
    elif f["p_in"] >= f["p_out"]:
        res.verdict = (f"**證偽**：歷建內熱食比例 {fmt_pct(f['p_in'])} 不低於歷建外 {fmt_pct(f['p_out'])}")
    elif f["p"] < 0.05:
        res.verdict = (f"**支持**：歷建內熱食比例顯著較低（{fmt_pct(f['p_in'])} 對 {fmt_pct(f['p_out'])}，"
                       f"p = {fmt_p(f['p'])}）")
    else:
        res.verdict = (f"**方向一致但未達顯著**：歷建內 {fmt_pct(f['p_in'])} < 歷建外 {fmt_pct(f['p_out'])}，"
                       f"但 p = {fmt_p(f['p'])}")

    # 描述：歷建內的業態分布，幫助解讀
    if df["in_h"].any():
        dist = (df[df["in_h"]][category_col].value_counts(normalize=True).rename("歷建內比例").to_frame()
                .join(df[~df["in_h"]][category_col].value_counts(normalize=True).rename("歷建外比例"), how="outer")
                .fillna(0).reset_index(names="業態大類"))
        res.tables["category_mix"] = dist
    return res


def figures(res: HypothesisResult) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import setup_cjk_font
    from .stats import wilson_ci

    setup_cjk_font()
    out = config.path("figures"); out.mkdir(parents=True, exist_ok=True)
    t = res.tables["contingency"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = [f"{r['']}\n(n={r['合計']})" for _, r in t.iterrows()]
    props = t["熱食比例"].values
    cis = [wilson_ci(int(r["熱食餐飲"]), int(r["合計"])) for _, r in t.iterrows()]
    err = np.array([[p - lo for p, (lo, _) in zip(props, cis)], [hi - p for p, (_, hi) in zip(props, cis)]])
    ax.bar(labels, props, color=["#c53030", "#4a6fa5"], width=0.55)
    ax.errorbar(labels, props, yerr=np.clip(err, 0, None), fmt="none", ecolor="#333", capsize=5)
    ax.set_ylabel("熱食餐飲的比例")
    ax.set_title("H5：歷史建築內外的熱食餐飲比例（誤差棒為 Wilson 95%）", fontsize=10)
    ax.set_ylim(0, max(0.5, float(np.nanmax(props)) * 1.5 + 0.05))
    for x, p in zip(labels, props):
        ax.text(x, p + 0.012, f"{p:.1%}", ha="center", fontsize=10)
    fig.tight_layout()
    p = out / "h5_heritage_hotfood.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    return [p]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category-col", default="places_category",
                    help="業態欄位；閘門一通過後可改成 LLM 的 category")
    args = ap.parse_args(argv)
    au = load_analysis_units()
    res = run(au, category_col=args.category_col)
    res.save()
    print(res.tables["contingency"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(res.summary())
    for p in figures(res):
        print("圖：", p)
    pd.DataFrame([{"hypothesis": "H5", "p_raw": res.p_for_holm, "verdict": res.verdict}]) \
        .to_csv(config.path("tables") / "h5_pvalue.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
