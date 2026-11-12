"""T11：H1 夜間經濟弱，多數店家傍晚就打烊。

預先登錄（docs/preregistration.md 第 3 節）：
  變數      closes_before_19（平日關門時間中位數 < 19:00）；業態大類；是否位於主街
  估計      比例與 95% 信賴區間：整體、分業態大類、主街對巷弄
  區間      主要＝以街區為單位的群集自助法；並列 Wilson 區間
  Holm p    整體比例的單尾精確二項檢定，H₀：p ≤ 0.5
  證偽      整體比例的 95% 信賴區間上界低於 50%
  缺失      沒有營業時間的 POI 排除於分母，並報告缺失比例；24 小時營業視為不在 19 點前關門

這一條只用 Places 的營業時間，不依賴 LLM，所以閘門一之前就能交付（提案書 8.8 備援方案）。

用法：python -m dadaocheng.t11_h1_hours
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .stats import HypothesisResult, Proportion, binom_test_greater, fmt_p, fmt_pct, proportion_with_ci

MIN_GROUP = 15  # 分組報告的最小樣本數；小於此只列出不下結論


def load_analysis_units() -> pd.DataFrame:
    p = config.path("final") / "poi_master.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在。請先跑 python -m dadaocheng.t04_features")
    df = pd.read_parquet(p)
    return df[df["analysis_unit"]].copy()


def run(au: pd.DataFrame, n_boot: int = 2000) -> HypothesisResult:
    res = HypothesisResult("H1", "夜間經濟弱，多數店家平日 19:00 前就打烊")
    have = au["closes_before_19"].notna()
    n_total, n_have = len(au), int(have.sum())
    res.notes.append(f"分析單位 {n_total} 家，其中 {n_have} 家有營業時間（{n_have / n_total:.1%}）；"
                     f"沒有營業時間的 {n_total - n_have} 家排除於分母，不當成「不營業」")
    status = au["hours_status"].value_counts().to_dict()
    res.notes.append("營業時間狀態：" + "、".join(f"{k} {v}" for k, v in status.items()))

    overall = proportion_with_ci(au["closes_before_19"], au["block_id"], "整體", n_boot=n_boot, seed_offset=11)
    rows = [overall]

    # 分業態大類
    for cat, g in au.groupby("places_category"):
        if int(g["closes_before_19"].notna().sum()) == 0:
            continue
        rows.append(proportion_with_ci(g["closes_before_19"], g["block_id"], f"業態：{cat}",
                                       n_boot=n_boot, seed_offset=12))
    # 主街對巷弄
    if "on_main_street" in au:
        for flag, label in [(True, "主街（迪化街）"), (False, "巷弄與其他街道")]:
            g = au[au["on_main_street"] == flag]
            if len(g):
                rows.append(proportion_with_ci(g["closes_before_19"], g["block_id"], f"區位：{label}",
                                               n_boot=n_boot, seed_offset=13))

    tab = pd.DataFrame([r.row() for r in rows])
    tab["樣本足夠"] = tab["n"] >= MIN_GROUP
    res.tables["proportions"] = tab

    p = binom_test_greater(overall.k, overall.n, 0.5)
    res.p_for_holm = p
    res.notes.append(f"整體比例 {fmt_pct(overall.p)}（{overall.k}/{overall.n}）"
                     f"；街區自助法 95% 區間 [{fmt_pct(overall.boot[0])}, {fmt_pct(overall.boot[1])}]"
                     f"；Wilson [{fmt_pct(overall.wilson[0])}, {fmt_pct(overall.wilson[1])}]"
                     f"；街區數 {overall.n_clusters}")
    res.notes.append(f"單尾精確二項檢定（H₀：p ≤ 0.5）p = {fmt_p(p)}")

    # 證偽判定依預先登錄：以主要版本（群集自助法）的上界為準
    hi = overall.boot[1]
    if np.isnan(hi):
        res.verdict = "無法判定（樣本不足）"
    elif hi < 0.5:
        res.verdict = f"**證偽**：95% 區間上界 {fmt_pct(hi)} < 50%，多數店家並非 19 點前打烊"
    elif overall.boot[0] > 0.5:
        res.verdict = f"**支持**：95% 區間下界 {fmt_pct(overall.boot[0])} > 50%"
    else:
        res.verdict = (f"**未證偽但也未達 50% 門檻**：區間 [{fmt_pct(overall.boot[0])}, {fmt_pct(hi)}] 跨越 50%，"
                       "資料不足以判斷「多數」")

    # 業態之間的差異（描述，不檢定；預先登錄沒有這一項）
    sub = tab[(tab["組別"].str.startswith("業態")) & tab["樣本足夠"]].sort_values("比例")
    if len(sub) >= 2:
        res.notes.append(f"業態間差異（描述）：最低 {sub.iloc[0]['組別']} {fmt_pct(sub.iloc[0]['比例'])}"
                         f"、最高 {sub.iloc[-1]['組別']} {fmt_pct(sub.iloc[-1]['比例'])}")
    return res


def figures(au: pd.DataFrame, tab: pd.DataFrame) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import setup_cjk_font

    setup_cjk_font()
    out = config.path("figures")
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # 關門時間分布
    fig, ax = plt.subplots(figsize=(7.5, 4))
    v = au["close_min_weekday"].dropna() / 60.0
    v = v[v <= 30]
    ax.hist(v, bins=np.arange(10, 27.5, 0.5), color="#4a6fa5", edgecolor="white")
    ax.axvline(19, color="#c53030", linestyle="--", linewidth=2)
    ax.text(19.1, ax.get_ylim()[1] * 0.92, "19:00", color="#c53030", fontsize=10)
    ax.set_xlabel("平日關門時間（小時，跨午夜以 24 以上表示）")
    ax.set_ylabel("店家數")
    ax.set_title(f"H1：平日關門時間分布（n = {len(v)}）", fontsize=11)
    ax.set_xticks(range(10, 28, 2))
    fig.tight_layout()
    p = out / "h1_close_time_hist.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 分組比例與區間
    t = tab[tab["n"] >= MIN_GROUP].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.45 * len(t) + 1.4)))
    y = np.arange(len(t))
    lo = t["比例"] - t["街區自助法 下界"]
    hi = t["街區自助法 上界"] - t["比例"]
    colors = ["#c53030" if g == "整體" else "#4a6fa5" for g in t["組別"]]
    ax.errorbar(t["比例"], y, xerr=[lo.clip(lower=0), hi.clip(lower=0)], fmt="o", color="#333",
                ecolor="#888", capsize=4, markersize=0)
    ax.scatter(t["比例"], y, c=colors, s=60, zorder=3)
    ax.axvline(0.5, color="#c53030", linestyle="--", linewidth=1.5)
    ax.set_yticks(y); ax.set_yticklabels([f"{g}（n={n}）" for g, n in zip(t["組別"], t["n"])], fontsize=9)
    ax.set_xlabel("平日 19:00 前關門的比例（點與 95% 街區自助法區間）")
    ax.set_xlim(0, 1)
    ax.set_title("H1：各組「傍晚就打烊」的比例", fontsize=11)
    fig.tight_layout()
    p = out / "h1_proportions.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    au = load_analysis_units()
    res = run(au, n_boot=args.n_boot)
    res.save()
    print(res.tables["proportions"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(res.summary())
    for p in figures(au, res.tables["proportions"]):
        print("圖：", p)
    pd.DataFrame([{"hypothesis": "H1", "p_raw": res.p_for_holm, "verdict": res.verdict}]) \
        .to_csv(config.path("tables") / "h1_pvalue.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
