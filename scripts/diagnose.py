"""決策診斷：把「H6 能不能識別」「H1 的結論有多穩」「in_heritage 是否過度納入」變成數字。

這支程式只讀 T04 的產出，不呼叫任何 API、不花錢、不改任何檔案，約 10 秒跑完。

為什麼需要它：
  - H6 的模型含街區固定效果。固定效果會吸收「街區之間」的差異，交互項只能靠
    「街區之內」的變異來識別。歷史密度是一個很平滑的面，街區內可能幾乎是常數——
    如果是這樣，H6 不是「不顯著」，而是**從一開始就識別不出來**。這兩件事的
    學術意義完全不同，必須先用數字分清楚。
  - H1 的 59% 建立在「缺營業時間是隨機缺失」的假設上。29% 的樣本缺這個欄位，
    所以要算出最壞情況的上下界（Manski bounds），看結論會不會翻轉。
  - in_heritage 幾乎全靠「距離 ≤ 15 公尺」判定。大稻埕街屋店面寬約 4–5 公尺，
    15 公尺會把左右鄰棟一起算進來。要知道這個比例有多高。

用法：python scripts/diagnose.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dadaocheng import config  # noqa: E402

pd.set_option("display.width", 200)


def rule(t: str) -> None:
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


def within_between(df: pd.DataFrame, var: str, group: str = "block_id") -> dict:
    """把一個變數拆成「街區之間」與「街區之內」的變異。

    固定效果模型只用得到 within 的部分。absorbed 就是固定效果吃掉的比例。
    """
    d = df[[var, group]].dropna()
    if d.empty or d[var].nunique() < 2:
        return {}
    x = d[var].astype(float)
    gm = x.groupby(d[group]).transform("mean")
    within = x - gm
    total_sd, within_sd = float(x.std(ddof=1)), float(within.std(ddof=1))
    return {"變數": var, "n": len(d), "總 SD": total_sd, "街區內 SD": within_sd,
            "街區內/總": within_sd / total_sd if total_sd else np.nan,
            "固定效果吸收掉的變異": 1 - (within_sd / total_sd) ** 2 if total_sd else np.nan,
            "平均值": float(x.mean())}


def main() -> None:
    p = config.path("final") / "poi_master.parquet"
    if not p.exists():
        raise SystemExit(f"{p} 不存在，請先跑 python -m dadaocheng.t04_features")
    df = pd.read_parquet(p)
    au = df[df["analysis_unit"]].copy()
    for c in ("in_heritage", "on_main_street", "north_of_minsheng"):
        if c in au:
            au[c] = au[c].astype("boolean").fillna(False).astype(bool)

    rule("0. 樣本與街區結構")
    print(f"poi_master {len(df)} 筆｜分析單位 {len(au)} 家｜街區 {au['block_id'].nunique()} 個")
    bs = au.groupby("block_id").size().sort_values()
    print(f"每街區 POI 數：最小 {bs.min()}、中位 {int(bs.median())}、最大 {bs.max()}")
    print("\n街區 × 業態大類 的細格數（H6 的交互項需要街區內的業態變異）：")
    ct = pd.crosstab(au["block_id"], au["places_category"])
    nonzero = (ct > 0).sum().sort_values(ascending=False)
    print(pd.DataFrame({"有該業態的街區數": nonzero,
                        "家數": au["places_category"].value_counts(),
                        "每街區平均": (au["places_category"].value_counts() / nonzero).round(1)}).to_string())
    small = int((ct.replace(0, np.nan).stack() < 5).sum())
    print(f"\n非空細格中少於 5 家的：{small} 個（細格太小時，固定效果加交互項會非常不穩）")

    rule("1. H6 的識別力：固定效果吃掉多少變異？（最關鍵）")
    rows = []
    for v in ["heritage_kde_150", "heritage_kde_300", "in_heritage", "dist_temple_m",
              "dist_mrt_m", "on_main_street", "bus_density"]:
        if v in au:
            r = within_between(au, v)
            if r:
                rows.append(r)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n判讀：")
    print("  「街區內/總」< 0.30 → 固定效果吃掉九成以上的變異，交互項幾乎沒有識別力。")
    print("  「街區內/總」> 0.60 → 變異主要在街區之內，固定效果可以放心用。")
    for _, r in t.iterrows():
        if r["變數"].startswith("heritage_kde") or r["變數"] == "in_heritage":
            ratio = r["街區內/總"]
            verdict = ("幾乎無法識別" if ratio < 0.3 else
                       "識別力偏弱" if ratio < 0.6 else "可以識別")
            print(f"  {r['變數']:<20} 街區內/總 = {ratio:.3f} → {verdict}")

    rule("2. H6 各業態的街區內變異（交互項是逐業態估計的）")
    rows = []
    for cat, g in au.groupby("places_category"):
        if cat == "其他" or len(g) < 20:
            continue
        for v in ["heritage_kde_150", "heritage_kde_300"]:
            r = within_between(g, v)
            if r:
                rows.append({"業態": cat, "家數": len(g), **{k: r[k] for k in ("變數", "街區內 SD", "街區內/總")}})
    if rows:
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    rule("3. 歷史密度與其他空間變數的相關（共線性）")
    cols = [c for c in ["heritage_kde_150", "heritage_kde_300", "dist_temple_m", "dist_mrt_m",
                        "on_main_street", "bus_density", "in_heritage"] if c in au]
    corr = au[cols].astype(float).corr()
    print(corr.round(3).to_string())
    print("\n若 heritage_kde 與 dist_temple_m／on_main_street 高度相關（|r| > 0.7），")
    print("即使不放固定效果，也很難把「歷史環境」和「位置本身」分開。")

    rule("4. H1：缺營業時間造成的上下界（Manski bounds）")
    n = len(au)
    has = au["closes_before_19"].notna()
    k = int(au.loc[has, "closes_before_19"].astype(bool).sum())
    n_has, n_miss = int(has.sum()), int((~has).sum())
    print(f"分析單位 {n} 家｜有營業時間 {n_has}（{n_has / n:.1%}）｜缺 {n_miss}（{n_miss / n:.1%}）")
    print(f"在有資料的樣本上：{k}/{n_has} = {k / n_has:.1%}（這是目前報的數字，假設缺失是隨機的）")
    lo, hi = k / n, (k + n_miss) / n
    print(f"最壞情況上下界：[{lo:.1%}, {hi:.1%}]"
          f"（缺的全部不早關 → 下界；全部早關 → 上界）")
    print(f"  50% 門檻{'在' if lo < 0.5 < hi else '不在'}上下界之內 →"
          f" {'「多數」的結論依賴缺失隨機的假設' if lo < 0.5 < hi else '結論對缺失型態穩健'}")
    print("\n缺營業時間的店長什麼樣（和有資料的比）：")
    cmp = pd.DataFrame({"缺營業時間": au.loc[~has, "places_category"].value_counts(normalize=True),
                        "有營業時間": au.loc[has, "places_category"].value_counts(normalize=True)})
    print((cmp * 100).round(1).to_string())
    print("\n來源與主街：")
    print(pd.DataFrame({"缺營業時間": [au.loc[~has, "source"].eq("osm").mean(),
                                       au.loc[~has, "on_main_street"].mean()],
                        "有營業時間": [au.loc[has, "source"].eq("osm").mean(),
                                       au.loc[has, "on_main_street"].mean()]},
                       index=["OSM 來源比例", "位於主街比例"]).round(3).to_string())

    rule("5. in_heritage 是怎麼判定的？（決定 H5 的可信度）")
    print(f"分析單位中 in_heritage：{int(au['in_heritage'].sum())} 家"
          f"（{au['in_heritage'].mean():.1%}）")
    hm_path = config.path("interim") / "heritage_match.csv"
    if hm_path.exists():
        hm = pd.read_csv(hm_path)
        hm = hm[hm["poi_id"].isin(au["poi_id"])]
        only_dist = hm[(~hm["in_footprint"].astype(bool)) & (~hm["address_match"].astype(bool))]
        print(f"配對組數 {len(hm)}｜其中只靠距離（沒有輪廓也沒有地址）{len(only_dist)}"
              f"（{len(only_dist) / len(hm):.1%}）")
        if "dist_m" in hm:
            d = hm["dist_m"].dropna()
            print(f"配對距離分布：中位 {d.median():.1f}、平均 {d.mean():.1f} 公尺")
            for cut in (5, 8, 10, 12, 15):
                pois = hm.loc[hm["dist_m"] <= cut, "poi_id"].nunique()
                extra = hm.loc[hm["in_footprint"].astype(bool) | hm["address_match"].astype(bool), "poi_id"]
                total = len(set(hm.loc[hm["dist_m"] <= cut, "poi_id"]) | set(extra))
                print(f"  門檻 {cut:>2} 公尺 → in_heritage {total} 家（{total / len(au):.1%}）")
        print("\n大稻埕街屋店面寬約 4–5 公尺。門檻愈大，愈會把左右鄰棟算成「在歷建內」。")
        print("上面這張表就是 H5 的敏感度分析：若各門檻下方向一致，證偽的結論才站得住。")
    else:
        print(f"（找不到 {hm_path}）")

    rule("5b. H5 的反轉是真的嗎？在歷建內 vs 在主街上，哪一個在驅動")
    if {"in_heritage", "on_main_street", "places_category"} <= set(au.columns):
        au["_hot"] = au["places_category"].eq("熱食餐飲")
        au["_cafe"] = au["places_category"].eq("咖啡茶飲甜點")
        print("in_heritage × on_main_street 的交叉表：")
        print(pd.crosstab(au["in_heritage"], au["on_main_street"],
                          rownames=["在歷建內"], colnames=["在主街"]).to_string())
        phi = float(au["in_heritage"].astype(float).corr(au["on_main_street"].astype(float)))
        print(f"\n兩者相關係數 = {phi:.3f}"
              + ("　← 高度相關：『在歷建內』可能只是『在主街上』的代理" if abs(phi) > 0.4 else
                 "　← 相關不高，兩者可以分開" if abs(phi) < 0.25 else "　← 中度相關，需要分層看"))
        print("\n熱食比例，依『是否主街』分層（分層後若差異消失，代表原本的反轉來自主街）：")
        tab = (au.groupby(["on_main_street", "in_heritage"])
                 .agg(家數=("poi_id", "size"), 熱食比例=("_hot", "mean"), 咖啡比例=("_cafe", "mean")))
        print(tab.round(3).to_string())
        for ms in (False, True):
            sub = au[au["on_main_street"] == ms]
            if sub["in_heritage"].nunique() == 2:
                a = int((sub["in_heritage"] & sub["_hot"]).sum()); b = int((sub["in_heritage"] & ~sub["_hot"]).sum())
                c = int((~sub["in_heritage"] & sub["_hot"]).sum()); d = int((~sub["in_heritage"] & ~sub["_hot"]).sum())
                from dadaocheng.stats import fisher_lower
                f = fisher_lower(a, b, c, d)
                lab = "主街" if ms else "巷弄"
                print(f"  {lab}：歷建內 {f['p_in']:.1%}（n={a + b}）對 歷建外 {f['p_out']:.1%}（n={c + d}）"
                      f"｜單尾 p = {f['p']:.3f}")
        print("\n咖啡茶飲甜點的比例差如果比熱食更大，支持『老屋再利用』的解釋：")
        print("  修復後的街屋被改成咖啡與餐飲空間，而不是『法規限制沒有作用』。")

    rule("6. 業態分類的粗糙度（決定 H6 要不要等 LLM）")
    vc = au["places_category"].value_counts()
    print(vc.to_string())
    print(f"\n「其他」{vc.get('其他', 0)} 家（{vc.get('其他', 0) / len(au):.1%}）"
          "，依預先登錄排除於 H6 → H6 的樣本會少掉這些。")
    print(f"H6 可用樣本：{len(au) - vc.get('其他', 0)} 家")
    if "traditional_subtype" in au:
        print(f"traditional_subtype 有值：{int(au['traditional_subtype'].notna().sum())} 家"
              f"（中藥 {int(au['traditional_subtype'].eq('中藥').sum())}、"
              f"南北貨 {int(au['traditional_subtype'].eq('南北貨').sum())}、"
              f"布業 {int(au['traditional_subtype'].eq('布業').sum())}、"
              f"茶行 {int(au['traditional_subtype'].eq('茶行').sum())}）")

    rule("7. H3(b) 文創的定義有多寬")
    cult = au["places_category"].isin(["選物與文創零售", "文化空間"])
    print(f"以 Places 類別定義的「文創」：{int(cult.sum())} 家（{cult.mean():.1%} 的分析單位）")
    print("這個比例若明顯偏高，代表 Places 的 store／gift_shop 把一般零售也收進來了，")
    print("H3(b) 等於用一個很鬆的定義在檢定，閘門一後應改用 LLM 的 wenqing_group 重跑。")
    if "likely_wenqing" in au:
        print(f"對照：likely_wenqing（T07 分層用的粗估）{au['likely_wenqing'].mean():.1%}")

    rule("8. 文本可得性對文青定義的影響")
    wp = config.path("interim") / "web_text.parquet"
    if wp.exists():
        web = pd.read_parquet(wp)
        m = au.merge(web[["poi_id", "text_status", "reason"]], on="poi_id", how="left")
        m["text_status"] = m["text_status"].fillna("name_only")
        print("分析單位的文本狀態：")
        print(m["text_status"].value_counts(normalize=True).round(3).to_string())
        print("\n依提案書 6.4，文青有兩條路：")
        print("  (1) 獨立咖啡或茶空間 → 只需店名與類別，name_only 也判得出來")
        print("  (2) 白名單業態 ＋ 至少一項風格指標 → **風格指標需要文本**")
        cafe = m["places_category"].eq("咖啡茶飲甜點")
        wl = m["places_category"].isin(["選物與文創零售", "文化空間"])
        with_text = m["text_status"].eq("website")
        print(f"\n走第 (1) 條的候選（咖啡茶飲甜點）：{int(cafe.sum())} 家")
        print(f"走第 (2) 條的候選（白名單業態）：{int(wl.sum())} 家，"
              f"其中有文本的只有 {int((wl & with_text).sum())} 家"
              f"（{(wl & with_text).sum() / max(wl.sum(), 1):.1%}）")
        print("→ 第 (2) 條路上的店家，絕大多數無法取得風格指標，會被判為非文青。")
        print("  這不是 LLM 的錯，是資料可得性造成的系統性低估，必須寫進限制。")
    else:
        print(f"（找不到 {wp}，請先跑 T05）")

    print("\n" + "=" * 72)
    print("把這份輸出整段貼回對話，就能決定 H6 的模型與 H5 的門檻。")
    print("=" * 72)


if __name__ == "__main__":
    # Windows PowerShell 的 > 會用主控台編碼重新編碼輸出，中文會變亂碼。
    # 用 --out 讓程式自己以 UTF-8 寫檔：python scripts/xxx.py --out outputs/tables/xxx.txt
    import argparse
    import contextlib

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--out", default=None, help="以 UTF-8 寫入這個檔案（取代 PowerShell 的 >）")
    _a = _ap.parse_args()
    if _a.out:
        Path(_a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(_a.out, "w", encoding="utf-8-sig") as _f, contextlib.redirect_stdout(_f):
            main()
        print(f"已寫入 {_a.out}")
    else:
        main()
