"""RQ1 彙總：跑 H1、H3、H5（以及 H6 若已完成），套 Holm 校正，產出簡報可直接用的摘要。

為什麼要有這支：Holm 校正必須在同一個地方、以預先登錄的家族大小（H1、H3、H5、H6 共 4 條）計算。
各條假說分開跑時各自不知道別人的 p 值，校正只能在這裡做。

H6 還沒完成時，這裡會以「假設 H6 的 p 值最大」計算最保守的校正，並在報表中標明尚未完成。

輸出：
  outputs/tables/rq1_holm.csv
  outputs/tables/rq1_summary.md      ← 貼進簡報的文字
用法：python -m dadaocheng.report_rq1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .stats import PREREG_FAMILY, fmt_p, holm


def collect_pvalues() -> dict[str, dict]:
    """讀各假說自己寫出的 p 值檔。沒跑的就是缺。"""
    d = config.path("tables")
    out: dict[str, dict] = {}
    for h in PREREG_FAMILY:
        p = d / f"{h.lower()}_pvalue.csv"
        if p.exists():
            r = pd.read_csv(p).iloc[0].to_dict()
            out[h] = {"p": float(r["p_raw"]) if pd.notna(r["p_raw"]) else float("nan"),
                      "verdict": r.get("verdict", "")}
    return out


def summary_markdown(h: pd.DataFrame, found: dict[str, dict], au_n: int, n_blocks: int) -> str:
    done = [k for k in PREREG_FAMILY if k in found]
    missing = [k for k in PREREG_FAMILY if k not in found]
    display = {"H1": "假說一", "H3": "假說二", "H5": "假說三", "H6": "假說四"}   # 簡報與論文用的編號
    claims = {"H1": "夜間經濟弱，多數店家傍晚就打烊",
              "H3": "業種分區：主街中藥、巷弄文創、民生西路以北南北貨",
              "H5": "歷史建築限制熱食與明火",
              "H6": "歷史環境的關聯因業態而異（主問題）"}
    L = ["# RQ1 假說檢定結果（自動產生）", "",
         f"分析單位：主範圍內營業中的商業 POI **{au_n}** 家，分布在 **{n_blocks}** 個街區。",
         f"顯著水準 α = 0.05，Holm 校正以預先登錄的家族大小 m = {len(PREREG_FAMILY)} 計算"
         f"（{'、'.join(PREREG_FAMILY)}）。", ""]
    if missing:
        L += [f"> **尚未完成：{'、'.join(missing)}。** 下表的校正已假設未完成者的 p 值最大，"
              "因此是最保守的版本；補完之後校正後 p 只會變小或不變，不會變大。", ""]
    L += ["| 假說 | 程式代號 | 主張 | 原始 p | Holm 乘數 | 校正後 p | 結論 |", "|---|---|---|---|---|---|---|"]
    hi = h.set_index("假說")
    for k in PREREG_FAMILY:
        if k in found:
            r = hi.loc[k]
            L.append(f"| {display[k]} | {k} | {claims[k]} | {fmt_p(r['原始 p'])} | {int(r['Holm 乘數'])} | "
                     f"{fmt_p(r['校正後 p'])} | {found[k]['verdict']} |")
        else:
            L.append(f"| {display[k]} | {k} | {claims[k]} | — | — | — | 尚未完成 |")
    L += ["", "## 判讀時要注意的三件事", "",
          "1. **這些是關聯，不是因果。** 觀察資料只能說明模式，不能說明機制。",
          "2. **樣本只含現存店家（存活者偏誤）。** 訪談指出新店兩三年就換，現存不等於適配。",
          f"3. **街區數只有 {n_blocks} 個。** 群集標準誤在群集數少時會低估，"
          "H6 的信賴區間要以 wild cluster bootstrap 為準。"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-all", action="store_true", help="先依序跑 H1、H3、H5 再彙總")
    ap.add_argument("--with-h6", action="store_true", help="--run-all 時也跑 H6（野自助法較久）")
    ap.add_argument("--n-boot-h6", type=int, default=9999)
    args = ap.parse_args(argv)
    if args.run_all:
        from . import t11_h1_hours, t12_h5_heritage, t13_h3_zoning

        for mod, name in [(t11_h1_hours, "H1"), (t13_h3_zoning, "H3"), (t12_h5_heritage, "H5")]:
            print(f"\n{'=' * 28} {name} {'=' * 28}")
            mod.main([])
        if args.with_h6:
            from . import t14_h6_density

            print(f"\n{'=' * 28} H6 {'=' * 28}")
            t14_h6_density.main(["--n-boot", str(args.n_boot_h6)])

    found = collect_pvalues()
    if not found:
        raise SystemExit("還沒有任何假說的 p 值檔。請先跑 python -m dadaocheng.report_rq1 --run-all")
    h = holm({k: v["p"] for k, v in found.items()})
    au = pd.read_parquet(config.path("final") / "poi_master.parquet")
    au = au[au["analysis_unit"]]
    md = summary_markdown(h, found, len(au), int(au["block_id"].nunique()))

    d = config.path("tables"); d.mkdir(parents=True, exist_ok=True)
    h.to_csv(d / "rq1_holm.csv", index=False, encoding="utf-8-sig")
    (d / "rq1_summary.md").write_text(md, encoding="utf-8")
    print(f"\n{'=' * 28} Holm 校正 {'=' * 28}")
    print(h.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n簡報用摘要：{d / 'rq1_summary.md'}")
    print(f"Holm 表：{d / 'rq1_holm.csv'}")


if __name__ == "__main__":
    main()
