"""T10：計算 κ 與閘門一報表。

規格（實作規劃 B T10）：
- 三個關鍵欄位的 Cohen's κ 與準確率（自助法 95% 信賴區間）、混淆矩陣、證據片段有效率、
  name_only 子集與 website 子集分開報告。
- 分層抽樣時，準確率要依抽樣機率加權。
- **只產出報表，不宣布通過。** 閘門一由全組看著數字決定（CLAUDE.md）。

三個關鍵欄位（提案書 8.4）：
  category            業態大類，直接比對
  is_wenqing          是否文青。LLM 這邊是程式從 fine_type、is_chain、風格指標推導（規則 11），
                      人工是抽查員依同一套規則判斷（docs/audit_manual.md）
  history_narrative   歷史敘事。複選欄位，依 實作審閱筆記.md 2.1 折成「有／無／不明」三類

為什麼要同時看準確率：文青大約只占一到兩成，這種類別不平衡會壓低 κ（盛行率悖論）。
κ 偏低時，要能分辨是「錯得多」還是「類別不平衡」，所以報表並列準確率與各類別召回率。

用法：python -m dadaocheng.t10_kappa_gate
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config
from .extraction import UNK
from .stats import cohen_kappa, confusion_frame, fmt_pct, kappa_bootstrap_ci, weighted_accuracy

HIST_LABELS = ["有", "無", UNK]
YESNO = ["是", "否", UNK]


def collapse_history(v) -> str | None:
    """複選的歷史敘事折成三類。"""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    parts = {p.strip() for p in s.replace("、", ";").replace(",", ";").split(";") if p.strip()}
    if not parts or parts == {UNK}:
        return UNK
    if parts == {"無"}:
        return "無"
    return "有" if parts & {"建築", "街區", "家族"} else UNK


def normalise_yesno(v) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if s in ("是", "Y", "y", "yes", "True", "TRUE", "1"):
        return "是"
    if s in ("否", "N", "n", "no", "False", "FALSE", "0"):
        return "否"
    return UNK


def load_human() -> pd.DataFrame:
    """讀四份抽查表單，合併成一張人工答案表。"""
    d = config.path("final")
    stems = sorted({f.stem for f in d.glob("audit_form_*.csv")} | {f.stem for f in d.glob("audit_form_*.xlsx")})
    if not stems:
        raise FileNotFoundError(f"{d} 下沒有 audit_form_*.xlsx／.csv。請先跑 T07，並請抽查員填完。")

    def n_answers(t: pd.DataFrame) -> int:
        cols = [c for c in ["h_category", "h_is_wenqing", "h_history_narrative"] if c in t]
        return int(t[cols].fillna("").astype(str).apply(lambda s: s.str.strip().ne("")).any(axis=1).sum())

    parts = []
    for stem in stems:
        cands = []
        if (d / f"{stem}.xlsx").exists():
            from .t07_audit_sample import XLSX_COLS
            x = pd.read_excel(d / f"{stem}.xlsx", sheet_name="填寫", dtype=str)
            x.columns = [c for c, *_ in XLSX_COLS][: len(x.columns)]
            cands.append((f"{stem}.xlsx", x))
        if (d / f"{stem}.csv").exists():
            cands.append((f"{stem}.csv", pd.read_csv(d / f"{stem}.csv", dtype=str, encoding="utf-8-sig")))
        filled = [(name, t, n_answers(t)) for name, t in cands]
        if sum(1 for *_, k in filled if k > 0) > 1:
            print(f"▲ {stem} 的 Excel 與 CSV 都有填答，採用 Excel 版。")
        # Excel 有填就用 Excel；否則用有填的那個；都沒填就用第一個
        pick = next((x for x in filled if x[0].endswith(".xlsx") and x[2] > 0), None) \
            or next((x for x in filled if x[2] > 0), None) or filled[0]
        t = pick[1].copy()
        t["form"] = pick[0]
        parts.append(t)
    h = pd.concat(parts, ignore_index=True)
    h["h_category"] = h["h_category"].astype(str).str.strip().replace({"": None, "nan": None})
    h["h_is_wenqing"] = h["h_is_wenqing"].apply(normalise_yesno)
    h["h_history"] = h["h_history_narrative"].apply(collapse_history)
    sample = pd.read_csv(d / "audit_sample.csv", dtype={"poi_id": str})
    keep = [c for c in ["poi_id", "stratum", "sampling_prob", "text_status"] if c in sample]
    return h.merge(sample[keep], on="poi_id", how="left", suffixes=("", "_s"))


def load_llm() -> pd.DataFrame:
    p = config.path("final") / "extractions.parquet"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在。請先跑 T08 --build。")
    e = pd.read_parquet(p)
    e["l_category"] = e["category"]
    e["l_is_wenqing"] = np.where(e["wenqing_group"].eq("非文青"), "否", "是")
    e.loc[e["category"].astype(str).eq(UNK), "l_is_wenqing"] = UNK
    e["l_history"] = e["history_narrative"].apply(collapse_history)
    return e


FIELD_SPECS = [
    ("category", "h_category", "l_category", None),
    ("is_wenqing", "h_is_wenqing", "l_is_wenqing", YESNO),
    ("history_narrative", "h_history", "l_history", HIST_LABELS),
]


def evaluate(m: pd.DataFrame, subset: str = "全部", n_boot: int = 2000) -> tuple[pd.DataFrame, dict]:
    rows, mats = [], {}
    for name, hc, lc, labels in FIELD_SPECS:
        sub = m[m[hc].notna() & m[lc].notna()]
        if sub.empty:
            rows.append({"子集": subset, "欄位": name, "可比對筆數": 0, "κ": np.nan})
            continue
        res = cohen_kappa(sub[hc], sub[lc], labels)
        lo, hi = kappa_bootstrap_ci(sub[hc], sub[lc], labels, n_boot=n_boot)
        correct = sub[hc].astype(str) == sub[lc].astype(str)
        wacc = weighted_accuracy(correct, sub["sampling_prob"]) if "sampling_prob" in sub else np.nan
        rows.append({"子集": subset, "欄位": name, "可比對筆數": res["n"], "κ": res["kappa"],
                     "κ 95% 下界": lo, "κ 95% 上界": hi, "準確率": res["accuracy"],
                     "加權準確率": wacc, "達 κ ≥ 0.6": bool(res["kappa"] >= 0.6)})
        mats[f"{subset}｜{name}"] = confusion_frame(res)
    return pd.DataFrame(rows), mats


def per_class_recall(m: pd.DataFrame) -> pd.DataFrame:
    """各類別的召回率：κ 偏低時用來分辨「錯得多」還是「類別不平衡」。"""
    out = []
    for name, hc, lc, _ in FIELD_SPECS:
        sub = m[m[hc].notna() & m[lc].notna()]
        for lab, g in sub.groupby(hc):
            out.append({"欄位": name, "人工標為": lab, "筆數": len(g),
                        "LLM 也標為此類的比例": float((g[lc].astype(str) == str(lab)).mean())})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)

    human, llm = load_human(), load_llm()
    m = human.merge(llm, on="poi_id", how="inner", suffixes=("", "_llm"))
    ts = m["text_status"] if "text_status" in m else m.get("text_status_llm")
    m = m.assign(_ts=ts)

    n_filled = int(m[["h_category", "h_is_wenqing", "h_history"]].notna().any(axis=1).sum())
    print(f"抽查樣本 {len(human)} 筆，已填 {n_filled} 筆，與 LLM 可對照 {len(m)} 筆")
    if n_filled < len(human):
        print(f"▲ 還有 {len(human) - n_filled} 筆沒填。κ 會以已填的部分計算，"
              "但抽樣代表性會打折，請在報告中註明。")

    frames, mats = [], {}
    f_all, mt = evaluate(m, "全部", args.n_boot); frames.append(f_all); mats.update(mt)
    for ts_name in ["website", "name_only"]:
        sub = m[m["_ts"] == ts_name]
        if len(sub) >= 5:
            f, mm = evaluate(sub, ts_name, args.n_boot); frames.append(f); mats.update(mm)
        else:
            print(f"（{ts_name} 子集只有 {len(sub)} 筆，不單獨報告）")
    report = pd.concat(frames, ignore_index=True)

    ev_cols = [c for c in llm.columns if c.startswith("evidence_valid_")]
    ev = pd.concat([llm[c] for c in ev_cols]).dropna() if ev_cols else pd.Series(dtype=float)
    extra = pd.DataFrame([
        {"指標": "證據片段有效率（全量）", "值": float(ev.astype(bool).mean()) if len(ev) else np.nan},
        {"指標": "抽查員認為其他欄位合理的比例",
         "值": float(human["llm_fields_reasonable"].apply(normalise_yesno).eq("是").mean())},
        {"指標": "抽查員認為證據支持答案的比例",
         "值": float(human["evidence_reasonable"].apply(normalise_yesno).eq("是").mean())},
        {"指標": "抽查時有看 Google Maps 的比例",
         "值": float(human["basis"].astype(str).str.contains("maps").mean())},
    ])

    tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
    report.to_csv(tables / "t10_kappa.csv", index=False, encoding="utf-8-sig")
    extra.to_csv(tables / "t10_extra_metrics.csv", index=False, encoding="utf-8-sig")
    recall = per_class_recall(m); recall.to_csv(tables / "t10_per_class_recall.csv", index=False, encoding="utf-8-sig")
    with open(tables / "t10_confusion.md", "w", encoding="utf-8") as f:
        for k, t in mats.items():
            f.write(f"## {k}\n\n{t.to_markdown(index=False)}\n\n")

    print()
    print(report.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(extra.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(recall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    key = report[report["子集"] == "全部"]
    print("\n" + "=" * 64)
    print("閘門一判定所需的數字（三個關鍵欄位，門檻 κ ≥ 0.6）：")
    for _, r in key.iterrows():
        mark = "達標" if r.get("達 κ ≥ 0.6") else "未達"
        print(f"  {r['欄位']:<20} κ = {r['κ']:.3f} "
              f"[{r.get('κ 95% 下界', float('nan')):.3f}, {r.get('κ 95% 上界', float('nan')):.3f}]"
              f"　準確率 {fmt_pct(r['準確率'])}　{mark}")
    print("\n本程式只產出報表，不宣布通過。請全組一起看著數字決定（CLAUDE.md）。")
    print("未達標時依提案書 8.4：先改 schema 一次 → 再換 Claude Sonnet 5.5 一次 → 仍未達則啟動 8.8 備援方案。")
    print(f"\n報表：{tables / 't10_kappa.csv'}、{tables / 't10_confusion.md'}")


if __name__ == "__main__":
    main()
