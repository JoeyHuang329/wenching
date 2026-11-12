"""T08 抽取品質診斷：失敗率 9.7%（驗收要 < 2%）與歷史敘事 99% 不明，原因在哪裡。

只讀快取與 extractions.parquet，不打任何 API、不花錢、不讀人工答案。

用法：
  python scripts/diagnose_t08.py > outputs/tables/diagnose_t08.txt
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dadaocheng import config, geo  # noqa: E402
from dadaocheng.clients import anthropic_caller, anthropic_response_text  # noqa: E402
from dadaocheng.extraction import build_request, user_content, validate_extraction  # noqa: E402
from dadaocheng.t08_batch_extract import load_inputs, retry_extra, retry_request  # noqa: E402


def kind(e: Exception) -> str:
    s = f"{type(e).__name__}: {e}"
    if type(e).__name__ == "JSONDecodeError":
        return "JSON 解析失敗"
    if "validation error" in s:
        # pydantic：取出是哪個欄位
        lines = [ln.strip() for ln in str(e).splitlines()]
        field = lines[1] if len(lines) > 1 else "?"
        return f"欄位驗證失敗：{field}"
    return s[:80]


def main() -> None:
    print("== 0. 邊界狀態 ==")
    print("主範圍邊界是 DRAFT：", geo.boundary_is_draft("boundary_main"))

    runs = int(config.settings()["llm_runs"])
    poi = load_inputs(include_buffer=True)
    caller, *_ = anthropic_caller()
    audit_p = config.path("final") / "audit_sample.csv"
    audit = set(pd.read_csv(audit_p, dtype={"poi_id": str})["poi_id"]) if audit_p.exists() else set()

    rows, examples, stops = [], {}, Counter()
    for _, r in poi.iterrows():
        req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
        src = user_content(r["name"], r["places_types"], r["text_status"], r["text"])
        for k in range(runs):
            rec = caller.cached_record(retry_request(req), extra=retry_extra(k)) \
                or caller.cached_record(req, extra={"run": k})       # 有截斷重跑就看重跑的
            if rec is None:
                rows.append({"poi_id": r["poi_id"], "text_status": r["text_status"], "問題": "沒有快取"})
                continue
            resp = rec["response"]
            stops[resp.get("stop_reason")] += 1
            try:
                validate_extraction(anthropic_response_text(resp), src, r["text_status"])
                prob = "ok"
            except Exception as e:  # noqa: BLE001
                prob = kind(e)
                if resp.get("stop_reason") == "max_tokens":
                    prob = "輸出被截斷（max_tokens）"
                examples.setdefault(prob, (r["name"], anthropic_response_text(resp)[-400:], str(e)[:400]))
            rows.append({"poi_id": r["poi_id"], "text_status": r["text_status"], "問題": prob,
                         "out_tokens": (resp.get("usage") or {}).get("output_tokens")})
    d = pd.DataFrame(rows)

    print(f"\n== 1. 失敗類型（共 {len(d)} 次呼叫）==")
    print(d["問題"].value_counts().to_string())
    print("\nstop_reason：", dict(stops))
    print("\n== 2. 失敗率 × 文本狀態 ==")
    print(pd.crosstab(d["text_status"], d["問題"].ne("ok"), normalize="index").round(3).to_string())
    print("\n輸出 token 數（ok vs 失敗）：")
    print(d.groupby(d["問題"].eq("ok"))["out_tokens"].describe().round(0).to_string())

    print("\n== 3. 每種失敗的一個例子（輸出結尾 400 字＋錯誤訊息）==")
    for prob, (name, tail, err) in examples.items():
        print(f"\n--- {prob} ｜ {name}\n輸出結尾：{tail}\n錯誤：{err}")

    print("\n== 4. 三次全失敗、完全沒有結果的 POI ==")
    per = d.assign(ok=d["問題"].eq("ok")).groupby("poi_id")["ok"].sum()
    dead = set(per[per == 0].index)
    print(f"{len(dead)} 家；其中抽查樣本 {len(dead & audit)} 家（這些在 T10 會變成缺值）")

    ep = config.path("final") / "extractions.parquet"
    if ep.exists():
        e = pd.read_parquet(ep)
        print("\n== 5. 歷史敘事 × 文本狀態（name_only 依規則一定是不明）==")
        print(pd.crosstab(e["text_status"], e["history_narrative"].fillna("（空）"), margins=True).to_string())
        if "history_narrative_raw" in e:
            web = e[e["text_status"] == "website"]
            n_inv = web["history_narrative_raw"].notna().sum()
            print(f"\nwebsite 店家中，歷史敘事因證據不是逐字子字串而被改成不明：{n_inv} / {len(web)}")
            print(web.loc[web["history_narrative_raw"].notna(), "history_narrative_raw"].value_counts().head(8).to_string())
        print("\n== 6. category × 文本狀態 ==")
        print(pd.crosstab(e["text_status"], e["category"], margins=True).to_string())
        print("\n== 7. 抽查樣本的覆蓋 ==")
        print(f"抽查 {len(audit)} 家，有抽取結果 {len(audit & set(e['poi_id']))} 家")


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
