"""T09：Gemini 交叉檢查（補充指標，不是閘門依據）。

規格（實作規劃 B T09）：
- 在 120 筆抽查樣本與約 300 筆隨機 POI 上，以 Gemini 跑同一份 prompt 一次。
- 計算與 Claude 的一致率。
- 免費方案有每分鐘與每日上限：遇 429 要退避重試，不要並行轟炸（clients.gemini_send_factory 已處理）。
- 只送公開的店家文本（CLAUDE.md 規則 8）：輸入只有店名、Places 類別與官網正文，
  和 Claude 完全相同，不含訪談、個資或 Google 評論。

**兩個模型一致不等於正確**（提案書第九節）：兩個模型可能一起錯。這裡算出來的一致率
只用來找出「容易判錯」的點位，正確率仍以 T10 的人工抽查為準。

用法：
  python -m dadaocheng.t09_gemini_crosscheck --run
  python -m dadaocheng.t09_gemini_crosscheck --report
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config
from .extraction import (FIELDS, KEY_FIELDS, apply_invalid_evidence, derive_wenqing_group, json_schema,
                         system_prompt, user_content, validate_extraction)

COMPARE_FIELDS = ["category", "fine_type", "is_chain", "history_narrative", "customer_type"]


def gemini_request(name, places_types, text_status, text, model: str) -> dict:
    """同一份系統指令與欄位定義，只是改用 Gemini 的格式。"""
    prompt = system_prompt() + "\n\n----\n\n" + user_content(name, places_types, text_status, text)
    return {"model": model, "contents": prompt,
            "config": {"response_mime_type": "application/json", "temperature": 0,
                       "response_json_schema": json_schema()}}


def select_pois(n_random: int, seed_offset: int = 9) -> pd.DataFrame:
    from .t08_batch_extract import load_inputs

    poi = load_inputs()
    audit_path = config.path("final") / "audit_sample.csv"
    ids = set()
    if audit_path.exists():
        ids |= set(pd.read_csv(audit_path, dtype={"poi_id": str})["poi_id"])
    else:
        print("▲ 還沒有抽查樣本（T07），只跑隨機 POI")
    pool = poi[poi["in_main"] & ~poi["poi_id"].isin(ids)]
    rng = config.rng(seed_offset)
    extra = pool.sample(min(n_random, len(pool)), random_state=int(rng.integers(0, 2**31 - 1)))
    sel = pd.concat([poi[poi["poi_id"].isin(ids)], extra]).drop_duplicates("poi_id")
    sel = sel.copy()
    sel["in_audit"] = sel["poi_id"].isin(ids)
    return sel.reset_index(drop=True)


def run(sel: pd.DataFrame, caller, send, model: str) -> pd.DataFrame:
    rows = []
    for i, (_, r) in enumerate(sel.iterrows(), 1):
        req = gemini_request(r["name"], r["places_types"], r["text_status"], r["text"], model)
        src = user_content(r["name"], r["places_types"], r["text_status"], r["text"])
        rec = {"poi_id": r["poi_id"], "in_audit": bool(r["in_audit"]), "text_status": r["text_status"]}
        try:
            resp = caller.call(req, send=send, estimate_cost=lambda _r: 0.0, extra={"model": model})
            flat = apply_invalid_evidence(validate_extraction(resp["text"], src, r["text_status"]))
            rec.update({f"g_{k}": v for k, v in flat.items() if k in FIELDS})
            rec["g_wenqing_group"] = derive_wenqing_group(flat)
            rec["error"] = ""
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(rec)
        if i % 25 == 0:
            print(f"  [{i}/{len(sel)}]")
    return pd.DataFrame(rows)


def agreement(gem: pd.DataFrame, ext: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = gem.merge(ext, on="poi_id", how="inner", suffixes=("", "_claude"))
    rows = []
    for f in COMPARE_FIELDS + ["wenqing_group"]:
        gc, cc = f"g_{f}", f
        if gc not in m or cc not in m:
            continue
        ok = m[gc].notna() & m[cc].notna()
        agree = (m.loc[ok, gc].astype(str) == m.loc[ok, cc].astype(str))
        rows.append({"欄位": f, "可比對筆數": int(ok.sum()),
                     "一致率": float(agree.mean()) if ok.any() else np.nan,
                     "抽查樣本上的一致率": float(agree[m.loc[ok, "in_audit"]].mean())
                     if (ok & m["in_audit"]).any() else np.nan})
    # 兩模型不一致的點位：T10 報表會標示出來，供人優先檢查
    dis = m[[c for c in ["poi_id", "in_audit", "text_status"] if c in m]].copy()
    for f in COMPARE_FIELDS + ["wenqing_group"]:
        if f"g_{f}" in m and f in m:
            dis[f"{f}_claude"] = m[f]
            dis[f"{f}_gemini"] = m[f"g_{f}"]
            dis[f"{f}_同"] = m[f].astype(str) == m[f"g_{f}"].astype(str)
    same_cols = [c for c in dis.columns if c.endswith("_同")]
    dis["不一致欄位數"] = (~dis[same_cols].astype(bool)).sum(axis=1) if same_cols else 0
    return pd.DataFrame(rows), dis.sort_values("不一致欄位數", ascending=False)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--n-random", type=int, default=300)
    args = ap.parse_args(argv)

    out = config.path("interim") / "gemini_crosscheck.parquet"
    if args.run:
        from .api_layer import ApiCaller, default_ledger
        from .clients import gemini_send_factory

        model = config.settings()["models"]["gemini_crosscheck"]
        ledger = default_ledger()
        caller = ApiCaller("gemini", config.path("llm_cache") / "gemini", ledger)
        send = gemini_send_factory(config.require_env("GEMINI_API_KEY"), model)
        sel = select_pois(args.n_random)
        print(f"交叉檢查 {len(sel)} 家（抽查樣本 {int(sel['in_audit'].sum())}、隨機 {len(sel) - int(sel['in_audit'].sum())}）"
              f"，模型 {model}")
        gem = run(sel, caller, send, model)
        gem.to_parquet(out, index=False)
        n_err = int((gem["error"] != "").sum())
        print(f"完成：{len(gem)} 筆，失敗 {n_err} 筆")
        if n_err:
            top = gem.loc[gem["error"] != "", "error"].value_counts().head(3)
            print("\n最常見的錯誤訊息（請整段貼給 Claude）：")
            for msg, k in top.items():
                print(f"  [{k} 筆] {msg}")
            if n_err == len(gem):
                print("\n▲ 全部失敗，通常不是資料問題，而是設定問題。最常見的三個原因：\n"
                      "  1. .env 沒有 GEMINI_API_KEY，或金鑰貼錯（錯誤訊息會有 API key／401／403）\n"
                      "  2. 模型名稱不存在：config/settings.yaml 的 models.gemini_crosscheck"
                      "（錯誤訊息會有 404／not found），請到 AI Studio 確認免費方案的模型名稱\n"
                      "  3. 免費額度用完或速率限制（429／RESOURCE_EXHAUSTED）：隔天再跑\n"
                      "  API 呼叫失敗的不會寫進快取，修好設定後直接重跑 --run 即可。")
        print(ledger.summary())

    gem = pd.read_parquet(out)
    ep = config.path("final") / "extractions.parquet"
    if not ep.exists():
        print("還沒有 extractions.parquet（T08 --build），只輸出 Gemini 原始結果。")
        return
    ext = pd.read_parquet(ep)
    agr, dis = agreement(gem[gem["error"] == ""], ext)
    tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
    agr.to_csv(tables / "t09_agreement.csv", index=False, encoding="utf-8-sig")
    dis.to_csv(config.path("interim") / "t09_disagreements.csv", index=False, encoding="utf-8-sig")
    print(agr.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n兩個模型一致不等於正確：兩個模型可能一起錯。")
    print("這個數字只用來找出容易判錯的點位，正確率以 T10 的人工抽查為準。")
    print(f"不一致清單：{config.path('interim') / 't09_disagreements.csv'}")


if __name__ == "__main__":
    main()
