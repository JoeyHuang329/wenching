"""T08：全量批次抽取（Anthropic Message Batches）。

規格（實作規劃 B T08）：
- 對所有 in_main 與緩衝區的 POI 跑三次抽取；custom_id 為 {poi_id}_{run}。
- 結果寫入 llm_cache/；以 pydantic 驗證；三次多數決；檢查證據片段是否為子字串。
- 驗收：失敗或格式錯誤 < 2%（失敗的以同步呼叫補跑）；產出 extractions。
- 批次是非同步的，可能要好幾小時。11/10 09:00 仍沒完成就把剩下的改用同步呼叫並行跑。

設計重點：
- **三次的快取鍵不同**（extra={"run": k}）。不這樣做的話，三次會讀到同一筆快取，
  「自我一致」就變成同一個答案抄三遍。
- 已經在快取裡的 (poi_id, run) 不會再送出，所以中斷、補跑、重跑都不會重複付費。
- 送出前先用 Ledger.check_step() 估算整批花費；超過 200 元會要求人確認（CLAUDE.md）。

用法：
  python -m dadaocheng.t08_batch_extract --estimate              # 只估花費與筆數
  python -m dadaocheng.t08_batch_extract --submit [--approve-step]
  python -m dadaocheng.t08_batch_extract --poll                  # 查進度，完成就取回並寫快取
  python -m dadaocheng.t08_batch_extract --fill-sync             # 剩下的改用同步呼叫補跑
  python -m dadaocheng.t08_batch_extract --build                 # 從快取組出 extractions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .clients import anthropic_actual_cost_factory, anthropic_estimate_factory, anthropic_response_text
from .extraction import (FIELDS, apply_invalid_evidence, build_request, derive_wenqing_group, majority_vote,
                         prompt_version, user_content, validate_extraction)

STATE = "batch_state.json"


def state_path() -> Path:
    return config.path("llm_cache") / STATE


def load_state() -> dict:
    p = state_path()
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"batches": []}


def save_state(st: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 輸入
# ---------------------------------------------------------------------------

def load_inputs(include_buffer: bool = True) -> pd.DataFrame:
    poi = pd.read_parquet(config.path("final") / "poi_master.parquet")
    if not include_buffer:
        poi = poi[poi["in_main"]]
    # 緩衝區的 POI 只用來算鄰域特徵，但文青比例等鄰域特徵需要它們的抽取結果
    poi = poi[poi["business_status"].isin(["OPERATIONAL", "UNKNOWN_OSM"]) | poi["in_main"]]
    wp = config.path("interim") / "web_text.parquet"
    if wp.exists():
        web = pd.read_parquet(wp)[["poi_id", "text_status", "text"]]
        poi = poi.merge(web, on="poi_id", how="left")
    else:
        poi["text_status"], poi["text"] = "name_only", ""
    poi["text_status"] = poi["text_status"].fillna("name_only")
    poi["text"] = poi["text"].fillna("")
    return poi.reset_index(drop=True)


RETRY_TAG = "max_tokens"


def retry_request(req: dict) -> dict:
    r = dict(req)
    r["max_tokens"] = int(config.settings()["models"].get("extraction_max_tokens_retry", 4000))
    return r


def retry_extra(run: int) -> dict:
    return {"run": int(run), "retry": RETRY_TAG}


def best_cached(caller, req: dict, run: int):
    """有截斷重跑的結果就用它，否則用原本的。"""
    rr = caller.cached(retry_request(req), extra=retry_extra(run))
    return rr if rr is not None else caller.cached(req, extra={"run": run})


def truncated_items(poi: pd.DataFrame, caller, runs: int) -> list[dict]:
    """原本的回應被 max_tokens 截斷、而且還沒重跑過的 (poi_id, run)。"""
    out = []
    for _, r in poi.iterrows():
        req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
        for k in range(runs):
            rec = caller.cached_record(req, extra={"run": k})
            if rec is None or (rec.get("response") or {}).get("stop_reason") != "max_tokens":
                continue
            rq = retry_request(req)
            if caller.cached(rq, extra=retry_extra(k)) is None:
                out.append({"poi_id": r["poi_id"], "run": k, "request": rq, "extra": retry_extra(k)})
    return out


def pending_items(poi: pd.DataFrame, caller, runs: int) -> list[dict]:
    """還沒有快取的 (poi_id, run)。"""
    out = []
    for _, r in poi.iterrows():
        req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
        for k in range(runs):
            if caller.cached(req, extra={"run": k}) is None:
                out.append({"poi_id": r["poi_id"], "run": k, "request": req})
    return out


# ---------------------------------------------------------------------------
# 批次送出與取回
# ---------------------------------------------------------------------------

def submit(items: list[dict], client, chunk: int = 10000, retry: bool = False) -> list[dict]:
    """送出批次，回傳 [{batch_id, n, custom_ids}]。"""
    batches = []
    for i in range(0, len(items), chunk):
        part = items[i:i + chunk]
        reqs = [{"custom_id": f"{it['poi_id']}_{it['run']}", "params": it["request"]} for it in part]
        b = client.messages.batches.create(requests=reqs)
        batches.append({"batch_id": b.id, "n": len(part), "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "prompt_version": prompt_version(), "retry": RETRY_TAG if retry else None})
        print(f"已送出批次 {b.id}（{len(part)} 筆）")
    return batches


def poll(client, caller, poi: pd.DataFrame, actual_cost) -> dict:
    """查每個批次的狀態；ended 的就取回並寫進快取。"""
    st = load_state()
    index = {r["poi_id"]: r for _, r in poi.iterrows()}
    summary = {"ended": 0, "in_progress": 0, "stored": 0, "errors": 0}
    for b in st.get("batches", []):
        if b.get("collected"):
            summary["ended"] += 1
            continue
        info = client.messages.batches.retrieve(b["batch_id"])
        status = getattr(info, "processing_status", "")
        counts = getattr(info, "request_counts", None)
        print(f"批次 {b['batch_id']}：{status}" + (f"　{counts}" if counts else ""))
        if status != "ended":
            summary["in_progress"] += 1
            continue
        summary["ended"] += 1
        for item in client.messages.batches.results(b["batch_id"]):
            cid = item.custom_id
            poi_id, _, run = cid.rpartition("_")
            r = index.get(poi_id)
            if r is None:
                continue
            res_type = getattr(item.result, "type", "")
            if res_type != "succeeded":
                summary["errors"] += 1
                continue
            msg = item.result.message
            resp = msg.model_dump(mode="json") if hasattr(msg, "model_dump") else dict(msg)
            req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
            if b.get("retry"):
                req = retry_request(req)
                caller.store(req, resp, actual_cost(req, resp), extra=retry_extra(int(run)))
            else:
                caller.store(req, resp, actual_cost(req, resp), extra={"run": int(run)})
            summary["stored"] += 1
        b["collected"] = True
    save_state(st)
    return summary


# ---------------------------------------------------------------------------
# 從快取組出 extractions
# ---------------------------------------------------------------------------

def build_extractions(poi: pd.DataFrame, caller, runs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (extractions, 問題清單)。"""
    rows, problems = [], []
    pv = prompt_version()
    for _, r in poi.iterrows():
        req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
        src = user_content(r["name"], r["places_types"], r["text_status"], r["text"])
        parsed = []
        for k in range(runs):
            cached = best_cached(caller, req, k)
            if cached is None:
                problems.append({"poi_id": r["poi_id"], "run": k, "problem": "missing"})
                continue
            try:
                flat = validate_extraction(anthropic_response_text(cached), src, r["text_status"])
                parsed.append(apply_invalid_evidence(flat))
            except Exception as e:  # noqa: BLE001
                problems.append({"poi_id": r["poi_id"], "run": k, "problem": f"{type(e).__name__}: {e}"[:200]})
        if not parsed:
            continue
        mv = majority_vote(parsed)
        rec = {"poi_id": r["poi_id"], "text_status": r["text_status"], "n_runs": len(parsed),
               "model": config.settings()["models"]["extraction"], "prompt_version": pv, **mv}
        rec["wenqing_group"] = derive_wenqing_group(mv)
        rows.append(rec)
    ext = pd.DataFrame(rows)
    return ext, pd.DataFrame(problems, columns=["poi_id", "run", "problem"])


def quality_report(ext: pd.DataFrame, problems: pd.DataFrame, n_expected: int, runs: int) -> pd.DataFrame:
    total = n_expected * runs
    fail = len(problems)
    ev_cols = [f"evidence_valid_{f}" for f in FIELDS if f"evidence_valid_{f}" in ext]
    ev = pd.concat([ext[c] for c in ev_cols]).dropna() if ev_cols else pd.Series(dtype=float)
    rows = [
        {"指標": "預期呼叫數", "值": total},
        {"指標": "失敗或格式錯誤", "值": fail},
        {"指標": "失敗率", "值": fail / total if total else np.nan},
        {"指標": "驗收（< 2%）", "值": bool(total and fail / total < 0.02)},
        {"指標": "有結果的 POI", "值": len(ext)},
        {"指標": "三次完全一致（3/3）", "值": float((ext["agreement"] == f"{runs}/{runs}").mean()) if len(ext) else np.nan},
        {"指標": "證據片段有效率", "值": float(ev.astype(bool).mean()) if len(ev) else np.nan},
    ]
    if len(ext):
        rows.append({"指標": "文青比例（推導）", "值": float((ext["wenqing_group"] != "非文青").mean())})
        for f in ("category", "history_narrative"):
            if f in ext:
                rows.append({"指標": f"{f} 為「不明」的比例", "值": float(ext[f].astype(str).eq("不明").mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_status(client) -> None:
    """批次的時間資訊。Message Batches 最長 24 小時；到期沒處理的請求會變成 expired，不收費。"""
    from datetime import datetime, timezone

    st = load_state()
    now = datetime.now(timezone.utc)
    if not st.get("batches"):
        print("沒有已送出的批次。")
        return
    for b in st["batches"]:
        info = client.messages.batches.retrieve(b["batch_id"])
        c = info.request_counts
        created, expires = info.created_at, info.expires_at
        waited = (now - created).total_seconds() / 3600 if created else float("nan")
        left = (expires - now).total_seconds() / 3600 if expires else float("nan")
        done = c.succeeded + c.errored + c.canceled + c.expired
        total = done + c.processing
        print(f"{b['batch_id']}")
        print(f"  狀態 {info.processing_status}｜已完成 {done}/{total}"
              f"（成功 {c.succeeded}、錯誤 {c.errored}、過期 {c.expired}、取消 {c.canceled}）")
        print(f"  建立 {created:%m/%d %H:%M} UTC｜已等 {waited:.1f} 小時｜距離到期 {left:.1f} 小時"
              f"｜已取回 {'是' if b.get('collected') else '否'}")
    print("\n判讀：批次最長 24 小時。已完成仍是 0 而且已等超過 6 小時，請到 Console 的 Batches 頁面看，"
          "並考慮 --fill-sync --only-audit 先讓抽查樣本有結果。")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--submit", action="store_true")
    g.add_argument("--poll", action="store_true")
    g.add_argument("--fill-sync", action="store_true")
    g.add_argument("--build", action="store_true")
    g.add_argument("--retry-truncated", action="store_true",
                   help="只重跑輸出被 max_tokens 截斷的呼叫（上限改用 extraction_max_tokens_retry）")
    g.add_argument("--status", action="store_true", help="印出每個批次的建立時間、已等待多久、到期時間（不取回結果）")
    ap.add_argument("--approve-step", action="store_true")
    ap.add_argument("--sync", action="store_true", help="--retry-truncated 時用同步呼叫（立即完成，價格約批次的兩倍）")
    ap.add_argument("--only-audit", action="store_true",
                    help="--fill-sync 時只補抽查樣本（data/final/audit_sample.csv）的 POI，"
                         "批次卡住時先讓 T09／T10 能往下做")
    ap.add_argument("--main-only", action="store_true", help="只跑主範圍（緩衝區的鄰域特徵會不完整）")
    ap.add_argument("--limit", type=int, default=None, help="--fill-sync 時一次最多補幾筆")
    args = ap.parse_args(argv)

    s = config.settings()
    runs = int(s["llm_runs"])
    poi = load_inputs(include_buffer=not args.main_only)
    from .clients import anthropic_caller

    caller, send, est, act = anthropic_caller()
    batch_est = anthropic_estimate_factory(caller.ledger, batch=True)
    batch_act = anthropic_actual_cost_factory(caller.ledger, batch=True)

    if args.estimate or args.submit:
        items = pending_items(poi, caller, runs)
        total = sum(batch_est(it["request"]) for it in items)
        print(f"POI {len(poi)} 家 × {runs} 次 = {len(poi) * runs} 次呼叫；"
              f"已快取 {len(poi) * runs - len(items)} 次，待送出 {len(items)} 次")
        print(f"批次介面預估花費：約新台幣 {total:.0f} 元（同步呼叫約 {total * 2:.0f} 元）")
        print(f"prompt 版本：{prompt_version()}")
        if args.estimate:
            return
        if not items:
            print("沒有待送出的項目。")
            return
        caller.ledger.check_step("anthropic", total, step=f"T08 全量批次 {len(items)} 次", approved=args.approve_step)
        import anthropic

        client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        st = load_state()
        st["batches"] = st.get("batches", []) + submit(items, client)
        st["prompt_version"] = prompt_version()
        save_state(st)
        print("\n送出後請每隔一段時間跑：python -m dadaocheng.t08_batch_extract --poll")
        return

    if args.poll:
        import anthropic

        client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        summary = poll(client, caller, poi, batch_act)
        print(summary)
        print(caller.ledger.summary())
        left = len(pending_items(poi, caller, runs))
        print(f"仍未取得結果：{left} 次呼叫")
        if left == 0:
            print("全部取得 → python -m dadaocheng.t08_batch_extract --build")
        return

    if args.retry_truncated:
        items = truncated_items(poi, caller, runs)
        if args.only_audit:
            ids = set(pd.read_csv(config.path("final") / "audit_sample.csv", dtype={"poi_id": str})["poi_id"])
            items = [it for it in items if it["poi_id"] in ids]
        if not items:
            print("沒有需要重跑的截斷呼叫。")
            return
        n_poi = len({it["poi_id"] for it in items})
        e = est if args.sync else batch_est
        total = sum(e(it["request"]) for it in items)
        print(f"截斷的呼叫 {len(items)} 次（{n_poi} 家），max_tokens 改為 {items[0]['request']['max_tokens']}")
        print(f"{'同步' if args.sync else '批次'}預估新台幣 {total:.0f} 元（以滿上限估，實際通常更少）")
        caller.ledger.check_step("anthropic", total, step=f"T08 截斷重跑 {len(items)} 次", approved=args.approve_step)
        if args.sync:
            for i, it in enumerate(items, 1):
                try:
                    caller.call(it["request"], send=send, estimate_cost=est, actual_cost=act, extra=it["extra"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  ▲ {it['poi_id']} run {it['run']}：{type(exc).__name__}: {str(exc)[:120]}")
                if i % 50 == 0:
                    print(f"  [{i}/{len(items)}]")
            print(caller.ledger.summary())
            print("完成 → python -m dadaocheng.t08_batch_extract --build")
        else:
            import anthropic

            client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
            st = load_state()
            st["batches"] = st.get("batches", []) + submit(items, client, retry=True)
            save_state(st)
            print("已送出。之後跑 --poll，全部取回後 --build")
        return

    if args.status:
        import anthropic

        client = anthropic.Anthropic(api_key=config.require_env("ANTHROPIC_API_KEY"))
        print_status(client)
        return

    if args.fill_sync:
        items = pending_items(poi, caller, runs)
        if args.only_audit:
            ap_path = config.path("final") / "audit_sample.csv"
            if not ap_path.exists():
                raise SystemExit(f"{ap_path} 不存在，請先跑 T07 --draw")
            ids = set(pd.read_csv(ap_path, dtype={"poi_id": str})["poi_id"])
            items = [it for it in items if it["poi_id"] in ids]
            print(f"只補抽查樣本：{len(ids)} 家")
        if args.limit:
            items = items[: args.limit]
        if not items:
            print("沒有需要補跑的項目。")
            return
        total = sum(est(it["request"]) for it in items)
        print(f"以同步呼叫補跑 {len(items)} 次，預估新台幣 {total:.0f} 元（同步沒有批次五折）")
        caller.ledger.check_step("anthropic", total, step=f"T08 同步補跑 {len(items)} 次", approved=args.approve_step)
        for i, it in enumerate(items, 1):
            caller.call(it["request"], send=send, estimate_cost=est, actual_cost=act, extra={"run": it["run"]})
            if i % 25 == 0:
                print(f"  [{i}/{len(items)}]")
        print(caller.ledger.summary())
        return

    # --build
    ext, problems = build_extractions(poi, caller, runs)
    if ext.empty:
        raise SystemExit("快取裡還沒有任何抽取結果。請先 --submit 與 --poll。")
    out = config.path("final") / "extractions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ext.to_parquet(out, index=False)
    rep = quality_report(ext, problems, len(poi), runs)
    tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
    rep.to_csv(tables / "t08_quality.csv", index=False, encoding="utf-8-sig")
    if len(problems):
        problems.to_csv(config.path("interim") / "t08_problems.csv", index=False, encoding="utf-8-sig")
    print(rep.to_string(index=False))
    print(f"\nextractions：{out}（{len(ext)} 筆）")
    print("文青分群：", ext["wenqing_group"].value_counts().to_dict())
    print("業態大類：", ext["category"].value_counts().to_dict())
    miss = len(problems[problems["problem"] == "missing"]) if len(problems) else 0
    if miss:
        print(f"\n▲ 還有 {miss} 次呼叫沒有結果：--poll 或 --fill-sync")


if __name__ == "__main__":
    main()
