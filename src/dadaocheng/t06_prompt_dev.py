"""T06：抽取 prompt 在 20 筆開發樣本上迭代。

規格（實作規劃 B T06）：
- 系統指令＋欄位定義＋值域放前面（prompt caching），POI 名稱、Places 類別、文本放最後。
- 結構化輸出，每個欄位都有「不明」；證據逐字引用。
- 只依提供的文本判斷；name_only 只能判斷 category 與 is_chain（本專案另加 fine_type），其他一律「不明」。
- 驗收：20 筆由黃旭寬與游采頻看過；證據片段有效率 ≥ 90%；打 tag prompt-frozen（人做）。
- 開發樣本與抽查樣本不能重疊：T07 抽樣時排除 dev_sample.csv。
- 本模組**不讀** audit_sample.csv 或任何人工標註（規則 3）。

用法：
  python -m dadaocheng.t06_prompt_dev --sample          # 抽 20 筆開發樣本（固定種子），寫 data/final/dev_sample.csv
  python -m dadaocheng.t06_prompt_dev --show-prompt     # 印出第一筆的完整請求（不呼叫 API）
  python -m dadaocheng.t06_prompt_dev --run             # 呼叫 Claude（每筆 1 次），輸出審閱表
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .extraction import FIELDS, build_request, derive_wenqing_group, prompt_version, user_content, validate_extraction


def load_inputs() -> pd.DataFrame:
    poi = pd.read_parquet(config.path("final") / "poi_master.parquet")
    web = pd.read_parquet(config.path("interim") / "web_text.parquet")
    df = poi.merge(web[["poi_id", "text_status", "text", "reason"]], on="poi_id", how="left")
    df["text_status"] = df["text_status"].fillna("name_only")
    df["text"] = df["text"].fillna("")
    return df


def draw_dev_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """從分析單位中抽 n 筆：一半有官網文本、一半 name_only；有文本的那一半再讓「可能文青」至少占一半。

    這是為了讓 prompt 在困難案例上也被看過，不是為了估計任何比例（估計用 T07 的隨機抽查）。
    """
    au = df[df["analysis_unit"]].copy()
    rng = np.random.default_rng(seed)
    web = au[au["text_status"] == "website"]
    no = au[au["text_status"] != "website"]
    n_web = min(len(web), n // 2 + n % 2)
    n_no = min(len(no), n - n_web)
    n_web = min(len(web), n - n_no)
    lw = web[web["likely_wenqing"] == True]  # noqa: E712
    k = min(len(lw), (n_web + 1) // 2)
    pick_lw = lw.sample(k, random_state=rng.integers(1e9)) if k else lw.iloc[:0]
    rest = web.drop(pick_lw.index)
    pick_web = rest.sample(min(len(rest), n_web - k), random_state=rng.integers(1e9))
    pick_no = no.sample(n_no, random_state=rng.integers(1e9))
    out = pd.concat([pick_lw, pick_web, pick_no])
    return out[["poi_id", "name", "places_types", "text_status"]].assign(purpose="prompt_dev")


def source_text(row) -> str:
    return user_content(row["name"], row["places_types"], row["text_status"], row["text"])


def run_dev(df: pd.DataFrame, caller, send, estimate, actual, approve_step: bool = False) -> pd.DataFrame:
    reqs = [build_request(r["name"], r["places_types"], r["text_status"], r["text"]) for _, r in df.iterrows()]
    est_total = sum(estimate(q) for q in reqs)
    caller.ledger.check_step("anthropic", est_total, step=f"T06 開發樣本 {len(reqs)} 筆", approved=approve_step)
    pv = prompt_version()
    rows = []
    from .clients import anthropic_response_text

    for (_, r), req in zip(df.iterrows(), reqs):
        resp = caller.call(req, send=send, estimate_cost=estimate, actual_cost=actual, extra={"run": 0, "purpose": "dev"})
        rec = {"poi_id": r["poi_id"], "name": r["name"], "text_status": r["text_status"], "prompt_version": pv,
               "stop_reason": resp.get("stop_reason")}
        try:
            flat = validate_extraction(anthropic_response_text(resp), source_text(r), r["text_status"])
            rec.update(flat)
            rec["wenqing_group"] = derive_wenqing_group(flat)
            rec["parse_error"] = ""
        except Exception as e:  # noqa: BLE001
            rec["parse_error"] = f"{type(e).__name__}: {e}"[:300]
        rows.append(rec)
    return pd.DataFrame(rows)


def evidence_valid_rate(res: pd.DataFrame) -> float:
    cols = [f"evidence_valid_{f}" for f in FIELDS if f"evidence_valid_{f}" in res]
    vals = pd.concat([res[c] for c in cols]).dropna()
    return float(vals.astype(bool).mean()) if len(vals) else float("nan")


def review_markdown(res: pd.DataFrame, inputs: pd.DataFrame) -> str:
    """給黃旭寬與游采頻看的審閱表：每筆列出輸入摘要、各欄位答案、證據與是否為子字串。"""
    lines = [f"# T06 開發樣本審閱（prompt 版本 {res['prompt_version'].iloc[0] if len(res) else ''}）", "",
             f"證據片段有效率：**{evidence_valid_rate(res):.1%}**（目標 ≥ 90%）", "",
             "請逐筆標記：答案錯誤的欄位、證據不支持答案的欄位、prompt 需要補的定義。", ""]
    by = inputs.set_index("poi_id")
    for _, r in res.iterrows():
        lines.append(f"## {r['poi_id']}　{r['name']}（{r['text_status']}）")
        txt = str(by.loc[r["poi_id"], "text"])[:300].replace("\n", " ") if r["poi_id"] in by.index else ""
        lines.append(f"> 文本開頭：{txt or '（無）'}")
        if r.get("parse_error"):
            lines.append(f"**解析失敗：** {r['parse_error']}")
            lines.append("")
            continue
        lines.append("")
        lines.append("| 欄位 | 答案 | 證據 | 子字串？ | 審閱意見 |")
        lines.append("|---|---|---|---|---|")
        for f in FIELDS:
            ok = r.get(f"evidence_valid_{f}")
            mark = "—" if ok is None or (isinstance(ok, float) and np.isnan(ok)) else ("✔" if ok else "✘")
            ev = str(r.get(f"evidence_{f}", "") or "").replace("|", "／")
            lines.append(f"| {f} | {r.get(f)} | {ev} | {mark} |  |")
        lines.append(f"| （推導）wenqing_group | {r.get('wenqing_group')} |  |  |  |")
        if r.get("name_only_violations"):
            lines.append(f"\n⚠ name_only 卻填了：{r['name_only_violations']}（已強制改為不明）")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", action="store_true")
    g.add_argument("--show-prompt", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--approve-step", action="store_true")
    args = ap.parse_args(argv)
    s = config.settings()
    dev_path = config.project_root() / s["extraction"]["dev_sample_file"]
    df = load_inputs()
    if args.sample:
        if dev_path.exists():
            print(f"{dev_path} 已存在，不覆寫（避免開發樣本被換掉）。要重抽請先手動刪除並記錄原因。")
            return
        dev = draw_dev_sample(df, int(s["dev_n"]), int(s["seed"]) + 6)
        dev_path.parent.mkdir(parents=True, exist_ok=True)
        dev.to_csv(dev_path, index=False, encoding="utf-8-sig")
        print(f"已抽 {len(dev)} 筆開發樣本 → {dev_path}")
        print(dev["text_status"].value_counts().to_string())
        return
    dev_ids = pd.read_csv(dev_path, dtype=str)["poi_id"]
    sub = df[df["poi_id"].isin(dev_ids)].reset_index(drop=True)
    if args.show_prompt:
        r = sub.iloc[0]
        print(json.dumps(build_request(r["name"], r["places_types"], r["text_status"], r["text"]), ensure_ascii=False, indent=2))
        print("\nprompt 版本：", prompt_version())
        return
    from .clients import anthropic_caller

    caller, send, est, act = anthropic_caller()
    res = run_dev(sub, caller, send, est, act, approve_step=args.approve_step)
    tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
    res.to_csv(tables / f"t06_dev_{prompt_version()}.csv", index=False, encoding="utf-8-sig")
    md = tables / f"t06_dev_review_{prompt_version()}.md"
    md.write_text(review_markdown(res, sub), encoding="utf-8")
    rate = evidence_valid_rate(res)
    print(f"prompt 版本：{prompt_version()}；解析失敗：{(res['parse_error'] != '').sum()} 筆；證據有效率：{rate:.1%}")
    print("（≥ 90% 且兩位審閱者同意後，由人打 tag prompt-frozen）" if rate >= s["extraction"]["evidence_valid_target"] else "未達 90%：修改 prompts/ 後另存新版本檔再跑。")
    print(f"審閱表：{md}")
    print(caller.ledger.summary())


if __name__ == "__main__":
    main()
