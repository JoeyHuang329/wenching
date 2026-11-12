"""T07–T10：抽查樣本、批次抽取、Gemini 交叉檢查、κ 閘門報表。外部服務全部用假的。"""

from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

from dadaocheng.extraction import FIELDS


def make_master(n=400, seed=0, wenqing_share=0.10):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        lw = rng.random() < wenqing_share
        rows.append({
            "poi_id": f"P{i:05d}", "place_id": f"pl{i}", "name": f"店{i}",
            "places_types": '["cafe"]' if lw else '["store"]',
            "places_primary_type": "cafe" if lw else "store",
            "analysis_unit": True, "in_main": True, "business_status": "OPERATIONAL",
            "likely_wenqing": lw, "block_id": f"B{i % 12:02d}",
            "lon": 121.509 + rng.normal(0, 0.001), "lat": 25.057 + rng.normal(0, 0.002),
            "places_category": "咖啡茶飲甜點" if lw else "一般零售與品牌",
        })
    return pd.DataFrame(rows)


def write_master(root, df, web=None):
    (root / "data" / "final").mkdir(parents=True, exist_ok=True)
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)
    df.to_parquet(root / "data" / "final" / "poi_master.parquet", index=False)
    if web is not None:
        web.to_parquet(root / "data" / "interim" / "web_text.parquet", index=False)
    return df


def make_web(df, website_frac=0.4, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df.iterrows():
        has = rng.random() < website_frac
        rows.append({"poi_id": r["poi_id"], "text_status": "website" if has else "name_only",
                     "text": f"{r['name']}位於大稻埕的老屋，販售手作選物。" * 3 if has else ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# T07
# ---------------------------------------------------------------------------

def test_t07_stratifies_and_records_probability(tmp_root):
    from dadaocheng.t07_audit_sample import draw, load_frame

    df = make_master(wenqing_share=0.10)
    write_master(tmp_root, df, make_web(df))
    au = load_frame()
    sample, log = draw(au, 120, dev_ids=set())
    assert len(sample) == 120
    assert set(log["stratum"]) == {"可能文青", "其他"}
    # 「可能文青」層的抽樣率要大約是另一層的兩倍
    p = log.set_index("stratum")["sampling_prob"]
    assert p["可能文青"] == pytest.approx(2 * p["其他"], rel=0.25)
    assert sample["sampling_prob"].notna().all()
    assert sample["poi_id"].is_unique
    assert sample["auditor"].value_counts().tolist() == [30, 30, 30, 30]


def test_t07_no_stratification_when_wenqing_common(tmp_root):
    from dadaocheng.t07_audit_sample import draw, load_frame

    df = make_master(wenqing_share=0.40)
    write_master(tmp_root, df, make_web(df))
    sample, log = draw(load_frame(), 120, dev_ids=set())
    assert list(log["stratum"]) == ["全體"]
    assert sample["sampling_prob"].nunique() == 1


def test_t07_excludes_dev_sample_and_is_reproducible(tmp_root):
    from dadaocheng.t07_audit_sample import check, draw, load_frame

    df = make_master()
    write_master(tmp_root, df, make_web(df))
    au = load_frame()
    dev = set(df["poi_id"].head(20))
    s1, _ = draw(au, 120, dev_ids=dev)
    s2, _ = draw(au, 120, dev_ids=dev)
    assert not (set(s1["poi_id"]) & dev)
    assert list(s1["poi_id"]) == list(s2["poi_id"])     # 固定種子
    assert check(s1, au, dev) == ["✔ 檢查通過"]
    bad = pd.concat([s1, s1.head(1)])
    assert any("重複" in m for m in check(bad, au, dev))


def test_t07_forms_hide_llm_answers(tmp_root):
    from dadaocheng.t07_audit_sample import main

    df = make_master()
    write_master(tmp_root, df, make_web(df))
    main(["--draw"])
    forms = sorted((tmp_root / "data" / "final").glob("audit_form_*.csv"))
    assert len(forms) == 4
    t = pd.read_csv(forms[0])
    assert len(t) == 30
    assert t[["h_category", "h_is_wenqing", "h_history_narrative"]].isna().all().all()
    # 表單裡不能出現任何 LLM 欄位
    assert not ({"category", "wenqing_group", "fine_type"} & set(t.columns))
    assert "maps_url" in t.columns and t["maps_url"].str.startswith("https://").all()


# ---------------------------------------------------------------------------
# T08
# ---------------------------------------------------------------------------

def good_output(name="店", with_text=True):
    out = {f: "不明" for f in FIELDS}
    out.update({"category": "咖啡茶飲甜點", "fine_type": "咖啡店", "is_chain": "否",
                "history_narrative": ["建築"] if with_text else ["不明"],
                "founded_year": "未提及"})
    ev = {f: "" for f in FIELDS}
    ev["category"] = ev["fine_type"] = name
    ev["is_chain"] = name
    if with_text:
        ev["history_narrative"] = "老屋"
    out["evidence"] = ev
    return out


class FakeBatches:
    """模擬 Anthropic Message Batches：送出後第二次 retrieve 才 ended。"""

    def __init__(self):
        self.store: dict[str, list] = {}
        self.polls: dict[str, int] = {}

    def create(self, requests):
        bid = f"batch_{len(self.store):03d}"
        self.store[bid] = requests
        self.polls[bid] = 0
        return types.SimpleNamespace(id=bid)

    def retrieve(self, bid):
        self.polls[bid] += 1
        status = "ended" if self.polls[bid] >= 2 else "in_progress"
        return types.SimpleNamespace(processing_status=status,
                                     request_counts=types.SimpleNamespace(succeeded=len(self.store[bid])))

    def results(self, bid):
        for i, r in enumerate(self.store[bid]):
            name = r["params"]["messages"][-1]["content"].split("\n")[0].replace("店名：", "")
            with_text = "<text>" in r["params"]["messages"][-1]["content"]
            # 每 50 筆故意壞一筆，測失敗率統計
            if i % 50 == 49:
                yield types.SimpleNamespace(custom_id=r["custom_id"],
                                            result=types.SimpleNamespace(type="errored"))
                continue
            msg = types.SimpleNamespace(model_dump=lambda mode=None, n=name, t=with_text: {
                "content": [{"type": "text", "text": json.dumps(good_output(n, t), ensure_ascii=False)}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1800, "output_tokens": 240, "cache_read_input_tokens": 1500,
                          "cache_creation_input_tokens": 0}})
            yield types.SimpleNamespace(custom_id=r["custom_id"],
                                        result=types.SimpleNamespace(type="succeeded", message=msg))


@pytest.fixture()
def fake_anthropic(monkeypatch):
    batches = FakeBatches()
    client = types.SimpleNamespace(messages=types.SimpleNamespace(batches=batches))
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: client)
    return batches


def setup_llm(tmp_root, monkeypatch, n=60):
    from dadaocheng import api_layer, clients

    df = make_master(n=n)
    write_master(tmp_root, df, make_web(df))
    ledger = api_layer.default_ledger()
    caller = api_layer.ApiCaller("anthropic", tmp_root / "data" / "llm_cache" / "anthropic", ledger)

    def send(req):
        name = req["messages"][-1]["content"].split("\n")[0].replace("店名：", "")
        with_text = "<text>" in req["messages"][-1]["content"]
        return {"content": [{"type": "text", "text": json.dumps(good_output(name, with_text), ensure_ascii=False)}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1800, "output_tokens": 240}}

    monkeypatch.setattr(clients, "anthropic_caller",
                        lambda root=None: (caller, send, clients.anthropic_estimate_factory(ledger),
                                           clients.anthropic_actual_cost_factory(ledger)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return df, caller, ledger


def test_t08_three_runs_have_separate_cache_keys(tmp_root, monkeypatch):
    """自我一致要真的跑三次：三次的快取鍵必須不同。"""
    from dadaocheng.extraction import build_request
    from dadaocheng.t08_batch_extract import load_inputs, pending_items

    df, caller, _ = setup_llm(tmp_root, monkeypatch, n=5)
    poi = load_inputs()
    items = pending_items(poi, caller, 3)
    assert len(items) == len(poi) * 3
    req = build_request(poi.iloc[0]["name"], poi.iloc[0]["places_types"],
                        poi.iloc[0]["text_status"], poi.iloc[0]["text"])
    caller.store(req, {"content": []}, 0.0, extra={"run": 0})
    assert caller.cached(req, extra={"run": 0}) is not None
    assert caller.cached(req, extra={"run": 1}) is None     # 第二次不會讀到第一次


def test_t08_submit_poll_build(tmp_root, monkeypatch, fake_anthropic, capsys):
    from dadaocheng import t08_batch_extract as t08

    df, caller, ledger = setup_llm(tmp_root, monkeypatch, n=60)
    t08.main(["--estimate"])
    assert "待送出" in capsys.readouterr().out
    t08.main(["--submit", "--approve-step"])
    t08.main(["--poll"])                     # 第一次還在跑
    assert ledger.spent("anthropic") == 0
    t08.main(["--poll"])                     # 第二次取回
    assert ledger.spent("anthropic") > 0
    t08.main(["--build"])
    ext = pd.read_parquet(tmp_root / "data" / "final" / "extractions.parquet")
    assert len(ext) > 0 and "wenqing_group" in ext
    rep = pd.read_csv(tmp_root / "outputs" / "tables" / "t08_quality.csv").set_index("指標")
    assert rep.loc["失敗或格式錯誤", "值"] != "0"
    # agreement 的分母是該 POI 實際拿到的次數：有一次失敗的就是 2/2，不是 2/3
    assert (ext["agreement"].str.split("/").str[1].astype(int) == ext["n_runs"]).all()
    assert set(ext["n_runs"]) == {2, 3}
    # 再 poll 一次不應重複計費
    spent = ledger.spent("anthropic")
    t08.main(["--poll"])
    assert ledger.spent("anthropic") == spent


def test_t08_fill_sync_completes_missing(tmp_root, monkeypatch, fake_anthropic):
    from dadaocheng import t08_batch_extract as t08

    setup_llm(tmp_root, monkeypatch, n=60)
    t08.main(["--submit", "--approve-step"])
    t08.main(["--poll"]); t08.main(["--poll"])
    from dadaocheng.t08_batch_extract import load_inputs, pending_items

    caller = t08.__dict__ and None
    from dadaocheng import clients

    c, *_ = clients.anthropic_caller()
    before = len(pending_items(load_inputs(), c, 3))
    assert before > 0          # 故意壞掉的那些
    t08.main(["--fill-sync", "--approve-step"])
    assert len(pending_items(load_inputs(), c, 3)) == 0


# ---------------------------------------------------------------------------
# T09
# ---------------------------------------------------------------------------

def test_t09_agreement(tmp_root, monkeypatch):
    from dadaocheng.t09_gemini_crosscheck import agreement

    ext = pd.DataFrame({"poi_id": ["A", "B", "C"], "category": ["咖啡茶飲甜點", "熱食餐飲", "文化空間"],
                        "fine_type": ["咖啡店", "其他", "獨立書店"], "is_chain": ["否"] * 3,
                        "history_narrative": ["建築", "無", "不明"], "customer_type": ["不明"] * 3,
                        "wenqing_group": ["咖啡與茶空間", "非文青", "閱讀與展演"]})
    gem = pd.DataFrame({"poi_id": ["A", "B", "C"], "in_audit": [True, False, False],
                        "text_status": ["website"] * 3,
                        "g_category": ["咖啡茶飲甜點", "咖啡茶飲甜點", "文化空間"],
                        "g_fine_type": ["咖啡店", "咖啡店", "獨立書店"], "g_is_chain": ["否"] * 3,
                        "g_history_narrative": ["建築", "無", "不明"], "g_customer_type": ["不明"] * 3,
                        "g_wenqing_group": ["咖啡與茶空間", "非文青", "閱讀與展演"]})
    agr, dis = agreement(gem, ext)
    cat = agr.set_index("欄位").loc["category"]
    assert cat["一致率"] == pytest.approx(2 / 3)
    assert dis.iloc[0]["poi_id"] == "B" and dis.iloc[0]["不一致欄位數"] >= 2


def test_t09_only_sends_public_text(tmp_root):
    """規則 8：送給 Gemini 的內容只能有店名、類別與官網正文。"""
    from dadaocheng.t09_gemini_crosscheck import gemini_request

    req = gemini_request("某店", '["cafe"]', "website", "老屋裡的咖啡店", "gemini-x")
    body = req["contents"]
    assert "某店" in body and "老屋裡的咖啡店" in body
    assert req["config"]["response_mime_type"] == "application/json"
    assert req["config"]["temperature"] == 0


# ---------------------------------------------------------------------------
# T10
# ---------------------------------------------------------------------------

def test_collapse_history_and_yesno(tmp_root):
    from dadaocheng.t10_kappa_gate import collapse_history, normalise_yesno

    assert collapse_history("建築;街區") == "有"
    assert collapse_history("無") == "無"
    assert collapse_history("不明") == "不明"
    assert collapse_history("建築、家族") == "有"
    assert collapse_history("") is None and collapse_history(None) is None
    assert normalise_yesno("是") == "是" and normalise_yesno("Y") == "是"
    assert normalise_yesno("否") == "否" and normalise_yesno("不明") == "不明"
    assert normalise_yesno("") is None


def test_t10_report(tmp_root, monkeypatch, capsys):
    """人工與 LLM 高度一致時 κ 要高；報表要含三個關鍵欄位與兩個子集。"""
    from dadaocheng import t10_kappa_gate as t10

    n = 120
    rng = np.random.default_rng(0)
    cats = ["咖啡茶飲甜點", "一般零售與品牌", "熱食餐飲", "文化空間"]
    sample_rows, form_rows, ext_rows = [], [], []
    for i in range(n):
        pid = f"P{i:05d}"
        cat = cats[i % len(cats)]
        wen = "是" if cat in ("咖啡茶飲甜點", "文化空間") else "否"
        hist = ["建築", "無", "不明"][i % 3]
        disagree = rng.random() < 0.12          # 一成多不一致
        ts = "website" if i % 2 == 0 else "name_only"
        sample_rows.append({"poi_id": pid, "stratum": "可能文青" if wen == "是" else "其他",
                            "sampling_prob": 0.2 if wen == "是" else 0.1, "text_status": ts})
        form_rows.append({"poi_id": pid, "auditor": f"A{i % 4}", "name": f"店{i}",
                          "h_category": cat, "h_is_wenqing": wen, "h_history_narrative": hist,
                          "basis": "text" if i % 5 else "text+maps",
                          "llm_fields_reasonable": "是" if i % 7 else "否",
                          "evidence_reasonable": "是" if i % 6 else "否"})
        l_cat = cats[(i + 1) % len(cats)] if disagree else cat
        ext_rows.append({"poi_id": pid, "category": l_cat,
                         "wenqing_group": "咖啡與茶空間" if (wen == "是") != disagree else "非文青",
                         "history_narrative": {"建築": "建築", "無": "無", "不明": "不明"}[hist],
                         "evidence_valid_category": True, "text_status": ts})
    d = tmp_root / "data" / "final"; d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sample_rows).to_csv(d / "audit_sample.csv", index=False, encoding="utf-8-sig")
    f = pd.DataFrame(form_rows)
    for i in range(4):
        f.iloc[i::4].to_csv(d / f"audit_form_{i + 1}_A{i}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ext_rows).to_parquet(d / "extractions.parquet", index=False)

    t10.main(["--n-boot", "200"])
    out = capsys.readouterr().out
    assert "本程式只產出報表，不宣布通過" in out
    rep = pd.read_csv(tmp_root / "outputs" / "tables" / "t10_kappa.csv")
    assert set(rep[rep["子集"] == "全部"]["欄位"]) == {"category", "is_wenqing", "history_narrative"}
    assert {"website", "name_only"} <= set(rep["子集"])
    full = rep[rep["子集"] == "全部"].set_index("欄位")
    assert full.loc["category", "κ"] > 0.6
    assert full.loc["category", "κ 95% 下界"] < full.loc["category", "κ"] < full.loc["category", "κ 95% 上界"]
    # 加權準確率與未加權不同（分層抽樣）
    assert full.loc["is_wenqing", "加權準確率"] != full.loc["is_wenqing", "準確率"]
    assert (tmp_root / "outputs" / "tables" / "t10_confusion.md").exists()
    assert (tmp_root / "outputs" / "tables" / "t10_per_class_recall.csv").exists()


def test_t10_handles_partially_filled_forms(tmp_root, capsys):
    from dadaocheng import t10_kappa_gate as t10

    d = tmp_root / "data" / "final"; d.mkdir(parents=True, exist_ok=True)
    rows = [{"poi_id": f"P{i:05d}", "stratum": "全體", "sampling_prob": 0.1, "text_status": "website"}
            for i in range(20)]
    pd.DataFrame(rows).to_csv(d / "audit_sample.csv", index=False, encoding="utf-8-sig")
    form = pd.DataFrame([{"poi_id": f"P{i:05d}", "h_category": "咖啡茶飲甜點" if i < 8 else "",
                          "h_is_wenqing": "是" if i < 8 else "", "h_history_narrative": "無" if i < 8 else "",
                          "basis": "text", "llm_fields_reasonable": "", "evidence_reasonable": ""}
                         for i in range(20)])
    form.to_csv(d / "audit_form_1_A.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"poi_id": f"P{i:05d}", "category": "咖啡茶飲甜點", "wenqing_group": "咖啡與茶空間",
                   "history_narrative": "無", "evidence_valid_category": True, "text_status": "website"}
                  for i in range(20)]).to_parquet(d / "extractions.parquet", index=False)
    t10.main(["--n-boot", "100"])
    assert "沒填" in capsys.readouterr().out


def test_forced_allocation_overrides_threshold():
    """likely_wenqing 比例高於 15% 時，--allocate 仍要強制分層並照配額抽。"""
    import numpy as np
    import pandas as pd

    from dadaocheng import t07_audit_sample as t07

    n = 400
    rng = np.random.default_rng(0)
    au = pd.DataFrame({
        "poi_id": [f"p{i:04d}" for i in range(n)],
        "analysis_unit": True,
        "likely_wenqing": rng.random(n) < 0.30,       # 30%，遠高於 15% 門檻
        "place_id": [f"pl{i}" for i in range(n)],
        "lon": 121.51, "lat": 25.055,
        "text_status": "ok", "text": "x",
    })
    sample, log = t07.draw(au, 120, set(), alloc={"可能文青": 50, "其他": 70})
    assert len(sample) == 120
    counts = sample["stratum"].value_counts().to_dict()
    assert counts == {"其他": 70, "可能文青": 50}
    assert log["stratum"].nunique() == 2
    # 兩層的 sampling_prob 不同，而且「可能文青」層被超額抽樣
    probs = dict(zip(log["stratum"], log["sampling_prob"]))
    assert probs["可能文青"] > probs["其他"]
    assert sample.groupby("stratum")["sampling_prob"].nunique().eq(1).all()

    # 沒給 alloc 時，30% 高於門檻 → 不分層
    s2, l2 = t07.draw(au, 120, set())
    assert l2["stratum"].tolist() == ["全體"] and len(s2) == 120


def test_allocation_larger_than_stratum_is_capped(capsys):
    import pandas as pd

    from dadaocheng import t07_audit_sample as t07

    au = pd.DataFrame({
        "poi_id": [f"q{i:03d}" for i in range(100)],
        "analysis_unit": True,
        "likely_wenqing": [True] * 20 + [False] * 80,
        "place_id": [f"pl{i}" for i in range(100)],
        "lon": 121.51, "lat": 25.055, "text_status": "ok", "text": "x",
    })
    sample, log = t07.draw(au, 120, set(), alloc={"可能文青": 50, "其他": 70})
    assert dict(zip(log["stratum"], log["抽出筆數"])) == {"可能文青": 20, "其他": 70}
    assert "母體只有 20 筆" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Excel 表單與範例（11/12：組員看 CSV 不知道怎麼填）
# ---------------------------------------------------------------------------

def test_t07_writes_excel_forms_with_dropdowns(tmp_root):
    from openpyxl import load_workbook

    from dadaocheng.t07_audit_sample import CATEGORY_CHOICES, main

    df = make_master()
    write_master(tmp_root, df, make_web(df))
    main(["--draw"])
    d = tmp_root / "data" / "final"
    xs = sorted(d.glob("audit_form_*.xlsx"))
    assert len(xs) == 4
    wb = load_workbook(xs[0])
    ws = wb["填寫"]
    assert ws.max_row == 31                       # 標題＋30 筆
    heads = [c.value for c in ws[1]]
    assert any("業態大類" in h for h in heads)
    # 表單裡不能出現 LLM 欄位
    assert not any(h in ("category", "wenqing_group", "fine_type") for h in heads)
    formulas = [dv.formula1 for dv in ws.data_validations.dataValidation]
    assert any(all(c in f for c in CATEGORY_CHOICES) for f in formulas)
    assert any("建築;街區" in f for f in formulas)
    assert "怎麼填" in wb.sheetnames
    assert (tmp_root / "docs" / "audit_form_example.xlsx").exists()


def test_t07_xlsx_regenerate_refuses_to_overwrite(tmp_root):
    from dadaocheng.t07_audit_sample import main

    df = make_master()
    write_master(tmp_root, df, make_web(df))
    main(["--draw"])
    with pytest.raises(SystemExit):
        main(["--xlsx"])
    main(["--xlsx", "--force"])


def test_example_rows_are_valid_choices_and_not_real(tmp_root):
    from dadaocheng.t07_audit_sample import (BASIS_CHOICES, CATEGORY_CHOICES, EXAMPLE_ROWS,
                                             HISTORY_CHOICES, WENQING_CHOICES)
    from dadaocheng.t10_kappa_gate import collapse_history

    for r in EXAMPLE_ROWS:
        assert r["name"].startswith("【範例】")
        assert r["h_category"] in CATEGORY_CHOICES
        assert r["h_is_wenqing"] in WENQING_CHOICES
        assert r["h_history_narrative"] in HISTORY_CHOICES
        assert r["basis"] in BASIS_CHOICES
        assert collapse_history(r["h_history_narrative"]) in ("有", "無", "不明")


def test_t10_prefers_filled_excel_over_empty_csv(tmp_root):
    from dadaocheng import t07_audit_sample as t07
    from dadaocheng import t10_kappa_gate as t10

    df = make_master()
    write_master(tmp_root, df, make_web(df))
    t07.main(["--draw"])
    d = tmp_root / "data" / "final"
    x = sorted(d.glob("audit_form_*.xlsx"))[0]
    from openpyxl import load_workbook

    wb = load_workbook(x)
    ws = wb["填寫"]
    heads = [c.value for c in ws[1]]
    col = {k: heads.index(h) + 1 for k, h in
           [("cat", next(h for h in heads if "業態" in h)), ("wen", next(h for h in heads if "是否文青" in h)),
            ("his", next(h for h in heads if "歷史" in h))]}
    for r in range(2, 7):
        ws.cell(r, col["cat"], "咖啡茶飲甜點"); ws.cell(r, col["wen"], "是"); ws.cell(r, col["his"], "建築;街區")
    wb.save(x)
    h = t10.load_human()
    got = h[h["form"] == x.name]
    assert len(got) == 30
    assert got["h_category"].notna().sum() == 5
    assert set(got["h_history"].dropna()) == {"有"}
    # 沒填的 Excel 表單 → 仍能讀（退回 CSV 或空的 Excel），不會報錯
    assert h["form"].nunique() == 4


def test_t08_status_reports_wait_time(tmp_root, capsys):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from dadaocheng import t08_batch_extract as t08

    t08.save_state({"batches": [{"batch_id": "msgbatch_x", "n": 10}]})
    now = datetime.now(timezone.utc)
    info = SimpleNamespace(
        processing_status="in_progress", created_at=now - timedelta(hours=2), expires_at=now + timedelta(hours=22),
        request_counts=SimpleNamespace(processing=10, succeeded=0, errored=0, canceled=0, expired=0))
    client = SimpleNamespace(messages=SimpleNamespace(batches=SimpleNamespace(retrieve=lambda _id: info)))
    t08.print_status(client)
    out = capsys.readouterr().out
    assert "已完成 0/10" in out and "已等 2.0 小時" in out and "距離到期 22.0 小時" in out


def test_diagnose_t08_runs_on_fake_results(tmp_root, monkeypatch, fake_anthropic, capsys):
    import sys
    from pathlib import Path

    from dadaocheng import t08_batch_extract as t08

    setup_llm(tmp_root, monkeypatch, n=40)
    t08.main(["--submit", "--approve-step"])
    t08.main(["--poll"]); t08.main(["--poll"])
    t08.main(["--build"])
    capsys.readouterr()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import diagnose_t08

    diagnose_t08.main()
    out = capsys.readouterr().out
    for sec in ["== 1. 失敗類型", "== 3. 每種失敗", "== 5. 歷史敘事", "== 7. 抽查樣本"]:
        assert sec in out


def test_t08_retry_truncated_only_reruns_truncated(tmp_root, monkeypatch, fake_anthropic, capsys):
    """截斷的回應用較大的 max_tokens 重跑；沒截斷的不動，build 優先用重跑結果。"""
    from dadaocheng import t08_batch_extract as t08
    from dadaocheng.extraction import build_request

    df, caller, ledger = setup_llm(tmp_root, monkeypatch, n=12)
    poi = t08.load_inputs()
    trunc = set()
    for i, (_, r) in enumerate(poi.iterrows()):
        req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
        good = {"content": [{"type": "text", "text": json.dumps(good_output(r["name"], r["text_status"] == "website"),
                                                                 ensure_ascii=False)}],
                "stop_reason": "end_turn", "usage": {"output_tokens": 300}}
        bad = {"content": [{"type": "text", "text": '{"category": "咖啡茶'}],
               "stop_reason": "max_tokens", "usage": {"output_tokens": 1200}}
        for k in range(3):
            is_bad = (i % 3 == 0 and k == 0)
            caller.store(req, bad if is_bad else good, 0.0, extra={"run": k})
            if is_bad:
                trunc.add((r["poi_id"], k))

    items = t08.truncated_items(poi, caller, 3)
    assert {(it["poi_id"], it["run"]) for it in items} == trunc
    assert all(it["request"]["max_tokens"] == 4000 for it in items)

    t08.main(["--build"])
    rep = pd.read_csv(tmp_root / "outputs" / "tables" / "t08_quality.csv").set_index("指標")
    assert int(float(rep.loc["失敗或格式錯誤", "值"])) == len(trunc)

    before = ledger.requests("anthropic")
    t08.main(["--retry-truncated", "--sync"])
    assert ledger.requests("anthropic") - before == len(trunc)     # 只重跑截斷的
    assert t08.truncated_items(poi, caller, 3) == []
    t08.main(["--retry-truncated", "--sync"])                      # 再跑一次：沒東西可跑
    assert "沒有需要重跑" in capsys.readouterr().out
    t08.main(["--build"])
    rep = pd.read_csv(tmp_root / "outputs" / "tables" / "t08_quality.csv").set_index("指標")
    assert int(float(rep.loc["失敗或格式錯誤", "值"])) == 0
    # 普通的 pending 不受影響（不會把全部重送）
    assert t08.pending_items(poi, caller, 3) == []


def test_t08_retry_truncated_batch_path(tmp_root, monkeypatch, fake_anthropic):
    from dadaocheng import t08_batch_extract as t08
    from dadaocheng.extraction import build_request

    df, caller, ledger = setup_llm(tmp_root, monkeypatch, n=6)
    poi = t08.load_inputs()
    r = poi.iloc[0]
    req = build_request(r["name"], r["places_types"], r["text_status"], r["text"])
    caller.store(req, {"content": [{"type": "text", "text": "{"}], "stop_reason": "max_tokens"}, 0.0, extra={"run": 1})
    t08.main(["--retry-truncated", "--approve-step"])
    st = t08.load_state()
    assert st["batches"][-1]["retry"] == "max_tokens"
    t08.main(["--poll"]); t08.main(["--poll"])
    assert caller.cached(t08.retry_request(req), extra=t08.retry_extra(1)) is not None
