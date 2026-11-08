"""T06：schema、prompt 順序（快取前綴固定）、證據子字串檢查、name_only 強制、文青推導、多數決、端到端（假 API）。"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

TEXT = "我們是迪化街上的一間小店，2015年在百年街屋裡開業，販售台灣職人手作的器物，也不定期舉辦講座。"


def good_output(**over):
    d = {
        "category": "選物與文創零售", "fine_type": "選物或文創商品", "is_chain": "否",
        "style_design": "不明", "style_local_story": "是", "style_handmade": "是", "style_reading": "是",
        "history_narrative": ["建築"], "customer_type": "不明", "founded_year": "2015",
        "evidence": {"category": "販售台灣職人手作的器物", "fine_type": "販售台灣職人手作的器物", "is_chain": "一間小店",
                     "style_design": "", "style_local_story": "迪化街上的一間小店", "style_handmade": "職人手作",
                     "style_reading": "不定期舉辦講座", "history_narrative": "百年街屋", "customer_type": "",
                     "founded_year": "2015年在百年街屋裡開業"},
    }
    d.update(over)
    return d


def test_schema_has_unknown_everywhere(tmp_root):
    from dadaocheng.extraction import json_schema

    s = json_schema()
    for k, v in s["properties"].items():
        if v.get("type") == "string" and "enum" in v:
            assert "不明" in v["enum"], k
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])
    assert "wenqing_group" not in s["properties"]   # 規則 11：不叫 LLM 判斷文青


def test_prompt_prefix_fixed_and_poi_last(tmp_root):
    from dadaocheng.extraction import build_request

    a = build_request("店A", '["cafe"]', "website", "文本A")
    b = build_request("店B", '["store"]', "name_only", None)
    assert a["system"] == b["system"]                          # 前綴相同 → 吃得到快取
    assert a["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert a["output_config"] == b["output_config"]
    assert a["messages"][0]["content"].startswith("店名：店A")
    assert "文本A" in a["messages"][-1]["content"] and "文本A" not in json.dumps(a["system"], ensure_ascii=False)
    assert a["model"] == "claude-haiku-5-5"


def test_evidence_substring_rules(tmp_root):
    from dadaocheng.extraction import evidence_is_substring

    assert evidence_is_substring("百年街屋", TEXT)
    assert evidence_is_substring("百年 街屋", TEXT)            # 只容忍空白差異
    assert not evidence_is_substring("百年老屋", TEXT)         # 改寫 → 無效
    assert not evidence_is_substring("", TEXT)


def test_validate_and_invalid_evidence(tmp_root):
    from dadaocheng.extraction import apply_invalid_evidence, user_content, validate_extraction

    src = user_content("小店", '["store"]', "website", TEXT)
    out = good_output()
    out["evidence"]["customer_type"] = ""
    out["evidence"]["style_handmade"] = "純手工打造"            # 不是子字串
    flat = validate_extraction(json.dumps(out, ensure_ascii=False), src, "website")
    assert flat["evidence_valid_category"] is True
    assert flat["evidence_valid_style_handmade"] is False
    assert flat["evidence_valid_customer_type"] is None        # 不明不檢查
    assert flat["founded_year"] == 2015
    fixed = apply_invalid_evidence(flat)
    assert fixed["style_handmade"] == "不明" and fixed["style_handmade_raw"] == "是"


def test_name_only_forced_unknown(tmp_root):
    from dadaocheng.extraction import user_content, validate_extraction

    src = user_content("小藝埕咖啡", '["cafe"]', "name_only", None)
    out = good_output(category="咖啡茶飲甜點", fine_type="咖啡店", is_chain="不明")
    out["evidence"].update({"category": "小藝埕咖啡", "fine_type": "小藝埕咖啡", "is_chain": ""})
    flat = validate_extraction(out, src, "name_only")
    assert flat["style_local_story"] == "不明" and flat["history_narrative"] == "不明"
    assert flat["founded_year"] is None
    assert "style_local_story" in flat["name_only_violations"]
    assert flat["evidence_valid_category"] is True


def test_history_unknown_cannot_mix(tmp_root):
    from dadaocheng.extraction import Extraction

    e = Extraction.model_validate(good_output(history_narrative=["無", "建築"]))
    assert e.history_narrative == ["不明"]


def test_bad_year_becomes_not_mentioned(tmp_root):
    from dadaocheng.extraction import Extraction

    assert Extraction.model_validate(good_output(founded_year="百年")).founded_year == "未提及"
    assert Extraction.model_validate(good_output(founded_year="3021")).founded_year == "未提及"


@pytest.mark.parametrize("rec,expected", [
    ({"is_chain": "否", "fine_type": "咖啡店"}, "咖啡與茶空間"),
    ({"is_chain": "是", "fine_type": "咖啡店"}, "非文青"),
    ({"is_chain": "不明", "fine_type": "茶館或茶空間"}, "咖啡與茶空間"),
    ({"is_chain": "否", "fine_type": "手搖飲或外帶飲料"}, "非文青"),
    ({"is_chain": "否", "fine_type": "獨立書店", "style_reading": "是"}, "閱讀與展演"),
    ({"is_chain": "否", "fine_type": "獨立書店"}, "非文青"),
    ({"is_chain": "否", "fine_type": "選物或文創商品", "style_design": "是"}, "選物與工藝"),
    ({"is_chain": "否", "fine_type": "甜點烘焙", "style_local_story": "是"}, "咖啡與茶空間"),
    ({"is_chain": "否", "fine_type": "中藥", "style_local_story": "是"}, "非文青"),
])
def test_derive_wenqing(tmp_root, rec, expected):
    from dadaocheng.extraction import derive_wenqing_group

    assert derive_wenqing_group(rec) == expected


def test_majority_vote(tmp_root):
    from dadaocheng.extraction import majority_vote

    runs = [{"category": "熱食餐飲", "evidence_category": "a"}, {"category": "熱食餐飲", "evidence_category": "b"},
            {"category": "其他", "evidence_category": "c"}]
    for r in runs:
        for f in ["fine_type", "is_chain", "style_design", "style_local_story", "style_handmade", "style_reading",
                  "history_narrative", "customer_type", "founded_year"]:
            r.setdefault(f, "不明")
    mv = majority_vote(runs)
    assert mv["category"] == "熱食餐飲" and mv["evidence_category"] == "a" and mv["agreement"] == "2/3"


def test_t06_never_reads_audit_answers():
    """規則 3：prompt 開發不得讀取抽查樣本的人工答案。"""
    src = (Path(__file__).resolve().parents[1] / "src" / "dadaocheng" / "t06_prompt_dev.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith(("#", '"', "-", "規則")))
    assert "audit_sample" not in code.replace("audit_sample.csv 或任何人工標註", "")


def test_run_dev_with_fake_api(tmp_root):
    """假的 Claude：回傳固定 JSON；檢查審閱表、證據有效率、快取（第二次跑不再付費）、步驟預算檢查。"""
    from dadaocheng.api_layer import ApiCaller, Ledger
    from dadaocheng.clients import anthropic_actual_cost_factory, anthropic_estimate_factory
    from dadaocheng.t06_prompt_dev import evidence_valid_rate, review_markdown, run_dev

    ledger = Ledger(tmp_root / "config" / "budget.yaml")
    caller = ApiCaller("anthropic", tmp_root / "data" / "llm_cache" / "anthropic", ledger)
    sent = []

    def send(req):
        sent.append(req)
        return {"content": [{"type": "text", "text": json.dumps(good_output(), ensure_ascii=False)}],
                "stop_reason": "end_turn", "usage": {"input_tokens": 1500, "output_tokens": 300,
                                                     "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 0}}

    df = pd.DataFrame({"poi_id": ["P1", "P2"], "name": ["小店", "小店二"], "places_types": ['["store"]', '["store"]'],
                       "text_status": ["website", "website"], "text": [TEXT, TEXT]})
    res = run_dev(df, caller, send, anthropic_estimate_factory(ledger), anthropic_actual_cost_factory(ledger))
    assert len(sent) == 2 and (res["parse_error"] == "").all()
    assert evidence_valid_rate(res) == 1.0
    assert (res["wenqing_group"] == "選物與工藝").all()
    assert "| category |" in review_markdown(res, df)
    spent = ledger.spent("anthropic")
    assert 0 < spent < 1
    run_dev(df, caller, send, anthropic_estimate_factory(ledger), anthropic_actual_cost_factory(ledger))
    assert len(sent) == 2 and ledger.spent("anthropic") == spent
