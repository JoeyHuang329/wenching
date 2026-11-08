"""LLM 抽取的 schema、prompt 組裝、輸出驗證、wenqing_group 推導（T06 開發、T08 全量共用）。

- schema 對應實作規劃 B 第 3.2 節，另加 fine_type（推導 wenqing_group 必須，見審閱筆記）。
- prompt：固定的系統指令＋欄位定義放 system（可吃 prompt caching），POI 內容放 user 最後。
- 規則 7：證據片段必須是輸入文本的子字串，由程式檢查；不是子字串的欄位標為無效。
- 規則 11：wenqing_group 由程式依規則推導，不叫 LLM 判斷。
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from . import config

UNK = "不明"
CATEGORIES = ["熱食餐飲", "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "文化空間", "一般零售與品牌", "其他", UNK]
FINE_TYPES = ["咖啡店", "茶館或茶空間", "手搖飲或外帶飲料", "甜點烘焙", "獨立書店", "連鎖或一般書店", "藝廊或展演空間",
              "選物或文創商品", "手作或工藝", "設計工作室", "中藥", "南北貨", "布業", "茶葉行", "其他", UNK]
YN = ["是", "否", UNK]
HISTORY = ["無", "建築", "街區", "家族", UNK]
CUSTOMER = ["目的型", "路過型", "批發", "在地民生", UNK]
STYLE_FIELDS = ["style_design", "style_local_story", "style_handmade", "style_reading"]
FIELDS = ["category", "fine_type", "is_chain", *STYLE_FIELDS, "history_narrative", "customer_type", "founded_year"]
NAME_ONLY_ALLOWED = {"category", "fine_type", "is_chain"}
KEY_FIELDS = ["category", "is_wenqing", "history_narrative"]  # 閘門一的三個關鍵欄位（is_wenqing 為推導值）


# ---------------------------------------------------------------------------
# JSON schema（給 output_config.format）
# ---------------------------------------------------------------------------

def json_schema() -> dict:
    def enum(vals):
        return {"type": "string", "enum": vals}

    props = {
        "category": enum(CATEGORIES),
        "fine_type": enum(FINE_TYPES),
        "is_chain": enum(YN),
        **{f: enum(YN) for f in STYLE_FIELDS},
        "history_narrative": {"type": "array", "items": enum(HISTORY)},
        "customer_type": enum(CUSTOMER),
        "founded_year": {"type": "string", "description": "四位數西元年，或「未提及」"},
        "evidence": {
            "type": "object",
            "properties": {f: {"type": "string"} for f in FIELDS},
            "required": FIELDS,
            "additionalProperties": False,
        },
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


class Extraction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Literal[tuple(CATEGORIES)]  # type: ignore[valid-type]
    fine_type: Literal[tuple(FINE_TYPES)]  # type: ignore[valid-type]
    is_chain: Literal[tuple(YN)]  # type: ignore[valid-type]
    style_design: Literal[tuple(YN)]  # type: ignore[valid-type]
    style_local_story: Literal[tuple(YN)]  # type: ignore[valid-type]
    style_handmade: Literal[tuple(YN)]  # type: ignore[valid-type]
    style_reading: Literal[tuple(YN)]  # type: ignore[valid-type]
    history_narrative: list[Literal[tuple(HISTORY)]]  # type: ignore[valid-type]
    customer_type: Literal[tuple(CUSTOMER)]  # type: ignore[valid-type]
    founded_year: str
    evidence: dict[str, str]

    @field_validator("history_narrative")
    @classmethod
    def _hist(cls, v):
        if not v:
            return [UNK]
        v = list(dict.fromkeys(v))
        if (("無" in v) or (UNK in v)) and len(v) > 1:
            # 「無／不明」不能與其他值並存：保守地改成不明
            return [UNK]
        return v

    @field_validator("founded_year")
    @classmethod
    def _year(cls, v):
        v = v.strip()
        if v == "未提及":
            return v
        if re.fullmatch(r"(1[6-9]\d\d|20[0-2]\d)", v):
            return v
        return "未提及"

    @field_validator("evidence")
    @classmethod
    def _ev(cls, v):
        return {f: (v.get(f) or "") for f in FIELDS}


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

def system_prompt() -> str:
    p = config.project_root() / config.settings()["extraction"]["prompt_file"]
    return p.read_text(encoding="utf-8").strip()


def prompt_version() -> str:
    h = hashlib.sha256((system_prompt() + json.dumps(json_schema(), ensure_ascii=False, sort_keys=True)).encode("utf-8"))
    name = config.settings()["extraction"]["prompt_file"].split("/")[-1].replace(".md", "")
    return f"{name}-{h.hexdigest()[:8]}"


def user_content(name: str, places_types: str | None, text_status: str, text: str | None) -> str:
    """POI 的內容（放在 prompt 最後）。訪談文字絕不會出現在這裡（規則 1）：來源只有 Places 名稱／類別與官網正文。"""
    types = places_types
    if isinstance(types, str) and types.startswith("["):
        try:
            types = "、".join(json.loads(types))
        except json.JSONDecodeError:
            pass
    parts = [f"店名：{name}", f"Google 類別：{types or '（無）'}", f"文本狀態：{text_status}"]
    if text_status == "name_only" or not text:
        parts.append("店家文本：（無）")
    else:
        parts.append("店家文本：\n<text>\n" + text.strip() + "\n</text>")
    return "\n".join(parts)


def build_request(name, places_types, text_status, text, model: str | None = None) -> dict:
    from .clients import anthropic_message_request

    s = config.settings()
    return anthropic_message_request(
        model=model or s["models"]["extraction"],
        system_prefix=system_prompt(),
        user_content=user_content(name, places_types, text_status, text),
        json_schema=json_schema(),
        max_tokens=int(s["models"]["extraction_max_tokens"]),
    )


# ---------------------------------------------------------------------------
# 驗證
# ---------------------------------------------------------------------------

def _norm_ws(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)


def evidence_is_substring(evidence: str, source: str) -> bool:
    """證據是否為輸入文本的子字串。只容忍空白與全半形差異（NFKC＋去空白），其餘必須逐字相同。"""
    if not evidence:
        return False
    return _norm_ws(evidence) in _norm_ws(source)


def field_is_unknown(field: str, value) -> bool:
    if field == "history_narrative":
        return value == [UNK]
    if field == "founded_year":
        return value == "未提及"
    return value == UNK


def validate_extraction(raw: dict | str, source_text: str, text_status: str) -> dict:
    """解析＋驗證一次 LLM 輸出。回傳扁平化的 dict（含 evidence_valid_*、name_only_violations）。"""
    data = json.loads(raw) if isinstance(raw, str) else raw
    ex = Extraction.model_validate(data)
    out = ex.model_dump()
    ev = out.pop("evidence")
    violations = []
    if text_status == "name_only":
        for f in FIELDS:
            if f in NAME_ONLY_ALLOWED:
                continue
            if not field_is_unknown(f, out[f]):
                violations.append(f)
                out[f] = [UNK] if f == "history_narrative" else ("未提及" if f == "founded_year" else UNK)
                ev[f] = ""
    flat = {}
    for f in FIELDS:
        v = out[f]
        flat[f] = ";".join(v) if isinstance(v, list) else v
        flat[f"evidence_{f}"] = ev.get(f, "")
        if field_is_unknown(f, v):
            flat[f"evidence_valid_{f}"] = None
        else:
            flat[f"evidence_valid_{f}"] = evidence_is_substring(ev.get(f, ""), source_text)
    flat["name_only_violations"] = ";".join(violations)
    flat["founded_year"] = None if out["founded_year"] == "未提及" else int(out["founded_year"])
    return flat


def apply_invalid_evidence(flat: dict) -> dict:
    """證據無效的欄位 → 視為「不明」（規則 7：不是子字串的欄位標為無效）。原值保留在 *_raw。"""
    out = dict(flat)
    for f in FIELDS:
        if out.get(f"evidence_valid_{f}") is False:
            out[f"{f}_raw"] = out[f]
            out[f] = None if f == "founded_year" else UNK
    return out


# ---------------------------------------------------------------------------
# wenqing_group 推導（規則 11）
# ---------------------------------------------------------------------------

def derive_wenqing_group(rec: dict, unknown_chain_as_independent: bool = True) -> str:
    """依提案書 6.4 推導文青分群。

    - 連鎖一律排除；is_chain 為「不明」時，預設視為獨立（【待確認】，見 preregistration.md）。
    - 咖啡店、茶館或茶空間：獨立即為文青（咖啡與茶空間）。
    - 甜點烘焙：需至少一項風格指標（「空間型甜點店」）（【待確認】）。
    - 獨立書店、藝廊或展演空間：需至少一項風格指標 → 閱讀與展演。
    - 選物或文創商品、手作或工藝、設計工作室：需至少一項風格指標 → 選物與工藝。
    - 其他一律非文青（包含轉型中的老店：中藥、南北貨等）。
    """
    chain = rec.get("is_chain")
    if chain == "是" or (chain == UNK and not unknown_chain_as_independent):
        return "非文青"
    ft = rec.get("fine_type")
    any_style = any(rec.get(f) == "是" for f in STYLE_FIELDS)
    if ft in ("咖啡店", "茶館或茶空間"):
        return "咖啡與茶空間"
    if ft == "甜點烘焙" and any_style:
        return "咖啡與茶空間"
    if ft in ("獨立書店", "藝廊或展演空間") and any_style:
        return "閱讀與展演"
    if ft in ("選物或文創商品", "手作或工藝", "設計工作室") and any_style:
        return "選物與工藝"
    return "非文青"


def majority_vote(runs: list[dict]) -> dict:
    """三次抽取的多數決（T08 用）。每個欄位取眾數；平手（三個都不同）→ 不明。回傳含 agreement。"""
    from collections import Counter

    out, agree = {}, []
    for f in FIELDS:
        vals = [r.get(f) for r in runs]
        c = Counter(vals).most_common()
        top, n = c[0]
        if n == 1 and len(runs) >= 3:
            top = None if f == "founded_year" else UNK
        out[f] = top
        agree.append(n)
        # 證據取第一個與多數值相同的 run
        for r in runs:
            if r.get(f) == top:
                out[f"evidence_{f}"] = r.get(f"evidence_{f}", "")
                out[f"evidence_valid_{f}"] = r.get(f"evidence_valid_{f}")
                break
    out["agreement"] = f"{min(agree)}/{len(runs)}"
    return out
