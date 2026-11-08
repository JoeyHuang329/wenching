"""各付費 API 的 send / 花費估算函式。全部要透過 api_layer.ApiCaller 使用。

- Google Places (New) Nearby Search：必帶 X-Goog-FieldMask（規則 5）。
- Anthropic Messages：固定前綴（系統指令＋schema）在前、POI 內容在後，才吃得到 prompt caching。
- Gemini 免費方案：遇 429 指數退避、不並行（CLAUDE.md「LLM」段）。只送公開店家文本（規則 8）。
"""

from __future__ import annotations

import math
import re
import time
from typing import Any

from .api_layer import ApiCaller, Ledger
from . import config

# ---------------------------------------------------------------------------
# 共用：token 粗估（保守：寧可高估）
# ---------------------------------------------------------------------------

_CJK = re.compile(r"[　-〿㐀-鿿豈-﫿＀-￯]")


def estimate_tokens(text: str) -> int:
    """中文約 1 字 1 token、英數約 3.5 字元 1 token，再乘 1.3 的安全係數。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(math.ceil((cjk + other / 3.5) * 1.3))


# ---------------------------------------------------------------------------
# Google Places
# ---------------------------------------------------------------------------

def places_field_mask() -> str:
    s = config.settings()["places"]
    forbidden = {f.lower() for f in s["forbidden_fields"]}
    for f in s["field_mask"]:
        if f.lower() in forbidden or f.split(".")[-1].lower() in forbidden:
            raise ValueError(f"欄位遮罩含禁止欄位 {f}（規則 5）")
    return ",".join(s["field_mask"])


def places_nearby_request(lat: float, lon: float, radius_m: float) -> dict:
    s = config.settings()["places"]
    return {
        "maxResultCount": int(s["max_result_count"]),
        "rankPreference": s["rank_preference"],
        "languageCode": s["language_code"],
        "regionCode": s["region_code"],
        "locationRestriction": {
            "circle": {"center": {"latitude": round(lat, 7), "longitude": round(lon, 7)}, "radius": round(float(radius_m), 2)}
        },
        # 不設 includedTypes（實作規劃 B T02）
    }


def places_send_factory(api_key: str, timeout: float = 30.0):
    import httpx

    endpoint = config.settings()["places"]["endpoint"]
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": places_field_mask(),
    }

    def send(request: dict) -> dict:
        r = httpx.post(endpoint, json=request, headers=headers, timeout=timeout)
        if r.status_code != 200:
            raise RuntimeError(f"Places API 回傳 {r.status_code}: {r.text[:500]}")
        return r.json()

    return send


def places_cost_factory(ledger: Ledger):
    """免費額度內 0 元；超過後以 budget.yaml 的單價計。"""
    p = ledger.pricing("google_places").get("nearby_search_enterprise", {})
    free = int(p.get("free_requests_per_month", 1000))
    usd_per_1000 = float(p.get("usd_per_1000_after_free", 35.0))

    def estimate(_req: dict) -> float:
        if ledger.requests("google_places") < free:
            return 0.0
        return usd_per_1000 / 1000 * ledger.fx

    return estimate


def places_caller(root=None) -> tuple[ApiCaller, Any, Any]:
    from .api_layer import default_ledger

    ledger = default_ledger(root)
    s = config.settings(root)
    caller = ApiCaller(
        "google_places", config.path("places_cache", root), ledger, max_requests=int(s["places_max_requests"])
    )
    send = places_send_factory(config.require_env("GOOGLE_PLACES_API_KEY"))
    return caller, send, places_cost_factory(ledger)


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def anthropic_message_request(
    model: str,
    system_prefix: str,
    user_content: str,
    json_schema: dict | None,
    max_tokens: int,
    cache_prefix: bool = True,
    temperature: float | None = None,
) -> dict:
    """組 Messages API 的請求本體。

    固定的系統指令放 system（加 cache_control），POI 內容放 user 最後。
    結構化輸出用 output_config.format（json_schema）。
    """
    system_block: dict[str, Any] = {"type": "text", "text": system_prefix}
    if cache_prefix:
        system_block["cache_control"] = {"type": "ephemeral"}
    req: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [system_block],
        "messages": [{"role": "user", "content": user_content}],
    }
    if json_schema is not None:
        req["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
    if temperature is not None:
        req["temperature"] = temperature
    return req


def _anthropic_prices(ledger: Ledger, model: str) -> dict:
    prices = ledger.pricing("anthropic").get(model)
    if not prices:
        raise KeyError(f"budget.yaml 沒有 {model} 的單價，請先補上")
    return prices


def anthropic_estimate_factory(ledger: Ledger, batch: bool = False):
    def estimate(req: dict) -> float:
        p = _anthropic_prices(ledger, req["model"])
        text = "".join(b.get("text", "") for b in req.get("system", []))
        text += "".join(m["content"] if isinstance(m["content"], str) else str(m["content"]) for m in req["messages"])
        if "output_config" in req:
            text += str(req["output_config"])
        in_tok = estimate_tokens(text)
        out_tok = int(req.get("max_tokens", 1000))  # 保守：以 max_tokens 計輸出
        usd = in_tok / 1e6 * p["input_per_mtok"] + out_tok / 1e6 * p["output_per_mtok"]
        if batch:
            usd *= p.get("batch_discount", 1.0)
        return usd * ledger.fx

    return estimate


def anthropic_actual_cost_factory(ledger: Ledger, batch: bool = False):
    def actual(req: dict, resp: dict) -> float:
        p = _anthropic_prices(ledger, req["model"])
        u = resp.get("usage", {}) or {}
        usd = (
            u.get("input_tokens", 0) / 1e6 * p["input_per_mtok"]
            + u.get("output_tokens", 0) / 1e6 * p["output_per_mtok"]
            + u.get("cache_read_input_tokens", 0) / 1e6 * p.get("cache_read_per_mtok", p["input_per_mtok"])
            + u.get("cache_creation_input_tokens", 0) / 1e6 * p.get("cache_write_per_mtok", p["input_per_mtok"] * 1.25)
        )
        if batch:
            usd *= p.get("batch_discount", 1.0)
        return usd * ledger.fx

    return actual


def anthropic_send_factory(api_key: str):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key, max_retries=3)

    def send(req: dict) -> dict:
        resp = client.messages.create(**req)
        return resp.model_dump(mode="json")

    return send


def anthropic_caller(root=None):
    from .api_layer import default_ledger

    ledger = default_ledger(root)
    caller = ApiCaller("anthropic", config.path("llm_cache", root) / "anthropic", ledger)
    send = anthropic_send_factory(config.require_env("ANTHROPIC_API_KEY"))
    return caller, send, anthropic_estimate_factory(ledger), anthropic_actual_cost_factory(ledger)


def anthropic_response_text(resp: dict) -> str:
    return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")


# ---------------------------------------------------------------------------
# Gemini 免費方案（T09、T18 才會用到；這裡先備好）
# ---------------------------------------------------------------------------

def gemini_send_factory(api_key: str, model: str, max_retries: int = 6, base_sleep: float = 4.0):
    from google import genai

    client = genai.Client(api_key=api_key)

    def send(req: dict) -> dict:
        delay = base_sleep
        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(model=model, contents=req["contents"], config=req.get("config"))
                return {"text": resp.text, "model": model}
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                raise
        raise RuntimeError("Gemini 連續 429，請稍後再跑（免費方案每日上限）。")

    return send
