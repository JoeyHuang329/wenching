"""T00：所有付費 API 共用的呼叫層（CLAUDE.md 規則 4）。

流程（每一次呼叫）：
  1. 以「provider + 請求內容（+ extra，例如第幾次 run）」的標準化 JSON 計算 SHA-256 當快取鍵。
  2. 有快取 → 直接回傳，不送出、不扣額度，只記一次 cache_hit。
  3. 沒快取 → 預估花費 → 檢查上限、七成門檻、請求數上限 → 送出。
  4. 成功才寫快取、記花費（用實際用量算，沒有就用預估）。失敗不寫快取、不扣錢。

「同樣的請求不可以付費兩次」由第 1–2 步保證。
「單一步驟 > 200 元要停下來問人」由 Ledger.check_step() 保證，請在每個會花錢的迴圈之前呼叫一次。

注意：本模組假設同一時間只有一個會花錢的程序在跑（budget.yaml 沒有跨程序鎖）。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import yaml


class BudgetExceeded(RuntimeError):
    """預估花費會讓某個帳戶超過硬上限。"""


class RequestLimitExceeded(BudgetExceeded):
    """請求數超過上限（例如 Places 900 次）。"""


class NeedsHumanApproval(RuntimeError):
    """CLAUDE.md「必須停下來問人」的情況：單一步驟 > 200 元，或累計達上限七成。"""


# ---------------------------------------------------------------------------
# 快取鍵
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def request_key(provider: str, request: dict, extra: dict | None = None) -> str:
    payload = {"provider": provider, "request": request, "extra": extra or {}}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 帳本（budget.yaml）
# ---------------------------------------------------------------------------

class Ledger:
    def __init__(self, budget_path: Path | str):
        self.path = Path(budget_path)
        self._load()

    # -- 讀寫 --
    def _load(self) -> None:
        with open(self.path, encoding="utf-8") as f:
            self.state: dict[str, Any] = yaml.safe_load(f) or {}
        for k in ("spent_twd", "requests", "cache_hits"):
            self.state.setdefault(k, {})

    def _save(self) -> None:
        # 原子寫入：先寫暫存檔再取代，避免中途當掉留下壞檔
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".budget_", suffix=".yaml")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.state, f, allow_unicode=True, sort_keys=False)
        os.replace(tmp, self.path)

    # -- 查詢 --
    def limit(self, provider: str) -> float:
        return float(self.state.get("limits_twd", {}).get(provider, 0.0))

    def spent(self, provider: str) -> float:
        return float(self.state["spent_twd"].get(provider, 0.0))

    def requests(self, provider: str) -> int:
        return int(self.state["requests"].get(provider, 0))

    @property
    def fx(self) -> float:
        return float(self.state.get("fx_usd_twd", 32))

    def pricing(self, provider: str) -> dict:
        return self.state.get("pricing", {}).get(provider, {})

    # -- 檢查 --
    def precheck(self, provider: str, est_twd: float, approved: bool = False) -> None:
        limit, spent = self.limit(provider), self.spent(provider)
        if spent + est_twd > limit + 1e-9:
            raise BudgetExceeded(
                f"[{provider}] 已花 {spent:.2f} 元＋預估 {est_twd:.2f} 元 會超過上限 {limit:.0f} 元。"
            )
        ratio = float(self.state.get("approval_ratio", 0.7))
        if limit > 0 and est_twd > 0 and spent + est_twd >= ratio * limit and not approved:
            raise NeedsHumanApproval(
                f"[{provider}] 累計將達上限的 {ratio:.0%}（{spent + est_twd:.2f} / {limit:.0f} 元）。請人確認後以 approved=True 重跑。"
            )

    def check_step(self, provider: str, est_twd: float, step: str, approved: bool = False) -> None:
        """在一個會花錢的步驟（整個迴圈、整批）開始之前呼叫。"""
        threshold = float(self.state.get("step_approval_twd", 200))
        if self.spent(provider) + est_twd > self.limit(provider) + 1e-9:
            raise BudgetExceeded(f"步驟「{step}」預估 {est_twd:.2f} 元，加上已花費會超過 {provider} 上限。")
        if est_twd > threshold and not approved:
            raise NeedsHumanApproval(
                f"步驟「{step}」預估 {est_twd:.2f} 元 > {threshold:.0f} 元。請人確認後加上 --approve-step 重跑。"
            )
        self.precheck(provider, est_twd, approved=approved)

    # -- 記帳 --
    def record(self, provider: str, cost_twd: float) -> None:
        self._load()  # 盡量讀最新狀態
        self.state["spent_twd"][provider] = round(self.spent(provider) + float(cost_twd), 6)
        self.state["requests"][provider] = self.requests(provider) + 1
        self._save()

    def record_hit(self, provider: str) -> None:
        self._load()
        self.state["cache_hits"][provider] = int(self.state["cache_hits"].get(provider, 0)) + 1
        self._save()

    def set_spent_for_test(self, provider: str, value: float) -> None:
        self.state["spent_twd"][provider] = value
        self._save()

    def summary(self) -> str:
        lines = []
        for p in self.state.get("limits_twd", {}):
            lines.append(
                f"{p:15s} 已花 {self.spent(p):8.2f} / 上限 {self.limit(p):6.0f} 元，"
                f"請求 {self.requests(p)} 次，快取命中 {self.state['cache_hits'].get(p, 0)} 次"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 呼叫層
# ---------------------------------------------------------------------------

class ApiCaller:
    """包住某一個 provider 的所有付費呼叫。

    send(request) -> response：真正送出請求的函式（由各 client 提供，測試時可換成假的）。
    estimate_cost(request) -> 新台幣：送出前的保守預估。
    actual_cost(request, response) -> 新台幣：依回應中的實際用量計算（可省略）。
    """

    def __init__(self, provider: str, cache_dir: Path | str, ledger: Ledger, max_requests: int | None = None):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.ledger = ledger
        self.max_requests = max_requests

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, request: dict, extra: dict | None = None) -> dict | None:
        p = self._cache_path(request_key(self.provider, request, extra))
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        return None

    def call(
        self,
        request: dict,
        send: Callable[[dict], dict],
        estimate_cost: Callable[[dict], float],
        actual_cost: Callable[[dict, dict], float] | None = None,
        extra: dict | None = None,
        approved: bool = False,
    ) -> dict:
        key = request_key(self.provider, request, extra)
        path = self._cache_path(key)
        if path.exists():
            self.ledger.record_hit(self.provider)
            with open(path, encoding="utf-8") as f:
                return json.load(f)["response"]

        if self.max_requests is not None and self.ledger.requests(self.provider) >= self.max_requests:
            raise RequestLimitExceeded(
                f"[{self.provider}] 已送出 {self.ledger.requests(self.provider)} 次，達上限 {self.max_requests}。請停下來回報。"
            )
        est = float(estimate_cost(request))
        self.ledger.precheck(self.provider, est, approved=approved)

        response = send(request)  # 失敗會直接丟出例外：不寫快取、不扣錢
        cost = float(actual_cost(request, response)) if actual_cost else est

        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "provider": self.provider,
            "extra": extra or {},
            "request": request,
            "response": response,
            "cost_twd": cost,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
        os.replace(tmp, path)
        self.ledger.record(self.provider, cost)
        return response


def default_ledger(root: Path | None = None) -> Ledger:
    from . import config

    return Ledger(config.path("budget", root))
