"""T00 驗收：快取不重複扣款、超過上限丟例外、需要人工核准的情況會停下來。"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from dadaocheng.api_layer import (
    ApiCaller,
    BudgetExceeded,
    Ledger,
    NeedsHumanApproval,
    RequestLimitExceeded,
    request_key,
)

REPO = Path(__file__).resolve().parents[1]


def make_caller(root, provider="anthropic", max_requests=None):
    ledger = Ledger(root / "config" / "budget.yaml")
    return ApiCaller(provider, root / "data" / "cache" / provider, ledger, max_requests=max_requests), ledger


def test_request_key_is_order_independent_and_run_sensitive():
    a = request_key("anthropic", {"b": 1, "a": [1, 2]})
    b = request_key("anthropic", {"a": [1, 2], "b": 1})
    assert a == b
    assert request_key("anthropic", {"a": 1}, extra={"run": 0}) != request_key("anthropic", {"a": 1}, extra={"run": 1})
    assert request_key("anthropic", {"a": 1}) != request_key("gemini", {"a": 1})


def test_second_identical_call_hits_cache_and_costs_nothing(tmp_root):
    caller, ledger = make_caller(tmp_root)
    calls = []

    def send(req):
        calls.append(req)
        return {"ok": True, "n": len(calls)}

    r1 = caller.call({"q": "同一個請求"}, send=send, estimate_cost=lambda r: 1.5)
    r2 = caller.call({"q": "同一個請求"}, send=send, estimate_cost=lambda r: 1.5)
    assert r1 == r2 == {"ok": True, "n": 1}
    assert len(calls) == 1
    state = yaml.safe_load((tmp_root / "config" / "budget.yaml").read_text(encoding="utf-8"))
    assert state["spent_twd"]["anthropic"] == pytest.approx(1.5)
    assert state["requests"]["anthropic"] == 1
    assert state["cache_hits"]["anthropic"] == 1


def test_cached_returns_same_shape_as_call(tmp_root):
    """cached() 必須和 call() 回傳同樣的東西，否則「先查快取再送出」的流程會拿到不同結構。"""
    caller, _ = make_caller(tmp_root)
    resp = caller.call({"q": 1}, send=lambda r: {"content": [{"text": "hi"}]}, estimate_cost=lambda r: 0.1)
    assert caller.cached({"q": 1}) == resp
    rec = caller.cached_record({"q": 1})
    assert rec["response"] == resp and rec["cost_twd"] == pytest.approx(0.1) and "fetched_at" in rec
    assert caller.cached({"q": 2}) is None


def test_store_then_cached_roundtrip(tmp_root):
    """批次介面寫入的結果，要能像 call() 的結果一樣讀回來，而且不會重複記帳。"""
    caller, ledger = make_caller(tmp_root)
    caller.store({"q": "batch"}, {"content": [{"text": "x"}]}, 0.25, extra={"run": 1})
    assert caller.cached({"q": "batch"}, extra={"run": 1}) == {"content": [{"text": "x"}]}
    assert caller.cached({"q": "batch"}, extra={"run": 0}) is None   # run 不同＝不同快取
    assert ledger.spent("anthropic") == pytest.approx(0.25)
    caller.store({"q": "batch"}, {"content": []}, 0.25, extra={"run": 1})
    assert ledger.spent("anthropic") == pytest.approx(0.25)


def test_actual_cost_overrides_estimate(tmp_root):
    caller, ledger = make_caller(tmp_root)
    caller.call({"q": 1}, send=lambda r: {"usage": 3}, estimate_cost=lambda r: 5.0, actual_cost=lambda req, resp: 0.25)
    assert ledger.spent("anthropic") == pytest.approx(0.25)


def test_exceeding_limit_raises_and_does_not_send(tmp_root):
    caller, ledger = make_caller(tmp_root)
    ledger.set_spent_for_test("anthropic", 999.0)
    sent = []
    with pytest.raises(BudgetExceeded):
        caller.call({"q": "x"}, send=lambda r: sent.append(r) or {}, estimate_cost=lambda r: 5.0, approved=True)
    assert sent == []


def test_seventy_percent_requires_human(tmp_root):
    caller, ledger = make_caller(tmp_root)
    ledger.set_spent_for_test("anthropic", 699.0)  # 上限 1000 的七成 = 700
    with pytest.raises(NeedsHumanApproval):
        caller.call({"q": "y"}, send=lambda r: {}, estimate_cost=lambda r: 2.0)
    # 人核准後可以繼續（仍不得超過上限）
    caller.call({"q": "y"}, send=lambda r: {"ok": 1}, estimate_cost=lambda r: 2.0, approved=True)


def test_step_over_200_twd_requires_human(tmp_root):
    _, ledger = make_caller(tmp_root)
    with pytest.raises(NeedsHumanApproval):
        ledger.check_step("anthropic", 250.0, step="T08 全量批次")
    ledger.check_step("anthropic", 250.0, step="T08 全量批次", approved=True)
    ledger.check_step("anthropic", 50.0, step="T06 開發樣本")


def test_step_over_limit_raises_even_if_approved(tmp_root):
    _, ledger = make_caller(tmp_root)
    with pytest.raises(BudgetExceeded):
        ledger.check_step("anthropic", 1500.0, step="過大", approved=True)


def test_request_count_limit_for_places(tmp_root):
    caller, ledger = make_caller(tmp_root, provider="google_places", max_requests=2)
    for i in range(2):
        caller.call({"cell": i}, send=lambda r: {"places": []}, estimate_cost=lambda r: 0.0)
    with pytest.raises(RequestLimitExceeded):
        caller.call({"cell": 99}, send=lambda r: {"places": []}, estimate_cost=lambda r: 0.0)
    # 已快取的請求不算新請求
    caller.call({"cell": 0}, send=lambda r: {"places": []}, estimate_cost=lambda r: 0.0)


def test_free_provider_with_zero_limit(tmp_root):
    caller, ledger = make_caller(tmp_root, provider="gemini")
    caller.call({"q": "free"}, send=lambda r: {"ok": 1}, estimate_cost=lambda r: 0.0)
    with pytest.raises(BudgetExceeded):
        caller.call({"q": "paid?"}, send=lambda r: {}, estimate_cost=lambda r: 0.01)


def test_failed_send_is_not_cached_or_charged(tmp_root):
    caller, ledger = make_caller(tmp_root)

    def boom(req):
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        caller.call({"q": "fail"}, send=boom, estimate_cost=lambda r: 1.0)
    assert ledger.spent("anthropic") == 0
    assert caller.cached({"q": "fail"}) is None


def test_env_and_data_are_gitignored():
    """驗收：.env 與 data/ 不在 git 中。"""
    for target in [".env", "data/raw/x.csv", "data/places_cache/a.json"]:
        res = subprocess.run(["git", "check-ignore", "-q", target], cwd=REPO)
        assert res.returncode == 0, f"{target} 沒有被 .gitignore 排除"
    res = subprocess.run(["git", "check-ignore", "-q", ".env.example"], cwd=REPO)
    assert res.returncode == 1, ".env.example 應該要能 commit"


def test_save_survives_windows_file_lock(tmp_root, monkeypatch):
    """Windows：budget.yaml 被其他程式短暫鎖住時，os.replace 會丟 PermissionError；應重試成功。"""
    import os as _os
    from dadaocheng import api_layer

    real = _os.replace
    fails = {"n": 3}

    def flaky(a, b):
        if fails["n"] > 0 and str(b).endswith("budget.yaml"):
            fails["n"] -= 1
            raise PermissionError(5, "存取被拒")
        return real(a, b)

    monkeypatch.setattr(api_layer.os, "replace", flaky)
    monkeypatch.setattr(api_layer.time, "sleep", lambda s: None)
    caller, ledger = make_caller(tmp_root)
    caller.call({"q": "lock"}, send=lambda r: {"ok": 1}, estimate_cost=lambda r: 1.0)
    assert Ledger(tmp_root / "config" / "budget.yaml").requests("anthropic") == 1
    assert not list((tmp_root / "config").glob(".budget_*.yaml"))


def test_save_falls_back_when_lock_persists(tmp_root, monkeypatch):
    from dadaocheng import api_layer

    real = api_layer.os.replace
    monkeypatch.setattr(api_layer.os, "replace",
                        lambda a, b: (_ for _ in ()).throw(PermissionError(5, "x")) if str(b).endswith("budget.yaml") else real(a, b))
    monkeypatch.setattr(api_layer.time, "sleep", lambda s: None)
    caller, ledger = make_caller(tmp_root)
    caller.call({"q": "lock2"}, send=lambda r: {"ok": 1}, estimate_cost=lambda r: 1.0)
    assert Ledger(tmp_root / "config" / "budget.yaml").requests("anthropic") == 1
    assert not list((tmp_root / "config").glob(".budget_*.yaml"))
