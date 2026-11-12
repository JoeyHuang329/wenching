"""T02：自適應網格的切割、最小格、去重與欄位遮罩。用假的 fetch，不呼叫 API。"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import box

from dadaocheng.t02_places_census import Cell, initial_cells, run_census, to_dataframe


def fake_world(n=600, seed=0, size=400):
    r = np.random.default_rng(seed)
    pts = r.uniform(0, size, size=(n, 2))
    # 一個極度密集的熱點（會在最小格仍然滿 20）
    hot = np.column_stack([np.full(40, 10.0) + r.uniform(0, 1, 40), np.full(40, 10.0) + r.uniform(0, 1, 40)])
    return np.vstack([pts, hot])


def make_fetch(points, max_results=20):
    calls = []

    def fetch(cell: Cell):
        calls.append(cell.cell_id)
        cx, cy = cell.center
        d = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
        idx = np.where(d <= cell.radius)[0]
        idx = idx[np.argsort(d[idx])][:max_results]  # DISTANCE 排序取前 20
        return [{"id": f"p{i}", "displayName": {"text": f"店{i}"}, "location": {"longitude": float(points[i, 0]), "latitude": float(points[i, 1])}} for i in idx]

    return fetch, calls


def test_cell_geometry():
    c = Cell("a", 0, 0, 200)
    assert c.radius == pytest.approx(200 * math.sqrt(2) / 2)
    kids = c.split()
    assert len(kids) == 4 and all(k.size == 100 for k in kids)
    assert {k.center for k in kids} == {(50, 50), (150, 50), (50, 150), (150, 150)}


def test_initial_cells_skip_outside():
    poly = box(0, 0, 450, 210)
    cells = initial_cells(poly, 200)
    assert len(cells) == 3 * 2


def test_census_finds_every_point_except_saturated_hotspot():
    pts = fake_world()
    poly = box(0, 0, 400, 400)
    fetch, calls = make_fetch(pts)
    res = run_census(poly, fetch, initial_size=200, min_size=25)
    df = to_dataframe(res)
    found = set(df["place_id"])
    normal = {f"p{i}" for i in range(600)}
    # 一般點全部找到（四分樹覆蓋完整）
    assert normal <= found
    # 熱點在最小格仍滿 20 → 被回報
    assert res.saturated_min_cells >= 1
    # 沒有重複 place_id
    assert df["place_id"].is_unique
    # 最小格不會再切
    assert min(c["size_m"] for c in res.cells) >= 25


def test_max_cells_guard():
    pts = fake_world()
    fetch, calls = make_fetch(pts)
    res = run_census(box(0, 0, 400, 400), fetch, 200, 25, max_cells=5)
    assert len(res.cells) == 5


def test_field_mask_has_no_atmosphere_review_fields(tmp_root):
    from dadaocheng.clients import places_field_mask, places_nearby_request

    mask = places_field_mask()
    assert "reviews" not in mask and "editorialSummary" not in mask
    assert "places.websiteUri" in mask and "places.regularOpeningHours" in mask
    req = places_nearby_request(25.05, 121.51, 141.42)
    assert req["maxResultCount"] == 20 and req["rankPreference"] == "DISTANCE"
    assert "includedTypes" not in req
    assert req["languageCode"] == "zh-TW" and req["regionCode"] == "TW"


def test_cached_cells_are_free_on_rerun(tmp_root):
    """普查重跑時，已查過的格子走快取，不增加請求數。"""
    from dadaocheng.api_layer import ApiCaller, Ledger
    from dadaocheng.clients import places_nearby_request

    ledger = Ledger(tmp_root / "config" / "budget.yaml")
    caller = ApiCaller("google_places", tmp_root / "data" / "places_cache", ledger, max_requests=900)
    sent = []
    send = lambda r: sent.append(r) or {"places": []}
    for _ in range(3):
        caller.call(places_nearby_request(25.05, 121.51, 100), send=send, estimate_cost=lambda r: 0.0)
    assert len(sent) == 1 and ledger.requests("google_places") == 1


def test_priority_polygon_is_resolved_first():
    pts = fake_world()
    fetch, calls = make_fetch(pts)
    main = box(0, 0, 200, 200)
    res = run_census(box(0, 0, 400, 400), fetch, 200, 25, priority_polygon=main)
    flags = [c["in_priority"] for c in res.cells]
    # 所有主範圍的格子都排在非主範圍之前
    first_non = flags.index(False)
    assert all(flags[:first_non]) and not any(flags[first_non:])


def test_limit_marks_pending_and_keeps_results():
    """請求上限用完：已查的結果保留、未查的格子標 pending、主範圍優先所以主範圍沒有 pending。"""
    from dadaocheng.api_layer import RequestLimitExceeded

    pts = fake_world()
    base, calls = make_fetch(pts)
    main = box(0, 0, 200, 200)
    full = run_census(box(0, 0, 400, 400), make_fetch(pts)[0], 200, 25, priority_polygon=main)
    n_main = sum(1 for c in full.cells if c["in_priority"])
    budget = {"n": n_main + 5}

    def limited(cell):
        if budget["n"] <= 0:
            raise RequestLimitExceeded("limit")
        budget["n"] -= 1
        return base(cell)

    res = run_census(box(0, 0, 400, 400), limited, 200, 25, priority_polygon=main)
    assert len(res.pending) > 0
    assert not any(c["in_priority"] for c in res.pending)
    assert len([c for c in res.cells if not c["pending"]]) == n_main + 5
    df = to_dataframe(res)
    assert len(df) > 0 and df["place_id"].is_unique


def test_fixed_origin_keeps_cells_aligned():
    """原點固定：同一個範圍結果與舊版相同；縮小或改邊界時，格子仍在同一張格網上（快取可沿用）。"""
    big = box(3.3, 7.1, 812.9, 433.0)
    old = initial_cells(big, 200)                       # 舊行為：原點＝外框左下角
    org = (big.bounds[0], big.bounds[1])
    new = initial_cells(big, 200, origin=org)
    assert [c.center for c in old] == [c.center for c in new]
    small = box(250.0, 100.0, 500.0, 300.0)
    centers_big = {c.center for c in new}
    sub = initial_cells(small, 200, origin=org)
    assert sub and all(c.center in centers_big for c in sub)
    wider = box(-300.0, -50.0, 900.0, 500.0)              # 邊界往外擴：舊格子仍是子集
    assert centers_big <= {c.center for c in initial_cells(wider, 200, origin=org)}
