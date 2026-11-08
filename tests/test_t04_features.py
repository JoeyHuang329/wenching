"""T04：營業時間（跨午夜、午休、24 小時、缺失）、街區合併、鄰域排除自己、核密度、端到端。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, box

REPO = Path(__file__).resolve().parents[1]


def P(day, h, m=0):
    return {"day": day, "hour": h, "minute": m}


def hours(periods):
    return json.dumps({"periods": periods})


def test_close_time_basic_and_threshold(tmp_root):
    from dadaocheng.t04_features import weekday_close_minutes

    week = [{"open": P(d, 10), "close": P(d, 18)} for d in range(1, 6)]
    assert weekday_close_minutes(hours(week)) == (18 * 60, "ok")


def test_cross_midnight(tmp_root):
    from dadaocheng.t04_features import weekday_close_minutes

    week = [{"open": P(d, 18), "close": P((d + 1) % 7, 2)} for d in range(1, 6)]
    m, st = weekday_close_minutes(hours(week))
    assert m == 26 * 60 and st == "ok"


def test_lunch_break_takes_last_close(tmp_root):
    from dadaocheng.t04_features import weekday_close_minutes

    week = []
    for d in range(1, 6):
        week += [{"open": P(d, 11), "close": P(d, 14)}, {"open": P(d, 17), "close": P(d, 21)}]
    assert weekday_close_minutes(hours(week))[0] == 21 * 60


def test_24h_missing_and_weekday_closed(tmp_root):
    from dadaocheng.t04_features import weekday_close_minutes

    assert weekday_close_minutes(hours([{"open": P(0, 0)}])) == (1440.0, "open_24h")
    assert weekday_close_minutes(None) == (None, "missing")
    assert weekday_close_minutes(hours([])) == (None, "missing")
    weekend = [{"open": P(6, 10), "close": P(6, 18)}, {"open": P(0, 10), "close": P(0, 18)}]
    assert weekday_close_minutes(hours(weekend)) == (None, "weekday_closed")


def test_median_of_weekdays(tmp_root):
    from dadaocheng.t04_features import add_hours

    week = [{"open": P(d, 9), "close": P(d, c)} for d, c in zip(range(1, 6), [17, 18, 18, 20, 21])]
    df = add_hours(pd.DataFrame({"regular_opening_hours": [hours(week), None]}))
    assert df.loc[0, "close_time_weekday"] == "18:00" and df.loc[0, "closes_before_19"] is True
    assert df.loc[1, "closes_before_19"] is None   # 缺失不是「不營業」


def test_context_excludes_self(tmp_root):
    from dadaocheng.t04_features import context_vectors

    x = np.array([0.0, 10.0, 500.0]); y = np.zeros(3)
    cats = np.array(["咖啡茶飲甜點", "熱食餐飲", "咖啡茶飲甜點"])
    ctx = context_vectors(x, y, cats, 100)
    # 第 0 點的鄰居只有第 1 點（熱食），自己不算
    assert ctx.loc[0, "poi_context_n"] == 1
    assert ctx.loc[0, "poi_context_cafe"] == 0 and ctx.loc[0, "poi_context_hotfood"] == 1
    assert np.isnan(ctx.loc[2, "poi_context_cafe"])


def test_kernel_density(tmp_root):
    from dadaocheng.t04_features import kernel_density

    k_near = kernel_density(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 150)
    k_far = kernel_density(np.array([1000.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 150)
    assert k_near[0] > k_far[0] > 0
    assert k_near[0] == pytest.approx(1e6 / (2 * np.pi * 150 ** 2))


def test_block_road_rule(tmp_root):
    from dadaocheng.t04_features import is_block_road

    assert is_block_road("tertiary", "民生西路")
    assert is_block_road("residential", "迪化街一段")
    assert not is_block_road("residential", "迪化街一段32巷")
    assert not is_block_road("service", "某停車場")
    assert not is_block_road("footway", None)


def test_merge_small_blocks(tmp_root):
    from dadaocheng.t04_features import merge_small_blocks

    # 一列四塊，POI 數 3, 20, 4, 30 → 最小的先併到共用邊界最長的鄰居
    geoms = [box(i * 100, 0, (i + 1) * 100, 100) for i in range(4)]
    blocks = gpd.GeoDataFrame({"raw_block": ["R0", "R1", "R2", "R3"]}, geometry=geoms, crs="EPSG:3826")
    counts = pd.Series({"R0": 3, "R1": 20, "R2": 4, "R3": 30})
    m = merge_small_blocks(blocks, counts, 15)
    assert m["R0"] == m["R1"]
    assert m["R2"] in (m["R1"], m["R3"])
    groups = pd.Series(counts.values, index=[m[k] for k in counts.index]).groupby(level=0).sum()
    assert (groups >= 15).all()


def test_places_category_and_subtype(tmp_root):
    from dadaocheng.t04_features import places_category, traditional_subtype

    assert places_category("cafe", '["cafe"]') == "咖啡茶飲甜點"
    assert places_category(None, '["book_store","store"]') == "文化空間"
    assert places_category(None, '["point_of_interest"]') == "其他"
    assert traditional_subtype("乾元參藥行") == "中藥"
    assert traditional_subtype("林復振南北貨") == "南北貨"
    assert traditional_subtype("小藝埕") is None


def test_build_poi_master_end_to_end(tmp_root):
    """用合成的道路格網與 POI 跑一次 T04，檢查欄位齊全、街區 ≥ 15、文資與主街判定合理。"""
    from dadaocheng import geo
    from dadaocheng.t04_features import build_poi_master

    interim = tmp_root / "data" / "interim"
    interim.mkdir(parents=True)
    for f in (REPO / "data" / "interim").glob("transit_*.geojson"):
        shutil.copy(f, interim / f.name)
    geo.make_buffer_geojson()
    main = geo.main_polygon_metric()
    minx, miny, maxx, maxy = main.bounds
    cx = (minx + maxx) / 2

    # 合成道路：一條南北向主街（迪化街一段）、三條東西向的路、一條民生西路
    def ll(coords):
        return LineString([geo.metric_to_lonlat(x, y) for x, y in coords])

    ys = [miny + (maxy - miny) * f for f in (0.25, 0.5, 0.75)]
    roads = gpd.GeoDataFrame(
        {"highway": ["residential", "tertiary", "tertiary", "tertiary", "residential"],
         "name": ["迪化街一段", "甲路", "民生西路", "乙路", "某巷"]},
        geometry=[ll([(cx, miny - 50), (cx, maxy + 50)])] + [ll([(minx - 100, y), (maxx + 100, y)]) for y in ys]
        + [ll([(cx - 50, miny), (cx - 50, maxy)])],
        crs=geo.WGS84,
    )
    rng = np.random.default_rng(0)
    rows = []
    for i in range(240):
        while True:
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            from shapely.geometry import Point
            if main.contains(Point(x, y)):
                break
        lon, lat = geo.metric_to_lonlat(x, y)
        week = [{"open": P(d, 10), "close": P(d, 18 if i % 3 else 21)} for d in range(1, 6)]
        rows.append({"poi_id": f"P{i:05d}", "place_id": f"pl{i}", "osm_id": None, "source": "places", "name": f"店{i}",
                     "lon": lon, "lat": lat, "in_main": True,
                     "places_primary_type": ["cafe", "restaurant", "book_store", "clothing_store"][i % 4],
                     "places_types": "[]", "osm_kind": None, "generic_type": False, "business_status": "OPERATIONAL",
                     "user_rating_count": int(rng.integers(0, 500)), "rating": 4.2,
                     "price_level": "PRICE_LEVEL_MODERATE", "website_uri": None,
                     "formatted_address": f"103台北市大同區迪化街一段{i}號", "regular_opening_hours": hours(week) if i % 10 else None,
                     "osm_opening_hours": None})
    poi = pd.DataFrame(rows)
    h_lon, h_lat = poi.loc[0, "lon"], poi.loc[0, "lat"]
    heritage = pd.DataFrame([
        {"heritage_id": "H1", "name": "某歷史建築", "type": "歷史建築", "address": "", "lon": h_lon, "lat": h_lat},
        {"heritage_id": "H2", "name": "只有地址的古蹟", "type": "古蹟", "address": "臺北市大同區迪化街一段5號", "lon": np.nan, "lat": np.nan},
        {"heritage_id": "H3", "name": "找不到的", "type": "古蹟", "address": "臺北市某路999號", "lon": np.nan, "lat": np.nan},
    ])
    heritage["addr_key"] = heritage["address"].apply(__import__("dadaocheng.names", fromlist=["x"]).normalize_address)

    res = build_poi_master(poi, roads, None, heritage, None)
    df = res["poi_master"]
    for col in ["block_id", "on_main_street", "north_of_minsheng", "in_heritage", "heritage_kde_150", "heritage_kde_300",
                "dist_mrt_m", "bus_density", "dist_temple_m", "closes_before_19", "poi_context_cafe", "price_level_ord"]:
        assert col in df.columns, col
    assert df.loc[df.poi_id == "P00000", "in_heritage"].item()
    assert df.loc[df.poi_id == "P00005", "in_heritage"].item()          # 地址比對
    assert (res["heritage_located"]["method"] == "unlocated").sum() == 1
    assert res["blocks"]["n_analysis_poi"].min() >= 15
    assert df["block_id"].notna().all()
    assert df["on_main_street"].any() and not df["on_main_street"].all()
    assert df["north_of_minsheng"].any() and not df["north_of_minsheng"].all()
    assert df["closes_before_19"].isna().sum() == 24   # 每 10 筆一筆沒有營業時間
    assert (df["dist_mrt_m"] > 0).all()
