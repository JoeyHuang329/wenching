"""缺值（NaN）處理的回歸測試。

背景：pandas 讀 parquet 回來的文字欄缺值是 `float('nan')`，而 **bool(float('nan')) 是 True**。
所以 `if v:` 會把缺值當成有值，接著呼叫 `.partition()`、`re.search()` 之類就會丟 AttributeError
或 TypEerror。2026-11-12 實際跑 T04 時就是這樣掛掉的（osm_kind 與道路 name 兩處）。

真實資料一定會有缺值：
  - Places 來源的 POI 沒有 osm_kind；OSM 來源的沒有 places_primary_type、評論數、營業時間
  - OSM 的巷弄常常沒有 name
  - 評論數 74%、營業時間 67%、官網 43%、價位 11% 才有值

所以每個「接受可能缺值欄位」的函式都要同時測 None、NaN、空字串、字串 "nan"。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MISSING = [None, float("nan"), np.nan, "", "  ", "nan", "NaN", "None", "<NA>", pd.NA]


def test_as_text_handles_every_missing_form(tmp_root):
    from dadaocheng.names import as_text

    for v in MISSING:
        assert as_text(v) == "", repr(v)
    assert as_text("台北") == "台北"
    assert as_text("  台北  ") == "台北"
    assert as_text(123) == "123"


@pytest.mark.parametrize("v", MISSING)
def test_places_category_survives_missing(tmp_root, v):
    """2026-11-12 的崩潰點：osm_kind 是 NaN 時 `if osm_kind:` 會通過，然後 .partition() 掛掉。"""
    from dadaocheng.t04_features import places_category

    assert places_category(v, v, v) == "其他"
    # 類別對不上任何對照表時會走到 osm_kind 分支，這是當初沒被測到的路徑
    assert places_category("lodging", '["lodging","point_of_interest"]', v) == "其他"
    assert places_category(v, v, "shop=fabric") == "一般零售與品牌"
    assert places_category("cafe", v, v) == "咖啡茶飲甜點"


@pytest.mark.parametrize("v", MISSING)
def test_is_block_road_survives_missing_name(tmp_root, v):
    """第二個崩潰點：OSM 巷弄沒有 name，re.search(NaN) 會丟 TypeError。"""
    from dadaocheng.t04_features import is_block_road

    assert is_block_road("residential", v) is False
    assert is_block_road(v, v) is False
    assert is_block_road("tertiary", v) is True      # 大路不看名稱
    assert is_block_road(v, "民生西路") is False      # 沒有 highway 等級就不切


@pytest.mark.parametrize("v", MISSING)
def test_other_entry_points_survive_missing(tmp_root, v):
    from dadaocheng.extraction import user_content
    from dadaocheng.heritage import split_city_district
    from dadaocheng.names import name_similarity, normalize_address, normalize_name
    from dadaocheng.t04_features import likely_wenqing, parse_types, traditional_subtype

    # 缺值不能變成字串 "nan"：否則所有缺地址的 POI 會共用同一個地址鍵
    assert normalize_name(v) == ""
    assert normalize_address(v) == ""
    assert split_city_district(v) == ("", "")
    assert traditional_subtype(v) is None
    assert likely_wenqing(v, v) is False
    assert parse_types(v) == []
    assert name_similarity(v, v) == 0.0
    assert name_similarity(v, "小藝埕") == 0.0
    out = user_content(v, v, v, v)
    assert "nan" not in out.lower() and "店家文本：（無）" in out


def test_parse_types_accepts_all_shapes(tmp_root):
    from dadaocheng.t04_features import parse_types

    assert parse_types('["cafe","food"]') == ["cafe", "food"]
    assert parse_types(["cafe", "food"]) == ["cafe", "food"]
    assert parse_types(np.array(["cafe", "food"])) == ["cafe", "food"]
    assert parse_types("cafe, food") == ["cafe", "food"]
    assert parse_types("[壞掉的 json") == []
    assert parse_types('["cafe", null]') == ["cafe"]


def test_missing_address_keys_do_not_collide(tmp_root):
    """缺地址的 POI 不能彼此「地址相同」，也不能和文資配上。"""
    from dadaocheng import geo
    from dadaocheng.t04_features import match_heritage

    h = pd.DataFrame([{"heritage_id": "H1", "name": "沒座標的古蹟", "type": "古蹟",
                       "address": "臺北市大同區迪化街1段99號", "lon": None, "lat": None,
                       "addr_key": "迪化街1段99號"}])
    pois = geo.points_gdf(pd.DataFrame({
        "poi_id": ["A", "B"], "name": ["缺地址一", "缺地址二"],
        "lon": [121.5097, 121.5098], "lat": [25.0570, 25.0571],
        "formatted_address": [np.nan, None]}))
    located, m = match_heritage(pois, h, None)
    assert located.iloc[0]["method"] == "unlocated"
    assert m.empty


def test_build_poi_master_with_realistic_missing(tmp_root):
    """整合：Places 與 OSM 混合、未對應類別、無名巷弄、缺營業時間與價位，全部走完 T04。"""
    import shutil

    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    from dadaocheng import geo
    from dadaocheng.t04_features import build_poi_master

    repo = Path(__file__).resolve().parents[1]
    interim = tmp_root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    for f in (repo / "data" / "interim").glob("transit_*.geojson"):
        shutil.copy(f, interim / f.name)
    if not (interim / "transit_mrt_stations.geojson").exists():
        pytest.skip("需要先跑 python -m dadaocheng.transit 產生交通圖層")
    geo.make_buffer_geojson()
    main = geo.main_polygon_metric()
    minx, miny, maxx, maxy = main.bounds
    cx = (minx + maxx) / 2
    rng = np.random.default_rng(0)

    rows = []
    for i in range(240):
        while True:
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if main.contains(Point(x, y)):
                break
        lon, lat = geo.metric_to_lonlat(x, y)
        is_osm = i % 7 == 0
        unmapped = i % 5 == 0
        has_hours = i % 3 != 0
        week = [{"open": {"day": d, "hour": 11, "minute": 0},
                 "close": {"day": d, "hour": 18, "minute": 30}} for d in range(1, 6)]
        rows.append({
            "poi_id": f"P{i:05d}",
            "place_id": np.nan if is_osm else f"ChIJ{i}",
            "osm_id": f"node/{i}" if is_osm else np.nan,
            "source": "osm" if is_osm else "places",
            "name": f"店{i}" if i % 11 else np.nan,
            "lon": lon, "lat": lat, "in_main": True,
            "places_primary_type": np.nan if (is_osm or unmapped) else "cafe",
            "places_types": np.nan if is_osm else ('["lodging"]' if unmapped else '["cafe"]'),
            "osm_kind": "shop=fabric" if is_osm else np.nan,
            "generic_type": False,
            "business_status": "UNKNOWN_OSM" if is_osm else "OPERATIONAL",
            "user_rating_count": np.nan if is_osm else int(rng.integers(0, 500)),
            "rating": np.nan, "price_level": np.nan if i % 9 else "PRICE_LEVEL_MODERATE",
            "website_uri": np.nan,
            "formatted_address": np.nan if is_osm else f"103臺北市大同區迪化街1段{i}號",
            "regular_opening_hours": json.dumps({"periods": week}) if has_hours else np.nan,
            "osm_opening_hours": np.nan,
        })
    poi = pd.DataFrame(rows)

    def ll(c):
        return LineString([geo.metric_to_lonlat(x, y) for x, y in c])

    roads = [{"highway": "residential", "name": "迪化街一段", "geometry": ll([(cx, miny - 50), (cx, maxy + 50)])}]
    for k, f in enumerate([0.25, 0.5, 0.75]):
        y = miny + (maxy - miny) * f
        roads.append({"highway": "tertiary", "name": "民生西路" if k == 1 else f"路{k}",
                      "geometry": ll([(minx - 100, y), (maxx + 100, y)])})
    for k in range(30):     # 無名巷弄：以前會讓 is_block_road 掛掉
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        roads.append({"highway": "service", "name": np.nan, "geometry": ll([(x, y), (x + 30, y + 20)])})
    roads = gpd.GeoDataFrame(roads, geometry="geometry", crs=geo.WGS84)
    bld = gpd.GeoDataFrame({"osm_id": ["way/1"], "name": [np.nan], "historic": [np.nan], "heritage": [np.nan]},
                           geometry=[Polygon([geo.metric_to_lonlat(a, b) for a, b in
                                              [(cx - 10, miny + 10), (cx + 10, miny + 10),
                                               (cx + 10, miny + 30), (cx - 10, miny + 30), (cx - 10, miny + 10)]])],
                           crs=geo.WGS84)
    heritage = pd.DataFrame([
        {"heritage_id": "H1", "name": "古蹟一", "type": "古蹟", "address": "臺北市大同區迪化街1段5號",
         "lon": poi.loc[5, "lon"], "lat": poi.loc[5, "lat"], "addr_key": "迪化街1段5號"},
        {"heritage_id": "H2", "name": "沒座標的", "type": "歷史建築", "address": "臺北市大同區迪化街1段9號",
         "lon": np.nan, "lat": np.nan, "addr_key": "迪化街1段9號"},
    ])
    temples = pd.DataFrame([{"name": "某百年廟", "lon": poi.loc[10, "lon"], "lat": poi.loc[10, "lat"],
                             "founded_year": 1880}])

    res = build_poi_master(poi, roads, bld, heritage, temples)
    df = res["poi_master"]
    assert len(df) == 240
    assert df["places_category"].notna().all()
    assert (df["places_category"] == "其他").any()          # 未對應的類別落到「其他」
    assert df["block_id"].notna().all()
    assert df.loc[df.poi_id == "P00005", "in_heritage"].item()
    assert df.loc[df.poi_id == "P00009", "in_heritage"].item()   # 靠地址配上
    assert (df["heritage_kde_300"] > 0).all()
    # 缺營業時間要留成缺失，不能變 False
    assert df["closes_before_19"].isna().sum() == sum(1 for i in range(240) if i % 3 == 0)
    assert df["price_level_ord"].notna().sum() == sum(1 for i in range(240) if i % 9 == 0)
    assert df["dist_mrt_m"].notna().all()
