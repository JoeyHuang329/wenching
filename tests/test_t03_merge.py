"""T03：名稱正規化、去重門檻、非商業過濾、poi_id 穩定性。"""

from __future__ import annotations

import json

import pandas as pd

from dadaocheng.names import name_similarity, normalize_address, normalize_name


def test_normalize_name_rules():
    assert normalize_name("ＡＢＣ咖啡　迪化店") == normalize_name("abc咖啡迪化")
    assert normalize_name("臺灣茶行") == "台灣茶行"
    assert normalize_name("書店") == "書店"          # 不會被砍成一個字
    assert normalize_name("星巴克 本店") == "星巴克"
    assert normalize_name("乾元參藥行") == "乾元參藥行"


def test_similarity_thresholds():
    assert name_similarity("星巴克 迪化店", "星巴克咖啡迪化門市") >= 60
    assert name_similarity("臺北書房", "台北書房") == 100
    assert name_similarity("永樂布業", "霞海城隍廟") < 60


def test_normalize_address():
    assert normalize_address("103台北市大同區迪化街一段61號") == "迪化街1段61號"
    assert normalize_address("臺北市大同區迪化街1段61之1號2樓") == "迪化街1段61-1號"
    assert normalize_address("迪化街一段32巷1號") == "迪化街1段32巷1號"


def _places_row(pid, name, lon, lat, primary="cafe", types=("cafe", "point_of_interest")):
    return {"place_id": pid, "name": name, "lon": lon, "lat": lat, "places_primary_type": primary,
            "places_types": json.dumps(list(types)), "business_status": "OPERATIONAL",
            "user_rating_count": 10, "rating": 4.5, "price_level": None, "website_uri": None,
            "formatted_address": "", "regular_opening_hours": None}


def test_merge_end_to_end(tmp_root):
    from dadaocheng import geo
    from dadaocheng.t03_osm_merge import merge

    (tmp_root / "data" / "interim").mkdir(parents=True)
    geo.make_buffer_geojson()
    lon, lat = 121.5095, 25.0580   # 草稿主範圍內
    places = pd.DataFrame([
        _places_row("A", "小藝埕咖啡", lon, lat),
        _places_row("B", "永樂國小", lon + 0.0005, lat, primary="primary_school", types=("primary_school", "school")),
        _places_row("C", "神秘地點", lon + 0.0007, lat, primary=None, types=("point_of_interest", "establishment")),
        _places_row("D", "遠方咖啡", 121.60, 25.10),   # 範圍外
    ])
    d = 0.00009  # 約 10 公尺
    osm = [
        {"type": "node", "id": 1, "lat": lat + d, "lon": lon, "tags": {"name": "小藝埕 咖啡", "amenity": "cafe"}},   # 合併
        {"type": "node", "id": 2, "lat": lat, "lon": lon + 0.0010, "tags": {"name": "OSM 才有的布行", "shop": "fabric"}},  # 補漏
        {"type": "node", "id": 3, "lat": lat, "lon": lon + 0.0011, "tags": {"name": "某公園", "leisure": "park"}},  # 非商業
        {"type": "node", "id": 4, "lat": lat - d, "lon": lon, "tags": {"name": "小藝埕咖啡館二館", "amenity": "cafe"}},  # 模稜兩可
    ]
    res = merge(places, osm)
    poi = res["poi"]
    assert set(poi["place_id"].dropna()) == {"A", "C"}
    assert poi.loc[poi.place_id == "A", "source"].item() == "both"
    assert poi.loc[poi.place_id == "C", "generic_type"].item()
    assert "OSM 才有的布行" in set(poi["name"])
    assert "某公園" not in set(poi["name"])
    assert poi["in_main"].all()
    assert "B" in set(res["dropped"]["place_id"])
    assert len(res["ambiguous"]) >= 1
    assert poi["poi_id"].str.match(r"P\d{5}").all() and poi["poi_id"].is_unique
    # 重跑 → poi_id 不變
    again = merge(places, osm)["poi"]
    assert dict(zip(poi.name, poi.poi_id)) == dict(zip(again.name, again.poi_id))


def test_human_decisions_applied(tmp_root):
    from dadaocheng import geo
    from dadaocheng.t03_osm_merge import merge

    (tmp_root / "data" / "interim").mkdir(parents=True)
    geo.make_buffer_geojson()
    lon, lat = 121.5095, 25.0580
    places = pd.DataFrame([_places_row("A", "小藝埕咖啡", lon, lat), _places_row("Z", "小藝埕咖啡", lon + 0.0001, lat)])
    osm = [{"type": "node", "id": 4, "lat": lat - 0.00009, "lon": lon, "tags": {"name": "小藝埕咖啡館二館", "amenity": "cafe"}}]
    base = merge(places, osm)
    assert "小藝埕咖啡館二館" not in set(base["poi"]["name"])          # 模稜兩可 → 先不加入
    assert {"A", "Z"} <= set(base["poi"]["place_id"])
    dec = pd.DataFrame([{"kind": "osm_vs_places", "a_id": "node/4", "b_id": "A", "decision": "different"},
                        {"kind": "places_vs_places", "a_id": "A", "b_id": "Z", "decision": "same"}])
    res = merge(places, osm, dec)["poi"]
    assert "小藝埕咖啡館二館" in set(res["name"])
    assert "Z" not in set(res["place_id"])
