"""文資與廟宇資料的載入、清理、百年篩選、驗證，以及向量化的 POI 配對。"""

from __future__ import annotations

import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon


def write_heritage_csv(root, rows):
    d = root / "data" / "raw" / "heritage"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(d / "heritage_list.csv", index=False, encoding="utf-8-sig")
    return d


def boch_record(case_id, name, city, dist, addr, lon=None, lat=None):
    rec = {"caseId": case_id, "caseName": name,
           "addresses": [{"itemNo": 1, "cityName": city, "distName": dist, "address": addr}]}
    if lon is not None:
        rec["longitude"], rec["latitude"] = lon, lat
    return rec


def test_boch_json_city_district_filter(tmp_root):
    """行政區名稱會跨縣市重複（基隆市也有中正區），所以必須先過縣市。"""
    from dadaocheng.heritage import load_heritage

    d = tmp_root / "data" / "raw" / "heritage"
    d.mkdir(parents=True)
    recs = [boch_record("1", "大稻埕某店屋", "臺北市", "大同區", "迪化街1段100號", 121.5097, 25.0570),
            boch_record("2", "基隆某廠", "基隆市", "中正區", "某路1號", 121.77, 25.13),
            boch_record("3", "台中某校", "臺中市", "南屯區", "某路2號", 120.64, 24.14),
            boch_record("4", "中山區某宅", "臺北市", "中山區", "某路3號", 121.52, 25.06)]
    (d / "boch_1.2.json").write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    h = load_heritage(verbose=False)
    assert set(h["heritage_id"]) == {"1", "4"}
    assert (h["type"] == "歷史建築").all()
    assert h.loc[h.heritage_id == "1", "address"].item().startswith("臺北市大同區")


def test_csv_and_json_merge_dedup(tmp_root):
    """同一 heritage_id 同時出現時以有座標者為準；JSON 補上 CSV 沒有的筆數。"""
    from dadaocheng.heritage import load_heritage

    d = write_heritage_csv(tmp_root, [
        {"heritage_id": "1", "name": "CSV 版", "type": "歷史建築",
         "address": "臺北市大同區迪化街1段100號", "lon": None, "lat": None},
    ])
    (d / "boch_1.1.json").write_text(json.dumps([
        boch_record("1", "JSON 版", "臺北市", "大同區", "迪化街1段100號", 121.5097, 25.0570),
        boch_record("9", "只有 JSON 的古蹟", "臺北市", "大同區", "貴德街73號", 121.5085, 25.0565),
    ], ensure_ascii=False), encoding="utf-8")
    h = load_heritage(verbose=False)
    assert len(h) == 2
    one = h[h.heritage_id == "1"].iloc[0]
    assert one["name"] == "JSON 版" and one["lon"] == pytest.approx(121.5097)
    assert (h["classify"] == "1.1").sum() == 2


def test_missing_monument_class_is_a_stop(tmp_root):
    from dadaocheng.heritage import load_heritage, validate

    write_heritage_csv(tmp_root, [{"heritage_id": "1", "name": "某店屋", "type": "歷史建築",
                                   "address": "臺北市大同區迪化街1段100號", "lon": 121.5097, "lat": 25.0570}])
    v = validate(load_heritage(verbose=False), pd.DataFrame())
    stop = v[v.level == "stop"]
    assert len(stop) == 1 and "1.1" in stop.iloc[0]["item"]


def test_temples_cleaning_and_centenarian_filter(tmp_root):
    """表頭誤貼成資料列、外縣市座標、未滿百年、年份未知，四種都要正確處理。"""
    from dadaocheng.heritage import load_temples

    d = tmp_root / "data" / "raw" / "heritage"
    d.mkdir(parents=True, exist_ok=True)
    (d / "temples.csv").write_text(
        "name,lon,lat,founded_year,source\n"
        "name,lon,lat,founded_year,source\n"              # 誤貼的表頭
        "霞海城隍廟,121.51015,25.05568,1859,x\n"           # 百年、範圍內 → 納入
        "真聖堂,121.5121658,25.0554591,1912,x\n"           # 1912 → 納入
        "萬和宮,120.6389131,24.1379245,1840,x\n"           # 台中座標 → 排除
        "新廟,121.5100,25.0570,1988,x\n"                   # 未滿百年 → 排除
        "年份不明廟,121.5100,25.0575,,x\n",                 # 年份未知 → 排除
        encoding="utf-8")
    keep, allt = load_temples(verbose=False)
    assert set(keep["name"]) == {"霞海城隍廟", "真聖堂"}
    assert len(allt) == 5 and "name" not in set(allt["name"])
    r = allt.set_index("name")
    assert not r.loc["萬和宮", "coord_ok"]
    assert "未滿百年" in r.loc["新廟", "exclude_reason"]
    assert "年份未知" in r.loc["年份不明廟", "exclude_reason"]


def test_temple_wrong_coord_is_a_stop(tmp_root):
    from dadaocheng.heritage import load_temples, validate

    d = tmp_root / "data" / "raw" / "heritage"
    d.mkdir(parents=True, exist_ok=True)
    (d / "temples.csv").write_text("name,lon,lat,founded_year,source\n萬和宮,120.638,24.137,1840,x\n", encoding="utf-8")
    write_heritage_csv(tmp_root, [{"heritage_id": "1", "name": "a", "type": "古蹟",
                                   "address": "臺北市大同區迪化街1段1號", "lon": 121.5097, "lat": 25.0570},
                                  {"heritage_id": "2", "name": "b", "type": "歷史建築",
                                   "address": "臺北市大同區迪化街1段3號", "lon": 121.5098, "lat": 25.0571}])
    _, allt = load_temples(verbose=False)
    v = validate(__import__("dadaocheng.heritage", fromlist=["x"]).load_heritage(verbose=False), allt)
    assert any("萬和宮" in i for i in v[v.level == "stop"]["item"])


def test_match_heritage_three_paths(tmp_root):
    """輪廓內、距離內、地址相符三條路徑，以及超過門檻的不配對。"""
    from dadaocheng import geo
    from dadaocheng.t04_features import match_heritage

    lon, lat = 121.5097, 25.0570
    x, y = geo.lonlat_to_metric(lon, lat)
    def off(dx, dy):
        return geo.metric_to_lonlat(x + dx, y + dy)

    h = pd.DataFrame([
        {"heritage_id": "H1", "name": "有座標的店屋", "type": "歷史建築", "address": "臺北市大同區迪化街1段1號",
         "lon": lon, "lat": lat, "addr_key": "迪化街1段1號"},
        {"heritage_id": "H2", "name": "沒座標的店屋", "type": "古蹟", "address": "臺北市大同區迪化街1段99號",
         "lon": None, "lat": None, "addr_key": "迪化街1段99號"},
    ])
    pts, addrs = [], []
    for dx, dy, a in [(0, 0, "無關路1號"), (10, 8, "無關路2號"), (40, 0, "無關路3號"),
                      (300, 300, "臺北市大同區迪化街1段99號")]:
        lo, la = off(dx, dy)
        pts.append((lo, la)); addrs.append(a)
    pois = geo.points_gdf(pd.DataFrame({
        "poi_id": ["A", "B", "C", "D"], "name": ["在上面", "同棟", "四十公尺外", "地址相符"],
        "lon": [p[0] for p in pts], "lat": [p[1] for p in pts], "formatted_address": addrs}))
    ring = [off(dx, dy) for dx, dy in [(-12, -12), (14, -12), (14, 14), (-12, 14), (-12, -12)]]
    blds = gpd.GeoDataFrame({"osm_id": ["way/1"]}, geometry=[Polygon(ring)], crs=geo.WGS84)

    located, m = match_heritage(pois, h, blds)
    assert set(located["method"]) == {"coord+footprint", "address"}
    got = set(m["poi_id"])
    assert {"A", "B", "D"} <= got and "C" not in got
    assert m.loc[m.poi_id == "B", "in_footprint"].item()
    assert m.loc[m.poi_id == "D", "address_match"].item()
    assert m.loc[m.poi_id == "A", "dist_m"].item() == pytest.approx(0.0, abs=0.5)
    assert m["poi_id"].is_unique or not m.duplicated(["poi_id", "heritage_id"]).any()


def test_match_heritage_without_buildings(tmp_root):
    """OSM 沒有建物輪廓時也要能跑（只靠距離與地址）。"""
    from dadaocheng import geo
    from dadaocheng.t04_features import match_heritage

    h = pd.DataFrame([{"heritage_id": "H1", "name": "x", "type": "歷史建築", "address": "a",
                       "lon": 121.5097, "lat": 25.0570, "addr_key": "a"}])
    pois = geo.points_gdf(pd.DataFrame({"poi_id": ["A"], "name": ["n"], "lon": [121.5097], "lat": [25.0570],
                                        "formatted_address": ["zzz"]}))
    located, m = match_heritage(pois, h, None)
    assert located.iloc[0]["method"] == "coord" and len(m) == 1
