"""scripts/build_boundary.py：從 OSM 道路中心線建邊界。

用合成的 OSM 資料測試，因為真實的 osm_raw.json 不在測試環境裡。合成資料的四條路
用的是從文資／寺廟／捷運站地址反推出來的實際位置（見 實作計畫 S04 的備援座標），
所以測試同時也在檢查那些座標能不能圍出一個合理的範圍。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

# 從文資地址反推的四個轉角（實作計畫 S04 備援座標）
SW = (121.5077, 25.0528)   # 南京西路 × 環河北路
SE = (121.5118, 25.0541)   # 南京西路 × 延平北路（黑美人大酒家在 121.511844, 25.054072）
NE = (121.5113, 25.0632)   # 民權西路 × 延平北路
NW = (121.5084, 25.0631)   # 民權西路 × 環河北路
ANCHOR = (121.51015, 25.05568)  # 霞海城隍廟


def _way(wid, name, coords, highway="secondary"):
    return {"type": "way", "id": wid, "tags": {"highway": highway, "name": name},
            "geometry": [{"lon": lo, "lat": la} for lo, la in coords]}


def _ext(p, q, f=0.25):
    """把線段 p→q 往兩端各延伸 f 倍，模擬道路延伸到研究範圍之外。"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    return (p[0] - dx * f, p[1] - dy * f), (q[0] + dx * f, q[1] + dy * f)


def _synthetic_osm():
    s0, s1 = _ext(SW, SE)
    n0, n1 = _ext(NW, NE)
    w0, w1 = _ext(SW, NW)
    e0, e1 = _ext(SE, NE)
    return [
        # 南京西路：分成兩個 way，中間有共用節點（測試 linemerge）
        _way(1, "南京西路", [s0, SW, (121.5098, 25.0534)]),
        _way(2, "南京西路", [(121.5098, 25.0534), SE, s1]),
        # 民權西路
        _way(3, "民權西路", [n0, NW, (121.5092, 25.0630), NE, n1]),
        # 環河北路：有「一段」後綴，測試開頭比對
        _way(4, "環河北路一段", [w0, SW, (121.50786, 25.05483), NW, w1]),
        # 延平北路：分一段／二段
        _way(5, "延平北路一段", [e0, SE, (121.511844, 25.054072)]),
        _way(6, "延平北路二段", [(121.511844, 25.054072), (121.511392, 25.062065), NE, e1]),
        # 範圍內的巷弄（不應該被當成界線）
        _way(7, "迪化街一段", [(121.5103, 25.0542), (121.5092, 25.0630)], "residential"),
        _way(8, None, [(121.5100, 25.0550), (121.5105, 25.0560)], "service"),
    ]


@pytest.fixture()
def osm_file(tmp_path):
    p = tmp_path / "osm_raw.json"
    p.write_text(json.dumps({"elements": _synthetic_osm()}, ensure_ascii=False), encoding="utf-8")
    return p


def test_finds_all_four_roads(tmp_root, osm_file):
    import build_boundary as bb

    roads = bb.load_named_roads(osm_file)
    assert set(roads) == {"south", "east", "north", "west"}
    # 每條都應該合併成單一 LineString（way 之間有共用節點）
    for side, line in roads.items():
        assert line.geom_type == "LineString", side
        assert line.length > 200, side


def test_builds_polygon_containing_anchor(tmp_root, osm_file):
    from shapely.geometry import Point

    import build_boundary as bb
    from dadaocheng import geo

    roads = bb.load_named_roads(osm_file)
    poly, notes = bb.build(roads)
    assert poly.is_valid
    assert len(notes) == 4
    assert all(n["判定方式"].startswith("交點") for n in notes), notes
    assert poly.contains(Point(*geo.lonlat_to_metric(*ANCHOR)))
    # 面積：約 400m × 1150m 的四邊形，0.3–0.6 平方公里之間才合理
    assert 0.30 < poly.area / 1e6 < 0.60, poly.area / 1e6


def test_includes_yanping_north_road_frontage(tmp_root, osm_file):
    """草稿邊界的東緣在延平北路以西，把沿街的文資都排除了；新邊界必須包含。"""
    from shapely.geometry import Point

    import build_boundary as bb
    from dadaocheng import geo

    poly, _ = bb.build(bb.load_named_roads(osm_file))
    for lon, lat, label in [
        (121.511844, 25.054072, "黑美人大酒家（延平北路2段1號）"),
        (121.511392, 25.062065, "延平北路2段241號"),
        (121.511134, 25.061694, "永樂國小（延平北路2段266號）"),
        (121.509210, 25.063006, "迪化街1段372號"),
        (121.5103, 25.0542, "迪化街1段13號"),
    ]:
        p = Point(*geo.lonlat_to_metric(lon, lat))
        assert poly.covers(p) or poly.exterior.distance(p) < 12, label


def test_excludes_points_outside(tmp_root, osm_file):
    from shapely.geometry import Point

    import build_boundary as bb
    from dadaocheng import geo

    poly, _ = bb.build(bb.load_named_roads(osm_file))
    for lon, lat, label in [
        (121.509456, 25.064939, "原枋隙教會（迪化街2段，民權西路以北）"),
        (121.5160, 25.0570, "延平北路以東約 450 公尺"),
        (121.5100, 25.0500, "南京西路以南約 300 公尺"),
        (121.5050, 25.0570, "環河北路以西（淡水河）"),
    ]:
        assert not poly.contains(Point(*geo.lonlat_to_metric(lon, lat))), label


def test_handles_gap_in_osm(tmp_root, tmp_path):
    """OSM 缺口時應該退回最近點，而不是整支爆掉。"""
    import build_boundary as bb

    els = _synthetic_osm()
    # 把環河北路縮短，讓它碰不到南京西路
    els[3] = _way(4, "環河北路一段", [(121.50790, 25.05400), (121.50786, 25.05483), NW, _ext(SW, NW)[1]])
    p = tmp_path / "gap.json"
    p.write_text(json.dumps({"elements": els}, ensure_ascii=False), encoding="utf-8")
    poly, notes = bb.build(bb.load_named_roads(p))
    assert poly.is_valid
    howto = [n["判定方式"] for n in notes]
    assert any("最近點" in h for h in howto), howto


def test_missing_road_raises_with_instructions(tmp_root, tmp_path):
    import build_boundary as bb

    els = [e for e in _synthetic_osm() if e["tags"].get("name") != "民權西路"]
    p = tmp_path / "nomin.json"
    p.write_text(json.dumps({"elements": els}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        bb.load_named_roads(p)
    assert "民權西路" in str(ei.value) and "--list-names" in str(ei.value)


def test_write_backs_up_draft_and_marks_final(tmp_root, osm_file):
    import build_boundary as bb
    from dadaocheng import config, geo

    before = config.path("boundary_main").read_text(encoding="utf-8")
    poly, notes = bb.build(bb.load_named_roads(osm_file))
    out = bb.write(poly, notes)
    assert out == config.path("boundary_main")

    gdf = geo.read_boundary("boundary_main")
    assert gdf.iloc[0]["status"] == "FINAL_FROM_OSM"
    assert "build_boundary.py" in gdf.iloc[0]["drawn_by"]
    corners = gdf.iloc[0]["corners"]
    if isinstance(corners, str):
        corners = json.loads(corners)
    assert corners[0]["lon"] == pytest.approx(SW[0], abs=3e-4)
    assert not geo.boundary_is_draft("boundary_main")

    backups = list(config.path("boundary_main").parent.glob("boundary_main.draft_backup_*.geojson"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == before


def test_inclusive_buffer_captures_frontage(tmp_root):
    """main_polygon_metric 的外擴要納入沿界線道路的店家，但不越界到下一個街區。"""
    from shapely.geometry import Point

    from dadaocheng import config, geo

    assert config.settings()["main_inclusive_buffer_m"] == 10
    raw = geo.main_polygon_metric(inclusive=False)
    inc = geo.main_polygon_metric()
    assert inc.area > raw.area
    assert inc.area - raw.area < 0.05 * raw.area + 40_000
    # 不論外擴與否，明顯在範圍外的點都不能被納入
    for lon, lat in [(121.5160, 25.0570), (121.5050, 25.0570), (121.509456, 25.064939)]:
        assert not inc.contains(Point(*geo.lonlat_to_metric(lon, lat)))


# ---------------------------------------------------------------------------
# 真實 OSM 的兩個特徵（11/12 使用者實跑時踩到）：
#   1. 同名前綴的巷弄（南京西路155巷、民權西路144巷5弄）
#   2. 主路本身是多段：分隔車道、缺口、遠處的同名路段（延平北路三段）
# ---------------------------------------------------------------------------

def _offset(coords, dx):
    """把一條線往東平移 dx 度，模擬分隔島另一側的車道。"""
    return [(lo + dx, la) for lo, la in coords]


def _messy_osm():
    els = _synthetic_osm()
    e0, e1 = _ext(SE, NE)
    s0, s1 = _ext(SW, SE)
    els += [
        # 巷弄：名稱以界線道路開頭，但不是界線
        _way(101, "南京西路155巷", [(121.5100, 25.0534), (121.5101, 25.0560)], "residential"),
        _way(102, "民權西路144巷5弄", [(121.5095, 25.0631), (121.5096, 25.0600)], "residential"),
        _way(103, "延平北路一段66巷", [(121.5117, 25.0560), (121.5090, 25.0561)], "residential"),
        _way(104, "環河北路二段69巷", [(121.5080, 25.0600), (121.5095, 25.0601)], "residential"),
        # 延平北路的另一側車道（約 10 公尺外），與主線不相連
        _way(105, "延平北路二段", _offset([SE, (121.511392, 25.062065), NE], 0.0001)),
        # 南京西路中間斷掉一小段（缺口約 15 公尺）
        _way(106, "南京西路", [(121.50995, 25.05345), (121.5104, 25.0536)]),
        # 遠處的同名路段（延平北路三段北端，在裁切半徑外）
        _way(107, "延平北路三段", [(121.5120, 25.0800), (121.5121, 25.0850)]),
    ]
    return els


@pytest.fixture()
def messy_osm(tmp_path):
    p = tmp_path / "messy.json"
    p.write_text(json.dumps({"elements": _messy_osm()}, ensure_ascii=False), encoding="utf-8")
    return p


def test_lane_names_are_excluded():
    import build_boundary as bb

    assert bb.is_main_road("南京西路", "南京西路")
    assert bb.is_main_road("延平北路二段", "延平北路")
    assert bb.is_main_road("環河北路 一段", "環河北路")
    assert bb.is_main_road("延平北路2段", "延平北路")
    for n in ["南京西路155巷", "民權西路144巷5弄", "延平北路一段66巷", "環河北路二段69巷", None, ""]:
        assert not bb.is_main_road(n, n[:4] if n else "南京西路"), n


def test_messy_osm_builds_same_boundary(tmp_root, osm_file, messy_osm):
    """加了巷弄、分隔車道、缺口、遠處路段之後，結果要和乾淨版幾乎一樣。"""
    import build_boundary as bb

    clean, _ = bb.build(bb.load_named_roads(osm_file))
    roads = bb.load_named_roads(messy_osm)
    # 延平北路有兩側車道 → 合併後是多段
    assert roads["east"].geom_type == "MultiLineString"
    poly, notes = bb.build(roads)
    assert poly.is_valid
    assert 0.30 < poly.area / 1e6 < 0.60
    # 與乾淨版的差異：面積差在 3% 以內，且對稱差很小
    assert abs(poly.area - clean.area) / clean.area < 0.03
    assert poly.symmetric_difference(clean).area / clean.area < 0.05


def test_messy_osm_road_segment_handles_multiline(tmp_root, messy_osm):
    from shapely.geometry import Point

    import build_boundary as bb
    from dadaocheng import geo

    roads = bb.load_named_roads(messy_osm)
    p1 = Point(*geo.lonlat_to_metric(*SE))
    p2 = Point(*geo.lonlat_to_metric(*NE))
    seg = bb.road_segment(roads["east"], p1, p2)
    assert seg.geom_type == "LineString"
    # 沿延平北路走，長度約等於兩轉角距離（約 1 公里），不會繞到遠處路段
    assert 0.95 * p1.distance(p2) < seg.length < 1.15 * p1.distance(p2)
    assert Point(seg.coords[0]).distance(p1) < 15
    assert Point(seg.coords[-1]).distance(p2) < 15


def test_verify_lists_excluded_heritage(tmp_root, osm_file, monkeypatch, capsys):
    import pandas as pd

    import build_boundary as bb
    from dadaocheng import heritage

    fake = pd.DataFrame([
        {"name": "黑美人大酒家", "address": "臺北市大同區延平北路二段1號", "district": "大同區",
         "lon": 121.511844, "lat": 25.054072},
        {"name": "對側某樓", "address": "臺北市大同區延平北路二段10號", "district": "大同區",
         "lon": 121.5124, "lat": 25.0570},
        {"name": "某店屋", "address": "臺北市大同區迪化街一段100號", "district": "大同區",
         "lon": 121.5100, "lat": 25.0570},
    ])
    monkeypatch.setattr(heritage, "load_heritage", lambda verbose=False: fake)
    poly, _ = bb.build(bb.load_named_roads(osm_file))
    bb.verify(poly)
    out = capsys.readouterr().out
    assert "已外擴 10 公尺" in out
    assert "對側某樓" in out            # 東側被排除，要列出來
    assert "黑美人大酒家" not in out.split("落在範圍外的文資")[1]   # 外擴後納入
