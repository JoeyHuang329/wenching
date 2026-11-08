"""交通資料：用使用者提供的真實檔案測試解析與特徵。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "transit"

pytestmark = pytest.mark.skipif(not (RAW / "mrt_stations.csv").exists(), reason="交通原始資料不在 data/raw/transit/")


def test_parse_mrt_stations():
    from dadaocheng.transit import parse_mrt_stations

    st = parse_mrt_stations(RAW / "mrt_stations.csv")
    assert len(st) >= 100
    dqt = st[st.name_zh == "大橋頭"].iloc[0]
    assert dqt.station_ids == "O12"
    assert dqt.geometry.x == pytest.approx(121.5130035) and dqt.geometry.y == pytest.approx(25.06322)
    # 轉乘站合併成一列
    zs = st[st.name_zh == "中山"].iloc[0]
    assert set(zs.station_ids.split(";")) == {"R11", "G14"}
    # 南港展覽館：BR24 與 BL23 同座標 → 合併
    ng = st[st.name_zh == "南港展覽館"]
    assert len(ng) == 1 and set(ng.iloc[0].station_ids.split(";")) == {"BR24", "BL23"}


def test_parse_bus_stops_utf8_truncation():
    from dadaocheng.transit import parse_bus_stops

    bs = parse_bus_stops(RAW / "busstop.shp")
    assert len(bs) == 3407
    assert bs["stop_id"].is_unique
    assert "迪化街" in set(bs["name"])
    assert bs["name_truncated"].any()      # dbf 欄寬 22 bytes 造成的截斷


def test_parse_bus_shapes():
    from dadaocheng.transit import parse_bus_shapes

    g = parse_bus_shapes(RAW / "bus_shapes_GetBusShape.json")
    assert len(g) > 1000
    assert set(g["go_back"].dropna().unique()) <= {0, 1}


def test_transit_features_near_daqiaotou(tmp_root):
    from dadaocheng import geo
    from dadaocheng.transit import parse_bus_shapes, parse_bus_stops, parse_mrt_stations, transit_features

    pts = geo.points_gdf(pd.DataFrame({"lon": [121.5128, 121.5100], "lat": [25.0628, 25.0560]}))
    f = transit_features(pts, parse_mrt_stations(RAW / "mrt_stations.csv"), parse_bus_stops(RAW / "busstop.shp"),
                         parse_bus_shapes(RAW / "bus_shapes_GetBusShape.json"))
    assert f.loc[0, "nearest_mrt"] == "大橋頭" and f.loc[0, "dist_mrt_m"] < 100
    assert f.loc[1, "dist_mrt_m"] > f.loc[0, "dist_mrt_m"]
    assert (f["bus_density"] > 0).all()
    assert (f["bus_route_count_150"] >= 0).all()
