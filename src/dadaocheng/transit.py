"""交通資料：捷運站、捷運路網、公車站牌、公車路線軌跡。

資料來源（皆為開放資料，原始檔放 data/raw/transit/）：
- mrt_stations.csv：臺北捷運車站資料（data.taipei，2026-09-04 更新）。座標 WGS84，欄位帶有多餘的單引號與大括號。
- mrt_routes_twd97.json：臺北都會區捷運路網圖（2026-10-13 更新），GeoJSON，EPSG:3826。只用來畫圖。
- busstop.shp/.shx/.dbf：公車站牌點位，WGS84（沒有 .prj），dbf 為 UTF-8，站名欄被截在 22 bytes，少數字會變成「�」。
- bus_shapes_GetBusShape.json：臺北市公車路線軌跡（utime 2025-12），WKT 線段，WGS84。TstBusShape.json 是較舊的版本。

產生的特徵（給 T04 用）：
- dist_mrt_m：到最近捷運站中心點的直線距離（EPSG:3826）。資料沒有出入口座標。
- nearest_mrt：最近捷運站名稱。
- bus_density：半徑 bus_density_radius_m 內的站牌數 ÷ 面積（站／平方公里）。
- bus_stop_count_300：同上的站牌數（未除面積，方便解釋）。
- bus_route_count_150：半徑內通過的不同公車路線數（補充特徵，未列入預先登錄）。
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import box

from . import config, geo

# ---------------------------------------------------------------------------
# 讀檔
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    return s.strip().strip("'").strip('"').strip()


def parse_mrt_stations(path: Path | str) -> gpd.GeoDataFrame:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            ids = [i.strip() for i in _strip(r["StationID"]).split(",") if i.strip()]
            name = _strip(r["StationName"]).strip("{}")
            zh, _, en = name.partition(",")
            pos = _strip(r["StationPosition"]).strip("{}")
            lon, lat = (float(v) for v in pos.split(","))
            rows.append(
                {
                    "station_ids": ";".join(ids),
                    "lines": ";".join(sorted({re.match(r"[A-Z]+", i).group(0) for i in ids})),
                    "name_zh": zh.strip(),
                    "name_en": en.strip(),
                    "lon": lon,
                    "lat": lat,
                    "address": _strip(r["StationAddress"]).replace("\n", " / "),
                    "update_time": _strip(r["UpdateTime"]),
                }
            )
    df = pd.DataFrame(rows)
    # 同站不同線分列（例如南港展覽館 BR24 與 BL23）：以名稱＋座標合併
    df = (
        df.groupby(["name_zh", "lon", "lat"], as_index=False)
        .agg({"station_ids": lambda s: ";".join(sorted(set(";".join(s).split(";")))),
              "lines": lambda s: ";".join(sorted(set(";".join(s).split(";")))),
              "name_en": "first", "address": " | ".join, "update_time": "max"})
    )
    return geo.points_gdf(df)


def parse_mrt_routes(path: Path | str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:3826")
    return gdf[["RouteName", "geometry"]].to_crs(geo.WGS84)


def parse_bus_stops(shp_path: Path | str) -> gpd.GeoDataFrame:
    import shapefile  # pyshp

    r = shapefile.Reader(str(shp_path), encoding="utf-8", encodingErrors="replace")
    recs, shapes = r.records(), r.shapes()
    rows = []
    for rec, shp in zip(recs, shapes):
        if not shp.points:
            continue
        lon, lat = shp.points[0]
        name = str(rec[1]).strip()
        rows.append({"stop_id": int(rec[0]), "name": name, "name_truncated": "�" in name, "lon": lon, "lat": lat})
    df = pd.DataFrame(rows).drop_duplicates("stop_id")
    return geo.points_gdf(df)


def parse_bus_shapes(json_path: Path | str) -> gpd.GeoDataFrame:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for d in data:
        try:
            g = shapely_wkt.loads(d["wkt"])
        except Exception:  # noqa: BLE001
            continue
        rows.append({"uni_route_id": str(d.get("UniRouteId")), "route_id": d.get("RouteID"),
                     "sub_route_id": d.get("SubRouteID"), "go_back": d.get("GoBack"),
                     "utime": d.get("utime"), "geometry": g})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=geo.WGS84)


# ---------------------------------------------------------------------------
# 裁切與輸出
# ---------------------------------------------------------------------------

def clip_window_metric():
    s = config.settings()
    poly = geo.study_polygon_metric()
    return poly.buffer(float(s["transit"]["clip_margin_m"]))


def build_transit_layers(out_dir: Path | None = None) -> dict[str, int]:
    """讀原始交通資料、裁切到研究範圍外擴 clip_margin_m，存成 GeoJSON（WGS84）。"""
    s = config.settings()
    t = s["transit"]
    root = config.project_root()
    out_dir = out_dir or config.path("interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    window = clip_window_metric()
    crs = geo.metric_crs()

    layers = {
        "transit_mrt_stations": parse_mrt_stations(root / t["mrt_stations_csv"]),
        "transit_mrt_routes": parse_mrt_routes(root / t["mrt_routes_geojson"]),
        "transit_bus_stops": parse_bus_stops(root / t["bus_stops_shp"]),
        "transit_bus_routes": parse_bus_shapes(root / t["bus_shapes_json"]),
    }
    counts = {}
    for name, gdf in layers.items():
        m = gdf.to_crs(crs)
        if name in ("transit_mrt_routes", "transit_bus_routes"):
            m = m[m.intersects(window)].copy()
            m["geometry"] = m.geometry.intersection(window)
        else:
            m = m[m.within(window)].copy()
        out = m.to_crs(geo.WGS84)
        out.to_file(out_dir / f"{name}.geojson", driver="GeoJSON")
        counts[name] = len(out)
    return counts


def load_layer(name: str) -> gpd.GeoDataFrame:
    p = config.path("interim") / f"{name}.geojson"
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在，請先跑 python -m dadaocheng.transit")
    return gpd.read_file(p)


# ---------------------------------------------------------------------------
# 特徵
# ---------------------------------------------------------------------------

def transit_features(pois: gpd.GeoDataFrame,
                     stations: gpd.GeoDataFrame | None = None,
                     stops: gpd.GeoDataFrame | None = None,
                     routes: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    """回傳與 pois 同索引的交通特徵表。pois 需為點（任何 CRS）。"""
    s = config.settings()["transit"]
    crs = geo.metric_crs()
    P = pois.to_crs(crs)
    stations = (stations if stations is not None else load_layer("transit_mrt_stations")).to_crs(crs)
    stops = (stops if stops is not None else load_layer("transit_bus_stops")).to_crs(crs)

    out = pd.DataFrame(index=pois.index)
    # 最近捷運站
    sx, sy = stations.geometry.x.values, stations.geometry.y.values
    px, py = P.geometry.x.values, P.geometry.y.values
    d = np.sqrt((px[:, None] - sx[None, :]) ** 2 + (py[:, None] - sy[None, :]) ** 2)
    j = d.argmin(axis=1)
    out["dist_mrt_m"] = d[np.arange(len(P)), j]
    out["nearest_mrt"] = stations["name_zh"].values[j]

    # 公車站牌密度
    r = float(s["bus_density_radius_m"])
    bx, by = stops.geometry.x.values, stops.geometry.y.values
    db = np.sqrt((px[:, None] - bx[None, :]) ** 2 + (py[:, None] - by[None, :]) ** 2)
    cnt = (db <= r).sum(axis=1)
    out[f"bus_stop_count_{int(r)}"] = cnt
    out["bus_density"] = cnt / (math.pi * r * r / 1e6)

    # 補充：通過的公車路線數
    if routes is not None or (config.path("interim") / "transit_bus_routes.geojson").exists():
        routes = (routes if routes is not None else load_layer("transit_bus_routes")).to_crs(crs)
        rr = float(s["bus_route_radius_m"])
        sidx = routes.sindex
        counts = []
        for pt in P.geometry:
            idx = sidx.query(pt.buffer(rr), predicate="intersects")
            counts.append(routes.iloc[idx]["uni_route_id"].nunique() if len(idx) else 0)
        out[f"bus_route_count_{int(rr)}"] = counts
    return out


def check_map(out_png: Path | None = None) -> Path:
    """交通資料檢查圖：主範圍、緩衝區、捷運站、站牌、路線。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import setup_cjk_font

    setup_cjk_font()
    crs = geo.metric_crs()
    main = geo.read_boundary("boundary_main").to_crs(crs)
    buf = geo.read_boundary("boundary_buffer").to_crs(crs)
    st = load_layer("transit_mrt_stations").to_crs(crs)
    bs = load_layer("transit_bus_stops").to_crs(crs)
    mr = load_layer("transit_mrt_routes").to_crs(crs)
    br = load_layer("transit_bus_routes").to_crs(crs)

    fig, ax = plt.subplots(figsize=(8, 9))
    br.plot(ax=ax, color="#bbbbbb", linewidth=0.4, alpha=0.5)
    mr.plot(ax=ax, color="#2b6cb0", linewidth=2)
    buf.boundary.plot(ax=ax, color="#888888", linestyle="--", linewidth=1)
    main.boundary.plot(ax=ax, color="#c53030", linewidth=2)
    bs.plot(ax=ax, color="#2f855a", markersize=6)
    st.plot(ax=ax, color="#2b6cb0", markersize=50, marker="s")
    for _, row in st.iterrows():
        ax.annotate(row["name_zh"], (row.geometry.x, row.geometry.y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    minx, miny, maxx, maxy = buf.total_bounds
    pad = 600
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal")
    ax.set_title("交通資料檢查圖（紅：主範圍；虛線：緩衝區；綠點：公車站牌；藍方塊：捷運站）", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    out_png = out_png or (config.path("figures") / "check_transit.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> None:
    if not config.path("boundary_buffer").exists():
        geo.make_buffer_geojson()
    if geo.boundary_is_draft():
        print("⚠ 注意：主範圍邊界仍是 DRAFT（Claude 粗估）。正式分析前請由彭郁媗重畫。")
    counts = build_transit_layers()
    for k, v in counts.items():
        print(f"{k}: {v} 筆")
    st = load_layer("transit_mrt_stations")
    main_poly = geo.main_polygon_metric()
    st_m = st.to_crs(geo.metric_crs())
    st_m["dist_to_main_m"] = st_m.geometry.distance(main_poly)
    print("\n離主範圍最近的捷運站：")
    print(st_m.sort_values("dist_to_main_m")[["name_zh", "station_ids", "dist_to_main_m"]].head(6).to_string(index=False))
    bs = load_layer("transit_bus_stops")
    print(f"\n站名被截斷（含 �）的站牌：{int(bs['name_truncated'].sum())} / {len(bs)}")
    print("檢查圖：", check_map())


if __name__ == "__main__":
    main()
