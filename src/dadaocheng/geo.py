"""空間共用工具。原始座標存 WGS84；距離與面積一律在 EPSG:3826 計算。"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from . import config

WGS84 = "EPSG:4326"


def metric_crs() -> str:
    return config.settings()["crs_metric"]


def read_boundary(key: str = "boundary_main") -> gpd.GeoDataFrame:
    """讀邊界（WGS84），回傳 GeoDataFrame。"""
    p = config.path(key)
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}。請先完成邊界（待執行任務描述.md 的 T00-H2）。")
    gdf = gpd.read_file(p)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84)
    return gdf.to_crs(WGS84)


def boundary_is_draft(key: str = "boundary_main") -> bool:
    with open(config.path(key), encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    return any(str(f.get("properties", {}).get("status", "")).upper().startswith("DRAFT") for f in feats)


def main_polygon_metric(inclusive: bool | None = None):
    """主範圍（EPSG:3826）。

    主範圍是用四條界線道路的「中心線」定義的，但 POI 與文資的座標是建物形心，
    面向界線道路的店家因此會落在中心線外側數公尺。inclusive 會把多邊形外擴
    main_inclusive_buffer_m（預設 10 公尺）把這些沿街店家納入；界線道路對側的
    建物形心距中心線通常超過 12 公尺，不會被誤納。
    設 inclusive=False 可取得未外擴的版本（敏感性分析用）。
    """
    poly = unary_union(read_boundary("boundary_main").to_crs(metric_crs()).geometry)
    if inclusive is False:
        return poly
    b = float(config.settings().get("main_inclusive_buffer_m", 0) or 0)
    return poly.buffer(b, join_style="mitre") if b > 0 else poly


def study_polygon_metric():
    """主範圍＋緩衝區（EPSG:3826）。"""
    return unary_union(read_boundary("boundary_buffer").to_crs(metric_crs()).geometry)


def make_buffer_geojson(out_path: Path | None = None) -> Path:
    """由主範圍外擴 buffer_m 公尺產生緩衝區邊界（含主範圍本身）。"""
    s = config.settings()
    main = read_boundary("boundary_main").to_crs(metric_crs())
    buf = unary_union(main.geometry).buffer(float(s["buffer_m"]), join_style="mitre")
    out = gpd.GeoDataFrame(
        {"name": ["主範圍＋緩衝區"], "buffer_m": [s["buffer_m"]], "status": ["DERIVED from boundary_main"]},
        geometry=[buf],
        crs=metric_crs(),
    ).to_crs(WGS84)
    out_path = out_path or config.path("boundary_buffer")
    out.to_file(out_path, driver="GeoJSON")
    return out_path


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(metric_crs())


def points_gdf(df, lon="lon", lat="lat") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon], df[lat]), crs=WGS84)


def add_metric_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    m = gdf.to_crs(metric_crs())
    gdf = gdf.copy()
    gdf["x"] = m.geometry.x.values
    gdf["y"] = m.geometry.y.values
    return gdf


def lonlat_to_metric(lon: float, lat: float) -> tuple[float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs(WGS84, metric_crs(), always_xy=True)
    return t.transform(lon, lat)


def metric_to_lonlat(x: float, y: float) -> tuple[float, float]:
    from pyproj import Transformer

    t = Transformer.from_crs(metric_crs(), WGS84, always_xy=True)
    return t.transform(x, y)
