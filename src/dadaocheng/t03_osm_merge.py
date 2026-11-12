"""T03：OSM 補漏、與 Places 合併去重、標記 in_main、濾掉非商業類別。

輸入：
  data/interim/places_raw.parquet         （T02）
  data/raw/osm/osm_raw.json               （Overpass 回應；沙盒或學校網路擋住時，由人到 overpass-turbo 下載）
輸出：
  data/interim/poi_merged.parquet         （所有範圍內 POI，含 source、in_main、poi_id）
  data/interim/dedup_ambiguous.csv        （模稜兩可的配對：給隊友 A 看，不自動合併）
  data/interim/noncommercial_dropped.csv  （被濾掉的非商業地點，方便抽查）
  data/interim/poi_id_map.csv             （poi_id 對照，重跑時保持穩定）

用法：
  python -m dadaocheng.t03_osm_merge --write-query    # 只寫出 Overpass 查詢檔
  python -m dadaocheng.t03_osm_merge --download       # 嘗試下載（需網路）
  python -m dadaocheng.t03_osm_merge --merge          # 合併去重
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon

from . import config, geo
from .names import name_similarity, normalize_name

OSM_DIR_KEY = "raw"


def osm_dir() -> Path:
    return config.path("raw") / "osm"


# ---------------------------------------------------------------------------
# Overpass 查詢
# ---------------------------------------------------------------------------

def download_bbox_lonlat() -> tuple[float, float, float, float]:
    s = config.settings()
    poly = geo.study_polygon_metric().buffer(float(s["osm"]["download_margin_m"]))
    minx, miny, maxx, maxy = poly.bounds
    lon0, lat0 = geo.metric_to_lonlat(minx, miny)
    lon1, lat1 = geo.metric_to_lonlat(maxx, maxy)
    return lat0, lon0, lat1, lon1  # Overpass 的順序：south, west, north, east


def overpass_query() -> str:
    s, w, n, e = download_bbox_lonlat()
    bb = f"{s:.6f},{w:.6f},{n:.6f},{e:.6f}"
    return f"""[out:json][timeout:180];
(
  nwr["shop"]({bb});
  nwr["craft"]({bb});
  nwr["office"]({bb});
  nwr["amenity"]({bb});
  nwr["tourism"]({bb});
  way["highway"]({bb});
  way["building"]({bb});
  nwr["historic"]({bb});
  nwr["heritage"]({bb});
);
out geom tags;
"""


def write_query() -> Path:
    d = osm_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / "overpass_query.overpassql"
    p.write_text(overpass_query(), encoding="utf-8")
    return p


def download(timeout: float = 200.0) -> Path:
    import httpx

    s = config.settings()
    q = overpass_query()
    r = httpx.post(s["osm"]["overpass_endpoint"], data={"data": q}, timeout=timeout,
                   headers={"User-Agent": s["scrape"]["user_agent"]})
    r.raise_for_status()
    out = osm_dir() / "osm_raw.json"
    out.write_bytes(r.content)
    return out


# ---------------------------------------------------------------------------
# 解析 OSM
# ---------------------------------------------------------------------------

def load_osm(path: Path | None = None) -> list[dict]:
    path = path or (osm_dir() / "osm_raw.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在。請依 待執行任務描述.md 的 T03-H1 下載 OSM 資料。")
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("elements", [])


def _element_geometry(el: dict):
    if el["type"] == "node":
        return Point(el["lon"], el["lat"])
    geom = el.get("geometry")
    if not geom:
        return None
    coords = [(g["lon"], g["lat"]) for g in geom if g]
    if len(coords) >= 4 and coords[0] == coords[-1]:
        return Polygon(coords)
    if len(coords) >= 2:
        return LineString(coords)
    return None


def is_commercial_osm(tags: dict) -> bool:
    s = config.settings()["osm"]
    if not tags.get("name"):
        return False
    if any(k in tags for k in ("shop", "craft", "office")):
        return True
    if tags.get("amenity") in s["commercial_amenity"]:
        return True
    if tags.get("tourism") in s["commercial_tourism"]:
        return True
    return False


def osm_pois(elements: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if not is_commercial_osm(tags):
            continue
        g = _element_geometry(el)
        if g is None:
            continue
        pt = g if isinstance(g, Point) else g.representative_point()
        kind = next((f"{k}={tags[k]}" for k in ("shop", "amenity", "craft", "office", "tourism") if k in tags), "")
        rows.append({
            "osm_id": f"{el['type']}/{el['id']}", "name": tags.get("name"), "osm_kind": kind,
            "osm_website": tags.get("website") or tags.get("contact:website"),
            "osm_opening_hours": tags.get("opening_hours"), "osm_brand": tags.get("brand"),
            "lon": pt.x, "lat": pt.y,
        })
    df = pd.DataFrame(rows, columns=["osm_id", "name", "osm_kind", "osm_website", "osm_opening_hours", "osm_brand", "lon", "lat"])
    return geo.points_gdf(df)


def _gdf(rows: list[dict], cols: list[str]) -> gpd.GeoDataFrame:
    """rows 可能是空的（例如該範圍 OSM 沒有建物）：仍回傳有 geometry 欄的空表。"""
    if not rows:
        return gpd.GeoDataFrame({c: [] for c in cols}, geometry=gpd.GeoSeries([], crs=geo.WGS84), crs=geo.WGS84)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=geo.WGS84)


def osm_roads(elements: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if el["type"] != "way" or "highway" not in tags:
            continue
        g = _element_geometry(el)
        if g is None:
            continue
        if isinstance(g, Polygon):
            g = LineString(g.exterior.coords)
        rows.append({"osm_id": f"way/{el['id']}", "highway": tags["highway"], "name": tags.get("name"), "geometry": g})
    return _gdf(rows, ["osm_id", "highway", "name"])


def osm_buildings(elements: list[dict]) -> gpd.GeoDataFrame:
    rows = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        if "building" not in tags:
            continue
        g = _element_geometry(el)
        if isinstance(g, Polygon) and g.is_valid:
            rows.append({"osm_id": f"{el['type']}/{el['id']}", "name": tags.get("name"),
                         "historic": tags.get("historic"), "heritage": tags.get("heritage"), "geometry": g})
    return _gdf(rows, ["osm_id", "name", "historic", "heritage"])


# ---------------------------------------------------------------------------
# 非商業過濾
# ---------------------------------------------------------------------------

GENERIC_TYPES = {"point_of_interest", "establishment"}


def classify_places_type(primary: str | None, types_json: str | None) -> tuple[bool, bool]:
    """回傳 (is_noncommercial, generic_type)。"""
    nonc = set(config.settings()["noncommercial_types"])
    types = set(json.loads(types_json)) if types_json else set()
    if primary and primary in nonc:
        return True, False
    meaningful = types - GENERIC_TYPES
    if not meaningful:
        return False, True
    if meaningful <= nonc:
        return True, False
    return False, False


# ---------------------------------------------------------------------------
# 合併去重
# ---------------------------------------------------------------------------

def match_osm_to_places(places: gpd.GeoDataFrame, osm: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (matches, ambiguous)。matches：osm_id → place_id 的自動合併；ambiguous：給人看的清單。"""
    s = config.settings()
    dmax = float(s["dedup_distance_m"])
    hi, lo = float(s["dedup"]["name_sim_high"]), float(s["dedup"]["name_sim_low"])
    sfx = s["dedup"]["strip_suffixes"]
    P = places.to_crs(geo.metric_crs())
    O = osm.to_crs(geo.metric_crs())
    matches, amb = [], []
    if len(P) == 0 or len(O) == 0:
        return pd.DataFrame(columns=["osm_id", "place_id", "dist_m", "sim"]), pd.DataFrame()
    sidx = P.sindex
    for _, o in O.iterrows():
        cand = sidx.query(o.geometry.buffer(dmax), predicate="intersects")
        best = None
        for j in cand:
            p = P.iloc[j]
            d = o.geometry.distance(p.geometry)
            if d > dmax:
                continue
            sim = name_similarity(o["name"], p["name"], sfx)
            if best is None or sim > best[1]:
                best = (p["place_id"], sim, d, p["name"])
        if best is None:
            continue
        pid, sim, d, pname = best
        if sim >= hi:
            matches.append({"osm_id": o["osm_id"], "place_id": pid, "dist_m": round(d, 1), "sim": sim})
        elif sim >= lo:
            amb.append({"kind": "osm_vs_places", "a_id": o["osm_id"], "a_name": o["name"], "b_id": pid, "b_name": pname,
                        "dist_m": round(d, 1), "sim": round(sim, 1), "decision": ""})
    # Places 之間：不同 place_id 但很近、名稱很像 → 只列出，不合併
    for i, p in P.iterrows():
        cand = sidx.query(p.geometry.buffer(dmax), predicate="intersects")
        for j in cand:
            q = P.iloc[j]
            if q["place_id"] <= p["place_id"]:
                continue
            d = p.geometry.distance(q.geometry)
            if d > dmax:
                continue
            sim = name_similarity(p["name"], q["name"], sfx)
            if sim >= hi:
                amb.append({"kind": "places_vs_places", "a_id": p["place_id"], "a_name": p["name"], "b_id": q["place_id"],
                            "b_name": q["name"], "dist_m": round(d, 1), "sim": round(sim, 1), "decision": ""})
    return pd.DataFrame(matches), pd.DataFrame(amb)


def assign_poi_ids(keys: pd.Series, map_path: Path) -> pd.Series:
    """穩定的 poi_id：已存在的鍵沿用舊編號，新鍵依字典序接在後面。"""
    if map_path.exists():
        m = pd.read_csv(map_path, dtype=str)
        mapping = dict(zip(m["key"], m["poi_id"]))
    else:
        mapping = {}
    next_n = max([int(v[1:]) for v in mapping.values()] or [0]) + 1
    for k in sorted(set(keys) - set(mapping)):
        mapping[k] = f"P{next_n:05d}"
        next_n += 1
    pd.DataFrame({"key": list(mapping), "poi_id": list(mapping.values())}).to_csv(map_path, index=False, encoding="utf-8-sig")
    return keys.map(mapping)


def load_decisions(path: Path | None = None) -> pd.DataFrame:
    """人工判斷檔：把 dedup_ambiguous.csv 另存成 dedup_decisions.csv 並填 decision（same／different／unsure）。"""
    path = path or (config.path("interim") / "dedup_decisions.csv")
    if not path.exists():
        return pd.DataFrame(columns=["kind", "a_id", "b_id", "decision"])
    d = pd.read_csv(path, dtype=str, encoding="utf-8-sig").fillna("")
    d["decision"] = d["decision"].str.strip().str.lower()
    return d


def merge(places_df: pd.DataFrame, osm_elements: list[dict], decisions: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """decisions：人工判斷（可省略）。
    - osm_vs_places：same → 視為同一家（source=both）；different → OSM 點以 osm 來源加入；其他 → 先不加入。
    - places_vs_places：same → 刪掉 b_id（保留 a_id）；其他 → 兩筆都保留。
    """
    crs = geo.metric_crs()
    study = geo.study_polygon_metric()
    main = geo.main_polygon_metric()

    places = geo.points_gdf(places_df)
    flags = places_df.apply(lambda r: classify_places_type(r.get("places_primary_type"), r.get("places_types")), axis=1)
    places["is_noncommercial"] = [f[0] for f in flags]
    places["generic_type"] = [f[1] for f in flags]
    dropped = places[places["is_noncommercial"]].drop(columns="geometry")
    places = places[~places["is_noncommercial"]].copy()

    osm = osm_pois(osm_elements)
    n_before = len(places) + len(osm)
    matches, amb = match_osm_to_places(places, osm)

    dec = decisions if decisions is not None else pd.DataFrame(columns=["kind", "a_id", "b_id", "decision"])
    ov = dec[dec["kind"] == "osm_vs_places"]
    same_ov = ov[ov["decision"] == "same"]
    if len(same_ov):
        extra = pd.DataFrame({"osm_id": same_ov["a_id"], "place_id": same_ov["b_id"], "dist_m": None, "sim": None})
        matches = pd.concat([matches, extra], ignore_index=True) if len(matches) else extra
    diff_osm = set(ov.loc[ov["decision"] == "different", "a_id"])
    drop_places = set(dec.loc[(dec["kind"] == "places_vs_places") & (dec["decision"] == "same"), "b_id"])
    places = places[~places["place_id"].isin(drop_places)].copy()

    places["source"] = "places"
    places["osm_id"] = None
    # concat 會把「只有一邊有的欄位」填成 NaN（float），而 bool(NaN) 是 True，
    # 下游用 `if osm_kind:` 判斷就會誤判。這裡先明確建成 object 欄並填 None。
    for c in ("osm_kind", "osm_website", "osm_opening_hours", "osm_brand"):
        if c not in places.columns:
            places[c] = pd.Series([None] * len(places), index=places.index, dtype=object)
    if len(matches):
        mm = matches.drop_duplicates("place_id").set_index("place_id")["osm_id"]
        has = places["place_id"].isin(mm.index)
        places.loc[has, "osm_id"] = places.loc[has, "place_id"].map(mm)
        places.loc[has, "source"] = "both"
    matched_osm = set(matches["osm_id"]) if len(matches) else set()
    amb_osm = set(amb.loc[amb["kind"] == "osm_vs_places", "a_id"]) if len(amb) else set()
    amb_osm -= diff_osm  # 人工判斷為不同店 → 加入
    # 模稜兩可的 OSM 點先不加入（等人判斷），避免重複計數
    osm_only = osm[~osm["osm_id"].isin(matched_osm | amb_osm)].copy()
    osm_only["source"] = "osm"
    osm_only["place_id"] = None
    osm_only["business_status"] = "UNKNOWN_OSM"
    osm_only["website_uri"] = osm_only["osm_website"]
    osm_only["generic_type"] = False
    for c in ("places_primary_type", "places_types", "formatted_address", "regular_opening_hours",
              "user_rating_count", "rating", "price_level"):
        if c not in osm_only.columns:
            osm_only[c] = pd.Series([None] * len(osm_only), index=osm_only.index, dtype=object)

    allp = pd.concat([places, osm_only], ignore_index=True, sort=False)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=geo.WGS84)
    m = allp.to_crs(crs)
    allp["in_study"] = m.within(study).values
    allp["in_main"] = m.within(main).values
    allp = allp[allp["in_study"]].copy()
    allp["x"] = m.loc[allp.index].geometry.x.values
    allp["y"] = m.loc[allp.index].geometry.y.values
    key = allp["place_id"].where(allp["place_id"].notna(), "osm:" + allp["osm_id"].astype(str))
    allp["poi_id"] = assign_poi_ids(key.astype(str), config.path("interim") / "poi_id_map.csv").values
    cols = ["poi_id", "place_id", "osm_id", "source", "name", "lon", "lat", "x", "y", "in_main",
            "places_primary_type", "places_types", "osm_kind", "generic_type", "business_status",
            "user_rating_count", "rating", "price_level", "website_uri", "formatted_address", "regular_opening_hours",
            "osm_opening_hours"]
    for c in cols:
        if c not in allp.columns:
            allp[c] = None
    out = pd.DataFrame(allp[cols]).sort_values("poi_id").reset_index(drop=True)
    # 寫進 parquet 之前把文字欄位的缺值統一成 None（object），避免讀回來變 NaN float
    for c in ("osm_kind", "places_primary_type", "places_types", "formatted_address",
              "regular_opening_hours", "osm_opening_hours", "website_uri", "price_level", "osm_id", "place_id"):
        if c in out.columns:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    stats = pd.DataFrame([{
        "places_in": len(places_df), "noncommercial_dropped": len(dropped), "osm_commercial": len(osm),
        "before_dedup": n_before, "auto_merged": len(matches), "ambiguous_pairs": len(amb),
        "human_decisions_applied": int((dec["decision"].isin(["same", "different"])).sum()),
        "after_dedup_in_study": len(out), "in_main": int(out["in_main"].sum()),
        "in_main_operational": int(((out["in_main"]) & (out["business_status"] == "OPERATIONAL")).sum()),
    }])
    return {"poi": out, "ambiguous": amb, "dropped": dropped, "stats": stats}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write-query", action="store_true")
    g.add_argument("--download", action="store_true")
    g.add_argument("--merge", action="store_true")
    args = ap.parse_args(argv)
    if args.write_query:
        print("已寫出：", write_query())
        return
    if args.download:
        write_query()
        print("已下載：", download())
        return
    places_df = pd.read_parquet(config.path("interim") / "places_raw.parquet")
    res = merge(places_df, load_osm(), load_decisions())
    d = config.path("interim")
    res["poi"].to_parquet(d / "poi_merged.parquet", index=False)
    res["ambiguous"].to_csv(d / "dedup_ambiguous.csv", index=False, encoding="utf-8-sig")
    res["dropped"].to_csv(d / "noncommercial_dropped.csv", index=False, encoding="utf-8-sig")
    print(res["stats"].T.to_string(header=False))
    print(f"模稜兩可清單（給隊友 A）：{d / 'dedup_ambiguous.csv'}")


if __name__ == "__main__":
    main()
