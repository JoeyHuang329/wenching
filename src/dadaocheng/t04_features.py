"""T04：文資比對、街區切割、結構化特徵，產生 poi_master。

輸入：
  data/interim/poi_merged.parquet                 （T03）
  data/raw/osm/osm_raw.json                       （T03 下載的 OSM：道路、建物）
  data/raw/heritage/heritage_list.csv             （人工下載整理：文資局／文化局的古蹟與歷史建築）
  data/raw/heritage/temples.csv                   （百年廟宇，人工確認；可選）
  data/interim/transit_*.geojson                  （python -m dadaocheng.transit）
輸出：
  data/final/poi_master.parquet
  data/interim/heritage_match.csv                 （給隊友 A 人工檢查）
  data/interim/blocks.geojson
  outputs/tables/t04_missing_report.csv
  outputs/figures/check_heritage.png、check_blocks.png、check_main_street.png

heritage_list.csv 欄位（見 待執行任務描述.md T04-H1）：
  heritage_id, name, type(古蹟/歷史建築/紀念建築…), address, lon, lat   （lon/lat 可空）
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import nearest_points, polygonize, unary_union

from . import config, geo
from .names import as_text, normalize_address, normalize_name

# ---------------------------------------------------------------------------
# 營業時間
# ---------------------------------------------------------------------------

def weekday_close_minutes(hours_json: str | dict | None) -> tuple[float | None, str]:
    """回傳 (週一到週五關門時間中位數，以「當天 0 點起算的分鐘」表示, 狀態)。

    - 跨午夜：close.day 與 open.day 不同時，加上 1440×天數差（例如 02:00 → 1560）。
    - 同一天多個時段（午休）：取當天最後一個關門時間。
    - 24 小時營業（open 0:00 且沒有 close）：視為 1440（不在 19:00 前關門）。
    - 平日全部沒開：回傳 (None, "weekday_closed")。
    - 沒有營業時間：回傳 (None, "missing") —— 是缺失，不是「不營業」。
    """
    if hours_json is None or (isinstance(hours_json, float) and math.isnan(hours_json)):
        return None, "missing"
    h = json.loads(hours_json) if isinstance(hours_json, str) else hours_json
    periods = (h or {}).get("periods") or []
    if not periods:
        return None, "missing"
    weekdays = config.settings()["hours"]["weekdays"]
    # 24 小時
    if len(periods) == 1 and "close" not in periods[0]:
        o = periods[0].get("open", {})
        if o.get("hour", 0) == 0 and o.get("minute", 0) == 0:
            return 1440.0, "open_24h"
    per_day: dict[int, float] = {}
    for p in periods:
        o, c = p.get("open"), p.get("close")
        if not o or not c:
            continue
        od, cd = int(o.get("day", 0)), int(c.get("day", 0))
        cm = int(c.get("hour", 0)) * 60 + int(c.get("minute", 0))
        om = int(o.get("hour", 0)) * 60 + int(o.get("minute", 0))
        delta_days = (cd - od) % 7
        if delta_days == 0 and cm <= om:   # 沒標隔天但關門早於開門 → 視為跨午夜
            delta_days = 1
        close_abs = cm + 1440 * delta_days
        if od in weekdays:
            per_day[od] = max(per_day.get(od, -1), close_abs)
    if not per_day:
        return None, "weekday_closed"
    return float(np.median(list(per_day.values()))), "ok"


def minutes_to_hhmm(m: float | None) -> str | None:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return None
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def add_hours(df: pd.DataFrame) -> pd.DataFrame:
    th = config.settings()["hours"]["close_threshold"]
    th_min = int(th[:2]) * 60 + int(th[3:])
    res = df["regular_opening_hours"].apply(weekday_close_minutes)
    df = df.copy()
    df["close_min_weekday"] = [r[0] for r in res]
    df["hours_status"] = [r[1] for r in res]
    df["close_time_weekday"] = df["close_min_weekday"].apply(minutes_to_hhmm)
    df["closes_before_19"] = df["close_min_weekday"].apply(lambda m: None if m is None or pd.isna(m) else bool(m < th_min))
    return df


# ---------------------------------------------------------------------------
# Places 類別 → 業態大類（粗分類）
# ---------------------------------------------------------------------------

def parse_types(types_json) -> list[str]:
    """Places 的 types 欄位。來源可能是 JSON 字串、list，或缺值。"""
    v = types_json
    if isinstance(v, (list, tuple, np.ndarray)):
        return [as_text(t) for t in v if as_text(t)]
    s = as_text(v)
    if not s:
        return []
    if s.startswith("["):
        try:
            return [as_text(t) for t in json.loads(s) if as_text(t)]
        except json.JSONDecodeError:
            return []
    return [t for t in (x.strip() for x in s.split(",")) if t]


def places_category(primary: str | None, types_json: str | None, osm_kind: str | None = None) -> str:
    """由 Places 類別（或 OSM 標籤）推 POI 的業態大類。

    三個參數都可能是缺值：Places 來源的 POI 沒有 osm_kind，OSM 來源的沒有 primaryType。
    """
    cmap: dict[str, list[str]] = config.settings()["places_category_map"]
    lookup = {t: cat for cat, ts in cmap.items() for t in ts}
    primary = as_text(primary)
    if primary and primary in lookup:
        return lookup[primary]
    for t in parse_types(types_json):
        if t in lookup:
            return lookup[t]
    kind = as_text(osm_kind)
    if kind:
        k, _, v = kind.partition("=")
        if v in lookup:
            return lookup[v]
        if k == "shop":
            return "一般零售與品牌"
        if k == "craft":
            return "選物與文創零售"
    return "其他"


def traditional_subtype(name: str | None) -> str | None:
    n = normalize_name(name)
    for sub, kws in config.settings()["traditional_subtypes"].items():
        if any(normalize_name(k) in n for k in kws):
            return sub
    return None


def likely_wenqing(primary: str | None, types_json: str | None) -> bool:
    lw = set(config.settings()["likely_wenqing_types"])
    return bool((as_text(primary) in lw) or (set(parse_types(types_json)) & lw))


# ---------------------------------------------------------------------------
# 文資比對
# ---------------------------------------------------------------------------

def load_heritage(path: Path | None = None) -> pd.DataFrame:
    """保留舊介面：實際載入與驗證在 dadaocheng.heritage。"""
    from .heritage import load_heritage as _load

    return _load()


def match_heritage(pois: gpd.GeoDataFrame, heritage: pd.DataFrame,
                   buildings: gpd.GeoDataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """比對 POI 與文資，回傳 (heritage_located：每筆文資的定位結果, poi_matches：POI × 文資配對)。

    判定（提案書 6.5 / 實作規劃 B T04）：
      1. POI 落在該文資所在的建物輪廓內 → 算。
      2. POI 與文資點位距離 ≤ match_distance_m → 算。
      3. POI 的正規化地址與文資地址相同 → 算（沒有座標的文資只能靠這個）。

    以空間索引向量化處理：文資 227 筆 × POI 4,000 筆用雙層迴圈要數分鐘，這裡是數秒。
    """
    crs = geo.metric_crs()
    dmax = float(config.settings()["heritage"]["match_distance_m"])
    P = pois.to_crs(crs).reset_index(drop=True)
    poi_addr = (pois["formatted_address"].apply(normalize_address).reset_index(drop=True)
                if "formatted_address" in pois else pd.Series([""] * len(P)))
    B = buildings.to_crs(crs).reset_index(drop=True) if buildings is not None and len(buildings) else None

    H = heritage.reset_index(drop=True).copy()
    has_xy = H["lon"].notna() & H["lat"].notna()
    HX = gpd.GeoDataFrame(H[has_xy], geometry=gpd.points_from_xy(H.loc[has_xy, "lon"], H.loc[has_xy, "lat"]),
                          crs=geo.WGS84).to_crs(crs).reset_index(drop=True)

    # 每筆文資落在哪個建物輪廓內
    fp_idx = pd.Series([pd.NA] * len(HX), dtype="object")
    if B is not None and len(HX):
        j = gpd.sjoin(HX[["geometry"]], B[["geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()]
        fp_idx = j["index_right"]

    located = []
    for i, r in H.iterrows():
        if has_xy.iloc[i]:
            k = int(HX.index[HX["heritage_id"] == r["heritage_id"]][0]) if len(HX) else -1
            fp = fp_idx.iloc[k] if k >= 0 else pd.NA
            method = "coord+footprint" if pd.notna(fp) else "coord"
            pt = HX.geometry.iloc[k]
            located.append({"heritage_id": r["heritage_id"], "name": r["name"], "type": r.get("type"),
                            "method": method, "x": pt.x, "y": pt.y})
        else:
            method = "address" if (r.get("addr_key") and (poi_addr == r["addr_key"]).any()) else "unlocated"
            located.append({"heritage_id": r["heritage_id"], "name": r["name"], "type": r.get("type"),
                            "method": method, "x": np.nan, "y": np.nan})
    located = pd.DataFrame(located)

    pairs: dict[tuple[str, str], dict] = {}

    def add(pi: int, hi: int, in_fp: bool, dist: float | None, addr: bool) -> None:
        pid = P.loc[pi, "poi_id"]
        hid = H.loc[hi, "heritage_id"]
        rec = pairs.setdefault((pid, hid), {"poi_id": pid, "poi_name": P.loc[pi, "name"], "heritage_id": hid,
                                            "heritage_name": H.loc[hi, "name"], "heritage_type": H.loc[hi].get("type"),
                                            "in_footprint": False, "dist_m": np.nan, "address_match": False,
                                            "human_check": ""})
        rec["in_footprint"] |= in_fp
        rec["address_match"] |= addr
        if dist is not None:
            rec["dist_m"] = dist if pd.isna(rec["dist_m"]) else min(rec["dist_m"], dist)

    hid_to_row = {h: i for i, h in enumerate(H["heritage_id"])}

    # (2) 距離 ≤ dmax：以 POI 的緩衝圓和文資點位做空間接合
    if len(HX) and len(P):
        buf = gpd.GeoDataFrame({"pi": P.index}, geometry=P.geometry.buffer(dmax), crs=crs)
        hit = gpd.sjoin(buf, HX[["heritage_id", "geometry"]], how="inner", predicate="intersects")
        for _, row in hit.iterrows():
            pi = int(row["pi"])
            hi = hid_to_row[row["heritage_id"]]
            d = float(P.geometry.iloc[pi].distance(HX.geometry.iloc[int(row["index_right"])]))
            if d <= dmax:
                add(pi, hi, False, round(d, 1), False)

    # (1) 建物輪廓內：POI 與文資落在同一棟
    if B is not None and len(HX):
        fp_of_h = {HX.loc[k, "heritage_id"]: int(v) for k, v in fp_idx.items() if pd.notna(v)}
        if fp_of_h:
            fps = B.loc[sorted(set(fp_of_h.values())), ["geometry"]].copy()
            fps["bidx"] = fps.index
            inside = gpd.sjoin(P[["geometry"]], fps, how="inner", predicate="within")
            by_b: dict[int, list[int]] = {}
            for pi, row in inside.iterrows():
                by_b.setdefault(int(row["bidx"]), []).append(int(pi))
            for hid, bidx in fp_of_h.items():
                for pi in by_b.get(bidx, []):
                    add(pi, hid_to_row[hid], True, None, False)

    # (3) 地址相同
    addr_map: dict[str, list[int]] = {}
    for pi, k in poi_addr.items():
        if k:
            addr_map.setdefault(k, []).append(int(pi))
    for hi, r in H.iterrows():
        k = r.get("addr_key")
        if k:
            for pi in addr_map.get(k, []):
                add(pi, hi, False, None, True)

    cols = ["poi_id", "poi_name", "heritage_id", "heritage_name", "heritage_type", "in_footprint", "dist_m",
            "address_match", "human_check"]
    matches = pd.DataFrame(list(pairs.values()), columns=cols)
    return located, matches


def kernel_density(px: np.ndarray, py: np.ndarray, hx: np.ndarray, hy: np.ndarray, bandwidth: float) -> np.ndarray:
    """高斯核密度（點／平方公里）。"""
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    hx, hy = np.asarray(hx, dtype=float), np.asarray(hy, dtype=float)
    if len(hx) == 0:
        return np.zeros(len(px))
    d2 = (px[:, None] - hx[None, :]) ** 2 + (py[:, None] - hy[None, :]) ** 2
    k = np.exp(-d2 / (2 * bandwidth ** 2)) / (2 * math.pi * bandwidth ** 2)
    return k.sum(axis=1) * 1e6


# ---------------------------------------------------------------------------
# 街區
# ---------------------------------------------------------------------------

MAJOR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link",
                  "primary_link", "secondary_link", "tertiary_link"}
_ROAD_NAME = re.compile(r"(路|街|大道)(.段)?$")
_LANE_NAME = re.compile(r"(巷|弄)")


def is_block_road(highway: str | None, name: str | None) -> bool:
    """切街區用的道路：大路一定切；residential/unclassified/living_street 只在「路／街」且不是巷弄時切。

    OSM 的 highway 與 name 都可能缺值（巷弄常常沒有名稱）。
    """
    excl = set(config.settings()["streets"]["exclude_highway"])
    highway = as_text(highway)
    name = as_text(name)
    if not highway or highway in excl:
        return False
    if highway in MAJOR_HIGHWAYS:
        return True
    if highway in {"residential", "unclassified", "living_street"}:
        return bool(name) and bool(_ROAD_NAME.search(name)) and not _LANE_NAME.search(name)
    return False


def make_blocks(roads: gpd.GeoDataFrame, main_poly) -> gpd.GeoDataFrame:
    crs = geo.metric_crs()
    R = roads.to_crs(crs)
    R = R[[is_block_road(h, n) for h, n in zip(R["highway"], R["name"])]]
    lines = list(R.geometry) + [main_poly.boundary]
    merged = unary_union(lines)
    polys = [p for p in polygonize(merged) if p.representative_point().within(main_poly)]
    polys = [p.intersection(main_poly) for p in polys]
    polys = [p for p in polys if p.area > 50]
    return gpd.GeoDataFrame({"raw_block": [f"R{i:03d}" for i in range(len(polys))]}, geometry=polys, crs=crs)


def assign_blocks(P: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame, col: str = "raw_block", snap_m: float = 25.0) -> pd.Series:
    """點落在哪個街區；落在路中心線上（不在任何街區內）的點，指派給 snap_m 內最近的街區。"""
    j = gpd.sjoin(P[["geometry"]], blocks[[col, "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = j[col]
    miss = out.isna()
    if miss.any():
        nn = gpd.sjoin_nearest(P.loc[miss, ["geometry"]], blocks[[col, "geometry"]], how="left", max_distance=snap_m)
        nn = nn[~nn.index.duplicated()]
        out.loc[miss] = nn[col]
    return out


def merge_small_blocks(blocks: gpd.GeoDataFrame, counts: pd.Series, min_size: int) -> dict[str, str]:
    """依預先登錄規則合併：POI 少於 min_size 的街區，與共用邊界最長的相鄰街區合併；從最小的開始，反覆到全部 ≥ min_size。

    回傳 raw_block → block_id 的對照。
    """
    geoms = {r: g for r, g in zip(blocks["raw_block"], blocks.geometry)}
    members = {r: {r} for r in geoms}
    n = {r: int(counts.get(r, 0)) for r in geoms}
    while True:
        small = [r for r in geoms if n[r] < min_size]
        if not small or len(geoms) == 1:
            break
        r = min(small, key=lambda k: (n[k], k))
        g = geoms[r]
        best, best_len = None, -1.0
        for k, h in geoms.items():
            if k == r:
                continue
            shared = g.boundary.intersection(h.boundary).length
            if shared > best_len or (shared == best_len and best is not None and k < best):
                best, best_len = k, shared
        if best is None or best_len <= 0:
            # 沒有相鄰（孤島）：併到距離最近的
            best = min((k for k in geoms if k != r), key=lambda k: g.distance(geoms[k]))
        geoms[best] = unary_union([geoms[best], g])
        members[best] |= members.pop(r)
        n[best] += n.pop(r)
        del geoms[r]
    mapping = {}
    for i, (rep, mem) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1]))):
        for m in mem:
            mapping[m] = f"B{i + 1:02d}"
    return mapping


# ---------------------------------------------------------------------------
# 主街、民生西路以北
# ---------------------------------------------------------------------------

def named_road_geometry(roads: gpd.GeoDataFrame, names: list[str]):
    crs = geo.metric_crs()
    R = roads.to_crs(crs)
    keys = {normalize_name(n) for n in names}
    sel = R[R["name"].apply(lambda n: normalize_name(n) in keys if n else False)]
    if sel.empty:
        return None
    return unary_union(list(sel.geometry))


def north_of(P: gpd.GeoDataFrame, road_geom) -> pd.Series:
    """點是否在某條（大致東西向的）道路以北：比較點與其在道路上最近點的 y。"""
    if road_geom is None:
        return pd.Series([None] * len(P), index=P.index, dtype=object)
    return P.geometry.apply(lambda p: bool(p.y > nearest_points(p, road_geom)[1].y))


# ---------------------------------------------------------------------------
# POI 上下文向量（100 公尺，排除自己）
# ---------------------------------------------------------------------------

CATEGORIES = ["熱食餐飲", "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "文化空間", "一般零售與品牌", "其他"]
_CAT_SLUG = {"熱食餐飲": "hotfood", "咖啡茶飲甜點": "cafe", "傳統業種": "traditional", "選物與文創零售": "select",
             "文化空間": "culture", "一般零售與品牌": "retail", "其他": "other"}


def context_vectors(x: np.ndarray, y: np.ndarray, cats: np.ndarray, radius: float) -> pd.DataFrame:
    """每個點 radius 內（不含自己）各類別比例；鄰居數為 0 時比例為 NaN。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cats = np.asarray(cats, dtype=object)
    d2 = (x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2
    near = d2 <= radius ** 2
    np.fill_diagonal(near, False)  # 排除自己（CLAUDE.md：鄰域特徵一律排除目標 POI 本身）
    total = near.sum(axis=1)
    out = {"poi_context_n": total}
    for c in CATEGORIES:
        k = (near & (cats[None, :] == c)).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"poi_context_{_CAT_SLUG[c]}"] = np.where(total > 0, k / np.maximum(total, 1), np.nan)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 組裝
# ---------------------------------------------------------------------------

def build_poi_master(poi: pd.DataFrame, roads: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame | None,
                     heritage: pd.DataFrame | None, temples: pd.DataFrame | None) -> dict:
    s = config.settings()
    crs = geo.metric_crs()
    G = geo.points_gdf(poi)
    P = G.to_crs(crs)
    main_poly = geo.main_polygon_metric()
    df = poi.copy()
    df["x"], df["y"] = P.geometry.x.values, P.geometry.y.values

    # 營業時間
    df = add_hours(df)
    # 粗分類
    df["places_category"] = [places_category(a, b, c) for a, b, c in zip(df["places_primary_type"], df["places_types"], df["osm_kind"])]
    df["traditional_subtype"] = df["name"].apply(traditional_subtype)
    df["likely_wenqing"] = [likely_wenqing(a, b) for a, b in zip(df["places_primary_type"], df["places_types"])]
    pmap = s["price_level_ordinal"]
    df["price_level_ord"] = df["price_level"].map(lambda v: pmap.get(as_text(v)))

    # 分析單位：主範圍、營業中、Places 來源
    df["analysis_unit"] = df["in_main"] & (df["business_status"] == "OPERATIONAL")

    # 街區
    blocks = make_blocks(roads, main_poly)
    raw = assign_blocks(P, blocks)
    df["raw_block"] = raw.values
    df.loc[~df["in_main"], "raw_block"] = None
    counts = df.loc[df["analysis_unit"]].groupby("raw_block").size()
    mapping = merge_small_blocks(blocks, counts, int(s["min_block_size"]))
    df["block_id"] = df["raw_block"].map(mapping)
    blocks["block_id"] = blocks["raw_block"].map(mapping)
    blocks_merged = blocks.dissolve(by="block_id", as_index=False)
    blocks_merged["n_analysis_poi"] = blocks_merged["block_id"].map(df.loc[df["analysis_unit"]].groupby("block_id").size()).fillna(0).astype(int)

    # 主街
    ms = named_road_geometry(roads, s["streets"]["main_street_names"])
    buf = float(s["streets"]["main_street_buffer_m"])
    df["dist_main_street_m"] = P.geometry.distance(ms).values if ms is not None else np.nan
    df["on_main_street"] = (df["dist_main_street_m"] <= buf) if ms is not None else None
    df["north_of_minsheng"] = north_of(P, named_road_geometry(roads, [s["minsheng_road_name"]])).values

    # 文資
    heritage_located = pd.DataFrame()
    hmatch = pd.DataFrame(columns=["poi_id", "heritage_id"])
    hx = np.array([]); hy = np.array([])
    if heritage is not None and len(heritage):
        heritage_located, hmatch = match_heritage(G, heritage, buildings)
        ok = heritage_located.dropna(subset=["x", "y"])
        hx, hy = ok["x"].values, ok["y"].values
    df["in_heritage"] = df["poi_id"].isin(set(hmatch["poi_id"])) if len(hmatch) else False
    if temples is not None and len(temples) and s["heritage"]["include_temples"]:
        T = geo.points_gdf(temples.dropna(subset=["lon", "lat"])).to_crs(crs)
        hx = np.concatenate([hx, T.geometry.x.values]); hy = np.concatenate([hy, T.geometry.y.values])
    for bw in s["kde_bandwidths_m"]:
        df[f"heritage_kde_{int(bw)}"] = kernel_density(df["x"].values, df["y"].values, hx, hy, float(bw))

    # 霞海城隍廟
    lm = s["landmarks"]["xiahai_temple"]
    tx, ty = geo.lonlat_to_metric(lm["lon"], lm["lat"])
    df["dist_temple_m"] = np.hypot(df["x"] - tx, df["y"] - ty)

    # 交通
    from .transit import transit_features

    tf = transit_features(G)
    for c in tf.columns:
        df[c] = tf[c].values

    # POI 上下文（用主範圍＋緩衝區的全部 POI 當鄰居，排除自己）
    ctx = context_vectors(df["x"].values, df["y"].values, df["places_category"].values, float(s["context_radius_m"]))
    for c in ctx.columns:
        df[c] = ctx[c].values

    return {"poi_master": df, "blocks": blocks_merged, "heritage_located": heritage_located, "heritage_match": hmatch,
            "main_street": ms, "heritage_xy": (hx, hy)}


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    au = df[df["analysis_unit"]]
    rows = [{"column": c, "missing_all": float(df[c].isna().mean()), "missing_analysis_unit": float(au[c].isna().mean()) if len(au) else np.nan}
            for c in df.columns]
    return pd.DataFrame(rows)


def check_maps(res: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import setup_cjk_font

    setup_cjk_font()
    crs = geo.metric_crs()
    df = res["poi_master"]
    main = gpd.GeoSeries([geo.main_polygon_metric()], crs=crs)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def base(title):
        fig, ax = plt.subplots(figsize=(6, 9))
        main.boundary.plot(ax=ax, color="#c53030", linewidth=1.5)
        ax.set_title(title, fontsize=10); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        return fig, ax

    # 文資
    fig, ax = base("文資點位（三角）與 in_heritage 的 POI（紅點）")
    ax.scatter(df["x"], df["y"], s=2, c="#999999")
    hit = df[df["in_heritage"] == True]  # noqa: E712
    ax.scatter(hit["x"], hit["y"], s=8, c="#c53030")
    hx, hy = res["heritage_xy"]
    ax.scatter(hx, hy, marker="^", s=30, c="#2b6cb0")
    p = out_dir / "check_heritage.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 街區
    fig, ax = base("合併後的街區（每塊 ≥ 15 個分析單位 POI）")
    res["blocks"].plot(ax=ax, column="block_id", cmap="tab20", alpha=0.5, edgecolor="k", linewidth=0.5)
    for _, b in res["blocks"].iterrows():
        c = b.geometry.representative_point()
        ax.annotate(f"{b['block_id']}\n{b['n_analysis_poi']}", (c.x, c.y), fontsize=6, ha="center")
    p = out_dir / "check_blocks.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)

    # 主街
    fig, ax = base("主街（藍線）與 on_main_street 的 POI（藍點）")
    if res["main_street"] is not None:
        gpd.GeoSeries([res["main_street"]], crs=crs).plot(ax=ax, color="#2b6cb0", linewidth=2)
    ax.scatter(df["x"], df["y"], s=2, c="#999999")
    on = df[df["on_main_street"] == True]  # noqa: E712
    ax.scatter(on["x"], on["y"], s=6, c="#2b6cb0")
    p = out_dir / "check_main_street.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-heritage", action="store_true",
                    help="文資清單還沒好時先跑其他欄位（in_heritage 全為 False、heritage_kde 全為 0）")
    ap.add_argument("--accept-warnings", action="store_true",
                    help="已知文資資料不完整仍要往下跑（請同時在 docs/deviations.md 記錄）")
    args = ap.parse_args(argv)
    from .heritage import load_heritage as load_h
    from .heritage import load_temples, print_validation, validate
    from .t03_osm_merge import load_osm, osm_buildings, osm_roads

    poi = pd.read_parquet(config.path("interim") / "poi_merged.parquet")
    els = load_osm()
    roads, buildings = osm_roads(els), osm_buildings(els)
    print(f"OSM：道路 {len(roads)} 條、建物輪廓 {len(buildings)} 個")

    heritage, temples, temples_all = None, None, pd.DataFrame()
    if not args.no_heritage:
        heritage = load_h()
        temples, temples_all = load_temples()
        v = validate(heritage, temples_all)
        print()
        n_stop = print_validation(v)
        tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
        v.to_csv(tables / "t04_heritage_validation.csv", index=False, encoding="utf-8-sig")
        if n_stop and not args.accept_warnings:
            print(f"\n有 {n_stop} 項「必須處理」。修好後重跑，"
                  "或確認影響後加 --accept-warnings 繼續（並寫進 docs/deviations.md）。")
            raise SystemExit(2)
        if n_stop:
            print(f"\n（已以 --accept-warnings 略過 {n_stop} 項警告，請記錄在 docs/deviations.md）")
        print()

    res = build_poi_master(poi, roads, buildings, heritage, temples)
    df = res["poi_master"]
    out = config.path("final") / "poi_master.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    res["blocks"].to_crs(geo.WGS84).to_file(config.path("interim") / "blocks.geojson", driver="GeoJSON")
    res["heritage_match"].to_csv(config.path("interim") / "heritage_match.csv", index=False, encoding="utf-8-sig")
    if len(res["heritage_located"]):
        res["heritage_located"].to_csv(config.path("interim") / "heritage_located.csv", index=False, encoding="utf-8-sig")
    tables = config.path("tables"); tables.mkdir(parents=True, exist_ok=True)
    mr = missing_report(df); mr.to_csv(tables / "t04_missing_report.csv", index=False, encoding="utf-8-sig")

    au = df[df["analysis_unit"]]
    print(f"poi_master：{len(df)} 筆；分析單位（主範圍且營業中）：{len(au)}")
    if len(res["heritage_located"]):
        hl = res["heritage_located"]
        ok = (hl["method"] != "unlocated")
        print(f"文資定位率：{ok.mean():.1%}（{int(ok.sum())}/{len(hl)}）"
              f"｜含建物輪廓 {int((hl['method'] == 'coord+footprint').sum())} 筆")
        if ok.mean() < 0.8:
            print("■ 文資定位率低於八成：請停下來，由隊友 A 人工補比對（CLAUDE.md）。")
        hm = res["heritage_match"]
        print(f"文資配對：{len(hm)} 組（POI {hm['poi_id'].nunique() if len(hm) else 0} 家）"
              f"｜輪廓內 {int(hm['in_footprint'].sum()) if len(hm) else 0}"
              f"、距離內 {int(hm['dist_m'].notna().sum()) if len(hm) else 0}"
              f"、地址相符 {int(hm['address_match'].sum()) if len(hm) else 0}")
    print(f"in_heritage 的分析單位 POI：{int(au['in_heritage'].sum())}")
    for c in ("heritage_kde_150", "heritage_kde_300"):
        if c in au:
            print(f"{c}：中位數 {au[c].median():.1f}、最大 {au[c].max():.1f}（點／平方公里）")
    print("closes_before_19（分析單位）：")
    print(au["closes_before_19"].value_counts(dropna=False).to_string())
    print("hours_status：", au["hours_status"].value_counts().to_dict())
    print(f"街區數：{df['block_id'].nunique()}；最小街區 POI 數：{res['blocks']['n_analysis_poi'].min()}"
          f"；中位數 {res['blocks']['n_analysis_poi'].median():.0f}")
    if df["block_id"].nunique() < 20:
        print("▲ 街區數少於 20：H6 的群集標準誤會低估，請依 實作審閱筆記.md 1.1 改用 wild cluster bootstrap。")
    print("業態大類（分析單位，依 Places 類別粗分）：")
    print(au["places_category"].value_counts().to_string())
    print(f"on_main_street：{int(au['on_main_street'].sum())} 家"
          f"｜north_of_minsheng：{int(au['north_of_minsheng'].sum())} 家")
    print("缺失比例最高的欄位（分析單位）：")
    print(mr.sort_values("missing_analysis_unit", ascending=False).head(8).to_string(index=False))
    for p in check_maps(res, config.path("figures")):
        print("檢查地圖：", p)
    print(f"\n下一步：T05 官網抓取（背景跑）→ T06 prompt。只需結構化資料的 H1／H5／H3 現在就可以跑：")
    print("  python -m dadaocheng.t11_h1_hours")
    print("  python -m dadaocheng.t12_h5_heritage")
    print("  python -m dadaocheng.t13_h3_zoning")


if __name__ == "__main__":
    main()
