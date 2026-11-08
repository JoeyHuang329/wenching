"""T04：文資比對、街區切割、結構化特徵，產生 poi_master。

輸入：
  data/interim/poi_merged.parquet                 （T03）
  data/raw/osm/osm_raw.json                       （T03 下載的 OSM：道路、建物）
  data/raw/heritage/heritage_list.csv             （人工下載整理：文資局／文化局的古蹟與歷史建築）
  data/raw/heritage/temples.csv                   （百年廟宇，人工確認；可選）
  data/interim/transit_*.geojson                  （python -m dadaocheng.transit）
輸出：
  data/final/poi_master.parquet
  data/interim/heritage_match.csv                 （給彭郁媗人工檢查）
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
from .names import normalize_address, normalize_name

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

def places_category(primary: str | None, types_json: str | None, osm_kind: str | None = None) -> str:
    cmap: dict[str, list[str]] = config.settings()["places_category_map"]
    lookup = {t: cat for cat, ts in cmap.items() for t in ts}
    if primary and primary in lookup:
        return lookup[primary]
    types = json.loads(types_json) if isinstance(types_json, str) and types_json else []
    for t in types:
        if t in lookup:
            return lookup[t]
    if osm_kind:
        k, _, v = osm_kind.partition("=")
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
    types = set(json.loads(types_json)) if isinstance(types_json, str) and types_json else set()
    return bool((primary in lw) or (types & lw))


# ---------------------------------------------------------------------------
# 文資比對
# ---------------------------------------------------------------------------

def load_heritage(path: Path | None = None) -> pd.DataFrame:
    path = path or (config.path("raw") / "heritage" / "heritage_list.csv")
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在。請依 待執行任務描述.md T04-H1 整理文資清單。")
    h = pd.read_csv(path, dtype={"heritage_id": str}, encoding="utf-8-sig")
    for c in ["lon", "lat"]:
        if c not in h.columns:
            h[c] = np.nan
    h["addr_key"] = h["address"].apply(normalize_address)
    return h


def match_heritage(pois: gpd.GeoDataFrame, heritage: pd.DataFrame, buildings: gpd.GeoDataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """回傳 (heritage_located：每筆文資的定位結果, poi_matches：POI × 文資的配對)。"""
    crs = geo.metric_crs()
    dmax = float(config.settings()["heritage"]["match_distance_m"])
    P = pois.to_crs(crs)
    poi_addr = pois["formatted_address"].apply(normalize_address) if "formatted_address" in pois else pd.Series("", index=pois.index)
    B = buildings.to_crs(crs) if buildings is not None and len(buildings) else None
    located, matches = [], []
    for _, h in heritage.iterrows():
        fp, pt, method = None, None, None
        if pd.notna(h["lon"]) and pd.notna(h["lat"]):
            pt = gpd.GeoSeries([Point(h["lon"], h["lat"])], crs=geo.WGS84).to_crs(crs).iloc[0]
            method = "coord"
            if B is not None:
                hit = B[B.contains(pt)]
                if len(hit):
                    fp = hit.geometry.iloc[0]
                    method = "coord+footprint"
        elif h["addr_key"]:
            same = poi_addr[poi_addr == h["addr_key"]]
            if len(same):
                pt = P.loc[same.index[0]].geometry
                method = "address"
                if B is not None:
                    hit = B[B.contains(pt)]
                    if len(hit):
                        fp = hit.geometry.iloc[0]
                        method = "address+footprint"
        located.append({"heritage_id": h["heritage_id"], "name": h["name"], "type": h.get("type"),
                        "method": method or "unlocated", "x": pt.x if pt is not None else np.nan,
                        "y": pt.y if pt is not None else np.nan})
        if pt is None:
            continue
        for idx, p in P.iterrows():
            in_fp = fp is not None and fp.buffer(0.5).contains(p.geometry)
            d = p.geometry.distance(pt)
            addr_same = bool(h["addr_key"]) and poi_addr.get(idx, "") == h["addr_key"]
            if in_fp or d <= dmax or addr_same:
                matches.append({"poi_id": p["poi_id"], "poi_name": p["name"], "heritage_id": h["heritage_id"],
                                "heritage_name": h["name"], "in_footprint": in_fp, "dist_m": round(d, 1),
                                "address_match": addr_same, "human_check": ""})
    return pd.DataFrame(located), pd.DataFrame(matches)


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


def is_block_road(highway: str, name: str | None) -> bool:
    """切街區用的道路：大路一定切；residential/unclassified/living_street 只在「路／街」且不是巷弄時切。"""
    excl = set(config.settings()["streets"]["exclude_highway"])
    if highway in excl:
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
    df["price_level_ord"] = df["price_level"].map(pmap)

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
    ap.add_argument("--no-heritage", action="store_true", help="文資清單還沒好時先跑其他欄位（in_heritage 全為 False）")
    args = ap.parse_args(argv)
    from .t03_osm_merge import load_osm, osm_buildings, osm_roads

    poi = pd.read_parquet(config.path("interim") / "poi_merged.parquet")
    els = load_osm()
    roads, buildings = osm_roads(els), osm_buildings(els)
    heritage = None if args.no_heritage else load_heritage()
    tp = config.project_root() / config.settings()["heritage"]["temples_csv"]
    temples = pd.read_csv(tp, encoding="utf-8-sig") if tp.exists() else None
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
        rate = (hl["method"] != "unlocated").mean()
        print(f"文資比對率（定位成功的文資比例）：{rate:.1%}（{(hl['method'] != 'unlocated').sum()}/{len(hl)}）")
        if rate < 0.8:
            print("⚠ 文資比對率低於八成：請停下來，由彭郁媗人工補比對（CLAUDE.md）。")
    print(f"in_heritage 的分析單位 POI：{int(au['in_heritage'].sum())}")
    print("closes_before_19（分析單位）：")
    print(au["closes_before_19"].value_counts(dropna=False).to_string())
    print("hours_status：", au["hours_status"].value_counts().to_dict())
    print(f"街區數：{df['block_id'].nunique()}；最小街區 POI 數：{res['blocks']['n_analysis_poi'].min()}")
    print("缺失比例最高的欄位：")
    print(mr.sort_values("missing_analysis_unit", ascending=False).head(10).to_string(index=False))
    for p in check_maps(res, config.path("figures")):
        print("檢查地圖：", p)


if __name__ == "__main__":
    main()
