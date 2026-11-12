"""從 OSM 道路中心線自動建構主範圍邊界（取代人工描繪）。

為什麼不用手畫：主範圍決定誰是分析單位，而「沿道路中心線」這件事用手畫會有數十公尺的
誤差——11/12 的草稿邊界就是因此把延平北路沿街的 23 筆文資中的 18 筆排除在外。
從 OSM 的道路幾何直接算，結果可重現、可稽核，而且人只需要在地圖上「驗收」而不是「描繪」。

做法：
  1. 從 data/raw/osm/osm_raw.json 找出四條界線道路的所有 way（名稱正規化後比對，含「一段／二段」）。
  2. 每條道路合併成一條線。
  3. 兩兩求交點得到四個轉角；若 OSM 有缺口而求不到交點，改用兩線的最近點。
  4. 每條道路取「兩個轉角之間」的實際幾何（所以邊界會跟著道路彎曲，不是直線）。
  5. 四段接成封閉環 → 多邊形。檢查它包含內部定錨點（霞海城隍廟）。
  6. 寫入 config/boundary_main.geojson，status 標 FINAL_FROM_OSM，並記錄來源與參數。

用法：
  python scripts/build_boundary.py --preview     # 只計算與驗收，不寫檔
  python scripts/build_boundary.py --write       # 寫入 config/boundary_main.geojson（會備份舊檔）
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon, mapping
from shapely.ops import linemerge, nearest_points, substring, unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dadaocheng import config, geo  # noqa: E402

# 四條界線道路。比對時用正規化後的「開頭相符」，所以「延平北路一段」「延平北路二段」都會命中。
BOUNDARY_ROADS = {
    "south": "南京西路",
    "east": "延平北路",
    "north": "民權西路",
    "west": "環河北路",
}
# 轉角的順序：西南 → 東南 → 東北 → 西北
CORNERS = [("south", "west"), ("south", "east"), ("north", "east"), ("north", "west")]
# 內部定錨點：霞海城隍廟（迪化街一段 61 號）。
ANCHOR = (121.51015, 25.05568)
# 只取定錨點周圍這個半徑內的道路幾何（排除遠處同名路段，例如延平北路三段北端）
CLIP_RADIUS_M = 1500
# 同一條路的不同 way 端點相距在這個距離內，視為相連（補 OSM 的小缺口、分隔島兩側車道）
BRIDGE_M = 30


def is_main_road(name, base: str) -> bool:
    """只接受「南京西路」「延平北路二段」這類主路名，排除「南京西路155巷」「民權西路144巷5弄」。"""
    if not name:
        return False
    s = re.sub(r"\s+", "", str(name)).replace("臺", "台")
    return re.fullmatch(re.escape(base) + r"([一二三四五六七八九十0-9０-９]+段)?", s) is not None


def load_named_roads(path: Path | None = None) -> dict[str, MultiLineString]:
    from dadaocheng.t03_osm_merge import load_osm, osm_roads

    roads = osm_roads(load_osm(path))
    if roads.empty:
        raise SystemExit("OSM 裡沒有道路。請先跑 python -m dadaocheng.t03_osm_merge --download")
    R = roads.to_crs(geo.metric_crs())
    ax, ay = geo.lonlat_to_metric(*ANCHOR)
    R = R[R.geometry.distance(Point(ax, ay)) <= CLIP_RADIUS_M]
    out, report = {}, []
    for side, base in BOUNDARY_ROADS.items():
        sel = R[R["name"].apply(lambda n: is_main_road(n, base))]
        if sel.empty:
            raise SystemExit(
                f"OSM 找不到名稱以「{base}」開頭的道路。\n"
                f"  請跑 python scripts/build_boundary.py --list-names 看 OSM 實際的道路名稱，"
                f"再改 BOUNDARY_ROADS 或改用手動備援（見 實作計畫 S04）。"
            )
        u = unary_union(list(sel.geometry))
        merged = linemerge(u) if u.geom_type != "LineString" else u
        out[side] = merged   # 可能是 MultiLineString（分隔車道、缺口），road_segment 會處理
        report.append({"side": side, "道路": base, "OSM way 數": len(sel),
                       "總長度（公尺）": round(float(sum(g.length for g in sel.geometry)), 1),
                       "合併後段數": len(getattr(merged, "geoms", [merged])),
                       "名稱": "、".join(sorted({str(n) for n in sel["name"].dropna().unique()}))})
    print("找到的界線道路：")
    import pandas as pd

    print(pd.DataFrame(report).to_string(index=False))
    return out


def list_names(path: Path | None = None) -> None:
    from dadaocheng.t03_osm_merge import load_osm, osm_roads
    import pandas as pd

    roads = osm_roads(load_osm(path))
    R = roads.to_crs(geo.metric_crs())
    R["len"] = R.geometry.length
    named = (R.dropna(subset=["name"]).groupby("name")
              .agg(way數=("name", "size"), 總長公尺=("len", lambda s: round(float(s.sum()), 1)))
              .sort_values("總長公尺", ascending=False))
    print(f"OSM 內有名稱的道路（前 40 條，共 {len(named)} 條）：")
    print(named.head(40).to_string())


def corner_point(a, b) -> tuple[Point, str]:
    """兩條道路的交點；沒有交點時用兩者的最近點（OSM 有缺口時）。"""
    inter = a.intersection(b)
    if not inter.is_empty:
        if inter.geom_type == "Point":
            return inter, "交點"
        pts = [g for g in getattr(inter, "geoms", [inter]) if g.geom_type == "Point"]
        if pts:
            # 多個交點（分隔車道會有 2–4 個）→ 取最靠近定錨點的，也就是內側車道的路口
            ax, ay = geo.lonlat_to_metric(*ANCHOR)
            a0 = Point(ax, ay)
            best = min(pts, key=lambda q: q.distance(a0))
            return best, (f"交點（{len(pts)} 個，取最靠近範圍中心者）" if len(pts) > 1 else "交點")
        return inter.centroid, "交疊線段的中點"
    p, q = nearest_points(a, b)
    d = p.distance(q)
    return Point((p.x + q.x) / 2, (p.y + q.y) / 2), f"最近點（相距 {d:.1f} 公尺，OSM 可能有缺口）"


def _graph(line) -> dict:
    """把道路幾何轉成圖：節點是頂點座標，邊是線段；不同段的端點在 BRIDGE_M 內補一條邊。"""
    parts = list(getattr(line, "geoms", [line]))
    G: dict = {}

    def add(a, b):
        w = float(np.hypot(a[0] - b[0], a[1] - b[1]))
        G.setdefault(a, {})[b] = min(w, G.get(a, {}).get(b, w))
        G.setdefault(b, {})[a] = min(w, G.get(b, {}).get(a, w))

    ends = []
    for k, part in enumerate(parts):
        cs = [tuple(map(float, c[:2])) for c in part.coords]
        for a, b in zip(cs, cs[1:]):
            add(a, b)
        ends += [(k, cs[0]), (k, cs[-1])]
    for i, (ki, a) in enumerate(ends):
        for kj, b in ends[i + 1:]:
            if ki != kj and np.hypot(a[0] - b[0], a[1] - b[1]) <= BRIDGE_M:
                add(a, b)
    return G


def _snap(line, p: Point, G: dict):
    """把點投影到最近的線段上，插入成新節點。"""
    parts = list(getattr(line, "geoms", [line]))
    best = None
    for part in parts:
        cs = [tuple(map(float, c[:2])) for c in part.coords]
        for a, b in zip(cs, cs[1:]):
            seg = LineString([a, b])
            d = seg.distance(p)
            if best is None or d < best[0]:
                q = seg.interpolate(seg.project(p))
                best = (d, a, b, (float(q.x), float(q.y)))
    _, a, b, q = best
    for u in (a, b):
        w = float(np.hypot(u[0] - q[0], u[1] - q[1]))
        G.setdefault(q, {})[u] = w
        G.setdefault(u, {})[q] = w
    return q


def road_segment(line, p1: Point, p2: Point):
    """取道路上 p1 到 p2 之間的實際幾何（沿道路網的最短路徑），從 p1 走到 p2。

    道路在 OSM 裡常是好幾段（分隔車道、缺口、同名但不相連的路段），
    所以不能直接對整條線取 substring，而是在道路網上走最短路徑。
    """
    if line.geom_type == "LineString":
        d1, d2 = line.project(p1), line.project(p2)
        seg = substring(line, min(d1, d2), max(d1, d2))
        if seg.is_empty or seg.length < 1:
            return LineString([p1, p2])
        cs = list(seg.coords)
        if Point(cs[0]).distance(p1) > Point(cs[-1]).distance(p1):
            cs = cs[::-1]
        return LineString(cs)

    G = _graph(line)
    s, e = _snap(line, p1, G), _snap(line, p2, G)
    dist, prev, pq = {s: 0.0}, {}, [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == e:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in G[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if e not in dist:
        print(f"  ▲ 兩個轉角之間的道路不相連（缺口 > {BRIDGE_M} 公尺），這一段改用直線連接")
        return LineString([p1, p2])
    path = [e]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return LineString(path[::-1])


def build(roads: dict) -> tuple[Polygon, list[dict]]:
    corners, notes = {}, []
    for s1, s2 in CORNERS:
        pt, how = corner_point(roads[s1], roads[s2])
        corners[(s1, s2)] = pt
        lon, lat = geo.metric_to_lonlat(pt.x, pt.y)
        notes.append({"轉角": f"{BOUNDARY_ROADS[s1]} × {BOUNDARY_ROADS[s2]}",
                      "判定方式": how, "lon": round(lon, 6), "lat": round(lat, 6)})
    ring: list[tuple[float, float]] = []
    order = [("south", ("south", "west"), ("south", "east")),
             ("east", ("south", "east"), ("north", "east")),
             ("north", ("north", "east"), ("north", "west")),
             ("west", ("north", "west"), ("south", "west"))]
    for side, c_from, c_to in order:
        seg = road_segment(roads[side], corners[c_from], corners[c_to])
        cs = list(seg.coords)
        ring.extend(cs)   # 不省略首點：相鄰兩條路的轉角投影點可能差幾公尺
    if ring[0] != ring[-1]:
        ring.append(ring[0])
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    ax, ay = geo.lonlat_to_metric(*ANCHOR)
    if not poly.contains(Point(ax, ay)):
        raise SystemExit("建出來的多邊形沒有包含霞海城隍廟，四條道路的接法可能有問題。"
                         "請跑 --list-names 檢查道路名稱，或改用手動備援（實作計畫 S04）。")
    return poly, notes


def verify(poly: Polygon) -> None:
    import pandas as pd

    from dadaocheng.heritage import load_heritage

    print(f"\n面積：{poly.area / 1e6:.3f} 平方公里")
    old_path = config.path("boundary_main")
    if old_path.exists():
        old = unary_union(geo.read_boundary("boundary_main").to_crs(geo.metric_crs()).geometry)
        print(f"舊邊界面積：{old.area / 1e6:.3f} 平方公里"
              f"｜新增 {(poly.difference(old)).area / 1e6:.3f}｜減少 {(old.difference(poly)).area / 1e6:.3f}")

    h = load_heritage(verbose=False).dropna(subset=["lon", "lat"])
    h = h[h["district"].eq("大同區") | h["address"].fillna("").str.contains("大同區")]
    # 和正式分析一樣外擴 main_inclusive_buffer_m，才能把面向界線道路的建物算進來
    b = float(config.settings().get("main_inclusive_buffer_m", 0) or 0)
    inc = poly.buffer(b, join_style="mitre") if b > 0 else poly
    rows, outside = [], []
    for base in [*BOUNDARY_ROADS.values(), "迪化街"]:
        g = h[h["address"].fillna("").str.contains(base)]
        k = 0
        for _, r in g.iterrows():
            pt = Point(*geo.lonlat_to_metric(r.lon, r.lat))
            if inc.contains(pt):
                k += 1
            else:
                outside.append({"道路": base, "名稱": r.get("name", ""), "地址": r.get("address", ""),
                                "距邊界（公尺）": round(inc.exterior.distance(pt), 1)})
        rows.append({"沿線道路": base, "文資筆數": len(g), "落在範圍內": k,
                     "比例": k / len(g) if len(g) else np.nan})
    print(f"\n驗收：沿街文資有沒有被納入（已外擴 {b:.0f} 公尺，與正式分析相同）")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.1%}"))
    if outside:
        print("\n落在範圍外的文資（請隊友 A 逐筆確認：段別在範圍外、或在道路對側，就是正確的排除）：")
        print(pd.DataFrame(outside).sort_values(["道路", "距邊界（公尺）"]).to_string(index=False))
    print("\n延平北路的比例不會到 100%：一段（南京西路以南）、三段（民權西路以北）與道路東側的文資本來就在範圍外。"
          "\n要看的是上面的清單裡，有沒有「二段、在道路西側、距邊界很近」卻被排除的。")

    pm = config.path("final") / "poi_master.parquet"
    if pm.exists():
        import pandas as pd

        poi = pd.read_parquet(pm)
        pts = [Point(*geo.lonlat_to_metric(a, b)) for a, b in zip(poi["lon"], poi["lat"])]
        now_in = np.array([poly.contains(p) for p in pts])
        was_in = poi["in_main"].astype(bool).to_numpy()
        print(f"\nPOI 變動：原本 in_main {was_in.sum()} 家 → 新邊界 {now_in.sum()} 家"
              f"（新增 {int((now_in & ~was_in).sum())}、移出 {int((~now_in & was_in).sum())}）")


def write(poly: Polygon, notes: list[dict]) -> Path:
    out = config.path("boundary_main")
    if out.exists():
        bak = out.with_suffix(f".draft_backup_{date.today():%Y%m%d}.geojson")
        shutil.copy(out, bak)
        print(f"舊檔已備份：{bak}")
    gdf = gpd.GeoDataFrame(
        [{"name": "大稻埕主範圍（南京西路、民權西路、環河北路、延平北路，沿道路中心線）",
          "status": "FINAL_FROM_OSM",
          "drawn_by": "scripts/build_boundary.py（由 OSM 道路中心線自動建構）",
          "date": f"{date.today():%Y-%m-%d}",
          "roads": "；".join(BOUNDARY_ROADS.values()),
          "corners": json.dumps(notes, ensure_ascii=False),
          "osm_attribution": "© OpenStreetMap contributors, ODbL"}],
        geometry=[poly], crs=geo.metric_crs()).to_crs(geo.WGS84)
    gdf.to_file(out, driver="GeoJSON")
    print(f"已寫入：{out}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preview", action="store_true", help="計算並驗收，不寫檔")
    g.add_argument("--write", action="store_true", help="寫入 config/boundary_main.geojson（先備份舊檔）")
    g.add_argument("--list-names", action="store_true", help="列出 OSM 的道路名稱（找不到界線道路時用）")
    ap.add_argument("--osm", default=None, help="osm_raw.json 的路徑（預設 data/raw/osm/osm_raw.json）")
    args = ap.parse_args(argv)
    path = Path(args.osm) if args.osm else None

    if args.list_names:
        list_names(path)
        return
    roads = load_named_roads(path)
    poly, notes = build(roads)
    import pandas as pd

    print("\n四個轉角：")
    print(pd.DataFrame(notes).to_string(index=False))
    verify(poly)
    if args.write:
        write(poly, notes)
        print("\n接下來（實作計畫 S05）：")
        print("  python -c \"from dadaocheng import geo; geo.make_buffer_geojson()\"")
        print("  python -m dadaocheng.transit")
        print("  python -m dadaocheng.t02_places_census --cache-only --extent-m 150")
    else:
        print("\n（--preview 模式，沒有寫檔。確認上面的驗收數字之後改用 --write）")


if __name__ == "__main__":
    main()
