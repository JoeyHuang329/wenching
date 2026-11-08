"""T02：Google Places Nearby Search（New）自適應網格普查。

規格（實作規劃 B T02）：
- 起始網格覆蓋「主範圍＋緩衝區」外框，邊長約 200 公尺；每格用外接圓（半徑＝半對角線）查詢。
- maxResultCount 20、rankPreference DISTANCE、zh-TW、TW、不設 includedTypes。
- 回傳滿 20 筆就把該格切成四格重查，最小邊長約 25 公尺。
- 欄位遮罩見 config；所有呼叫走 T00 呼叫層（快取＋計數＋上限 900）。

用法：
  python -m dadaocheng.t02_places_census --dry-run        # 只算起始格數，不呼叫 API
  python -m dadaocheng.t02_places_census --single-cell    # 只跑 1 格，給人到 Cloud Console 確認 SKU
  python -m dadaocheng.t02_places_census --run            # 全部（已快取的格子不會再付費）

重跑是安全的：同一個格子的請求內容相同，會直接讀快取。
"""

from __future__ import annotations

import argparse
import json
import math
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from shapely.geometry import box

from . import config, geo


@dataclass
class Cell:
    cell_id: str
    x0: float
    y0: float
    size: float
    depth: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return self.x0 + self.size / 2, self.y0 + self.size / 2

    @property
    def radius(self) -> float:
        return self.size * math.sqrt(2) / 2

    def polygon(self):
        return box(self.x0, self.y0, self.x0 + self.size, self.y0 + self.size)

    def split(self) -> list["Cell"]:
        h = self.size / 2
        return [
            Cell(f"{self.cell_id}.{k}", self.x0 + dx * h, self.y0 + dy * h, h, self.depth + 1)
            for k, (dx, dy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)])
        ]


@dataclass
class CensusResult:
    places: list[dict] = field(default_factory=list)
    cells: list[dict] = field(default_factory=list)

    @property
    def saturated_min_cells(self) -> int:
        return sum(1 for c in self.cells if c["saturated_at_min"])


def initial_cells(polygon, size: float, skip_outside: bool = True) -> list[Cell]:
    minx, miny, maxx, maxy = polygon.bounds
    nx = math.ceil((maxx - minx) / size)
    ny = math.ceil((maxy - miny) / size)
    cells = []
    for j in range(ny):
        for i in range(nx):
            c = Cell(f"g{j:02d}_{i:02d}", minx + i * size, miny + j * size, size)
            if skip_outside and not c.polygon().intersects(polygon):
                continue
            cells.append(c)
    return cells


def run_census(
    polygon,
    fetch: Callable[[Cell], list[dict]],
    initial_size: float,
    min_size: float,
    max_results: int = 20,
    skip_outside: bool = True,
    max_cells: int | None = None,
    priority_polygon=None,
) -> CensusResult:
    """四分樹普查。fetch(cell) 回傳該格的 places 清單（已經過呼叫層）。

    priority_polygon（通常是主範圍）：與它相交的格子（含其子格）優先處理。
    這樣即使請求數撞到上限，主範圍也已經完整，缺的只會是緩衝區。
    """
    res = CensusResult()
    counter = itertools.count()
    heap: list = []

    def push(c: Cell) -> None:
        pri = 0 if (priority_polygon is None or c.polygon().intersects(priority_polygon)) else 1
        heapq.heappush(heap, (pri, c.depth, next(counter), c))

    for c in initial_cells(polygon, initial_size, skip_outside):
        push(c)
    while heap:
        if max_cells is not None and len(res.cells) >= max_cells:
            break
        cell = heapq.heappop(heap)[3]
        places = fetch(cell)
        full = len(places) >= max_results
        can_split = cell.size / 2 >= min_size - 1e-6
        res.cells.append(
            {"cell_id": cell.cell_id, "depth": cell.depth, "size_m": cell.size, "n": len(places),
             "split": full and can_split, "saturated_at_min": full and not can_split,
             "cx": cell.center[0], "cy": cell.center[1],
             "in_priority": priority_polygon is None or cell.polygon().intersects(priority_polygon)}
        )
        for p in places:
            res.places.append({**p, "_cell_id": cell.cell_id, "_depth": cell.depth})
        if full and can_split:
            for child in cell.split():
                if not skip_outside or child.polygon().intersects(polygon):
                    push(child)
    return res


def flatten_place(p: dict) -> dict:
    loc = p.get("location", {}) or {}
    return {
        "place_id": p.get("id"),
        "name": (p.get("displayName") or {}).get("text"),
        "lon": loc.get("longitude"),
        "lat": loc.get("latitude"),
        "places_primary_type": p.get("primaryType"),
        "places_types": json.dumps(p.get("types", []), ensure_ascii=False),
        "business_status": p.get("businessStatus"),
        "formatted_address": p.get("formattedAddress"),
        "regular_opening_hours": json.dumps(p.get("regularOpeningHours"), ensure_ascii=False) if p.get("regularOpeningHours") else None,
        "user_rating_count": p.get("userRatingCount"),
        "rating": p.get("rating"),
        "price_level": p.get("priceLevel"),
        "website_uri": p.get("websiteUri"),
        "cell_id": p.get("_cell_id"),
        "cell_depth": p.get("_depth"),
    }


def to_dataframe(res: CensusResult) -> pd.DataFrame:
    df = pd.DataFrame([flatten_place(p) for p in res.places])
    if df.empty:
        return df
    # 去掉格子重疊造成的重複 place_id：保留最深（最小格）那一筆
    df = df.sort_values("cell_depth", ascending=False).drop_duplicates("place_id").reset_index(drop=True)
    return df


def make_fetch(caller, send, estimate, approved: bool = False):
    from .clients import places_nearby_request

    def fetch(cell: Cell) -> list[dict]:
        lon, lat = geo.metric_to_lonlat(*cell.center)
        req = places_nearby_request(lat, lon, cell.radius)
        resp = caller.call(req, send=send, estimate_cost=estimate, approved=approved)
        return resp.get("places", []) or []

    return fetch


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--single-cell", action="store_true")
    g.add_argument("--run", action="store_true")
    ap.add_argument("--approve", action="store_true", help="人已確認可繼續（七成門檻時用）")
    args = ap.parse_args(argv)

    s = config.settings()
    ps = s["places"]
    if geo.boundary_is_draft():
        print("⚠ 主範圍邊界仍是 DRAFT。正式普查前請先換成彭郁媗畫的邊界，並重新產生緩衝區。")
    if not config.path("boundary_buffer").exists():
        geo.make_buffer_geojson()
    poly = geo.study_polygon_metric()
    cells0 = initial_cells(poly, ps["initial_cell_m"], ps["skip_cells_outside_polygon"])
    print(f"起始格數：{len(cells0)}（邊長 {ps['initial_cell_m']} 公尺，研究範圍面積 {poly.area / 1e6:.2f} 平方公里）")
    if args.dry_run:
        # 粗估：每格平均若需切到 50 公尺，請求數 ≈ 起始格 × (1+4+16)
        print(f"請求數粗估：最少 {len(cells0)}，若每格都切到 50 公尺約 {len(cells0) * 21}；上限 {s['places_max_requests']}")
        return

    from .clients import places_caller

    caller, send, estimate = places_caller()
    if args.single_cell:
        # 選最靠近主範圍中心的起始格
        cx, cy = geo.main_polygon_metric().centroid.coords[0]
        cell = min(cells0, key=lambda c: (c.center[0] - cx) ** 2 + (c.center[1] - cy) ** 2)
        places = make_fetch(caller, send, estimate, args.approve)(cell)
        print(f"格子 {cell.cell_id}：回傳 {len(places)} 筆")
        if places:
            print("第一筆的欄位：", sorted(places[0].keys()))
            print(json.dumps(flatten_place(places[0]), ensure_ascii=False, indent=2))
        print("\n下一步：到 Cloud Console → APIs & Services → Places API (New) → Metrics／Billing，"
              "確認這 1 次請求被計在 Nearby Search Enterprise。確認後再跑 --run。")
        print(caller.ledger.summary())
        return

    res = run_census(poly, make_fetch(caller, send, estimate, args.approve), ps["initial_cell_m"], ps["min_cell_m"],
                     ps["max_result_count"], ps["skip_cells_outside_polygon"],
                     priority_polygon=geo.main_polygon_metric())
    df = to_dataframe(res)
    out = config.path("interim") / "places_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    cells = pd.DataFrame(res.cells)
    cells.to_csv(config.path("interim") / "places_cells_log.csv", index=False, encoding="utf-8-sig")
    print(f"格子數：{len(cells)}；回傳筆數（含重複）：{len(res.places)}；去重後 place_id：{len(df)}")
    print(f"最小格（{ps['min_cell_m']} 公尺）仍滿 20 筆的格數：{res.saturated_min_cells}（>0 代表該處可能有漏，請回報）")
    print(caller.ledger.summary())
    print(f"已存：{out}")


if __name__ == "__main__":
    main()
