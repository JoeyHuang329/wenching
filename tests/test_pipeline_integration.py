"""整合測試：依序呼叫 T02→T03→T04→T05→T06 的命令列入口（main），所有外部服務換成假的。

目的：抓出模組之間的銜接錯誤（欄位名稱、檔案路徑、資料型態）。不檢驗任何研究結論。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (REPO / "data" / "interim" / "transit_bus_stops.geojson").exists(),
                                reason="需要先跑 python -m dadaocheng.transit")


def synthetic_world(n=400, seed=1):
    from dadaocheng import geo

    main = geo.main_polygon_metric()
    study = geo.study_polygon_metric()
    rng = np.random.default_rng(seed)
    minx, miny, maxx, maxy = study.bounds
    types = [("cafe", ["cafe", "food"]), ("restaurant", ["restaurant", "food"]), ("book_store", ["book_store", "store"]),
             ("gift_shop", ["gift_shop", "store"]), ("pharmacy", ["pharmacy"]), ("clothing_store", ["clothing_store"]),
             ("primary_school", ["primary_school", "school"])]
    places = []
    i = 0
    while len(places) < n:
        x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
        if not study.contains(Point(x, y)):
            continue
        # 主範圍內密度加倍
        if not main.contains(Point(x, y)) and rng.random() < 0.5:
            continue
        lon, lat = geo.metric_to_lonlat(x, y)
        pt, ty = types[i % len(types)]
        close = 18 if i % 3 else 21
        places.append({
            "id": f"ChIJ{i:06d}", "displayName": {"text": f"測試店{i}" if pt != "pharmacy" else f"測試{i}參藥行"},
            "location": {"latitude": lat, "longitude": lon}, "types": ty, "primaryType": pt,
            "businessStatus": "OPERATIONAL" if i % 17 else "CLOSED_PERMANENTLY",
            "formattedAddress": f"103台灣台北市大同區迪化街一段{i}號",
            "regularOpeningHours": {"periods": [{"open": {"day": d, "hour": 10, "minute": 0},
                                                 "close": {"day": d, "hour": close, "minute": 0}} for d in range(1, 6)]},
            "userRatingCount": int(rng.integers(0, 800)), "rating": 4.3, "priceLevel": "PRICE_LEVEL_MODERATE",
            "websiteUri": (f"https://shop{i}.example.tw/" if i % 4 == 0 else (f"https://www.facebook.com/s{i}" if i % 4 == 1 else None)),
        })
        i += 1
    return places


def fake_places_send(world):
    from dadaocheng import geo

    pts = np.array([[p["location"]["longitude"], p["location"]["latitude"]] for p in world])
    xy = np.array([geo.lonlat_to_metric(lo, la) for lo, la in pts])

    def send(req):
        c = req["locationRestriction"]["circle"]
        cx, cy = geo.lonlat_to_metric(c["center"]["longitude"], c["center"]["latitude"])
        d = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy)
        idx = np.where(d <= c["radius"])[0]
        idx = idx[np.argsort(d[idx])][: req["maxResultCount"]]
        return {"places": [world[k] for k in idx]}

    return send


def synthetic_osm():
    from dadaocheng import geo

    main = geo.main_polygon_metric()
    minx, miny, maxx, maxy = main.bounds
    cx = (minx + maxx) / 2

    def way(i, coords, tags):
        return {"type": "way", "id": i, "tags": tags,
                "geometry": [dict(zip(("lon", "lat"), geo.metric_to_lonlat(x, y))) for x, y in coords]}

    els = [way(1, [(cx, miny - 100), (cx, maxy + 100)], {"highway": "residential", "name": "迪化街一段"})]
    for k, f in enumerate([0.2, 0.4, 0.6, 0.8]):
        y = miny + (maxy - miny) * f
        nm = "民生西路" if k == 1 else f"測試路{k}"
        els.append(way(10 + k, [(minx - 200, y), (maxx + 200, y)], {"highway": "tertiary", "name": nm}))
    lon, lat = geo.metric_to_lonlat(cx + 20, miny + 30)
    els.append({"type": "node", "id": 99, "lon": lon, "lat": lat, "tags": {"name": "OSM布行", "shop": "fabric"}})
    return {"elements": els}


def test_full_pipeline_with_fakes(tmp_root, monkeypatch, capsys):
    from dadaocheng import api_layer, clients, geo, t02_places_census, t03_osm_merge, t04_features, t05_scrape, t06_prompt_dev
    from dadaocheng.extraction import FIELDS

    interim = tmp_root / "data" / "interim"
    interim.mkdir(parents=True)
    for f in (REPO / "data" / "interim").glob("transit_*.geojson"):
        shutil.copy(f, interim / f.name)
    geo.make_buffer_geojson()

    # ---- T02 ----
    world = synthetic_world()
    ledger = api_layer.default_ledger()
    caller = api_layer.ApiCaller("google_places", tmp_root / "data" / "places_cache", ledger, max_requests=900)
    monkeypatch.setattr(clients, "places_caller", lambda root=None: (caller, fake_places_send(world), clients.places_cost_factory(ledger)))
    t02_places_census.main(["--run"])
    raw = pd.read_parquet(interim / "places_raw.parquet")
    assert len(raw) == len({p["id"] for p in world})        # 普查找回全部地點
    assert ledger.requests("google_places") <= 900

    # ---- T03 ----
    osm_dir = tmp_root / "data" / "raw" / "osm"; osm_dir.mkdir(parents=True)
    (osm_dir / "osm_raw.json").write_text(json.dumps(synthetic_osm(), ensure_ascii=False), encoding="utf-8")
    t03_osm_merge.main(["--write-query"])
    assert "out geom" in (osm_dir / "overpass_query.overpassql").read_text(encoding="utf-8")
    t03_osm_merge.main(["--merge"])
    merged = pd.read_parquet(interim / "poi_merged.parquet")
    assert "OSM布行" in set(merged["name"])
    assert not merged["places_primary_type"].eq("primary_school").any()

    # ---- T04 ----
    hdir = tmp_root / "data" / "raw" / "heritage"; hdir.mkdir(parents=True)
    inmain = merged[merged.in_main]
    # 歷史建築與古蹟都要有，否則 T04 的文資驗證會擋下來（提案書 6.5 需要兩類）
    rows = []
    for k in range(12):
        r = inmain.iloc[k]
        rows.append({"heritage_id": f"H{k:02d}", "name": f"測試文資{k}",
                     "type": "古蹟" if k % 4 == 0 else "歷史建築",
                     "address": f"臺北市大同區迪化街1段{k}號", "lon": r.lon, "lat": r.lat})
    pd.DataFrame(rows).to_csv(hdir / "heritage_list.csv", index=False, encoding="utf-8-sig")
    (hdir / "temples.csv").write_text(
        "name,lon,lat,founded_year,source\n"
        f"測試百年廟,{inmain.iloc[20].lon},{inmain.iloc[20].lat},1880,test\n"
        f"測試新廟,{inmain.iloc[21].lon},{inmain.iloc[21].lat},1990,test\n", encoding="utf-8")
    au = inmain.iloc[0]
    t04_features.main([])
    pm = pd.read_parquet(tmp_root / "data" / "final" / "poi_master.parquet")
    assert pm["analysis_unit"].sum() > 50
    assert pm.loc[pm.poi_id == au.poi_id, "in_heritage"].item()
    assert (pm["heritage_kde_300"] > 0).any()
    # 未滿百年的廟不計入密度
    chk = pd.read_csv(tmp_root / "outputs" / "tables" / "t04_heritage_validation.csv")
    assert (chk["level"] == "stop").sum() == 0
    for fig in ["check_heritage.png", "check_blocks.png", "check_main_street.png"]:
        assert (tmp_root / "outputs" / "figures" / fig).exists()

    # ---- T05 ----
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        n = request.url.host.replace("shop", "").split(".")[0]
        body = f"<html><body><article><p>測試店{n}位於大稻埕的老屋，{2000 + int(n) % 20}年創立，販售手作選物。" + "本店重視設計與在地故事。" * 30 + "</p></article></body></html>"
        return httpx.Response(200, html=body)

    fetcher = t05_scrape.Fetcher(client=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))
    fetcher.sleep = lambda s: None
    monkeypatch.setattr(t05_scrape, "Fetcher", lambda client=None: fetcher)
    t05_scrape.main(["--run"])
    web = pd.read_parquet(interim / "web_text.parquet")
    assert set(web["text_status"]) == {"website", "name_only"}
    assert (web.loc[web.url.fillna("").str.contains("facebook"), "text_status"] == "name_only").all()

    # ---- T06 ----
    t06_prompt_dev.main(["--sample"])
    dev = pd.read_csv(tmp_root / "data" / "final" / "dev_sample.csv")
    assert len(dev) == 20 and dev["poi_id"].is_unique

    def fake_claude(req):
        content = req["messages"][-1]["content"]
        name = content.split("\n")[0].replace("店名：", "")
        has_text = "<text>" in content
        out = {f: "不明" for f in FIELDS}
        out.update({"category": "選物與文創零售", "fine_type": "選物或文創商品", "is_chain": "不明",
                    "history_narrative": ["不明"], "founded_year": "未提及"})
        ev = {f: "" for f in FIELDS}
        ev["category"] = ev["fine_type"] = name
        if has_text:
            out.update({"style_design": "是", "history_narrative": ["建築"]})
            ev.update({"style_design": "重視設計", "history_narrative": "大稻埕的老屋"})
        out["evidence"] = ev
        return {"content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False)}], "stop_reason": "end_turn",
                "usage": {"input_tokens": 1800, "output_tokens": 250}}

    acaller = api_layer.ApiCaller("anthropic", tmp_root / "data" / "llm_cache" / "anthropic", ledger)
    monkeypatch.setattr(clients, "anthropic_caller", lambda root=None: (acaller, fake_claude, clients.anthropic_estimate_factory(ledger), clients.anthropic_actual_cost_factory(ledger)))
    t06_prompt_dev.main(["--show-prompt"])
    t06_prompt_dev.main(["--run"])
    out = capsys.readouterr().out
    assert "證據有效率：100.0%" in out
    reviews = list((tmp_root / "outputs" / "tables").glob("t06_dev_review_*.md"))
    assert reviews and "選物與工藝" in reviews[0].read_text(encoding="utf-8")
    assert 0 < ledger.spent("anthropic") < 5

    # ---- T07：抽查樣本（必須排除開發樣本）----
    from dadaocheng import report_rq1, t07_audit_sample

    t07_audit_sample.main(["--draw"])
    sample = pd.read_csv(tmp_root / "data" / "final" / "audit_sample.csv", dtype={"poi_id": str})
    dev_ids = set(pd.read_csv(tmp_root / "data" / "final" / "dev_sample.csv", dtype=str)["poi_id"])
    assert not (set(sample["poi_id"]) & dev_ids)
    assert len(sorted((tmp_root / "data" / "final").glob("audit_form_*.csv"))) == 4
    t07_audit_sample.main(["--check"])
    assert "✔ 檢查通過" in capsys.readouterr().out

    # ---- 只需結構化資料的三條假說 ----
    report_rq1.main(["--run-all"])
    holm = pd.read_csv(tmp_root / "outputs" / "tables" / "rq1_holm.csv")
    assert {"H1", "H3", "H5"} <= set(holm["假說"])
    md = (tmp_root / "outputs" / "tables" / "rq1_summary.md").read_text(encoding="utf-8")
    assert "尚未完成：H6" in md
    for fig in ["h1_close_time_hist.png", "h1_proportions.png", "h3_zoning_map.png", "h5_heritage_hotfood.png"]:
        assert (tmp_root / "outputs" / "figures" / fig).exists()
