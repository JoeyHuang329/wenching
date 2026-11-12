"""T11/T12/T13：H1、H5、H3 的端到端檢定。

用「答案已知」的合成 poi_master：資料裡刻意放進 H1 成立、H5 成立、H3 兩個描述成立，
檢定結果必須抓到這些模式；另外用一組「沒有模式」的資料確認不會誤報。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def make_poi_master(n_per_block=24, n_blocks=16, seed=0, *, signal=True):
    """產生合成的 poi_master。signal=False 時各欄位與位置無關（虛無資料）。"""
    rng = np.random.default_rng(seed)
    rows = []
    cats = ["熱食餐飲", "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "文化空間", "一般零售與品牌"]
    for b in range(n_blocks):
        block = f"B{b:02d}"
        north = b >= n_blocks // 2                       # 一半街區在民生西路以北
        for i in range(n_per_block):
            on_main = (i % 3 == 0)                       # 三分之一面向主街
            if signal:
                # H1：85% 的店 19 點前關
                closes = rng.random() < 0.85
                # H3(a) 中藥在主街；(c) 南北貨在北；(b) 文創在巷弄
                if on_main and rng.random() < 0.30:
                    cat, sub = "傳統業種", "中藥"
                elif north and not on_main and rng.random() < 0.25:
                    cat, sub = "傳統業種", "南北貨"
                elif not on_main and rng.random() < 0.30:
                    cat, sub = rng.choice(["選物與文創零售", "文化空間"]), None
                else:
                    cat, sub = rng.choice(cats), None
                # H5：歷建內幾乎沒有熱食
                in_h = rng.random() < 0.18
                if in_h and cat == "熱食餐飲" and rng.random() < 0.9:
                    cat = "咖啡茶飲甜點"
            else:
                closes = rng.random() < 0.45
                cat = rng.choice(cats)
                sub = {"中藥": "中藥", "南北貨": "南北貨"}.get(
                    str(rng.choice(["中藥", "南北貨", None, None, None, None])), None)
                if cat != "傳統業種":
                    sub = None
                in_h = rng.random() < 0.18
            rows.append({
                "poi_id": f"P{b:02d}{i:03d}", "name": f"店{b}_{i}", "analysis_unit": True,
                "block_id": block, "places_category": cat, "traditional_subtype": sub,
                "on_main_street": on_main, "north_of_minsheng": north, "in_heritage": in_h,
                "closes_before_19": closes,
                "close_min_weekday": (17.5 if closes else 21.0) * 60 + rng.normal(0, 30),
                "hours_status": "ok",
                "x": 301000 + (b % 4) * 120 + rng.normal(0, 25),
                "y": 2771500 + (b // 4) * 300 + rng.normal(0, 60),
                "user_rating_count": int(rng.integers(0, 400)),
            })
    df = pd.DataFrame(rows)
    # T04 產出的 closes_before_19 是 object 欄（缺失為 None），這裡保持一致
    df["closes_before_19"] = df["closes_before_19"].astype(object)
    miss = df.sample(frac=0.12, random_state=seed).index
    df.loc[miss, "closes_before_19"] = None
    df.loc[miss, "close_min_weekday"] = np.nan
    df.loc[miss, "hours_status"] = "missing"
    return df


@pytest.fixture()
def poi_master(tmp_root):
    def _write(df):
        d = tmp_root / "data" / "final"
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "poi_master.parquet", index=False)
        (tmp_root / "data" / "interim").mkdir(parents=True, exist_ok=True)
        return df
    return _write


def test_h1_detects_early_closing(tmp_root, poi_master):
    from dadaocheng.t11_h1_hours import run

    au = poi_master(make_poi_master(signal=True))
    res = run(au, n_boot=400)
    tab = res.tables["proportions"].set_index("組別")
    overall = tab.loc["整體"]
    assert overall["比例"] == pytest.approx(0.85, abs=0.06)
    assert overall["街區自助法 下界"] > 0.5
    assert res.p_for_holm < 0.001
    assert "支持" in res.verdict
    # 缺失不進分母
    assert overall["n"] < len(au)
    assert any("沒有營業時間" in n for n in res.notes)
    # 分業態與主街／巷弄都有列出
    assert any(i.startswith("業態：") for i in tab.index)
    assert "區位：主街（迪化街）" in tab.index


def test_h1_null_data_does_not_support(tmp_root, poi_master):
    from dadaocheng.t11_h1_hours import run

    au = poi_master(make_poi_master(seed=7, signal=False))
    res = run(au, n_boot=400)
    assert res.p_for_holm > 0.05
    assert "支持" not in res.verdict


def test_h5_detects_and_reports_power(tmp_root, poi_master):
    from dadaocheng.t12_h5_heritage import run

    au = poi_master(make_poi_master(signal=True))
    res = run(au)
    t = res.tables["contingency"]
    assert t.loc[0, "熱食比例"] < t.loc[1, "熱食比例"]
    assert res.p_for_holm < 0.05 and "支持" in res.verdict
    assert "mantel_haenszel" in res.tables
    assert "category_mix" in res.tables

    small = au[au["in_heritage"] | (au.index % 7 == 0)].copy()
    small.loc[small["in_heritage"], "in_heritage"] = [True] * 10 + [False] * (int(small["in_heritage"].sum()) - 10)
    r2 = run(small)
    if r2.tables["contingency"].loc[0, "合計"] < 20:
        assert "檢定力不足" in r2.verdict


def test_h3_detects_two_of_three(tmp_root, poi_master):
    from dadaocheng.t13_h3_zoning import run

    au = poi_master(make_poi_master(signal=True))
    res = run(au, n_perm=299)
    t = res.tables["claims"]
    assert len(t) == 3
    assert int((t["置換檢定 p"] < 0.05).sum()) >= 2
    assert "支持" in res.verdict
    # 觀測比例要高於全體基準
    for _, r in t.iterrows():
        if r["置換檢定 p"] < 0.05:
            assert r["業種內比例"] > r["全體 POI 在該區位的比例"]
    assert set(res.tables["clq"]["業種"]) == {"中藥", "南北貨", "文創"}


def test_h3_null_data_is_falsified(tmp_root, poi_master):
    from dadaocheng.t13_h3_zoning import run

    au = poi_master(make_poi_master(seed=11, signal=False))
    res = run(au, n_perm=299)
    assert int((res.tables["claims"]["置換檢定 p"] < 0.05).sum()) <= 1
    assert "證偽" in res.verdict


def test_h3_with_fewer_than_two_valid_p_values_is_not_testable(poi_master, monkeypatch):
    from dadaocheng import t13_h3_zoning

    au = poi_master(make_poi_master(seed=11, signal=False))
    p_values = iter([0.01, np.nan, np.nan])

    def fake_permutation_share(*args, **kwargs):
        return {"p": next(p_values), "n_target": 0, "k_in_region": 0,
                "observed": np.nan, "baseline": np.nan, "underpowered": True}

    monkeypatch.setattr(t13_h3_zoning, "permutation_share", fake_permutation_share)
    res = t13_h3_zoning.run(au)

    assert np.isnan(res.p_for_holm)
    assert "無法檢定" in res.verdict


def test_h3_with_one_significant_and_one_nonsignificant_claim_is_inconclusive(poi_master, monkeypatch):
    from dadaocheng import t13_h3_zoning

    au = poi_master(make_poi_master(seed=11, signal=False))
    p_values = iter([0.01, 0.20, np.nan])

    def fake_permutation_share(*args, **kwargs):
        return {"p": next(p_values), "n_target": 0, "k_in_region": 0,
                "observed": np.nan, "baseline": np.nan, "underpowered": True}

    monkeypatch.setattr(t13_h3_zoning, "permutation_share", fake_permutation_share)
    res = t13_h3_zoning.run(au)

    assert res.p_for_holm == pytest.approx(0.20)
    assert "未能支持或證偽" in res.verdict


def test_report_rq1_holm_and_summary(tmp_root, poi_master, capsys):
    from dadaocheng import report_rq1, t11_h1_hours, t12_h5_heritage, t13_h3_zoning

    poi_master(make_poi_master(signal=True))
    t11_h1_hours.main(["--n-boot", "300"])
    t13_h3_zoning.main(["--n-perm", "199"])
    t12_h5_heritage.main([])
    report_rq1.main([])
    d = tmp_root / "outputs" / "tables"
    h = pd.read_csv(d / "rq1_holm.csv").set_index("假說")
    assert {"H1", "H3", "H5"} <= set(h.index)
    # Holm 以 m=4 計算，所以最小的 p 乘 4
    assert h["Holm 乘數"].max() == 4
    md = (d / "rq1_summary.md").read_text(encoding="utf-8")
    assert "尚未完成：H6" in md and "存活者偏誤" in md
    assert "| H6 |" in md
