"""T14：H6 歷史密度 × 業態。用合成資料驗證三件事：
有異質性時要支持、沒有異質性且樣本夠時要證偽、MDE 不依賴業態斜率的估計值。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

CATS = ["熱食餐飲", "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "一般零售與品牌", "文化空間", "其他"]


def make_master(n_blocks=17, per=80, slopes=None, noise=1.0, seed=0):
    slopes = slopes or {c: 0.0 for c in CATS}
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(n_blocks):
        bmean = rng.normal(0, 25)            # 街區之間的歷史密度差異（會被固定效果吸收）
        fe = rng.normal(0, 0.6)              # 街區層級的關注度差異
        for i in range(per):
            cat = CATS[rng.integers(0, len(CATS))]
            kde = 120 + bmean + rng.normal(0, 12)
            rows.append({"poi_id": f"P{b:02d}{i:03d}", "analysis_unit": True, "block_id": f"B{b:02d}",
                         "places_category": cat, "heritage_kde_300": kde, "heritage_kde_150": kde * 2,
                         "on_main_street": bool(rng.random() < 0.25),
                         "dist_mrt_m": 500 + rng.normal(0, 100),
                         "price_level_ord": rng.choice([np.nan, 1, 2, 3], p=[0.8, 0.1, 0.05, 0.05]),
                         "_kde": kde, "_fe": fe})
    df = pd.DataFrame(rows)
    z = (df["_kde"] - df["_kde"].mean()) / df["_kde"].std(ddof=0)
    s = df["places_category"].map(slopes).fillna(0.0)
    ly = 3.0 + df["_fe"] + s * z + rng.normal(0, noise, len(df))
    df["user_rating_count"] = np.round(np.expm1(np.clip(ly, 0, None))).astype(int)
    df.loc[df.sample(frac=0.05, random_state=seed).index, "user_rating_count"] = np.nan
    return df.drop(columns=["_kde", "_fe"])


@pytest.fixture()
def write_master(tmp_root):
    def _w(df):
        d = tmp_root / "data" / "final"; d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / "poi_master.parquet", index=False)
        return df
    return _w


def test_h6_supports_when_slopes_differ(tmp_root, write_master):
    from dadaocheng import t14_h6_density as t14

    au = write_master(make_master(slopes={"選物與文創零售": 0.6, "一般零售與品牌": 0.5}, noise=0.8))
    res, specs = t14.run(au, n_boot=199, slope_p=False)
    assert res.p_for_holm < 0.05
    assert res.verdict.startswith("**支持**")
    sl = res.tables["slopes_main"].set_index("業態")
    assert sl.loc["選物與文創零售", "斜率"] > 0.3
    assert abs(sl.loc["熱食餐飲", "斜率"]) < 0.25
    # 排除規則：其他不進樣本、文化空間不在估計樣本
    assert "其他" not in set(sl.index) and "文化空間" not in set(sl.index)
    assert res.tables["specs"]["規格"].str.startswith("M1").any()


def test_h6_falsified_with_null_and_precise_data(tmp_root, write_master):
    from dadaocheng import t14_h6_density as t14

    au = write_master(make_master(n_blocks=19, per=400, noise=0.15, seed=4))
    res, _ = t14.run(au, n_boot=199, slope_p=False)
    assert res.p_for_holm > 0.05
    assert res.tables["interactions_main"]["區間落在 ±0.10 內"].all()
    assert res.verdict.startswith("**證偽**")


def test_h6_noisy_null_is_not_falsified_and_mde_flags_it(tmp_root, write_master):
    from dadaocheng import t14_h6_density as t14

    au = write_master(make_master(n_blocks=16, per=60, noise=1.2, seed=7))
    res, _ = t14.run(au, n_boot=199, slope_p=False)
    m = res.tables["mde"]
    assert (m["MDE（log 單位）"] > 0.10).any()
    if res.p_for_holm >= 0.05:
        assert "無法證偽" in res.verdict or "區間太寬" in res.verdict


def test_mde_does_not_depend_on_true_heterogeneity(tmp_root, write_master):
    """MDE 用虛無模型的殘差：真實斜率改變時，MDE 只會因殘差變大而小幅變動，不會反映斜率本身。"""
    from dadaocheng import t14_h6_density as t14

    a = make_master(seed=2, noise=1.0)
    b = make_master(seed=2, noise=1.0, slopes={"選物與文創零售": 0.05})
    ma = t14.mde(t14.prepare(a)[0].query("cat in @t14.MODEL_CATS").reset_index(drop=True))
    mb = t14.mde(t14.prepare(b)[0].query("cat in @t14.MODEL_CATS").reset_index(drop=True))
    assert np.allclose(ma["MDE（log 單位）"], mb["MDE（log 單位）"], rtol=0.15)


def test_h6_cli_writes_outputs_and_holm_picks_it_up(tmp_root, write_master, capsys):
    from dadaocheng import report_rq1, t14_h6_density as t14

    write_master(make_master(slopes={"選物與文創零售": 0.6}, noise=0.8))
    t14.main(["--mde-only"])
    d = tmp_root / "outputs" / "tables"
    assert (d / "h6_mde.csv").exists()
    meta = json.loads((d / "h6_mde.json").read_text(encoding="utf-8"))
    assert meta["G"] == 17
    assert not (d / "h6_pvalue.csv").exists()          # MDE 階段不產生 p 值

    t14.main(["--n-boot", "199"])
    for f in ["h6_specs.csv", "h6_slopes_main.csv", "h6_interactions_main.csv", "h6_loco.csv",
              "h6_identification.csv", "h6_culture_describe.csv", "h6_pvalue.csv"]:
        assert (d / f).exists(), f
    loco = pd.read_csv(d / "h6_loco.csv")
    assert len(loco) == 18 and loco.iloc[0]["剔除的街區"] == "（全部保留）"
    assert (tmp_root / "outputs" / "figures" / "h6_forest.png").exists()

    # 敏感性：有 tag 時不覆蓋 h6_pvalue.csv
    before = (d / "h6_pvalue.csv").read_text(encoding="utf-8")
    t14.main(["--n-boot", "99", "--kde-col", "heritage_kde_150", "--tag", "bw150"])
    assert (d / "h6_bw150_slopes_main.csv").exists()
    assert (d / "h6_pvalue.csv").read_text(encoding="utf-8") == before

    # report_rq1 讀得到 H6
    for h in ["h1", "h3", "h5"]:
        pd.DataFrame([{"hypothesis": h.upper(), "p_raw": 0.5, "verdict": "x"}]).to_csv(d / f"{h}_pvalue.csv", index=False)
    report_rq1.main([])
    md = (d / "rq1_summary.md").read_text(encoding="utf-8")
    assert "尚未完成" not in md and "假說四" in md


def test_h6_uses_llm_chain_control_when_available(tmp_root, write_master):
    from dadaocheng import t14_h6_density as t14

    au = write_master(make_master())
    ex = pd.DataFrame({"poi_id": au["poi_id"], "is_chain": np.where(np.arange(len(au)) % 7 == 0, "是", "否"),
                       "category": au["places_category"]})
    res, specs = t14.run(au, n_boot=99, extractions=ex, slope_p=False)
    assert "chain:是" in specs["M1"]["names"]
    assert any("是否連鎖來自 LLM" in n for n in res.notes)
    res2, specs2 = t14.run(au, n_boot=99, slope_p=False)
    assert "chain:是" not in specs2["M1"]["names"]
    assert any("不含是否連鎖" in n for n in res2.notes)
