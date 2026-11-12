"""文資與百年廟宇資料的載入、清理與驗證（T04 的輸入）。

為什麼要單獨一個模組：`heritage_kde_*` 同時進入 H5（分層）與 H6（主自變數），
這兩條是提案書的核心假說。輸入資料少一類（例如古蹟）或多一個錯誤座標，
結果會整個偏掉，而且從最後的迴歸表上看不出來。所以載入時就把品質問題變成明確的警告。

資料來源與分類代碼（文化部文資局國家文化資產網）：
  1.1 古蹟        ← 大稻埕的陳天來故居、新芳春茶行、仁安醫院等都在這一類
  1.2 歷史建築    ← 使用者已下載（boch_1.2.json / heritage_list.csv）
提案書 6.5 節的「歷史環境密度」要求「歷建、古蹟、百年廟宇」三者，所以 1.1 不能缺。

支援三種輸入，可同時存在，以 heritage_id 去重（JSON 優先，因為欄位較完整）：
  data/raw/heritage/heritage_list.csv   人工整理的表（欄位見 待執行任務描述.md T04-H1）
  data/raw/heritage/boch_*.json         直接放文資局 API 的回應，檔名含分類代碼
  data/raw/heritage/temples.csv         百年廟宇（人工整理）
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, geo
from .names import as_text, normalize_address

HERITAGE_COLS = ["heritage_id", "name", "type", "address", "lon", "lat", "classify", "source_file"]


def heritage_dir() -> Path:
    return config.path("raw") / "heritage"


# ---------------------------------------------------------------------------
# 文資局 JSON
# ---------------------------------------------------------------------------

CLASSIFY_NAMES = {"1.1": "古蹟", "1.2": "歷史建築", "1.3": "紀念建築", "1.4": "聚落建築群",
                  "1.5": "考古遺址", "1.6": "史蹟", "1.7": "文化景觀"}


def classify_from_filename(path: Path) -> str:
    m = re.search(r"(\d\.\d)", path.name)
    return m.group(1) if m else ""


def _addresses(rec: dict) -> tuple[str, str, str, str]:
    """回傳 (完整地址字串, 第一個地址（含縣市與行政區）, 縣市, 行政區)。

    文資局 JSON 把縣市、行政區與門牌分開存，而 CSV 的地址是整串。
    這裡一律補成「縣市＋行政區＋門牌」，下游的過濾與地址正規化才會一致。
    """
    a = rec.get("addresses") or []
    if not isinstance(a, list):
        return str(a), str(a), "", ""
    parts, city, dist = [], "", ""
    for item in a:
        if isinstance(item, dict):
            city = city or str(item.get("cityName", "")).strip()
            dist = dist or str(item.get("distName", "")).strip()
            addr = str(item.get("address", "")).strip()
            if addr:
                parts.append(f"{item.get('cityName', '') or city}{item.get('distName', '') or dist}{addr}")
        elif item:
            parts.append(str(item))
    return "；".join(parts), (parts[0] if parts else ""), city, dist


_CITY_DIST = re.compile(r"(?:臺|台)?北市|新北市|基隆市|桃園市|(?:臺|台)中市|(?:臺|台)南市|高雄市|[^\s]{1,3}[縣市]")


def split_city_district(addr: str | None) -> tuple[str, str]:
    """從整串地址切出 (縣市, 行政區)。CSV 的地址用得到。"""
    a = as_text(addr).replace("臺", "台")
    m = re.match(r"\s*\d{0,6}\s*(台北市|新北市|基隆市|桃園市|台中市|台南市|高雄市|[^\s]{2,3}[縣市])\s*([^\s]{1,4}[區鄉鎮市])?", a)
    if not m:
        return "", ""
    return m.group(1) or "", m.group(2) or ""


def load_boch_json(path: Path) -> pd.DataFrame:
    """讀文資局 API 的 JSON（list of 案件）。不在這裡過濾行政區，交給 load_heritage 統一處理。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # 有些匯出會包一層
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    cls = classify_from_filename(path)
    rows = []
    for rec in data:
        if not isinstance(rec, dict) or not rec.get("caseId"):
            continue
        full, first, city, dist = _addresses(rec)
        rows.append({
            "heritage_id": str(rec["caseId"]),
            "name": rec.get("caseName"),
            "type": CLASSIFY_NAMES.get(cls, cls or "未標示"),
            "address": first,
            "address_all": full,
            "city": city,
            "district": dist,
            "lon": rec.get("longitude"),
            "lat": rec.get("latitude"),
            "classify": cls,
            "source_file": path.name,
        })
    df = pd.DataFrame(rows)
    for c in ("lon", "lat"):
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


# ---------------------------------------------------------------------------
# 文資清單（CSV ＋ JSON 合併）
# ---------------------------------------------------------------------------

def load_heritage(districts: list[str] | None = None, cities: list[str] | None = None,
                  verbose: bool = True) -> pd.DataFrame:
    """合併 heritage_dir 下的 heritage_list.csv 與所有 boch_*.json。

    放全國或全縣市的 JSON 也可以：會先依 cities 再依 districts 過濾。
    行政區名稱在不同縣市會重複（基隆市中正區、台中市中區…），所以兩層都要過。
    """
    s = config.settings()["heritage"]
    districts = districts if districts is not None else s.get("districts")
    cities = cities if cities is not None else s.get("cities")
    d = heritage_dir()
    frames = []
    csv_path = d / "heritage_list.csv"
    if csv_path.exists():
        c = pd.read_csv(csv_path, dtype={"heritage_id": str}, encoding="utf-8-sig")
        for col in ("lon", "lat"):
            c[col] = pd.to_numeric(c.get(col), errors="coerce")
        for col in ("classify", "address_all", "city", "district"):
            if col not in c:
                c[col] = None
        c["source_file"] = csv_path.name
        # CSV 沒寫 classify 時，依 type 反推
        rev = {v: k for k, v in CLASSIFY_NAMES.items()}
        c["classify"] = c["classify"].fillna(c["type"].map(rev))
        frames.append(c)
    for jp in sorted(d.glob("boch_*.json")):
        frames.append(load_boch_json(jp))
    if not frames:
        raise FileNotFoundError(
            f"{d} 下找不到 heritage_list.csv 或 boch_*.json。請依 待執行任務描述.md T04-H1 下載文資資料。"
        )
    h = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("city", "district"):
        if col not in h:
            h[col] = None
    # 缺縣市或行政區欄位的（CSV 來源）從地址字串補
    need = h["city"].isna() | (h["city"].astype(str).str.strip() == "")
    if need.any():
        cd = h.loc[need, "address"].apply(split_city_district)
        h.loc[need, "city"] = [a for a, _ in cd]
        h.loc[need, "district"] = [b for _, b in cd]
    h["city"] = h["city"].fillna("").str.replace("臺", "台", regex=False)
    h["district"] = h["district"].fillna("")

    n_all = len(h)
    if cities:
        keep_city = {c.replace("臺", "台") for c in cities}
        h = h[h["city"].isin(keep_city) | (h["city"] == "")]
    if districts:
        # 行政區名稱在不同縣市會重複（例如基隆市與台北市都有中正區），所以要先過縣市再過行政區
        h = h[h["district"].isin(districts) | (h["district"] == "")]
    n_filtered = n_all - len(h)

    # 距離過濾：行政區範圍比研究區大得多（中正區含紀州庵、萬華區含寶藏巖），
    # 這些點離大稻埕數公里，在 300 公尺帶寬的高斯核下貢獻實質為 0。
    # 留著只會讓「文資清單 N 筆」這個數字失去意義，所以依距離濾掉並回報。
    max_d = s.get("max_distance_m")
    n_far = 0
    if max_d:
        from shapely.geometry import Point

        poly = geo.study_polygon_metric().buffer(float(max_d))
        def _near(lon, lat):
            if pd.isna(lon) or pd.isna(lat):
                return True          # 沒有座標的留著，改走地址比對
            x, y = geo.lonlat_to_metric(float(lon), float(lat))
            return bool(poly.contains(Point(x, y)))
        near = [_near(a, b) for a, b in zip(h["lon"], h["lat"])]
        n_far = len(h) - int(sum(near))
        h = h[near]

    # 去重：同一 heritage_id 保留有座標的那一筆；都有座標時以 JSON 為準（欄位較完整）
    h["_has_xy"] = h["lon"].notna() & h["lat"].notna()
    h["_from_json"] = h["source_file"].str.endswith(".json")
    h = (h.sort_values(["heritage_id", "_has_xy", "_from_json"], ascending=[True, False, False])
          .drop_duplicates("heritage_id", keep="first").drop(columns=["_has_xy", "_from_json"]))
    h["addr_key"] = h["address"].apply(normalize_address)
    h = h.reset_index(drop=True)
    if verbose:
        types = "、".join(f"{k} {v}" for k, v in h["type"].value_counts().items())
        src = "、".join(f"{k} {v}" for k, v in h["source_file"].value_counts().items())
        print(f"文資清單：{len(h)} 筆（{types}）｜來源：{src}"
              + (f"｜依縣市與行政區濾掉 {n_filtered} 筆" if n_filtered else "")
              + (f"｜距研究範圍超過 {max_d:.0f} 公尺濾掉 {n_far} 筆" if n_far else ""))
    return h


# ---------------------------------------------------------------------------
# 百年廟宇
# ---------------------------------------------------------------------------

TEMPLE_COLS = ["name", "lon", "lat", "founded_year"]


def load_temples(verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀並清理廟宇檔，回傳 (百年廟宇, 全部含排除原因)。

    清理：
      - 丟掉誤貼成資料列的表頭（第一欄等於 "name"）。
      - 經緯度轉數值；缺值或超出研究範圍的標為 bad_coord。
      - 創立年份：只留 <= settings.heritage.temple_founded_before（預設 1926，即 2026 年起算滿百年）。
        年份空白者不計入密度，但保留在報表中，因為「查不到年份」不等於「不是百年廟」。
    """
    s = config.settings()["heritage"]
    path = heritage_dir() / s["temples_csv"].split("/")[-1]
    if not path.exists():
        if verbose:
            print(f"（沒有 {path.name}，歷史環境密度只用文資點位）")
        return pd.DataFrame(columns=TEMPLE_COLS), pd.DataFrame()
    t = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    t.columns = [c.strip() for c in t.columns]
    t = t[t["name"].notna() & (t["name"].str.strip().str.lower() != "name")].copy()
    for c in ("lon", "lat"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
    t["founded_year"] = pd.to_numeric(t["founded_year"], errors="coerce")
    cutoff = int(s.get("temple_founded_before", 1926))

    # 座標合理性：必須落在研究範圍外擴一段距離內（避免同名廟宇抓到其他縣市）
    tol = float(s.get("temple_coord_tolerance_m", 2000))
    poly = geo.study_polygon_metric().buffer(tol)
    def inside(lon, lat):
        if pd.isna(lon) or pd.isna(lat):
            return False
        from shapely.geometry import Point
        x, y = geo.lonlat_to_metric(float(lon), float(lat))
        return bool(poly.contains(Point(x, y)))
    t["coord_ok"] = [inside(a, b) for a, b in zip(t["lon"], t["lat"])]
    t["is_centenarian"] = t["founded_year"].notna() & (t["founded_year"] <= cutoff)

    def reason(r):
        if not r["coord_ok"]:
            return "座標缺失或不在研究範圍內" if pd.notna(r["lon"]) else "沒有座標"
        if pd.isna(r["founded_year"]):
            return "創立年份未知"
        if r["founded_year"] > cutoff:
            return f"創立於 {int(r['founded_year'])} 年，未滿百年"
        return ""
    t["exclude_reason"] = t.apply(reason, axis=1)
    keep = t[t["coord_ok"] & t["is_centenarian"]].copy()
    if verbose:
        print(f"廟宇檔 {len(t)} 筆 → 納入歷史密度的百年廟宇 {len(keep)} 筆"
              f"（{cutoff} 年以前創立且座標在範圍內）")
        for _, r in t[t["exclude_reason"] != ""].iterrows():
            print(f"  排除 {r['name']}：{r['exclude_reason']}")
    return keep[TEMPLE_COLS].reset_index(drop=True), t


# ---------------------------------------------------------------------------
# 驗證報表
# ---------------------------------------------------------------------------

def validate(heritage: pd.DataFrame, temples_all: pd.DataFrame) -> pd.DataFrame:
    """回傳警告表。level=stop 代表應該停下來問人（CLAUDE.md），warn 代表寫進研究限制。"""
    s = config.settings()["heritage"]
    rows: list[dict] = []

    def add(level, item, detail):
        rows.append({"level": level, "item": item, "detail": detail})

    want = set(s.get("required_classify", ["1.1", "1.2"]))
    have = set(heritage["classify"].dropna().astype(str))
    for c in sorted(want - have):
        add("stop", f"缺少分類 {c}（{CLASSIFY_NAMES.get(c, '')}）",
            f"提案書 6.5 節的歷史環境密度需要「歷建、古蹟、百年廟宇」。"
            f"目前清單沒有任何 {CLASSIFY_NAMES.get(c, c)}，heritage_kde 與 in_heritage 會系統性偏低（影響 H5、H6）。"
            f"請依 待執行任務描述.md T04-H1 下載 classifyCode={c} 後放進 data/raw/heritage/boch_{c}.json。")

    n_xy = int((heritage["lon"].notna() & heritage["lat"].notna()).sum())
    rate = n_xy / len(heritage) if len(heritage) else 0.0
    add("info" if rate >= 0.8 else "stop", "文資座標覆蓋率",
        f"{n_xy}/{len(heritage)} = {rate:.1%}" + ("" if rate >= 0.8 else "（低於八成，CLAUDE.md 要求停下來人工補比對）"))
    no_xy = heritage[heritage["lon"].isna() | heritage["lat"].isna()]
    for _, r in no_xy.iterrows():
        add("warn", f"沒有座標：{r['name']}", f"{r.get('address') or ''}（改以地址比對；比對不到就不會進入密度計算）")

    # 座標是否落在研究範圍附近
    tol = float(s.get("heritage_coord_tolerance_m", 3000))
    poly = geo.study_polygon_metric().buffer(tol)
    from shapely.geometry import Point
    far = []
    for _, r in heritage.dropna(subset=["lon", "lat"]).iterrows():
        x, y = geo.lonlat_to_metric(float(r["lon"]), float(r["lat"]))
        if not poly.contains(Point(x, y)):
            far.append(r["name"])
    if far:
        add("info", f"{len(far)} 筆文資座標離研究範圍 {tol:.0f} 公尺以上",
            "；".join(far[:6]) + ("…" if len(far) > 6 else "")
            + "。這些是同一行政區但不在大稻埕的文資（例如中正區的紀州庵、萬華區的寶藏巖）。"
            "在 300 公尺帶寬的高斯核下貢獻實質為 0，不影響結果；"
            "要讓清單只留相關範圍，可調小 settings.heritage.max_distance_m。")

    if len(temples_all):
        bad = temples_all[~temples_all["coord_ok"]]
        for _, r in bad.iterrows():
            add("stop" if pd.notna(r["lon"]) else "warn", f"廟宇座標有問題：{r['name']}",
                f"lon={r['lon']}, lat={r['lat']} — {r['exclude_reason']}。"
                "若是別的縣市同名廟宇，請改成正確座標或刪掉這一列。")
        unknown = temples_all[temples_all["founded_year"].isna() & temples_all["coord_ok"]]
        if len(unknown):
            add("warn", f"{len(unknown)} 間廟宇年份未知，未計入百年廟宇",
                "；".join(unknown["name"].tolist()) + "。查到年份後會改變 heritage_kde，請在 deviations.md 記錄。")
    return pd.DataFrame(rows, columns=["level", "item", "detail"])


def print_validation(v: pd.DataFrame) -> int:
    """印出警告，回傳 stop 等級的筆數。"""
    icons = {"stop": "■ 必須處理", "warn": "▲ 請注意", "info": "· 資訊"}
    for lv in ("stop", "warn", "info"):
        for _, r in v[v["level"] == lv].iterrows():
            print(f"{icons[lv]}　{r['item']}")
            if r["detail"]:
                print(f"            {r['detail']}")
    return int((v["level"] == "stop").sum())


def main() -> None:
    h = load_heritage()
    temples, temples_all = load_temples()
    v = validate(h, temples_all)
    print()
    n_stop = print_validation(v)
    out = config.path("tables")
    out.mkdir(parents=True, exist_ok=True)
    v.to_csv(out / "t04_heritage_validation.csv", index=False, encoding="utf-8-sig")
    h.to_csv(config.path("interim") / "heritage_merged.csv", index=False, encoding="utf-8-sig")
    temples_all.to_csv(config.path("interim") / "temples_checked.csv", index=False, encoding="utf-8-sig")
    print(f"\n報表：{out / 't04_heritage_validation.csv'}")
    if n_stop:
        print(f"有 {n_stop} 項需要處理才建議往下跑 T04。")


if __name__ == "__main__":
    main()
