#!/usr/bin/env python3
"""用文化部文化資產局（文資局）開放資料，回填 heritage_list.csv 的 address / lon / lat。

用法：
  1. 瀏覽器下載（不要用爬蟲，data.gov.tw 與部分站台禁止自動化存取）：
       https://data.boch.gov.tw/opendata/v2/assetsCase/1.2.json   （歷史建築）
     若之後補古蹟(1.1)、紀念建築(1.4)，同樣下載後可重複執行本腳本。
     存成 data/raw/heritage/boch_1.2.json
  2. python3 tools/fill_heritage_geo.py data/raw/heritage/boch_1.2.json
     （可一次給多個 json）

行為：
  - 只自動填「名稱正規化後完全相同」的紀錄（臺/台、一段/1段、空白與標點視為相同；
    名稱內的「?」視為萬用字元，因為原始 Big5 CSV 缺字）。
  - 只覆寫空白欄位，不會動已有值。
  - 近似比對只列在報告中，請人工確認，不會自動填入。
  - 經緯度若不在臺灣範圍或疑似對調，會留空並列入報告。
"""
import argparse
import csv
import difflib
import json
import re
import sys
from pathlib import Path

CSV_PATH = Path("data/raw/heritage/heritage_list.csv")
REPORT_PATH = Path("data/raw/heritage/heritage_fill_report.csv")
NUM = str.maketrans("一二三四五六七八九", "123456789")


def norm(s: str) -> str:
    s = (s or "").replace("臺", "台").replace("?", "◆").replace("？", "◆")
    s = re.sub(r"([一二三四五六七八九])段", lambda m: m.group(1).translate(NUM) + "段", s)
    return re.sub(r"[^\w◆]+", "", s)


def name_equal(a: str, b: str) -> bool:
    """a 可含 ◆ 萬用字元（單字）。"""
    if a == b:
        return True
    if "◆" in a:
        return re.fullmatch(re.escape(a).replace("◆", "."), b) is not None
    return False


def load_records(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
        raise SystemExit(f"{path}: 找不到記錄陣列，請檢查 JSON 結構")
    return data


def pick_address(rec) -> str:
    a = rec.get("addresses")
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, list) and a:
        a = a[0]
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, dict):
        for k in ("address", "addr", "fullAddress", "location", "name"):
            if isinstance(a.get(k), str) and a[k].strip():
                return a[k].strip()
        vals = [v for v in a.values() if isinstance(v, str) and v.strip()]
        return "".join(vals)
    return ""


def pick_coord(rec):
    def f(k):
        try:
            return float(rec.get(k))
        except (TypeError, ValueError):
            return None
    return f("longitude"), f("latitude")


def is_taipei(rec) -> bool:
    blob = json.dumps(rec.get("addresses", ""), ensure_ascii=False) + str(rec.get("govInstitutionName", ""))
    return "臺北市" in blob or "台北市" in blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+", type=Path)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args()

    recs = []
    for p in args.json_files:
        recs += load_records(p)
    index = {}
    for r in recs:
        index.setdefault(norm(r.get("caseName", "")), []).append(r)
    all_keys = list(index)

    with args.csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = f.name and list(rows[0].keys())

    report, first_shown = [], False
    for row in rows:
        key = norm(row["name"])
        cands = index.get(key, [])
        if not cands and "◆" in key:
            cands = [r for k in all_keys if name_equal(key, k) for r in index[k]]
        cands = [r for r in cands if is_taipei(r)] or cands
        if not cands:
            close = difflib.get_close_matches(key, all_keys, n=2, cutoff=0.8)
            report.append((row["heritage_id"], row["name"], "no_match", "; ".join(close)))
            continue
        rec = cands[0]
        if len(cands) > 1:
            report.append((row["heritage_id"], row["name"], "multiple_match", f"{len(cands)} 筆，取第一筆"))
        if not first_shown:
            print("第一筆比對成功的 addresses 原始內容（請確認 pick_address 取得正確）：")
            print(json.dumps(rec.get("addresses"), ensure_ascii=False)[:300])
            first_shown = True
        if not row["address"]:
            row["address"] = pick_address(rec)
        lon, lat = pick_coord(rec)
        if lon is not None and lat is not None:
            if 21 <= lon <= 26 and 119 <= lat <= 123:  # 疑似對調
                lon, lat = lat, lon
            if 119 <= lon <= 123 and 21 <= lat <= 26:
                row["lon"] = row["lon"] or f"{lon:.6f}"
                row["lat"] = row["lat"] or f"{lat:.6f}"
            else:
                report.append((row["heritage_id"], row["name"], "bad_coord", f"{lon},{lat}"))
        else:
            report.append((row["heritage_id"], row["name"], "no_coord_in_source", ""))
        if not row["address"]:
            report.append((row["heritage_id"], row["name"], "no_address_in_source", ""))

    with args.csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["heritage_id", "name", "status", "detail"])
        w.writerows(report)

    n = len(rows)
    print(f"完成：{n} 筆；有地址 {sum(bool(r['address']) for r in rows)}；"
          f"有座標 {sum(bool(r['lon'] and r['lat']) for r in rows)}；報告 {len(report)} 筆 → {REPORT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
