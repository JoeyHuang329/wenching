"""T07：抽出 120 筆人工抽查樣本，產生四份抽查表單。

規格（實作規劃 B T07 ＋ 提案書 8.4）：
- 只從 in_main 且營業中的 POI 抽，排除 T06 的開發樣本。
- 先以 Places 類別粗估「可能文青」比例：低於 15% 就分兩層，「可能文青」層抽樣率加倍，
  並記錄每筆的 sampling_prob（T10 的加權準確率與 PPI 都要用）。
- 四份表單各 30 筆，**不顯示 LLM 答案**（CLAUDE.md 規則 3）。
- 種子固定；各層筆數與抽樣機率寫入紀錄檔。

這支程式**不讀取**任何 LLM 抽取結果或人工答案，所以在 T08 送出前後跑都一樣。

用法：
  python -m dadaocheng.t07_audit_sample --draw          # 抽樣並產生表單（CSV＋Excel）
  python -m dadaocheng.t07_audit_sample --xlsx          # 不重抽，只從 audit_sample.csv 重新產生 Excel 表單
  python -m dadaocheng.t07_audit_sample --example       # 產生填寫範例 docs/audit_form_example.xlsx
  python -m dadaocheng.t07_audit_sample --check         # 檢查已抽的樣本（不重抽）

抽查員請填 Excel 版（audit_form_*.xlsx）：有下拉選單、文字會換行。
T10 會優先讀 Excel 版；Excel 沒填任何答案時才退回讀 CSV 版。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import config

AUDITORS = ["隊友 A", "隊友 B", "隊友 D", "隊友 C"]
FORM_COLS = ["poi_id", "auditor", "name", "places_types", "text_status", "maps_url", "text",
             "h_category", "h_is_wenqing", "h_history_narrative", "basis",
             "llm_fields_reasonable", "evidence_reasonable", "note"]


def maps_url(row) -> str:
    if row.get("place_id"):
        return f"https://www.google.com/maps/place/?q=place_id:{row['place_id']}"
    return f"https://www.google.com/maps/search/?api=1&query={row['lat']},{row['lon']}"


def load_frame() -> pd.DataFrame:
    poi = pd.read_parquet(config.path("final") / "poi_master.parquet")
    au = poi[poi["analysis_unit"]].copy()
    wp = config.path("interim") / "web_text.parquet"
    if wp.exists():
        web = pd.read_parquet(wp)[["poi_id", "text_status", "text"]]
        au = au.merge(web, on="poi_id", how="left")
    else:
        au["text_status"], au["text"] = "name_only", ""
        print("▲ 還沒有 web_text.parquet（T05 未完成）：表單的文本欄會是空的。"
              "建議 T05 跑完再抽，抽查員才看得到和 LLM 一樣的輸入。")
    au["text_status"] = au["text_status"].fillna("name_only")
    au["text"] = au["text"].fillna("")
    return au


def draw(au: pd.DataFrame, n: int, dev_ids: set[str], min_wenqing_share: float = 0.15,
         seed_offset: int = 7, alloc: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分層隨機抽樣。回傳 (樣本, 分層紀錄)。

    sampling_prob 是「該筆被抽中的機率」＝該層抽出數 ÷ 該層母體數。
    T10 的加權準確率與 PPI 都要依 1/sampling_prob 加權，才會回到母體。

    alloc 指定各層的固定抽出筆數（例如 {"可能文青": 50, "其他": 70}），會強制分層，
    不管 likely_wenqing 的比例有沒有低於 min_wenqing_share。用在比例高於門檻、
    但「真文青」在樣本中仍然太少、is_wenqing 的 κ 會太不精確的時候。
    """
    pool = au[~au["poi_id"].isin(dev_ids)].copy()
    share = float(pool["likely_wenqing"].mean())
    stratified = alloc is not None or share < min_wenqing_share
    rng = config.rng(seed_offset)

    if alloc is not None:
        pool["stratum"] = np.where(pool["likely_wenqing"], "可能文青", "其他")
        sizes = pool["stratum"].value_counts().to_dict()
        unknown = set(alloc) - {"可能文青", "其他"}
        if unknown:
            raise SystemExit(f"--allocate 只接受「可能文青」與「其他」，收到 {sorted(unknown)}")
        take = {st: min(int(k), int(sizes.get(st, 0))) for st, k in alloc.items()}
        for st, k in alloc.items():
            if int(k) > int(sizes.get(st, 0)):
                print(f"▲「{st}」層母體只有 {sizes.get(st, 0)} 筆，少於指定的 {k} 筆，已改抽全部。")
        if sum(take.values()) != n:
            print(f"▲ 指定的配額合計 {sum(take.values())} 筆，與 --n {n} 不同；以配額為準。")
    elif not stratified:
        pool["stratum"] = "全體"
        take = {"全體": min(n, len(pool))}
    else:
        pool["stratum"] = np.where(pool["likely_wenqing"], "可能文青", "其他")
        n_w = int(pool["stratum"].eq("可能文青").sum())
        n_o = len(pool) - n_w
        # 「可能文青」層抽樣率加倍：解 n_w_take/n_w = 2 × n_o_take/n_o 且兩者合計 n
        w_take = min(n_w, int(round(n * (2 * n_w) / (2 * n_w + n_o))))
        o_take = min(n_o, n - w_take)
        w_take = min(n_w, n - o_take)
        take = {"可能文青": w_take, "其他": o_take}

    parts, log = [], []
    for st, g in pool.groupby("stratum"):
        k = take.get(st, 0)
        sel = g.sample(k, random_state=int(rng.integers(0, 2**31 - 1))) if k else g.iloc[:0]
        prob = k / len(g) if len(g) else np.nan
        sel = sel.copy()
        sel["sampling_prob"] = prob
        parts.append(sel)
        log.append({"stratum": st, "母體筆數": len(g), "抽出筆數": k, "sampling_prob": prob})
    sample = pd.concat(parts).sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    sample["auditor"] = [AUDITORS[i % len(AUDITORS)] for i in range(len(sample))]
    sample["maps_url"] = sample.apply(maps_url, axis=1)
    return sample, pd.DataFrame(log)


def write_forms(sample: pd.DataFrame) -> list:
    d = config.path("final")
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, a in enumerate(AUDITORS):
        g = sample[sample["auditor"] == a].copy()
        for c in ["h_category", "h_is_wenqing", "h_history_narrative", "basis",
                  "llm_fields_reasonable", "evidence_reasonable", "note"]:
            g[c] = ""
        g["basis"] = "text"
        p = d / f"audit_form_{i + 1}_{a}.csv"
        g[FORM_COLS].to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Excel 表單：下拉選單、換行、凍結窗格。抽查員填這個，不要填 CSV。
# ---------------------------------------------------------------------------

CATEGORY_CHOICES = ["熱食餐飲", "咖啡茶飲甜點", "傳統業種", "選物與文創零售", "文化空間",
                    "一般零售與品牌", "其他", "不明"]
WENQING_CHOICES = ["是", "否", "不明"]
HISTORY_CHOICES = ["無", "不明", "建築", "街區", "家族", "建築;街區", "建築;家族", "街區;家族", "建築;街區;家族"]
BASIS_CHOICES = ["text", "text+maps"]
GENERIC = {"point_of_interest", "establishment"}
TEXT_LIMIT = 30000          # Excel 單格上限 32,767 字

XLSX_COLS = [
    # (欄位, 標題, 寬度, 是否要填)
    ("poi_id", "poi_id\n（不要改）", 10, False),
    ("name", "店名", 26, False),
    ("places_types", "Google 類別", 22, False),
    ("text_status", "文本狀態", 10, False),
    ("maps_url", "Google Maps", 12, False),
    ("text", "店家文本（LLM 看到的就是這些）", 60, False),
    ("h_category", "① 業態大類", 16, True),
    ("h_is_wenqing", "② 是否文青", 11, True),
    ("h_history_narrative", "③ 歷史敘事", 15, True),
    ("basis", "④ 判斷依據", 11, True),
    ("note", "備註（為什麼這樣判斷、哪裡不確定）", 34, True),
    ("llm_fields_reasonable", "第二輪：LLM 其他欄位合理嗎\n（現在先空著）", 16, False),
    ("evidence_reasonable", "第二輪：LLM 證據合理嗎\n（現在先空著）", 16, False),
]


def pretty_types(types_json) -> str:
    try:
        ts = json.loads(types_json) if isinstance(types_json, str) and types_json else []
    except ValueError:
        return str(types_json or "")
    return "、".join(x for x in ts if x not in GENERIC)


def _write_xlsx(rows: pd.DataFrame, path, title: str) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()
    ws = wb.active
    ws.title = "填寫"
    fill_in = PatternFill("solid", fgColor="FFF2CC")       # 要填的欄：淡黃
    fill_later = PatternFill("solid", fgColor="EDEDED")    # 第二輪：灰
    head_font = Font(bold=True)
    wrap_top = Alignment(wrap_text=True, vertical="top")

    for j, (col, head, width, to_fill) in enumerate(XLSX_COLS, 1):
        c = ws.cell(row=1, column=j, value=head)
        c.font, c.alignment = head_font, Alignment(wrap_text=True, vertical="center")
        if to_fill:
            c.fill = fill_in
        elif col in ("llm_fields_reasonable", "evidence_reasonable"):
            c.fill = fill_later
        ws.column_dimensions[c.column_letter].width = width
    ws.row_dimensions[1].height = 45

    for i, (_, r) in enumerate(rows.iterrows(), 2):
        for j, (col, _, _, to_fill) in enumerate(XLSX_COLS, 1):
            v = r.get(col, "")
            v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else v
            if col == "places_types":
                v = pretty_types(v)
            if col == "text":
                v = str(v)
                if not v.strip():
                    v = "（沒有文本：只有店名與 Google 類別）"
                elif len(v) > TEXT_LIMIT:
                    v = v[:TEXT_LIMIT] + "……（以下省略）"
            c = ws.cell(row=i, column=j, value=str(v) if col != "maps_url" else "開啟地圖")
            c.alignment = wrap_top
            if col == "maps_url" and v:
                c.hyperlink = str(v)
                c.font = Font(color="0563C1", underline="single")
            if to_fill:
                c.fill = fill_in
        ws.row_dimensions[i].height = 90

    n = len(rows) + 1
    col_of = {col: ws.cell(row=1, column=j).column_letter for j, (col, *_ ) in enumerate(XLSX_COLS, 1)}
    for col, choices in [("h_category", CATEGORY_CHOICES), ("h_is_wenqing", WENQING_CHOICES),
                         ("h_history_narrative", HISTORY_CHOICES), ("basis", BASIS_CHOICES)]:
        dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True,
                            showErrorMessage=True, errorTitle="請用下拉選單",
                            error="只能填下拉選單裡的值。")
        ws.add_data_validation(dv)
        dv.add(f"{col_of[col]}2:{col_of[col]}{max(n, 2)}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{col_of['evidence_reasonable']}{max(n, 1)}"

    guide = wb.create_sheet("怎麼填")
    lines = [
        title, "",
        "只填淡黃色的四欄 ①②③④ 和備註。灰色兩欄是第二輪（LLM 結果出來之後）才填，現在先空著。",
        "每一列：先讀「店家文本」→ 需要時點「開啟地圖」→ 用下拉選單填 ①②③④。",
        "",
        "① 業態大類：選最主要的一個。不是店家（寺廟、工會、公司辦公室、事務所）→「其他」。看不出來 →「不明」。",
        "② 是否文青：照規則，不要憑感覺。",
        "    連鎖 → 否",
        "    獨立咖啡店、可以坐下來喝茶的茶館 → 是（不需要其他條件）",
        "    獨立書店、藝廊、展演、選物、文創、手作工藝、設計工作室，而且看得到一項風格指標 → 是",
        "       風格指標：設計或美學主張／在地或歷史故事／手作或小量生產／閱讀或展覽活動",
        "    甜點店：要有風格指標才是；路邊點心、雞蛋糕、傳統糕餅 → 否",
        "    手搖飲、中藥行、南北貨、布行、一般零售、餐廳 → 否（包裝得再好也一樣）",
        "    資料不足以判斷 → 不明",
        "③ 歷史敘事：只看「文本裡有沒有講」，不是看店老不老。",
        "    講到老屋、街屋、歷史建築 → 建築；講大稻埕的歷史 → 街區；講傳承、第幾代、創立於某年 → 家族",
        "    可以複選（下拉選單有組合）。文本夠多但沒講歷史 → 無；沒有文本或太少 → 不明",
        "④ 判斷依據：只看表單 → text；有點開地圖看照片或評論 → text+maps",
        "",
        "常見的坑",
        "  · Google 類別常常是錯的：雞蛋糕攤被標成 cafe、珠寶店只有 point_of_interest。以店名和文本為準。",
        "  · 文本常常是網站選單、購物車、公司登記資料。那不是「沒有文本」，是「有文本但沒講歷史」→ ③ 填 無。",
        "  · 店名有「老店」「百年」不算歷史敘事，文本要真的講到才算。",
        "  · 不要因為認得這家店就填資料裡沒有的資訊；如果你點了地圖，④ 要填 text+maps。",
        "  · 看到個人電話、姓名不要抄進備註，寫「含個資」就好。",
        "  · 填過的答案不要回頭改。改變想法就寫在備註。不要和其他人討論答案，直到四個人都填完。",
    ]
    for i, s in enumerate(lines, 1):
        guide.cell(row=i, column=1, value=s).alignment = Alignment(wrap_text=True)
    guide.column_dimensions["A"].width = 120
    wb.save(path)


def write_forms_xlsx(sample: pd.DataFrame) -> list:
    d = config.path("final")
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, a in enumerate(AUDITORS):
        g = sample[sample["auditor"] == a].copy()
        for c in ["h_category", "h_is_wenqing", "h_history_narrative", "llm_fields_reasonable",
                  "evidence_reasonable", "note"]:
            g[c] = ""
        g["basis"] = "text"
        p = d / f"audit_form_{i + 1}_{a}.xlsx"
        _write_xlsx(g, p, f"抽查表單：{a}（{len(g)} 筆）")
        paths.append(p)
    return paths


EXAMPLE_ROWS = [
    # 全部是虛構店家，不在抽查樣本裡。抽查員不可以拿範例的答案去套真實樣本。
    dict(name="【範例】稻光咖啡", places_types='["cafe","coffee_shop"]', text_status="name_only", text="",
         h_category="咖啡茶飲甜點", h_is_wenqing="是", h_history_narrative="不明", basis="text",
         note="店名與類別都是咖啡店，看不出是連鎖 → 獨立咖啡店直接算文青。沒有文本，所以歷史敘事是「不明」。"),
    dict(name="【範例】某某連鎖咖啡 迪化店", places_types='["cafe"]', text_status="name_only", text="",
         h_category="咖啡茶飲甜點", h_is_wenqing="否", h_history_narrative="不明", basis="text",
         note="店名有分店名（迪化店），是全台連鎖品牌 → 連鎖一律否。"),
    dict(name="【範例】永和蔘藥行", places_types='["health","store"]', text_status="name_only", text="",
         h_category="傳統業種", h_is_wenqing="否", h_history_narrative="不明", basis="text",
         note="中藥行屬傳統業種，不論包裝如何都不是文青。沒有文本 → 歷史不明。"),
    dict(name="【範例】大成貿易有限公司", places_types='["store"]', text_status="website",
         text="大成貿易有限公司的董監事資料\n營業項目：一般進出口貿易業務……",
         h_category="其他", h_is_wenqing="否", h_history_narrative="無", basis="text",
         note="公司登記資料，不是店面 → 其他。有文本但沒提到歷史 → 無（不是不明）。"),
    dict(name="【範例】稻埕小物所", places_types='["gift_shop","store"]', text_status="website",
         text="我們在百年街屋裡販售台灣職人手作的陶器與布品，每件都是小量生產……",
         h_category="選物與文創零售", h_is_wenqing="是", h_history_narrative="建築", basis="text",
         note="選物店＋風格指標（手作、小量生產）→ 是。提到百年街屋 → 建築。"),
    dict(name="【範例】三代布莊", places_types='["home_goods_store","store"]', text_status="website",
         text="創立於1950年，目前由第三代經營，專營棉麻布料批發零售。",
         h_category="傳統業種", h_is_wenqing="否", h_history_narrative="家族", basis="text",
         note="布行是傳統業種 → 否。講到創立年份與第三代 → 家族。"),
    dict(name="【範例】花園旅店咖啡廳", places_types='["restaurant","cafe"]', text_status="website",
         text="旅店座落於迪化街，保留傳統三進二中庭的街屋建築結構……附設咖啡廳提供下午茶。",
         h_category="咖啡茶飲甜點", h_is_wenqing="是", h_history_narrative="建築;街區", basis="text",
         note="獨立的咖啡廳 → 是。講到街屋建築與迪化街 → 建築＋街區（下拉選單選組合）。"),
    dict(name="【範例】小魚雞蛋糕", places_types='["cafe","food"]', text_status="name_only", text="",
         h_category="咖啡茶飲甜點", h_is_wenqing="否", h_history_narrative="不明", basis="text",
         note="Google 標成 cafe，但其實是雞蛋糕攤。甜點類要有風格指標才算文青，這裡沒有 → 否。"),
    dict(name="【範例】大稻埕某某宮", places_types='["place_of_worship"]', text_status="name_only", text="",
         h_category="其他", h_is_wenqing="否", h_history_narrative="不明", basis="text",
         note="寺廟不是店家 → 其他，並在備註寫「非商業」。"),
    dict(name="【範例】豐盛", places_types='["point_of_interest"]', text_status="name_only", text="",
         h_category="一般零售與品牌", h_is_wenqing="否", h_history_narrative="不明", basis="text+maps",
         note="只看店名與類別完全看不出來；點開地圖看照片是珠寶店 → 一般零售。因為看了地圖，④ 填 text+maps。"
              "如果沒看地圖，三欄都填「不明」也是正確的。"),
]


def write_example(path=None):
    path = path or (config.project_root() / "docs" / "audit_form_example.xlsx")
    rows = pd.DataFrame(EXAMPLE_ROWS)
    rows["poi_id"] = [f"EX{i:02d}" for i in range(1, len(rows) + 1)]
    rows["maps_url"] = ""
    rows["llm_fields_reasonable"] = ""
    rows["evidence_reasonable"] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_xlsx(rows, path, "填寫範例（全部是虛構店家，只看格式與判斷方式，不要拿去套真實樣本）")
    return path


def check(sample: pd.DataFrame, au: pd.DataFrame, dev_ids: set[str]) -> list[str]:
    msgs = []
    if sample["poi_id"].duplicated().any():
        msgs.append("■ 樣本有重複的 poi_id")
    if set(sample["poi_id"]) & dev_ids:
        msgs.append("■ 抽查樣本與開發樣本重疊（違反 CLAUDE.md 規則 3）")
    if not set(sample["poi_id"]) <= set(au["poi_id"]):
        msgs.append("■ 樣本中有不在分析單位內的 POI")
    counts = sample["auditor"].value_counts()
    # 樣本數不一定能被抽查員人數整除，差 1 筆是可接受的
    if counts.min() < 1 or (counts.max() - counts.min()) > 1:
        msgs.append(f"▲ 每人筆數不平均：{counts.to_dict()}")
    if "sampling_prob" not in sample or sample["sampling_prob"].isna().any():
        msgs.append("■ 有筆數缺 sampling_prob，T10 的加權與 PPI 會錯")
    for c in ["h_category", "h_is_wenqing", "h_history_narrative"]:
        if c in sample.columns and sample[c].astype(str).str.strip().ne("").any():
            msgs.append(f"▲ {c} 已經有填答；重抽會蓋掉人工答案")
    return msgs or ["✔ 檢查通過"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--draw", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--xlsx", action="store_true", help="不重抽，從 audit_sample.csv 重新產生 Excel 表單")
    g.add_argument("--example", action="store_true", help="產生填寫範例 docs/audit_form_example.xlsx")
    ap.add_argument("--force", action="store_true", help="已經抽過仍重抽（會蓋掉表單，請先備份已填的答案）")
    ap.add_argument("--allocate", default=None, metavar="可能文青=50,其他=70",
                    help="強制分層並指定各層抽出筆數。likely_wenqing 比例高於 15% 門檻、"
                         "但真文青在樣本中仍太少時用（見 裁決列表 N-03）。")
    args = ap.parse_args(argv)
    alloc = None
    if args.allocate:
        try:
            alloc = {k.strip(): int(v) for k, v in (kv.split("=") for kv in args.allocate.split(","))}
        except ValueError:
            raise SystemExit("--allocate 格式應為「可能文青=50,其他=70」")

    s = config.settings()
    out = config.path("final") / "audit_sample.csv"
    if args.example:
        print("範例：", write_example())
        return
    if args.xlsx:
        if not out.exists():
            raise SystemExit(f"{out} 不存在，請先 --draw")
        sample = pd.read_csv(out, dtype={"poi_id": str})
        existing = [p for p in config.path("final").glob("audit_form_*.xlsx")]
        if existing and not args.force:
            raise SystemExit("Excel 表單已存在，重新產生會蓋掉已填的答案。確定要重產請加 --force。")
        for p in write_forms_xlsx(sample):
            print("表單：", p)
        return
    au = load_frame()
    dev_path = config.project_root() / s["extraction"]["dev_sample_file"]
    dev_ids = set(pd.read_csv(dev_path, dtype=str)["poi_id"]) if dev_path.exists() else set()
    if not dev_ids:
        print("▲ 還沒有開發樣本（T06 --sample）：抽查樣本無法排除它，兩者可能重疊。")

    if args.check:
        if not out.exists():
            raise SystemExit(f"{out} 不存在，請先 --draw")
        sample = pd.read_csv(out, dtype={"poi_id": str})
        for m in check(sample, au, dev_ids):
            print(m)
        return

    if out.exists() and not args.force:
        print(f"{out} 已存在，不重抽（避免蓋掉已填的人工答案）。確定要重抽請加 --force。")
        return

    sample, log = draw(au, int(s["audit_n"]), dev_ids, alloc=alloc)
    sample.to_csv(out, index=False, encoding="utf-8-sig")
    log_path = config.path("interim") / "audit_sampling_log.csv"
    log.to_csv(log_path, index=False, encoding="utf-8-sig")
    (config.path("interim") / "audit_sampling_meta.json").write_text(
        json.dumps({"seed": int(s["seed"]), "seed_offset": 7, "audit_n": int(s["audit_n"]),
                    "dev_excluded": len(dev_ids), "pool": int(len(au) - len(dev_ids & set(au["poi_id"]))),
                    "likely_wenqing_share": float(au["likely_wenqing"].mean()),
                    "stratified": bool(log["stratum"].nunique() > 1),
                    "forced_allocation": alloc}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"母體（分析單位扣掉開發樣本）：{len(au) - len(dev_ids & set(au['poi_id']))} 家")
    print(f"Places 類別粗估的「可能文青」比例：{au['likely_wenqing'].mean():.1%}"
          f"（低於 15% 自動分層；本次{'有' if log['stratum'].nunique() > 1 else '沒有'}分層"
          f"{'，用 --allocate 指定配額' if alloc else ''}）")
    print(log.to_string(index=False))
    print(f"\n抽出 {len(sample)} 筆 → {out}")
    print(f"分層紀錄 → {log_path}")
    print("文本狀態分布：", sample["text_status"].value_counts().to_dict())
    for p in write_forms(sample):
        print("表單（CSV，備份用）：", p)
    for p in write_forms_xlsx(sample):
        print("表單（Excel，請抽查員填這個）：", p)
    print("填寫範例：", write_example())
    print("\n提醒：表單上沒有 LLM 答案，請抽查員先填三個關鍵欄位，再看 LLM 的結果（docs/audit_manual.md）。")


if __name__ == "__main__":
    main()
