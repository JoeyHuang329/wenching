# 大稻埕文青選址研究｜資料流程程式庫

**題目：** 訪談驅動的假說生成與大型語言模型資料建構：大稻埕文青產業選址之文化特徵檢驗與缺失因素分析
**團隊：** 黃旭寬、隊友 C、隊友 D、隊友 B、隊友 A｜**指導：** 江穎慧老師（地政學系）
**硬性日期：** 11/11 22:00 結果凍結 → 11/12 簡報 → 11/15 海報

這個程式庫把提案書第八節的研究方法做成可重跑、可稽核的程式。目前完成到**實作規劃 B 的 T00–T06**（骨架、預先登錄草稿、普查、OSM 去重、結構化特徵、官網抓取、抽取 prompt 開發），並已整合使用者提供的捷運與公車開放資料。

> **第一次打開請先讀：**
> 1. `接下來的實作計畫.md`：依 11/12 現況重排的時程與取捨（最新）。
> 2. `T04_審閱_資料品質與範圍邊界.md`：文資缺古蹟、主範圍東緣畫錯的影響與修法。
> 3. `待執行任務描述.md`：每一步的指令與人工任務。
> 4. `實作審閱筆記.md`：打 prereg tag 之前必須決定的事。
> 5. `CLAUDE.md`：給 Claude Code 的規則與禁區。

---

## 目錄

1. 現況一覽
2. 快速開始（Windows）
3. 資料流程
4. 資料夾結構
5. 設定檔
6. 規則如何在程式中落實
7. 交通資料說明
8. 測試
9. 關鍵 tag

---

## 1. 現況一覽

| 任務卡 | 程式 | 測試 | 實際執行 | 卡在哪裡 |
|---|---|---|---|---|
| T00 骨架與預算守門員 | ✔ | ✔ | ✔ | 需要人建立金鑰與服務端上限（T00-H2） |
| T01 預先登錄 | 草稿 ✔ | — | — | 9 個【待確認】要人決定，再打 tag |
| 交通資料 | ✔ | ✔ | ✔ 已在本機跑過 | 主範圍改正式版後重跑一次 |
| T02 Places 普查 | ✔ | ✔（模擬 API） | ✘ | 金鑰、Console 確認計費 SKU |
| T03 OSM 去重 | ✔ | ✔ | ✘ | 下載 OSM、人工判斷模稜兩可清單 |
| T04 結構化特徵 | ✔ | ✔ | ✘ | 文資清單、百年廟宇清單、地圖檢查 |
| T05 官網抓取 | ✔ | ✔（模擬網站） | ✘ | 網路；在背景跑 |
| T06 prompt 開發 | ✔ | ✔（模擬 Claude） | ✘ | 金鑰、兩人審閱、打 tag |
| T07 抽查樣本 | ✔ | ✔ | ✘ | 等 T05；表單要四位組員填 |
| T08 批次抽取 | ✔ | ✔（模擬批次） | ✘ | 等 T06 打 tag |
| T09 Gemini 交叉檢查 | ✔ | ✔ | ✘ | 等 T08 |
| T10 κ 閘門報表 | ✔ | ✔ | ✘ | 等人工抽查 |
| T11 H1 夜間經濟 | ✔ | ✔ | ✘ | **只需 T04，閘門一之前就能跑** |
| T12 H5 歷建與熱食 | ✔ | ✔ | ✘ | **只需 T04** |
| T13 H3 業種分區 | ✔ | ✔ | ✘ | **只需 T04** |
| T14–T18 | 尚未撰寫 | | | H6、消融、RQ3 |

整合測試 `tests/test_pipeline_integration.py` 會用模擬的外部服務，依序跑 T02→T06 的命令列入口，確認各步驟之間的檔案與欄位銜接正確。

---

## 2. 快速開始（Windows）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
pip install -e ".[dev]"
copy .env.example .env      # 填入三把金鑰（不要貼進任何對話）
python -m pytest
python scripts\check_rules.py
```

macOS／Linux 把 `.\.venv\Scripts\Activate.ps1` 換成 `source .venv/bin/activate`。

---

## 3. 資料流程

```
（人）主範圍邊界 ──► make_buffer_geojson ──► boundary_buffer.geojson
                                   │
交通原始檔 ──► transit ────────────┼──► data/interim/transit_*.geojson ─┐
                                   ▼                                    │
T02 Places 普查 ──► places_raw.parquet                                  │
                         │                                              │
T03 OSM 下載＋去重 ◄─────┘ ──► poi_merged.parquet ＋ dedup_ambiguous.csv（人判斷）
                         │                                              │
T04 特徵 ◄───────────────┴── 文資清單、百年廟宇（人整理）◄───────────────┘
   └──► data/final/poi_master.parquet ＋ 三張檢查地圖（人檢查）
                         │
T05 官網抓取 ◄───────────┘ ──► web_text.parquet
                         │
T06 prompt 開發 ◄────────┘ ──► dev_sample.csv ──► 審閱表（人看）──► tag prompt-frozen
```

各步驟的指令：

| 步驟 | 指令 |
|---|---|
| 文資驗證 | `python -m dadaocheng.heritage` |
| 緩衝區 | `python -c "from dadaocheng import geo; geo.make_buffer_geojson()"` |
| 交通 | `python -m dadaocheng.transit` |
| T02 | `python -m dadaocheng.t02_places_census --dry-run`，接著 `--single-cell`，最後 `--run` |
| T03 | `python -m dadaocheng.t03_osm_merge --download`（或 `--write-query` 後手動下載），接著 `--merge` |
| T04 | `python -m dadaocheng.t04_features`（文資未就緒時加 `--no-heritage`） |
| T05 | `python -m dadaocheng.t05_scrape --run`，接著 `--report` |
| T06 | `python -m dadaocheng.t06_prompt_dev --sample`，接著 `--show-prompt`，最後 `--run` |
| T07 | `python -m dadaocheng.t07_audit_sample --draw`，接著 `--check` |
| T08 | `--estimate` → `--submit --approve-step` → `--poll`（反覆）→ `--fill-sync` → `--build` |
| T09 | `python -m dadaocheng.t09_gemini_crosscheck --run` |
| T10 | `python -m dadaocheng.t10_kappa_gate` |
| H1/H3/H5 | `python -m dadaocheng.report_rq1 --run-all`（只需 T04） |

所有花錢的步驟都可以安全重跑：相同請求會讀快取，不會再付費。

---

## 4. 資料夾結構

```
dadaocheng/
├── CLAUDE.md                 給 Claude Code 的規則（每次新對話會讀）
├── README.md                 本檔
├── 待執行任務描述.md          需要人做的步驟（依任務卡編號）
├── 實作審閱筆記.md            方法風險與建議
├── pyproject.toml            套件與相依
├── .env.example              金鑰範本（.env 不進 git）
├── config/
│   ├── settings.yaml         種子、半徑、門檻、模型、欄位遮罩、類別對照
│   ├── budget.yaml           各 API 上限、單價、累計花費（程式寫入）
│   ├── boundary_main.geojson 主範圍（目前為 DRAFT）
│   └── boundary_buffer.geojson 由主範圍外擴 300 公尺產生
├── docs/
│   ├── proposal.md           提案書定稿
│   ├── implementation_A.md   實作規劃 A（時程分工）
│   ├── implementation_B.md   實作規劃 B（技術規格與任務卡）
│   ├── decision_review.md    決策審閱意見
│   ├── preregistration.md    預先登錄草稿（prereg tag 後唯讀）
│   ├── deviations.md         偏離紀錄
│   ├── audit_manual.md       抽查手冊草稿
│   ├── evidence_cards.md     證據卡範本（隊友 B）
│   └── places_terms.md       條款整理範本（隊友 D）
├── prompts/extraction_v1.md  抽取 prompt（改版請另存 v2、v3）
├── src/dadaocheng/
│   ├── config.py             設定、路徑、種子、.env
│   ├── api_layer.py          T00 呼叫層：快取、記帳、上限、停下來問人
│   ├── clients.py            Places／Claude／Gemini 的請求與花費函式
│   ├── geo.py                邊界、座標轉換
│   ├── names.py              店名與地址正規化、相似度
│   ├── transit.py            捷運、公車資料與交通特徵
│   ├── t02_places_census.py
│   ├── t03_osm_merge.py
│   ├── t04_features.py
│   ├── t05_scrape.py
│   ├── extraction.py         schema、驗證、證據檢查、文青推導、多數決
│   ├── t06_prompt_dev.py
│   ├── heritage.py           文資與百年廟宇的載入、清理、驗證
│   ├── stats.py              信賴區間、Holm、置換檢定、CLQ、Fisher、MH、κ
│   ├── t07_audit_sample.py
│   ├── t08_batch_extract.py
│   ├── t09_gemini_crosscheck.py
│   ├── t10_kappa_gate.py
│   ├── t11_h1_hours.py、t12_h5_heritage.py、t13_h3_zoning.py
│   ├── report_rq1.py         Holm 校正與簡報摘要
│   └── plotting.py           中文字型
├── scripts/check_rules.py    CLAUDE.md 規則自動檢查
├── tests/                    pytest
├── data/                     整個不進 git
│   ├── raw/transit/          已放入使用者提供的交通開放資料
│   ├── raw/osm/、raw/heritage/
│   ├── places_cache/、web_cache/、llm_cache/
│   ├── interim/、final/
└── outputs/figures/、tables/、rq3_cases/
```

---

## 5. 設定檔

- **`config/settings.yaml`：** 實作規劃 B 第 4 節的所有初始值不變，另外加上各任務需要的參數（Places 欄位遮罩、非商業類別、去重門檻、主街名單、交通半徑、社群網域、Places 類別對照、傳統業種關鍵字）。改動會影響預先登錄的值時，要寫進 `docs/deviations.md`。
- **`config/budget.yaml`：** 上限分別是 Claude 1,000 元、Google 1,500 元、Gemini 0 元，匯率 32。單價於 2026-11-08 依官方定價頁查證：Haiku 5.5 每百萬 token 輸入 0.10、輸出 0.50 美元，批次五折；Sonnet 5.5 為 2／10 美元。`spent_twd` 等欄位由程式寫入，不要手動改。

---

## 6. 規則如何在程式中落實

| CLAUDE.md 規則 | 落實方式 | 自動檢查 |
|---|---|---|
| 1 訪談不進程式庫、不進 LLM | prompt 只組 Places 名稱、類別與官網正文；`.gitignore` 排除整稿檔名 | `check_rules.py` |
| 2 預先登錄 tag 後唯讀 | 程式不寫此檔 | `check_rules.py`（比對 tag） |
| 3 抽查與開發樣本分開 | T06 不讀抽查檔；開發樣本檔不覆寫 | 測試 |
| 4 付費 API 走同一呼叫層 | `api_layer.ApiCaller`：快取、記帳、上限、七成與 200 元門檻 | `check_rules.py`＋測試 |
| 5 Places 欄位遮罩 | `clients.places_field_mask()` 拒絕禁止欄位 | 測試＋`check_rules.py` |
| 6 社群不抓 | `t05_scrape.classify_url`；轉址到社群也算 | 測試 |
| 7 證據必須是子字串 | `extraction.evidence_is_substring`；無效欄位視為「不明」 | 測試 |
| 8 Gemini 只送公開文本 | Gemini 只接 `user_content()`（只有店名、類別、官網正文） | — |
| 9 金鑰只在 .env | `config.require_env`；User-Agent 無 email | `check_rules.py` |
| 10 種子 | `config.rng()`、`settings.seed` | `check_rules.py` |
| 11 文青由程式推導 | `extraction.derive_wenqing_group`；schema 沒有 wenqing 欄位 | 測試 |

「必須停下來問人」的情況中，可以寫成程式的都已寫成例外或警告：單步 > 200 元、累計達七成、Places 請求超過 900、`name_only` 超過六成、文資比對率低於八成、邊界仍是草稿。

---

## 7. 交通資料說明

| 檔案（`data/raw/transit/`） | 內容 | 處理 |
|---|---|---|
| `mrt_stations.csv` | 臺北捷運車站（2026-09-04），WGS84 | 去掉多餘引號與大括號；轉乘站與同站不同線合併成一列 |
| `mrt_routes_twd97.json` | 捷運路網（2026-10-13），EPSG:3826 | 只用於畫圖 |
| `busstop.shp/.shx/.dbf` | 公車站牌 3,407 站，WGS84，dbf 為 UTF-8 | 站名被截在 22 bytes，以替代字元讀入並標記 `name_truncated` |
| `bus_shapes_GetBusShape.json` | 公車路線軌跡（2025-12），WKT | 主要使用 |
| `bus_shapes_TstBusShape.json` | 同上的較舊版本 | 保留，不使用 |

產生的特徵：

- `dist_mrt_m`、`nearest_mrt`：直線距離，量到車站中心點。
- `bus_stop_count_300`、`bus_density`：300 公尺內站牌數，以及每平方公里站數。
- `bus_route_count_150`：補充特徵，不進預先登錄的模型。

以草稿邊界試跑：主範圍附近有 321 個站牌、14 個捷運站。最近的是大橋頭站（距邊界約 174 公尺），其次是北門站（約 400 公尺）。檢查圖在 `outputs/figures/check_transit.png`。

---

## 8. 測試

```powershell
python -m pytest                          # 全部
python -m pytest tests/test_t00_api_layer.py -v
```

| 檔案 | 驗證什麼 |
|---|---|
| `test_t00_api_layer.py` | 快取不重複扣款、上限、七成門檻、單步 200 元、請求數上限、失敗不扣款、`.env` 與 `data/` 被忽略 |
| `test_t02_census.py` | 格子幾何、四分樹找回所有點、最小格飽和回報、主範圍優先、欄位遮罩、重跑讀快取 |
| `test_t03_merge.py` | 名稱與地址正規化、相似度門檻、非商業過濾、模稜兩可清單、人工判斷套用、`poi_id` 穩定 |
| `test_t04_features.py` | 跨午夜、午休、24 小時、缺失、街區合併 ≥ 15、鄰域排除自己、核密度、端到端 |
| `test_transit.py` | 用真實交通檔案：解析、合併轉乘站、站名截斷、大橋頭附近的特徵 |
| `test_t05_scrape.py` | 社群排除、robots、限速、關於頁、截斷、續跑、六成提醒 |
| `test_t06_extraction.py` | schema、快取前綴固定、證據子字串、`name_only` 強制、文青推導、多數決、不讀抽查答案、模擬呼叫 |
| `test_heritage.py` | 縣市與行政區過濾、CSV 與 JSON 合併、缺古蹟會擋、廟宇清理與百年篩選、三條比對路徑 |
| `test_stats.py` | Wilson 與群集自助法、Holm（m=4）、置換檢定、CLQ、Fisher、MH、κ |
| `test_t07_t10_audit.py` | 分層抽樣與機率、表單不含 LLM 答案、三次快取鍵不同、批次送出取回補跑、κ 報表 |
| `test_t11_t13_hypotheses.py` | H1/H3/H5 在「有訊號」與「無訊號」資料上的行為 |
| `test_pipeline_integration.py` | T02→T07＋H1/H3/H5 全流程（模擬外部服務） |

---

## 9. 關鍵 tag

`prereg` → `census-done` → `prompt-frozen` → `gate1` → `results-frozen`。只 tag 程式碼與文件，資料不進 git。
