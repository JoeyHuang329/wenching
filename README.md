# 從訪談到開放資料：大稻埕歷史環境與業態分布的假說檢驗及大型語言模型測量評估

**團隊：** 文青不迷路

**原提案題目：** 訪談驅動的假說生成與大型語言模型資料建構：大稻埕文青產業選址之文化特徵檢驗與缺失因素分析

這個程式庫把研究方法做成可重跑、可稽核的程式：從 Google Places 普查、OSM 去重、文化資產比對，到 LLM 抽取、人工抽查、四條假說的檢定與 Holm 校正。四條核心假說（假說一到四）都已經有結果；RQ2 的消融實驗與 RQ3 的案例分析尚未撰寫。

> **第一次打開請先讀：**
> 1. `CLAUDE.md`：不可違反的規則（訪談不進程式庫與 LLM、預先登錄唯讀、付費 API 只走呼叫層等）。
> 2. `論文稿_上_緒論設計與方法.md`、`論文稿_下_結果與討論.md`：研究的完整書面版。
> 3. `假說四_H6_結果報告.md`：主問題的結果、執行順序與要記入的偏離。
> 4. `執行計畫_1112_1115.md`、`新裁決列表_1112.md`：步驟卡 S01–S14 與裁決 N-01–N-10。
> 5. `撰寫備忘_低信心主張_待補資料_後續規劃.md`：尚待驗證的主張與剩餘工作。

---

## 目錄

1. 研究結果摘要
2. 現況一覽
3. 快速開始（Windows）
4. 資料流程與指令
5. 重要的資料處理決定
6. 資料夾結構
7. 設定檔
8. 規則如何在程式中落實
9. 交通資料說明
10. 測試
11. 已知問題與待辦
12. 關鍵 tag
13. 編號對照

---

## 1. 研究結果摘要

分析單位：主範圍內營業中的商業 POI 2,022 家，分布在 16 個街區。顯著水準 α = 0.05，四條核心假說以 Holm 校正（m = 4）。

| 假說 | 程式代號 | 主張 | 原始 p | Holm 校正後 p | 結論 |
|---|---|---|---|---|---|
| 假說一 | H1 | 夜間經濟弱，多數店家傍晚就打烊 | < 0.001 | < 0.001 | **支持**：58.0% 平日 19:00 前打烊（街區自助法 95% 區間 52.9%–62.1%）。缺營業時間 28.7%，最壞情況界限涵蓋 50% |
| 假說二 | H3 | 主街中藥、巷弄文創、民生西路以北南北貨 | 0.001 | 0.003 | **支持**：中藥與南北貨分區成立；「文創集中在巷弄」不成立（定義太寬） |
| 假說三 | H5 | 歷史建築限制熱食與明火 | 0.989 | 1.000 | **證偽**：歷建內熱食 11.5%、歷建外 8.1%。探索性分層：主街方向符合假說，巷弄相反 |
| 假說四 | H6 | 歷史環境與店家關注度的關聯因業態而異（主問題） | 0.650 | 1.000 | **無法證偽，也未獲支持**：事前最小可檢測效果 0.34–0.81 log 單位，大於證偽門檻 0.10 |

**LLM 測量品質（閘門一）**：119 筆人工抽查。三個關鍵欄位的 Cohen's κ 分別為業態大類 0.48、是否文青 0.31、歷史敘事 −0.02，都未達 0.6，因此全組決定後續路徑之前，假說一到四的業態一律使用 Places 類別。分析單位中只有 14.9% 有可讀的網站文本，這是測量品質的主要限制。

**花費**：Google Places 新台幣 129.92 元（1,116 次請求），Claude 新台幣 120.83 元，Gemini 0 元。

---

## 2. 現況一覽

| 任務 | 程式 | 實際執行 | 備註 |
|---|---|---|---|
| T00 骨架與預算守門員 | ✔ | ✔ | |
| T01 預先登錄 | 草稿 ✔ | — | 【待確認】需人工定稿並打 `prereg` tag |
| 邊界建構 `build_boundary.py` | ✔ | ✔ | `status: FINAL_FROM_OSM` |
| 交通資料 | ✔ | ✔ | |
| T02 Places 普查 | ✔ | ✔ | 969 格、4,337 個地點 |
| T03 OSM 去重 | ✔ | ✔ | 自動合併 290 對、人工判斷 206 對 |
| T04 結構化特徵 | ✔ | ✔ | 2,022 家、16 個街區 |
| T05 官網抓取 | ✔ | ✔ | 有文本 14.6%（全部 POI）／14.9%（分析單位） |
| T06 prompt 開發 | ✔ | ? | 是否有 `prompt-frozen` tag 待確認 |
| T07 抽查樣本 | ✔ | ✔ | 分層 50／70；Excel 表單已回收 119 筆 |
| T08 批次抽取 | ✔ | ✔ | 截斷重跑後失敗率 2.8%（仍有 361 次被截斷） |
| T09 Gemini 交叉檢查 | ✔ | O | 模型名稱無效，見第 11 節 |
| T10 κ 閘門報表 | ✔ | ✔ | 三欄皆未達 0.6 |
| T11 假說一（H1） | ✔ | ✔ | |
| T12 假說三（H5） | ✔ | ✔ | |
| T13 假說二（H3） | ✔ | ✔ | |
| T14 假說四（H6） | ✔ | ✔ | 由 Claude 代跑（本機被 Windows 擋住，見第 11 節） |
| T15 消融（RQ2） | ✘ | ✘ | 尚未撰寫 |
| T16–T17 RQ3 | ✘ | ✘ | 範圍縮為附錄 B 的五個已知案例 |

---

## 3. 快速開始（Windows）

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install -e ".[dev]"
copy .env.example .env      # 填入金鑰（不要貼進任何對話）
python -m pytest            # 應為 173 passed
python scripts\check_rules.py
```

- 一律用 `python -m pip`，不要直接執行 `pip.exe`。
- **如果出現「應用程式控制原則已封鎖此檔案」**：代表 Windows 的「智慧型應用程式控制」擋住了 pandas、pyarrow 的 `.pyd` 檔。處理方式見第 11 節。
- 診斷腳本的輸出請用 `--out` 寫檔，不要用 PowerShell 的 `>`，否則中文會變亂碼。

macOS／Linux 把 `.\.venv\Scripts\Activate.ps1` 換成 `source .venv/bin/activate`。

---

## 4. 資料流程與指令

```
OSM 道路 ──► build_boundary ──► boundary_main.geojson ──► make_buffer_geojson ──► boundary_buffer.geojson
                                                                     │
交通原始檔 ──► transit ──────────────────────────────────────────────┤
                                                                     ▼
T02 Places 普查 ──► T03 OSM 去重（人工判斷模稜兩可）──► T04 特徵 ◄── 文資局古蹟／歷建、百年廟宇
                                                          │
                                                          ▼
                                             data/final/poi_master.parquet
                       ┌──────────────────────────────────┼──────────────────────────────┐
                       ▼                                  ▼                              ▼
              T05 官網抓取 ──► T08 LLM 抽取       T07 抽查樣本 ──► 人工標註        H1／H3／H5（只需 T04）
                                   │                       │                             │
                                   └──► T10 κ 閘門 ◄───────┘                             │
                                   └──► T09 Gemini 交叉                                  │
                                                                                         ▼
                                                     T14 H6（事前 MDE → 估計）──► report_rq1（Holm）
```

| 步驟 | 指令 |
|---|---|
| 主範圍邊界 | `python scripts\build_boundary.py --preview` → 確認後 `--write` |
| 緩衝區 | `python -c "import sys; sys.path.insert(0,'src'); from dadaocheng import geo; print(geo.make_buffer_geojson())"` |
| 文資驗證 | `python -m dadaocheng.heritage` |
| 交通 | `python -m dadaocheng.transit` |
| T02 | `python -m dadaocheng.t02_places_census --cache-only --extent-m 150`（只讀快取）；補查用 `--run --max-requests N --approve` |
| T03 | `python -m dadaocheng.t03_osm_merge --merge` |
| T04 | `python -m dadaocheng.t04_features`（文資驗證有警告時，確認後加 `--accept-warnings`） |
| T05 | `python -m dadaocheng.t05_scrape --run`，接著 `--report` |
| T07 | `python -m dadaocheng.t07_audit_sample --draw --allocate 可能文青=50,其他=70`，接著 `--check`；`--xlsx` 只重產 Excel 表單；`--example` 產生填寫範例 |
| T08 | `--estimate` → `--submit --approve-step` → `--poll`（反覆）→ `--fill-sync` → `--build`；批次狀態用 `--status`；截斷重跑用 `--retry-truncated --sync` |
| T08 診斷 | `python scripts\diagnose_t08.py --out outputs\tables\diagnose_t08.txt` |
| T09 | `python -m dadaocheng.t09_gemini_crosscheck --run` |
| T10 | `python -m dadaocheng.t10_kappa_gate` |
| H1／H3／H5 | `python -m dadaocheng.report_rq1 --run-all` |
| **H6** | **先** `python -m dadaocheng.t14_h6_density --mde-only`（記進 `deviations.md`），**再** `python -m dadaocheng.t14_h6_density` |
| H6 敏感性 | `--kde-col heritage_kde_150 --tag bw150`；`--exclude-missing-reviews --tag noreviewzero`；`--category-col category --tag llmcat`（須全組同意） |
| Holm 彙總 | `python -m dadaocheng.report_rq1`（`--run-all --with-h6` 會一次跑完四條） |
| 全面診斷 | `python scripts\diagnose.py --out outputs\tables\diagnose_after_boundary.txt` |

所有花錢的步驟都可以安全重跑：相同的請求會讀快取，不會再付費。

---

## 5. 重要的資料處理決定

以下每一項都要在 `docs/deviations.md` 有對應紀錄。

| 決定 | 內容 | 依據 |
|---|---|---|
| 邊界由程式建構 | 從 OSM 取四條界線主路的中心線，排除巷弄，兩兩求交點後沿道路接成多邊形；手繪草稿漏掉了延平北路沿街 | `scripts/build_boundary.py` |
| 主範圍外擴 10 公尺 | 邊界沿道路中心線，店家座標是建物形心；外擴才能納入沿街店家。敏感性分析用 0 與 15 公尺 | `settings.main_inclusive_buffer_m`、`geo.main_polygon_metric` |
| 網格原點固定 | 重畫邊界不會讓 Places 快取失效 | `data/interim/places_grid_origin.json` |
| 抽查強制分層 | 「可能文青」比例 17.3%，高於 15% 的自動分層門檻，所以改用固定配額 50／70，並以抽樣機率倒數加權 | `t07 --allocate` |
| 抽查母體 | 抽樣時用的是草稿邊界的分析單位；120 筆都位在最終邊界內 | 撰寫備忘 L-01 |
| LLM 截斷重跑 | 只重跑被 `max_tokens` 截斷的呼叫，用較大的上限；其餘結果不動 | `t08 --retry-truncated` |
| 業態用 Places 類別 | 閘門一未通過，假說一到四的業態都不依賴 LLM | 提案書 8.8 |
| 缺評論數記為 0 | 缺 `user_rating_count` 的店全都有 place_id、全都沒有評分，而且沒有任何一家評論數為 0，代表 Places 不回傳零評論的欄位 | `t14 --exclude-missing-reviews` 為敏感性 |
| 是否連鎖只用一個指標 | LLM 結果是不明 4,112、是 173、否 3，所以模型只放「是連鎖」 | `t14_h6_density.design` |
| 事前 MDE | 只用設計矩陣與虛無模型的殘差計算，在估計業態斜率之前執行並存檔 | `t14 --mde-only` |

---

## 6. 資料夾結構

```
dadaocheng/
├── CLAUDE.md                     規則（每次新對話會讀）
├── README.md                     本檔
├── 論文稿_上／下、計畫提案_1／2、簡報大綱、假說四結果報告、執行計畫、裁決列表、撰寫備忘
├── pyproject.toml、.env.example
├── config/
│   ├── settings.yaml             種子、門檻、模型、欄位遮罩、類別對照
│   ├── budget.yaml               各 API 上限、單價、累計花費（程式寫入）
│   ├── boundary_main.geojson     主範圍（FINAL_FROM_OSM）
│   └── boundary_buffer.geojson   主範圍外擴 300 公尺
├── docs/
│   ├── proposal.md、implementation_A.md、implementation_B.md、decision_review.md
│   ├── preregistration.md        預先登錄（prereg tag 後唯讀）
│   ├── deviations.md             偏離紀錄
│   ├── audit_manual.md           抽查手冊
│   ├── audit_form_example.xlsx   抽查填寫範例（虛構店家）
│   ├── evidence_cards.md、places_terms.md
├── prompts/extraction_v1.md      抽取 prompt
├── src/dadaocheng/
│   ├── config.py、api_layer.py、clients.py
│   ├── geo.py                    邊界、座標轉換、主範圍外擴
│   ├── names.py                  店名與地址正規化（含 NaN 防護）
│   ├── heritage.py、transit.py
│   ├── t02_places_census.py … t06_prompt_dev.py
│   ├── extraction.py             schema、證據檢查、文青推導、多數決
│   ├── t07_audit_sample.py       分層抽樣、CSV 與 Excel 表單
│   ├── t08_batch_extract.py      批次、同步補跑、截斷重跑
│   ├── t09_gemini_crosscheck.py、t10_kappa_gate.py
│   ├── t11_h1_hours.py、t12_h5_heritage.py、t13_h3_zoning.py
│   ├── t14_h6_density.py         假說四：三個規格、野自助法、MDE、逐一剔除街區
│   ├── stats.py                  區間、Holm、置換、CLQ、Fisher、MH、κ、群集 OLS、野自助法
│   ├── report_rq1.py             Holm 校正與摘要
│   └── plotting.py
├── scripts/
│   ├── check_rules.py            CLAUDE.md 規則自動檢查
│   ├── build_boundary.py         由 OSM 建構主範圍
│   ├── diagnose.py               識別力、共線性、缺值、H5 分層等診斷
│   └── diagnose_t08.py           LLM 抽取失敗原因診斷
├── tests/                        pytest（173 個）
├── data/                         整個不進 git
└── outputs/figures/、tables/
```

---

## 7. 設定檔

- **`config/settings.yaml`**
  - `main_inclusive_buffer_m: 10`：主範圍外擴距離。
  - `models.extraction_max_tokens: 1200`：抽取的輸出上限；`extraction_max_tokens_retry` 是截斷重跑時用的上限。程式庫預設 4000，你本機實際用的是 1800。
  - `models.gemini_crosscheck`：目前的 `gemini-flash-latest` 無效，需要改成有效的名稱。
  - 其餘為種子 `20261108`、Places 欄位遮罩、去重門檻、類別對照、傳統業種關鍵字等。
  - 改動會影響預先登錄的值時，要寫進 `docs/deviations.md`。
- **`config/budget.yaml`**：上限為 Claude 1,000 元、Google 1,500 元、Gemini 0 元。`spent_twd` 等欄位由程式寫入，不要手動改。

---

## 8. 規則如何在程式中落實

| CLAUDE.md 規則 | 落實方式 | 自動檢查 |
|---|---|---|
| 1 訪談不進程式庫、不進 LLM | prompt 只組 Places 名稱、類別與官網正文 | `check_rules.py` |
| 2 預先登錄 tag 後唯讀 | 程式不寫此檔 | `check_rules.py` |
| 3 抽查與開發樣本分開 | 抽樣排除開發樣本；表單不顯示 LLM 答案；範例用虛構店家 | 測試 |
| 4 付費 API 走同一呼叫層 | `api_layer.ApiCaller`：快取、記帳、上限、單步 200 元門檻 | `check_rules.py`＋測試 |
| 5 Places 欄位遮罩 | 拒絕評論全文等禁止欄位 | 測試＋`check_rules.py` |
| 6 社群不抓 | `t05_scrape.classify_url`；轉址到社群也算 | 測試 |
| 7 證據必須是子字串 | `extraction.evidence_is_substring`；無效欄位改為「不明」 | 測試 |
| 8 Gemini 只送公開文本 | 只接 `user_content()` | — |
| 9 金鑰只在 .env | `config.require_env`；User-Agent 不含 email | `check_rules.py` |
| 10 種子 | `config.rng()`、`settings.seed` | `check_rules.py` |
| 11 文青由程式推導 | `extraction.derive_wenqing_group` | 測試 |

---

## 9. 交通資料說明

| 檔案（`data/raw/transit/`） | 內容 | 處理 |
|---|---|---|
| `mrt_stations.csv` | 臺北捷運車站，WGS84 | 轉乘站合併成一列 |
| `mrt_routes_twd97.json` | 捷運路網，EPSG:3826 | 只用於畫圖 |
| `busstop.shp/.shx/.dbf` | 公車站牌 3,407 站 | 站名被截在 22 bytes，標記 `name_truncated` |
| `bus_shapes_GetBusShape.json` | 公車路線軌跡 | 主要使用 |

產生的特徵：`dist_mrt_m`（到最近捷運站中心點的直線距離；資料沒有出入口）、`bus_stop_count_300`、`bus_density`。離主範圍最近的捷運站是大橋頭站（約 160 公尺）。

---

## 10. 測試

```powershell
python -m pytest                    # 全部（173 個）
python -m pytest tests\test_t14_h6.py -v
```

| 檔案 | 驗證什麼 |
|---|---|
| `test_t00_api_layer.py` | 快取不重複扣款、上限、門檻、Windows 檔案鎖重試 |
| `test_t02_census.py` | 四分樹找回所有點、網格原點固定、請求上限時保留結果 |
| `test_t03_merge.py` | 正規化、相似度門檻、人工判斷套用、`poi_id` 穩定 |
| `test_t04_features.py` | 營業時間處理、街區合併、鄰域排除自己、核密度 |
| `test_missing_values.py` | NaN 在各欄位不會讓程式崩潰或被當成字串 |
| `test_build_boundary.py` | 道路名稱比對排除巷弄、分隔車道與 OSM 缺口、外擴 10 公尺 |
| `test_heritage.py` | 縣市與行政區過濾、廟宇清理、三條比對路徑 |
| `test_transit.py` | 交通檔解析與特徵 |
| `test_t05_scrape.py` | 社群排除、robots、限速、續跑 |
| `test_t06_extraction.py` | schema、證據子字串、`name_only` 強制、文青推導、多數決 |
| `test_t07_t10_audit.py` | 分層與固定配額、Excel 表單與範例、截斷重跑只重跑截斷的、批次狀態、κ 報表 |
| `test_t11_t13_hypotheses.py` | 假說一到三在有訊號與無訊號資料上的行為 |
| `test_t14_h6.py` | 假說四：有異質性時支持、精確虛無資料時證偽、MDE 不依賴斜率、零評論處理、輸出與 Holm |
| `test_stats.py` | 各種區間、Holm、置換、CLQ、κ、野自助法（虛無為真時 p 不偏小）、組內去平均 |
| `test_pipeline_integration.py` | 全流程（模擬外部服務） |

---

## 11. 已知問題與待辦

| 問題 | 影響 | 處理 |
|---|---|---|
| 仍有 361 次抽取被截斷 | 失敗率 2.8%，高於 2% 的驗收標準；集中在有網站文本的店家 | 需要第二層重跑（尚未實作），或接受並記入限制 |
| 預先登錄尚未打 tag | H1、H3、H5 的部分規格是看過初步結果後才定案的 | 定稿、打 `prereg` tag，並記錄時間 |
| 文青推導的「是否連鎖不明」規則 | 程式預設把不明視為獨立，與建議的保守規則不同 | 全組決定後修改 `derive_wenqing_group` 的預設 |
| 閘門一的後續路徑 | 改 schema、換模型，或啟動備援 | 全組決定 |
| T15 消融、T16–T17 RQ3 | 尚未撰寫 | 依時程決定是否在海報前完成 |
| 參考文獻 | 卷期頁碼未逐筆核對 | 投稿或海報前查證 |

---

## 12. 關鍵 tag

`prereg` → `census-done` → `prompt-frozen` → `gate1` → `results-frozen`。只 tag 程式碼與文件，資料不進 git。

---

## 13. 編號對照

程式、檔名與輸出表沿用程式代號；簡報與論文使用「假說一到四」。

| 簡報／論文 | 程式代號 | 程式 | 輸出前綴 |
|---|---|---|---|
| 假說一 | H1 | `t11_h1_hours.py` | `h1_` |
| 假說二 | H3 | `t13_h3_zoning.py` | `h3_` |
| 假說三 | H5 | `t12_h5_heritage.py` | `h5_` |
| 假說四 | H6 | `t14_h6_density.py` | `h6_` |
| （已移除） | H7 | — | — |
