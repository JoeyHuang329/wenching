# 實作規劃 B：技術規格與 Claude Code 任務卡

2026/11/8 · 對應文件：專案提案書（定稿）、實作規劃 A、CLAUDE.md

這份是給**黃旭寬與 Claude Code** 用的。前半是規格（資料夾、資料表、設定值），後半是 19 張任務卡。每張卡寫明輸入、輸出、驗收條件，以及可以直接貼給 Claude Code 的指令。這份不含程式碼。

---

## 1. 技術選型

| 用途 | 套件或服務 | 備註 |
|---|---|---|
| 空間運算 | geopandas、shapely、pyproj | 距離一律在 EPSG:3826 計算 |
| OSM | Overpass API（HTTP） | 快取回應 |
| Google Places | Places API（New）`places:searchNearby` | 必帶 `X-Goog-FieldMask` |
| 網頁抓取與正文抽取 | httpx、trafilatura | 遵守 robots.txt |
| 主抽取 LLM | Anthropic SDK，`claude-haiku-5-5`，Message Batches | 結構化輸出 |
| 長尾 Agent、RQ3 草稿 | `claude-sonnet-5-5` | 有預算上限 |
| 交叉檢查、RQ3 判官 | Google Gen AI SDK，Gemini 免費方案的 Flash 系列模型 | 模型名稱以 AI Studio 顯示為準；注意每日請求數 |
| 資料驗證 | pydantic | 每筆 LLM 輸出都要通過驗證 |
| 統計 | statsmodels、scipy | 固定效果、群集標準誤、Fisher、Holm |
| 偏誤修正 | `ppi_py`（先確認函式與假設） | 分層抽樣時改用手動加權公式 |
| 機器學習 | scikit-learn、lightgbm | `GroupKFold` |
| 圖 | matplotlib | 中文字型要設定（Noto Sans TC 或微軟正黑體） |

---

## 2. 資料夾結構

```
dadaocheng/
├── CLAUDE.md
├── .env                     # 金鑰，不 commit
├── config/
│   ├── settings.yaml        # 種子、半徑、門檻、模型名稱
│   ├── budget.yaml          # 各 API 的花費上限與目前累計
│   ├── boundary_main.geojson
│   └── boundary_buffer.geojson
├── docs/
│   ├── preregistration.md   # tag prereg 之後唯讀
│   ├── deviations.md
│   ├── evidence_cards.md    # 隊友 B 撰寫
│   ├── places_terms.md      # 隊友 D 撰寫
│   └── audit_manual.md
├── data/                    # 整個資料夾不 commit
│   ├── raw/                 # OSM、文資、交通的原始下載
│   ├── places_cache/        # Places 回應（依條款處理）
│   ├── web_cache/           # 抓取的網頁
│   ├── llm_cache/           # 所有 LLM 原始回應
│   ├── interim/
│   └── final/
├── src/                     # 程式，依任務卡編號命名
├── tests/
└── outputs/
    ├── figures/
    ├── tables/
    └── rq3_cases/           # 每個案例一個資料夾，交給隊友
```

訪談整稿**不放進**這個資料夾。

---

## 3. 核心資料表

### 3.1 `poi_master`（每個 POI 一列）

| 欄位 | 型別 | 來源 | 說明 |
|---|---|---|---|
| poi_id | 字串 | 自建 | 穩定編號，例如 `P00001` |
| place_id | 字串 | Places | 可為空（僅 OSM 來源） |
| name | 字串 | Places／OSM | |
| lon、lat | 浮點 | Places／OSM | WGS84 |
| x、y | 浮點 | 換算 | EPSG:3826，公尺 |
| in_main | 布林 | 計算 | 是否在主範圍 |
| block_id | 字串 | 計算 | 街區編號 |
| on_main_street | 布林 | 計算 | 是否面向主街（迪化街一、二段等，清單寫在 config） |
| places_primary_type、places_types | 字串 | Places | |
| business_status | 字串 | Places | 只保留 `OPERATIONAL` 進分析 |
| user_rating_count | 整數 | Places | H6 結果變數 |
| price_level | 字串 | Places | H6 控制變數 |
| website_uri | 字串 | Places | 文本入口 |
| close_time_weekday | 時間 | 由營業時間計算 | 週一到週五關門時間的中位數 |
| closes_before_19 | 布林 | 計算 | H1 |
| in_heritage | 布林 | 文資比對 | H5、H6 |
| heritage_kde_150、heritage_kde_300 | 浮點 | 計算 | 歷史密度 |
| dist_mrt_m、bus_density | 浮點 | 計算 | |
| dist_temple_m | 浮點 | 計算 | 到霞海城隍廟 |
| poi_context_* | 浮點 | 計算 | 100 公尺內各業態比例（以 Places 類別計算） |
| source | 字串 | 自建 | `places`、`osm`、`both` |

### 3.2 `extractions`（每個 POI 一列，三次多數決之後）

| 欄位 | 值域 |
|---|---|
| poi_id | |
| text_status | `website`、`agent`、`name_only` |
| category | 熱食餐飲、咖啡茶飲甜點、傳統業種、選物與文創零售、文化空間、一般零售與品牌、其他 |
| is_chain | 是、否、不明 |
| style_design、style_local_story、style_handmade、style_reading | 是、否、不明 |
| wenqing_group | 咖啡與茶空間、閱讀與展演、選物與工藝、非文青 |
| history_narrative | 無、建築、街區、家族（可複選，以分號分隔） |
| customer_type | 目的型、路過型、批發、在地民生、不明 |
| founded_year | 整數或空值 |
| evidence_* | 每個欄位一段引文；必須是輸入文本的子字串 |
| evidence_valid_* | 布林，由程式檢查 |
| agreement | 三次是否一致：`3/3`、`2/3`、`1/3` |
| model、prompt_version | |

`wenqing_group` 由程式依提案書第 6.4 節的規則，從 `category`、`is_chain`、`style_*` **推導**，不直接叫 LLM 判斷。這樣定義是透明的，改規則也不用重跑 LLM。

### 3.3 `audit`（120 筆抽查）

| 欄位 | 說明 |
|---|---|
| poi_id、stratum、sampling_prob | 抽樣資訊 |
| auditor | 抽查人 |
| h_category、h_is_wenqing、h_history_narrative | 人工判斷（看 LLM 答案之前填） |
| basis | 判斷依據：`text`（只看文本）或 `text+maps`（加看 Google Maps 頁面） |
| llm_fields_reasonable | 看完 LLM 答案後：其他欄位是否合理（是／否＋備註） |
| evidence_reasonable | 證據片段是否支持答案 |

---

## 4. 設定值（`config/settings.yaml` 的初始內容）

| 鍵 | 值 | 說明 |
|---|---|---|
| seed | 20261108 | 所有隨機性共用 |
| crs_metric | EPSG:3826 | |
| buffer_m | 300 | 緩衝區 |
| context_radius_m | 100 | POI 上下文與鄰域特徵 |
| kde_bandwidths_m | [150, 300] | 歷史密度 |
| dedup_distance_m | 30 | 去重距離門檻 |
| min_block_size | 15 | 街區太小就合併 |
| places_max_requests | 900 | 超過就停（Enterprise 免費額度 1,000） |
| llm_runs | 3 | 自我一致 |
| audit_n | 120 | |
| dev_n | 20 | prompt 開發樣本 |
| kappa_threshold | 0.6 | |
| agent_max_pois | 100 | |
| agent_max_steps | 8 | |
| rq3_cases_target | 30 | |

`config/budget.yaml`：Claude 上限 1,000 元、Google 上限 1,500 元、Gemini 0 元；匯率 32。

---

## 5. 任務卡

每張卡的格式：**目標 → 輸入 → 輸出 → 驗收條件 → 給 Claude Code 的指令 → 注意**。預估時間是 Claude Code 協助下的實際工時。

### T00　專案骨架與預算守門員（11/8 上午，1 小時）

- **目標：** 建立第 2 節的資料夾結構；寫一個所有付費 API 共用的呼叫層，負責快取、計數與預算上限。
- **輸出：** 資料夾、`config/*.yaml`、`.gitignore`、呼叫層模組、單元測試。
- **驗收：** 相同請求第二次呼叫時直接讀快取、不扣額度；模擬超過上限時拋出例外；`.env` 與 `data/` 不在 git 中。
- **指令：**「讀 CLAUDE.md 與實作規劃 B 第 2、4 節。建立資料夾結構與設定檔。寫一個 API 呼叫層：以請求內容雜湊為快取鍵、記錄每次花費到 budget.yaml、超過上限丟出例外。先寫測試再寫實作。先給我計畫，不要動手。」
- **注意：** Windows 請設 `PYTHONUTF8=1`。

### T01　預先登錄（11/8 下午，30 分鐘，人主導）

- **目標：** 把提案書第 5.1 節的表格、顯著水準 0.05、Holm 校正（只用在 H1、H3、H5、H6）、第 6 節定義、街區合併規則，寫進 `docs/preregistration.md`。
- **驗收：** `git tag prereg` 已推到私有遠端。
- **指令：**「依提案書第 5、6 節，草擬 preregistration.md，逐條列出每個假說的變數、模型、檢定與證偽條件。不要加入提案書沒有的分析。」
- **注意：** 由黃旭寬讀過再 commit。之後 Claude Code 不得修改這個檔案。

### T02　Places 網格普查（11/8 下午，2 小時）

- **目標：** 以自適應網格呼叫 Nearby Search，取得範圍內所有地點。
- **做法規格：**
  - 起始網格：覆蓋「主範圍＋緩衝區」外框，邊長約 200 公尺；每格用外接圓（半徑＝半對角線）查詢。
  - `maxResultCount` 20、`rankPreference` DISTANCE、`languageCode` zh-TW、`regionCode` TW、不設 `includedTypes`。
  - 回傳滿 20 筆就把該格切成四格重查，最小邊長約 25 公尺。
  - 欄位遮罩：`places.id, places.displayName, places.location, places.types, places.primaryType, places.businessStatus, places.formattedAddress, places.regularOpeningHours, places.userRatingCount, places.rating, places.priceLevel, places.websiteUri`。
- **輸出：** `data/places_cache/`、`data/interim/places_raw.parquet`。
- **驗收：** 總請求數 ≤ 900；最小格仍滿 20 筆的格數回報出來；去掉格子重疊造成的重複 place_id。
- **指令：**「依實作規劃 B 的 T02 規格寫 Places 普查。先只跑 1 個格子給我看回應內容，我到 Cloud Console 確認計費 SKU 之後，你再跑全部。所有呼叫走 T00 的呼叫層。」
- **注意：** 第一次正式跑之前，到 Console 確認被計在 Nearby Search Enterprise。學校、公園、停車場等非商業類別在 T03 濾掉，不要因此增加請求。

### T03　OSM 補漏、去重與範圍標記（11/8 晚上，1.5 小時）

- **目標：** 下載 OSM 的商業類節點與道路；與 Places 合併去重；標記 `in_main`；濾掉非商業類別。
- **驗收：** 輸出去重前後筆數；模稜兩可的配對（距離 < 30 公尺但名稱相似度中等）輸出成清單給隊友 A 看；非商業類別清單寫在 config。
- **注意：** 名稱正規化要處理全形與半形、「臺」與「台」、「店」「分店」「本店」等字尾。

### T04　文資比對、街區切割與結構化特徵（11/9 上午，3 小時）

- **目標：** 產生 `poi_master` 的所有結構化欄位。
- **規格：**
  - 文資：有座標就用「POI 落在建物輪廓內，或距離 ≤ 15 公尺」；只有地址就正規化地址後比對。輸出比對表給隊友 A 人工檢查。
  - 街區：以 OSM 道路（排除小巷）切割主範圍；不足 15 個 POI 的街區與相鄰街區合併。
  - 關門時間：取週一到週五的關門時間中位數；跨午夜的營業時間要正確處理；沒有營業時間的標成缺失，不當成「不營業」。
- **驗收：** 每個欄位的缺失比例；文資比對率；`closes_before_19` 的分布；三張檢查地圖（文資點位、街區、主街）。
- **注意：** 所有距離在 EPSG:3826 計算。核密度的帶寬來自 config。

### T05　官網文本抓取（11/8 晚上開始，背景執行）

- **目標：** 對有 `website_uri` 的 POI 抓取正文。
- **規格：**
  - 網址是 facebook、instagram、threads、line 等社群平台的，**一律不抓**，標為 `name_only`。
  - 遵守 robots.txt；每個網域每秒最多 1 個請求；逾時 10 秒；User-Agent 用通用研究用名稱。
  - 以 trafilatura 取正文；若首頁太短，嘗試同網域的「關於」頁（about、story、品牌故事）一次。
  - 正文截到約 1,500 token；存網址、抓取時間、雜湊。
- **驗收：** 回報 `website`、`name_only` 的比例。**超過六成是 `name_only` 時告訴我**，這會影響閘門一與研究限制。
- **注意：** 台灣小店很多只有粉專，`name_only` 比例偏高是正常的。

### T06　抽取 prompt 開發（11/9 下午，2–3 小時）

- **目標：** 在 20 筆開發樣本上讓 prompt 穩定。
- **規格：**
  - 系統指令＋欄位定義＋值域放在前面（可吃到 prompt caching）；POI 的名稱、Places 類別、文本放在最後。
  - 結構化輸出對應第 3.2 節，每個欄位都有「不明」；證據片段要求逐字引用。
  - 明確指示：只依提供的文本判斷，不要用你對這家店的既有知識；`name_only` 的 POI 只能判斷 `category` 與 `is_chain`，其他欄位一律「不明」。
- **驗收：** 20 筆的結果由黃旭寬與隊友 B 看過；證據片段有效率 ≥ 90%；**打 tag `prompt-frozen`**。
- **注意：** 開發樣本與抽查樣本不能重疊（T07 抽樣時要排除）。

### T07　抽查樣本（11/9 下午，30 分鐘）

- **目標：** 抽出 120 筆抽查樣本，產生抽查表單。
- **規格：** 只從 `in_main` 且營業中的 POI 抽，排除開發樣本。先以 Places 類別粗估「可能文青」比例：低於 15% 就分兩層，「可能文青」層抽樣率加倍，記錄每筆的 `sampling_prob`。
- **輸出：** `data/final/audit_sample.csv`、四份抽查表單（每人 30 筆），表單上**先不顯示** LLM 答案。
- **驗收：** 種子固定；各層筆數與抽樣機率寫入 `deviations.md` 以外的紀錄檔。

### T08　全量批次抽取（11/9 傍晚送出）

- **目標：** 對所有 `in_main` 與緩衝區的 POI 跑三次抽取。
- **規格：** Message Batches，`custom_id` 為 `{poi_id}_{run}`；結果寫入 `llm_cache/`；以 pydantic 驗證；三次多數決；檢查證據片段是否為子字串。
- **驗收：** 失敗或格式錯誤的筆數 < 2%（失敗的以同步呼叫補跑）；產出 `extractions`。
- **注意：** 批次是非同步的，可能要好幾小時。11/10 09:00 仍沒完成就把剩下的改用同步呼叫並行跑。

### T09　Gemini 交叉檢查（11/10 上午，1 小時）

- **目標：** 在 120 筆抽查樣本與約 300 筆隨機 POI 上，以 Gemini 跑同一份 prompt 一次，計算與 Claude 的一致率。
- **注意：** 免費方案有每分鐘與每日的請求上限，程式要遇到 429 就退避重試，不要並行轟炸。只送公開的店家文本。這不是閘門的依據，只是補充指標。

### T10　計算 κ 與閘門一報表（11/10 13:00，1 小時）

- **輸出：** 三個關鍵欄位的 Cohen's κ 與準確率（自助法 95% 信賴區間）、混淆矩陣、證據片段有效率、`name_only` 子集與 `website` 子集分開報告。
- **注意：** 分層抽樣時，準確率要依抽樣機率加權。**Claude Code 只產出報表，不宣布通過。**

### T11　H1 夜間經濟（11/10 下午，1 小時）

- 比例與 95% 信賴區間（整體、分業態大類、主街對巷弄）；關門時間的分布圖。

### T12　H5 歷建與熱食（11/10 下午，1 小時）

- 2×2 表、Fisher 精確檢定、以街區分層的 Mantel–Haenszel；若熱食的類別來自 LLM，附 PPI 修正版。
- 文資內 POI 數若少於 20，報告「檢定力不足」，不要改用其他檢定湊顯著。

### T13　H3 業種分區（11/10 晚上，2 小時）

- 主街對巷弄的業態比例差；中藥、文創、南北貨三組的 CLQ 與置換檢定（固定位置、洗牌標籤，999 次）；業態分布地圖。
- 「傳統業種」裡的中藥與南北貨要分開，依 Places 類別或名稱關鍵字細分，規則寫在 config。

### T14　H6 與 PPI 修正（11/11 上午，3 小時）

- **模型：** log(1＋評論數) ＝ 歷史密度（300 公尺）× 業態大類 ＋ 控制變數（店齡或其缺失指標、連鎖、主街、捷運距離、價位）＋ 街區固定效果；標準誤以街區群集。
- **兩個版本：** 直接版；PPI 修正版（業態大類與連鎖來自 LLM 時）。若 `ppi_py` 不支援固定效果加群集標準誤，就先以去平均處理固定效果，再用其 OLS 版本，並在 `deviations.md` 記錄。
- **敏感度（時間允許）：** 納入大橋頭重跑一次。
- **圖：** 各業態的歷史密度斜率與信賴區間（森林圖）。

### T15　S 對 S+L 消融（11/11 下午，2 小時）

- **規格：** 目標是 `wenqing_group`（四類）；`GroupKFold` 以街區分組；L2 logistic 與 LightGBM；鄰域特徵排除自己；指標為 macro-F1 與對數損失；S 與 S+L 的差以街區自助法求信賴區間；另在 120 筆抽查樣本上對人工標籤評估。
- **注意：** 寫一個單元測試，專門檢查「100 公尺內文青比例」不含目標 POI 本身。

### T16　RQ3 統計層（11/11 下午，2 小時）

- **規格：** 「成功」＝營業中且評論數高於同業態大類中位數。對每個案例：置換重要度 × 單側標準化缺口的前五名；Gower 距離最近的三家成功同群店家；固定附上「資料無法觀察」清單（個別租金與租期、房東意願、經營者的其他收入或家族資源）。
- **案例挑選：** 附錄 B 的已知案例＋每個文青群各挑幾個低適配的店址，共 10–30 個。
- **輸出：** `outputs/rq3_cases/{case_id}/summary.md` 與 `summary.json`。

### T17　RQ3 證據包（11/12 下午，1 小時）

- 每個案例的 `evidence.md`：結構化事實、店家文本片段、相關假說的檢驗結果、證據卡摘要，每項有編號與來源。**不含任何訪談原文。**

### T18　RQ3 LLM 草稿與忠實度檢查（11/13 上午，2 小時）

- Sonnet 5.5 依 `summary.json` 與 `evidence.md` 寫 200 字以內的說明，每句標證據編號；程式檢查編號存在；Gemini 逐句判斷證據是否支持；以規則攔下因果處方式的語句。
- 輸出 `draft.md` 與忠實度表，交給隊友做人文詮釋。

---

## 6. 依賴關係

```
T00 → T01 → T02 → T03 → T04 ─┬→ T11、T12、T13
               └→ T05 → T06 → T07 → T08 → T09、T10（閘門一）→ T13、T14、T15 → T16 → T17 → T18
```

T11 與 T12 只依賴結構化資料，可以在閘門一之前先做。時間緊的話，這是最先可以放進簡報的結果。
