# CLAUDE.md：大稻埕文青選址研究（定稿版）

> 放在程式庫根目錄，Claude Code 每次開新對話都會讀取。這一版取代先前的 CLAUDE.md。
> 規則與禁區寫在這裡；要做什麼寫在 `docs/implementation_B.md`（實作規劃 B）的任務卡。

## 專案脈絡

- **題目：** 訪談驅動的假說生成與大型語言模型資料建構：大稻埕文青產業選址之文化特徵檢驗與缺失因素分析。
- **設計：** 見 `docs/proposal.md`（提案書定稿）。檢驗 H1、H3、H5、H6 四條核心假說與探索性的 H7。**H2（租金）與 H4（輪替與存活）已移除**，不要實作，也不要下載實價登錄或商業登記資料。
- **分析單位：** 主範圍內營業中的商業 POI，約 2,000–3,000 筆；緩衝區的 POI 只用來算鄰域特徵。
- **硬性日期：** 11/11 22:00 結果凍結 → 11/12 簡報 → 11/15 海報。**能交付比完美重要。** 遇到兩條路時，選比較簡單、可驗證的那條。
- **環境：** Windows 筆電，47 GB RAM，RTX 5070 Laptop 8 GB。主流程不需要 GPU。

## 不可違反的規則

1. **訪談文字不進程式庫、不送進任何 LLM。** 訪談只透過 `docs/evidence_cards.md`（人工撰寫）進入分析。
2. **`docs/preregistration.md` 在 `prereg` tag 之後唯讀。** 規格變更寫進 `docs/deviations.md`（日期、原因、影響）。
3. **抽查樣本與開發樣本分開。** prompt 只能在 20 筆開發樣本上迭代。`audit_sample.csv` 的人工答案在 `prompt-frozen` 之前不得讀取、不得當成 few-shot 範例。抽查表單上不顯示 LLM 答案。
4. **所有付費 API 都走同一個呼叫層：** 先查快取再呼叫、記錄花費、超過 `config/budget.yaml` 上限就丟出例外。同樣的請求不可以付費兩次。
5. **Places 每個請求都要帶欄位遮罩。** 不要求 `reviews`、`editorialSummary` 等 Atmosphere 欄位。Google 評論文字不取得、不送進 LLM。長期只保存 `place_id`；其他內容依 `docs/places_terms.md` 的結論處理。
6. **社群平台不抓取。** 網址是 facebook、instagram、threads、line 的，一律標為 `name_only`。
7. **LLM 只能依提供的文本判斷。** 證據片段必須是輸入文本的子字串，由程式檢查。不是子字串的欄位標為無效。
8. **免費的 Gemini 只送公開的店家文本**，不送任何訪談、個人資料或 Places 的評論內容。
9. **金鑰只放在 `.env`。** User-Agent 用通用研究用名稱，不放任何人的 email。
10. **隨機性一律用 `config/settings.yaml` 的種子。**
11. **`wenqing_group` 由程式依規則推導，** 不叫 LLM 直接判斷是否為文青。規則：獨立（非連鎖）咖啡或茶空間即為文青；白名單業態需加上至少一項風格指標；連鎖一律排除。

## 必須停下來問人的情況

- 單一步驟預估花費超過新台幣 200 元，或任一帳戶累計達上限的七成。
- Places 請求數超過 900。
- 閘門一（κ）與結果凍結的判定：**只產出報表，由人決定。**
- 需要改動預先登錄。
- 官網文本的 `name_only` 比例超過六成。
- 文資比對率低於八成。
- 在資料中看到疑似訪談內容、個人姓名或電話。
- 任何資料來源或 API 連續失敗 30 分鐘。

## 各階段最容易出錯的地方

**空間：**
- 原始座標存 WGS84，距離與面積一律在 EPSG:3826 計算。
- 去重用「距離 ≤ 30 公尺且名稱相似」；模稜兩可的輸出清單給人看，不要自動合併。
- 名稱正規化：全形與半形、「臺」與「台」、「店」「分店」「本店」。
- 營業時間跨午夜要正確處理；沒有營業時間是缺失，不是「不營業」。

**Places：**
- `maxResultCount` 最多 20，沒有分頁。回傳滿 20 筆就把格子切成四格重查。
- 第一次正式跑之前，先跑 1 個請求，請人到 Cloud Console 確認計費 SKU。

**LLM：**
- 固定指令與 schema 放在 prompt 前面，POI 內容放最後，才吃得到 prompt caching。
- 批次介面是非同步的，可能要等好幾小時；`custom_id` 用 `{poi_id}_{run}`。
- Gemini 免費方案遇到 429 要退避重試，不要並行。
- `name_only` 的 POI 只能填 `category` 與 `is_chain`，其他欄位一律「不明」。

**統計與模型：**
- 分層抽樣時，準確率與 PPI 都要依 `sampling_prob` 加權。
- H6：街區固定效果加上以街區群集的標準誤。用 `ppi_py` 前先確認它支援什麼；不支援就照提案書做，並記錄在 `deviations.md`。
- 交叉驗證以 `GroupKFold` 依街區分組。
- **鄰域特徵一律排除目標 POI 本身**，並有單元測試。
- 檢定力不足時照實報告，不要換檢定湊顯著。

**RQ3：**
- 證據包不含任何訪談原文。
- LLM 草稿每句都要有證據編號，程式檢查編號是否存在；以規則攔下因果處方式的語句（例如「只要…就能」「保證」「導致失敗」）。
- 對已退場的店家，不歸咎於經營者。

## 給人類的使用建議

- 一張任務卡開一個新對話：「讀 CLAUDE.md 與實作規劃 B 的 T0X，先用計畫模式給我步驟。」看過計畫再讓它寫。
- 不要讓 Claude Code 把大型 CSV 讀進對話，請它用抽樣列印或摘要統計。
- 長時間的工作（抓取、批次）用腳本在背景跑，不要在對話裡一筆一筆處理。Pro 方案有使用額度限制。
- 關鍵節點打 tag：`prereg`、`census-done`、`prompt-frozen`、`gate1`、`results-frozen`。
- Claude Code 說「完成了」的時候，挑一條規則問它：「你怎麼確認第 3 條沒有被違反？」

## 程式庫現況（2026-11-08 補充，由 Claude 加入）

- **已實作：** T00–T06 的程式碼與測試（`python -m pytest`）。執行狀態與需要人做的步驟見 `待執行任務描述.md`；方法風險見 `實作審閱筆記.md`。
- **模組與任務卡對應：** `src/dadaocheng/` 下 `api_layer.py`（T00）、`t02_places_census.py`、`t03_osm_merge.py`、`t04_features.py`、`t05_scrape.py`、`t06_prompt_dev.py`；`extraction.py` 是 T06／T08 共用的 schema、驗證、多數決與 `wenqing_group` 推導。
- **付費呼叫一律用 `clients.py` 的工廠函式＋`api_layer.ApiCaller`。** 自我一致的第 k 次呼叫要在 `extra={"run": k}` 帶入，否則會讀到第一次的快取。
- **交通資料已在 `data/raw/transit/`：** 捷運站 CSV、捷運路網（EPSG:3826）、公車站牌 shapefile（UTF-8、WGS84、無 .prj）、公車軌跡（以 GetBusShape 為準）。`python -m dadaocheng.transit` 會裁切並輸出到 `data/interim/transit_*.geojson`。
- **`config/boundary_main.geojson` 若屬性 status 為 DRAFT，就還不是正式邊界**；程式會印出警告。
- **抽取 schema 比實作規劃 B 3.2 多一個 `fine_type`**（推導文青所需）。
- 每次說「完成」之前，跑 `python scripts/check_rules.py` 與 `python -m pytest`。
