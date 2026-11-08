"""自動檢查 CLAUDE.md 中可以用程式驗證的規則。每次 commit 前跑一次：

    python scripts/check_rules.py

檢查項目：
  規則 1  程式庫中沒有訪談整稿（檔名或內容關鍵字）
  規則 2  prereg tag 之後 preregistration.md 沒被改過
  規則 4  src/ 中呼叫付費 API 的地方都經過 api_layer（沒有直接 httpx.post 到 Places 或 anthropic.messages.create 在 clients.py 之外）
  規則 5  Places 欄位遮罩沒有 reviews / editorialSummary
  規則 9  沒有金鑰樣式的字串、.env 沒被追蹤、User-Agent 沒有 email
  規則 10 settings.yaml 有 seed
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
problems: list[str] = []


def tracked_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        return [ROOT / l for l in out.splitlines() if l.strip()]
    except Exception:  # noqa: BLE001
        return [p for p in ROOT.rglob("*") if p.is_file() and "data" not in p.parts and ".git" not in p.parts]


files = tracked_files()
text_files = [p for p in files if p.suffix in {".py", ".md", ".yaml", ".yml", ".txt", ".csv", ".json", ".toml", ".geojson", ".overpassql"}]

# 規則 1
INTERVIEW_MARKERS = ["整稿①", "整稿②", "整稿③", "整稿④", "逐字稿待修", "語者1"]
for p in files:
    if re.search(r"整稿|逐字稿|transcript|interview", p.name, re.I):
        problems.append(f"規則 1：疑似訪談檔案被追蹤：{p.relative_to(ROOT)}")
for p in text_files:
    t = p.read_text(encoding="utf-8", errors="ignore")
    if any(m in t for m in INTERVIEW_MARKERS) and p.name != "check_rules.py":
        problems.append(f"規則 1：{p.relative_to(ROOT)} 含疑似訪談整稿內容")

# 規則 2
try:
    tags = subprocess.run(["git", "tag", "--list", "prereg"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if tags:
        diff = subprocess.run(["git", "diff", "--name-only", "prereg", "--", "docs/preregistration.md"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        if diff:
            problems.append("規則 2：prereg tag 之後 docs/preregistration.md 被修改了（變更請寫進 deviations.md）")
except Exception:  # noqa: BLE001
    pass

# 規則 4
for p in (ROOT / "src").rglob("*.py"):
    if p.name in {"clients.py", "api_layer.py"}:
        continue
    t = p.read_text(encoding="utf-8")
    if re.search(r"places\.googleapis\.com|messages\.create\(|generate_content\(", t):
        problems.append(f"規則 4：{p.relative_to(ROOT)} 直接呼叫付費 API，應改走 clients.py＋api_layer")

# 規則 5、9、10
settings = yaml.safe_load((ROOT / "config" / "settings.yaml").read_text(encoding="utf-8"))
mask = settings["places"]["field_mask"]
if any(re.search(r"review|editorialSummary|generativeSummary", f, re.I) for f in mask):
    problems.append("規則 5：Places 欄位遮罩含 Atmosphere 評論類欄位")
if re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", settings["scrape"]["user_agent"]):
    problems.append("規則 9：User-Agent 含 email")
if "seed" not in settings:
    problems.append("規則 10：settings.yaml 沒有 seed")
KEY_PAT = re.compile(r"(sk-ant-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,})")
for p in text_files:
    if KEY_PAT.search(p.read_text(encoding="utf-8", errors="ignore")):
        problems.append(f"規則 9：{p.relative_to(ROOT)} 疑似含 API 金鑰")
if any(p.name == ".env" for p in files):
    problems.append("規則 9：.env 被 git 追蹤")

if problems:
    print("發現問題：")
    for x in problems:
        print(" -", x)
    sys.exit(1)
print(f"可自動檢查的規則都通過（檢查了 {len(files)} 個檔案）。")
