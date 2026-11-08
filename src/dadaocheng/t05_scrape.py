"""T05：店家官網正文抓取（背景執行、可中斷續跑）。

規格（實作規劃 B T05）：
- 社群平台網址（facebook、instagram、threads、line…）一律不抓，標為 name_only（規則 6）。
- 遵守 robots.txt；每個網域每秒最多 1 個請求；逾時 10 秒；User-Agent 用通用研究用名稱（規則 9：不放 email）。
- trafilatura 取正文；首頁太短就嘗試同網域的「關於」頁一次。
- 正文截到約 1,500 token；存網址、抓取時間、雜湊。
- 回報 website／name_only 比例；name_only 超過六成時提醒（要回報給人）。

輸出：
  data/web_cache/{url 雜湊}.json        每個網址一筆（續跑時跳過）
  data/interim/web_text.parquet         每個 POI 一列：poi_id, text_status, reason, url, final_url, fetched_at, text_sha256, text
用法：
  python -m dadaocheng.t05_scrape --run            （在背景跑：Windows 用 start /b，或另開一個終端機）
  python -m dadaocheng.t05_scrape --report         （只彙整已抓到的結果）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd

from . import config
from .clients import estimate_tokens


# ---------------------------------------------------------------------------
# 網址分類
# ---------------------------------------------------------------------------

def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def is_social(url: str | None) -> bool:
    if not url:
        return False
    host = domain_of(url)
    for d in config.settings()["scrape"]["social_domains"]:
        if host == d or host.endswith("." + d):
            return True
    return False


def classify_url(url: str | None) -> tuple[str, str]:
    """回傳 (text_status 預設值, reason)。website 代表「可以嘗試抓」。"""
    if not url or not isinstance(url, str) or not url.strip():
        return "name_only", "no_website"
    u = url.strip()
    if not u.startswith(("http://", "https://")):
        return "name_only", "invalid_url"
    if is_social(u):
        return "name_only", "social_platform"
    return "website", "to_fetch"


# ---------------------------------------------------------------------------
# 截斷
# ---------------------------------------------------------------------------

def truncate_to_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    # 盡量在句子或段落結尾切
    m = max(cut.rfind("\n"), cut.rfind("。"), cut.rfind("."))
    return cut[: m + 1] if m > lo * 0.8 else cut


# ---------------------------------------------------------------------------
# 抓取器
# ---------------------------------------------------------------------------

class Fetcher:
    def __init__(self, client=None):
        import httpx

        s = config.settings()["scrape"]
        self.s = s
        self.client = client or httpx.Client(
            headers={"User-Agent": s["user_agent"], "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.5"},
            timeout=float(s["timeout_s"]), follow_redirects=True,
        )
        self.robots: dict[str, RobotFileParser | None] = {}
        self.last_hit: dict[str, float] = {}
        self.sleep = time.sleep

    def _throttle(self, url: str) -> None:
        d = domain_of(url)
        gap = float(self.s["per_domain_interval_s"])
        t = self.last_hit.get(d)
        now = time.monotonic()
        if t is not None and now - t < gap:
            self.sleep(gap - (now - t))
        self.last_hit[d] = time.monotonic()

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.robots:
            rp = RobotFileParser()
            try:
                self._throttle(url)
                r = self.client.get(base + "/robots.txt")
                if r.status_code in (401, 403):
                    rp.disallow_all = True
                elif r.status_code >= 400:
                    rp.allow_all = True
                else:
                    rp.parse(r.text.splitlines())
            except Exception:  # noqa: BLE001  robots 讀不到 → 視為允許（慣例），但仍限速
                rp.allow_all = True
            self.robots[base] = rp
        rp = self.robots[base]
        return rp.can_fetch(self.s["user_agent"], url)

    def get(self, url: str):
        self._throttle(url)
        return self.client.get(url)


def extract_text(html: str, url: str | None = None) -> str:
    import trafilatura

    txt = trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_recall=True) or ""
    return re.sub(r"\n{3,}", "\n\n", txt).strip()


def find_about_link(html: str, base_url: str) -> str | None:
    pats = [p.lower() for p in config.settings()["scrape"]["about_patterns"]]
    base_dom = domain_of(base_url)
    for m in re.finditer(r"<a\b[^>]*href=[\"']([^\"'#]+)[\"'][^>]*>(.*?)</a>", html, flags=re.I | re.S):
        href, label = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip().lower()
        full = urljoin(base_url, href)
        if domain_of(full) != base_dom:
            continue
        hay = (href.lower() + " " + label)
        if any(p in hay for p in pats):
            return full
    return None


def fetch_site(url: str, fetcher: Fetcher) -> dict:
    s = config.settings()["scrape"]
    rec = {"url": url, "final_url": None, "about_url": None, "status": None, "text": "",
           "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not fetcher.allowed(url):
        rec["status"] = "robots_disallowed"
        return rec
    try:
        r = fetcher.get(url)
    except Exception as e:  # noqa: BLE001
        rec["status"] = f"error:{type(e).__name__}"
        return rec
    rec["final_url"] = str(r.url)
    if r.status_code >= 400:
        rec["status"] = f"http_{r.status_code}"
        return rec
    if is_social(str(r.url)):
        rec["status"] = "redirect_to_social"
        return rec
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype and ctype:
        rec["status"] = "not_html"
        return rec
    html = r.text
    text = extract_text(html, str(r.url))
    if len(text) < int(s["min_homepage_chars"]):
        about = find_about_link(html, str(r.url))
        if about and fetcher.allowed(about):
            try:
                ra = fetcher.get(about)
                if ra.status_code < 400:
                    rec["about_url"] = str(ra.url)
                    more = extract_text(ra.text, str(ra.url))
                    text = (text + "\n\n" + more).strip() if more else text
            except Exception:  # noqa: BLE001
                pass
    text = truncate_to_tokens(text, int(s["max_tokens"]))
    rec["text"] = text
    rec["status"] = "ok" if text else "empty_text"
    return rec


# ---------------------------------------------------------------------------
# 批次
# ---------------------------------------------------------------------------

def cache_path(url: str) -> Path:
    return config.path("web_cache") / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"


def run(pois: pd.DataFrame, fetcher: Fetcher | None = None, log=print) -> pd.DataFrame:
    fetcher = fetcher or Fetcher()
    out = []
    todo = pois[["poi_id", "website_uri"]].copy()
    n = len(todo)
    for i, (pid, url) in enumerate(zip(todo["poi_id"], todo["website_uri"]), 1):
        status, reason = classify_url(url)
        row = {"poi_id": pid, "url": url if isinstance(url, str) else None, "final_url": None, "fetched_at": None,
               "text_sha256": None, "text": "", "fetch_status": reason}
        if status == "website":
            cp = cache_path(url.strip())
            if cp.exists():
                rec = json.loads(cp.read_text(encoding="utf-8"))
            else:
                rec = fetch_site(url.strip(), fetcher)
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            row.update({"final_url": rec["final_url"], "fetched_at": rec["fetched_at"], "fetch_status": rec["status"]})
            if rec["status"] == "ok" and rec["text"]:
                row["text"] = rec["text"]
                row["text_sha256"] = hashlib.sha256(rec["text"].encode("utf-8")).hexdigest()
            else:
                status = "name_only"
        row["text_status"] = status
        row["reason"] = reason if status == "name_only" and reason != "to_fetch" else row["fetch_status"]
        out.append(row)
        if i % 50 == 0:
            log(f"[{i}/{n}] 已處理")
    return pd.DataFrame(out)


def report(df: pd.DataFrame) -> str:
    alert = float(config.settings()["scrape"]["name_only_alert_ratio"])
    share = df["text_status"].value_counts(normalize=True)
    lines = ["text_status 比例：", share.to_string(), "", "原因：", df["reason"].value_counts().to_string()]
    no = float(share.get("name_only", 0))
    if no > alert:
        lines.append(f"\n⚠ name_only 比例 {no:.1%} 超過 {alert:.0%}：請告訴黃旭寬（影響閘門一與研究限制，CLAUDE.md 必須停下來問人）。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--source", default=None, help="POI 表（預設：data/final/poi_master.parquet，沒有就用 data/interim/poi_merged.parquet）")
    args = ap.parse_args(argv)
    src = Path(args.source) if args.source else config.path("final") / "poi_master.parquet"
    if not src.exists():
        src = config.path("interim") / "poi_merged.parquet"
    pois = pd.read_parquet(src)
    out_path = config.path("interim") / "web_text.parquet"
    if args.report:
        print(report(pd.read_parquet(out_path)))
        return
    df = run(pois)
    df.to_parquet(out_path, index=False)
    print(report(df))
    print(f"已存：{out_path}")


if __name__ == "__main__":
    main()
