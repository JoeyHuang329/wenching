"""T05：社群排除、robots、限速、關於頁、截斷、續跑（用 httpx.MockTransport，不連網）。"""

from __future__ import annotations

import httpx
import pandas as pd
import pytest

LONG = "大稻埕老屋裡的小店。" * 80
SHORT_HOME = "<html><body><p>歡迎光臨</p><a href='/about-us'>關於我們</a></body></html>"
ABOUT = "<html><body><article><p>" + ("我們在迪化街的百年街屋裡經營選物店，販售台灣手作。" * 20) + "</p></article></body></html>"


def make_client(calls):
    def handler(request: httpx.Request):
        calls.append(str(request.url))
        host, path = request.url.host, request.url.path
        if path == "/robots.txt":
            if host == "blocked.tw":
                return httpx.Response(200, text="User-agent: *\nDisallow: /")
            return httpx.Response(404)
        if host == "short.tw" and path == "/":
            return httpx.Response(200, html=SHORT_HOME)
        if host == "short.tw" and path == "/about-us":
            return httpx.Response(200, html=ABOUT)
        if host == "long.tw":
            return httpx.Response(200, html="<html><body><article><p>" + LONG + "</p></article></body></html>")
        if host == "redir.tw":
            return httpx.Response(301, headers={"location": "https://www.facebook.com/someshop"})
        if host == "www.facebook.com":
            return httpx.Response(200, html="<html>fb</html>")
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_classify_url(tmp_root):
    from dadaocheng.t05_scrape import classify_url

    assert classify_url(None) == ("name_only", "no_website")
    assert classify_url("https://www.facebook.com/abc") == ("name_only", "social_platform")
    assert classify_url("https://m.facebook.com/abc")[1] == "social_platform"
    assert classify_url("https://instagram.com/x")[1] == "social_platform"
    assert classify_url("https://lin.ee/abc")[1] == "social_platform"
    assert classify_url("https://shop.example.tw") == ("website", "to_fetch")


def test_truncate(tmp_root):
    from dadaocheng.clients import estimate_tokens
    from dadaocheng.t05_scrape import truncate_to_tokens

    t = truncate_to_tokens(LONG, 200)
    assert estimate_tokens(t) <= 200 and LONG.startswith(t)


def test_run_end_to_end(tmp_root):
    from dadaocheng.t05_scrape import Fetcher, run

    calls = []
    f = Fetcher(client=make_client(calls))
    sleeps = []
    f.sleep = lambda s: sleeps.append(s)
    pois = pd.DataFrame({"poi_id": ["P1", "P2", "P3", "P4", "P5", "P6"],
                         "website_uri": ["https://short.tw/", "https://long.tw/", "https://blocked.tw/",
                                         "https://www.instagram.com/x", None, "https://redir.tw/"]})
    df = run(pois, fetcher=f, log=lambda *_: None).set_index("poi_id")
    assert df.loc["P1", "text_status"] == "website" and "百年街屋" in df.loc["P1", "text"]   # 首頁太短 → 關於頁
    assert df.loc["P2", "text_status"] == "website"
    assert df.loc["P3", "text_status"] == "name_only" and df.loc["P3", "reason"] == "robots_disallowed"
    assert df.loc["P4", "reason"] == "social_platform"
    assert df.loc["P5", "reason"] == "no_website"
    assert df.loc["P6", "reason"] == "redirect_to_social"
    assert not any("instagram" in c for c in calls)          # 社群不抓
    assert not any(c.startswith("https://blocked.tw/") and not c.endswith("robots.txt") for c in calls)
    assert sleeps, "同網域連續請求應該要限速"
    assert df.loc["P2", "text_sha256"] and df.loc["P2", "fetched_at"]

    # 續跑：快取命中，不再發請求
    n = len(calls)
    run(pois, fetcher=Fetcher(client=make_client(calls)), log=lambda *_: None)
    assert all(c.endswith("robots.txt") for c in calls[n:]) or len(calls) == n


def test_report_alert(tmp_root):
    from dadaocheng.t05_scrape import report

    df = pd.DataFrame({"text_status": ["name_only"] * 7 + ["website"] * 3, "reason": ["x"] * 10})
    assert "超過" in report(df)
