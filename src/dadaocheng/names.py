"""店名與地址正規化（T03 去重、T04 文資地址比對共用）。

處理：全形／半形（NFKC）、「臺」→「台」、空白與標點、英文大小寫、常見分店字尾。
"""

from __future__ import annotations

import math
import re
import unicodedata

from rapidfuzz import fuzz


def as_text(v) -> str:
    """把可能是 None、NaN 或非字串的值轉成乾淨字串。

    pandas 讀回來的缺值是 float('nan')，而 **bool(float('nan')) 是 True**，
    所以 `if v:` 會把缺值當成有值，接著呼叫字串方法就會丟 AttributeError。
    所有接受「可能缺值的欄位」的函式都要先過這一關。
    """
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "nat", "<na>", "null") else s

_PUNCT = re.compile(r"[\s\-‐‑‒–—―_·・•．.,，、。:：;；!！?？'\"“”‘’()（）\[\]【】{}《》〈〉<>/／\\|&＆+＋~～*＊#＃@＠]")
_DEFAULT_SUFFIXES = ["分店", "本店", "總店", "門市", "店"]


def normalize_name(name: str | None, suffixes: list[str] | None = None) -> str:
    name = as_text(name)
    if not name:
        return ""
    s = unicodedata.normalize("NFKC", name).lower()
    s = s.replace("臺", "台")
    s = _PUNCT.sub("", s)
    sfx = sorted(suffixes or _DEFAULT_SUFFIXES, key=len, reverse=True)
    for x in sfx:
        # 只去掉一次字尾，且去掉後至少留 2 個字（避免「書店」→「書」）
        if s.endswith(x) and len(s) - len(x) >= 2:
            s = s[: -len(x)]
            break
    return s


SUBSTRING_CAP = 84.0  # 低於預設的自動合併門檻 85：只靠「包含關係」不能自動合併


def name_similarity(a: str | None, b: str | None, suffixes: list[str] | None = None) -> float:
    """0–100。

    - 正規化後完全相同 → 100。
    - 主分數＝max(ratio, token_set_ratio)。
    - 一方是另一方的子字串（例如「小藝埕咖啡」與「小藝埕咖啡館二館」）時，最多只給 SUBSTRING_CAP，
      讓它落進「模稜兩可、給人看」的區間，而不是自動合併（二館、分館常是不同的店）。
    """
    na, nb = normalize_name(a, suffixes), normalize_name(b, suffixes)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    base = float(max(fuzz.ratio(na, nb), fuzz.token_set_ratio(na, nb)))
    if min(len(na), len(nb)) >= 2 and (na in nb or nb in na):
        return min(max(base, SUBSTRING_CAP), SUBSTRING_CAP)
    return min(base, 99.0)


_CN_NUM = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9"}


def normalize_address(addr: str | None) -> str:
    """把地址正規化成「路名段巷弄號」的鍵，供文資比對。

    - 去掉郵遞區號、國名、「台北市」「大同區」與樓層。
    - 「一段」→「1段」；全形數字→半形；「臺」→「台」。
    - 「之」號保留（例如 61之1號 → 61-1號）。
    """
    addr = as_text(addr)
    if not addr:
        return ""
    s = unicodedata.normalize("NFKC", addr).replace("臺", "台")
    s = re.sub(r"^\d{3,6}", "", s)
    s = re.sub(r"(台灣|taiwan)", "", s, flags=re.I)
    s = re.sub(r"台北市|大同區|萬華區|中山區", "", s)
    s = re.sub(r"[\s,，]", "", s)
    s = re.sub(r"([一二三四五六七八九])段", lambda m: _CN_NUM[m.group(1)] + "段", s)
    s = re.sub(r"(\d+)之(\d+)號", r"\1-\2號", s)
    s = re.sub(r"(\d+)號之(\d+)", r"\1-\2號", s)
    s = re.sub(r"\d+樓.*$|[bB]\d+.*$|地下\d*樓.*$", "", s)
    m = re.match(r"(.+?(?:路|街|大道)(?:\d+段)?(?:\d+巷)?(?:\d+弄)?\d+(?:-\d+)?號)", s)
    return m.group(1) if m else s
