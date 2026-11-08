"""matplotlib 中文字型設定（Windows：微軟正黑體；其他：Noto Sans CJK TC）。"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager

_CANDIDATES = [
    "Microsoft JhengHei", "Noto Sans TC", "Noto Sans CJK TC", "Noto Sans CJK JP",
    "PingFang TC", "Heiti TC", "WenQuanYi Zen Hei", "Source Han Sans TC",
]


def setup_cjk_font() -> str | None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None
