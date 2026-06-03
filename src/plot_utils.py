"""matplotlib 中文字型設定。

產圖前呼叫一次 :func:`setup_chinese_font`,即可讓中文標題 / legend 正常顯示
(否則在無 CJK 字型 fallback 時會變成 □□□ tofu)。
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt


def setup_chinese_font() -> str:
    """設定 matplotlib 中文字型,回傳實際使用的字型名稱(找不到則回傳空字串)。"""
    candidates = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "SimHei"]
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return font
    return ""
