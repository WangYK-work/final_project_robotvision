"""Shared tool labels, gestures and fixed RGB/BGR colors."""
TOOLS = (
    ("PAINT", "画笔", "Pen", "只伸食指"),
    ("ERASE", "橡皮", "Eraser", "食指、中指"),
    ("CLEAR", "清空画板", "Clear", "食指、中指、无名指"),
    ("VIEW", "视图", "View", "除拇指外四指"),
    ("VIEW", "视图", "View", "五指张开"),
    ("COLOR", "选择颜色", "Colors", "拇指、小指"),
    ("PORTRAIT", "肖像", "Portrait", "拇指、食指、中指尖聚拢"),
)
PALETTE = (
    ("黑", "Black", "#000000"), ("红", "Red", "#E53935"),
    ("橙", "Orange", "#FB8C00"), ("黄", "Yellow", "#FDD835"),
    ("绿", "Green", "#43A047"), ("蓝", "Blue", "#1E88E5"),
    ("紫", "Purple", "#8E24AA"), ("棕", "Brown", "#795548"),
)

UI_TOOLS = tuple(dict((entry[0], entry) for entry in TOOLS).values())

def tool_label(tool):
    if tool == "VIEW":
        return "4/5 · 视图 / View"
    for number, (key, chinese, english, _) in enumerate(TOOLS, 1):
        if key == tool:
            return f"{number} · {chinese} / {english}"
    return tool

def palette_bgr(index):
    value = PALETTE[index][2].lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r
