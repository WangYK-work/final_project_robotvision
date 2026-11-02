"""Local viewport and non-destructive overview / hand toolbar rendering."""
import cv2
import numpy as np
from config import W, H, BOARD_W, BOARD_H, TOOLBAR_W

TOOLS = [("PAINT", "1  Pen"), ("ERASE", "2  Eraser"),
         ("CLEAR", "3  Clear all"), ("ZOOM", "4  Zoom"),
         ("PAN", "5  Move view"), ("PICK", "6  Eyedropper"), ("PORTRAIT", "7  Portrait")]

class Viewport:
    def __init__(self):
        self.zoom = 2.0
        self.cx, self.cy = BOARD_W / 2, BOARD_H / 2

    def bounds(self):
        width = round(BOARD_W / self.zoom)
        height = round(BOARD_H / self.zoom)
        x = min(max(round(self.cx - width / 2), 0), BOARD_W - width)
        y = min(max(round(self.cy - height / 2), 0), BOARD_H - height)
        return x, y, width, height

    def clamp(self):
        x, y, width, height = self.bounds()
        self.cx, self.cy = x + width / 2, y + height / 2

    def set_zoom(self, value):
        self.zoom = min(8.0, max(1.0, float(value)))
        self.clamp()

    def move(self, dx, dy):
        self.cx += dx
        self.cy += dy
        self.clamp()

    def to_world(self, pos):
        x, y, width, height = self.bounds()
        return (min(BOARD_W - 1, max(0, x + int(pos[0] * width / W))),
                min(BOARD_H - 1, max(0, y + int(pos[1] * height / H))))

    @staticmethod
    def left_cursor(hand):
        if hand is None:
            return None
        x, y = hand["index_pos"]
        return (min(W + TOOLBAR_W - 1, max(0, round(x * (W + TOOLBAR_W) / W))),
                min(H - 1, max(0, y)))

    @staticmethod
    def hit(pos):
        if pos and 8 <= pos[0] < TOOLBAR_W - 8:
            for i, (tool, _) in enumerate(TOOLS):
                if 10 + i * 56 <= pos[1] < 58 + i * 56:
                    return tool
        return None

    def render(self, canvas, right_pos, left_pos, tool, color, portrait_cursor=None):
        x, y, width, height = self.bounds()
        local = cv2.resize(canvas[y:y + height, x:x + width], (W, H))
        if right_pos:
            pos = tuple(int(v) for v in right_pos)
            radius = max(3, round(30 * W / width)) if tool == "ERASE" else 7
            cv2.circle(local, pos, radius + 1, (255, 255, 255), 3)
            cv2.circle(local, pos, radius, (30, 30, 30), 1)
        panel = np.full((H, W + TOOLBAR_W, 3), 242, np.uint8)
        panel[:, TOOLBAR_W:] = local
        for i, (key, label) in enumerate(TOOLS):
            top = 10 + i * 56
            fill = (215, 235, 215) if key == tool else (255, 255, 255)
            cv2.rectangle(panel, (8, top), (TOOLBAR_W - 9, top + 47), fill, -1)
            cv2.rectangle(panel, (8, top), (TOOLBAR_W - 9, top + 47), (100, 100, 100), 1)
            cv2.putText(panel, label, (15, top + 29), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.rectangle(panel, (15, 420), (48, 453), color, -1)
        cv2.putText(panel, "{:.2f}x".format(self.zoom), (60, 443), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 30, 30), 1)
        if left_pos:
            cv2.circle(panel, left_pos, 11, (255, 255, 255), 4)
            cv2.circle(panel, left_pos, 9, (255, 100, 0), 2)
        overview = canvas.copy()
        # 半透明黑色边框仅用于预览，不修改作品像素。
        overlay = overview.copy()
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0, 0, 0), 8)
        overview = cv2.addWeighted(overlay, .45, overview, .55, 0)
        if portrait_cursor is not None:
            px, py = portrait_cursor
            if x <= px < x + width and y <= py < y + height:
                local_pos = (TOOLBAR_W + int((px - x) * W / width), int((py - y) * H / height))
                cv2.drawMarker(panel, local_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(overview, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
        return panel, overview
