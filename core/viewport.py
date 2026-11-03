"""Local viewport and non-destructive overview / palette rendering."""
import cv2
import numpy as np
from config import W, H, BOARD_W, BOARD_H
from core.tool_catalog import PALETTE, palette_bgr

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
    def palette_rects():
        margin, gap = 20, 12
        cell_w = (W - 2 * margin - 3 * gap) // 4
        cell_h = (H - 2 * margin - gap) // 2
        return [(margin + (i % 4) * (cell_w + gap),
                 margin + (i // 4) * (cell_h + gap), cell_w, cell_h) for i in range(8)]

    @classmethod
    def palette_hit(cls, pos):
        if pos is not None:
            for i, (x, y, w, h) in enumerate(cls.palette_rects()):
                if x <= pos[0] < x + w and y <= pos[1] < y + h:
                    return i
        return None

    def render(self, canvas, right_pos, tool, color, portrait_cursor=None):
        x, y, width, height = self.bounds()
        local = cv2.resize(canvas[y:y + height, x:x + width], (W, H))
        if tool == "COLOR":
            local[:] = (247, 245, 242)
            hovered = self.palette_hit(right_pos)
            for i, (px, py, pw, ph) in enumerate(self.palette_rects()):
                bgr = palette_bgr(i)
                cv2.rectangle(local, (px, py), (px + pw - 1, py + ph - 1), bgr, -1)
                if i == hovered:
                    cv2.rectangle(local, (px - 3, py - 3), (px + pw + 2, py + ph + 2), (180, 95, 25), 3)
                label_color = (255, 255, 255) if sum(bgr) < 430 else (20, 20, 20)
                cv2.putText(local, f"{i + 1} {PALETTE[i][1]}", (px + 8, py + ph - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, label_color, 1, cv2.LINE_AA)
        if right_pos is not None:
            pos = tuple(int(v) for v in right_pos)
            radius = max(3, round(30 * W / width)) if tool == "ERASE" else 7
            cv2.circle(local, pos, radius + 1, (255, 255, 255), 3)
            cv2.circle(local, pos, radius, (30, 30, 30), 1)
        overview = canvas.copy()
        overlay = overview.copy()
        cv2.rectangle(overlay, (x, y), (x + width - 1, y + height - 1), (0, 0, 0), 8)
        overview = cv2.addWeighted(overlay, .45, overview, .55, 0)
        if portrait_cursor is not None:
            px, py = portrait_cursor
            if x <= px < x + width and y <= py < y + height:
                local_pos = (int((px - x) * W / width), int((py - y) * H / height))
                cv2.drawMarker(local, local_pos, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(overview, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
        return local, overview
