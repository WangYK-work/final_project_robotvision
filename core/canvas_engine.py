import cv2
import numpy as np
from config import W, H, BRUSH_THICKNESS, ERASER_THICKNESS, COLOR_BRUSH, COLOR_ERASER

class CanvasEngine:
    def __init__(self):
        self.canvas = np.zeros((H, W, 3), np.uint8)
        self.prev_pos = None
        self.trajectory = []

    def clear(self):
        self.canvas = np.zeros((H, W, 3), np.uint8)

    def draw_or_erase(self, pos, tool):
        if not pos:
            self.prev_pos = None
            self.trajectory = []
            return

        if tool == "PAINT":
            if self.prev_pos:
                cv2.line(self.canvas, self.prev_pos, pos, COLOR_BRUSH, BRUSH_THICKNESS)
            self.prev_pos = pos
            self.trajectory.append(pos)
        elif tool == "ERASE":
            cv2.circle(self.canvas, pos, ERASER_THICKNESS // 2, COLOR_ERASER, -1)
            self.prev_pos = None

    def render_preview(self, cursor_pos, tool):
        display = self.canvas.copy()
        if cursor_pos:
            color = (255, 255, 255) if tool == "PAINT" else (0, 0, 255)
            radius = 6 if tool == "PAINT" else ERASER_THICKNESS // 2
            cv2.circle(display, cursor_pos, radius, color, 2)
        return display