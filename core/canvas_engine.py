import cv2
import numpy as np
import math
from config import BOARD_W as W, BOARD_H as H, BRUSH_THICKNESS, ERASER_THICKNESS, COLOR_BRUSH, COLOR_ERASER, COLOR_BACKGROUND

class CanvasEngine:
    def __init__(self):
        self.canvas = np.full((H, W, 3), COLOR_BACKGROUND, np.uint8)
        self.stroke_color = COLOR_BRUSH
        self.prev_pos = None
        self.trajectory = []
        self.stroke_base = None

    def clear(self):
        self.canvas = np.full((H, W, 3), COLOR_BACKGROUND, np.uint8)
        self.prev_pos = None
        self.trajectory = []
        self.stroke_base = None

    def draw_or_erase(self, pos, tool, color=COLOR_BRUSH):
        if not pos:
            completed = self.trajectory if tool == "PAINT" else []
            self.prev_pos = None
            self.trajectory = []
            return completed

        if tool == "PAINT":
            if not self.trajectory:
                self.stroke_base = self.canvas.copy()
                self.stroke_color = tuple(color)
            if self.prev_pos is None:
                cv2.circle(self.canvas, pos, max(1, BRUSH_THICKNESS // 2), self.stroke_color, -1)
            if self.prev_pos:
                cv2.line(self.canvas, self.prev_pos, pos, self.stroke_color, BRUSH_THICKNESS)
            self.prev_pos = pos
            self.trajectory.append(pos)
        elif tool == "ERASE":
            cv2.circle(self.canvas, pos, ERASER_THICKNESS // 2, COLOR_ERASER, -1)
            self.prev_pos = None
            self.trajectory = []
            self.stroke_base = None
        return []

    def replace_last_stroke(self, shape):
        if self.stroke_base is None:
            return
        self.canvas = self.stroke_base
        kind, geometry = shape
        if kind == "circle":
            center, radius = geometry
            cv2.circle(self.canvas, center, radius, self.stroke_color, BRUSH_THICKNESS)
        else:
            cv2.polylines(self.canvas, [geometry], True, self.stroke_color, BRUSH_THICKNESS)
        self.stroke_base = None

    def is_closed_candidate(self):
        points = self.trajectory
        if len(points) < 20:
            return False
        xs, ys = zip(*points)
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if width < 50 or height < 50:
            return False
        diagonal = math.hypot(width, height)
        path_length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        return path_length >= 2.0 * diagonal and math.dist(points[0], points[-1]) <= 0.15 * diagonal

    def render_preview(self, cursor_pos, tool):
        display = self.canvas.copy()
        if cursor_pos:
            color = (40, 40, 40) if tool == "PAINT" else (0, 0, 255)
            radius = 6 if tool == "PAINT" else ERASER_THICKNESS // 2
            cv2.circle(display, cursor_pos, radius, color, 2)
        return display
