import cv2
import numpy as np
import math
from config import W, H, BRUSH_THICKNESS, ERASER_THICKNESS, COLOR_BRUSH, COLOR_ERASER

class CanvasEngine:
    def __init__(self):
        self.canvas = np.zeros((H, W, 3), np.uint8)
        self.prev_pos = None
        self.trajectory = []
        self.stroke_base = None

    def clear(self):
        self.canvas = np.zeros((H, W, 3), np.uint8)
        self.prev_pos = None
        self.trajectory = []
        self.stroke_base = None

    def draw_or_erase(self, pos, tool):
        if not pos:
            completed = self.trajectory if tool == "PAINT" else []
            self.prev_pos = None
            self.trajectory = []
            return completed

        if tool == "PAINT":
            if not self.trajectory:
                self.stroke_base = self.canvas.copy()
            if self.prev_pos:
                cv2.line(self.canvas, self.prev_pos, pos, COLOR_BRUSH, BRUSH_THICKNESS)
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
            cv2.circle(self.canvas, center, radius, COLOR_BRUSH, BRUSH_THICKNESS)
        else:
            cv2.polylines(self.canvas, [geometry], True, COLOR_BRUSH, BRUSH_THICKNESS)
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
            color = (255, 255, 255) if tool == "PAINT" else (0, 0, 255)
            radius = 6 if tool == "PAINT" else ERASER_THICKNESS // 2
            cv2.circle(display, cursor_pos, radius, color, 2)
        return display
