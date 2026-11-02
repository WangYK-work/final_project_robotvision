from core.processor import BaseProcessor
import cv2
import numpy as np

class ShapeRecognizerProcessor(BaseProcessor):
    def process(self, frame, canvas, left_hand, right_hand, state):
        points = state.custom_data.pop("completed_stroke", None)
        if points and state.enable_shape:
            state.custom_data["recognized_shape"] = self.recognize(points)
        return frame, canvas

    @staticmethod
    def recognize(points):
        if len(points) < 12:
            return None
        contour = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
        x, y, w, h = cv2.boundingRect(contour)
        diagonal = float(np.hypot(w, h))
        if w < 30 or h < 30 or np.linalg.norm(contour[0, 0] - contour[-1, 0]) > 0.22 * diagonal:
            return None
        perimeter = cv2.arcLength(contour, True)
        area = abs(cv2.contourArea(contour))
        if perimeter < 100 or area < 900 or area < 0.30 * w * h:
            return None
        vertices = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
        if not cv2.isContourConvex(vertices):
            return None
        if len(vertices) == 3:
            return "triangle", vertices
        if len(vertices) == 4:
            return "rectangle", cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32).reshape(-1, 1, 2)
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity >= 0.72 and 0.65 <= w / h <= 1.54:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            return "circle", ((round(cx), round(cy)), round(radius))
        return None
