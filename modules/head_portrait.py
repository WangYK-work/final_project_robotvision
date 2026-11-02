"""Offline head cutout and black/gold ink illustration using existing MediaPipe."""
import cv2
import numpy as np


class HeadPortrait:
    def __init__(self):
        self.segmenter = None
        self.mesh = None

    def extract(self, frame, face):
        # Models are bundled with the project's pinned MediaPipe dependency.
        if self.segmenter is None:
            import mediapipe as mp
            self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
            self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=False)
        fx, fy, fw, fh = (int(v) for v in face)
        x0, y0 = max(0, fx - fw), max(0, fy - fh)
        x1, y1 = min(frame.shape[1], fx + 2 * fw), min(frame.shape[0], fy + 2 * fh)
        crop = frame[y0:y1, x0:x1].copy()
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        person = self.segmenter.process(rgb).segmentation_mask
        result = self.mesh.process(rgb)
        if person is None or not result.multi_face_landmarks:
            raise ValueError('未能稳定提取头部，请正面入镜、让头发与背景有明显区别后重试。')
        height, width = crop.shape[:2]
        points = np.array([(lm.x * width, lm.y * height) for lm in result.multi_face_landmarks[0].landmark])
        # Face jaw follows the chin and excludes neck/shoulders. The upper arc
        # extends beyond the forehead to include hair, then intersects person mask.
        jaw_ids = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152,
                   377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        jaw = points[jaw_ids]
        if jaw[0, 0] > jaw[-1, 0]:
            jaw = jaw[::-1]
        left, right = jaw[0], jaw[-1]
        cx = (left[0] + right[0]) / 2
        rx = max((right[0] - left[0]) / 2 + .20 * fw, .55 * fw)
        cy = min(left[1], right[1]) - .05 * fh
        top = min(points[10, 1] - .48 * fh, fy - y0 - .35 * fh)
        ry = max(.65 * fh, cy - top)
        angles = np.linspace(0, np.pi, 60)
        arc = np.column_stack([cx + rx * np.cos(angles), cy - ry * np.sin(angles)])
        polygon = np.concatenate([jaw, arc]).round().astype(np.int32)
        geometry = np.zeros((height, width), np.uint8)
        cv2.fillPoly(geometry, [polygon], 255)
        confidence = np.clip((person - .25) / .45, 0, 1)
        alpha = confidence * (geometry.astype(np.float32) / 255)
        alpha = cv2.GaussianBlur(alpha, (3, 3), .7)
        yy, xx = np.nonzero(alpha > .08)
        if len(xx) < fw * fh * .2:
            raise ValueError('头部前景不足，请改善光线并更换对比明显的背景。')
        xa, xb, ya, yb = max(0, xx.min()-2), min(width, xx.max()+3), max(0, yy.min()-2), min(height, yy.max()+3)
        return crop[ya:yb, xa:xb], alpha[ya:yb, xa:xb]

    @staticmethod
    def stylize(image, alpha, style="gold"):
        if style == "grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), .45)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if style != "gold":
            raise ValueError("未知肖像风格")
        smooth = cv2.bilateralFilter(image, 7, 55, 55)
        gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
        values = gray[alpha > .5]
        low, high = np.percentile(values, [3, 97]) if len(values) else (0, 255)
        tone = np.clip((gray.astype(np.float32) - low) / max(50, high - low), 0, 1)
        # Original sharp five-tone black/gold palette, including solid black
        # shadows. Only the artificial periodic hatching has been removed.
        palette = np.array([[35, 30, 26], [52, 92, 125], [77, 165, 211],
                            [160, 211, 237], [239, 248, 251]], np.uint8)
        levels = np.minimum(4, (tone * 5).astype(np.int32))
        art = palette[levels]
        edges = cv2.Canny(gray, 45, 110) > 0
        art[edges] = palette[0]
        return art

    def close(self):
        for model in (self.segmenter, self.mesh):
            if model is not None:
                model.close()
        self.segmenter = self.mesh = None
