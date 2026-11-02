"""Head cutout and progressive black/gold illustration drawing; no external model download."""
import time
import cv2
import numpy as np


class PortraitPainter:
    def __init__(self, duration=24.0):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(path)
        from modules.head_portrait import HeadPortrait
        self.head = HeadPortrait()
        self.duration = duration
        self.active = False
        self.cursor = None
        self.progress = 0
        self.last_time = None

    @staticmethod
    def victory(hand):
        if hand is None or list(hand['fingers']) != [True, True, False, False]:
            return False
        points = hand['landmarks']
        palm = max(1.0, float(np.linalg.norm(np.subtract(points[5], points[17]))))
        return np.linalg.norm(np.subtract(points[8], points[12])) / palm > .25

    def start(self, frame, canvas, bounds, now=None, style="gold"):
        if self.detector.empty():
            raise ValueError('人脸检测器加载失败，请检查 OpenCV 安装。')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            raise ValueError('未检测到正面人脸，请面向摄像头、改善光线，放下胜利手势后重试。')
        face = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        image, alpha = self.head.extract(frame, face)
        art = self.head.stylize(image, alpha, style)
        # Portrait placement is independent of the local viewport.
        self.begin_image(art, canvas, (0, 0, canvas.shape[1], canvas.shape[0]), now, alpha)

    def begin_image(self, image, canvas, bounds, now=None, alpha=None):
        x, y, width, height = bounds
        scale = min(width * .90 / image.shape[1], height * .92 / image.shape[0])
        tw, th = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
        resized = cv2.resize(image, (tw, th), interpolation=cv2.INTER_NEAREST)
        self.target = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR) if resized.ndim == 2 else resized
        self.x, self.y = x + (width - tw) // 2, y + (height - th) // 2
        self.backup = canvas[self.y:self.y + th, self.x:self.x + tw].copy()
        self.alpha = np.ones((th, tw), np.float32) if alpha is None else cv2.resize(alpha, (tw, th), interpolation=cv2.INTER_LINEAR)
        self.alpha = np.clip(self.alpha, 0, 1)
        self.target = np.round(self.target * self.alpha[:, :, None] + self.backup * (1 - self.alpha[:, :, None])).astype(np.uint8)
        rows = np.repeat(np.arange(th), tw)
        cols = np.tile(np.arange(tw), th).reshape(th, tw)
        cols[1::2] = cols[1::2, ::-1]
        cols = cols.reshape(-1)
        keep = self.alpha[rows, cols] > .01
        self.rows, self.cols = rows[keep], cols[keep]
        if not len(self.rows):
            raise ValueError('未提取到头部，请重新拍摄。')
        self.progress = 0
        self.fraction = 0.0
        self.active = True
        self.cursor = None
        self.last_time = time.monotonic() if now is None else now

    def advance(self, canvas, now=None):
        if not self.active:
            return False
        now = time.monotonic() if now is None else now
        dt = min(.15, max(0.0, now - self.last_time))
        self.last_time = now
        self.fraction += dt * len(self.rows) / self.duration
        end = min(len(self.rows), int(self.fraction))
        if end > self.progress:
            rr, cc = self.rows[self.progress:end], self.cols[self.progress:end]
            canvas[self.y + rr, self.x + cc] = self.target[rr, cc]
            self.cursor = (int(self.x + cc[-1]), int(self.y + rr[-1]))
            self.progress = end
        self.active = self.progress < len(self.rows)
        return not self.active

    def cancel(self, canvas):
        if self.active:
            th, tw = self.backup.shape[:2]
            canvas[self.y:self.y + th, self.x:self.x + tw] = self.backup
        self.active = False
        self.cursor = None

    @property
    def percent(self):
        return round(100 * self.progress / len(self.rows)) if hasattr(self, 'rows') else 0
