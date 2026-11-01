from core.processor import BaseProcessor
import cv2

class ShapeRecognizerProcessor(BaseProcessor):
    def process(self, frame, canvas, left_hand, right_hand, state):
        # C同学的逻辑写在这里
        # 识别逻辑在此实现
        return frame, canvas