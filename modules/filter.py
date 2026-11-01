from core.processor import BaseProcessor
import cv2

class FilterProcessor(BaseProcessor):
    def __init__(self):
        self.prev_pos = None

    def process(self, frame, canvas, left_hand, right_hand, state):
        # B同学的逻辑写在这里
        # 如果以后要改滤波算法，改这个文件。
        return frame, canvas