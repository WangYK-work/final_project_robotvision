from core.processor import BaseProcessor
import math

class FilterProcessor(BaseProcessor):
    def __init__(self):
        self.prev_pos = None
        self.prev_mode = None

    def reset(self):
        self.prev_pos = None

    def process(self, frame, canvas, left_hand, right_hand, state):
        mode = state.filter_mode
        if mode != self.prev_mode:
            self.reset()
            self.prev_mode = mode

        drawing = right_hand is not None and right_hand.get("draw_active", False)
        if not drawing:
            self.reset()
            return frame, canvas

        raw = right_hand["index_pos"]
        if mode == "none" or self.prev_pos is None:
            pos = (float(raw[0]), float(raw[1]))
        else:
            speed = math.dist(raw, self.prev_pos)
            alpha = 0.35 if mode == "fixed" else min(0.85, max(0.18, speed / 30.0))
            pos = tuple(self.prev_pos[i] + alpha * (raw[i] - self.prev_pos[i]) for i in range(2))
        self.prev_pos = pos
        right_hand["filtered_index_pos"] = tuple(round(v) for v in pos)
        return frame, canvas
