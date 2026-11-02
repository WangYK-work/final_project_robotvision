import math


class PinchDrawGate:
    """Right-hand pinch starts a stroke; releasing it ends the stroke."""

    def __init__(self, start_ratio=0.45, stop_ratio=0.65, stable_frames=2):
        self.start_ratio = start_ratio
        self.stop_ratio = stop_ratio
        self.stable_frames = stable_frames
        self.active = False
        self.near_frames = 0

    def update(self, hand):
        if hand is None:
            self.active = False
            self.near_frames = 0
            return False

        landmarks = hand["landmarks"]
        palm_width = max(1.0, math.dist(landmarks[5], landmarks[17]))
        ratio = math.dist(landmarks[4], landmarks[8]) / palm_width
        if self.active:
            if ratio >= self.stop_ratio:
                self.active = False
                self.near_frames = 0
        else:
            self.near_frames = self.near_frames + 1 if ratio <= self.start_ratio else 0
            if self.near_frames >= self.stable_frames:
                self.active = True
                self.near_frames = 0
        return self.active
