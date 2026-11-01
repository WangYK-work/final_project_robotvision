import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        left_hand = None
        right_hand = None
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                self.mp_draw.draw_landmarks(frame, lms, self.mp_hands.HAND_CONNECTIONS)
                
                h, w = frame.shape[:2]
                # 1. 完整坐标 (21个点)
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lms.landmark]
                # 2. 手指状态
                fingers = [
                    lms.landmark[8].y < lms.landmark[6].y,
                    lms.landmark[12].y < lms.landmark[10].y,
                    lms.landmark[16].y < lms.landmark[14].y,
                    lms.landmark[20].y < lms.landmark[18].y
                ]
                # 3. 食指指尖
                index_pos = landmarks[8]
                
                hand_data = {"landmarks": landmarks, "fingers": fingers, "index_pos": index_pos}
                if label == 'Left': left_hand = hand_data
                elif label == 'Right': right_hand = hand_data
                
        return frame, left_hand, right_hand