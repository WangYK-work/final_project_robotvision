import cv2
import platform
from core.hand_tracker import HandTracker
from core.canvas_engine import CanvasEngine
from core.gesture_gate import PinchDrawGate
from core.state import state
from config import W, H

class VideoPipeline:
    def __init__(self):
        self.tracker = HandTracker()
        self.canvas_engine = CanvasEngine()
        self.processors = [] # 插件列表

        # 注册插件（新插件全往这里加）
        from modules.filter import FilterProcessor
        from modules.recognizer import ShapeRecognizerProcessor
        self.filter_processor = FilterProcessor()
        self.shape_processor = ShapeRecognizerProcessor()
        self.processors.extend([self.filter_processor, self.shape_processor])
        self.clear_count = 0
        self.clear_latched = False
        self.stroke_locked = False
        self.draw_gate = PinchDrawGate()

    def process_frame(self):
        # 1. 跨平台自动选择摄像头后端
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows
        elif platform.system() == "Linux":
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)   # Linux / WSL
        else:
            cap = cv2.VideoCapture(0)                 # Mac 等其他系统

        # 2. 设置通用摄像头参数
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        try:
            while True:
                ret, frame = cap.read()
                if not ret: 
                    break
                
                # 画面水平翻转，符合直觉
                frame = cv2.flip(frame, 1)
                
                # 确保画面尺寸与画布一致
                if frame.shape[1] != W or frame.shape[0] != H:
                    frame = cv2.resize(frame, (W, H))

                # 3. 提取手部数据 (感知)
                frame, left_hand, right_hand = self.tracker.process(frame)

                # 4. 左手状态机控制
                clear_gesture = False
                if left_hand:
                    fingers = left_hand["fingers"]
                    clear_gesture = fingers[:3] == [True, True, True] and not fingers[3]
                    # 伸出食指(1)：画笔
                    if not clear_gesture and fingers[0] and not fingers[1]:
                        state.tool = "PAINT"
                    # 伸出食指+中指(2)：橡皮擦
                    elif not clear_gesture and fingers[0] and fingers[1] and not fingers[2]:
                        state.tool = "ERASE"
                self.clear_count = self.clear_count + 1 if clear_gesture else 0
                if not clear_gesture:
                    self.clear_latched = False
                clear_now = self.clear_count >= 3 and not self.clear_latched
                if clear_now:
                    self.clear_latched = True
                    self.canvas_engine.clear()
                    self.filter_processor.reset()
                    state.custom_data.pop("completed_stroke", None)
                    state.custom_data.pop("recognized_shape", None)

                # 5. 右手先悬停定位，捏合后才允许落笔
                right_drawing = self.draw_gate.update(right_hand)
                if right_hand is not None:
                    right_hand["draw_active"] = right_drawing
                if not right_drawing:
                    self.stroke_locked = False
                if clear_now and right_drawing:
                    self.stroke_locked = True

                # 滤波必须在持久画布绘图之前执行
                frame, _ = self.filter_processor.process(
                    frame, self.canvas_engine.canvas, left_hand, right_hand, state
                )

                # 切换到橡皮擦时，先结束上一笔画笔轨迹
                completed = []
                if state.tool != "PAINT" and self.canvas_engine.trajectory:
                    completed = self.canvas_engine.draw_or_erase(None, "PAINT")

                # 6. 右手基础绘画
                cursor_pos = right_hand["index_pos"] if right_hand else None
                if right_drawing and not self.stroke_locked:
                    cursor_pos = right_hand["filtered_index_pos"]
                    # 已完成的画笔笔画需先规整，下一帧再执行擦除。
                    if not (completed and state.tool == "ERASE"):
                        self.canvas_engine.draw_or_erase(cursor_pos, state.tool)
                    if (state.tool == "PAINT" and state.enable_shape
                            and self.canvas_engine.is_closed_candidate()
                            and self.shape_processor.recognize(self.canvas_engine.trajectory)):
                        completed = self.canvas_engine.draw_or_erase(None, "PAINT")
                        self.stroke_locked = True
                else:
                    completed = completed or self.canvas_engine.draw_or_erase(None, "PAINT")

                # 7. 笔画结束后识别，并替换持久画布中的原始笔迹
                if completed:
                    state.custom_data["completed_stroke"] = completed
                frame, _ = self.shape_processor.process(
                    frame, self.canvas_engine.canvas, left_hand, right_hand, state
                )
                shape = state.custom_data.pop("recognized_shape", None)
                if shape is not None:
                    self.canvas_engine.replace_last_stroke(shape)
                display_canvas = self.canvas_engine.render_preview(cursor_pos, state.tool)

                # 8. 输出给 Gradio 网页
                yield (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    cv2.cvtColor(display_canvas, cv2.COLOR_BGR2RGB), 
                    state.tool + ("（松开捏合后可再画）" if self.stroke_locked else ("（绘制中）" if right_drawing else "（悬停定位，捏合落笔）"))
                )
        finally:
            cap.release()
