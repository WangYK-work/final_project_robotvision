import cv2
import platform
from core.hand_tracker import HandTracker
from core.canvas_engine import CanvasEngine
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
        self.processors.append(FilterProcessor())
        self.processors.append(ShapeRecognizerProcessor())

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
                if left_hand:
                    fingers = left_hand["fingers"]
                    # 伸出食指(1)：画笔
                    if fingers[0] and not fingers[1]: 
                        state.tool = "PAINT"
                    # 伸出食指+中指(2)：橡皮擦
                    elif fingers[0] and fingers[1] and not fingers[2]: 
                        state.tool = "ERASE"

                # 5. 右手基础绘画
                cursor_pos = None
                if right_hand and right_hand["fingers"][0] and not right_hand["fingers"][1]:
                    cursor_pos = right_hand["index_pos"]
                    self.canvas_engine.draw_or_erase(cursor_pos, state.tool)
                else:
                    self.canvas_engine.draw_or_erase(None, state.tool)

                # 6. 生成带光标的预览画板
                display_canvas = self.canvas_engine.render_preview(cursor_pos, state.tool)

                # 7. 魔法时刻：执行所有插件
                for processor in self.processors:
                    frame, display_canvas = processor.process(frame, display_canvas, left_hand, right_hand, state)

                # 8. 输出给 Gradio 网页
                yield (
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    cv2.cvtColor(display_canvas, cv2.COLOR_BGR2RGB), 
                    state.tool
                )
        finally:
            cap.release()