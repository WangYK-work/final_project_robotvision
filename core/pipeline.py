import cv2
import platform
import math
import time
from modules.portrait import PortraitPainter
from threading import RLock
from core.hand_tracker import HandTracker
from core.canvas_engine import CanvasEngine
from core.gesture_gate import PinchDrawGate
from core.viewport import Viewport
from core.state import state
from config import W, H

class VideoPipeline:
    def __init__(self):
        self.tracker = HandTracker()
        self.canvas_engine = CanvasEngine()
        self.viewport = Viewport()
        self.lock = RLock()
        from modules.filter import FilterProcessor
        from modules.recognizer import ShapeRecognizerProcessor
        self.filter_processor = FilterProcessor()
        self.shape_processor = ShapeRecognizerProcessor()
        self.processors = [self.filter_processor, self.shape_processor]
        self.draw_gate = PinchDrawGate()
        self.left_gate = PinchDrawGate()
        self.left_down = False
        self.stroke_locked = False
        self.portrait = PortraitPainter()
        self.victory_since = None
        self.victory_latched = False
        self.drag = None
        self.last_color = None
        self.last_zoom = None
        self.message = "左手指向工具并捏合确认；右手捏合操作。"

    def finish_stroke(self):
        points = self.canvas_engine.draw_or_erase(None, "PAINT")
        if points and state.enable_shape:
            shape = self.shape_processor.recognize(points)
            if shape is not None:
                self.canvas_engine.replace_last_stroke(shape)
        self.canvas_engine.stroke_base = None
        self.filter_processor.reset()

    def select_tool(self, tool):
        with self.lock:
            self.portrait.cancel(self.canvas_engine.canvas)
            self.victory_since = None
            self.victory_latched = False
            self.finish_stroke()
            self.drag = None
            self.stroke_locked = True  # 右手松开后才执行新工具
            if tool == "CLEAR":
                self.canvas_engine.clear()
                state.tool = "PAINT"
                self.message = "已清空整张大画板。"
            else:
                state.tool = tool
                self.message = {"PAINT": "右手捏合绘画，松开结束。", "ERASE": "右手捏合擦除。",
                    "ZOOM": "右手捏合后向上移动放大、向下移动缩小。",
                    "PAN": "右手捏合并拖动局部画面，松开结束。",
                    "PORTRAIT": "面向镜头，右手保持胜利手势约 0.6 秒拍摄去背景大头肖像；可点击取消绘制。",
                    "PICK": "右手定位后捏合取色，或用鼠标点击局部画板。"}[tool]
            state.eyedropper_active = state.tool == "PICK"

    def pick(self, pos):
        world = self.viewport.to_world(pos)
        state.color = tuple(int(c) for c in self.canvas_engine.canvas[world[1], world[0]])
        state.tool = "PAINT"
        state.eyedropper_active = False
        self.stroke_locked = True
        self.message = "已取色，松开右手捏合后可继续绘画。"

    def cancel_portrait(self):
        with self.lock:
            was_active = self.portrait.active
            if was_active or state.tool == "PORTRAIT":
                self.select_tool("PAINT")
                self.message = "已取消肖像绘制，恢复绘制前内容。" if was_active else "已退出肖像工具，完成的肖像保留。"

    def step(self, frame, left_hand, right_hand, raw_frame=None):
        with self.lock:
            left_pos = self.viewport.left_cursor(left_hand)
            left_down = self.left_gate.update(left_hand)
            if left_down and not self.left_down:
                selected = self.viewport.hit(left_pos)
                if selected:
                    self.select_tool(selected)
            self.left_down = left_down
            if state.tool == "PORTRAIT":
                now = time.monotonic()
                victory = self.portrait.victory(right_hand)
                if not victory:
                    self.victory_since = None
                    self.victory_latched = False
                elif not self.victory_latched and not self.portrait.active:
                    if self.victory_since is None:
                        self.victory_since = now
                    if now - self.victory_since >= .6:
                        self.victory_latched = True
                        try:
                            source = raw_frame if raw_frame is not None else frame
                            self.portrait.start(source, self.canvas_engine.canvas, self.viewport.bounds(), now, style=state.portrait_style)
                            self.message = "肖像绘制中，可点击取消绘制或选择其他工具。"
                        except ValueError as exc:
                            self.message = str(exc)
                if self.portrait.active:
                    done = self.portrait.advance(self.canvas_engine.canvas, now)
                    self.message = "肖像绘制完成；放下胜利手势后可重新触发，或选择画笔。" if done else "肖像绘制中 {}%（可取消）".format(self.portrait.percent)
                    if done:
                        self.victory_latched = True
                        self.portrait.cursor = None
            active = self.draw_gate.update(right_hand)
            if not active:
                self.stroke_locked = False
                self.drag = None
                self.finish_stroke()
            cursor = right_hand["index_pos"] if right_hand else None
            if right_hand is not None:
                right_hand["draw_active"] = active and not self.stroke_locked and state.tool in ("PAINT", "ERASE")
            self.filter_processor.process(frame, self.canvas_engine.canvas, left_hand, right_hand, state)
            if active and not self.stroke_locked and cursor:
                if state.tool in ("PAINT", "ERASE"):
                    cursor = right_hand.get("filtered_index_pos", cursor)
                    self.canvas_engine.draw_or_erase(self.viewport.to_world(cursor), state.tool, state.color)
                    if (state.tool == "PAINT" and state.enable_shape and self.canvas_engine.is_closed_candidate()
                            and self.shape_processor.recognize(self.canvas_engine.trajectory)):
                        self.finish_stroke()
                        self.stroke_locked = True
                elif state.tool == "PICK":
                    self.pick(cursor)
                elif state.tool in ("PAN", "ZOOM"):
                    if self.drag is None:
                        self.drag = (cursor, self.viewport.cx, self.viewport.cy, self.viewport.zoom, self.viewport.bounds())
                    origin, cx, cy, zoom, bounds = self.drag
                    if state.tool == "PAN":
                        self.viewport.cx = cx - (cursor[0] - origin[0]) * bounds[2] / W
                        self.viewport.cy = cy - (cursor[1] - origin[1]) * bounds[3] / H
                        self.viewport.clamp()
                    else:
                        self.viewport.set_zoom(zoom * math.exp((origin[1] - cursor[1]) / 180))
            local, overview = self.viewport.render(self.canvas_engine.canvas, None if state.tool == "PORTRAIT" else cursor, left_pos, state.tool, state.color, self.portrait.cursor if self.portrait.active else None)
            b, g, r = state.color
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            import colorsys
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            status = "{} | {} | {} | {:.2f}x".format(state.tool, hex_color, self.message, self.viewport.zoom)
            color_outputs = (hex_color, h * 360, s * 100, v * 100) if self.last_color != state.color else ({"__type__": "update"},) * 4
            zoom_output = self.viewport.zoom if self.last_zoom != self.viewport.zoom else {"__type__": "update"}
            self.last_color = state.color
            self.last_zoom = self.viewport.zoom
            return (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(local, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overview, cv2.COLOR_BGR2RGB), status, *color_outputs, zoom_output)

    def process_frame(self):
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        cap = cv2.VideoCapture(0, backend)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(cv2.flip(frame, 1), (W, H))
                raw_frame = frame.copy()
                frame, left, right = self.tracker.process(frame)
                yield self.step(frame, left, right, raw_frame)
        finally:
            cap.release()
            self.portrait.head.close()
