import cv2
import math
import platform
import time
from modules.portrait import PortraitPainter
from threading import RLock
from core.hand_tracker import HandTracker
from core.canvas_engine import CanvasEngine
from core.gesture_gate import PinchDrawGate, NumberGestureSelector, ViewGestureGate, pinch_ratio
from core.tool_catalog import TOOLS, PALETTE, palette_bgr, tool_label
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
        self.view_gate = ViewGestureGate()
        self.number_selector = NumberGestureSelector()
        self.stroke_locked = False
        self.portrait = PortraitPainter()
        self.victory_since = None
        self.victory_latched = False
        self.drag = None
        self.message = "左手数字手势保持 0.5 秒选工具；清屏需 1.2 秒。"

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
            tool = "VIEW" if tool in ("PAN", "ZOOM") else tool
            self.portrait.cancel(self.canvas_engine.canvas)
            self.victory_since = None
            self.victory_latched = False
            self.finish_stroke()
            self.drag = None
            self.view_gate.reset()
            self.stroke_locked = True  # 右手松开后才执行新工具
            if tool == "CLEAR":
                self.canvas_engine.clear()
                state.tool = "PAINT"
                self.message = "已清空整张大画板。"
            else:
                state.tool = tool
                self.message = {"PAINT": "右手捏合绘画，松开结束。", "ERASE": "右手捏合擦除。",
                    "VIEW": "拇指＋食指捏合移动；拇指＋中指捏合上拖放大、下拖缩小。切换前先松开两指。",
                    "PORTRAIT": "面向镜头，右手保持胜利手势约 0.6 秒拍摄；黑金/灰度只画头部，梵高/毕加索绘制全画面，可取消。",
                    "COLOR": "右手指向大色块并捏合，或用鼠标点击；选完返回画笔。"}[tool]
            state.eyedropper_active = False

    def select_color(self, index):
        with self.lock:
            if not 0 <= index < len(PALETTE):
                return
            self.finish_stroke()
            state.color = palette_bgr(index)
            state.tool = "PAINT"
            state.eyedropper_active = False
            self.stroke_locked = True
            self.message = f"已选择{PALETTE[index][0]}色 / {PALETTE[index][1]}，松开右手后可绘画。"

    def click_local(self, pos):
        with self.lock:
            if state.tool == "COLOR":
                index = self.viewport.palette_hit(pos)
                if index is not None:
                    self.select_color(index)

    def set_zoom(self, value):
        with self.lock:
            if self.portrait.busy:
                self.cancel_portrait()
            self.finish_stroke()
            self.stroke_locked = True
            self.drag = None
            self.view_gate.reset()
            self.viewport.set_zoom(value)

    def cancel_portrait(self):
        with self.lock:
            was_active = self.portrait.busy
            if was_active or state.tool == "PORTRAIT":
                self.select_tool("PAINT")
                self.message = "已取消肖像绘制，恢复绘制前内容。" if was_active else "已退出肖像工具，完成的肖像保留。"

    def step(self, frame, left_hand, right_hand, raw_frame=None):
        with self.lock:
            now = time.monotonic()
            number = self.number_selector.update(left_hand, now)
            if number is not None:
                selected = TOOLS[number - 1][0]
                if selected == "CLEAR" or selected != state.tool:
                    self.select_tool(selected)
            if state.tool == "PORTRAIT":
                now = time.monotonic()
                victory = self.portrait.victory(right_hand)
                if not victory:
                    self.victory_since = None
                    self.victory_latched = False
                elif not self.victory_latched and not self.portrait.busy:
                    if self.victory_since is None:
                        self.victory_since = now
                    if now - self.victory_since >= .6:
                        self.victory_latched = True
                        try:
                            source = raw_frame if raw_frame is not None else frame
                            self.portrait.start(source, self.canvas_engine.canvas, self.viewport.bounds(), now, style=state.portrait_style, strength=state.portrait_strength)
                            self.message = "肖像绘制中，可点击取消绘制或选择其他工具。"
                        except ValueError as exc:
                            self.message = str(exc)
                if self.portrait.job is not None:
                    try:
                        self.message = self.portrait.poll_generation(self.canvas_engine.canvas, now)
                    except ValueError as exc:
                        self.message = str(exc)
                        self.victory_latched = True
                if self.portrait.active:
                    done = self.portrait.advance(self.canvas_engine.canvas, now)
                    self.message = "肖像绘制完成；放下胜利手势后可重新触发，或选择画笔。" if done else "肖像绘制中 {}%（可取消）".format(self.portrait.percent)
                    if done:
                        self.victory_latched = True
                        self.portrait.cursor = None
            view_mode = self.view_gate.update(right_hand) if state.tool == "VIEW" else None
            active = self.draw_gate.update(right_hand) if state.tool != "VIEW" else view_mode is not None
            if not active:
                if right_hand is None or pinch_ratio(right_hand) >= self.draw_gate.stop_ratio:
                    self.stroke_locked = False
                self.drag = None
                self.finish_stroke()
            cursor = right_hand["index_pos"] if right_hand else None
            if right_hand is not None:
                right_hand["draw_active"] = active and not self.stroke_locked and state.tool in ("PAINT", "ERASE")
            self.filter_processor.process(frame, self.canvas_engine.canvas, left_hand, right_hand, state)
            if active and (not self.stroke_locked or state.tool == "VIEW") and cursor:
                if state.tool in ("PAINT", "ERASE"):
                    cursor = right_hand.get("filtered_index_pos", cursor)
                    self.canvas_engine.draw_or_erase(self.viewport.to_world(cursor), state.tool, state.color)
                    if (state.tool == "PAINT" and state.enable_shape and self.canvas_engine.is_closed_candidate()
                            and self.shape_processor.recognize(self.canvas_engine.trajectory)):
                        self.finish_stroke()
                        self.stroke_locked = True
                elif state.tool == "COLOR":
                    index = self.viewport.palette_hit(cursor)
                    if index is not None:
                        self.select_color(index)
                elif state.tool == "VIEW":
                    if view_mode == "ZOOM":
                        p = right_hand["landmarks"]
                        cursor = tuple((p[4][i] + p[12][i]) / 2 for i in (0, 1))
                    if self.drag is None:
                        self.drag = (cursor, self.viewport.cx, self.viewport.cy, self.viewport.zoom, self.viewport.bounds())
                    origin, cx, cy, zoom, bounds = self.drag
                    if view_mode == "PAN":
                        self.viewport.cx = cx - (cursor[0] - origin[0]) * bounds[2] / W
                        self.viewport.cy = cy - (cursor[1] - origin[1]) * bounds[3] / H
                        self.viewport.clamp()
                    else:
                        self.viewport.set_zoom(zoom * math.exp((origin[1] - cursor[1]) / 180))
            local, overview = self.viewport.render(
                self.canvas_engine.canvas, None if state.tool == "PORTRAIT" else cursor,
                state.tool, state.color, self.portrait.cursor if self.portrait.active else None)
            b, g, r = state.color
            hex_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
            color_name = next((f"{cn} / {en}" for cn, en, code in PALETTE if code.lower() == hex_color), hex_color)
            action = {"PAN": "移动中", "ZOOM": "缩放中"}.get(view_mode, "等待捏合") if state.tool == "VIEW" else ""
            status = f"{action} {tool_label(state.tool)} · {self.message} · {self.viewport.zoom:.2f}×"
            candidate = self.number_selector.candidate
            gesture_status = "左手掌心朝向镜头，做数字手势选择工具。"
            if candidate is not None:
                label = tool_label(TOOLS[candidate - 1][0])
                percent = round(self.number_selector.progress * 100)
                gesture_status = f"{label} · 确认进度 {percent}%" if percent < 100 else f"{label} · 已确认，保持不重复触发"
            return (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), cv2.cvtColor(local, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(overview, cv2.COLOR_BGR2RGB), status,
                    (hex_color, color_name), gesture_status, self.viewport.zoom, state.tool)

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
            with self.lock:
                self.portrait.cancel(self.canvas_engine.canvas)
                self.portrait.head.close()
