import colorsys
import gradio as gr
from core.pipeline import VideoPipeline
from core.state import state
from core.tool_catalog import UI_TOOLS, tool_label


def create_ui():
    pipeline = VideoPipeline()
    with gr.Blocks(title="Smart AirBoard", css=".gradio-container {max-width: 98vw !important;}") as demo:
        gr.Markdown("# Smart AirBoard")
        gr.Markdown("左手数字手势选工具，保持 0.5 秒确认（清屏 1.2 秒）；右手捏合绘画、擦除或拖动。\n\n"
                    "1 画笔 · 2 橡皮 · 3 清屏 · 4/5 视图（拇指＋食指移动，拇指＋中指上下缩放） · 6 固定色板 · 7 肖像。下方保留色盘与 HSB 精确调色。")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                out_video = gr.Image(label="摄像头", height=280, interactive=False)
                out_local = gr.Image(label="局部画板 · 选色时点击色块", height=300, interactive=False)
            with gr.Column(scale=3, min_width=600):
                out_board = gr.Image(label="完整大画板 · 肖像绘制区域", height=780, interactive=False)
        buttons = []
        with gr.Row():
            for key, *_ in UI_TOOLS:
                button = gr.Button(tool_label(key), variant="primary" if state.tool == key else "secondary", min_width=110)
                button.click(lambda chosen=key: pipeline.select_tool(chosen), queue=False)
                buttons.append(button)
        gesture_status = gr.Markdown("左手掌心朝向镜头，做数字手势选择工具。")
        cancel_portrait = gr.Button("取消肖像绘制（恢复本次绘制前内容）")
        cancel_portrait.click(pipeline.cancel_portrait, queue=False)
        out_state = gr.Textbox(label="当前状态", interactive=False)
        with gr.Row():
            color_picker = gr.ColorPicker(value="#000000", label="画笔颜色")
            zoom = gr.Slider(1, 8, value=2, step=.1, label="局部放大倍数（也可选视图后捏合中指上下拖动）")
        portrait_style = gr.Radio(
            choices=[("黑金手绘（默认）", "gold"), ("灰度", "grayscale"), ("梵高 · 全画面", "van_gogh"), ("毕加索 · 全画面", "picasso")],
            value=state.portrait_style, label="肖像风格（下一次拍摄生效；黑金/灰度只画头部；梵高/毕加索保留背景）"
        )
        portrait_strength = gr.Slider(.1, 3, value=state.portrait_strength, step=.1, label="神经画风强度（仅梵高/毕加索，下次拍摄生效；建议 0.3）")
        gr.Markdown("神经画风处理整张摄像头画面，保留人物和背景。选择 7 后，右手胜利手势拍照，后台生成后逐步绘制；可随时取消或切换工具。毕加索选项迁移色彩与纹理，不保证五官重排。")
        def set_portrait_strength(value):
            with pipeline.lock:
                state.portrait_strength = float(value)
        portrait_strength.input(set_portrait_strength, inputs=portrait_strength, queue=False)
        def set_portrait_style(value):
            with pipeline.lock:
                state.portrait_style = value
        portrait_style.change(set_portrait_style, inputs=portrait_style, queue=False)
        with gr.Accordion("HSB 精确调色", open=False):
            hue = gr.Slider(0, 360, value=0, step=1, label="色相 H")
            saturation = gr.Slider(0, 100, value=0, step=1, label="饱和度 S (%)")
            brightness = gr.Slider(0, 100, value=0, step=1, label="亮度 B (%)")

        manual_color = None

        def set_color(value):
            nonlocal manual_color
            value = (value or "#000000").lstrip("#")
            r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
            with pipeline.lock:
                pipeline.finish_stroke()
                state.color = (b, g, r)
                manual_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                pipeline.stroke_locked = True
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            return h * 360, s * 100, v * 100

        def set_hsb(h, s, v):
            nonlocal manual_color
            r, g, b = (round(c * 255) for c in colorsys.hsv_to_rgb((h % 360) / 360, s / 100, v / 100))
            with pipeline.lock:
                pipeline.finish_stroke()
                state.color = (b, g, r)
                manual_color = "#{:02x}{:02x}{:02x}".format(r, g, b)
                pipeline.stroke_locked = True
            return "#{:02x}{:02x}{:02x}".format(r, g, b)

        def click_local(evt: gr.SelectData):
            pipeline.click_local(evt.index)

        color_picker.input(set_color, inputs=color_picker, outputs=[hue, saturation, brightness], queue=False)
        for slider in (hue, saturation, brightness):
            slider.input(set_hsb, inputs=[hue, saturation, brightness], outputs=color_picker, queue=False)
        zoom.input(pipeline.set_zoom, inputs=zoom, queue=False)
        out_local.select(click_local, queue=False)
        with gr.Row():
            filter_mode = gr.Radio(choices=[("原始", "none"), ("固定平滑", "fixed"), ("自适应平滑", "adaptive")], value=state.filter_mode, label="轨迹滤波模式")
            enable_shape = gr.Checkbox(value=state.enable_shape, label="自动规整圆形、矩形、三角形")
        def set_filter(value):
            with pipeline.lock:
                state.filter_mode = value
        def set_shape(value):
            with pipeline.lock:
                state.enable_shape = value
        filter_mode.change(set_filter, inputs=filter_mode, queue=False)
        enable_shape.change(set_shape, inputs=enable_shape, queue=False)
        def stream():
            previous_color = previous_tool = previous_zoom = None
            for video, local, board, status, color, gesture, scale, tool in pipeline.process_frame():
                code = color[0]
                if code != previous_color and code != manual_color:
                    r, g, b = (int(code[i:i + 2], 16) / 255 for i in (1, 3, 5))
                    h, sat, val = colorsys.rgb_to_hsv(r, g, b)
                    color_outputs = (code, h * 360, sat * 100, val * 100)
                else:
                    color_outputs = (gr.skip(),) * 4
                zoom_output = scale if scale != previous_zoom else gr.skip()
                tool_outputs = [gr.update(variant="primary" if key == tool else "secondary") for key, *_ in UI_TOOLS] if tool != previous_tool else [gr.skip() for _ in UI_TOOLS]
                previous_color, previous_tool, previous_zoom = code, tool, scale
                yield video, local, board, status, *color_outputs, zoom_output, gesture, *tool_outputs

        demo.load(stream, outputs=[out_video, out_local, out_board, out_state, color_picker, hue, saturation, brightness, zoom, gesture_status, *buttons], show_progress="hidden")
    return demo
