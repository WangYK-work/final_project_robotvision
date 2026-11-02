import colorsys
import gradio as gr
from core.pipeline import VideoPipeline
from core.state import state
from config import W, H, TOOLBAR_W


def create_ui():
    pipeline = VideoPipeline()
    with gr.Blocks(title="Smart AirBoard", css=".gradio-container {max-width: 98vw !important;}") as demo:
        gr.Markdown("# Smart AirBoard")
        gr.Markdown("左手食指移动蓝色光标，到局部画板左侧按钮上捏合确认；松开后可再次选择。右手食指定位，捏合执行当前工具。\n\n"
                    "工具：1 画笔 · 2 橡皮 · 3 清空大画板 · 4 缩放 · 5 移动视图 · 6 吸管 · 7 肖像（右手胜利手势保持 0.6 秒，去背景大头肖像）。缩放时右手捏合向上放大、向下缩小；移动时捏合拖动。")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=340):
                out_video = gr.Image(label="摄像头", height=280, interactive=False)
                out_local = gr.Image(label="局部画板 · 左侧工具栏", height=300, interactive=False)
            with gr.Column(scale=3, min_width=600):
                out_board = gr.Image(label="完整大画板 · 肖像绘制区域", height=780, interactive=False)
        cancel_portrait = gr.Button("取消肖像绘制（恢复本次绘制前内容）")
        cancel_portrait.click(pipeline.cancel_portrait, queue=False)
        out_state = gr.Textbox(label="当前状态", interactive=False)
        with gr.Row():
            color_picker = gr.ColorPicker(value="#000000", label="画笔颜色")
            zoom = gr.Slider(1, 8, value=2, step=.1, label="局部放大倍数（也可选 Zoom 后用右手操作）")
        portrait_style = gr.Radio(
            choices=[("黑金手绘（默认）", "gold"), ("灰度", "grayscale")],
            value=state.portrait_style, label="肖像风格（下一次拍摄生效；两种均去背景、只画大头）"
        )
        def set_portrait_style(value):
            with pipeline.lock:
                state.portrait_style = value
        portrait_style.change(set_portrait_style, inputs=portrait_style, queue=False)
        with gr.Accordion("HSB 精确调色", open=False):
            hue = gr.Slider(0, 360, value=0, step=1, label="色相 H")
            saturation = gr.Slider(0, 100, value=0, step=1, label="饱和度 S (%)")
            brightness = gr.Slider(0, 100, value=0, step=1, label="亮度 B (%)")

        def set_color(value):
            value = (value or "#000000").lstrip("#")
            r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
            with pipeline.lock:
                state.color = (b, g, r)
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            return h * 360, s * 100, v * 100

        def set_hsb(h, s, v):
            r, g, b = (round(c * 255) for c in colorsys.hsv_to_rgb((h % 360) / 360, s / 100, v / 100))
            with pipeline.lock:
                state.color = (b, g, r)
            return "#{:02x}{:02x}{:02x}".format(r, g, b)

        def set_zoom(value):
            with pipeline.lock:
                if pipeline.portrait.active:
                    pipeline.cancel_portrait()
                pipeline.finish_stroke()
                pipeline.stroke_locked = True
                pipeline.drag = None
                pipeline.viewport.set_zoom(value)

        def click_local(evt: gr.SelectData):
            with pipeline.lock:
                x, y = evt.index
                tool = pipeline.viewport.hit((x, y))
                if tool:
                    pipeline.select_tool(tool)
                elif state.tool == "PICK" and TOOLBAR_W <= x < TOOLBAR_W + W and 0 <= y < H:
                    pipeline.pick((x - TOOLBAR_W, y))

        color_picker.input(set_color, inputs=color_picker, outputs=[hue, saturation, brightness], queue=False)
        for slider in (hue, saturation, brightness):
            slider.input(set_hsb, inputs=[hue, saturation, brightness], outputs=color_picker, queue=False)
        zoom.input(set_zoom, inputs=zoom, queue=False)
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
        demo.load(pipeline.process_frame, outputs=[out_video, out_local, out_board, out_state, color_picker, hue, saturation, brightness, zoom])
    return demo
