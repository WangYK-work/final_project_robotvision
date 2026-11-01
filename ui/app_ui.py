import gradio as gr
from core.pipeline import VideoPipeline
from core.state import state

def create_ui():
    pipeline = VideoPipeline()
    custom_css = ".gradio-container { max-width: 98vw !important; }"

    with gr.Blocks(title="Smart AirBoard", css=custom_css) as demo:
        gr.Markdown("# 🎨 Smart AirBoard - 插件式架构")
        gr.Markdown("左手伸出食指/食指加中指切换画笔/橡皮擦；左手伸出食指、中指、无名指清屏。右手先移动食指定位，拇指与食指捏合开始绘画，松开结束；闭合图形会自动结束，松开捏合后可画下一笔。")
        with gr.Row():
            out_video = gr.Image(label="摄像头", elem_id="video", height=360, scale=1, min_width=280)
            out_canvas = gr.Image(label="画板", elem_id="canvas", height=720, scale=3, min_width=640)
        out_state = gr.Textbox(label="当前工具")
        with gr.Row():
            filter_mode = gr.Radio(
                choices=[("原始", "none"), ("固定平滑", "fixed"), ("自适应平滑", "adaptive")],
                value=state.filter_mode, label="轨迹滤波模式"
            )
            enable_shape = gr.Checkbox(value=state.enable_shape, label="自动规整圆形、矩形、三角形")

        def set_filter_mode(value):
            state.filter_mode = value

        def set_enable_shape(value):
            state.enable_shape = value

        filter_mode.change(set_filter_mode, inputs=filter_mode)
        enable_shape.change(set_enable_shape, inputs=enable_shape)
        
        demo.load(pipeline.process_frame, outputs=[out_video, out_canvas, out_state])
    return demo
