import gradio as gr
from core.pipeline import VideoPipeline

def create_ui():
    pipeline = VideoPipeline()
    custom_css = ".gradio-container { max-width: 98% !important; } #video, #canvas { height: 65vh !important; }"

    with gr.Blocks(title="Smart AirBoard", css=custom_css) as demo:
        gr.Markdown("# 🎨 Smart AirBoard - 插件式架构")
        with gr.Row():
            out_video = gr.Image(label="摄像头", elem_id="video")
            out_canvas = gr.Image(label="画板", elem_id="canvas")
        out_state = gr.Textbox(label="当前工具")
        
        demo.load(pipeline.process_frame, outputs=[out_video, out_canvas, out_state])
    return demo