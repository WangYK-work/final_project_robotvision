class AppState:
    def __init__(self):
        self.tool = "PAINT"       # 工具键定义见 core/tool_catalog.py
        self.color = (0, 0, 0)  # 当前画笔颜色
        self.eyedropper_active = False  # 兼容旧接口，吸管入口已由固定色板替代
        self.portrait_style = "gold"  # gold 或 grayscale，下一次拍摄生效
        self.portrait_strength = 0.3  # 神经画风强度，下一次拍摄生效
        self.filter_mode = "none" # 滤波模式
        self.enable_shape = True  # 是否开启图形识别
        self.custom_data = {}     # 自由扩展区：想加什么开关放这里

# 全局单例：所有文件只要 import state 就能读写同一份数据
state = AppState()