class AppState:
    def __init__(self):
        self.tool = "PAINT"       # 当前工具：PAINT 或 ERASE
        self.color = (0, 0, 0)  # 当前画笔颜色
        self.eyedropper_active = False  # 点击画板取色，随后退出
        self.portrait_style = "gold"  # gold 或 grayscale，下一次拍摄生效
        self.filter_mode = "none" # 滤波模式
        self.enable_shape = True  # 是否开启图形识别
        self.custom_data = {}     # 自由扩展区：想加什么开关放这里

# 全局单例：所有文件只要 import state 就能读写同一份数据
state = AppState()