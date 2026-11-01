class BaseProcessor:
    """所有功能插件的基类"""
    def process(self, frame, canvas, left_hand, right_hand, state):
        """
        插件处理接口（所有插件必须实现这个方法）
        
        参数:
            frame: 摄像头原图 (numpy array, BGR格式)
            canvas: 当前画板图 (numpy array, BGR格式)
            left_hand: 左手数据，无手则为 None
            right_hand: 右手数据，无手则为 None
            state: 全局状态对象（AppState）
            
        返回:
            frame, canvas
        """
        return frame, canvas