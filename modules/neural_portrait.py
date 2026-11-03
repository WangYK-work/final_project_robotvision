"""Isolated neural style worker; the camera environment never imports PyTorch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import cv2

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'experiments/neural_style'


class NeuralPortraitJob:
    def __init__(self, image, style, strength):
        if style not in ('van_gogh', 'picasso'):
            raise ValueError('未知神经画风。')
        interpreter = Path(os.environ.get('AIRBOARD_NEURAL_PYTHON', str(Path(sys.prefix).parent / 'ai-learning' / ('python.exe' if os.name == 'nt' else 'bin/python'))))
        if not interpreter.is_file():
            raise ValueError('未找到神经网络 Python，请设置 AIRBOARD_NEURAL_PYTHON 为已安装 PyTorch 的 Python 路径。')
        reference = EXPERIMENT / 'assets' / (style + '.jpg')
        if not reference.is_file() or not (EXPERIMENT / 'models/vgg19-dcbb9e9d.pth').is_file():
            raise ValueError('缺少画风素材或模型；请在 ai-learning 环境运行 experiments/neural_style/prepare.py。')
        self.temp = tempfile.TemporaryDirectory(prefix='airboard-portrait-')
        self.folder = Path(self.temp.name)
        self.output = self.folder / 'result.png'
        self.log_path = self.folder / 'progress.log'
        self.started = time.monotonic()
        self.percent = 0
        self.closed = False
        try:
            if not cv2.imwrite(str(self.folder / 'input.png'), image):
                raise ValueError('无法保存临时肖像。')
            self.log = self.log_path.open('wb')
            self.process = subprocess.Popen([
                str(interpreter), '-u', str(EXPERIMENT / 'transfer.py'),
                '--content', str(self.folder / 'input.png'), '--style', str(reference),
                '--output', str(self.output), '--size', '256', '--steps', '200',
                '--strength', str(float(strength)), '--device', 'auto',
            ], stdout=self.log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        except Exception as exc:
            if hasattr(self, 'log'):
                self.log.close()
            self.temp.cleanup()
            raise ValueError(f'无法启动神经肖像：{exc}') from exc

    def poll(self):
        """Return BGR image on completion, None while working; never wait for GPU."""
        if time.monotonic() - self.started > 300:
            self.cancel()
            raise ValueError('神经肖像处理超时（5 分钟），已取消；可使用黑金或灰度重试。')
        log = self.log_path.read_text(errors='replace')
        for line in log.splitlines():
            if line.startswith('{'):
                try:
                    self.percent = min(99, round(json.loads(line)['step'] / 2))
                except (ValueError, KeyError):
                    pass
        code = self.process.poll()
        if code is None:
            return None
        image = cv2.imread(str(self.output)) if code == 0 and self.output.exists() else None
        self.log.close()
        self.temp.cleanup()
        self.closed = True
        if image is None:
            detail = next((line for line in reversed(log.splitlines()) if line.strip()), '没有生成结果')
            raise ValueError('神经肖像生成失败：' + detail[-350:])
        return image

    def cancel(self):
        if self.closed:
            return
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
        # Reap and remove temporary photo files without blocking the camera lock.
        def cleanup():
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            finally:
                self.log.close()
                self.temp.cleanup()
        threading.Thread(target=cleanup, daemon=True).start()
