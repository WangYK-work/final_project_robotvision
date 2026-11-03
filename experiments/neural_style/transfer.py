"""Local Gatys-style image optimization with frozen ImageNet VGG19 features.

No camera or web service; intended as a standalone experiment before integration.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps, ImageDraw
from torchvision.models import vgg19
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'models/vgg19-dcbb9e9d.pth'


def device_for(name):
    if name == 'auto':
        name = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS 不可用；请在 Mac 本机终端运行，或使用 --device cpu。')
    return torch.device(name)


def load_image(path, size, device):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        if min(image.size) < 32:
            raise ValueError('图片短边需至少 32 像素。')
        return pil_to_tensor(image).float().div(255).unsqueeze(0).to(device)


def gram(feature):
    _, channels, height, width = feature.shape
    flat = feature.reshape(channels, height * width)
    return flat @ flat.t() / (channels * height * width)


class NeuralStyleTransfer:
    """Reusable RGB PIL interface; callback and cancel support future background jobs."""
    def __init__(self, weights=WEIGHTS, device='auto'):
        self.device = device_for(device)
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f'缺少模型 {weights}，请先运行 prepare.py。')
        digest = hashlib.sha256()
        with weights.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if not digest.hexdigest().startswith('dcbb9e9d'):
            raise ValueError('VGG19 权重校验失败，请重新运行 prepare.py 下载。')
        # Load on CPU before moving only the feature extractor to GPU.
        network = vgg19(weights=None)
        network.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        self.features = network.features[:30].eval().requires_grad_(False).to(self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([.229, .224, .225], device=self.device).view(1, 3, 1, 1)

    def extract(self, image):
        value = (image - self.mean) / self.std
        result = {}
        # relu1_1, relu2_1, relu3_1, relu4_1, relu4_2(content), relu5_1
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 6, 11, 20, 22, 29):
                result[index] = value.clone()
        return result

    def run(self, content, style, steps=200, strength=1.0, progress=None, cancelled=None):
        if steps < 1 or not 0 < strength <= 10:
            raise ValueError('steps 必须为正数，strength 需在 (0, 10]。')
        content, style = content.to(self.device), style.to(self.device)
        with torch.no_grad():
            content_features = self.extract(content)
            style_features = self.extract(style)
            targets = {k: gram(v) for k, v in style_features.items() if k != 22}
        output = content.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([output], lr=.02)
        history = []
        start = time.perf_counter()
        for step in range(1, steps + 1):
            if cancelled and cancelled():
                raise InterruptedError('已取消风格迁移。')
            optimizer.zero_grad(set_to_none=True)
            features = self.extract(output)
            content_loss = (features[22] - content_features[22]).square().mean()
            style_loss = sum((gram(features[k]) - target).square().mean() for k, target in targets.items()) / len(targets)
            tv = (output[:, :, 1:] - output[:, :, :-1]).abs().mean() + (output[:, :, :, 1:] - output[:, :, :, :-1]).abs().mean()
            loss = content_loss + strength * 1e6 * style_loss + .05 * tv
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                output.clamp_(0, 1)
            if step == 1 or step % 25 == 0 or step == steps:
                entry = dict(step=step, total=float(loss.detach().cpu()), content=float(content_loss.detach().cpu()), style=float(style_loss.detach().cpu()), elapsed_seconds=round(time.perf_counter()-start, 3))
                history.append(entry)
                if progress:
                    progress(entry)
        image = to_pil_image(output.detach().cpu().squeeze(0))
        return image, history


def main():
    parser = argparse.ArgumentParser(description='本地神经风格迁移：肖像 + 参考画作')
    parser.add_argument('--content', type=Path, default=ROOT / 'assets/portrait.png')
    parser.add_argument('--style', type=Path, default=ROOT / 'assets/van_gogh.jpg')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/van_gogh.png')
    parser.add_argument('--size', type=int, choices=(128, 256, 384, 512), default=256)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.output.resolve() in (args.content.resolve(), args.style.resolve()):
        parser.error('输出路径不能覆盖输入图片。')
    start = time.perf_counter()
    device = device_for(args.device)
    print(f'Device: {device}; size: {args.size}; steps: {args.steps}', flush=True)
    content, style = (load_image(p, args.size, device) for p in (args.content, args.style))
    engine = NeuralStyleTransfer(device=str(device))
    image, history = engine.run(content, style, args.steps, args.strength,
                                progress=lambda event: print(json.dumps(event), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    preview = Image.new('RGB', (args.size * 3, args.size + 30), 'white')
    draw = ImageDraw.Draw(preview)
    for i, (label, item) in enumerate(zip(('Content', 'Style reference', 'Neural result'), (to_pil_image(content.cpu().squeeze(0)), to_pil_image(style.cpu().squeeze(0)), image))):
        item = ImageOps.contain(item, (args.size, args.size))
        preview.paste(item, (i*args.size + (args.size-item.width)//2, 30))
        draw.text((i*args.size+8, 8), label, fill='black')
    preview.save(args.output.with_name(args.output.stem + '_comparison.jpg'))
    report = dict(device=str(device), size=args.size, steps=args.steps, strength=args.strength,
                  total_seconds=round(time.perf_counter()-start, 3), content=str(args.content), style=str(args.style), history=history)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
