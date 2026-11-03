# 本地神经风格迁移实验

这是可以独立运行的神经样例，也作为 7 号肖像的后台工作程序，不改变 virtual-painter 依赖。无需 API key，不上传照片。首次下载约 549 MiB 的官方 VGG19 权重，之后运行不需要网络。

## 在这台 Mac 上运行

在项目目录的终端执行：

```bash
conda activate ai-learning
# 已下载的文件会跳过；换电脑或缺少素材时执行
python experiments/neural_style/prepare.py
# 默认公开肖像 + 梵高《星月夜》，长边 256，200 步
python experiments/neural_style/transfer.py
# 换成自己的照片，路径有空格时保留引号
python experiments/neural_style/transfer.py --content "/路径/你的照片.jpg" --output experiments/neural_style/outputs/my_portrait.png
# 毕加索参考画
python experiments/neural_style/transfer.py --style experiments/neural_style/assets/picasso.jpg --output experiments/neural_style/outputs/picasso.png
```

首次运行的 Device 应显示 mps；auto 会在没有可用 GPU 时退回 CPU。支持 `--device mps` 强制要求 Apple GPU。这里是单张照片优化，不是实时视频滤镜。

输出位于 `outputs/`：结果 PNG、原图/画作/结果对照 JPG、包含参数/耗时/损失的 JSON。照片、下载的素材和大模型均已加入 Git 忽略列表。

实测数据见 [RESULTS.md](RESULTS.md)。

## 如何比较

- `--strength 0.3` / `1` / `3`：弱、中、强（调整风格损失权重，不是简单透明叠加）。
- `--size 256` / `512`：保持原比例，不强行拉成正方形。高分辨率更慢。
- `--steps 200` / `400`：更多优化步骤不保证更符合个人审美。
- `--style /路径/参考画.jpg`：支持任意本地参考画，包括莫奈；画风取决于参考图，不是一个艺术家名称标签。

建议固定照片、分辨率、步数，分别对比三档强度；记录风格可辨认程度、人物辨识度、耗时和失败例子。第一次运行有模型加载和 GPU 初始化开销，JSON 的 total_seconds 与 history 最后一项的优化耗时分开记录。

## 原理与范围

使用冻结的 ImageNet VGG19 特征提取网络，优化输出图片像素：内容损失保留人物结构，多个卷积层的 Gram 矩阵损失迁移纹理与颜色，少量总变分损失抑制噪点。不是重新训练 VGG，也不是大语言模型或扩散模型。标准 NST 能产生笔触/色彩变化，但不能保证毕加索式五官重排或保持人脸身份；参考画的构图也不会直接搬过来。

`NeuralStyleTransfer.run()` 提供 progress 回调和 cancelled 检查；输入为 0–1 RGB tensor，输出为 RGB PIL 图片与损失日志。7 号已通过 modules/neural_portrait.py 调用独立进程，避免在摄像头逐帧锁里优化；生成结束后转 BGR，使用完整画面、不使用头部 alpha，再交给 PortraitPainter.begin_image。已用公开照片验证完整流程，真实摄像头拍摄仍需实机体验。

## 来源

- 方法参考：[PyTorch 官方神经风格迁移教程](https://docs.pytorch.org/tutorials/advanced/neural_style_tutorial.html)。本样例另行实现 Adam 优化、MPS 选择和日志输出。
- 官方预训练权重：[VGG19](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth)，校验 SHA256 前缀 dcbb9e9d；以 weights_only=True 加载。
- 公开肖像：scikit-image 的 astronaut.png（NASA 宇航员 Eileen Collins），[来源与说明](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)。仅用于测试，不代表用户照片。
- 梵高《星月夜》、毕加索 1907 自画像参考图：[jcjohnson/neural-style 示例输入](https://github.com/jcjohnson/neural-style/tree/master/examples/inputs)。素材按源站条款使用，下载文件不随项目提交；公开展示或再分发时核对相应图片权利。
