# 最简训练与推理指南

以下命令在**项目根目录**执行，使用带 CUDA GPU 的 Linux、WSL 或 Bash 环境。

## 1. 准备环境和配置

```bash
pip install -r RS_onestep/requirements.txt
cp RS_onestep/config.example.json RS_onestep/config.json
```

把训练用的高清图像放入 `data/train_hr/`；可以有子目录，支持 JPG、JPEG、PNG。打开 `RS_onestep/config.json`，至少确认这些字段：

```text
"train_dir": "data/train_hr",
"output_dir": "runs/rs_onestep",
"sd21": "stabilityai/stable-diffusion-2-1-base",
"taesd": "madebyollin/taesd",
"siglip2": "google/siglip2-so400m-patch16-512"
```

模型字段可以保留示例 ID，也可以改成已经下载好的本地模型目录。只需提供高清训练图，程序会自动生成退化的训练输入。

## 2. 开始训练

```bash
bash RS_onestep/train.sh RS_onestep/config.json
```

训练检查点保存在 `runs/rs_onestep/checkpoints/`。训练结束后使用 `final.pt`；训练中也可以使用 `step-XXXXXXX.pt`。从检查点继续训练：

```bash
bash RS_onestep/train.sh RS_onestep/config.json --resume runs/rs_onestep/checkpoints/step-0001000.pt
```

时间步为 800 的旧检查点不能用于当前时间步 500 的模型；请使用本版本训练生成的检查点。

## 3. 推理

把待处理的**原始低分辨率图像**放入 `data/test_lr/`，默认输出尺寸为输入的 4 倍：

```bash
bash RS_onestep/infer.sh --checkpoint runs/rs_onestep/checkpoints/final.pt --input data/test_lr --output outputs
```

`--input` 也可指向单张图片。如果输入图已经预先放大到目标尺寸，命令末尾加 `--upscale 1`。

当前训练和推理使用同一个时间步 **500**；生成器损失为 `1×潜变量 MSE + 2×VGG LPIPS + gan_weight×对抗损失`，不使用 FDL。
