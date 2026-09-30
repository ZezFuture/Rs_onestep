# RS_onestep

SD2.1 单步 x4 图像超分。训练读取 HR 图像，并使用 `train_tiny.sh` 实际调用的 RealESRGAN 退化生成同尺寸 LR 条件图；生成器采用 HYPIR 的 LR 潜变量到单步 `x0` 路径，VAE 换成 TAESD，判别器换成指定的 SigLIP2 成对评分模型。

## 安装与权重

从包含 `RS_onestep` 的目录运行：

```bash
pip install -r RS_onestep/requirements.txt
cp RS_onestep/config.example.json RS_onestep/config.json
```

编辑 `config.json` 中的三类预训练模型路径。可以填本地 Diffusers/Transformers 模型目录，也可以填可下载的模型 ID：

| 字段 | 所需权重 |
| --- | --- |
| `sd21` | SD2.1 Diffusers 模型，须含 `unet/`、`text_encoder/`、`tokenizer/`、`scheduler/`。示例为 SD2.1 base。 |
| `taesd` | SD 潜空间版 TAESD `AutoencoderTiny`，示例 `madebyollin/taesd`；不能使用 TAESDXL。 |
| `siglip2` | 与指定参考评分器相同的 SigLIP2 视觉模型及 image processor。 |
| `reward_checkpoint` | 可选，指定参考文件的 `save_reward_model()` 产生的权重文件。若为 `null`，SigLIP2 主干仍加载预训练权重，成对融合模块和评分头随机初始化后参与 GAN 训练。所用 SigLIP2 主干须与该文件匹配。 |

`config.example.json` 的模型 ID 是示例来源，不包含权重。LPIPS 使用预训练 VGG 权重，首次使用时须能从依赖包指定来源取得权重，或预先准备好对应缓存。复制的 `realesrgan.py` 需要 BasicSR、OpenCV、PyYAML 和 torchvision。`basicsr_compat.py` 仅为 BasicSR 1.4.2 适配 torchvision 的模块导入路径，不改变退化计算。训练入口使用 Accelerate 和 CUDA，可使用单卡或多卡；`precision` 支持 `bf16` 或 `fp32`。

## 数据与真实退化

`train_dir` 指向一个 HR 图像根目录，或多个根目录组成的 JSON 数组；递归读取 JPG/JPEG/PNG，例如：

```text
data/train_hr/
  scene001.png
  subdir/scene002.jpg
```

不需要预制 LR。每次读取 HR 后，数据集按 `train_tiny.py` 使用的 `MyDataset_blind_plus(deterministic_k=False, resolution=512)` 裁切；任一边不足目标尺寸时以 OpenCV bicubic 调整到目标尺寸。随后调用复制的 `RealESRGAN_degradation.degrade_process(..., resize_bak=True)`：水平翻转、随机模糊核、第一阶段模糊/缩放/高斯或泊松噪声/JPEG、第二阶段可选模糊/缩放/噪声、随机顺序的 sinc 滤波与 JPEG、随机插值放回 HR 大小、8 位量化。操作顺序、概率和范围保留在 `params_realesrgan_seesr.yml` 中，默认倍率为退化实现里的 **4**。返回的 LR/HR 均是同尺寸 RGB、`[-1,1]`、`3×H×W` 张量。调退化强度请编辑 YAML；它是训练所加载的文件。

训练入口使用 `conditioning_pixel_values` 作为 LR、`output_pixel_values` 作为 HR。判别器直接接收两张图；生成器将 LR 编码为唯一输入潜变量。没有独立 noise latent。

## 单步模型及梯度

SD2.1 UNet 基础权重冻结，配置中的 `lora_modules` 和 `lora_rank` 控制可训练 LoRA；CLIP 文本编码器冻结，以同一 prompt 构造条件。实际传给 UNet 的时间步固定是整数 **500**；它是 1000 步训练噪声计划的下标，不是推理时间步列表的序号。代码从同一 scheduler 的 `alphas_cumprod[500]` 读取系数，按 `prediction_type` (`epsilon` 或 `v_prediction`) 将 UNet 输出还原成 `x0`。输入是 LR 编码潜变量本身。训练和推理调用同一个 `OneStepSR.forward()`。HYPIR-main 自带的 SD2 训练和推理配置均为 `model_t=200, coeff_t=200`；本项目按当前要求使用 500。

TAESD 输入/输出图像范围是 `[-1,1]`，潜变量为 4 通道、空间缩小 8 倍。代码验证所载模型的通道、空间倍率、`scaling_factor=1.0` 和 `shift_factor=0.0`。TAESD 编码输出直接对接 SD2.1 的 4 通道 UNet，不套用 SD2.1 原始重型 VAE 的 `0.18215` 缩放；`x0` 再由 TAESD 解码回 RGB。`latent_magnitude` 与 `latent_shift` 是 TAESD 将潜变量存成图像时使用的映射，不属于本训练/推理计算图。

LoRA 的默认 rank 为 64、alpha 为 128、dropout 为 0，对应 Z-Image 实际训练脚本；这些值由配置中的 `lora_rank`、`lora_alpha` 和 `lora_dropout` 控制。

有两份独立的 TAESD 编码器：

1. `taesd.encoder` 从预训练 TAESD 初始化，完整训练，编码 LR，进入 UNet 和生成器优化器。
2. `reference_encoder` 是预训练编码器的独立深拷贝，冻结且不进入优化器，只对 HR 编码以提供 latent MSE 监督。

TAESD 解码器冻结，但解码操作保留对输入潜变量的梯度；解码结果裁剪到 `[-1,1]` 后，LPIPS 和对抗损失仍能反传到 LoRA 和可训练 LR 编码器。检查点只保存可训练编码器；恢复时冻结编码器与解码器从原始 TAESD 权重重载。

SigLIP2 评分器接收 `(LR, HR)` 与 `(LR, 生成 HR)`，其预处理在原评分器中将图像变为 `[0,1]`、缩放到指定大小，并用 image processor 的均值和标准差归一化。评分器输出的是偏好分数，不是预先校准的真假 logit。因此判别器使用同一 LR 下的 **分数差** `score(real)-score(fake)` 做 pairwise logistic 目标 `softplus(score(fake)-score(real))`；生成器用相反目标 `softplus(score(real)-score(fake))`。当 batch 大于 1 时，还用错配 LR/HR 作为负样本。`score_l2_weight` 抑制评分整体漂移。

每个更新周期先更新判别器：生成图像在 `no_grad` 中产生，SigLIP2 主干冻结，成对融合模块与评分头更新。随后冻结判别器参数并更新生成器；冻结的视觉主干仍允许梯度沿生成图像回传。`grad_accum_steps` 个微批次共享一个判别器更新和一个生成器更新。LoRA、可训练 TAESD 编码器和判别器各有独立学习率。

生成器使用 Z-Image 的 clean 与 LPIPS 监督；FDL 按当前要求移除。计算过程在 `losses.py`：

```text
loss_clean = MSE(pred_latent, frozen_TAESD_encoder(HR))
loss_lpips = LPIPS_VGG(pred_image, HR)                  # 图像范围 [-1, 1]
loss_base = 1.0 * loss_clean + 2.0 * loss_lpips
loss_G = loss_base + gan_weight * pairwise_GAN
```

`loss_clean` 是潜变量 MSE；没有像素 L1。Z-Image 原模型会计算 `loss_flow` 供记录，但它不进入实际的 `loss_base`，且脚本权重为 0；该项依赖 Z-Image 的噪声/flow 架构，在本项目指定的 HYPIR 单步生成器中不计算。Z-Image 的 VSD/CSD 分支在这条训练路径中没有启用。对抗项是 `define.md` 要求 SigLIP2 交替训练而增加的部分。

训练工程采用 Z-Image 活跃脚本的批次与优化设置：每卡 batch 1、目标全局 batch 12，`grad_accum_steps=null` 时按 GPU 数自动计算累积；100 个 epoch 上限、50000 个更新步上限；AdamW 的 β 为 0.9/0.999、`eps=1e-8`、权重衰减 0.01；LoRA 与 TAESD 编码器学习率均为 `5e-5`，恒定学习率加 500 步 warmup；每 1000 步保存、每 500 步清理缓存、每步记录最近 100 次更新的损失均值，并写入 TensorBoard。判别器新增独立学习率、优化器和同类调度器；其默认学习率为 `1e-5`。以上参数均可在 JSON 中调整。

## 与 HYPIR-main 默认训练的差异

本项目沿用 HYPIR 的 SD2.1 LR 潜变量单步生成路径；训练数据与 clean、LPIPS 监督以 Z-Image 的实际训练入口为准，按当前要求使用 TAESD、SigLIP2、时间步 500，并移除 FDL。

| 项目 | HYPIR-main `sd2_train.yaml` / 实际训练 | RS_onestep |
| --- | --- | --- |
| 时间步 | UNet `model_t=200`，scheduler `coeff_t=200`；推理默认同为 200 | UNet 与 `x0` 系数均用 500；训练与推理相同 |
| 数据 | HYPIR 的 `RealESRGANDataset` 加 `RealESRGANBatchTransform`，包含队列和锐化 | Z-Image 的 `MyDataset_blind_plus` 裁剪及原样复制的退化代码/YAML |
| VAE | 冻结的 SD2.1 `AutoencoderKL`，编码 LR 时从分布采样 | TAESD：完整训练 LR 编码器，冻结独立 HR 参考编码器和解码器 |
| 监督 | 图像 MSE×1、VGG LPIPS×5、GAN×0.5 | 潜变量 MSE×1、VGG LPIPS×2，再加可配置 GAN |
| 判别器 | `ImageConvNextDiscriminator`，图像真/假目标 | SigLIP2 成对 LR/SR 评分器，分数差的 pairwise logistic 目标 |
| LoRA | rank 256、alpha 256 | rank 64、alpha 128，沿用 Z-Image 脚本数值 |
| 更新与训练量 | G/D 交替的各一个累积周期；默认 batch 6、累积 1、30000 步、EMA | 每周期先 D 后 G，各使用同一组微批次；每卡 batch 1、目标全局 batch 12、最多 50000 步；无 EMA |
| 优化与保存 | AdamW，G/D 学习率均 `1e-5`；每 500 步保存 | AdamW，G LoRA/编码器均 `5e-5`、D 为 `1e-5`；500 步 warmup、每 1000 步保存，另存最终检查点 |

## 训练、恢复与推理

```bash
bash RS_onestep/train.sh RS_onestep/config.json
bash RS_onestep/train.sh RS_onestep/config.json --resume runs/rs_onestep/checkpoints/step-0001000.pt

# 按需指定参与训练的 GPU；脚本据此确定进程数与梯度累积
CUDA_VISIBLE_DEVICES=0,1,2 bash RS_onestep/train.sh RS_onestep/config.json

bash RS_onestep/infer.sh \
  --checkpoint runs/rs_onestep/checkpoints/step-0001000.pt \
  --input examples/lr --output outputs

# 输入已经放大到目标尺寸时：
bash RS_onestep/infer.sh \
  --checkpoint runs/rs_onestep/checkpoints/step-0001000.pt \
  --input examples/upscaled_lr.png --output outputs --upscale 1
```

`--input` 可为单张图或目录；目录递归处理，输出保留相对路径并转为 PNG。原始 LR 默认用 bicubic 放大 4 倍到生成器所需尺寸，映射到 `[-1,1]`，填充到 64 的倍数，再按 `--tile-size`/`--tile-overlap` 分块生成并加权融合，最终裁回目标尺寸。训练退化最后的回放大插值仍保持原实现的随机选择；推理的 bicubic 仅处理外部原始 LR，进入 `OneStepSR.forward()` 时两边都是同尺寸的 LR 条件图。

检查点存于 `output_dir/checkpoints/step-XXXXXXX.pt` 和训练结束时的 `final.pt`，包括步数、配置、UNet LoRA、可训练 TAESD 编码器、SigLIP2 可训练模块、两个优化器和学习率调度器状态、随机状态，以及 GPU 数/有效梯度累积数。`--resume` 从该文件恢复，要求相同 GPU 数与累积数。旧时间步 800 的检查点不能用于当前时间步 500 的模型。基础 SD2.1、冻结 TAESD 部分和冻结 SigLIP2 主干从配置路径重载；若权重移动，推理可用 `--sd21` 与 `--taesd` 覆盖。恢复后的数据加载器重新洗牌，因此同一步之后的数据顺序不保证逐样本相同；多卡随机状态也不保证逐卡精确重放。

常用参数在 JSON 中设置：`resolution`、`batch_size`、`target_global_batch_size`、`grad_accum_steps`、`max_steps`、`save_every`、`log_every`、`precision`、三组学习率及 clean/LPIPS/GAN 损失权重。训练指标同时输出到终端、`output_dir/train.jsonl` 和 `output_dir/logs` 下的 TensorBoard 事件文件。

## 源码对应

| 参考项目实际调用 | 本项目位置 |
| --- | --- |
| `train_tiny.sh` → `train_tiny.py` → `MyDataset_blind_plus` | `train.sh` → `train.py` → `data.py:BlindSRDataset` |
| `MyDataset_blind_plus.__getitem__` 的裁切、`degrade_process(..., resize_bak=True)` 和 `[-1,1]` 转换 | `data.py:BlindSRDataset.__getitem__` |
| `sr_utils/realesrgan.py:RealESRGAN_degradation` | `realesrgan.py:RealESRGAN_degradation`，原样复制 |
| `sr_utils/params_realesrgan_seesr.yml` | `params_realesrgan_seesr.yml`，原样复制 |
| `SRmodel_tiny.py` 中的 `loss_clean`、`loss_lpips` 与 `loss_base` | `losses.py:ZImageSupervision` 和 `train.py` 的生成器更新；原 FDL 项按当前要求移除 |
| `train_tiny.py` 的累积、AdamW、学习率调度、日志和检查点组织 | `train.py` 和 `config.example.json`；另接入 SigLIP2 判别器更新 |
| `infer_tiny_my.sh` → `inference_tiny.py` 的权重加载/分块推理组织 | `infer.sh` → `infer.py` 的检查点加载/分块推理组织；生成架构改用下表 HYPIR 路径 |

| HYPIR 关键模块 | 本项目位置 |
| --- | --- |
| `HYPIR/trainer/sd2.py:init_generator` 的 SD2.1 UNet、文本编码器、LoRA | `model.py:OneStepSR.__init__` |
| `HYPIR/trainer/base.py:prepare_batch_inputs` 的 LR 编码和 prompt 条件 | `model.py:prompt_embeddings`、`OneStepSR.forward`；训练端 `train.py` |
| `HYPIR/trainer/sd2.py:forward_generator` 的 LR latent → UNet → `x0` → 解码 | `model.py:OneStepSR.forward` |
| `HYPIR/enhancer/sd2.py:prepare_inputs/forward_generator` 的推理时间步与同一单步前向 | `model.py:OneStepSR.forward`、`infer.py:enhance` |
| `HYPIR/trainer/base.py:init_vae/init_discriminator/optimize_*` 的连接点 | `model.py` 的双 TAESD 编码器与解码器、`discriminator.py` 和 `train.py` 的交替更新 |

指定 SigLIP2 文件中的模型类和输入预处理保留在 `siglip2_pair_sr_reward_optional_fidelity.py`，删去未使用的排序/ReFL 训练器和演示代码；本项目的加载与可训练参数保存放在 `discriminator.py`。

复制和改编的 Z-Image 源码遵循本目录附带的 `LICENSE`。

## 验证范围

按任务要求，未运行训练、推理或实际测试。已静态检查权重加载路径、`3×512×512 → 4×64×64 → 3×512×512` 张量关系、两阶段梯度隔离和检查点保存/恢复路径；运行兼容性及输出质量仍需在具备 GPU 的环境验证。
