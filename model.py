"""HYPIR's LR-latent SD2.1 UNet path with two TAESD encoders."""

from copy import deepcopy

import torch
from torch import nn
from diffusers import AutoencoderTiny, DDPMScheduler, UNet2DConditionModel
from peft import LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer


TIMESTEP = 500  # Diffusion training timestep, not an inference-list position.


class OneStepSR(nn.Module):
    def __init__(self, sd21: str, taesd: str, lora_rank: int, lora_alpha: int,
                 lora_dropout: float, lora_modules: list[str]):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(sd21, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd21, subfolder="text_encoder")
        self.text_encoder.eval().requires_grad_(False)

        self.unet = UNet2DConditionModel.from_pretrained(sd21, subfolder="unet")
        self.unet.eval().requires_grad_(False)
        self.unet.add_adapter(LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=lora_modules, init_lora_weights="gaussian",
        ))
        if not any(p.requires_grad for p in self.unet.parameters()):
            raise RuntimeError("No trainable UNet LoRA parameters")

        self.taesd = AutoencoderTiny.from_pretrained(taesd)
        cfg = self.taesd.config
        if (cfg.in_channels != 3 or cfg.out_channels != 3 or cfg.latent_channels != 4
                or self.unet.config.in_channels != 4 or self.unet.config.out_channels != 4):
            raise ValueError("TAESD and SD2.1 must use RGB images and four latent channels")
        self.latent_scale = float(cfg.scaling_factor)
        if self.latent_scale != 1.0 or float(getattr(cfg, "shift_factor", 0.0)) != 0.0:
            raise ValueError("Use SD TAESD with scaling_factor=1 and shift_factor=0")
        self.spatial_scale = int(cfg.upsampling_scaling_factor) ** (len(cfg.encoder_block_out_channels) - 1)
        if self.spatial_scale != 8:
            raise ValueError("SD TAESD must have image-to-latent spatial factor 8")
        # The reference branch has its own parameters and never joins an optimizer.
        self.reference_encoder = deepcopy(self.taesd.encoder).eval().requires_grad_(False)
        self.taesd.requires_grad_(False)
        self.taesd.encoder.requires_grad_(True)
        self.taesd.decoder.eval()

        scheduler = DDPMScheduler.from_pretrained(sd21, subfolder="scheduler")
        if scheduler.config.num_train_timesteps <= TIMESTEP:
            raise ValueError(f"Scheduler has no training timestep {TIMESTEP}")
        if scheduler.config.prediction_type not in ("epsilon", "v_prediction"):
            raise ValueError("Expected epsilon or v_prediction SD scheduler")
        if scheduler.config.clip_sample or scheduler.config.thresholding:
            raise ValueError("This x0 formula requires an unclipped SD2.1 scheduler")
        self.prediction_type = scheduler.config.prediction_type
        self.register_buffer("alpha_t", scheduler.alphas_cumprod[TIMESTEP].float(), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.unet.eval()  # HYPIR keeps the pretrained base in eval mode.
        self.text_encoder.eval()
        self.reference_encoder.eval()
        self.taesd.decoder.eval()
        return self

    @torch.no_grad()
    def prompt_embeddings(self, prompt: str, batch_size: int, device: torch.device):
        ids = self.tokenizer(
            [prompt] * batch_size, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        ).input_ids.to(device)
        return self.text_encoder(ids).last_hidden_state

    @torch.no_grad()
    def encode_reference(self, hr: torch.Tensor):
        # EncoderTiny accepts [-1, 1] and internally maps it to [0, 1].
        return self.reference_encoder(hr).float() * self.latent_scale

    def forward(self, lr: torch.Tensor, text_embeddings: torch.Tensor):
        if lr.ndim != 4 or lr.shape[1] != 3:
            raise ValueError("Expected NCHW RGB low-quality image")
        if lr.shape[-2] % 64 or lr.shape[-1] % 64:
            raise ValueError("Image height and width must be multiples of 64")
        # No independently sampled noise latent. The LR encoding is the sole UNet sample.
        z_lr = self.taesd.encode(lr).latents * self.latent_scale
        expected = (lr.shape[0], 4, lr.shape[-2] // 8, lr.shape[-1] // 8)
        if tuple(z_lr.shape) != expected:
            raise RuntimeError(f"TAESD latent shape {tuple(z_lr.shape)} != {expected}")
        timesteps = torch.full((lr.shape[0],), TIMESTEP, dtype=torch.long, device=lr.device)
        prediction = self.unet(z_lr, timesteps, encoder_hidden_states=text_embeddings).sample.float()
        z_lr = z_lr.float()
        alpha = self.alpha_t.to(dtype=z_lr.dtype)
        if self.prediction_type == "epsilon":
            z0 = (z_lr - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
        else:
            z0 = alpha.sqrt() * z_lr - (1 - alpha).sqrt() * prediction
        image = self.taesd.decode(z0 / self.latent_scale).sample.float().clamp(-1, 1)
        return image, z0

    def lora_state(self):
        return {name: value.detach().cpu() for name, value in self.unet.state_dict().items() if "lora_" in name}

    def load_lora_state(self, state):
        expected = {name for name in self.unet.state_dict() if "lora_" in name}
        if set(state) != expected:
            raise ValueError(f"UNet LoRA keys differ: missing={expected - set(state)}, extra={set(state) - expected}")
        self.unet.load_state_dict(state, strict=False)
