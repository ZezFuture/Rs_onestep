"""Clean latent MSE and VGG LPIPS supervision."""

import torch.nn.functional as F
from torch import nn


class ZImageSupervision(nn.Module):
    """Clean latent MSE and VGG LPIPS."""

    def __init__(self, config: dict):
        super().__init__()
        self.clean_weight = float(config["clean_loss_weight"])
        self.lpips_weight = float(config["lpips_loss_weight"])
        if min(self.clean_weight, self.lpips_weight) < 0:
            raise ValueError("Supervision loss weights must be nonnegative")
        if self.lpips_weight:
            import lpips

            self.lpips = lpips.LPIPS(net=config.get("lpips_net", "vgg")).eval().requires_grad_(False)
        else:
            self.lpips = None
    def train(self, mode: bool = True):
        super().train(False)
        return self

    def forward(self, pred_image, hr_image, pred_latent, hr_latent):
        # train_tiny.py's "clean" loss is latent MSE, not pixel MSE.
        clean = F.mse_loss(pred_latent.float(), hr_latent.float())
        zero = clean.new_zeros(())
        # LPIPS receives [-1, 1] float inputs under the outer training autocast.
        pred = pred_image.float()
        hr = hr_image.float().clamp(-1, 1)
        perceptual = self.lpips(pred, hr).mean() if self.lpips is not None else zero
        total = self.clean_weight * clean + self.lpips_weight * perceptual
        return total, {"clean": clean, "lpips": perceptual}
