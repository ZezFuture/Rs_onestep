"""Single-image or recursive-directory inference from a training checkpoint."""

import argparse
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import OneStepSR, TIMESTEP


EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

def load_or_create_text_embedding(model, prompt, device, cache_path="text_emb.pt"):
    cache_path = Path(cache_path)

    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("prompt") == prompt:
            print(f"[Text Embedding] Loading from {cache_path}")
            return cache["embedding"].to(device)
        print(f"[Text Embedding] Prompt changed, recomputing...")

    print(f"[Text Embedding] Computing: {prompt}")
    with torch.no_grad():
        text = model.prompt_embeddings(prompt, 1, device)

    torch.save({"prompt": prompt, "embedding": text.detach().cpu()}, cache_path)
    print(f"[Text Embedding] Saved to {cache_path}")
    return text

def files_at(path: Path):
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if files:
            return files
    raise ValueError(f"No supported images at {path}")


def image_tensor(path: Path, device):
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, tile_size - overlap))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def tile_weights(start, length, total, overlap, device):
    weights = torch.ones(length, device=device, dtype=torch.float32)
    if start > 0:
        weights[:overlap] = torch.linspace(0, 1, overlap, device=device)
    if start + length < total:
        weights[-overlap:] = torch.linspace(1, 0, overlap, device=device)
    return weights


def enhance(model, image, text, tile_size, overlap, amp):
    height, width = image.shape[-2:]
    image = F.pad(image, (0, (-width) % 64, 0, (-height) % 64), mode="replicate")
    padded_h, padded_w = image.shape[-2:]
    output = torch.zeros((1, 3, padded_h, padded_w), dtype=torch.float32, device=image.device)
    weight_sum = torch.zeros((1, 1, padded_h, padded_w), dtype=torch.float32, device=image.device)
    for top in tile_starts(padded_h, tile_size, overlap):
        for left in tile_starts(padded_w, tile_size, overlap):
            tile = image[..., top:top + tile_size, left:left + tile_size]
            h, w = tile.shape[-2:]
            with amp():
                pred, _ = model(tile * 2 - 1, text)
            wy = tile_weights(top, h, padded_h, min(overlap, h), image.device)
            wx = tile_weights(left, w, padded_w, min(overlap, w), image.device)
            weight = (wy[:, None] * wx[None, :])[None, None]
            output[..., top:top + h, left:left + w] += pred.float() * weight
            weight_sum[..., top:top + h, left:left + w] += weight

    return (output / weight_sum.clamp_min(1e-6))[..., :height, :width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image path or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sd21", help="Override SD2.1 location from checkpoint")
    parser.add_argument("--taesd", help="Override TAESD location from checkpoint")
    parser.add_argument("--prompt", help="Override prompt from checkpoint")
    parser.add_argument("--upscale",type=float,default=4.0,help="Bicubic pre-upscale factor before restoration; use 1.0 to disable resizing")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--gray-output", action="store_true", help="Average RGB channels and save as grayscale")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires a CUDA GPU")
    if args.tile_size < 64 or args.tile_size % 64 or args.tile_overlap < 64 or args.tile_overlap % 64:
        raise ValueError("tile size and overlap must be multiples of 64")
    if args.tile_overlap >= args.tile_size:
        raise ValueError("tile overlap must be smaller than tile size")
    device = torch.device("cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = state["config"]
    if state["timestep"] != TIMESTEP:
        raise ValueError(f"Checkpoint does not use training timestep {TIMESTEP}")
    model = OneStepSR(args.sd21 or config["sd21"], args.taesd or config["taesd"],
                      int(config["lora_rank"]), int(config["lora_alpha"]),
                      float(config["lora_dropout"]), config["lora_modules"])
    if (state["prediction_type"] != model.prediction_type
            or abs(state["alpha_t"] - float(model.alpha_t)) > 1e-7):
        raise ValueError("Checkpoint scheduler does not match SD2.1 weights")
    model.load_lora_state(state["unet_lora"])
    model.taesd.encoder.load_state_dict(state["taesd_lr_encoder"], strict=True)
    model.to(device).eval()
    precision = config["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support checkpoint bf16 precision")
    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()

    source = Path(args.input)
    destination = Path(args.output)

    prompt = args.prompt or config["prompt"]
    text = load_or_create_text_embedding(model, prompt, device, "text_emb.pt")

    with torch.inference_mode():
        for path in files_at(source):
            image = image_tensor(path, device)

            if args.upscale <= 0:
                raise ValueError("upscale must be positive")

            if args.upscale != 1:
                image = F.interpolate(image, scale_factor=args.upscale, mode="bicubic", align_corners=False).clamp(0, 1)
            sr = enhance(model, image, text, args.tile_size, args.tile_overlap, amp)
            sr = ((sr[0] + 1) * 0.5).clamp(0, 1)
            if args.gray_output:
                array = (sr.mean(dim=0).cpu().numpy() * 255).round().astype(np.uint8)
            else:
                array = (sr.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            relative = path.relative_to(source) if source.is_dir() else Path(path.name)
            output_path = destination / relative.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.gray_output:
                Image.fromarray(array, "L").save(output_path)
            else:
                Image.fromarray(array, "RGB").save(output_path)
            print(f"{path} -> {output_path}", flush=True)

if __name__ == "__main__":
    main()
