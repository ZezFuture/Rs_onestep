"""Z-Image-style training loop with HYPIR generator and SigLIP2 GAN updates."""

import argparse
import gc
import json
import os
import random
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data import BlindSRDataset
from discriminator import build_discriminator, load_trainable_state, trainable_state
from losses import ZImageSupervision
from model import OneStepSR, TIMESTEP


def save_checkpoint(path, step, epoch, config, model, discriminator, architecture,
                    optimizer_g, optimizer_d, scheduler_g, scheduler_d, loss_window,
                    world_size, grad_accum_steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step, "epoch": epoch, "config": config,
        "world_size": world_size, "grad_accum_steps_effective": grad_accum_steps,
        "timestep": TIMESTEP, "prediction_type": model.prediction_type,
        "alpha_t": float(model.alpha_t),
        "discriminator_architecture": architecture,
        "unet_lora": model.lora_state(),
        "taesd_lr_encoder": {k: v.detach().cpu() for k, v in model.taesd.encoder.state_dict().items()},
        "discriminator": trainable_state(discriminator),
        "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(),
        "scheduler_g": scheduler_g.state_dict(), "scheduler_d": scheduler_d.state_dict(),
        "loss_window": list(loss_window),
        "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def restore_checkpoint(state, model, discriminator, optimizer_g, optimizer_d, scheduler_g, scheduler_d):
    if (state["timestep"] != TIMESTEP or state["prediction_type"] != model.prediction_type
            or abs(state["alpha_t"] - float(model.alpha_t)) > 1e-7):
        raise ValueError("Checkpoint timestep or scheduler differs from supplied SD2.1 model")
    model.load_lora_state(state["unet_lora"])
    model.taesd.encoder.load_state_dict(state["taesd_lr_encoder"], strict=True)
    load_trainable_state(discriminator, state["discriminator"])
    optimizer_g.load_state_dict(state["optimizer_g"])
    optimizer_d.load_state_dict(state["optimizer_d"])
    scheduler_g.load_state_dict(state["scheduler_g"])
    scheduler_d.load_state_dict(state["scheduler_d"])
    random.setstate(state["rng_python"])
    np.random.set_state(state["rng_numpy"])
    torch.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    return int(state["step"]), int(state["epoch"]), state["loss_window"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="JSON configuration")
    parser.add_argument("--resume", help="Full checkpoint written by this script")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    precision = config["precision"]
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be bf16 or fp32")
    logging_dir = Path(config["output_dir"]) / "logs"
    accelerator = Accelerator(
        mixed_precision="bf16" if precision == "bf16" else "no",
        log_with=config["report_to"] if config["report_to"] != "none" else None,
        project_config=ProjectConfiguration(project_dir=config["output_dir"], logging_dir=str(logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    if accelerator.device.type != "cuda":
        raise RuntimeError("Training requires CUDA GPUs")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU does not support bf16; use fp32")
    set_seed(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])
    if accelerator.is_main_process:
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)
        (Path(config["output_dir"]) / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )
    accelerator.wait_for_everyone()

    batch_size = int(config["batch_size"])
    target_global = int(config["target_global_batch_size"])
    if config["grad_accum_steps"] is None:
        denominator = accelerator.num_processes * batch_size
        if target_global % denominator:
            raise ValueError("target_global_batch_size must be divisible by GPU count × batch_size")
        accum = target_global // denominator
    else:
        accum = int(config["grad_accum_steps"])
    max_steps = int(config["max_steps"])
    max_epochs = int(config["max_epochs"])
    save_every = int(config["save_every"])
    log_every = int(config["log_every"])
    eval_freq = int(config["eval_freq"])
    window_size = int(config["loss_window_size"])
    if min(batch_size, accum, max_steps, max_epochs, save_every, log_every, eval_freq, window_size) < 1:
        raise ValueError("Batch, step, epoch, log and checkpoint counts must be positive")

    dataset = BlindSRDataset(config["train_dir"], int(config["resolution"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=int(config["num_workers"]), pin_memory=True, drop_last=True)
    if not len(loader):
        raise ValueError("Dataset has fewer images than batch_size")
    model = OneStepSR(config["sd21"], config["taesd"], int(config["lora_rank"]),
                      int(config["lora_alpha"]), float(config["lora_dropout"]),
                      config["lora_modules"])
    resume_state = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume_state is not None:
        for key in ("sd21", "taesd", "siglip2", "lora_rank", "lora_alpha",
                    "lora_dropout", "lora_modules", "resolution",
                    "batch_size", "target_global_batch_size", "grad_accum_steps", "lr_scheduler",
                    "lr_warmup_steps", "lr_scheduler_d", "lr_warmup_steps_d", "max_steps",
                    "prompt", "lr_lora", "lr_encoder", "lr_discriminator", "adam_beta1",
                    "adam_beta2", "adam_epsilon", "weight_decay", "clean_loss_weight",
                    "lpips_loss_weight", "lpips_net", "gan_weight",
                    "wrong_lr_weight", "score_l2_weight"):
            if config[key] != resume_state["config"][key]:
                raise ValueError(f"Resume config differs in {key}")
        if (resume_state["world_size"] != accelerator.num_processes
                or resume_state["grad_accum_steps_effective"] != accum):
            raise ValueError("Resume requires the same GPU count and gradient accumulation")
    discriminator, architecture = build_discriminator(
        config,
        architecture=resume_state["discriminator_architecture"] if resume_state else None,
        reward_path=None if resume_state else config.get("reward_checkpoint"),
    )
    supervision = ZImageSupervision(config).to(accelerator.device).eval()
    d_params = [parameter for parameter in discriminator.parameters() if parameter.requires_grad]
    g_lora = [parameter for parameter in model.unet.parameters() if parameter.requires_grad]
    g_encoder = list(model.taesd.encoder.parameters())
    if not d_params or not g_lora or not g_encoder:
        raise RuntimeError("Generator and discriminator must have trainable parameters")
    betas = (float(config["adam_beta1"]), float(config["adam_beta2"]))
    optimizer_g = torch.optim.AdamW([
        {"name": "sr_lora", "params": g_lora, "lr": float(config["lr_lora"]),
         "weight_decay": float(config["weight_decay"])},
        {"name": "taesd_encoder", "params": g_encoder, "lr": float(config["lr_encoder"]),
         "weight_decay": float(config["weight_decay"])},
    ], betas=betas, eps=float(config["adam_epsilon"]))
    optimizer_d = torch.optim.AdamW(d_params, lr=float(config["lr_discriminator"]),
                                    weight_decay=float(config["weight_decay"]),
                                    betas=betas, eps=float(config["adam_epsilon"]))
    scheduler_g = get_scheduler(
        config["lr_scheduler"], optimizer=optimizer_g,
        num_warmup_steps=int(config["lr_warmup_steps"]) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    scheduler_d = get_scheduler(
        config["lr_scheduler_d"], optimizer=optimizer_d,
        num_warmup_steps=int(config["lr_warmup_steps_d"]) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    model, discriminator, optimizer_g, optimizer_d, loader, scheduler_g, scheduler_d = accelerator.prepare(
        model, discriminator, optimizer_g, optimizer_d, loader, scheduler_g, scheduler_d
    )
    raw_g = accelerator.unwrap_model(model)
    raw_d = accelerator.unwrap_model(discriminator)
    d_params = [parameter for parameter in raw_d.parameters() if parameter.requires_grad]
    g_lora = [parameter for parameter in raw_g.unet.parameters() if parameter.requires_grad]
    g_encoder = list(raw_g.taesd.encoder.parameters())
    if not len(loader):
        raise ValueError("Prepared dataloader has no batches on this process")
    if resume_state:
        step, epoch, prior_window = restore_checkpoint(
            resume_state, raw_g, raw_d, optimizer_g, optimizer_d, scheduler_g, scheduler_d
        )
    else:
        step, epoch, prior_window = 0, 0, []
    loss_window = deque(prior_window, maxlen=window_size)
    accelerator.init_trackers("rs_onestep", config={k: str(v) for k, v in config.items()})
    if accelerator.is_main_process:
        print(f"images={len(dataset)} processes={accelerator.num_processes} "
              f"batch_per_gpu={batch_size} grad_accum={accum} "
              f"global_batch={accelerator.num_processes * batch_size * accum}", flush=True)
    progress = tqdm(range(max_steps), initial=step, disable=not accelerator.is_local_main_process)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator, epoch
        try:
            return next(iterator)
        except StopIteration:
            epoch += 1
            if epoch >= max_epochs:
                return None
            iterator = iter(loader)
            return next(iterator)

    while step < max_steps and epoch < max_epochs:
        batches = []
        for _ in range(accum):
            batch = next_batch()
            if batch is None:
                break
            batches.append(batch)
        if not batches:
            break
        micro_count = len(batches)
        raw_g.train()
        raw_d.train()
        raw_d.vision_model.eval()
        for parameter in d_params:
            parameter.requires_grad_(True)
        optimizer_d.zero_grad(set_to_none=True)
        d_total = 0.0
        for micro, batch in enumerate(batches):
            lr = batch["conditioning_pixel_values"].to(accelerator.device, non_blocking=True)
            hr = batch["output_pixel_values"].to(accelerator.device, non_blocking=True)
            text = raw_g.prompt_embeddings(config["prompt"], len(lr), accelerator.device)
            with torch.no_grad(), accelerator.autocast():
                fake, _ = raw_g(lr, text)
            wrong = len(lr) > 1 and float(config["wrong_lr_weight"]) > 0
            lr_inputs = [lr, lr] + ([lr.roll(1, 0)] if wrong else [])
            sr_inputs = [hr, fake.detach()] + ([hr] if wrong else [])
            sync = accelerator.no_sync(discriminator) if micro < micro_count - 1 else nullcontext()
            with sync, accelerator.autocast():
                scores = discriminator(torch.cat(lr_inputs), torch.cat(sr_inputs)).float().chunk(len(lr_inputs))
                real_score, fake_score = scores[:2]
                loss_d = F.softplus(fake_score - real_score).mean()
                if wrong:
                    loss_d = loss_d + float(config["wrong_lr_weight"]) * F.softplus(
                        scores[2] - real_score
                    ).mean()
                loss_d = loss_d + float(config["score_l2_weight"]) * 0.5 * (
                    real_score.square().mean() + fake_score.square().mean()
                )
                accelerator.backward(loss_d / micro_count)
            d_total += loss_d.detach().item() / micro_count
        accelerator.clip_grad_norm_(d_params, float(config["max_grad_norm"]))
        optimizer_d.step()
        scheduler_d.step()
        optimizer_d.zero_grad(set_to_none=True)

        raw_d.eval()
        for parameter in d_params:
            parameter.requires_grad_(False)
        optimizer_g.zero_grad(set_to_none=True)
        totals = {"G": 0.0, "base": 0.0, "clean": 0.0,
                  "lpips": 0.0, "gan": 0.0}
        for micro, batch in enumerate(batches):
            lr = batch["conditioning_pixel_values"].to(accelerator.device, non_blocking=True)
            hr = batch["output_pixel_values"].to(accelerator.device, non_blocking=True)
            text = raw_g.prompt_embeddings(config["prompt"], len(lr), accelerator.device)
            sync = accelerator.no_sync(model) if micro < micro_count - 1 else nullcontext()
            with sync, accelerator.autocast():
                fake, pred_latent = model(lr, text)
                with torch.no_grad():
                    target_latent = raw_g.encode_reference(hr)
                    real_score = raw_d(lr, hr).float()
                fake_score = raw_d(lr, fake).float()
                adversarial = F.softplus(real_score - fake_score).mean()
                base_loss, parts = supervision(fake, hr, pred_latent, target_latent)
                loss_g = base_loss + float(config["gan_weight"]) * adversarial
                accelerator.backward(loss_g / micro_count)
            for key, value in (("G", loss_g), ("base", base_loss), ("gan", adversarial),
                               ("clean", parts["clean"]), ("lpips", parts["lpips"])):
                totals[key] += value.detach().item() / micro_count
        accelerator.clip_grad_norm_(g_lora + g_encoder, float(config["max_grad_norm"]))
        optimizer_g.step()
        scheduler_g.step()
        step += 1
        progress.update(1)
        loss_window.append(totals["G"])
        if step == 1 or step % log_every == 0:
            names = ("D", "G", "base", "clean", "lpips", "gan")
            values = torch.tensor([d_total] + [totals[key] for key in names[1:]],
                                  device=accelerator.device)
            values = accelerator.reduce(values, reduction="mean").tolist()
            metrics = dict(zip(names, values))
            metrics.update({"step": step, "epoch": epoch,
                            "window_loss": sum(loss_window) / len(loss_window),
                            "lr_sr_lora": scheduler_g.get_last_lr()[0],
                            "lr_taesd_encoder": scheduler_g.get_last_lr()[1],
                            "lr_discriminator": scheduler_d.get_last_lr()[0]})
            accelerator.log(metrics, step=step)
            if accelerator.is_main_process:
                print(json.dumps(metrics), flush=True)
                with (Path(config["output_dir"]) / "train.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(metrics) + "\n")
        if step % save_every == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                path = Path(config["output_dir"]) / "checkpoints" / f"step-{step:07d}.pt"
                save_checkpoint(path, step, epoch, config, raw_g, raw_d, architecture,
                                optimizer_g, optimizer_d, scheduler_g, scheduler_d, loss_window,
                                accelerator.num_processes, accum)
                print(f"Saved {path}", flush=True)
            accelerator.wait_for_everyone()
        if step % eval_freq == 0:
            gc.collect()
            torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        path = Path(config["output_dir"]) / "checkpoints" / "final.pt"
        save_checkpoint(path, step, epoch, config, raw_g, raw_d, architecture,
                        optimizer_g, optimizer_d, scheduler_g, scheduler_d, loss_window,
                        accelerator.num_processes, accum)
        print(f"Saved {path}", flush=True)
    accelerator.end_training()


if __name__ == "__main__":
    main()
