#!/usr/bin/env python3
"""
train_flux_nvs.py — Training script for FLUX-based Novel View Synthesis
========================================================================
Supports:
  - Multi-GPU DDP (torch.distributed)
  - bf16 mixed precision (AMP)
  - Gradient checkpointing
  - 8-bit AdamW (bitsandbytes)
  - Checkpoint save/resume
  - Wandb logging (optional)
  - USCILab3D dataset integration

Usage:
  # Single GPU (debug)
  python train_flux_nvs.py --config configs/flux_nvs.yaml

  # Multi-GPU DDP
  torchrun --nproc_per_node=4 train_flux_nvs.py --config configs/flux_nvs.yaml

  # With LoRA (faster, less VRAM)
  torchrun --nproc_per_node=4 train_flux_nvs.py --config configs/flux_nvs.yaml --lora

Compatible with Python 3.9+.
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load YAML config, merging with defaults."""
    defaults = {
        # Model
        "flux_model_id": "black-forest-labs/FLUX.1-dev",
        "flux_dtype": "bfloat16",
        "img_height": 720,
        "img_width": 1280,
        "n_ref_images": 4,
        "use_lora": False,
        "lora_rank": 64,
        "lora_alpha": 64,
        "lambda_x0_recon": 0.0,
        "use_dynamic_mask": True,
        "guidance_scale": 1.0,

        # Training
        "learning_rate": 1e-5,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 0.01,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 4,
        "max_steps": 50000,
        "warmup_steps": 1000,
        "save_every_steps": 5000,
        "eval_every_steps": 2500,
        "log_every_steps": 50,

        # Data
        "data_root": "/data/USCILab3D",
        "mapping_dir": "/tmp/velodyne_cam_mappings",
        "projection_dir": None,
        "semantic_dir": None,
        "extracted_dir": None,
        "global_poses_path": None,
        "max_depth": 80.0,
        "max_dt_ms": 100.0,
        "samples_per_session": 200,
        "num_workers": 4,

        # Checkpointing
        "output_dir": "./checkpoints/flux_nvs",
        "resume_from": None,

        # Logging
        "use_wandb": False,
        "wandb_project": "uscilab3d-flux-nvs",
        "wandb_run_name": None,

        # Hardware
        "gradient_checkpointing": True,
        "use_8bit_adam": True,
    }

    if path and os.path.exists(path):
        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        defaults.update(user_config)

    return defaults


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    """Initialize DDP if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def is_main_process(rank):
    return rank == 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, config):
    """Build optimizer with separate param groups."""
    # Separate trainable params
    transformer_params = []
    adapter_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "transformer" in name:
            transformer_params.append(param)
        else:
            adapter_params.append(param)

    param_groups = [
        {"params": transformer_params, "lr": config["learning_rate"]},
        {"params": adapter_params, "lr": config["learning_rate"] * 5.0},
        # Adapters get 5x LR — they start from scratch
    ]

    if config["use_8bit_adam"]:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                weight_decay=config["adam_weight_decay"],
                eps=config["adam_epsilon"],
            )
            logger.info("Using 8-bit AdamW (bitsandbytes)")
        except ImportError:
            logger.warning("bitsandbytes not found, falling back to torch AdamW")
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                weight_decay=config["adam_weight_decay"],
                eps=config["adam_epsilon"],
            )
    else:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config["learning_rate"],
            betas=(config["adam_beta1"], config["adam_beta2"]),
            weight_decay=config["adam_weight_decay"],
            eps=config["adam_epsilon"],
        )

    return optimizer


def build_scheduler(optimizer, config):
    """Cosine schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR

    warmup = config["warmup_steps"]
    total = config["max_steps"]

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataloader(config, rank, world_size):
    """Build USCILab3D dataloader with distributed sampler."""
    from uscilab3d_dataset import USCILab3DDataset, uscilab3d_collate

    dataset = USCILab3DDataset(
        data_root=config["data_root"],
        mapping_dir=config["mapping_dir"],
        projection_dir=config.get("projection_dir"),
        semantic_dir=config.get("semantic_dir"),
        extracted_dir=config.get("extracted_dir"),
        global_poses_path=config.get("global_poses_path"),
        n_refs=config["n_ref_images"],
        max_depth=config["max_depth"],
        max_dt_ms=config["max_dt_ms"],
        samples_per_session=config["samples_per_session"],
        img_height=config["img_height"],
        img_width=config["img_width"],
    )

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True)
        shuffle = False

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size_per_gpu"],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config["num_workers"],
        collate_fn=uscilab3d_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config["num_workers"] > 0,
    )

    return loader, sampler


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, step, config, rank):
    """Save model checkpoint."""
    if not is_main_process(rank):
        return

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save only trainable params + adapters (not the frozen FLUX weights)
    unwrapped = model.module if isinstance(model, DDP) else model

    trainable_state = {}
    for name, param in unwrapped.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu()

    # Also save non-parameter buffers from our custom modules
    for name, buf in unwrapped.cond_encoder.named_buffers():
        trainable_state["cond_encoder." + name] = buf.cpu()
    for name, buf in unwrapped.ref_adapter.named_buffers():
        trainable_state["ref_adapter." + name] = buf.cpu()

    ckpt = {
        "step": step,
        "model_state": trainable_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
    }

    path = out_dir / "checkpoint_step{:06d}.pt".format(step)
    torch.save(ckpt, path)

    # Also save as "latest"
    latest_path = out_dir / "checkpoint_latest.pt"
    torch.save(ckpt, latest_path)

    logger.info("Saved checkpoint → %s", path)


def load_checkpoint(model, optimizer, scheduler, config):
    """Resume from checkpoint if available."""
    resume_path = config.get("resume_from")
    if resume_path is None:
        # Try "latest"
        latest = Path(config["output_dir"]) / "checkpoint_latest.pt"
        if latest.exists():
            resume_path = str(latest)

    if resume_path is None or not os.path.exists(resume_path):
        return 0

    logger.info("Resuming from %s", resume_path)
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)

    unwrapped = model.module if isinstance(model, DDP) else model
    # Load trainable params
    missing, unexpected = [], []
    model_state = ckpt.get("model_state", {})
    for name, param in unwrapped.named_parameters():
        if name in model_state:
            param.data.copy_(model_state[name])
        elif param.requires_grad:
            missing.append(name)

    if missing:
        logger.warning("Missing keys in checkpoint: %s", missing[:5])

    # Load optimizer + scheduler
    if "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception as e:
            logger.warning("Could not load optimizer state: %s", e)

    if "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception as e:
            logger.warning("Could not load scheduler state: %s", e)

    step = ckpt.get("step", 0)
    logger.info("Resumed at step %d", step)
    return step


# ---------------------------------------------------------------------------
# Eval / debug sample
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_debug_sample(model, dataloader, step, config, device, rank):
    """Generate a debug sample and save to disk."""
    if not is_main_process(rank):
        return

    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.eval()

    # Grab one batch
    batch = next(iter(dataloader))
    refs, depth, semantic, target = batch

    refs = [r[:1].to(device) for r in refs]  # just 1 sample
    depth = depth[:1].to(device)
    semantic = semantic[:1].to(device)
    target = target[:1].to(device)

    rgb_pred = unwrapped.sample(
        refs, depth, semantic,
        num_steps=20,  # fewer steps for debug
        guidance_scale=config.get("guidance_scale", 3.5),
    )

    # Save
    out_dir = Path(config["output_dir"]) / "debug_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.utils import save_image
    save_image(rgb_pred[0], out_dir / "pred_step{:06d}.png".format(step))
    save_image(target[0], out_dir / "target_step{:06d}.png".format(step))
    save_image(depth[0].repeat(3, 1, 1), out_dir / "depth_step{:06d}.png".format(step))

    logger.info("Debug sample saved → %s/", out_dir)
    unwrapped.train()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config):
    rank, local_rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)

    if is_main_process(rank):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info("Rank %d/%d, device: %s", rank, world_size, device)

    # ----- Build model -----
    from flux_nvs import FluxNVS, FluxNVSConfig

    model_config = FluxNVSConfig(
        flux_model_id=config["flux_model_id"],
        flux_dtype=config["flux_dtype"],
        img_height=config["img_height"],
        img_width=config["img_width"],
        n_ref_images=config["n_ref_images"],
        use_lora=config["use_lora"],
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        learning_rate=config["learning_rate"],
        lambda_x0_recon=config["lambda_x0_recon"],
        use_dynamic_mask=config["use_dynamic_mask"],
        guidance_scale=config["guidance_scale"],
    )

    model = FluxNVS(model_config)
    model.load_pretrained(device=device)

    # Gradient checkpointing on transformer
    if config["gradient_checkpointing"]:
        model.transformer.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled on transformer")

    # Freeze VAE (already frozen in load_pretrained)
    # Optionally freeze transformer for initial adapter warm-up
    # (uncomment below for a 2-phase training strategy)
    # for p in model.transformer.parameters():
    #     p.requires_grad = False

    # Move to device
    model = model.to(device)

    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=True)

    # ----- Data -----
    loader, sampler = build_dataloader(config, rank, world_size)
    logger.info("Dataset: %d samples", len(loader.dataset))

    # ----- Optimizer + Scheduler -----
    optimizer = build_optimizer(
        model.module if isinstance(model, DDP) else model,
        config
    )
    scheduler = build_scheduler(optimizer, config)

    # ----- Resume -----
    global_step = load_checkpoint(model, optimizer, scheduler, config)

    # ----- Wandb -----
    if config["use_wandb"] and is_main_process(rank):
        import wandb
        wandb.init(
            project=config["wandb_project"],
            name=config.get("wandb_run_name"),
            config=config,
        )

    # ----- Training -----
    logger.info("Starting training from step %d → %d", global_step, config["max_steps"])
    logger.info("Batch: %d/GPU × %d GPUs × %d accum = %d effective",
                config["batch_size_per_gpu"], world_size,
                config["gradient_accumulation_steps"],
                config["batch_size_per_gpu"] * world_size * config["gradient_accumulation_steps"])

    accum_steps = config["gradient_accumulation_steps"]
    epoch = 0

    while global_step < config["max_steps"]:
        if sampler is not None:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(loader):
            if global_step >= config["max_steps"]:
                break

            refs, depth, semantic, target = batch
            refs = [r.to(device) for r in refs]
            depth = depth.to(device)
            semantic = semantic.to(device)
            target = target.to(device)

            # Forward
            unwrapped = model.module if isinstance(model, DDP) else model
            metrics = unwrapped.training_step(
                target_img=target,
                ref_imgs=refs,
                depth=depth,
                semantic=semantic,
                dynamic_mask=None,  # TODO: add when SAM3 masks are ready
            )

            loss = metrics["loss"] / accum_steps
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                # Clip gradients
                nn.utils.clip_grad_norm_(
                    [p for p in (model.module if isinstance(model, DDP) else model).parameters()
                     if p.requires_grad],
                    config["max_grad_norm"]
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Logging
                if global_step % config["log_every_steps"] == 0 and is_main_process(rank):
                    lr = scheduler.get_last_lr()[0]
                    logger.info(
                        "step=%d  loss=%.4f  flow=%.4f  x0=%.4f  cond_scale=%.4f  lr=%.2e",
                        global_step,
                        metrics["loss"].item(),
                        metrics["loss_flow"].item(),
                        metrics["loss_x0"].item(),
                        metrics["cond_scale"].item(),
                        lr,
                    )

                    if config["use_wandb"]:
                        import wandb
                        wandb.log({
                            "loss": metrics["loss"].item(),
                            "loss_flow": metrics["loss_flow"].item(),
                            "loss_x0": metrics["loss_x0"].item(),
                            "cond_scale": metrics["cond_scale"].item(),
                            "lr": lr,
                            "step": global_step,
                        })

                # Save checkpoint
                if global_step % config["save_every_steps"] == 0:
                    save_checkpoint(model, optimizer, scheduler, global_step,
                                    config, rank)

                # Debug sample
                if global_step % config["eval_every_steps"] == 0:
                    generate_debug_sample(model, loader, global_step,
                                          config, device, rank)

        epoch += 1

    # Final save
    save_checkpoint(model, optimizer, scheduler, global_step, config, rank)
    logger.info("Training complete at step %d", global_step)

    cleanup_distributed()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train FLUX NVS")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA fine-tuning")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.lora:
        config["use_lora"] = True
    if args.lr is not None:
        config["learning_rate"] = args.lr
    if args.batch_size is not None:
        config["batch_size_per_gpu"] = args.batch_size
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.resume is not None:
        config["resume_from"] = args.resume
    if args.wandb:
        config["use_wandb"] = True

    train(config)


if __name__ == "__main__":
    main()
