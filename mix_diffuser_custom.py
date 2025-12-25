import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import numpy as np
from pathlib import Path as _Path
from PIL import Image as _Image
import torchvision.transforms as _T
import cv2
import os
import argparse
from pathlib import Path as _Path
from PIL import Image as _Image
import random as _random
import re as _re


# Positional / timestep embedding
def sinusoidal_embedding(timesteps, dim):
    """
    timesteps: [B] tensor of ints or floats
    returns: [B, dim]
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# Stable Diffusion VAE wrapper
class SDVAE(nn.Module):
    SD_LATENT_SCALE = 0.18215

    def __init__(self, torch_dtype=torch.float32, device="cuda"):
        super().__init__()
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder="vae",
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()

    @staticmethod
    def _to_vae_range(x_01):
        return x_01 * 2.0 - 1.0

    @staticmethod
    def _to_image_range(x_m11):
        return (x_m11 + 1.0) / 2.0

    def encode(self, x_01):
        x_m11 = self._to_vae_range(x_01)
        posterior = self.vae.encode(x_m11).latent_dist
        z = posterior.mean * self.SD_LATENT_SCALE
        return z

    def decode(self, z):
        z_dec = z / self.SD_LATENT_SCALE
        x_m11 = self.vae.decode(z_dec).sample
        return self._to_image_range(x_m11).clamp(0, 1)


# Cross-attention utilities used by custom UNet
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, q_input, kv_input):
        B, Nq, D = q_input.shape
        Nk = kv_input.shape[1]
        Q = self.to_q(q_input).view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.to_k(kv_input).view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.to_v(kv_input).view(B, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = out.permute(0, 2, 1, 3).reshape(B, Nq, D)
        return self.to_out(out)

class ReferenceMixingLayer(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, ref_tokens, qry_tokens):
        h = self.cross_attn(qry_tokens, ref_tokens)
        x = self.norm1(qry_tokens + h)
        h2 = self.ff(x)
        return self.norm2(x + h2)

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False, num_heads=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.use_attn = use_attn
        if use_attn:
            self.mix = ReferenceMixingLayer(out_ch, num_heads)

    def forward(self, x, ref_tokens=None, t_emb=None):
        h = self.act(self.norm1(self.conv1(x)))
        if t_emb is not None:
            h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(self.norm2(self.conv2(h)))
        if self.use_attn and ref_tokens is not None:
            B, C, H, W = h.shape
            tokens = h.flatten(2).permute(0, 2, 1)
            tokens = self.mix(ref_tokens, tokens)
            h = tokens.permute(0, 2, 1).view(B, C, H, W)
        return h

class CrossAttentionUNet(nn.Module):
    def __init__(self, in_ch=4, base_ch=128, time_dim=256, num_heads=4, use_ref=False):

        super().__init__()
        self.time_dim = time_dim
        self.num_heads = num_heads

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

        # Encoder
        self.enc1 = UNetBlock(in_ch, base_ch, time_dim)
        self.enc2 = UNetBlock(base_ch, base_ch * 2, time_dim)
        self.enc3 = UNetBlock(base_ch * 2, base_ch * 4, time_dim)

        # Bottleneck cross-attn mixing
        self.mix = ReferenceMixingLayer(base_ch * 4, num_heads=num_heads)

        # Decoder
        self.dec3 = UNetBlock(base_ch * 6, base_ch * 2, time_dim)
        self.dec2 = UNetBlock(base_ch * 3, base_ch,     time_dim)
        self.dec1 = UNetBlock(base_ch,     base_ch,     time_dim)

        self.out_conv = nn.Conv2d(base_ch, in_ch, kernel_size=1)

        self.use_ref = use_ref

    def forward(self, x, ref_tokens_dict, t):
        """
        x: [B,4,H8,W8] noisy latents (+ optional cond map added outside)
        ref_tokens_dict: dict of multi-scale tokens from RefFeatureAdapter
        t: [B] long timesteps
        """
        t_emb = sinusoidal_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        e1 = self.enc1(x, t_emb)
        e2 = self.down(e1); e2 = self.enc2(e2, t_emb)
        e3 = self.down(e2); e3 = self.enc3(e3, t_emb)

        # FIXED bottleneck for experimentation
        if self.use_ref:
            B, C, H, W = e3.shape
            qry_tokens = e3.flatten(2).permute(0, 2, 1)
            ref_tokens = ref_tokens_dict["bottleneck"]
            mixed_tokens = self.mix(ref_tokens, qry_tokens)
            b = mixed_tokens.permute(0, 2, 1).view(B, C, H, W)
        else:
            b = e3 
        
        # Decoder
        d3 = self.up(b)
        if d3.shape[-2:] != e2.shape[-2:]:
            d3 = F.interpolate(d3, size=e2.shape[-2:], mode="nearest")
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3, t_emb)

        # d2 should match e1 spatially
        d2 = self.up(d3)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2, t_emb)
        d1 = self.dec1(d2, t_emb)
        return self.out_conv(d1)

# Reference features → multi-scale tokens
class RefFeatureAdapter(nn.Module):
    def __init__(self, in_ch=4, base_ch=128):
        super().__init__()
        self.target_specs = {
            "enc1": base_ch,
            "enc2": base_ch * 2,
            "dec2": base_ch,
            "dec3": base_ch * 2,
            "bottleneck": base_ch * 4
        }
        self.proj = nn.ModuleDict({
            k: nn.Conv2d(in_ch, c, 1) for k, c in self.target_specs.items()
        })

    @staticmethod
    def _tokens_from_map(feat_map):
        B, C, H, W = feat_map.shape
        return feat_map.flatten(2).permute(0, 2, 1)

    def forward(self, z_ref_map, base_hw):
        H8, W8 = base_hw
        sizes = {
            "enc1": (H8, W8),
            "enc2": (H8 // 2, W8 // 2),
            "dec2": (H8 // 2, W8 // 2),
            "dec3": (H8 // 4, W8 // 4),
            "bottleneck": (H8 // 8, W8 // 8)
        }
        out = {}
        for k, tgt_ch in self.target_specs.items():
            Ht, Wt = sizes[k]
            pooled = F.adaptive_avg_pool2d(z_ref_map, (Ht, Wt))
            proj = self.proj[k](pooled)
            out[k] = self._tokens_from_map(proj)
        return out



# CNN downsampler
class CNNDownsampler(nn.Module):
    def __init__(self, in_channels=4, latent_ch=4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(64, latent_ch, 4, 2, 1)
        )

    def forward(self, x):
        return self.model(x)

class DiffusionLatentFusionSynthesizer(nn.Module):
    def __init__(self, base_ch=128, num_heads=4, time_dim=256,
                 device="cuda", vae_dtype=torch.float32, num_inference_steps=50):
        super().__init__()
        self.cond_norm  = nn.GroupNorm(1, 4).to(device)
        self.cond_scale = nn.Parameter(torch.tensor(0.1, device = device))
        self.device = device
        self.vae = SDVAE(torch_dtype=vae_dtype, device=device)
        self.cond_encoder = CNNDownsampler(in_channels=4, latent_ch=4).to(device)
        self.ref_adapter = RefFeatureAdapter(base_ch=base_ch).to(device)
        self.sd_unet = CrossAttentionUNet(
            in_ch=4,
            base_ch=base_ch,
            time_dim=time_dim,
            num_heads=num_heads,
            use_ref=True
        ).to(device)

        # DDIM scheduler
        self.scheduler = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        self.num_inference_steps = num_inference_steps
        self.ref_reduce = nn.Conv2d(4 * 4, 4, kernel_size=1).to(device)  # 4 refs * 4 latent ch

    def encode_references(self, refs_01):
        with torch.no_grad():
            zs = [self.vae.encode(r.to(self.device)) for r in refs_01]  # each [B,4,H8,W8]
        z_stack = torch.cat(zs, dim=1)          # [B,16,H8,W8]
        z_ref_map = self.ref_reduce(z_stack)    # [B,4,H8,W8]
        return z_ref_map
    
    def sample(
        self,
        refs_01,
        depth,
        semantic,
        height,
        width,
        zero_refs: bool = False,
    ):
        """
        refs_01: list of [B,3,H,W] refs (0..1 or 0..255)
        depth:   [B,1,H,W]
        semantic:[B,1 or 3,H,W]
        zero_refs: if True, ignore image content and use all-zero ref latents
        """
        B = depth.shape[0]
        device = self.device

        refs_01 = [
            r.to(device).float() / 255.0
            if r.dtype in (torch.uint8, torch.int8, torch.int16)
            else r.to(device).float()
            for r in refs_01
        ]
        depth = depth.to(device).float()
        semantic = semantic.to(device).float()

        # reference features to tokens dict at multiple scales
        H8, W8 = height // 8, width // 8

        if zero_refs:
            # ignore content, just shape + zeros
            z_ref_map = torch.zeros(B, 4, H8, W8, device=device)
        else:
            z_ref_map = self.encode_references(refs_01)  # [B,4,H/8,W/8]

        ref_tokens_dict = self.ref_adapter(z_ref_map, (H8, W8))  # dict of [B,L,C]

        # depth+semantic to latent map aligned with z
        cond_in = torch.cat([depth, semantic], dim=1)
        z_cond_map = self.cond_encoder(cond_in)                  # [B,4,H/8,W/8]

        latents = torch.randn(B, 4, H8, W8, device=device)
        self.scheduler.set_timesteps(self.num_inference_steps, device=device)

        for t_scalar in self.scheduler.timesteps:
            t = torch.full((B,), int(t_scalar), device=device, dtype=torch.long)

            latents_in = latents + self.cond_scale * self.cond_norm(z_cond_map)

            noise_pred = self.sd_unet(
                latents_in,
                ref_tokens_dict,
                t,
            )

            latents = self.scheduler.step(noise_pred, t_scalar, latents).prev_sample

        rgb = self.vae.decode(latents)
        return rgb

# Training loop updated to optimize SD UNet directly
import torchvision.transforms.functional as TF

def save_debug_sample(model, refs_paths, depth_path, sem_path,
                      device, epoch, out_dir="debug_samples"):
    os.makedirs(out_dir, exist_ok=True)

    refs = [load_rgb(_Path(p)).to(device) for p in refs_paths]

    _, _, H, W = refs[0].shape

    depth_raw = load_gray(_Path(depth_path))
    depth_smoothed = extract_pseudo_depth(depth_raw)
    depth = TF.resize(depth_smoothed.to(device), [H, W], antialias=True).float()
    semantic_raw = load_semantic_rgb(_Path(sem_path))
    semantic = TF.resize(
        semantic_raw.to(device),
        [H, W],
        interpolation=TF.InterpolationMode.NEAREST
    ).float()

    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        rgb = model.sample(refs, depth, semantic, H, W)

    out_path = f"{out_dir}/epoch_{epoch:03d}_flux.png"
    _T.ToPILImage()(rgb[0].cpu()).save(out_path)

    print(f"[debug] Saved sample for epoch {epoch} → {out_path}")


def save_debug_sample_from_tensors(model, refs, depth, semantic, target, device, epoch, out_dir="debug_overfit"):
    os.makedirs(out_dir, exist_ok=True)

    B, _, H, W = target.shape

    with torch.no_grad():
        rgb = model.sample(refs, depth, semantic, H, W)

    out_path = f"{out_dir}/epoch_{epoch:03d}.png"
    _T.ToPILImage()(rgb[0].cpu()).save(out_path)
    print(f"[debug] Saved overfit sample → {out_path}")


def train_diffusion(model, dataloader, num_epochs=10, lr=1e-4, device="cuda"):
    """
    dataloader should yield batches like: (refs, depth, semantic, target)
      - refs: list/tuple of length N_ref, each [B,3,H,W] in 0..255 uint8 or 0..1 float
      - depth: [B,1,H,W]
      - semantic: [B,1,H,W] or [B,3,H,W]
      - target: [B,3,H,W]
    """
    # --- optimizer ---
    params = [
        {"params": model.sd_unet.parameters(),      "lr": lr},
        {"params": model.cond_encoder.parameters(), "lr": lr},
        {"params": model.cond_norm.parameters(),    "lr": lr},
        {"params": [model.cond_scale],              "lr": lr},
        {"params": model.ref_adapter.parameters(),  "lr": lr},
        {"params": model.ref_reduce.parameters(),   "lr": lr},
    ]
    optimizer = optim.AdamW(params, lr=lr)
    scheduler = model.scheduler

    T_total = scheduler.config.num_train_timesteps
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T_total]

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            refs, depth, semantic, target = batch

            refs = [
                r.to(device).float() / 255.0
                if r.dtype in (torch.uint8, torch.int8, torch.int16)
                else r.to(device).float()
                for r in refs
            ]
            depth    = depth.to(device).float()
            semantic = semantic.to(device).float()
            target   = (
                target.to(device).float() / 255.0
                if target.dtype in (torch.uint8, torch.int8, torch.int16)
                else target.to(device).float()
            )

            with torch.no_grad():
                z0 = model.vae.encode(target)   # [B,4,H/8,W/8]

            B, C, H8, W8 = z0.shape

            t = torch.randint(0, T_total, (B,), device=device).long()    # [B]
            alpha_bar = alphas_cumprod[t].view(B, 1, 1, 1)               # [B,1,1,1]

            noise         = torch.randn_like(z0)
            sqrt_ab       = alpha_bar.sqrt()
            sqrt_1mab     = (1.0 - alpha_bar).sqrt()
            noisy_latents = sqrt_ab * z0 + sqrt_1mab * noise

            # encode conditioning
            cond_in    = torch.cat([depth, semantic], dim=1)  # [B,4,H,W]
            z_cond_map = model.cond_encoder(cond_in)          # [B,4,H/8,W/8]

            # encode references
            z_ref_map      = model.encode_references(refs)    # [B,4,H/8,W/8]
            ref_tokens_dict = model.ref_adapter(z_ref_map, (H8, W8))

            # UNet input: noisy_latents + cond 
            latents_with_cond = noisy_latents + model.cond_scale * model.cond_norm(z_cond_map)

            noise_pred = model.sd_unet(
                latents_with_cond,
                ref_tokens_dict,
                t,
            )  

            # epsilon loss
            loss_eps = F.mse_loss(noise_pred, noise)

            # x0 reconstruction loss (latent space)
            x0_pred    = (noisy_latents - sqrt_1mab * noise_pred) / (sqrt_ab + 1e-8)
            lambda_lat = 0.05
            loss_lat   = F.l1_loss(x0_pred, z0)

            loss = loss_eps + lambda_lat * loss_lat

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

            # optional: occasional x0 debug for the *current* batch
            if batch_idx % 390 == 0:
                model.eval()
                with torch.no_grad():
                    recon     = model.vae.decode(x0_pred)  # [B,3,H,W]
                    target_im = model.vae.decode(z0)

                os.makedirs("debug_x0", exist_ok=True)
                _T.ToPILImage()(recon[0].cpu().clamp(0,1)).save(
                    f"debug_x0/epoch{epoch:02d}_batch{batch_idx:04d}_recon.png"
                )
                _T.ToPILImage()(target_im[0].cpu().clamp(0,1)).save(
                    "debug_x0/target.png"
                )
                model.train()

        print(f"Epoch {epoch} finished.")

        '''
                save_debug_sample(
                    model,
                    refs_paths=[
                        "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00001.jpg",
                        "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00050.jpg",
                        "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00100.jpg",
                        "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00200.jpg"
                    ],
                    depth_path="/lab/student/DIFFU/train_data/15-deg-left-1/depth_00340.png",
                    sem_path="/lab/student/DIFFU/train_data/15-deg-left-1/classgt_00340.png",
                    device=device,
                    epoch=epoch,
                    out_dir="debug_samples"
                )
                '''
    '''
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, batch in enumerate(dataloader):
            # Unpack dataset batch
            refs, depth, semantic, target = batch
            # Normalize
            refs = [r.to(device).float() / 255.0 if r.dtype in (torch.uint8, torch.int8, torch.int16) else r.to(device).float() for r in refs]
            depth = depth.to(device).float()
            semantic = semantic.to(device).float()
            target = target.to(device).float() / 255.0 if target.dtype in (torch.uint8, torch.int8, torch.int16) else target.to(device).float()

            # Encode target image to latent
            with torch.no_grad():
                z0 = model.vae.encode(target)  # [B,4,H/8,W/8]

            B, C, H8, W8 = z0.shape

            # random timestep
            t = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device).long()

            # noise and add to latent
            noise = torch.randn_like(z0)
            noisy_latents = scheduler.add_noise(z0, noise, t)

            # encode references & conditions
            z_ref_map = model.encode_references(refs)               # [B,4,H/8,W/8]
            # refs -> multi-scale tokens dict
            ref_tokens_dict = model.ref_adapter(z_ref_map, (H8, W8))
            cond_in = torch.cat([depth, semantic], dim=1)
            z_cond_map = model.cond_encoder(cond_in)                # [B,4,H/8,W/8]

            # custom UNet predicts noise directly
            noise_pred = model.sd_unet(
                noisy_latents + (model.cond_scale * model.cond_norm(z_cond_map)),
                ref_tokens_dict,
                t
            )  # [B,4,H8,W8]

            # after you have t, noise_pred, noise, and scheduler
            alphas_cumprod = scheduler.alphas_cumprod.to(device)              # [T]
            alpha_bar = alphas_cumprod[t].view(-1, 1, 1, 1)                  # [B,1,1,1]
            snr = alpha_bar / (1.0 - alpha_bar + 1e-8)                       # SNR_t

            # "Balanced" loss weight; clamp to avoid extremes (5.0 is a common cap)
            w = torch.minimum(snr, torch.full_like(snr, 5.0)) / (snr + 1.0)

            loss_eps = (w * (noise_pred - noise)**2).mean()

            sqrt_ab   = alpha_bar.sqrt()
            sqrt_1mab = (1.0 - alpha_bar).sqrt()
            x0_pred = (noisy_latents - sqrt_1mab * noise_pred) / (sqrt_ab + 1e-8)

            lambda_lat = 0.05  # small weight
            loss_lat = F.l1_loss(x0_pred, z0)

            loss = loss_eps + lambda_lat * loss_lat

            # Backprop
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            #if batch_idx % 100 == 0:
            print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        # ---------------------------------------------
        #  After each epoch save a debug sample
        # ---------------------------------------------
        if (epoch + 1) % 10 == 0:  # change N=1 to bigger if you want fewer samples
            save_debug_sample(
                model,
                refs_paths=[
                    "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00001.jpg",
                    "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00050.jpg",
                    "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00100.jpg",
                    "/lab/student/DIFFU/train_data/15-deg-left-1/rgb_00200.jpg"
                ],
                depth_path="/lab/student/DIFFU/train_data/15-deg-left-1/depth_00340.png",
                sem_path="/lab/student/DIFFU/train_data/15-deg-left-1/classgt_00340.png",
                device=device,
                epoch=epoch,
                out_dir="debug_samples"
            )
            '''

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _is_image(p: _Path):
    return p.suffix.lower() in IMG_EXTS and p.is_file()

def load_rgb(path: _Path):
    im = _Image.open(path).convert("RGB")
    return _T.ToTensor()(im).unsqueeze(0)  # [1,3,H,W] in 0..1

def load_gray(path: _Path):
    im = _Image.open(path).convert("L")
    return _T.ToTensor()(im).unsqueeze(0)  # [1,1,H,W] in 0..1

def load_semantic_rgb(path):
    im = _Image.open(path).convert("RGB")
    return _T.ToTensor()(im).unsqueeze(0)   # [1,3,H,W], values 0..1

def natural_key(s: str):
    # natural sort key: "img2" < "img10"
    return [int(t) if t.isdigit() else t.lower() for t in _re.split(r"(\d+)", s)]

def _gather_triplets_from_subfolder(sf: _Path):
    files = [p for p in sf.iterdir() if _is_image(p)]
    if not files:
        raise ValueError(f"No images found in {sf}")

    by_tag = {"rgb": [], "depth": [], "sem": []}
    for p in files:
        n = p.stem.lower()
        if "depth" in n or n.endswith("_d"):
            by_tag["depth"].append(p)
        elif "sem" in n or "label" in n or "classgt" in n or "class_gt" in n:
            by_tag["sem"].append(p)
        else:
            by_tag["rgb"].append(p)

    for k in by_tag:
        by_tag[k].sort(key=lambda p: natural_key(p.name))

    N = min(len(by_tag["rgb"]), len(by_tag["depth"]), len(by_tag["sem"]))
    if N == 0:
        raise ValueError(f"Could not align triplets in {sf}.")
    return list(zip(by_tag["rgb"][:N], by_tag["depth"][:N], by_tag["sem"][:N]))

def extract_pseudo_depth(depth_raw):  
    # depth_raw: [1,1,H,W] torch float in [0,1]
    arr = depth_raw.squeeze().cpu().numpy()  # [H,W]

    # Convert to uint8
    arr8 = (arr * 255).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    arr_eq = clahe.apply(arr8)  # uint8, enhanced contrast

    # Convert back to float and renormalize
    arr_f = arr_eq.astype(np.float32) / 255.0
    out = torch.from_numpy(arr_f).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

    depth_smoothed = F.avg_pool2d(out, kernel_size=5, stride=1, padding=2)

    return depth_smoothed

class FolderFiveDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, choose_random: bool = True,
                 samples_per_subfolder: int = 400):
        super().__init__()
        self.root = _Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{root} is not a directory")

        self.subfolders = sorted(
            [p for p in self.root.iterdir() if p.is_dir()],
            key=lambda p: natural_key(p.name)
        )
        if not self.subfolders:
            raise ValueError(f"No subfolders found inside {root}")

        self.triples_per_sf = []
        self.valid_subfolders = []

        for sf in self.subfolders:
            triples = _gather_triplets_from_subfolder(sf)

            # *** IMPORTANT: require at least 5 triplets ***
            if len(triples) < 5:
                print(f"[WARN] Skipping {sf}: only {len(triples)} triplets (need ≥5).")
                continue

            self.valid_subfolders.append(sf)
            self.triples_per_sf.append(triples)

        if not self.triples_per_sf:
            raise ValueError("No subfolders with at least 5 triplets were found.")

        self.choose_random = choose_random
        self.samples_per_subfolder = samples_per_subfolder

    def __len__(self):
        # Each *valid* subfolder will be sampled multiple times per epoch
        return len(self.triples_per_sf) * self.samples_per_subfolder

    def __getitem__(self, idx):
        import random

        # Map the global index back to a "valid" subfolder
        sf_idx = idx % len(self.triples_per_sf)
        triples = self.triples_per_sf[sf_idx]
        N = len(triples)
        assert N >= 5, "Logic error: we should only keep subfolders with ≥5 triplets."

        # *** IMPORTANT CHANGE: ALWAYS pick exactly 5 triplets ***
        selected = random.sample(triples, k=5)  # 5 distinct triplets

        # Choose which of those 5 is the target
        k = random.randrange(5) if self.choose_random else 4

        rgb_k_path, depth_k_path, sem_k_path = selected[k]
        target   = load_rgb(rgb_k_path)               # [1,3,H,W]
        depth_raw = load_gray(depth_k_path)
        depth = extract_pseudo_depth(depth_raw)            # [1,1,H,W]
        semantic = load_semantic_rgb(sem_k_path) # [1,3,H,W]

        # Refs = the other 4 images
        refs = [load_rgb(p) for j, (p, _, _) in enumerate(selected) if j != k]
        assert len(refs) == 4, "Each sample must have exactly 4 references."

        return refs, depth, semantic, target

def refs_collate(batch):
    """
    Collate batch of items:
      refs(list of 4 [1,3,H,W]), depth[1,1,H,W], semantic[1,1,H,W], target[1,3,H,W]
    Output:
      refs_list where each element is [B,3,H,W], depth [B,1,H,W], semantic [B,1,H,W], target [B,3,H,W]
    """
    B = len(batch)

    # Sanity check: every sample must have same number of refs (4)
    n_ref = len(batch[0][0])
    for b in range(1, B):
        assert len(batch[b][0]) == n_ref, "All samples in batch must have same # of refs"

    refs_stacked = []
    for r_idx in range(n_ref):
        tensors = [batch[b][0][r_idx] for b in range(B)]
        refs_stacked.append(torch.cat(tensors, dim=0))  # [B,3,H,W]

    depth = torch.cat([b[1] for b in batch], dim=0)     # [B,1,H,W]
    semantic = torch.cat([b[2] for b in batch], dim=0)  # [B,1,H,W]
    target = torch.cat([b[3] for b in batch], dim=0)    # [B,3,H,W]

    return refs_stacked, depth, semantic, target


def _run_infer(args):
    device = args.device
    model = DiffusionLatentFusionSynthesizer(device=device, num_inference_steps=args.steps)
    model.eval()
    # load trained weights (if provided)
    if args.weights is not None:
        ckpt = torch.load(args.weights, map_location=device, weights_only=True)
        missing = model.load_state_dict(ckpt, strict=False)
        if getattr(missing, "missing_keys", None) or getattr(missing, "unexpected_keys", None):
            print("[infer] load_state_dict(strict=False):")
            if getattr(missing, "missing_keys", None):
                print("  missing:", missing.missing_keys)
            if getattr(missing, "unexpected_keys", None):
                print("  unexpected:", missing.unexpected_keys)
        model.eval()

    refs = [load_rgb(_Path(p)).to(device) for p in args.ref]

    import torchvision.transforms.functional as TF
    _, _, H, W = refs[0].shape
    depth_raw = load_gray(_Path(args.depth))
    depth_smoothed = extract_pseudo_depth(depth_raw)
    depth    = TF.resize(depth_smoothed.to(device), [H, W], antialias=True).float()
    semantic_raw = load_semantic_rgb(_Path(args.semantic))
    semantic = TF.resize(semantic_raw.to(device), [H, W], interpolation=TF.InterpolationMode.NEAREST).float()

    _, _, H, W = refs[0].shape
    with torch.no_grad():
        rgb = model.sample(refs, depth, semantic, H, W)  # [1,3,H,W] if batch=1

    out_img = _T.ToPILImage()(rgb[0].cpu())
    out_img.save(args.out)
    print(f"[infer] saved → {args.out}")


def _run_train(args):
    device = args.device
    model = DiffusionLatentFusionSynthesizer(device=device, num_inference_steps=args.steps)

    ds = FolderFiveDataset(args.data_root, choose_random=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     collate_fn=refs_collate, drop_last=False)

    # optional resume
    if getattr(args, "resume", None):
        ckpt_path = _Path(args.resume)
        if ckpt_path.is_file():
            print(f"[train] loading checkpoint from {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    train_diffusion(model, dl, num_epochs=args.epochs, lr=args.lr, device=device)

    # save weights
    save_path = _Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[train] done. Weights saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Inference
    p_inf = subparsers.add_parser("infer", help="Run sampling without ground-truth")
    p_inf.add_argument("--ref", type=str, nargs="+", required=True, help="Path(s) to 1+ reference RGB images")
    p_inf.add_argument("--depth", type=str, required=True, help="Path to depth map for target view")
    p_inf.add_argument("--semantic", type=str, required=True, help="Path to semantic map for target view")
    p_inf.add_argument("--out", type=str, default="out.png", help="Output image path")
    p_inf.add_argument("--steps", type=int, default=50)
    p_inf.add_argument("--device", type=str, default="cuda")
    p_inf.add_argument("--weights", type=str, default=None, help="Path to trained weights produced by `train`")

    # Training
    p_tr = subparsers.add_parser("train", help="Train from a folder-of-subfolders")
    p_tr.add_argument("--data_root", type=str, required=True,
                      help="Root dir; each subfolder contains 5 RGB + 5 depth + 5 semantic")
    p_tr.add_argument("--epochs", type=int, default=10)
    p_tr.add_argument("--batch_size", type=int, default=2)
    p_tr.add_argument("--lr", type=float, default=1e-4)
    p_tr.add_argument("--device", type=str, default="cuda")
    p_tr.add_argument("--steps", type=int, default=50, help="DDIM steps during sampling (not used in training)")
    p_tr.add_argument("--save_path", type=str, default="trained_weights.pth",
                      help="File path to save trained weights (default: trained_weights.pth)")
    p_tr.add_argument("--resume", type=str, default=None,
                      help="Optional checkpoint path to resume training")

    args = parser.parse_args()

    if args.mode == "infer":
        _run_infer(args)
    elif args.mode == "train":
        _run_train(args)

if __name__ == "__main__":
    main()