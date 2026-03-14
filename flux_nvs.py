#!/usr/bin/env python3
"""
flux_nvs.py — FLUX-based Novel View Synthesis for USCILab3D
=============================================================
Backbone:   FLUX.1-dev (12B DiT, pretrained)
Input:      target depth (static, from LiDAR) +
            target semantic (static, from SAM3) +
            4 reference RGBs (KD-tree cross-trajectory retrieval)
Output:     target RGB image (1280x720)

Key design choices:
  - Pretrained FLUX.1-dev backbone (frozen initially, then fine-tuned)
  - ControlNet-style conditioning: depth + semantic → parallel DiT branch
  - IP-Adapter-style reference injection: 4 ref images → CLIP/VAE embeddings
    → cross-attention injection into the main transformer
  - Rectified flow matching (FLUX native, NOT epsilon/v-prediction)
  - bf16 training on H100/H200 (bf16 required for FLUX stability)
  - Dynamic masking: zero out dynamic regions in depth/semantic conditioning

Compatible with Python 3.9+.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FluxNVSConfig:
    """All hyperparameters in one place."""

    # --- Backbone ---
    flux_model_id: str = "black-forest-labs/FLUX.1-dev"
    flux_dtype: str = "bfloat16"  # MUST be bf16 for training stability

    # --- Image / latent dimensions ---
    img_height: int = 720
    img_width: int = 1280
    # FLUX packs 2x2 patches → latent is H/16, W/16 in packed form
    # VAE downsamples 8x → latent is H/8, W/8 before packing
    vae_scale_factor: int = 8
    patch_size: int = 2  # FLUX packs latents in 2x2 patches

    # --- Conditioning ---
    n_ref_images: int = 4
    depth_channels: int = 1
    semantic_channels: int = 3  # RGB semantic from SAM3
    cond_channels: int = 4      # depth(1) + semantic(3)

    # --- ControlNet-style condition encoder ---
    cond_embed_dim: int = 64    # hidden dim inside condition encoder

    # --- Reference image adapter ---
    ref_proj_dim: int = 768     # project ref features to this dim
    ref_num_tokens: int = 16    # number of tokens per reference image

    # --- Training ---
    learning_rate: float = 1e-5
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    num_train_timesteps: int = 1000
    guidance_scale: float = 1.0  # guidance embedding during training

    # --- Loss ---
    # FLUX uses flow matching: target = noise - clean
    # Optionally add a weighted x0-reconstruction term
    lambda_x0_recon: float = 0.0   # set > 0 to add L1 x0-recon loss

    # --- Dynamic masking ---
    use_dynamic_mask: bool = True  # zero-out dynamic pixels in cond

    # --- LoRA (optional) ---
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 64


# ---------------------------------------------------------------------------
# Condition Encoder: depth + semantic → latent-space control signal
# ---------------------------------------------------------------------------

class ConditionEncoder(nn.Module):
    """
    Encode depth (1ch) + semantic (3ch) maps into a latent-space control
    signal that matches FLUX's packed latent shape.

    Input:  (B, 4, H, W) where H,W = full image resolution
    Output: (B, (H/16)*(W/16), C_packed) matching FLUX's hidden_states shape
    """

    def __init__(self, config: FluxNVSConfig):
        super().__init__()
        C = config.cond_embed_dim
        in_ch = config.cond_channels  # 4 = depth(1) + semantic(3)
        latent_ch = 16  # FLUX latent channels (after packing: 4ch * 2*2 = 16)

        # Downsample from full res to latent res (÷8) via strided convs
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, C, 4, stride=2, padding=1),      # ÷2
            nn.SiLU(),
            nn.GroupNorm(8, C),
            nn.Conv2d(C, C * 2, 4, stride=2, padding=1),      # ÷4
            nn.SiLU(),
            nn.GroupNorm(8, C * 2),
            nn.Conv2d(C * 2, C * 4, 4, stride=2, padding=1),  # ÷8 → latent res
            nn.SiLU(),
            nn.GroupNorm(8, C * 4),
            nn.Conv2d(C * 4, latent_ch, 1),                    # project to 16ch
        )
        # Zero-init the final conv so conditioning starts as identity
        nn.init.zeros_(self.encoder[-1].weight)
        nn.init.zeros_(self.encoder[-1].bias)

    def forward(self, depth, semantic, dynamic_mask=None):
        """
        depth:        (B, 1, H, W)
        semantic:     (B, 3, H, W)
        dynamic_mask: (B, 1, H, W) binary, 1 = dynamic (to zero out)

        Returns: (B, L, 16) where L = (H/16)*(W/16) — packed latent tokens
        """
        cond = torch.cat([depth, semantic], dim=1)  # (B, 4, H, W)

        if dynamic_mask is not None:
            # Zero out dynamic regions in ALL condition channels
            static_mask = 1.0 - dynamic_mask  # 1 = static, 0 = dynamic
            cond = cond * static_mask

        z_cond = self.encoder(cond)  # (B, 16, H/8, W/8)

        # Pack into FLUX format: (B, 16, H/8, W/8) → (B, (H/16)*(W/16), 16*4)
        # Wait — FLUX packs (B, C, H, W) → (B, H/2 * W/2, C * 2 * 2)
        # Our z_cond is at latent res (H/8, W/8), so pack with patch_size=2:
        z_cond = rearrange(
            z_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            ph=2, pw=2
        )  # (B, (H/16)*(W/16), 64)

        return z_cond


# ---------------------------------------------------------------------------
# Reference Image Adapter: 4 ref images → cross-attention tokens
# ---------------------------------------------------------------------------

class ReferenceAdapter(nn.Module):
    """
    Encode 4 reference RGB images into a set of tokens that can be
    injected into FLUX's encoder_hidden_states (alongside T5 text tokens).

    Uses frozen FLUX VAE to encode refs, then a learnable projector
    to create compact reference tokens.
    """

    def __init__(self, config: FluxNVSConfig):
        super().__init__()
        self.n_refs = config.n_ref_images
        self.n_tokens = config.ref_num_tokens

        # VAE latent channels for FLUX = 16 (after packing 4ch * 2x2)
        # Before packing: 4 channels. After VAE + scale/shift.
        # We'll process pre-packed VAE latents: (B*N_ref, 4, H/8, W/8)

        # Reduce spatial dims and project to token space
        # Input: (B*N_ref, 4, H/8, W/8) where H/8=90, W/8=160
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(4, 32, 4, stride=4, padding=0),  # → (B, 32, H/32, W/32)
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=4, padding=0),  # → (B, 64, H/128, W/128)
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                # → (B, 64, 4, 4) = 1024 total
        )

        # Project flattened features to ref_num_tokens × joint_attention_dim
        # FLUX joint_attention_dim = 4096 for the full model
        # We'll use a projection to match whatever dim FLUX expects
        self.token_proj = nn.Sequential(
            nn.Linear(64 * 4 * 4, 512),
            nn.SiLU(),
            nn.Linear(512, self.n_tokens * config.ref_proj_dim),
        )
        self.proj_dim = config.ref_proj_dim

        # Final projection to FLUX's joint_attention_dim (set after loading FLUX)
        # Will be initialized in set_flux_dim()
        self.flux_proj = None

    def set_flux_dim(self, joint_attention_dim: int):
        """Called after loading FLUX to match its hidden dim."""
        self.flux_proj = nn.Linear(self.proj_dim, joint_attention_dim)
        # Zero-init so refs start as no-op
        nn.init.zeros_(self.flux_proj.weight)
        nn.init.zeros_(self.flux_proj.bias)

    def forward(self, ref_latents: List[torch.Tensor]):
        """
        ref_latents: list of N_ref tensors, each (B, 4, H/8, W/8)
                     Already encoded by frozen VAE + scaled/shifted.

        Returns: (B, N_ref * n_tokens, joint_attention_dim)
        """
        B = ref_latents[0].shape[0]

        all_tokens = []
        for z_ref in ref_latents:
            # (B, 4, H/8, W/8) → spatial reduce
            feat = self.spatial_reduce(z_ref)    # (B, 64, 4, 4)
            feat = feat.flatten(1)                # (B, 1024)
            tokens = self.token_proj(feat)        # (B, n_tokens * proj_dim)
            tokens = tokens.view(B, self.n_tokens, self.proj_dim)
            all_tokens.append(tokens)

        # Concatenate all reference tokens
        ref_tokens = torch.cat(all_tokens, dim=1)  # (B, N_ref * n_tokens, proj_dim)

        # Project to FLUX dim
        if self.flux_proj is not None:
            ref_tokens = self.flux_proj(ref_tokens)

        return ref_tokens


# ---------------------------------------------------------------------------
# Main Model: FluxNVS
# ---------------------------------------------------------------------------

class FluxNVS(nn.Module):
    """
    FLUX-based Novel View Synthesizer.

    Wraps the pretrained FLUX.1-dev transformer with:
      1. ConditionEncoder for depth+semantic maps
      2. ReferenceAdapter for cross-trajectory reference images
      3. Frozen VAE for encoding/decoding
      4. Flow matching training objective

    The FLUX transformer itself is fine-tuned (full or LoRA).
    """

    def __init__(self, config: FluxNVSConfig):
        super().__init__()
        self.config = config

        # These will be set by load_pretrained()
        self.transformer = None
        self.vae = None
        self.vae_scaling_factor = None
        self.vae_shift_factor = None

        # Learnable components
        self.cond_encoder = ConditionEncoder(config)
        self.ref_adapter = ReferenceAdapter(config)

        # Condition injection scale (learnable, starts at 0)
        self.cond_scale = nn.Parameter(torch.tensor(0.0))

    def load_pretrained(self, device="cuda"):
        """
        Load pretrained FLUX.1-dev components.
        Call this AFTER moving to device.
        """
        from diffusers import AutoencoderKL, FluxTransformer2DModel

        dtype = getattr(torch, self.config.flux_dtype)
        model_id = self.config.flux_model_id

        logger.info("Loading FLUX VAE from %s ...", model_id)
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=dtype,
        ).to(device)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.vae_scaling_factor = self.vae.config.scaling_factor
        self.vae_shift_factor = self.vae.config.shift_factor

        logger.info("Loading FLUX Transformer from %s ...", model_id)
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            torch_dtype=dtype,
        ).to(device)

        # Get FLUX's joint_attention_dim and set ref adapter projection
        flux_dim = self.transformer.config.joint_attention_dim
        self.ref_adapter.set_flux_dim(flux_dim)
        self.ref_adapter = self.ref_adapter.to(device=device, dtype=dtype)
        self.cond_encoder = self.cond_encoder.to(device=device, dtype=dtype)
        self.cond_scale = self.cond_scale.to(device=device, dtype=dtype)

        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()

        logger.info("FLUX NVS model loaded. Transformer params: %.1fM",
                     sum(p.numel() for p in self.transformer.parameters()) / 1e6)

    def _apply_lora(self):
        """Apply LoRA adapters to transformer attention layers."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
            ],
            lora_dropout=0.0,
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        trainable = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.transformer.parameters())
        logger.info("LoRA applied: %.2fM / %.1fM trainable (%.1f%%)",
                     trainable / 1e6, total / 1e6, 100 * trainable / total)

    # ----- VAE helpers -----

    @torch.no_grad()
    def encode_image(self, img_01: torch.Tensor) -> torch.Tensor:
        """
        Encode image to FLUX latent space.
        img_01: (B, 3, H, W) in [0, 1]
        Returns: (B, 4, H/8, W/8) scaled and shifted latent
        """
        x = img_01 * 2.0 - 1.0  # [0,1] → [-1,1]
        z = self.vae.encode(x).latent_dist.sample()
        z = (z - self.vae_shift_factor) * self.vae_scaling_factor
        return z

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode FLUX latent to image.
        z: (B, 4, H/8, W/8) scaled and shifted latent
        Returns: (B, 3, H, W) in [0, 1]
        """
        z = z / self.vae_scaling_factor + self.vae_shift_factor
        x = self.vae.decode(z).sample
        return ((x + 1.0) / 2.0).clamp(0, 1)

    # ----- Pack / Unpack latents for FLUX -----

    @staticmethod
    def pack_latents(z: torch.Tensor) -> torch.Tensor:
        """
        Pack (B, C, H, W) → (B, H/2 * W/2, C * 4) for FLUX transformer.
        """
        return rearrange(z, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

    @staticmethod
    def unpack_latents(z_packed: torch.Tensor, h: int, w: int, c: int = 4) -> torch.Tensor:
        """
        Unpack (B, L, C*4) → (B, C, H, W).
        h, w = latent height/2, latent width/2 (packed dims)
        """
        return rearrange(
            z_packed, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=h, w=w, ph=2, pw=2, c=c
        )

    # ----- Position IDs -----

    @staticmethod
    def make_img_ids(h_packed: int, w_packed: int, device, dtype) -> torch.Tensor:
        """
        FLUX image position IDs: (h_packed * w_packed, 3)
        """
        img_ids = torch.zeros(h_packed, w_packed, 3, device=device, dtype=dtype)
        img_ids[..., 1] = torch.arange(h_packed, device=device, dtype=dtype)[:, None]
        img_ids[..., 2] = torch.arange(w_packed, device=device, dtype=dtype)[None, :]
        return rearrange(img_ids, "h w c -> (h w) c")

    @staticmethod
    def make_txt_ids(seq_len: int, device, dtype) -> torch.Tensor:
        """FLUX text position IDs: (seq_len, 3) of zeros."""
        return torch.zeros(seq_len, 3, device=device, dtype=dtype)

    # ----- Build context (replaces T5 text embeddings with our conditioning) -----

    def build_encoder_hidden_states(
        self,
        ref_latents: List[torch.Tensor],
        cond_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build encoder_hidden_states and pooled_projections for FLUX.

        Instead of T5 text tokens, we use:
          - Reference image tokens (from ReferenceAdapter)
          - Condition tokens (from ConditionEncoder) — optionally

        Returns:
          encoder_hidden_states: (B, L_ctx, joint_attention_dim)
          pooled_projections:    (B, joint_attention_dim)
        """
        # Reference tokens: (B, N_ref * n_tokens, flux_dim)
        ref_tokens = self.ref_adapter(ref_latents)

        # Pooled projection: mean-pool ref tokens
        pooled = ref_tokens.mean(dim=1)  # (B, flux_dim)

        # encoder_hidden_states = just ref tokens
        # (condition is injected by adding cond_tokens to hidden_states)
        return ref_tokens, pooled

    # ----- Training forward -----

    def training_step(
        self,
        target_img: torch.Tensor,       # (B, 3, H, W) in [0, 1]
        ref_imgs: List[torch.Tensor],    # N_ref × (B, 3, H, W) in [0, 1]
        depth: torch.Tensor,             # (B, 1, H, W) normalized [0, 1]
        semantic: torch.Tensor,          # (B, 3, H, W) normalized [0, 1]
        dynamic_mask: Optional[torch.Tensor] = None,  # (B, 1, H, W) binary
    ) -> dict:
        """
        One training step. Returns dict with 'loss' and diagnostics.

        Uses FLUX rectified flow matching:
          zt = (1 - t) * z0 + t * noise
          target = noise - z0
          loss = MSE(pred, target)
        """
        device = target_img.device
        dtype = next(self.transformer.parameters()).dtype
        B = target_img.shape[0]

        # 1) Encode target to latent
        z0 = self.encode_image(target_img.to(dtype))  # (B, 4, H/8, W/8)

        # 2) Encode reference images
        ref_latents = [self.encode_image(r.to(dtype)) for r in ref_imgs]

        # 3) Encode depth + semantic conditions
        cond_tokens = self.cond_encoder(
            depth.to(dtype), semantic.to(dtype),
            dynamic_mask.to(dtype) if dynamic_mask is not None else None
        )  # (B, L_packed, 64)

        # 4) Build encoder hidden states from references
        encoder_hidden_states, pooled_projections = self.build_encoder_hidden_states(
            ref_latents, cond_tokens
        )

        # 5) Sample timesteps (logit-normal distribution, FLUX style)
        # t ~ sigmoid(Normal(0, 1))
        u = torch.randn(B, device=device, dtype=dtype)
        t_01 = torch.sigmoid(u)  # (B,) in (0, 1)
        timesteps = t_01 * self.config.num_train_timesteps  # scale for transformer

        # 6) Create noisy latents via rectified flow
        noise = torch.randn_like(z0)
        t_expand = t_01.view(B, 1, 1, 1)
        z_t = (1.0 - t_expand) * z0 + t_expand * noise  # (B, 4, H/8, W/8)

        # 7) Add condition signal to noisy latents
        # Pack z_t for FLUX
        h_lat, w_lat = z0.shape[2], z0.shape[3]
        h_packed, w_packed = h_lat // 2, w_lat // 2

        z_t_packed = self.pack_latents(z_t)  # (B, L, 16)

        # Add scaled condition tokens
        # cond_tokens is (B, L_packed, 64) — need to match z_t_packed's 16 channels
        # Actually cond_tokens shape: (B, h_packed*w_packed, 16*4=64)
        # We need to add it to z_t_packed which is (B, h_packed*w_packed, 16)
        # Use a learnable scale and match dims
        if cond_tokens.shape[-1] != z_t_packed.shape[-1]:
            # Truncate or pad cond_tokens to match packed dim
            # The condition encoder outputs 16ch, packed to 64.
            # z_t_packed is 4ch packed to 16. We add cond as residual.
            # Project cond_tokens down to match
            if not hasattr(self, '_cond_proj'):
                self._cond_proj = nn.Linear(
                    cond_tokens.shape[-1], z_t_packed.shape[-1]
                ).to(device=device, dtype=dtype)
                nn.init.zeros_(self._cond_proj.weight)
                nn.init.zeros_(self._cond_proj.bias)
            cond_signal = self._cond_proj(cond_tokens)
        else:
            cond_signal = cond_tokens

        hidden_states = z_t_packed + self.cond_scale * cond_signal

        # 8) Position IDs
        img_ids = self.make_img_ids(h_packed, w_packed, device, dtype)
        txt_ids = self.make_txt_ids(encoder_hidden_states.shape[1], device, dtype)

        # 9) Guidance embedding
        guidance = torch.full((B,), self.config.guidance_scale,
                              device=device, dtype=dtype)

        # 10) Forward through FLUX transformer
        noise_pred_packed = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timesteps / self.config.num_train_timesteps,
            txt_ids=txt_ids,
            img_ids=img_ids,
            guidance=guidance,
            return_dict=False,
        )[0]  # (B, L, 16)

        # 11) Unpack prediction
        noise_pred = self.unpack_latents(noise_pred_packed, h_packed, w_packed, c=4)

        # 12) Compute loss (flow matching: target = noise - z0)
        target = noise - z0
        loss_flow = F.mse_loss(noise_pred.float(), target.float())

        loss = loss_flow

        # Optional x0-reconstruction loss
        loss_x0 = torch.tensor(0.0, device=device)
        if self.config.lambda_x0_recon > 0:
            # x0_pred = z_t - t * noise_pred  (from rectified flow)
            x0_pred = z_t - t_expand * noise_pred
            loss_x0 = F.l1_loss(x0_pred.float(), z0.float())
            loss = loss + self.config.lambda_x0_recon * loss_x0

        return {
            "loss": loss,
            "loss_flow": loss_flow.detach(),
            "loss_x0": loss_x0.detach(),
            "cond_scale": self.cond_scale.detach(),
        }

    # ----- Inference -----

    @torch.no_grad()
    def sample(
        self,
        ref_imgs: List[torch.Tensor],   # N_ref × (B, 3, H, W) in [0, 1]
        depth: torch.Tensor,             # (B, 1, H, W) normalized
        semantic: torch.Tensor,          # (B, 3, H, W) normalized
        dynamic_mask: Optional[torch.Tensor] = None,
        num_steps: int = 28,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Sample a novel view.
        Returns: (B, 3, H, W) in [0, 1]
        """
        device = depth.device
        dtype = next(self.transformer.parameters()).dtype
        B = depth.shape[0]
        H, W = self.config.img_height, self.config.img_width
        h_lat, w_lat = H // self.config.vae_scale_factor, W // self.config.vae_scale_factor
        h_packed, w_packed = h_lat // 2, w_lat // 2

        # Encode refs and conditions
        ref_latents = [self.encode_image(r.to(dtype)) for r in ref_imgs]
        cond_tokens = self.cond_encoder(
            depth.to(dtype), semantic.to(dtype),
            dynamic_mask.to(dtype) if dynamic_mask is not None else None
        )

        encoder_hidden_states, pooled_projections = self.build_encoder_hidden_states(
            ref_latents, cond_tokens
        )

        img_ids = self.make_img_ids(h_packed, w_packed, device, dtype)
        txt_ids = self.make_txt_ids(encoder_hidden_states.shape[1], device, dtype)

        # Start from pure noise
        z_t = torch.randn(B, 4, h_lat, w_lat, device=device, dtype=dtype)

        # Linear timestep schedule from 1.0 → 0.0
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_curr  # negative (going from 1 → 0)

            z_t_packed = self.pack_latents(z_t)

            # Add conditioning
            if hasattr(self, '_cond_proj') and self._cond_proj is not None:
                cond_signal = self._cond_proj(cond_tokens)
            else:
                cond_signal = cond_tokens
            hidden_states = z_t_packed + self.cond_scale * cond_signal

            guidance = torch.full((B,), guidance_scale, device=device, dtype=dtype)
            t_batch = torch.full((B,), t_curr, device=device, dtype=dtype)

            # Predict velocity
            v_pred_packed = self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=t_batch,
                txt_ids=txt_ids,
                img_ids=img_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

            v_pred = self.unpack_latents(v_pred_packed, h_packed, w_packed, c=4)

            # Euler step: z_{t+dt} = z_t + dt * v_pred
            z_t = z_t + dt * v_pred

        # Decode final latent
        rgb = self.decode_latent(z_t)
        return rgb


# ---------------------------------------------------------------------------
# Convenience: build model
# ---------------------------------------------------------------------------

def build_model(config: Optional[FluxNVSConfig] = None,
                device: str = "cuda") -> FluxNVS:
    """Build and load the FluxNVS model."""
    if config is None:
        config = FluxNVSConfig()
    model = FluxNVS(config)
    model.load_pretrained(device=device)
    return model
