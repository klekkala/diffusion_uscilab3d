
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T

@torch.no_grad()
def debug_refs_denoise_step(model, batch_A, batch_B, device,
                            t_debug: int = 400,
                            out_dir: str = "debug_refs_step"):
    """
    Check effect of references in the SAME one-step denoising regime
    as training (x_t -> x0), instead of full DDIM sampling.

    batch_A, batch_B: (refs_list, depth, semantic, target) from your DataLoader
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    refs_A, depth_A, sem_A, target_A = batch_A
    refs_B, depth_B, sem_B, target_B = batch_B  # only refs_B are used

    device = device or model.device
    depth_A = depth_A.to(device).float()
    sem_A   = sem_A.to(device).float()

    # --- preprocess target_A like in training ---
    target = target_A.to(device).float()
    if target.dtype in (torch.uint8, torch.int8, torch.int16):
        target = target / 255.0

    # encode target → latent z0
    z0 = model.vae.encode(target)              # [B,4,H8,W8]
    B, C, H8, W8 = z0.shape

    # pick a fixed timestep and noise (same for all configs)
    scheduler = model.scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(device)  # [T]
    t = torch.full((B,), t_debug, device=device, dtype=torch.long)

    alpha_bar = alphas_cumprod[t].view(B, 1, 1, 1)
    sqrt_ab   = alpha_bar.sqrt()
    sqrt_1mab = (1.0 - alpha_bar).sqrt()

    noise = torch.randn_like(z0)
    x_t   = sqrt_ab * z0 + sqrt_1mab * noise       # forward diffusion

    # ----- shared conditioning (depth+semantic) -----
    cond_in   = torch.cat([depth_A, sem_A], dim=1) # [B,4,H,W] if 1+3
    z_cond    = model.cond_encoder(cond_in)        # [B,4,H8,W8]
    latents_in_base = x_t + model.cond_scale * model.cond_norm(z_cond)

    # convenience: normalize refs to device/float
    def _prep_refs(refs_list):
        return [
            r.to(device).float() / 255.0
            if r.dtype in (torch.uint8, torch.int8, torch.int16)
            else r.to(device).float()
            for r in refs_list
        ]

    refs_A = _prep_refs(refs_A)
    refs_B = _prep_refs(refs_B)

    # ----- 1) correct refs A -----
    z_ref_A   = model.encode_references(refs_A)          # [B,4,H8,W8]
    tokens_A  = model.ref_adapter(z_ref_A, (H8, W8))

    eps_A = model.sd_unet(latents_in_base, tokens_A, t)
    x0_A  = (x_t - sqrt_1mab * eps_A) / (sqrt_ab + 1e-8)
    img_A = model.vae.decode(x0_A)[0].cpu().clamp(0, 1)

    # ----- 2) shuffled refs B (same cond, different refs) -----
    z_ref_B   = model.encode_references(refs_B)
    tokens_B  = model.ref_adapter(z_ref_B, (H8, W8))

    eps_B = model.sd_unet(latents_in_base, tokens_B, t)
    x0_B  = (x_t - sqrt_1mab * eps_B) / (sqrt_ab + 1e-8)
    img_B = model.vae.decode(x0_B)[0].cpu().clamp(0, 1)

    # ----- 3) zero refs -----
    z_ref_zero  = torch.zeros(B, 4, H8, W8, device=device)
    tokens_zero = model.ref_adapter(z_ref_zero, (H8, W8))

    eps_Z = model.sd_unet(latents_in_base, tokens_zero, t)
    x0_Z  = (x_t - sqrt_1mab * eps_Z) / (sqrt_ab + 1e-8)
    img_Z = model.vae.decode(x0_Z)[0].cpu().clamp(0, 1)

    # ----- target for reference -----
    target_im = target_A[0].cpu().float()
    if target_im.max() > 1:
        target_im = target_im / 255.0

    # ----- save -----
    T.ToPILImage()(target_im).save(os.path.join(out_dir, "A_target.png"))
    T.ToPILImage()(img_A).save(os.path.join(out_dir, "A_correct_refs.png"))
    T.ToPILImage()(img_B).save(os.path.join(out_dir, "A_shuffled_refs_B.png"))
    T.ToPILImage()(img_Z).save(os.path.join(out_dir, "A_zero_refs.png"))

    print(f"[debug] Saved denoise-step ref-debug images to {out_dir}")

from mix_diffuser_custom import FolderFiveDataset, refs_collate, DiffusionLatentFusionSynthesizer

# then run the script:
if __name__ == "__main__":
    device = "cuda"

    # rebuild dataset exactly like training
    ds = FolderFiveDataset("/lab/student/DIFFU/train_data", choose_random=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True,
                                    collate_fn=refs_collate)

    batch_A = next(iter(dl))
    batch_B = next(iter(dl))

    model = DiffusionLatentFusionSynthesizer(device=device, num_inference_steps=50)
    model.load_state_dict(torch.load("trained_weights.pth", map_location=device), strict=False)
    model.to(device)

    debug_refs_denoise_step(model, batch_A, batch_B, device, t_debug=400,
                            out_dir="debug_refs_step")




