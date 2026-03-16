# FLUX NVS — Diffusion Training Runbook

Depth+semantic conditioned novel-view synthesis for USCILab3D.
Backbone: FLUX.1-dev (12B params, rectified flow matching).

---

## Architecture Overview

```
Input (conditioning):
  ├─ target depth map (static LiDAR, 1280×720)
  ├─ target semantic map (static SAM3, 1280×720)
  └─ 4 reference RGB images (KD-tree cross-trajectory nearest)

Output:
  └─ target RGB image (1280×720)

Dynamic handling:
  └─ SAM3 masks → zero out dynamic regions in conditioning
  └─ masked loss (no gradients on dynamic pixels)
```

---

## Prerequisites

These come from the 3d2d_ann pipeline (separate RUNBOOK):
- Step 0 (COMPLETE): velodyne-camera mappings
- Step 4: global pose graph (for cross-trajectory KD-tree)
- Step 6: 3D→2D projections (for depth maps)
- Step 7: SAM3 semantic labels (for semantic conditioning + dynamic masks)

**None of the above need to be done before writing/debugging the FLUX code.**
They are needed before real training data is available.

---

## Hardware

| Phase | Machine | GPUs | Notes |
|-------|---------|------|-------|
| Debug/dev | Work server | A100 / H100 | Small batches, verify loss goes down |
| Full train | Rented cloud | 4× H100 80GB | ~$500-750 for 50K steps |
| Fallback | igpu15 | 4× V100 32GB | Slower (~7 days for 50K steps) |

---

## STEP 0: Environment Setup

```bash
# On whichever machine you're using
cd diffusion_uscilab3d

# Create conda env (Python 3.11)
conda create -n flux_nvs python=3.11 -y
conda activate flux_nvs

# Install deps
pip install -r requirements_flux.txt

# Verify FLUX loads
python -c "
from diffusers import FluxTransformer2DModel
import torch
model = FluxTransformer2DModel.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    subfolder='transformer',
    torch_dtype=torch.bfloat16
)
print('FLUX loaded, params:', sum(p.numel() for p in model.parameters()) / 1e9, 'B')
del model
torch.cuda.empty_cache()
print('OK')
"
```

**Hugging Face auth:** FLUX.1-dev is gated. You need:
```bash
huggingface-cli login
# paste your HF token (must have accepted FLUX.1-dev license)
```

---

## STEP 1: Prepare Training Data

This step depends on 3d2d_ann pipeline Steps 4-7 being complete.
Until then, you can debug with synthetic/dummy data.

### 1a. Generate dummy data for debugging

```bash
python -c "
import torch, os, json
os.makedirs('/tmp/flux_debug_data', exist_ok=True)

# 100 fake samples
for i in range(100):
    d = {
        'target_rgb': torch.randint(0, 255, (3, 720, 1280), dtype=torch.uint8),
        'target_depth': torch.rand(1, 720, 1280),
        'target_semantic': torch.randint(0, 20, (1, 720, 1280), dtype=torch.uint8),
        'ref_rgbs': torch.randint(0, 255, (4, 3, 720, 1280), dtype=torch.uint8),
        'dynamic_mask': torch.zeros(1, 720, 1280, dtype=torch.bool),
    }
    torch.save(d, '/tmp/flux_debug_data/sample_{:04d}.pt'.format(i))

print('Created 100 dummy samples')
"
```

### 1b. Real data preparation (after 3d2d_ann pipeline completes)

```bash
python prepare_training_data.py \
    --data-root /data/USCILab3D \
    --global-poses /tmp/pose_graph/global_poses.parquet \
    --projections-dir /tmp/projections \
    --semantic-dir /tmp/semantic_labels \
    --output-dir /data/flux_nvs_training \
    --skip-broken \
    --num-refs 4 \
    --ref-strategy kdtree-cross-trajectory \
    2>&1 | tee /tmp/prepare_flux_data.log
```

**Note:** `prepare_training_data.py` does not exist yet. Create it when
the upstream data is ready. It should:
1. Load global poses → build KD-tree of all keyframe positions
2. For each target frame, find 4 nearest reference frames from OTHER sessions
3. Load depth from 3D→2D projections
4. Load semantic map from SAM3 output
5. Build dynamic mask (SAM3 classes: person, car, bicycle, etc.)
6. Save as `.pt` files or sharded tar for webdataset

---

## STEP 2: Debug Training (Work Server)

Quick sanity check — verify loss decreases, no OOM, gradients flow.

```bash
cd diffusion_uscilab3d

# Single-GPU debug run (small batch, few steps)
python train_flux_nvs.py \
    --config configs/flux_nvs.yaml \
    --data-dir /tmp/flux_debug_data \
    --output-dir /tmp/flux_debug_run \
    --max-steps 200 \
    --batch-size 1 \
    --gradient-accumulation 1 \
    --log-every 10 \
    --save-every 100 \
    --no-wandb \
    2>&1 | tee /tmp/flux_debug.log
```

**What to check:**
- [ ] Loss decreases from ~1.0 to <0.5 within 100 steps on dummy data
- [ ] No OOM on A100 80GB with batch_size=1
- [ ] Gradient norms are reasonable (0.1–10.0 range)
- [ ] Checkpoint saves correctly
- [ ] bf16 mixed precision works (check for NaN loss)

### Multi-GPU debug (2 GPUs)

```bash
torchrun --nproc_per_node=2 train_flux_nvs.py \
    --config configs/flux_nvs.yaml \
    --data-dir /tmp/flux_debug_data \
    --output-dir /tmp/flux_debug_2gpu \
    --max-steps 200 \
    --batch-size 1 \
    --gradient-accumulation 2 \
    --no-wandb \
    2>&1 | tee /tmp/flux_debug_2gpu.log
```

**What to check (multi-GPU):**
- [ ] DDP syncs correctly (loss values match across ranks initially)
- [ ] Effective batch size = batch_size × num_gpus × gradient_accumulation
- [ ] Checkpointing works with DDP (saves/loads without errors)

---

## STEP 3: Full Training (Rented 4× H100)

### 3a. Server setup

```bash
# After SSH into rented server
git clone https://github.com/YOUR_ORG/diffusion_uscilab3d.git
cd diffusion_uscilab3d

conda create -n flux_nvs python=3.11 -y
conda activate flux_nvs
pip install -r requirements_flux.txt

huggingface-cli login

# Copy training data (rsync from iLab or download)
rsync -avzP igpu15:/data/flux_nvs_training/ /data/flux_nvs_training/
```

### 3b. Launch training

```bash
# 4× H100, effective batch 32
torchrun --nproc_per_node=4 train_flux_nvs.py \
    --config configs/flux_nvs.yaml \
    --data-dir /data/flux_nvs_training \
    --output-dir /data/flux_nvs_checkpoints \
    --max-steps 50000 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --lr 1e-5 \
    --warmup-steps 1000 \
    --log-every 50 \
    --save-every 5000 \
    --wandb-project flux-nvs-uscilab3d \
    2>&1 | tee /data/flux_train.log
```

**Training config summary:**
| Param | Value |
|-------|-------|
| Effective batch size | 2 × 4 GPUs × 4 accum = 32 |
| Learning rate | 1e-5 (cosine decay) |
| Warmup | 1,000 steps |
| Total steps | 50,000 |
| Precision | bf16 |
| Optimizer | 8-bit AdamW |
| Gradient checkpointing | ON |
| EMA | 0.9999 |
| Loss | rectified flow matching + 0.05 × L1 x0-reconstruction |

**Estimated time:** ~1-2 days on 4× H100 80GB
**Estimated cost:** ~$500-750

### 3c. Monitor training

```bash
# Watch loss in real time
tail -f /data/flux_train.log | grep "step.*loss"

# Or use wandb dashboard
# https://wandb.ai/YOUR_ORG/flux-nvs-uscilab3d
```

**Healthy training signs:**
- Loss drops from ~1.0 → ~0.3 in first 5K steps
- Loss plateaus around 0.1-0.2 by 20K steps
- Gradient norms stay < 5.0 (clipped at 1.0)
- No NaN/Inf in loss

### 3d. Save best checkpoint back to iLab

```bash
# After training completes
rsync -avzP /data/flux_nvs_checkpoints/checkpoint-best/ \
    igpu15:/data/flux_nvs_checkpoints/best/
```

---

## STEP 4: Evaluation & Inference

```bash
# Generate novel views for a test session
python inference_flux_nvs.py \
    --checkpoint /data/flux_nvs_checkpoints/best \
    --data-dir /data/flux_nvs_training/test \
    --output-dir /tmp/flux_nvs_outputs \
    --num-samples 50 \
    --guidance-scale 3.5 \
    --num-inference-steps 28 \
    2>&1 | tee /tmp/flux_inference.log
```

**Metrics to compute:**
- LPIPS (perceptual similarity to ground truth)
- SSIM / PSNR (pixel-level)
- FID (distribution-level, need 1K+ samples)
- Visual inspection (most important — does it look real at 1280×720?)

**Note:** `inference_flux_nvs.py` does not exist yet. Create after
training works. It should:
1. Load checkpoint
2. For each test sample, run FLUX denoising with conditioning
3. Save output images
4. Compute metrics vs ground truth

---

## Files in This Repo

| File | Purpose |
|------|---------|
| `flux_nvs.py` | FLUX backbone wrapper (ConditionEncoder, ReferenceAdapter, FluxNVS) |
| `train_flux_nvs.py` | DDP training script (bf16, grad checkpointing, 8-bit AdamW, wandb) |
| `configs/flux_nvs.yaml` | Training config (hyperparams, paths) |
| `requirements_flux.txt` | Python dependencies |
| `uscilab3d_dataset.py` | Dataset adapter for USCILab3D |
| `RUNBOOK.md` | This file |

---

## Troubleshooting

**OOM on A100 80GB:**
- Reduce batch_size to 1
- Enable gradient checkpointing (should be ON by default)
- Reduce image resolution for debugging (512×512)

**NaN loss:**
- Check bf16 overflow — try fp32 for first 100 steps
- Reduce learning rate to 5e-6
- Check input normalization (images should be [0, 1] or [-1, 1])

**FLUX download fails:**
- Accept license at https://huggingface.co/black-forest-labs/FLUX.1-dev
- Check HF_TOKEN is set correctly
- Use `--local-files-only` if you pre-downloaded weights

**DDP hangs:**
- Check NCCL env vars: `NCCL_P2P_DISABLE=1` if peer-to-peer fails
- Try `NCCL_SOCKET_IFNAME=eth0` (or your network interface)
- Ensure all GPUs visible: `nvidia-smi` should show all 4

---

## Timeline

```
NOW ─────────── Debug on work server (Steps 0-2)
                     │
3d2d_ann pipeline ── Steps 4,6,7 complete
completes                │
                    Prepare real training data (Step 1b)
                         │
                    Full training on rented 4×H100 (Step 3)
                         │
                    Eval & iterate (Step 4)
```
