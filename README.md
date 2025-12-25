# Diffusion_uscilab3d

This repository implements a **pose-conditioned diffusion pipeline** for rendering target RGB images given geometry, semantics, and a cross-trajectory reference image. The method is designed for **robotics datasets with accurate ground-truth poses**, such as **USCilab3D** and **NCLT**, where multiple cameras are rigidly mounted on a mobile platform.

Given a target camera pose \(T_{\text{target}}\), the system renders the corresponding RGB image using a **two-stage process**:

1. **Data preprocessing & reference retrieval** (geometry-driven, dataset-dependent)  
2. **Diffusion-based RGB synthesis** (dataset-agnostic model)

---

## Overview

For each target image to be rendered:
- The **target camera pose** is known from robot ground truth.
- A **reference image** is retrieved from a *different trajectory* based on geometric overlap.
- The model is conditioned on:
  - target depth map
  - target semantic map
  - reference RGB image
  - target camera pose

The output is a synthesized **target RGB image** aligned with the target pose.

---

## Data Preprocessing

The preprocessing stage is responsible for constructing diffusion-ready training pairs. This stage is **heavily adapted to the structure of the dataset** being used.

### Reference Camera Retrieval

Given:
- a target camera pose \(T^w_{c,\text{target}}\)
- a set of candidate camera poses from another trajectory

we retrieve the reference camera pose that **maximally overlaps** with the target pose.

#### Pose computation

Each dataset provides:
- robot ground-truth pose \(T^w_{\text{gt}}\) at each timestamp
- fixed camera extrinsics \(T^{\text{gt}}_{c_i}\) for each camera \(i\)

The camera pose in world coordinates is computed as:

\[
T^w_{c_i} = T^w_{\text{gt}} \cdot T^{\text{gt}}_{c_i}
\]

---

### KD-Tree Based Overlap Ranking

To efficiently search over large trajectories (hundreds of thousands of timestamps):

1. **All candidate camera centers** \((x, y, z)\) are indexed using a **KD-tree**.
2. For a given target camera center:
   - retrieve the top-\(K\) nearest candidates by Euclidean distance
3. Re-rank the shortlist using **orientation alignment**:

\[
\text{score} = -\alpha \lVert p - p_t \rVert + \beta \langle f, f_t \rangle
\]

where:
- \(p, p_t\) are camera centers
- \(f, f_t\) are camera forward directions
- \(\alpha, \beta\) are weighting constants

This approach is:
- **fast** (logarithmic search)
- **robust to seasonal changes**
- **independent of image appearance**

Optional time-based subsampling (e.g. every 0.1s) is used to scale to very large datasets.

---

## Robot Ground-Truth Pose and Camera Pose

In both **USCilab3D** and **NCLT**, the dataset provides:
- robot ground-truth poses as 6-DoF \((x, y, z, \text{roll}, \text{pitch}, \text{yaw})\)
- camera intrinsic calibration
- camera extrinsic calibration relative to the robot body

With \(n\) cameras mounted on the robot, the pose of each camera at time \(t\) is computed as:

\[
T^w_{c_i}(t) = T^w_{\text{gt}}(t) \cdot T^{\text{gt}}_{c_i}
\]

These camera poses are used for:
- reference image retrieval
- pose-conditioned diffusion
- cross-trajectory alignment

---

## Diffusion Model (Stage 2)

The diffusion model takes as input:
- **Reference RGB image** (appearance prior)
- **Target depth map** (geometry constraint)
- **Target semantic map** (scene structure constraint)
- **Target camera pose embedding**

and outputs:
- **Target RGB image**

The model is designed so that:
- geometry and semantics dominate spatial layout
- the reference image contributes appearance information
- the target pose controls viewpoint consistency

This separation allows the model to remain stable even when reference and target images come from different trajectories or seasons.

---

## Supported / Intended Datasets

- **USCilab3D**
- **NCLT**
- Other robotics datasets with:
  - accurate ground-truth poses
  - fixed multi-camera rigs
  - depth and/or semantic supervision

---
