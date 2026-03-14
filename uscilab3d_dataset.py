#!/usr/bin/env python3
"""
uscilab3d_dataset.py
====================
PyTorch Dataset for the USCILab3D diffusion model.

Reads from pipeline outputs:
  - RGB images:    extracted from bag files (cam1-5)
  - Depth maps:    projected LiDAR -> camera (from project_3d_to_2d.py Parquet)
  - Semantic maps:  SAM3 segmentation masks (per-frame PNG)
  - Poses:          global_poses.parquet (from pose graph pipeline)
  - Velodyne map:   velodyne_cam_mapping.parquet (velodyne scan <-> camera image index)

Reference images are retrieved from OTHER sessions via KD-tree pose search
(same physical area, different trajectory).

Robust to missing data:
  - Cameras that were disconnected during a run
  - Missing velodyne scans (sensor not turned on)
  - Missing semantic masks (SAM3 not yet run on all frames)

Compatible with Python 3.9+ (no 3.12-only syntax).
"""

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("pyarrow required: pip install pyarrow")

try:
    from scipy.spatial import cKDTree
except ImportError:
    raise ImportError("scipy required: pip install scipy")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_WIDTH = 1280
IMG_HEIGHT = 720
CAM_IDS = [1, 2, 3, 4, 5]

# Default intrinsics from pipeline_config.yaml (MATLAB calibration, 1280x720)
DEFAULT_INTRINSICS = {
    1: {"fx": 633.3955, "fy": 633.1321, "cx": 638.5496, "cy": 373.7600,
        "dist": [-0.0560, 0.0253, 0.0, 0.0, 0.0]},
    2: {"fx": 630.8010, "fy": 630.6018, "cx": 626.5921, "cy": 372.1486,
        "dist": [-0.0610, 0.0708, 0.0, 0.0, 0.0]},
    3: {"fx": 628.5559, "fy": 627.9042, "cx": 644.4944, "cy": 371.4139,
        "dist": [-0.0729, 0.1014, 0.0, 0.0, 0.0]},
    4: {"fx": 630.0727, "fy": 629.5730, "cx": 642.9044, "cy": 360.8640,
        "dist": [-0.0635, 0.0718, 0.0, 0.0, 0.0]},
    5: {"fx": 638.3315, "fy": 638.1820, "cx": 639.1726, "cy": 366.2761,
        "dist": [-0.0607, 0.0808, 0.0, 0.0, 0.0]},
}

# Default extrinsics: T_cam_velo (velodyne -> camera), from pipeline_config.yaml
DEFAULT_EXTRINSICS = {
    1: np.array([
        [ 0.997787, -0.0603672, -0.0278649, -0.0225125],
        [-0.0302716, -0.0393293, -0.998767,  0.162101],
        [ 0.0591969,  0.997402,  -0.0410699, -0.178768],
        [ 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64),
    2: np.array([
        [ 0.374325,  0.926928, -0.0261801, -0.0220061],
        [ 0.032223, -0.0412179, -0.99863,   0.162078],
        [-0.926738,  0.372969, -0.0452975, -0.178564],
        [ 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64),
    3: np.array([
        [-0.846798, -0.531271, -0.0261801,  0.153234],
        [-0.00192764, 0.0522831, -0.99863,  0.168518],
        [ 0.531912, -0.845588, -0.0452975, -0.421818],
        [ 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64),
    4: np.array([
        [-0.780203,  0.624979, -0.0261801, -0.100335],
        [ 0.048741,  0.0190146, -0.99863,   0.167401],
        [-0.623625, -0.780411, -0.0452975, -0.25064],
        [ 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64),
    5: np.array([
        [ 0.25608,  -0.966302, -0.0261801, -0.0337689],
        [-0.0505004, 0.0136728, -0.99863,   0.160828],
        [ 0.965336,  0.257051, -0.0452975, -0.144208],
        [ 0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64),
}


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------

def quat_to_R(qw, qx, qy, qz):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) + 1e-12
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz),  2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),      1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def make_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def cam_forward_z(R_w_c):
    """Camera +Z forward direction in world frame."""
    f = R_w_c[:, 2]
    return f / (np.linalg.norm(f) + 1e-12)


# ---------------------------------------------------------------------------
# Depth map rendering from LiDAR projection Parquet
# ---------------------------------------------------------------------------

def render_depth_map(projection_parquet, cam_id, width=IMG_WIDTH, height=IMG_HEIGHT,
                     max_depth=80.0):
    """
    Read a per-scan projection Parquet file and render a depth image.

    Expected Parquet columns: u, v, depth  (from project_3d_to_2d.py)
    Returns: (1, H, W) float32 tensor, depth in meters, 0 where no data.
    """
    try:
        table = pq.read_table(str(projection_parquet), columns=["u", "v", "depth"])
    except Exception as e:
        logger.warning("Failed to read projection %s: %s", projection_parquet, e)
        return None

    u = table.column("u").to_numpy().astype(np.int32)
    v = table.column("v").to_numpy().astype(np.int32)
    depth = table.column("depth").to_numpy().astype(np.float32)

    # Filter valid
    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height) & (depth > 0) & (depth < max_depth)
    u, v, depth = u[mask], v[mask], depth[mask]

    depth_map = np.zeros((height, width), dtype=np.float32)
    # Z-buffer: keep closest point per pixel
    depth_map[:] = max_depth + 1.0
    np.minimum.at(depth_map, (v, u), depth)
    depth_map[depth_map > max_depth] = 0.0

    return torch.from_numpy(depth_map).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def load_rgb_from_bag(bag_path, img_idx, cam_id):
    """
    Load a single image from a ROS bag by index.
    Uses rosbags (pure Python) to avoid ROS dependency.

    Returns: (1, 3, H, W) float32 tensor in [0, 1], or None on failure.
    """
    try:
        from rosbags.rosbag1 import Reader as Rosbag1Reader
        from rosbags.serde import deserialize_cdr, ros1_to_cdr
    except ImportError:
        raise ImportError("rosbags required: pip install rosbags")

    topic = "/cam{}/image_raw/compressed".format(cam_id)
    bag_path = Path(bag_path)
    if not bag_path.exists():
        return None

    try:
        with Rosbag1Reader(bag_path) as reader:
            conns = [c for c in reader.connections if c.topic == topic]
            if not conns:
                return None
            count = 0
            for conn, timestamp, rawdata in reader.messages(connections=conns):
                if count == img_idx:
                    msg = deserialize_cdr(ros1_to_cdr(rawdata, conn.msgtype), conn.msgtype)
                    img_bytes = bytes(msg.data)
                    # Decode JPEG
                    import cv2
                    arr = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is None:
                        return None
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    return tensor.unsqueeze(0)  # (1, 3, H, W)
                count += 1
    except Exception as e:
        logger.warning("Failed to read bag %s idx %d: %s", bag_path, img_idx, e)
    return None


def load_rgb_from_extracted(extracted_dir, cam_id, img_idx):
    """
    Load RGB from pre-extracted JPEG/PNG directory.
    Looks for: {extracted_dir}/cam{cam_id}/{img_idx:06d}.jpg (or .png)

    Returns: (1, 3, H, W) float32 tensor in [0, 1], or None.
    """
    import torchvision.transforms as T
    from PIL import Image

    for ext in [".jpg", ".jpeg", ".png"]:
        path = Path(extracted_dir) / "cam{}".format(cam_id) / "{:06d}{}".format(img_idx, ext)
        if path.exists():
            img = Image.open(path).convert("RGB")
            tensor = T.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)
            return tensor
    return None


def load_semantic_mask(semantic_dir, cam_id, img_idx):
    """
    Load SAM3 semantic mask.
    Expected: {semantic_dir}/cam{cam_id}/{img_idx:06d}_semantic.png  (RGB or index)

    Returns: (1, 3, H, W) float32 tensor in [0, 1], or None.
    """
    import torchvision.transforms as T
    from PIL import Image

    for pattern in [
        "{:06d}_semantic.png",
        "{:06d}_sem.png",
        "{:06d}.png",
    ]:
        path = Path(semantic_dir) / "cam{}".format(cam_id) / pattern.format(img_idx)
        if path.exists():
            img = Image.open(path).convert("RGB")
            tensor = T.ToTensor()(img).unsqueeze(0)  # (1, 3, H, W)
            return tensor
    return None


# ---------------------------------------------------------------------------
# Session index: one session's worth of aligned (rgb, depth, semantic) frames
# ---------------------------------------------------------------------------

class SessionIndex:
    """
    Index for a single session (e.g., 2023_03_27/0).
    Holds:
      - velodyne_cam_mapping: which velodyne scan maps to which camera image
      - global poses: T_world_body for each keyframe
      - paths to projection parquets, semantic masks, bag files
    """

    def __init__(
        self,
        session_id: str,             # e.g. "2023_03_27/0"
        session_path: Path,          # /data/USCILab3D/2023_03_27/0
        mapping_parquet: Path,       # velodyne_cam_mapping.parquet
        projection_dir: Optional[Path],  # /tmp/projections/2023_03_27/0
        semantic_dir: Optional[Path],    # /tmp/semantic_labels/2023_03_27/0
        extracted_dir: Optional[Path],   # pre-extracted images (if available)
        poses: Optional[dict] = None,    # {keyframe_idx: (x,y,z,qw,qx,qy,qz)}
    ):
        self.session_id = session_id
        self.session_path = Path(session_path)
        self.projection_dir = Path(projection_dir) if projection_dir else None
        self.semantic_dir = Path(semantic_dir) if semantic_dir else None
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.poses = poses or {}

        # Read velodyne-camera mapping
        self.mapping = self._load_mapping(mapping_parquet)

    def _load_mapping(self, parquet_path):
        """
        Load velodyne_cam_mapping.parquet.
        Columns: velodyne_idx, velodyne_ts,
                 cam{1-5}_bag, cam{1-5}_img_idx, cam{1-5}_ts, cam{1-5}_dt_ms
        Returns list of dicts, one per velodyne scan.
        """
        try:
            table = pq.read_table(str(parquet_path))
            df_dict = table.to_pydict()
        except Exception as e:
            logger.warning("Cannot load mapping %s: %s", parquet_path, e)
            return []

        n = len(df_dict.get("velodyne_idx", []))
        rows = []
        for i in range(n):
            row = {"velodyne_idx": df_dict["velodyne_idx"][i]}
            if "velodyne_ts" in df_dict:
                row["velodyne_ts"] = df_dict["velodyne_ts"][i]
            for cam_id in CAM_IDS:
                bag_col = "cam{}_bag".format(cam_id)
                idx_col = "cam{}_img_idx".format(cam_id)
                ts_col = "cam{}_ts".format(cam_id)
                dt_col = "cam{}_dt_ms".format(cam_id)
                if bag_col in df_dict:
                    row["cam{}_bag".format(cam_id)] = df_dict[bag_col][i]
                if idx_col in df_dict:
                    row["cam{}_img_idx".format(cam_id)] = df_dict[idx_col][i]
                if dt_col in df_dict:
                    row["cam{}_dt_ms".format(cam_id)] = df_dict[dt_col][i]
            rows.append(row)
        return rows

    def get_valid_frames(self, cam_id, max_dt_ms=100.0):
        """
        Return list of (velodyne_idx, bag_path, img_idx) for frames where:
          - camera mapping exists
          - temporal offset < max_dt_ms
          - bag file exists
        """
        frames = []
        for row in self.mapping:
            bag_col = "cam{}_bag".format(cam_id)
            idx_col = "cam{}_img_idx".format(cam_id)
            dt_col = "cam{}_dt_ms".format(cam_id)

            if bag_col not in row or idx_col not in row:
                continue
            bag_name = row[bag_col]
            img_idx = row[idx_col]
            if bag_name is None or img_idx is None:
                continue

            # Check temporal offset
            dt = row.get(dt_col, 0.0)
            if dt is not None and abs(dt) > max_dt_ms:
                continue

            bag_path = self.session_path / "cam{}".format(cam_id) / bag_name
            frames.append((row["velodyne_idx"], bag_path, img_idx))
        return frames


# ---------------------------------------------------------------------------
# Cross-trajectory reference retrieval (KD-tree)
# ---------------------------------------------------------------------------

class CrossTrajectoryIndex:
    """
    KD-tree index over camera poses from ALL sessions.
    Used to find the best reference image from a DIFFERENT session
    that overlaps with a given target camera pose.
    """

    def __init__(self, extrinsics=None):
        """
        extrinsics: dict cam_id -> 4x4 T_cam_velo (velodyne -> camera).
                    Default: USCILab3D calibration from pipeline_config.yaml.
        """
        self.T_cam_velo = extrinsics or DEFAULT_EXTRINSICS
        # We store T_velo_cam (inverse) for computing camera pose in world frame
        # T_world_cam = T_world_body * T_body_velo * T_velo_cam
        # Since T_cam_velo = T_cam_body (our extrinsics are velodyne->camera),
        # T_body_cam = inv(T_cam_velo) assuming velodyne is at body origin
        self.T_body_cam = {}
        for cam_id, T_cv in self.T_cam_velo.items():
            self.T_body_cam[cam_id] = np.linalg.inv(T_cv)

        # Will be populated by add_session()
        self.centers = []      # list of (3,) arrays
        self.forwards = []     # list of (3,) arrays
        self.meta = []         # list of (session_id, velodyne_idx, cam_id)
        self.tree = None
        self._built = False

    def add_session(self, session_id, poses_dict, cam_ids=None):
        """
        Add all camera poses from a session to the index.

        poses_dict: {keyframe_idx: (x, y, z, qw, qx, qy, qz)}
        cam_ids: which cameras to index (default: all 5)
        """
        if cam_ids is None:
            cam_ids = list(self.T_body_cam.keys())

        for kf_idx, pose in poses_dict.items():
            x, y, z, qw, qx, qy, qz = pose
            R_w_b = quat_to_R(qw, qx, qy, qz)
            T_w_b = make_T(R_w_b, np.array([x, y, z]))

            for cam_id in cam_ids:
                if cam_id not in self.T_body_cam:
                    continue
                T_w_c = T_w_b @ self.T_body_cam[cam_id]
                center = T_w_c[:3, 3].astype(np.float32)
                fwd = cam_forward_z(T_w_c[:3, :3]).astype(np.float32)

                self.centers.append(center)
                self.forwards.append(fwd)
                self.meta.append((session_id, kf_idx, cam_id))

        self._built = False  # need to rebuild tree

    def build(self):
        """Build/rebuild the KD-tree."""
        if len(self.centers) == 0:
            raise ValueError("No poses added to the index.")
        centers_arr = np.array(self.centers, dtype=np.float64)
        self.tree = cKDTree(centers_arr)
        self._built = True
        logger.info("CrossTrajectoryIndex: built KD-tree with %d entries", len(self.centers))

    def query(self, target_center, target_forward, exclude_session,
              top_k=200, alpha=1.0, beta=8.0, min_align_cos=None):
        """
        Find best reference from a DIFFERENT session.

        Returns: (session_id, keyframe_idx, cam_id, score) or None.
        """
        if not self._built:
            self.build()

        k = min(top_k, len(self.centers))
        dists, idxs = self.tree.query(target_center.astype(np.float64), k=k)
        if np.isscalar(idxs):
            idxs = np.array([int(idxs)])
            dists = np.array([float(dists)])

        best = None
        for dist, idx in zip(dists, idxs):
            idx = int(idx)
            sess_id, kf_idx, cam_id = self.meta[idx]

            # Skip same session
            if sess_id == exclude_session:
                continue

            fwd = self.forwards[idx].astype(np.float32)
            align = float(np.dot(fwd, target_forward.astype(np.float32)))

            if min_align_cos is not None and align < min_align_cos:
                continue

            score = -alpha * float(dist) + beta * align
            if best is None or score > best[3]:
                best = (sess_id, kf_idx, cam_id, score)

        return best


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class USCILab3DDataset(Dataset):
    """
    Diffusion training dataset for USCILab3D.

    Each sample yields:
      - refs:     list of N_ref (1, 3, H, W) reference RGB tensors from other trajectories
      - depth:    (1, 1, H, W) depth map from LiDAR projection
      - semantic: (1, 3, H, W) semantic mask from SAM3
      - target:   (1, 3, H, W) target RGB image

    Handles missing data gracefully: skips frames with missing depth,
    missing semantic maps, or failed bag reads.
    """

    def __init__(
        self,
        data_root: str,
        mapping_dir: str,                          # /tmp/velodyne_cam_mappings
        projection_dir: Optional[str] = None,      # /tmp/projections
        semantic_dir: Optional[str] = None,        # /tmp/semantic_labels
        extracted_dir: Optional[str] = None,       # pre-extracted images root
        global_poses_path: Optional[str] = None,   # /tmp/pose_graph/global_poses.parquet
        cam_ids: Optional[List[int]] = None,
        n_refs: int = 4,
        max_depth: float = 80.0,
        max_dt_ms: float = 100.0,
        samples_per_session: int = 200,
        img_height: int = IMG_HEIGHT,
        img_width: int = IMG_WIDTH,
        use_bag_reader: bool = True,               # if False, only use extracted images
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mapping_dir = Path(mapping_dir)
        self.projection_dir = Path(projection_dir) if projection_dir else None
        self.semantic_dir = Path(semantic_dir) if semantic_dir else None
        self.extracted_dir = Path(extracted_dir) if extracted_dir else None
        self.cam_ids = cam_ids or CAM_IDS
        self.n_refs = n_refs
        self.max_depth = max_depth
        self.max_dt_ms = max_dt_ms
        self.samples_per_session = samples_per_session
        self.img_height = img_height
        self.img_width = img_width
        self.use_bag_reader = use_bag_reader

        # Load global poses if available
        self.global_poses = self._load_global_poses(global_poses_path)

        # Discover sessions
        self.sessions = {}    # session_id -> SessionIndex
        self.ref_index = CrossTrajectoryIndex()
        self._discover_sessions()

        # Build flat sample list: (session_id, cam_id, velodyne_idx, bag_path, img_idx)
        self.samples = self._build_sample_list()
        logger.info("USCILab3DDataset: %d samples across %d sessions",
                     len(self.samples), len(self.sessions))

    def _load_global_poses(self, path):
        """
        Load global_poses.parquet -> {session_id: {keyframe_idx: (x,y,z,qw,qx,qy,qz)}}
        """
        if path is None or not Path(path).exists():
            return {}
        try:
            table = pq.read_table(str(path))
            d = table.to_pydict()
        except Exception as e:
            logger.warning("Cannot load global poses: %s", e)
            return {}

        poses = {}
        n = len(d.get("session", []))
        for i in range(n):
            sess = d["session"][i]
            kf = d["keyframe_idx"][i]
            pose_tuple = (
                d["x"][i], d["y"][i], d["z"][i],
                d["qw"][i], d["qx"][i], d["qy"][i], d["qz"][i],
            )
            if sess not in poses:
                poses[sess] = {}
            poses[sess][kf] = pose_tuple
        return poses

    def _discover_sessions(self):
        """Find all sessions that have velodyne_cam_mapping.parquet."""
        if not self.mapping_dir.exists():
            logger.warning("Mapping dir does not exist: %s", self.mapping_dir)
            return

        for date_dir in sorted(self.mapping_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            for seq_dir in sorted(date_dir.iterdir()):
                if not seq_dir.is_dir():
                    continue
                mapping_pq = seq_dir / "velodyne_cam_mapping.parquet"
                if not mapping_pq.exists():
                    continue

                session_id = "{}/{}".format(date_dir.name, seq_dir.name)
                session_path = self.data_root / date_dir.name / seq_dir.name

                proj_dir = None
                if self.projection_dir:
                    p = self.projection_dir / date_dir.name / seq_dir.name
                    if p.exists():
                        proj_dir = p

                sem_dir = None
                if self.semantic_dir:
                    s = self.semantic_dir / date_dir.name / seq_dir.name
                    if s.exists():
                        sem_dir = s

                ext_dir = None
                if self.extracted_dir:
                    e = Path(self.extracted_dir) / date_dir.name / seq_dir.name
                    if e.exists():
                        ext_dir = e

                session_poses = self.global_poses.get(session_id, {})

                si = SessionIndex(
                    session_id=session_id,
                    session_path=session_path,
                    mapping_parquet=mapping_pq,
                    projection_dir=proj_dir,
                    semantic_dir=sem_dir,
                    extracted_dir=ext_dir,
                    poses=session_poses,
                )
                self.sessions[session_id] = si

                # Add to cross-trajectory index if we have poses
                if session_poses:
                    self.ref_index.add_session(session_id, session_poses, self.cam_ids)

        # Build KD-tree if we have any poses
        if len(self.ref_index.centers) > 0:
            self.ref_index.build()

    def _build_sample_list(self):
        """Build list of valid (session_id, cam_id, vel_idx, bag_path, img_idx) tuples."""
        samples = []
        for session_id, si in self.sessions.items():
            for cam_id in self.cam_ids:
                frames = si.get_valid_frames(cam_id, max_dt_ms=self.max_dt_ms)
                if not frames:
                    continue
                # Subsample if too many frames
                if len(frames) > self.samples_per_session:
                    step = max(1, len(frames) // self.samples_per_session)
                    frames = frames[::step][:self.samples_per_session]
                for vel_idx, bag_path, img_idx in frames:
                    samples.append((session_id, cam_id, vel_idx, str(bag_path), img_idx))
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_rgb(self, session_id, cam_id, bag_path, img_idx):
        """Load RGB image, trying extracted dir first, then bag."""
        si = self.sessions[session_id]
        if si.extracted_dir is not None:
            tensor = load_rgb_from_extracted(si.extracted_dir, cam_id, img_idx)
            if tensor is not None:
                return tensor
        if self.use_bag_reader:
            return load_rgb_from_bag(bag_path, img_idx, cam_id)
        return None

    def _load_depth(self, session_id, cam_id, vel_idx):
        """Load depth map from projection Parquet."""
        si = self.sessions[session_id]
        if si.projection_dir is None:
            return None
        # Expected: projection_dir/cam{cam_id}/scan_{vel_idx:06d}.parquet
        for pattern in [
            "cam{}/scan_{:06d}.parquet",
            "cam{}/{:06d}.parquet",
            "scan_{:06d}_cam{}.parquet",
        ]:
            path = si.projection_dir / pattern.format(cam_id, vel_idx)
            if path.exists():
                return render_depth_map(path, cam_id, max_depth=self.max_depth)
        return None

    def _load_semantic(self, session_id, cam_id, img_idx):
        """Load SAM3 semantic mask."""
        si = self.sessions[session_id]
        if si.semantic_dir is None:
            return None
        return load_semantic_mask(si.semantic_dir, cam_id, img_idx)

    def _get_camera_pose(self, session_id, vel_idx, cam_id):
        """
        Compute camera pose in world frame for this frame.
        Returns (center_3, forward_3) or (None, None).
        """
        si = self.sessions[session_id]
        # Find nearest keyframe pose
        if not si.poses:
            return None, None

        # vel_idx ~= keyframe_idx for now (they're close enough for KD-tree retrieval)
        # In a refined version, we'd interpolate between keyframes
        nearest_kf = min(si.poses.keys(), key=lambda k: abs(k - vel_idx))
        if abs(nearest_kf - vel_idx) > 50:  # too far, skip
            return None, None

        x, y, z, qw, qx, qy, qz = si.poses[nearest_kf]
        R_w_b = quat_to_R(qw, qx, qy, qz)
        T_w_b = make_T(R_w_b, np.array([x, y, z]))
        T_body_cam = np.linalg.inv(DEFAULT_EXTRINSICS.get(cam_id, np.eye(4)))
        T_w_c = T_w_b @ T_body_cam
        center = T_w_c[:3, 3].astype(np.float32)
        fwd = cam_forward_z(T_w_c[:3, :3]).astype(np.float32)
        return center, fwd

    def _find_reference_frames(self, session_id, vel_idx, cam_id, n_refs):
        """
        Find n_refs reference frames from OTHER sessions using KD-tree.
        Returns list of (ref_session_id, ref_kf_idx, ref_cam_id).
        Falls back to random frames from other sessions if KD-tree fails.
        """
        center, fwd = self._get_camera_pose(session_id, vel_idx, cam_id)

        refs = []
        if center is not None and fwd is not None and self.ref_index._built:
            # Query KD-tree for top matches
            # We need n_refs, so query with a bigger top_k and deduplicate
            k = min(500, len(self.ref_index.centers))
            if k > 0:
                dists, idxs = self.ref_index.tree.query(center.astype(np.float64), k=k)
                if np.isscalar(idxs):
                    idxs = [int(idxs)]
                    dists = [float(dists)]

                seen_sessions = set()
                for dist, idx in zip(dists, idxs):
                    idx = int(idx)
                    sess_id, kf_idx, c_id = self.ref_index.meta[idx]
                    if sess_id == session_id:
                        continue
                    fwd_ref = self.ref_index.forwards[idx].astype(np.float32)
                    align = float(np.dot(fwd_ref, fwd))
                    if align < 0.3:  # facing too different a direction
                        continue
                    refs.append((sess_id, kf_idx, c_id))
                    if len(refs) >= n_refs:
                        break

        # Fallback: random frames from other sessions
        if len(refs) < n_refs:
            other_sessions = [s for s in self.sessions.keys() if s != session_id]
            while len(refs) < n_refs and other_sessions:
                other_sid = random.choice(other_sessions)
                other_si = self.sessions[other_sid]
                frames = other_si.get_valid_frames(cam_id, max_dt_ms=self.max_dt_ms)
                if frames:
                    vel_i, bag_p, img_i = random.choice(frames)
                    refs.append((other_sid, vel_i, cam_id))
                else:
                    other_sessions.remove(other_sid)

        return refs[:n_refs]

    def __getitem__(self, idx):
        """
        Returns: (refs, depth, semantic, target)
          refs: list of n_refs tensors, each (1, 3, H, W)
          depth: (1, 1, H, W)
          semantic: (1, 3, H, W)
          target: (1, 3, H, W)

        If any component is missing, retries with a random other sample.
        """
        max_retries = 10
        for attempt in range(max_retries):
            try:
                sample_idx = (idx + attempt) % len(self.samples)
                result = self._load_sample(sample_idx)
                if result is not None:
                    return result
            except Exception as e:
                logger.debug("Sample %d attempt %d failed: %s", idx, attempt, e)
                continue

        # Ultimate fallback: return zeros (should be very rare)
        logger.warning("All retries exhausted for idx %d, returning zeros", idx)
        H, W = self.img_height, self.img_width
        dummy_refs = [torch.zeros(1, 3, H, W) for _ in range(self.n_refs)]
        dummy_depth = torch.zeros(1, 1, H, W)
        dummy_sem = torch.zeros(1, 3, H, W)
        dummy_target = torch.zeros(1, 3, H, W)
        return dummy_refs, dummy_depth, dummy_sem, dummy_target

    def _load_sample(self, sample_idx):
        """Try to load a single sample. Returns tuple or None."""
        session_id, cam_id, vel_idx, bag_path, img_idx = self.samples[sample_idx]

        # 1) Target RGB
        target = self._load_rgb(session_id, cam_id, bag_path, img_idx)
        if target is None:
            return None

        # 2) Depth map (required)
        depth = self._load_depth(session_id, cam_id, vel_idx)
        if depth is None:
            # If no projection data yet, synthesize from zeros (training can proceed
            # with depth=0 but results will be poor — this is a placeholder)
            depth = torch.zeros(1, 1, self.img_height, self.img_width)

        # Normalize depth to [0, 1]
        depth = depth / self.max_depth

        # 3) Semantic mask
        semantic = self._load_semantic(session_id, cam_id, img_idx)
        if semantic is None:
            # Placeholder: zeros until SAM3 is run
            semantic = torch.zeros(1, 3, self.img_height, self.img_width)

        # 4) Reference images from other trajectories
        ref_specs = self._find_reference_frames(session_id, vel_idx, cam_id, self.n_refs)
        refs = []
        for ref_sid, ref_vel_idx, ref_cam_id in ref_specs:
            ref_si = self.sessions.get(ref_sid)
            if ref_si is None:
                continue
            # Find the corresponding image
            ref_frames = ref_si.get_valid_frames(ref_cam_id, max_dt_ms=self.max_dt_ms)
            # Find frame closest to ref_vel_idx
            best_frame = None
            best_dist = float("inf")
            for rv, rbag, ridx in ref_frames:
                d = abs(rv - ref_vel_idx)
                if d < best_dist:
                    best_dist = d
                    best_frame = (rv, rbag, ridx)
            if best_frame is not None:
                rv, rbag, ridx = best_frame
                ref_rgb = self._load_rgb(ref_sid, ref_cam_id, str(rbag), ridx)
                if ref_rgb is not None:
                    refs.append(ref_rgb)

        # Pad with zeros if not enough refs
        while len(refs) < self.n_refs:
            refs.append(torch.zeros(1, 3, self.img_height, self.img_width))

        refs = refs[:self.n_refs]

        # Ensure consistent sizes (resize if needed)
        H, W = self.img_height, self.img_width
        target = self._ensure_size(target, H, W)
        depth = self._ensure_size(depth, H, W)
        semantic = self._ensure_size(semantic, H, W)
        refs = [self._ensure_size(r, H, W) for r in refs]

        return refs, depth, semantic, target

    @staticmethod
    def _ensure_size(tensor, H, W):
        """Resize tensor if it doesn't match expected dimensions."""
        if tensor.shape[-2] != H or tensor.shape[-1] != W:
            tensor = F.interpolate(tensor, size=(H, W), mode="bilinear", align_corners=False)
        return tensor


# ---------------------------------------------------------------------------
# Collate function (matches Yutao's refs_collate)
# ---------------------------------------------------------------------------

def uscilab3d_collate(batch):
    """
    Collate batch:
      Input:  list of (refs_list, depth, semantic, target)
      Output: (refs_stacked, depth, semantic, target)
        refs_stacked: list of n_ref tensors each (B, 3, H, W)
        depth:    (B, 1, H, W)
        semantic: (B, 3, H, W)
        target:   (B, 3, H, W)
    """
    B = len(batch)
    n_ref = len(batch[0][0])

    refs_stacked = []
    for r_idx in range(n_ref):
        tensors = [batch[b][0][r_idx] for b in range(B)]
        refs_stacked.append(torch.cat(tensors, dim=0))

    depth = torch.cat([b[1] for b in batch], dim=0)
    semantic = torch.cat([b[2] for b in batch], dim=0)
    target = torch.cat([b[3] for b in batch], dim=0)

    return refs_stacked, depth, semantic, target


# ---------------------------------------------------------------------------
# CLI for testing / stats
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="USCILab3D Diffusion Dataset")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--mapping-dir", required=True)
    parser.add_argument("--projection-dir", default=None)
    parser.add_argument("--semantic-dir", default=None)
    parser.add_argument("--extracted-dir", default=None)
    parser.add_argument("--global-poses", default=None)
    parser.add_argument("--stats", action="store_true", help="Print dataset stats and exit")
    parser.add_argument("--test-load", type=int, default=0,
                        help="Try loading N random samples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ds = USCILab3DDataset(
        data_root=args.data_root,
        mapping_dir=args.mapping_dir,
        projection_dir=args.projection_dir,
        semantic_dir=args.semantic_dir,
        extracted_dir=args.extracted_dir,
        global_poses_path=args.global_poses,
    )

    if args.stats:
        print("Sessions:      {}".format(len(ds.sessions)))
        print("Total samples: {}".format(len(ds)))
        print("KD-tree entries: {}".format(len(ds.ref_index.centers)))
        for sid, si in sorted(ds.sessions.items()):
            n_frames = len(si.mapping)
            n_poses = len(si.poses)
            print("  {} : {} velodyne scans, {} poses".format(sid, n_frames, n_poses))

    if args.test_load > 0:
        import time
        indices = random.sample(range(len(ds)), min(args.test_load, len(ds)))
        for i, idx in enumerate(indices):
            t0 = time.time()
            refs, depth, semantic, target = ds[idx]
            dt = time.time() - t0
            print("Sample {:3d} (idx={:6d}): target={} depth={} sem={} refs={}x{} ({:.2f}s)".format(
                i, idx,
                tuple(target.shape),
                tuple(depth.shape),
                tuple(semantic.shape),
                len(refs),
                tuple(refs[0].shape) if refs else "?",
                dt,
            ))


if __name__ == "__main__":
    main()
