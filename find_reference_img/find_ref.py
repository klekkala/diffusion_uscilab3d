#!/usr/bin/env python3
"""
NCLT cross-trajectory reference retrieval (KD-tree shortlist + orientation re-rank)
PATCHED for:
  - time-based subsampling (e.g., keep one pose every 0.1s)
  - memory-safe metadata arrays (no Python tuple-per-candidate)
  - optional local refinement around the best subsampled match

Given:
  - a TARGET ground-truth pose (timestamp + x,y,z,roll,pitch,yaw) from session A
  - a TARGET camera id (0..5)
Searches:
  - all (timestamp, cam_id) camera poses in session B (optionally subsampled)
Returns:
  - best matching (timestamp, cam_id) in session B based on:
      score = -alpha * distance + beta * forward_alignment

Assumptions:
  - Ground truth CSV format (whitespace-separated):
      timestamp  x  y  z  roll  pitch  yaw
    roll/pitch/yaw are in radians (your sample supports this).
  - Camera extrinsics are provided as JSON with 4x4 matrices (body->cam_i).
  - Camera forward axis assumed +Z in camera frame (OpenCV-like). Toggle to +X if needed.

Performance:
  - With 835,469 timestamps, you SHOULD subsample (e.g. 0.1s) to keep the KD-tree small.
  - Metadata stored in NumPy arrays; centers/forwards stored as float32 to reduce memory.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception as e:
    raise ImportError("Requires scipy. Install: pip install scipy") from e


# ----------------------------
# SE(3) helpers
# ----------------------------

def rpy_to_R_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rz = np.array([[cy, -sy, 0.0],
                   [sy,  cy, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    Ry = np.array([[ cp, 0.0, sp],
                   [0.0, 1.0, 0.0],
                   [-sp, 0.0, cp]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0,  cr, -sr],
                   [0.0,  sr,  cr]], dtype=np.float64)
    return Rz @ Ry @ Rx


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_w_b_from_pose6(pose6: np.ndarray) -> np.ndarray:
    """pose6 = [x,y,z,roll,pitch,yaw] with angles in radians."""
    x, y, z, roll, pitch, yaw = map(float, pose6.tolist())
    R = rpy_to_R_zyx(roll, pitch, yaw)
    t = np.array([x, y, z], dtype=np.float64)
    return make_T(R, t)


def cam_forward(R_w_c: np.ndarray, axis: str = "z") -> np.ndarray:
    """
    Unit forward vector in world frame.
    Usually +Z for OpenCV-like cameras. Use +X via --forward_axis x if needed.
    """
    if axis.lower() == "z":
        f = R_w_c[:, 2]
    elif axis.lower() == "x":
        f = R_w_c[:, 0]
    else:
        raise ValueError("forward axis must be 'z' or 'x'")
    n = np.linalg.norm(f) + 1e-12
    return f / n


# ----------------------------
# Data loading + subsampling
# ----------------------------

def load_gt_csv_whitespace(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads NCLT GT poses. Expected columns:
      timestamp  x  y  z  roll  pitch  yaw
    Returns:
      ts: (N,) int64
      pose6: (N,6) float64 [x,y,z,roll,pitch,yaw]
    """
    arr = np.loadtxt(str(path), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 7:
        raise ValueError(f"Expected >=7 columns in {path}, got {arr.shape[1]}")
    ts = arr[:, 0].astype(np.int64)
    pose6 = arr[:, 1:7].astype(np.float64)
    return ts, pose6


def subsample_by_time(ts: np.ndarray, pose6: np.ndarray, dt_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep one pose every dt_seconds based on timestamp deltas.
    NCLT timestamps are typically in microseconds.
    """
    if dt_seconds <= 0:
        return ts, pose6
    dt_us = int(dt_seconds * 1e6)

    keep = np.empty(ts.shape[0], dtype=bool)
    keep[:] = False

    last = -10**18
    for i, t in enumerate(ts):
        t_int = int(t)
        if t_int - last >= dt_us:
            keep[i] = True
            last = t_int

    return ts[keep], pose6[keep]


def load_extrinsics_json(path: Path) -> Dict[int, np.ndarray]:
    """
    JSON format:
    {
      "0": [[...],[...],[...],[...]],
      "1": [[...],[...],[...],[...]],
      ...
    }
    """
    obj = json.loads(path.read_text())
    out: Dict[int, np.ndarray] = {}
    for k, v in obj.items():
        cam_id = int(k)
        M = np.array(v, dtype=np.float64)
        if M.shape != (4, 4):
            raise ValueError(f"Extrinsic for cam {cam_id} is not 4x4, got {M.shape}")
        out[cam_id] = M
    if not out:
        raise ValueError("No extrinsics loaded.")
    return out


# ----------------------------
# KD-tree index (memory-safe)
# ----------------------------

class CandidateIndex:
    """
    Holds:
      centers: (M,3) float32
      forwards:(M,3) float32
      meta_ts: (M,) int64
      meta_cam:(M,) uint8
      tree: KD-tree over centers
    """
    def __init__(self, centers, forwards, meta_ts, meta_cam, tree):
        self.centers = centers
        self.forwards = forwards
        self.meta_ts = meta_ts
        self.meta_cam = meta_cam
        self.tree = tree


def build_candidate_index(other_ts: np.ndarray,
                          other_pose6: np.ndarray,
                          T_b_ci: Dict[int, np.ndarray],
                          forward_axis: str = "z",
                          dtype_vec=np.float32) -> CandidateIndex:
    cam_ids = np.array(sorted(T_b_ci.keys()), dtype=np.int32)
    n_pose = other_ts.shape[0]
    n_cam = cam_ids.shape[0]
    M = n_pose * n_cam

    centers = np.empty((M, 3), dtype=dtype_vec)
    forwards = np.empty((M, 3), dtype=dtype_vec)
    meta_ts = np.empty((M,), dtype=np.int64)
    meta_cam = np.empty((M,), dtype=np.uint8)

    j = 0
    for ts, pose6 in zip(other_ts, other_pose6):
        T_w_b = T_w_b_from_pose6(pose6)
        for cam_id in cam_ids:
            T_w_c = T_w_b @ T_b_ci[int(cam_id)]
            centers[j, :] = T_w_c[:3, 3].astype(dtype_vec)
            forwards[j, :] = cam_forward(T_w_c[:3, :3], axis=forward_axis).astype(dtype_vec)
            meta_ts[j] = int(ts)
            meta_cam[j] = int(cam_id)
            j += 1

    tree = cKDTree(centers.astype(np.float64))  # scipy expects float64 internally
    return CandidateIndex(centers=centers, forwards=forwards, meta_ts=meta_ts, meta_cam=meta_cam, tree=tree)


# ----------------------------
# Query + optional refinement
# ----------------------------

def query_best(index: CandidateIndex,
               target_pose6: np.ndarray,
               target_cam_id: int,
               T_b_ci: Dict[int, np.ndarray],
               topK: int = 200,
               alpha: float = 1.0,
               beta: float = 8.0,
               forward_axis: str = "z",
               min_align: Optional[float] = None) -> Tuple[float, float, float, int, int]:
    """
    Returns:
      (score, dist, align, timestamp_other, cam_id_other)
    """
    if target_cam_id not in T_b_ci:
        raise KeyError(f"target_cam_id {target_cam_id} not in extrinsics keys {sorted(T_b_ci.keys())}")

    # Target camera pose in world
    T_w_b_t = T_w_b_from_pose6(target_pose6)
    T_w_c_t = T_w_b_t @ T_b_ci[target_cam_id]
    pt = T_w_c_t[:3, 3]
    ft = cam_forward(T_w_c_t[:3, :3], axis=forward_axis).astype(np.float32)

    k = min(int(topK), index.centers.shape[0])
    dists, idxs = index.tree.query(pt.astype(np.float64), k=k)
    if np.isscalar(idxs):
        idxs = np.array([int(idxs)], dtype=np.int64)
        dists = np.array([float(dists)], dtype=np.float64)

    best = None  # (score, dist, align, ts, cam)
    for dist, idx in zip(dists, idxs):
        idx = int(idx)
        align = float(np.dot(index.forwards[idx].astype(np.float32), ft))
        if min_align is not None and align < float(min_align):
            continue
        score = -alpha * float(dist) + beta * align
        ts = int(index.meta_ts[idx])
        cam = int(index.meta_cam[idx])
        if best is None or score > best[0]:
            best = (score, float(dist), align, ts, cam)

    if best is None:
        raise RuntimeError("No candidate survived min_align filtering. Lower --min_align_deg or omit it.")
    return best


def refine_locally_fullrate(other_ts_full: np.ndarray,
                            other_pose6_full: np.ndarray,
                            T_b_ci: Dict[int, np.ndarray],
                            target_pose6: np.ndarray,
                            target_cam_id: int,
                            approx_ts: int,
                            approx_cam: int,
                            window_sec: float,
                            alpha: float,
                            beta: float,
                            forward_axis: str,
                            min_align: Optional[float]) -> Tuple[float, float, float, int, int]:
    """
    Local refinement around approx_ts using the FULL-RATE other session GT.
    Searches timestamps within +/- window_sec around approx_ts, across all cams (or you can restrict).
    """
    if window_sec <= 0:
        # no refinement; return approx as-is but with a recomputed score for consistency
        return approx_ts, approx_cam  # type: ignore

    # Build target cam pose
    T_w_b_t = T_w_b_from_pose6(target_pose6)
    T_w_c_t = T_w_b_t @ T_b_ci[target_cam_id]
    pt = T_w_c_t[:3, 3]
    ft = cam_forward(T_w_c_t[:3, :3], axis=forward_axis)

    win_us = int(window_sec * 1e6)
    lo = approx_ts - win_us
    hi = approx_ts + win_us

    # Find range in sorted timestamps (NCLT GT is typically time-ordered)
    i0 = int(np.searchsorted(other_ts_full, lo, side="left"))
    i1 = int(np.searchsorted(other_ts_full, hi, side="right"))
    i0 = max(0, min(i0, other_ts_full.shape[0]))
    i1 = max(0, min(i1, other_ts_full.shape[0]))

    cam_ids = sorted(T_b_ci.keys())

    best = None  # (score, dist, align, ts, cam)
    for i in range(i0, i1):
        ts = int(other_ts_full[i])
        T_w_b = T_w_b_from_pose6(other_pose6_full[i])
        for cam_id in cam_ids:
            T_w_c = T_w_b @ T_b_ci[cam_id]
            pc = T_w_c[:3, 3]
            dist = float(np.linalg.norm(pc - pt))
            align = float(np.dot(cam_forward(T_w_c[:3, :3], axis=forward_axis), ft))
            if min_align is not None and align < float(min_align):
                continue
            score = -alpha * dist + beta * align
            if best is None or score > best[0]:
                best = (score, dist, align, ts, int(cam_id))

    if best is None:
        # fallback to approx (should be rare)
        return (float("-inf"), float("inf"), -1.0, approx_ts, approx_cam)
    return best


# ----------------------------
# Target selection
# ----------------------------

def parse_target_pose(args) -> np.ndarray:
    if args.target_pose is not None:
        vals = [float(x) for x in args.target_pose.strip().replace(",", " ").split()]
        if len(vals) != 6:
            raise ValueError("--target_pose must have 6 values: x y z roll pitch yaw")
        return np.array(vals, dtype=np.float64)

    if args.target_timestamp is None or args.target_gt_csv is None:
        raise ValueError("Provide either --target_pose or (--target_timestamp and --target_gt_csv).")

    ts_arr, pose6_arr = load_gt_csv_whitespace(Path(args.target_gt_csv))
    tgt_ts = int(args.target_timestamp)
    idx = int(np.argmin(np.abs(ts_arr - tgt_ts)))
    if args.verbose:
        print(f"[info] target_timestamp={tgt_ts}, nearest_in_target_csv={ts_arr[idx]}, dt={int(ts_arr[idx]-tgt_ts)}")
    return pose6_arr[idx]


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other_gt_csv", type=str, required=True,
                    help="Path to OTHER session GT CSV: ts x y z roll pitch yaw.")
    ap.add_argument("--extrinsics_json", type=str, required=True,
                    help="JSON file of body->cam extrinsics (4x4) for cams.")
    ap.add_argument("--target_cam", type=int, required=True, help="Target camera id (0..5).")

    ap.add_argument("--target_pose", type=str, default=None,
                    help="Target pose: 'x y z roll pitch yaw' (angles in radians).")
    ap.add_argument("--target_timestamp", type=int, default=None,
                    help="Target timestamp (int) used with --target_gt_csv for nearest pose.")
    ap.add_argument("--target_gt_csv", type=str, default=None,
                    help="Path to TARGET session GT CSV (only needed if using --target_timestamp).")

    ap.add_argument("--topK", type=int, default=200, help="KD-tree shortlist size.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Distance weight.")
    ap.add_argument("--beta", type=float, default=8.0, help="Alignment weight.")
    ap.add_argument("--forward_axis", type=str, default="z", choices=["z", "x"],
                    help="Forward axis in camera frame (usually z).")
    ap.add_argument("--min_align_deg", type=float, default=None,
                    help="Optional: discard candidates with align dot < cos(min_align_deg).")
    ap.add_argument("--subsample_dt", type=float, default=0.1,
                    help="Subsample other GT by time in seconds (0 disables). Example: 0.1 keeps one pose every 0.1s.")
    ap.add_argument("--refine_window_sec", type=float, default=0.0,
                    help="Optional local refinement using full-rate OTHER GT within +/- window (seconds). 0 disables.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    other_gt_csv = Path(args.other_gt_csv)
    extr_path = Path(args.extrinsics_json)

    # Load extrinsics
    T_b_ci = load_extrinsics_json(extr_path)

    # Target pose
    target_pose6 = parse_target_pose(args)

    # Optional min_align
    min_align = None
    if args.min_align_deg is not None:
        min_align = math.cos(math.radians(float(args.min_align_deg)))

    # Load OTHER full-rate GT if refinement enabled
    if args.refine_window_sec and args.refine_window_sec > 0:
        other_ts_full, other_pose6_full = load_gt_csv_whitespace(other_gt_csv)
    else:
        other_ts_full, other_pose6_full = None, None  # type: ignore

    # Load OTHER GT (for KD-tree), then subsample
    other_ts, other_pose6 = load_gt_csv_whitespace(other_gt_csv)
    if args.subsample_dt and args.subsample_dt > 0:
        other_ts, other_pose6 = subsample_by_time(other_ts, other_pose6, dt_seconds=float(args.subsample_dt))

    if args.verbose:
        print(f"[info] other_gt_csv rows(full)={len(load_gt_csv_whitespace(other_gt_csv)[0])}")
        print(f"[info] other_gt_csv rows(used)={len(other_ts)} after subsample_dt={args.subsample_dt}")
        print(f"[info] cams={sorted(T_b_ci.keys())}")
        print(f"[info] scoring: score=-{args.alpha}*dist + {args.beta}*align, topK={args.topK}")
        if min_align is not None:
            print(f"[info] min_align dot >= {min_align:.6f} (from {args.min_align_deg} deg)")
        if args.refine_window_sec and args.refine_window_sec > 0:
            print(f"[info] local refinement enabled: +/- {args.refine_window_sec} sec on full-rate other GT")

    # Build KD-tree index
    index = build_candidate_index(other_ts, other_pose6, T_b_ci, forward_axis=args.forward_axis, dtype_vec=np.float32)

    # Query best on subsampled KD-tree
    score, dist, align, best_ts, best_cam = query_best(
        index=index,
        target_pose6=target_pose6,
        target_cam_id=args.target_cam,
        T_b_ci=T_b_ci,
        topK=args.topK,
        alpha=args.alpha,
        beta=args.beta,
        forward_axis=args.forward_axis,
        min_align=min_align
    )

    # Optional refinement in full-rate data around best_ts
    if args.refine_window_sec and args.refine_window_sec > 0:
        assert other_ts_full is not None and other_pose6_full is not None
        score2, dist2, align2, ts2, cam2 = refine_locally_fullrate(
            other_ts_full=other_ts_full,
            other_pose6_full=other_pose6_full,
            T_b_ci=T_b_ci,
            target_pose6=target_pose6,
            target_cam_id=args.target_cam,
            approx_ts=best_ts,
            approx_cam=best_cam,
            window_sec=float(args.refine_window_sec),
            alpha=float(args.alpha),
            beta=float(args.beta),
            forward_axis=args.forward_axis,
            min_align=min_align
        )
        # If refinement found something better, replace
        if score2 > score:
            score, dist, align, best_ts, best_cam = score2, dist2, align2, ts2, cam2

    print("BEST_MATCH")
    print(f"  timestamp_other: {best_ts}")
    print(f"  cam_id_other:    {best_cam}")
    print(f"  dist_m:          {dist:.6f}")
    print(f"  align_dot:       {align:.6f}")
    print(f"  score:           {score:.6f}")


if __name__ == "__main__":
    main()
