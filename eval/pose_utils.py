from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class TrajectoryMetrics:
    rmse: float
    mean: float
    median: float
    std: float
    min: float
    max: float


def read_kitti_poses(path: str) -> np.ndarray:
    """Reads KITTI odometry poses.

    Each line: 12 floats for a 3x4 row-major matrix [R|t].
    Returns: (N, 4, 4) float64.
    """
    mats = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                raise ValueError(f"Expected 12 floats per line in {path}, got {len(vals)}")
            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = np.array(vals, dtype=np.float64).reshape(3, 4)
            mats.append(T)
    if not mats:
        raise ValueError(f"No poses found in {path}")
    return np.stack(mats, axis=0)


def write_kitti_poses(path: str, Ts: np.ndarray) -> None:
    Ts = np.asarray(Ts)
    if Ts.ndim != 3 or Ts.shape[1:] != (4, 4):
        raise ValueError("Ts must have shape (N,4,4)")
    with open(path, "w", encoding="utf-8") as f:
        for T in Ts:
            row = T[:3, :4].reshape(-1)
            f.write(" ".join(f"{x:.12g}" for x in row) + "\n")


def trajectory_positions(Ts: np.ndarray) -> np.ndarray:
    return np.asarray(Ts)[:, :3, 3]


def umeyama_alignment(
    src: np.ndarray, dst: np.ndarray, with_scale: bool
) -> Tuple[np.ndarray, float]:
    """Umeyama alignment from src->dst.

    Args:
        src: (N,3) points
        dst: (N,3) points
        with_scale: estimate isotropic scale

    Returns:
        T: (4,4) transform such that p_dst ~= (s*R*p_src + t)
        s: scale (1.0 if with_scale=False)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src and dst must both be (N,3)")

    n = src.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points for alignment")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst

    cov = (Y.T @ X) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = (X**2).sum() / n
        scale = float(np.trace(np.diag(D) @ S) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T, scale


def apply_sim3_to_poses(T_align: np.ndarray, scale: float, Ts: np.ndarray) -> np.ndarray:
    """Applies a similarity transform (s,R,t) to a list of SE(3) poses.

    We treat each pose as mapping from local->world. We align by:
        p' = s R p + t
    For poses, the rotation becomes R' = R_align * R_pose
    and translation becomes t' = s * R_align * t_pose + t_align

    Args:
        T_align: (4,4) with R,t (scale handled separately)
        scale: scalar s
        Ts: (N,4,4)

    Returns:
        (N,4,4)
    """
    Ts = np.asarray(Ts, dtype=np.float64)
    R_a = T_align[:3, :3]
    t_a = T_align[:3, 3]
    out = np.tile(np.eye(4), (Ts.shape[0], 1, 1)).astype(np.float64)
    out[:, :3, :3] = R_a @ Ts[:, :3, :3]
    out[:, :3, 3] = (scale * (R_a @ Ts[:, :3, 3].T)).T + t_a
    return out


def angle_from_rotation(R: np.ndarray) -> float:
    """Returns rotation angle in radians for a 3x3 rotation matrix."""
    tr = float(np.trace(R))
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def summarize_errors(err: np.ndarray) -> TrajectoryMetrics:
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(np.mean(err**2)))
    return TrajectoryMetrics(
        rmse=rmse,
        mean=float(np.mean(err)),
        median=float(np.median(err)),
        std=float(np.std(err)),
        min=float(np.min(err)),
        max=float(np.max(err)),
    )


def relative_transforms(Ts: np.ndarray, delta: int) -> np.ndarray:
    Ts = np.asarray(Ts, dtype=np.float64)
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if Ts.shape[0] <= delta:
        raise ValueError("Trajectory too short for given delta")

    inv = np.linalg.inv(Ts[:-delta])
    rel = inv @ Ts[delta:]
    return rel
