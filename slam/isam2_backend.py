from __future__ import annotations

from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.symbol_shorthand import X


def _pose3_from_matrix(t: np.ndarray) -> gtsam.Pose3:
    rot = gtsam.Rot3(t[:3, :3])
    trans = gtsam.Point3(float(t[0, 3]), float(t[1, 3]), float(t[2, 3]))
    return gtsam.Pose3(rot, trans)


def _matrix_from_pose3(pose: gtsam.Pose3) -> np.ndarray:
    return pose.matrix()


@dataclass(frozen=True)
class IsamUpdateStats:
    frame_idx: int
    used_fallback_motion: bool


class Isam2PoseGraph:
    """Incremental SE(3) pose graph using odometry and optional loop-closure factors."""

    def __init__(
        self,
        trans_sigma_m: float = 0.15,
        rot_sigma_rad: float = 0.05,
        loop_trans_sigma_m: float = 0.25,
        loop_rot_sigma_rad: float = 0.08,
        prior_trans_sigma_m: float = 1e-6,
        prior_rot_sigma_rad: float = 1e-6,
        relinearize_skip: int = 1,
    ) -> None:
        params = gtsam.ISAM2Params()
        params.relinearizeSkip = relinearize_skip
        self.isam = gtsam.ISAM2(params)

        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_init = gtsam.Values()

        self.odom_sigmas = np.array(
            [
                rot_sigma_rad,
                rot_sigma_rad,
                rot_sigma_rad,
                trans_sigma_m,
                trans_sigma_m,
                trans_sigma_m,
            ],
            dtype=np.float64,
        )
        self.loop_sigmas = np.array(
            [
                loop_rot_sigma_rad,
                loop_rot_sigma_rad,
                loop_rot_sigma_rad,
                loop_trans_sigma_m,
                loop_trans_sigma_m,
                loop_trans_sigma_m,
            ],
            dtype=np.float64,
        )
        self.prior_sigmas = np.array(
            [
                prior_rot_sigma_rad,
                prior_rot_sigma_rad,
                prior_rot_sigma_rad,
                prior_trans_sigma_m,
                prior_trans_sigma_m,
                prior_trans_sigma_m,
            ],
            dtype=np.float64,
        )

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(self.odom_sigmas)
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(self.loop_sigmas)
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(self.prior_sigmas)

        self.latest_idx = 0
        identity = np.eye(4, dtype=np.float64)
        self.pending_graph.add(gtsam.PriorFactorPose3(X(0), _pose3_from_matrix(identity), self.prior_noise))
        self.pending_init.insert(X(0), _pose3_from_matrix(identity))
        self.isam.update(self.pending_graph, self.pending_init)
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_init = gtsam.Values()

    def _add_odometry_with_noise(
        self,
        prev_idx: int,
        curr_idx: int,
        t_prev_to_curr: np.ndarray,
        noise,
    ) -> IsamUpdateStats:
        measurement = _pose3_from_matrix(t_prev_to_curr)
        self.pending_graph.add(gtsam.BetweenFactorPose3(X(prev_idx), X(curr_idx), measurement, noise))

        estimate = self.isam.calculateEstimate()
        if not estimate.exists(X(prev_idx)):
            raise KeyError(f"Missing pose estimate for index {prev_idx}")

        prev_pose = estimate.atPose3(X(prev_idx))
        curr_guess = prev_pose.compose(measurement)
        self.pending_init.insert(X(curr_idx), curr_guess)

        self.isam.update(self.pending_graph, self.pending_init)
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_init = gtsam.Values()
        self.latest_idx = curr_idx

        return IsamUpdateStats(frame_idx=curr_idx, used_fallback_motion=False)

    def _add_loop_closure_with_noise(
        self,
        src_idx: int,
        dst_idx: int,
        t_src_to_dst: np.ndarray,
        noise,
    ) -> None:
        if src_idx == dst_idx:
            raise ValueError("Loop closure requires different node indices")

        if src_idx > self.latest_idx or dst_idx > self.latest_idx:
            raise ValueError("Loop closure indices must already exist in graph")

        measurement = _pose3_from_matrix(t_src_to_dst)
        self.pending_graph.add(gtsam.BetweenFactorPose3(X(src_idx), X(dst_idx), measurement, noise))

        self.isam.update(self.pending_graph, gtsam.Values())
        self.pending_graph = gtsam.NonlinearFactorGraph()

    def _confidence_to_scale(
        self,
        confidence: float,
        min_scale: float,
        max_scale: float,
        aggressive: bool,
    ) -> float:
        conf = float(np.clip(confidence, 1e-3, 1.0))
        raw_scale = (1.0 / conf) if aggressive else (1.0 / np.sqrt(conf))
        return float(np.clip(raw_scale, min_scale, max_scale))

    def _robustify_noise(self, base_noise, robust_kernel: str | None, robust_k: float):
        if robust_kernel is None:
            return base_noise

        if robust_k <= 0.0:
            raise ValueError("robust_k must be > 0")

        kernel = robust_kernel.strip().lower()
        if kernel == "huber":
            m_estimator = gtsam.noiseModel.mEstimator.Huber.Create(robust_k)
        elif kernel == "cauchy":
            m_estimator = gtsam.noiseModel.mEstimator.Cauchy.Create(robust_k)
        else:
            raise ValueError("robust_kernel must be one of: huber, cauchy")

        return gtsam.noiseModel.Robust.Create(m_estimator, base_noise)

    def _confidence_scaled_odom_noise(
        self,
        confidence: float,
        min_scale: float,
        max_scale: float,
        aggressive: bool = False,
    ):
        scale = self._confidence_to_scale(
            confidence=confidence,
            min_scale=min_scale,
            max_scale=max_scale,
            aggressive=aggressive,
        )
        return gtsam.noiseModel.Diagonal.Sigmas(self.odom_sigmas * scale)

    def _confidence_scaled_loop_noise(
        self,
        confidence: float,
        min_scale: float,
        max_scale: float,
        aggressive: bool = False,
    ):
        scale = self._confidence_to_scale(
            confidence=confidence,
            min_scale=min_scale,
            max_scale=max_scale,
            aggressive=aggressive,
        )
        return gtsam.noiseModel.Diagonal.Sigmas(self.loop_sigmas * scale)

    def add_odometry(self, prev_idx: int, curr_idx: int, t_prev_to_curr: np.ndarray) -> IsamUpdateStats:
        if curr_idx != prev_idx + 1:
            raise ValueError("Only consecutive frame factors are supported in this baseline")
        return self._add_odometry_with_noise(prev_idx, curr_idx, t_prev_to_curr, self.odom_noise)

    def add_odometry_confidence_weighted(
        self,
        prev_idx: int,
        curr_idx: int,
        t_prev_to_curr: np.ndarray,
        confidence: float,
        min_scale: float = 0.85,
        max_scale: float = 4.0,
        aggressive: bool = False,
        robust_kernel: str | None = None,
        robust_k: float = 1.5,
    ) -> IsamUpdateStats:
        if curr_idx != prev_idx + 1:
            raise ValueError("Only consecutive frame factors are supported in this baseline")
        if min_scale <= 0.0:
            raise ValueError("min_scale must be > 0")
        if max_scale < min_scale:
            raise ValueError("max_scale must be >= min_scale")

        noise = self._confidence_scaled_odom_noise(
            confidence,
            min_scale,
            max_scale,
            aggressive=aggressive,
        )
        noise = self._robustify_noise(noise, robust_kernel=robust_kernel, robust_k=robust_k)
        return self._add_odometry_with_noise(prev_idx, curr_idx, t_prev_to_curr, noise)

    def add_loop_closure(self, src_idx: int, dst_idx: int, t_src_to_dst: np.ndarray) -> None:
        self._add_loop_closure_with_noise(
            src_idx=src_idx,
            dst_idx=dst_idx,
            t_src_to_dst=t_src_to_dst,
            noise=self.loop_noise,
        )

    def add_loop_closure_confidence_weighted(
        self,
        src_idx: int,
        dst_idx: int,
        t_src_to_dst: np.ndarray,
        confidence: float,
        min_scale: float = 0.8,
        max_scale: float = 8.0,
        aggressive: bool = True,
        robust_kernel: str | None = "huber",
        robust_k: float = 1.5,
    ) -> None:
        if min_scale <= 0.0:
            raise ValueError("min_scale must be > 0")
        if max_scale < min_scale:
            raise ValueError("max_scale must be >= min_scale")

        noise = self._confidence_scaled_loop_noise(
            confidence=confidence,
            min_scale=min_scale,
            max_scale=max_scale,
            aggressive=aggressive,
        )
        noise = self._robustify_noise(noise, robust_kernel=robust_kernel, robust_k=robust_k)
        self._add_loop_closure_with_noise(src_idx, dst_idx, t_src_to_dst, noise)

    def pose_matrix(self, idx: int) -> np.ndarray:
        estimate = self.isam.calculateEstimate()
        if not estimate.exists(X(idx)):
            raise KeyError(f"Estimate missing node X({idx})")
        return _matrix_from_pose3(estimate.atPose3(X(idx)))

    def trajectory_matrices(self) -> np.ndarray:
        estimate = self.isam.calculateEstimate()
        mats = []
        for i in range(self.latest_idx + 1):
            if not estimate.exists(X(i)):
                raise KeyError(f"Estimate missing node X({i})")
            mats.append(_matrix_from_pose3(estimate.atPose3(X(i))))
        return np.stack(mats, axis=0)
