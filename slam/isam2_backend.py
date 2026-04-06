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

        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
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
        )
        self.loop_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
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
        )
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
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
        )

        self.latest_idx = 0
        identity = np.eye(4, dtype=np.float64)
        self.pending_graph.add(gtsam.PriorFactorPose3(X(0), _pose3_from_matrix(identity), self.prior_noise))
        self.pending_init.insert(X(0), _pose3_from_matrix(identity))
        self.isam.update(self.pending_graph, self.pending_init)
        self.pending_graph = gtsam.NonlinearFactorGraph()
        self.pending_init = gtsam.Values()

    def _make_diagonal_noise(self, trans_sigma_m: float, rot_sigma_rad: float) -> gtsam.noiseModel.Diagonal:
        return gtsam.noiseModel.Diagonal.Sigmas(
            np.array(
                [rot_sigma_rad, rot_sigma_rad, rot_sigma_rad,
                 trans_sigma_m, trans_sigma_m, trans_sigma_m],
                dtype=np.float64,
            )
        )

    def add_odometry(self, prev_idx: int, curr_idx: int, t_prev_to_curr: np.ndarray) -> IsamUpdateStats:
        if curr_idx != prev_idx + 1:
            raise ValueError("Only consecutive frame factors are supported in this baseline")

        measurement = _pose3_from_matrix(t_prev_to_curr)
        self.pending_graph.add(gtsam.BetweenFactorPose3(X(prev_idx), X(curr_idx), measurement, self.odom_noise))

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

    def add_odometry_with_sigma(
        self,
        prev_idx: int,
        curr_idx: int,
        t_prev_to_curr: np.ndarray,
        trans_sigma_m: float,
        rot_sigma_rad: float,
    ) -> IsamUpdateStats:
        """Like add_odometry but with caller-supplied per-factor noise sigmas."""
        if curr_idx != prev_idx + 1:
            raise ValueError("Only consecutive frame factors are supported")

        noise = self._make_diagonal_noise(trans_sigma_m, rot_sigma_rad)
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

    def add_loop_closure(self, src_idx: int, dst_idx: int, t_src_to_dst: np.ndarray) -> None:
        if src_idx == dst_idx:
            raise ValueError("Loop closure requires different node indices")

        if src_idx > self.latest_idx or dst_idx > self.latest_idx:
            raise ValueError("Loop closure indices must already exist in graph")

        measurement = _pose3_from_matrix(t_src_to_dst)
        self.pending_graph.add(
            gtsam.BetweenFactorPose3(X(src_idx), X(dst_idx), measurement, self.loop_noise)
        )

        self.isam.update(self.pending_graph, gtsam.Values())
        self.pending_graph = gtsam.NonlinearFactorGraph()

    def add_loop_closure_with_sigma(
        self,
        src_idx: int,
        dst_idx: int,
        t_src_to_dst: np.ndarray,
        trans_sigma_m: float,
        rot_sigma_rad: float,
    ) -> None:
        """Like add_loop_closure but with caller-supplied per-factor noise sigmas."""
        if src_idx == dst_idx:
            raise ValueError("Loop closure requires different node indices")
        if src_idx > self.latest_idx or dst_idx > self.latest_idx:
            raise ValueError("Loop closure indices must already exist in graph")

        noise = self._make_diagonal_noise(trans_sigma_m, rot_sigma_rad)
        measurement = _pose3_from_matrix(t_src_to_dst)
        self.pending_graph.add(
            gtsam.BetweenFactorPose3(X(src_idx), X(dst_idx), measurement, noise)
        )
        self.isam.update(self.pending_graph, gtsam.Values())
        self.pending_graph = gtsam.NonlinearFactorGraph()

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
