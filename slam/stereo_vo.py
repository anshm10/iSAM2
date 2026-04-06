from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .kitti_loader import StereoCalibration


def _reproj_residuals(
    object_points_n3: np.ndarray,
    image_points_n2: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    obj = np.asarray(object_points_n3, dtype=np.float64).reshape(-1, 1, 3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, None)
    pred = proj.reshape(-1, 2)
    return (pred - np.asarray(image_points_n2, dtype=np.float64).reshape(-1, 2)).ravel()


def stereo_pnp_observability_condition(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    eps: float = 1e-5,
) -> float:
    """Approximate pose observability from Gauss–Newton information (JᵀJ) condition number."""
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3).copy()
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3).copy()
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = int(object_points.shape[0])
    if n < 6:
        return float("inf")

    r_mat, _ = cv2.Rodrigues(rvec)

    def rotated_rvec(delta_r: np.ndarray) -> np.ndarray:
        d_r, _ = cv2.Rodrigues(delta_r)
        r_new, _ = cv2.Rodrigues(r_mat @ d_r)
        return r_new.reshape(3)

    f0 = _reproj_residuals(object_points, image_points, rvec, tvec, camera_matrix)
    jac = np.zeros((f0.size, 6), dtype=np.float64)
    for j in range(3):
        dr = np.zeros(3, dtype=np.float64)
        dr[j] = eps
        r1 = rotated_rvec(dr)
        f1 = _reproj_residuals(object_points, image_points, r1, tvec, camera_matrix)
        jac[:, j] = (f1 - f0) / eps
    for j in range(3):
        dt = np.zeros(3, dtype=np.float64)
        dt[j] = eps
        f1 = _reproj_residuals(object_points, image_points, rvec, tvec + dt, camera_matrix)
        jac[:, 3 + j] = (f1 - f0) / eps

    h = jac.T @ jac
    h = 0.5 * (h + h.T)
    eig = np.linalg.eigvalsh(h)
    pos = eig[eig > 1e-18]
    if pos.size == 0:
        return float("inf")
    lam_max = float(pos.max())
    lam_min = float(pos.min())
    if lam_min <= 0.0:
        return float("inf")
    return lam_max / lam_min


@dataclass(frozen=True)
class MotionEstimate:
    t_prev_to_curr: np.ndarray
    inliers: int
    total_matches: int
    pnp_condition_number: float = float("nan")


class StereoVisualOdometry:
    """Stereo VO front-end producing relative transforms between consecutive frames."""

    def __init__(
        self,
        calib: StereoCalibration,
        max_features: int = 3000,
        ratio_test: float = 0.75,
        max_stereo_y_diff_px: float = 2.0,
        min_stereo_disparity_px: float = 0.5,
        min_pnp_inliers: int = 25,
    ) -> None:
        self.calib = calib
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher_hamming = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test = ratio_test
        self.max_stereo_y_diff_px = max_stereo_y_diff_px
        self.min_stereo_disparity_px = min_stereo_disparity_px
        self.min_pnp_inliers = min_pnp_inliers

    def _detect(self, image: np.ndarray) -> Tuple[list[cv2.KeyPoint], np.ndarray | None]:
        return self.detector.detectAndCompute(image, None)

    def _knn_ratio_matches(self, desc_a: np.ndarray, desc_b: np.ndarray) -> list[cv2.DMatch]:
        knn = self.matcher_hamming.knnMatch(desc_a, desc_b, k=2)
        good: list[cv2.DMatch] = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good

    def estimate_prev_to_curr(
        self,
        left_prev: np.ndarray,
        right_prev: np.ndarray,
        left_curr: np.ndarray,
        compute_observability: bool = False,
    ) -> MotionEstimate | None:
        kp_l_prev, desc_l_prev = self._detect(left_prev)
        kp_r_prev, desc_r_prev = self._detect(right_prev)
        kp_l_curr, desc_l_curr = self._detect(left_curr)

        if (
            desc_l_prev is None
            or desc_r_prev is None
            or desc_l_curr is None
            or len(kp_l_prev) < 20
            or len(kp_r_prev) < 20
            or len(kp_l_curr) < 20
        ):
            return None

        stereo_matches = self._knn_ratio_matches(desc_l_prev, desc_r_prev)
        if len(stereo_matches) < 20:
            return None

        obj_pts_prev: list[np.ndarray] = []
        desc_for_obj: list[np.ndarray] = []
        used_kp_prev: list[cv2.KeyPoint] = []

        fx = self.calib.k_left[0, 0]
        fy = self.calib.k_left[1, 1]
        cx = self.calib.k_left[0, 2]
        cy = self.calib.k_left[1, 2]
        baseline = self.calib.baseline_m

        for m in stereo_matches:
            pt_l = kp_l_prev[m.queryIdx].pt
            pt_r = kp_r_prev[m.trainIdx].pt

            disparity = pt_l[0] - pt_r[0]
            y_diff = abs(pt_l[1] - pt_r[1])
            if disparity <= self.min_stereo_disparity_px or y_diff > self.max_stereo_y_diff_px:
                continue

            z = fx * baseline / disparity
            x = (pt_l[0] - cx) * z / fx
            y = (pt_l[1] - cy) * z / fy
            if not np.isfinite(x + y + z):
                continue
            if z <= 0.0 or z > 80.0:
                continue

            obj_pts_prev.append(np.array([x, y, z], dtype=np.float32))
            desc_for_obj.append(desc_l_prev[m.queryIdx])
            used_kp_prev.append(kp_l_prev[m.queryIdx])

        if len(obj_pts_prev) < 20:
            return None

        desc_for_obj_arr = np.asarray(desc_for_obj, dtype=np.uint8)
        temporal_matches = self._knn_ratio_matches(desc_for_obj_arr, desc_l_curr)
        if len(temporal_matches) < 15:
            return None

        object_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for m in temporal_matches:
            object_points.append(obj_pts_prev[m.queryIdx])
            image_points.append(np.array(kp_l_curr[m.trainIdx].pt, dtype=np.float32))

        object_points_arr = np.asarray(object_points, dtype=np.float32)
        image_points_arr = np.asarray(image_points, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_arr,
            image_points_arr,
            self.calib.k_left,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=3.0,
            confidence=0.999,
            iterationsCount=200,
        )

        if not success or inliers is None or len(inliers) < self.min_pnp_inliers:
            return None

        cond = float("nan")
        if compute_observability:
            idx = inliers.reshape(-1)
            obj_in = np.asarray(object_points_arr[idx], dtype=np.float64)
            img_in = np.asarray(image_points_arr[idx], dtype=np.float64)
            r_ref = np.asarray(rvec, dtype=np.float64).reshape(3, 1).copy()
            t_ref = np.asarray(tvec, dtype=np.float64).reshape(3, 1).copy()
            cv2.solvePnPRefineLM(
                obj_in.astype(np.float32),
                img_in.astype(np.float32),
                self.calib.k_left,
                None,
                r_ref,
                t_ref,
            )
            rvec = r_ref
            tvec = t_ref
            cond = stereo_pnp_observability_condition(
                obj_in,
                img_in,
                rvec.reshape(3),
                tvec.reshape(3),
                self.calib.k_left,
            )

        rmat, _ = cv2.Rodrigues(rvec)

        # PnP returns transform that maps points from prev camera frame into current frame.
        t_curr_prev = np.eye(4, dtype=np.float64)
        t_curr_prev[:3, :3] = rmat.astype(np.float64)
        t_curr_prev[:3, 3] = tvec.reshape(3).astype(np.float64)

        # iSAM2 between-factor convention below uses prev->curr pose increment (prev_T_curr),
        # so invert curr_T_prev.
        t_prev_curr = np.linalg.inv(t_curr_prev)

        return MotionEstimate(
            t_prev_to_curr=t_prev_curr,
            inliers=int(len(inliers)),
            total_matches=int(len(temporal_matches)),
            pnp_condition_number=cond,
        )

    def estimate_prev_to_curr_2d2d(
        self,
        left_prev: np.ndarray,
        left_curr: np.ndarray,
        min_inliers: int = 30,
    ) -> MotionEstimate | None:
        """Fallback relative pose using only 2D-2D correspondences and epipolar geometry.

        Translation magnitude is arbitrary (up-to-scale) and should be scaled by caller.
        """
        kp_prev, desc_prev = self._detect(left_prev)
        kp_curr, desc_curr = self._detect(left_curr)
        if (
            desc_prev is None
            or desc_curr is None
            or len(kp_prev) < 20
            or len(kp_curr) < 20
        ):
            return None

        matches = self._knn_ratio_matches(desc_prev, desc_curr)
        if len(matches) < 20:
            return None

        pts_prev = np.asarray([kp_prev[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_curr = np.asarray([kp_curr[m.trainIdx].pt for m in matches], dtype=np.float32)

        e, inlier_mask = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            self.calib.k_left,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if e is None or inlier_mask is None:
            return None

        inlier_count, r, t, _ = cv2.recoverPose(e, pts_prev, pts_curr, self.calib.k_left)
        if inlier_count < min_inliers:
            return None

        t_prev_curr = np.eye(4, dtype=np.float64)
        t_prev_curr[:3, :3] = r.astype(np.float64)
        t_prev_curr[:3, 3] = t.reshape(3).astype(np.float64)
        return MotionEstimate(
            t_prev_to_curr=t_prev_curr,
            inliers=int(inlier_count),
            total_matches=int(len(matches)),
        )
