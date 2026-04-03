from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoCalibration:
    p0: np.ndarray
    p1: np.ndarray
    k_left: np.ndarray
    baseline_m: float


class KITTISequenceLoader:
    """Loads KITTI odometry sequence data (stereo grayscale + calibration + timestamps)."""

    def __init__(self, dataset_root: str | Path, sequence: str) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.sequence = f"{int(sequence):02d}"
        self.sequence_dir = self.dataset_root / "sequences" / self.sequence

        if not self.sequence_dir.exists():
            raise FileNotFoundError(f"Sequence directory not found: {self.sequence_dir}")

        self.image0_dir = self.sequence_dir / "image_0"
        self.image1_dir = self.sequence_dir / "image_1"
        self.times_path = self.sequence_dir / "times.txt"
        self.calib_path = self.sequence_dir / "calib.txt"
        self.gt_path = self.dataset_root / "poses" / f"{self.sequence}.txt"

        self.calib = self._read_calibration(self.calib_path)
        self.timestamps = self._read_timestamps(self.times_path)
        self.left_images = sorted(self.image0_dir.glob("*.png"))
        self.right_images = sorted(self.image1_dir.glob("*.png"))

        if not self.left_images:
            raise ValueError(f"No left images found in {self.image0_dir}")
        if len(self.left_images) != len(self.right_images):
            raise ValueError(
                f"Left/right image count mismatch: {len(self.left_images)} vs {len(self.right_images)}"
            )
        if len(self.timestamps) != len(self.left_images):
            raise ValueError(
                f"Timestamp/image count mismatch: {len(self.timestamps)} vs {len(self.left_images)}"
            )

    @staticmethod
    def _read_timestamps(path: Path) -> np.ndarray:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return np.asarray([float(x.strip()) for x in lines], dtype=np.float64)

    @staticmethod
    def _read_calibration(path: Path) -> StereoCalibration:
        entries: Dict[str, np.ndarray] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            key, values = line.split(":", maxsplit=1)
            nums = np.asarray([float(x) for x in values.split()], dtype=np.float64)
            if nums.size != 12:
                raise ValueError(f"Expected 12 numbers for {key} in {path}, got {nums.size}")
            entries[key.strip()] = nums.reshape(3, 4)

        if "P0" not in entries or "P1" not in entries:
            raise ValueError("calib.txt must contain P0 and P1")

        p0 = entries["P0"]
        p1 = entries["P1"]
        k_left = p0[:, :3]

        fx = p1[0, 0]
        if abs(fx) < 1e-12:
            raise ValueError("Invalid calibration: fx is zero")

        # For rectified KITTI stereo pairs: baseline = -Tx/fx where Tx is P1[0,3].
        baseline_m = -p1[0, 3] / fx

        return StereoCalibration(p0=p0, p1=p1, k_left=k_left, baseline_m=float(baseline_m))

    def num_frames(self) -> int:
        return len(self.left_images)

    def read_stereo(self, idx: int) -> Tuple[np.ndarray, np.ndarray, float]:
        if idx < 0 or idx >= self.num_frames():
            raise IndexError(f"Frame index out of bounds: {idx}")

        left = cv2.imread(str(self.left_images[idx]), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(self.right_images[idx]), cv2.IMREAD_GRAYSCALE)

        if left is None or right is None:
            raise ValueError(f"Failed to read stereo images at index {idx}")

        return left, right, float(self.timestamps[idx])

    def read_ground_truth(self) -> np.ndarray | None:
        if not self.gt_path.exists():
            return None

        mats: List[np.ndarray] = []
        for line in self.gt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                raise ValueError(f"GT line does not have 12 values in {self.gt_path}")
            t = np.eye(4, dtype=np.float64)
            t[:3, :4] = np.asarray(vals, dtype=np.float64).reshape(3, 4)
            mats.append(t)

        if not mats:
            return None

        return np.stack(mats, axis=0)
