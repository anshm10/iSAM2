from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np

from run_isam2_kitti import run_sequence


def _pick(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _read_pose_file(path: Path) -> np.ndarray:
    mats: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        vals = np.fromstring(line, sep=" ", dtype=np.float64)
        t = np.eye(4, dtype=np.float64)
        t[:3, :4] = vals.reshape(3, 4)
        mats.append(t)
    return np.stack(mats, axis=0)


def _raw_final_pose_errors(gt: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    rel = np.linalg.inv(gt[-1]) @ est[-1]
    trans = float(np.linalg.norm(rel[:3, 3]))
    cos_theta = float((np.trace(rel[:3, :3]) - 1.0) * 0.5)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    rot_deg = float(np.degrees(np.arccos(cos_theta)))
    return trans, rot_deg


def run_seq_compare(dataset_root: Path, seq: str, out_dir: Path) -> Path:
    seq_out = out_dir / f"seq_{seq}"
    seq_out.mkdir(parents=True, exist_ok=True)

    modes = ["base", "loop", "confidence"]
    results: dict[str, dict] = {}

    for mode in modes:
        est_path = seq_out / f"poses_est_{seq}_{mode}.txt"
        metrics_path = seq_out / f"metrics_{seq}_{mode}.json"
        log_path = seq_out / f"run_{seq}_{mode}.log"

        t0 = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as logf:
            with contextlib.redirect_stdout(logf), contextlib.redirect_stderr(logf):
                res = run_sequence(
                    dataset_root=str(dataset_root),
                    seq=seq,
                    max_frames=0,
                    output_path=est_path,
                    metrics_path=metrics_path,
                    min_inliers=25,
                    fallback_no_motion=True,
                    skip_metrics=False,
                    mode=mode,
                    conf_min_noise_scale=0.8,
                    conf_max_noise_scale=2.5,
                )
        res["runtime_sec"] = float(time.perf_counter() - t0)
        res["run_log"] = str(log_path)
        results[mode] = res

    gt_path = dataset_root / "poses" / f"{seq}.txt"
    gt = _read_pose_file(gt_path)

    mode_maps: dict[str, dict] = {mode: {} for mode in modes}
    for mode in modes:
        r = results[mode]
        m = r["metrics"]
        mm = mode_maps[mode]

        mm["EXTRA.runtime_sec"] = r.get("runtime_sec")
        mm["EXTRA.est_path"] = r.get("est_path")
        mm["EXTRA.metrics_path"] = r.get("metrics_path")
        mm["EXTRA.run_log"] = r.get("run_log")
        mm["EXTRA.accepted_transitions"] = r.get("accepted_transitions")
        mm["EXTRA.fallback_transitions"] = r.get("fallback_transitions")
        mm["EXTRA.loop_closures_added"] = r.get("loop_closures_added")

        est = _read_pose_file(Path(r["est_path"]))
        mm["EXTRA.num_poses_est"] = int(est.shape[0])
        mm["EXTRA.num_poses_gt"] = int(gt.shape[0])
        trans, rot = _raw_final_pose_errors(gt, est)
        mm["EXTRA.raw_final_pose_trans_error_m"] = trans
        mm["EXTRA.raw_final_pose_rot_error_deg"] = rot

        mm["ATE.align"] = _pick(m, ["ATE", "align"])
        mm["ATE.scale"] = _pick(m, ["ATE", "scale"])
        mm["ATE.num_poses"] = _pick(m, ["ATE", "num_poses"])

        for comp in ["ATE_trans_m", "ATE_rot_deg"]:
            for stat in ["rmse", "mean", "median", "std", "min", "max"]:
                mm[f"ATE.{comp}.{stat}"] = _pick(m, ["ATE", comp, stat])

        for d in ["1", "10", "100"]:
            mm[f"RPE.d{d}.count"] = _pick(m, ["RPE", "deltas", d, "count"])
            for comp in ["RPE_trans_m", "RPE_rot_deg"]:
                for stat in ["rmse", "mean", "median", "std", "min", "max"]:
                    mm[f"RPE.d{d}.{comp}.{stat}"] = _pick(m, ["RPE", "deltas", d, comp, stat])

        conf = r.get("confidence_weighting")
        if conf:
            for k, v in conf.items():
                mm[f"CONF.{k}"] = v

    all_metrics = sorted(set().union(*(mode_maps[m].keys() for m in modes)))

    lines: list[str] = []
    lines.append(f"# Sequence {seq}: Base vs Loop vs Confidence")
    lines.append("")
    lines.append(f"- ground_truth: {gt_path}")
    lines.append(f"- dataset_root: {dataset_root}")
    lines.append("")
    lines.append("| Metric | base | loop | confidence |")
    lines.append("|---|---:|---:|---:|")
    for metric in all_metrics:
        lines.append(
            f"| {metric} | {_fmt(mode_maps['base'].get(metric))} | {_fmt(mode_maps['loop'].get(metric))} | {_fmt(mode_maps['confidence'].get(metric))} |"
        )

    out_md = seq_out / f"seq_{seq}_modes_comparison.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = seq_out / f"seq_{seq}_modes_summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return out_md


def main() -> None:
    dataset_root = Path(
        "/gpfs/accounts/rob530w26s001_class_root/rob530w26s001_class/shared_data/dataset"
    )
    out_dir = Path("output/seq89_mode_compare")
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for seq in ["08", "09"]:
        outputs.append(run_seq_compare(dataset_root, seq, out_dir))

    print("Generated comparison tables:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
