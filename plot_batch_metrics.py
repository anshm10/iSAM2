#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot metrics from run_all_gt_sequences summary")
    ap.add_argument(
        "--summary-json",
        default="output/batch_gt/summary_metrics.json",
        help="Path to summary_metrics.json from batch runner",
    )
    ap.add_argument(
        "--output-dir",
        default="output/batch_gt/plots",
        help="Directory to write generated plots",
    )
    return ap.parse_args()


def _extract(rows: list[dict], key: str) -> np.ndarray:
    vals = []
    for r in rows:
        v = r.get(key)
        vals.append(np.nan if v is None else float(v))
    return np.asarray(vals, dtype=np.float64)


def _bar_plot(seqs: list[str], values: np.ndarray, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(seqs))
    mask = np.isfinite(values)

    ax.bar(x[mask], values[mask], color="#2a9d8f", width=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _multi_line_plot(seqs: list[str], rows: list[dict], out_path: Path) -> None:
    x = np.arange(len(seqs))
    d1 = _extract(rows, "rpe_trans_rmse_d1_m")
    d10 = _extract(rows, "rpe_trans_rmse_d10_m")
    d100 = _extract(rows, "rpe_trans_rmse_d100_m")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if np.isfinite(d1).any():
        ax.plot(x, d1, marker="o", linewidth=2, label="delta=1")
    if np.isfinite(d10).any():
        ax.plot(x, d10, marker="s", linewidth=2, label="delta=10")
    if np.isfinite(d100).any():
        ax.plot(x, d100, marker="^", linewidth=2, label="delta=100")

    ax.set_xticks(x)
    ax.set_xticklabels(seqs)
    ax.set_ylabel("RPE translational RMSE (m)")
    ax.set_title("RPE Translational RMSE by Sequence")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_json)
    out_dir = Path(args.output_dir)

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = data["rows"]
    seqs = [r["seq"] for r in rows]

    ate_rmse = _extract(rows, "ate_rmse_m")
    ate_rot = _extract(rows, "ate_rot_rmse_deg")
    runtime = _extract(rows, "runtime_sec")
    fallback = _extract(rows, "fallback_transitions")

    _bar_plot(
        seqs,
        ate_rmse,
        ylabel="ATE RMSE (m)",
        title="ATE Translational RMSE by Sequence",
        out_path=out_dir / "ate_trans_rmse.png",
    )
    _bar_plot(
        seqs,
        ate_rot,
        ylabel="ATE Rotation RMSE (deg)",
        title="ATE Rotational RMSE by Sequence",
        out_path=out_dir / "ate_rot_rmse.png",
    )
    _bar_plot(
        seqs,
        runtime,
        ylabel="Runtime (seconds)",
        title="Runtime by Sequence",
        out_path=out_dir / "runtime_sec.png",
    )
    _bar_plot(
        seqs,
        fallback,
        ylabel="Fallback transitions (count)",
        title="VO Fallback Count by Sequence",
        out_path=out_dir / "fallback_count.png",
    )
    _multi_line_plot(seqs, rows, out_dir / "rpe_trans_rmse_lines.png")

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
