#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CopyTask:
    source: str
    target: str
    label: str


COPY_TASKS = [
    CopyTask("output/base_seq00_allframes", "output/experiments/base/seq00_allframes", "main_base_seq00"),
    CopyTask("output/loop_seq00_allframes", "output/experiments/loop_closure/seq00_allframes", "main_loop_seq00"),
    CopyTask("output/confidence_seq00_allframes", "output/experiments/confidence_v1/seq00_allframes", "main_conf_v1_seq00"),
    CopyTask("output/confidence_v2_seq00_allframes", "output/experiments/confidence_v2/seq00_allframes", "main_conf_v2_seq00"),
    CopyTask("output/loop_conf_seq00_allframes", "output/experiments/loop_closure_confidence_v1/seq00_allframes", "main_loop_conf_v1_seq00"),
    CopyTask("output/loop_conf_v2_seq00_allframes", "output/experiments/loop_closure_confidence_v2/seq00_allframes", "main_loop_conf_v2_seq00"),
    CopyTask("output/batch_gt_smoke", "output/experiments/base/batch_gt_smoke", "main_batch_gt_smoke"),
    CopyTask("output/method_compare_seq00", "output/comparisons/method_compare_seq00", "main_method_compare_seq00"),
    CopyTask("output/method_compare_seq00_w_deg", "output/comparisons/method_compare_seq00_w_deg", "main_method_compare_seq00_w_deg"),
    CopyTask("output/method_compare_seq00_w_deg_info", "output/comparisons/method_compare_seq00_w_deg_info", "main_method_compare_seq00_w_deg_info"),
    CopyTask(
        "ALFRED_iSAM2-degeneracy2.0/output/degeneracy_seq00_allframes",
        "output/experiments/degeneracy/seq00_allframes",
        "alfred_degeneracy_seq00",
    ),
    CopyTask(
        "ALFRED_iSAM2-degeneracy2.0/output/degeneracy_loop_seq00_allframes",
        "output/experiments/degeneracy_loop/seq00_allframes",
        "alfred_degeneracy_loop_seq00",
    ),
    CopyTask(
        "MO_iSAM2-info_weight/output/baseline_no_loops",
        "output/experiments/mo_baseline_no_loops/seq00_allframes",
        "mo_baseline_no_loops_seq00",
    ),
    CopyTask(
        "MO_iSAM2-info_weight/output/baseline_with_loops",
        "output/experiments/mo_baseline_with_loops/seq00_allframes",
        "mo_baseline_with_loops_seq00",
    ),
    CopyTask(
        "MO_iSAM2-info_weight/output/infoweighted_with_loops",
        "output/experiments/info_weighted_loop/seq00_allframes",
        "mo_infoweighted_loop_seq00",
    ),
    CopyTask("MO_iSAM2-info_weight/output/difference_summary.json", "output/comparisons/mo/difference_summary.json", "mo_diff_summary"),
    CopyTask("MO_iSAM2-info_weight/output/seq00_ate_summary_bars.png", "output/comparisons/mo/seq00_ate_summary_bars.png", "mo_seq00_summary_plot"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Copy existing method/report outputs into a unified non-destructive layout under output/."
        )
    )
    ap.add_argument(
        "--repo-root",
        default=".",
        help="Repository root path (default: current working directory)",
    )
    ap.add_argument(
        "--manifest",
        default="output/manifests/output_migration_manifest.json",
        help="Path for migration manifest JSON",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying files",
    )
    return ap.parse_args()


def _count_dir_stats(path: Path) -> tuple[int, int]:
    files = 0
    bytes_total = 0
    for p in path.rglob("*"):
        if p.is_file():
            files += 1
            bytes_total += p.stat().st_size
    return files, bytes_total


def _copy_entry(src: Path, dst: Path) -> tuple[int, int]:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
        return _count_dir_stats(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return 1, dst.stat().st_size


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()

    manifest_entries: list[dict] = []

    for task in COPY_TASKS:
        src = repo_root / task.source
        dst = repo_root / task.target

        if not src.exists():
            manifest_entries.append(
                {
                    "label": task.label,
                    "status": "missing_source",
                    "source": str(src),
                    "target": str(dst),
                    "files": 0,
                    "bytes": 0,
                }
            )
            print(f"[SKIP] missing source: {src}")
            continue

        if args.dry_run:
            if src.is_dir():
                files, bytes_total = _count_dir_stats(src)
            else:
                files, bytes_total = 1, src.stat().st_size
            manifest_entries.append(
                {
                    "label": task.label,
                    "status": "dry_run",
                    "source": str(src),
                    "target": str(dst),
                    "files": files,
                    "bytes": bytes_total,
                }
            )
            print(f"[DRY-RUN] {src} -> {dst} ({files} files, {bytes_total} bytes)")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        files, bytes_total = _copy_entry(src, dst)
        manifest_entries.append(
            {
                "label": task.label,
                "status": "copied",
                "source": str(src),
                "target": str(dst),
                "files": files,
                "bytes": bytes_total,
            }
        )
        print(f"[COPY] {src} -> {dst} ({files} files, {bytes_total} bytes)")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "dry_run": bool(args.dry_run),
        "entries": manifest_entries,
    }

    manifest_path = repo_root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
