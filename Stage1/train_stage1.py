#!/usr/bin/env python3
"""Run MonoEv Stage-1 with exactly one camera alignment and one Gaussian refinement.

Pipeline:
  1. Align the original G0 to the reference image (camera_alignment.py).
  2. Refine the original G0 with that fixed camera (gaussian_refinement.py).
  3. Copy the camera and refined Gaussian to the final directory.
  4. Save initial (post-alignment) and final (post-refinement) metrics to metric.json.

Usage:
    python Stage1/train_stage1.py --config Stage1/fan_stage1.ini
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import subprocess
import sys
from pathlib import Path


STAGE1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE1_DIR.parent


def _clean_value(value: str | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if text.lower() in ("", "none", "null"):
        return None

    return text


def _resolve_path(value: str, base_dir: Path) -> Path:
    p = Path(value).expanduser()

    if not p.is_absolute():
        p = base_dir / p

    return p.resolve()


def _read_config(config_path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser(interpolation=None)

    loaded = cfg.read(config_path, encoding="utf-8")

    if not loaded:
        raise FileNotFoundError(f"Cannot read config: {config_path}")

    return cfg


def _cfg_path(
    cfg: configparser.ConfigParser,
    config_path: Path,
    section: str,
    key: str,
    default: Path | None = None,
) -> Path:
    if cfg.has_option(section, key):
        value = _clean_value(cfg.get(section, key))

        if value is not None:
            return _resolve_path(value, config_path.parent)

    if default is None:
        raise KeyError(f"Missing [{section}] {key}")

    return default.resolve()


def _run(label: str, command: list[str]) -> None:
    print("\n" + "=" * 78, flush=True)
    print(label, flush=True)
    print("=" * 78, flush=True)
    print("COMMAND:", " ".join(command), flush=True)

    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} was not produced: {path}"
        )


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data if isinstance(data, dict) else {"value": data}


def _parse_metric_value(value: str):
    """Parse known numeric/JSON metric values; preserve all other text."""
    value = value.strip()
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        pass
    try:
        result = float(value)
        if result == result and result not in (float("inf"), float("-inf")):
            return result
    except ValueError:
        pass
    return value


def _read_text_metrics(path: Path) -> dict:
    """Retain the full original log plus structured key/value entries."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    parsed = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        if key:
            parsed[key] = _parse_metric_value(value)
    return {"values": parsed, "raw_text": raw}


def _optional_json(path: Path):
    return _read_json(path) if path.is_file() else None


def _optional_text(path: Path):
    return _read_text_metrics(path) if path.is_file() else None


def _metrics_from_txt(path: Path):
    result = _optional_text(path)
    return result["values"] if result is not None else None


def _numeric_delta(initial, final):
    if not isinstance(initial, dict) or not isinstance(final, dict):
        return None
    output = {}
    for key in sorted(set(initial) & set(final)):
        first, last = initial[key], final[key]
        if (isinstance(first, (int, float)) and not isinstance(first, bool)
                and isinstance(last, (int, float)) and not isinstance(last, bool)):
            output[key] = last - first
    return output


def _write_final_metrics(
    output_path: Path,
    config_path: Path,
    original_g0: Path,
    camera_path: Path,
    final_g0: Path,
    alignment_dir: Path,
    refinement_dir: Path,
) -> None:
    """Aggregate actual two-stage before/after evaluations without recomputing them.

    Camera alignment comparison: same raw G0, pose changes.
    Refinement comparison: same aligned camera, G0 -> corrected G0.
    They are different experiments and must NOT be numerically interchanged.
    """
    alignment_eval = alignment_dir / "before_after_evaluation"
    refinement_eval = refinement_dir / "before_after_evaluation"
    camera_metrics_path = alignment_dir / "camera_alignment_metrics.json"
    summary_path = refinement_dir / "stage1_summary.txt"

    cam_initial = _optional_json(alignment_eval / "initial_metrics.json")
    cam_final = _optional_json(alignment_eval / "final_metrics.json")
    cam_comparison = _optional_json(alignment_eval / "metrics_comparison.json")
    ref_initial = _metrics_from_txt(refinement_eval / "initial_metrics.txt")
    ref_final = _metrics_from_txt(refinement_eval / "final_metrics.txt")

    # Refuse to mislabel the post-alignment camera metrics as G0-before-refinement.
    # A missing evaluation is null; it is never silently replaced by a different one.
    data = {
        "pipeline": "MonoEv Stage-1",
        "execution": {"camera_alignment_runs": 1, "gaussian_refinement_runs": 1},
        "config": str(config_path),
        "original_g0_ply": str(original_g0),
        "aligned_camera_json": str(camera_path),
        "corrected_g0_ply": str(final_g0),
        "camera_alignment": {
            "comparison": "same_original_G0_initial_pose_vs_final_pose",
            "initial_definition": (cam_comparison or {}).get("initial_definition"),
            "initial_metrics": cam_initial,
            "final_metrics": cam_final,
            "delta_final_minus_initial": _numeric_delta(cam_initial, cam_final),
            "full_comparison": cam_comparison,
            "original_alignment_report": _optional_json(camera_metrics_path),
            "evaluation_dir": str(alignment_eval),
        },
        "gaussian_refinement": {
            "comparison": "same_aligned_camera_original_G0_vs_corrected_G0",
            "initial_metrics": ref_initial,
            "final_metrics": ref_final,
            "delta_final_minus_initial": _numeric_delta(ref_initial, ref_final),
            "metrics_comparison_text": str(refinement_eval / "metrics_comparison.txt")
                if (refinement_eval / "metrics_comparison.txt").is_file() else None,
            "summary": _optional_text(summary_path),
            "evaluation_dir": str(refinement_eval),
        },
        "notes": [
            "Camera alignment: the raw G0 is fixed and the pose changes.",
            "Gaussian refinement: the aligned camera is fixed and G0 changes.",
            "Compare only metrics inside the same stage, not across different definitions.",
            "Absent or uncomputed metrics remain null; no synthetic values are inserted.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _copy_evaluations(alignment_dir: Path, refinement_dir: Path, final_dir: Path) -> None:
    """Keep metric files AND diagnostic renders beside final corrected_g0.ply."""
    mappings = (
        (alignment_dir / "before_after_evaluation", final_dir / "camera_alignment_evaluation"),
        (refinement_dir / "before_after_evaluation", final_dir / "gaussian_refinement_evaluation"),
    )
    for source, destination in mappings:
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
            print(f"[evaluation] copied {source} -> {destination}", flush=True)
        else:
            print(f"[evaluation] missing: {source}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--config",
        default=str(
            STAGE1_DIR
            / "fan_stage1.ini"
        ),
        help="Stage-1 INI file.",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help=(
            "Do not delete round1/final "
            "directories before starting."
        ),
    )

    parser.add_argument(
        "--round1-refinement-iterations",
        type=int,
        default=None,
        help=(
            "Optional override for "
            "Round-1 Gaussian refinement "
            "iterations."
        ),
    )

    args = parser.parse_args()

    config_path = (
        Path(args.config)
        .expanduser()
        .resolve()
    )

    cfg = _read_config(
        config_path
    )

    # ==========================================================
    # INPUT G0
    # ==========================================================

    original_g0 = _cfg_path(
        cfg,
        config_path,
        "Path",
        "initial_gaussian_ply",
    )

    _require_file(
        original_g0,
        "Original G0",
    )

    # ==========================================================
    # PIPELINE DIRECTORIES
    # ==========================================================

    pipeline_root = _cfg_path(
        cfg,
        config_path,
        "Stage1 Pipeline",
        "stage1_pipeline_root",
        STAGE1_DIR,
    )

    round1_dir = _cfg_path(
        cfg,
        config_path,
        "Stage1 Pipeline",
        "stage1_round1_dir",
        pipeline_root / "round1",
    )

    final_dir = _cfg_path(
        cfg,
        config_path,
        "Stage1 Pipeline",
        "stage1_final_dir",
        pipeline_root / "fan_stage1_final",
    )

    # Round 1
    r1_alignment = (
        round1_dir
        / "alignment"
    )

    r1_refinement = (
        round1_dir
        / "refinement"
    )


    # ==========================================================
    # CLEAN OUTPUTS
    # ==========================================================

    if not args.keep_existing:
        for directory in (
            round1_dir,
            final_dir,
        ):
            if directory.exists():
                print(
                    f"[clean] removing "
                    f"{directory}",
                    flush=True,
                )

                shutil.rmtree(
                    directory
                )

    for directory in (
        r1_alignment,
        r1_refinement,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ==========================================================
    # SCRIPTS
    # ==========================================================

    camera_script = (
        STAGE1_DIR
        / "camera_alignment.py"
    )

    refinement_script = (
        STAGE1_DIR
        / "gaussian_refinement.py"
    )

    _require_file(
        camera_script,
        "camera_alignment.py",
    )

    _require_file(
        refinement_script,
        "gaussian_refinement.py",
    )

    # ==========================================================
    # OUTPUT FILES
    # ==========================================================

    r1_camera = (
        r1_alignment
        / "fixed_camera.json"
    )

    r1_corrected = (
        r1_refinement
        / "corrected_g0.ply"
    )

    final_camera = (
        final_dir
        / "fixed_camera.json"
    )

    final_g0 = (
        final_dir
        / "corrected_g0.ply"
    )

    final_metrics = (
        final_dir
        / "metric.json"
    )

    # ==========================================================
    # STEP 1
    #
    # ORIGINAL G0 -> ROUND1 CAMERA ALIGNMENT
    # ==========================================================

    cmd = [
        sys.executable,
        str(camera_script),
        "--config",
        str(config_path),
        "--g0",
        str(original_g0),
        "--output",
        str(r1_alignment),
        "--camera-output",
        str(r1_camera),
        "--camera-prior",
        "none",
    ]

    _run(
        "ROUND 1 / CAMERA ALIGNMENT",
        cmd,
    )

    _require_file(
        r1_camera,
        "Round-1 fixed camera",
    )

    # ==========================================================
    # STEP 2
    #
    # ORIGINAL G0 + ROUND1 CAMERA
    # -> ONE AND ONLY GAUSSIAN REFINEMENT
    # ==========================================================

    cmd = [
        sys.executable,
        str(refinement_script),
        "--config",
        str(config_path),
        "--g0",
        str(original_g0),
        "--camera",
        str(r1_camera),
        "--output",
        str(r1_refinement),
    ]

    if (
        args.round1_refinement_iterations
        is not None
    ):
        cmd += [
            "--iterations",
            str(
                args.round1_refinement_iterations
            ),
        ]

    _run(
        "ROUND 1 / GAUSSIAN REFINEMENT",
        cmd,
    )

    _require_file(
        r1_corrected,
        "Round-1 corrected G0",
    )

    # ==========================================================
    # FINAL PACKAGE: NO SECOND CAMERA ALIGNMENT
    # ==========================================================
    # Ensure final output directory exists
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(r1_camera, final_camera)
    shutil.copy2(r1_corrected, final_g0)
    _require_file(final_camera, "Final fixed camera")
    _require_file(final_g0, "Final corrected G0")

    _copy_evaluations(r1_alignment, r1_refinement, final_dir)

    _write_final_metrics(
        final_metrics,
        config_path,
        original_g0,
        final_camera,
        final_g0,
        r1_alignment,
        r1_refinement,
    )

    # ==========================================================
    # COMPLETE
    # ==========================================================

    print(
        "\n"
        + "=" * 78,
        flush=True,
    )

    print(
        "STAGE-1 PIPELINE COMPLETE",
        flush=True,
    )

    print(
        "=" * 78,
        flush=True,
    )

    print(
        f"Round-1 camera : "
        f"{r1_camera}",
        flush=True,
    )

    print(
        f"Round-1 G0     : "
        f"{r1_corrected}",
        flush=True,
    )

    print(
        f"Final camera   : "
        f"{final_camera}",
        flush=True,
    )

    print(
        f"Final G0       : "
        f"{final_g0}",
        flush=True,
    )

    print(
        f"Final metrics  : "
        f"{final_metrics}",
        flush=True,
    )

    print("NOTE: Exactly one camera alignment and one Gaussian refinement.", flush=True)
    print("NOTE: Final camera is identical to the first aligned camera.", flush=True)
    print(f"Camera evaluations: {final_dir / 'camera_alignment_evaluation'}", flush=True)
    print(f"Refinement evaluations: {final_dir / 'gaussian_refinement_evaluation'}", flush=True)


if __name__ == "__main__":
    main()
