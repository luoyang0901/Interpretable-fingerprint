#!/usr/bin/env python3
"""Run TextCNN(FPtand) over seeds 0-100 for multiple branches.

The script wraps the user's existing Combine.py (or equivalent training script),
launches one isolated run per seed, supports resume, CPU/GPU execution, and
writes machine-readable manifests.

Two seed modes are supported:
- model_only: fixed data split; model initialization seed varies. Required for
  averaging predictions into a valid ensemble on the same held-out samples.
- split_and_model: both split seed and model seed vary. Useful for repeated-split
  uncertainty statistics, but predictions cannot be averaged across seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Job:
    branch: str
    seed: int
    split_seed: int
    model_seed: int
    fd_path: Path
    fa_path: Path
    y_path: Path
    output_dir: Path
    log_path: Path
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="JSON configuration file.")
    p.add_argument("--branches", nargs="*", help="Optional subset of branch names.")
    p.add_argument("--only-seed", type=int, help="Run only one seed; useful for SLURM arrays.")
    p.add_argument("--max-workers", type=int, help="Override runtime.max_workers.")
    p.add_argument("--overwrite", action="store_true", help="Delete and rerun completed seed folders.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def validate_array_file(path: Path, expected_n_bits: int = 1024) -> tuple[int, ...]:
    import numpy as np

    arr = np.load(path, mmap_mode="r")
    if arr.ndim == 2 and arr.shape[1] != expected_n_bits:
        raise ValueError(f"Expected {expected_n_bits} columns in {path}, got {arr.shape}")
    if arr.ndim not in (1, 2):
        raise ValueError(f"Unsupported NPY shape for {path}: {arr.shape}")
    return tuple(int(x) for x in arr.shape)


def prewarm_matplotlib(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(cache_dir)
    subprocess.run(
        [sys.executable, "-c", "import matplotlib.pyplot as plt; print('matplotlib cache ready')"],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_command(job: Job, combine_script: Path, training: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        str(combine_script),
        "--fd-path", str(job.fd_path),
        "--fa-path", str(job.fa_path),
        "--y-path", str(job.y_path),
        "--output-dir", str(job.output_dir),
        "--profile", str(training.get("profile", "strong")),
        "--split-method", str(training.get("split_method", "hspxy")),
        "--split-seed", str(job.split_seed),
        "--model-seeds", str(job.model_seed),
        "--test-size", str(training.get("test_size", 0.20)),
        "--valid-fraction-of-trainval", str(training.get("valid_fraction_of_trainval", 0.125)),
        "--batch-size", str(training.get("batch_size", 32)),
        "--epochs", str(training.get("epochs", 300)),
        "--patience", str(training.get("patience", 40)),
        "--lr", str(training.get("lr", 1e-3)),
        "--weight-decay", str(training.get("weight_decay", 1e-4)),
        "--grad-clip", str(training.get("grad_clip", 5.0)),
        "--max-len", str(training.get("max_len", 200)),
        "--embedding-dim", str(training.get("embedding_dim", 128)),
        "--channels", str(training.get("channels", 128)),
        "--dropout", str(training.get("dropout", 0.35)),
        "--kernel-sizes", str(training.get("kernel_sizes", "3,5,7")),
        "--hidden-dim", str(training.get("hidden_dim", 256)),
        "--loss", str(training.get("loss", "huber")),
        "--device", job.device,
    ]
    return cmd


def run_job(
    job: Job,
    combine_script: Path,
    training: dict[str, Any],
    runtime: dict[str, Any],
    overwrite: bool,
    dry_run: bool,
    config_hash: str,
) -> dict[str, Any]:
    summary_path = job.output_dir / "run_summary.json"
    if summary_path.exists() and not overwrite:
        return {
            "branch": job.branch,
            "seed": job.seed,
            "split_seed": job.split_seed,
            "model_seed": job.model_seed,
            "device": job.device,
            "status": "skipped_complete",
            "return_code": 0,
            "started_utc": "",
            "finished_utc": utc_now(),
            "elapsed_seconds": 0.0,
            "output_dir": str(job.output_dir),
            "log_path": str(job.log_path),
        }

    if overwrite and job.output_dir.exists():
        shutil.rmtree(job.output_dir)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)

    command = build_command(job, combine_script, training)
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return {
            "branch": job.branch,
            "seed": job.seed,
            "split_seed": job.split_seed,
            "model_seed": job.model_seed,
            "device": job.device,
            "status": "dry_run",
            "return_code": 0,
            "started_utc": utc_now(),
            "finished_utc": utc_now(),
            "elapsed_seconds": 0.0,
            "output_dir": str(job.output_dir),
            "log_path": str(job.log_path),
        }

    env = os.environ.copy()
    threads = str(runtime.get("threads_per_job", 1))
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[name] = threads
    env["MPLCONFIGDIR"] = str(Path(runtime["matplotlib_cache_dir"]).resolve())

    # Multi-GPU mode: device strings such as cuda:0 are mapped through CUDA_VISIBLE_DEVICES.
    effective_device = job.device
    if job.device.startswith("cuda:"):
        gpu_index = job.device.split(":", 1)[1]
        env["CUDA_VISIBLE_DEVICES"] = gpu_index
        command[-1] = "cuda"
        effective_device = f"cuda:{gpu_index}"

    metadata = {
        "branch": job.branch,
        "seed": job.seed,
        "split_seed": job.split_seed,
        "model_seed": job.model_seed,
        "device": effective_device,
        "command": command,
        "config_sha256": config_hash,
        "started_utc": utc_now(),
    }
    (job.output_dir / "job_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    start = time.perf_counter()
    started = utc_now()
    with open(job.log_path, "w", encoding="utf-8") as log_handle:
        proc = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - start
    finished = utc_now()

    status = "complete" if proc.returncode == 0 and summary_path.exists() else "failed"
    return {
        "branch": job.branch,
        "seed": job.seed,
        "split_seed": job.split_seed,
        "model_seed": job.model_seed,
        "device": effective_device,
        "status": status,
        "return_code": int(proc.returncode),
        "started_utc": started,
        "finished_utc": finished,
        "elapsed_seconds": round(elapsed, 3),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["branch"], int(r["seed"]))))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    config = load_json(config_path)
    config_hash = sha256_file(config_path)

    combine_script = resolve_path(config_dir, config["combine_script"])
    if not combine_script.exists():
        raise FileNotFoundError(f"Training script not found: {combine_script}")

    output_root = resolve_path(config_dir, config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "config_used.json")

    training = dict(config.get("training", {}))
    runtime = dict(config.get("runtime", {}))
    runtime.setdefault("max_workers", 1)
    runtime.setdefault("threads_per_job", 1)
    runtime.setdefault("devices", ["cpu"])
    runtime["matplotlib_cache_dir"] = str(output_root / ".mpl_cache")
    prewarm_matplotlib(Path(runtime["matplotlib_cache_dir"]))

    max_workers = int(args.max_workers or runtime["max_workers"])
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    devices = [str(x) for x in runtime.get("devices", ["cpu"])]
    if not devices:
        devices = ["cpu"]

    seed_start = int(config.get("seed_start", 0))
    seed_end = int(config.get("seed_end", 100))
    seeds = [args.only_seed] if args.only_seed is not None else list(range(seed_start, seed_end + 1))
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers.")

    seed_mode = str(config.get("seed_mode", "model_only"))
    fixed_split_seed = int(training.get("split_seed", 12))
    if seed_mode not in {"model_only", "split_and_model"}:
        raise ValueError("seed_mode must be 'model_only' or 'split_and_model'.")

    branch_cfg = config["branches"]
    selected_branches = args.branches or list(branch_cfg.keys())
    missing_names = set(selected_branches).difference(branch_cfg)
    if missing_names:
        raise KeyError(f"Unknown branches requested: {sorted(missing_names)}")

    jobs: list[Job] = []
    array_report: dict[str, Any] = {}
    for branch_index, branch in enumerate(selected_branches):
        spec = branch_cfg[branch]
        fd = resolve_path(config_dir, spec["fd"])
        fa = resolve_path(config_dir, spec["fa"])
        y = resolve_path(config_dir, spec["y"])
        for path in [fd, fa, y]:
            if not path.exists():
                raise FileNotFoundError(path)
        fd_shape = validate_array_file(fd)
        fa_shape = validate_array_file(fa)
        y_shape = validate_array_file(y)
        if fd_shape[0] != fa_shape[0] or fd_shape[0] != y_shape[0]:
            raise ValueError(f"Sample count mismatch in {branch}: fd={fd_shape}, fa={fa_shape}, y={y_shape}")
        array_report[branch] = {"fd": fd_shape, "fa": fa_shape, "y": y_shape}

        for position, seed in enumerate(seeds):
            split_seed = fixed_split_seed if seed_mode == "model_only" else int(seed)
            model_seed = int(seed)
            device = devices[(branch_index * len(seeds) + position) % len(devices)]
            out_dir = output_root / branch / f"seed_{seed:03d}"
            log_path = output_root / "logs" / branch / f"seed_{seed:03d}.log"
            jobs.append(
                Job(
                    branch=branch,
                    seed=int(seed),
                    split_seed=split_seed,
                    model_seed=model_seed,
                    fd_path=fd,
                    fa_path=fa,
                    y_path=y,
                    output_dir=out_dir,
                    log_path=log_path,
                    device=device,
                )
            )

    run_header = {
        "campaign_name": config.get("campaign_name", output_root.name),
        "seed_mode": seed_mode,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "selected_branches": selected_branches,
        "max_workers": max_workers,
        "devices": devices,
        "config_sha256": config_hash,
        "combine_script": str(combine_script),
        "combine_script_sha256": sha256_file(combine_script),
        "arrays": array_report,
        "started_utc": utc_now(),
    }
    (output_root / "campaign_metadata.json").write_text(json.dumps(run_header, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    print(f"Running {len(jobs)} jobs with max_workers={max_workers}; seed_mode={seed_mode}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                run_job,
                job,
                combine_script,
                training,
                runtime,
                args.overwrite,
                args.dry_run,
                config_hash,
            ): job
            for job in jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "branch": job.branch,
                    "seed": job.seed,
                    "split_seed": job.split_seed,
                    "model_seed": job.model_seed,
                    "device": job.device,
                    "status": f"exception:{type(exc).__name__}",
                    "return_code": -1,
                    "started_utc": "",
                    "finished_utc": utc_now(),
                    "elapsed_seconds": 0.0,
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                }
            rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}] {row['branch']} seed={row['seed']} -> {row['status']}", flush=True)
            write_manifest(rows, output_root / "run_manifest.csv")

    write_manifest(rows, output_root / "run_manifest.csv")
    failures = [r for r in rows if r["status"] not in {"complete", "skipped_complete", "dry_run"}]
    final = dict(run_header)
    final.update(
        {
            "finished_utc": utc_now(),
            "n_jobs": len(rows),
            "n_complete_or_skipped": len(rows) - len(failures),
            "n_failed": len(failures),
        }
    )
    (output_root / "campaign_complete.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    if failures:
        print(f"WARNING: {len(failures)} jobs failed. Inspect run_manifest.csv and logs/.")
        sys.exit(2)
    print("Campaign completed successfully.")


if __name__ == "__main__":
    main()
