#!/usr/bin/env python3
"""Run fixed-model-seed or nested split/model-seed TextCNN campaigns.

Configuration modes:
- fixed_model_seeds: one target-blind fixed split seed and many model seeds;
- nested: Cartesian product of split_seeds and model_seeds, allowing partition
  variability and training stochasticity to be summarized separately.

Every run calls Combine.py with one split seed and one model seed and records a
machine-readable manifest. Runs are resumable.
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
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def expand_seed_spec(value: Any, default: list[int]) -> list[int]:
    if value is None:
        return default
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    if isinstance(value, dict):
        start = int(value.get("start", 0))
        end = int(value.get("end", start))
        step = int(value.get("step", 1))
        return list(range(start, end + 1, step))
    raise TypeError(f"Unsupported seed specification: {value!r}")


@dataclass(frozen=True)
class Job:
    branch: str
    split_seed: int
    model_seed: int
    fd: Path
    fa: Path
    y: Path
    group: Path | None
    metadata: Path | None
    output_dir: Path
    log_path: Path
    device: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--branches", nargs="*")
    p.add_argument("--max-workers", type=int)
    p.add_argument("--only-split-seed", type=int)
    p.add_argument("--only-model-seed", type=int)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_command(job: Job, combine: Path, training: dict[str, Any]) -> list[str]:
    cmd = [
        sys.executable,
        str(combine),
        "--fd-path", str(job.fd),
        "--fa-path", str(job.fa),
        "--y-path", str(job.y),
        "--output-dir", str(job.output_dir),
        "--profile", str(training.get("profile", "strong")),
        "--split-method", str(training.get("split_method", "structure_ks")),
        "--split-seed", str(job.split_seed),
        "--model-seeds", str(job.model_seed),
        "--encoding-mode", str(training.get("encoding_mode", "role_aware")),
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
        "--high-pce-threshold", str(training.get("high_pce_threshold", 16.0)),
        "--device", job.device,
    ]
    if job.group is not None:
        cmd += ["--group-path", str(job.group)]
    if job.metadata is not None:
        cmd += ["--metadata-csv", str(job.metadata)]
    precomputed = training.get("precomputed_split")
    if precomputed:
        cmd += ["--precomputed-split", str(Path(precomputed).resolve())]
    return cmd


def run_job(
    job: Job,
    combine: Path,
    training: dict[str, Any],
    runtime: dict[str, Any],
    config_hash: str,
    overwrite: bool,
    dry_run: bool,
) -> dict[str, Any]:
    summary = job.output_dir / "run_summary.json"
    if summary.exists() and not overwrite:
        return {
            "branch": job.branch,
            "split_seed": job.split_seed,
            "model_seed": job.model_seed,
            "status": "skipped_complete",
            "return_code": 0,
            "output_dir": str(job.output_dir),
            "log_path": str(job.log_path),
            "elapsed_seconds": 0.0,
            "finished_utc": utc_now(),
        }
    if overwrite and job.output_dir.exists():
        shutil.rmtree(job.output_dir)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(job, combine, training)
    metadata = {
        "branch": job.branch,
        "split_seed": job.split_seed,
        "model_seed": job.model_seed,
        "command": command,
        "config_sha256": config_hash,
        "started_utc": utc_now(),
    }
    (job.output_dir / "job_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return {
            "branch": job.branch,
            "split_seed": job.split_seed,
            "model_seed": job.model_seed,
            "status": "dry_run",
            "return_code": 0,
            "output_dir": str(job.output_dir),
            "log_path": str(job.log_path),
            "elapsed_seconds": 0.0,
            "finished_utc": utc_now(),
        }
    env = os.environ.copy()
    threads = str(runtime.get("threads_per_job", 1))
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        env[key] = threads
    effective_device = job.device
    if job.device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = job.device.split(":", 1)[1]
        command[-1] = "cuda"
    start = time.perf_counter()
    with job.log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - start
    status = "complete" if proc.returncode == 0 and summary.exists() else "failed"
    return {
        "branch": job.branch,
        "split_seed": job.split_seed,
        "model_seed": job.model_seed,
        "device": effective_device,
        "status": status,
        "return_code": int(proc.returncode),
        "output_dir": str(job.output_dir),
        "log_path": str(job.log_path),
        "elapsed_seconds": round(elapsed, 3),
        "finished_utc": utc_now(),
    }


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: (x.get("branch", ""), int(x.get("split_seed", 0)), int(x.get("model_seed", 0)))))


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    base = config_path.parent
    cfg = load_json(config_path)
    config_hash = sha256_file(config_path)
    combine = resolve(base, cfg.get("combine_script", "Combine.py"))
    if combine is None or not combine.exists():
        raise FileNotFoundError(combine)
    output_root = resolve(base, cfg["output_root"])
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "config_used.json")

    mode = str(cfg.get("campaign_mode", "fixed_model_seeds"))
    if mode not in {"fixed_model_seeds", "nested"}:
        raise ValueError("campaign_mode must be fixed_model_seeds or nested")
    model_seeds = expand_seed_spec(cfg.get("model_seeds"), list(range(101)))
    if mode == "fixed_model_seeds":
        split_seeds = [int(cfg.get("fixed_split_seed", cfg.get("training", {}).get("split_seed", 12)))]
    else:
        split_seeds = expand_seed_spec(cfg.get("split_seeds"), list(range(10)))
    if args.only_split_seed is not None:
        split_seeds = [args.only_split_seed]
    if args.only_model_seed is not None:
        model_seeds = [args.only_model_seed]

    runtime = dict(cfg.get("runtime", {}))
    max_workers = int(args.max_workers or runtime.get("max_workers", 1))
    devices = [str(x) for x in runtime.get("devices", ["cpu"])] or ["cpu"]
    selected_branches = args.branches or list(cfg["branches"].keys())
    jobs: list[Job] = []
    for bpos, branch in enumerate(selected_branches):
        spec = cfg["branches"][branch]
        fd = resolve(base, spec["fd"])
        fa = resolve(base, spec["fa"])
        y = resolve(base, spec["y"])
        group = resolve(base, spec.get("group"))
        metadata = resolve(base, spec.get("metadata_csv"))
        for p in [fd, fa, y, group, metadata]:
            if p is not None and not p.exists():
                raise FileNotFoundError(p)
        for spos, split_seed in enumerate(split_seeds):
            for mpos, model_seed in enumerate(model_seeds):
                run_name = f"run_s{split_seed:03d}_m{model_seed:03d}"
                device = devices[(bpos * len(split_seeds) * len(model_seeds) + spos * len(model_seeds) + mpos) % len(devices)]
                jobs.append(Job(
                    branch=branch,
                    split_seed=int(split_seed),
                    model_seed=int(model_seed),
                    fd=fd, fa=fa, y=y, group=group, metadata=metadata,
                    output_dir=output_root / branch / run_name,
                    log_path=output_root / "logs" / branch / f"{run_name}.log",
                    device=device,
                ))

    header = {
        "campaign_name": cfg.get("campaign_name", output_root.name),
        "campaign_mode": mode,
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "branches": selected_branches,
        "combine_script": str(combine),
        "combine_script_sha256": sha256_file(combine),
        "config_sha256": config_hash,
        "n_jobs": len(jobs),
        "started_utc": utc_now(),
    }
    (output_root / "campaign_metadata.json").write_text(json.dumps(header, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    print(f"Running {len(jobs)} jobs; mode={mode}; max_workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_job, job, combine, dict(cfg.get("training", {})), runtime,
                config_hash, args.overwrite, args.dry_run,
            ): job
            for job in jobs
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "branch": job.branch,
                    "split_seed": job.split_seed,
                    "model_seed": job.model_seed,
                    "status": f"exception:{type(exc).__name__}",
                    "return_code": -1,
                    "output_dir": str(job.output_dir),
                    "log_path": str(job.log_path),
                    "elapsed_seconds": 0.0,
                    "finished_utc": utc_now(),
                }
            rows.append(row)
            write_manifest(rows, output_root / "run_manifest.csv")
            print(f"[{len(rows)}/{len(jobs)}] {row['branch']} s={row['split_seed']} m={row['model_seed']} -> {row['status']}")
    failures = [r for r in rows if r["status"] not in {"complete", "skipped_complete", "dry_run"}]
    complete = dict(header)
    complete.update({"finished_utc": utc_now(), "n_failed": len(failures)})
    (output_root / "campaign_complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    if failures:
        print(f"WARNING: {len(failures)} jobs failed. See run_manifest.csv and logs/.")
        sys.exit(2)
    print("Campaign completed successfully.")


if __name__ == "__main__":
    main()
