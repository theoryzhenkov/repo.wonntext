#!/usr/bin/env python
"""Dispatch training jobs across the sophon 8xH100 box via Ray.

Two scheduling modes:

* Default (Ray-scheduled): each job requests ``num_gpus`` (default 1) and Ray
  assigns ``CUDA_VISIBLE_DEVICES`` from the GPUs it manages. Use when the box is
  ours. Jobs queue when all GPUs are busy. Logs go to
  ``/nvme/theo/logs/<name>.log``.

* Pinned (``--gpu_ids 0,1``): jobs run with ``num_gpus=0`` and the driver
  assigns ``CUDA_VISIBLE_DEVICES`` from the given pool, with concurrency capped
  at the pool size (or ``--max_concurrent``). Use on a shared box to run only on
  free GPUs without contending with other users.

Usage:
  python scripts/ray_runner.py jobs.json                       # Ray-scheduled
  python scripts/ray_runner.py --gpu_ids 0,1 jobs.json         # pinned to 2 GPUs
  python scripts/ray_runner.py --gpus 1 jobs.json               # GPUs per job

jobs.json: [{"name": "wonn_t16", "cmd": "PYTHONPATH=src python -m wonntext.train ..."}, ...]
Commands run with cwd=/nvme/theo/wonntext and /nvme/theo/env.sh sourced, with the
venv on PATH.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import ray

REPO = "/nvme/theo/wonntext"
LOGDIR = "/nvme/theo/logs"


def _run(name: str, cmd: str, gpu_id: str | None) -> dict:
    """Execute one job; ``gpu_id`` pins CUDA_VISIBLE_DEVICES (None = let Ray decide)."""
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    logpath = Path(LOGDIR) / f"{name}.log"
    env = dict(os.environ)
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    reported_gpu = gpu_id if gpu_id is not None else env.get("CUDA_VISIBLE_DEVICES", "?")
    wrapped = (
        f"source /nvme/theo/env.sh && cd {REPO} && "
        f"PATH=/nvme/theo/venv/bin:$PATH {cmd}"
    )
    t0 = time.time()
    with logpath.open("w") as f:
        f.write(f"# job={name} gpu={reported_gpu}\n# cmd={cmd}\n\n")
        f.flush()
        rc = subprocess.run(
            ["bash", "-lc", wrapped], stdout=f, stderr=subprocess.STDOUT, env=env
        ).returncode
    return {
        "name": name,
        "rc": rc,
        "gpu": reported_gpu,
        "secs": round(time.time() - t0, 1),
        "log": str(logpath),
    }


# num_gpus=1 by default; overridden to 0 in pinned mode (driver assigns the GPU).
run_job = ray.remote(num_gpus=1)(_run)


def _dispatch_pinned(jobs: list[dict], gpu_pool: list[int], cap: int) -> None:
    runner = run_job.options(num_gpus=0)
    free = [str(g) for g in gpu_pool]
    pending: list[tuple[object, str]] = []  # (ref, gpu_id)
    todo = list(jobs)
    print(f"pinned to gpus {gpu_pool} (concurrency {cap}); {len(todo)} jobs queued\n",
          flush=True)
    while todo or pending:
        while todo and len(pending) < cap and free:
            gpu = free.pop(0)
            j = todo.pop(0)
            pending.append((runner.remote(j["name"], j["cmd"], gpu), gpu))
        done, _ = ray.wait([r for r, _ in pending], num_returns=1)
        survivors = []
        for ref, gpu in pending:
            if ref in done:
                free.append(gpu)
                r = ray.get(ref)
                status = "OK" if r["rc"] == 0 else f"FAILED(rc={r['rc']})"
                print(f"[{status}] {r['name']} gpu={r['gpu']} {r['secs']}s "
                      f"-> {r['log']}", flush=True)
            else:
                survivors.append((ref, gpu))
        pending = survivors
    print("\nall jobs finished.")


def _dispatch_ray(jobs: list[dict], gpus_per_job: float) -> None:
    runner = run_job.options(num_gpus=gpus_per_job)
    pending = [runner.remote(j["name"], j["cmd"], None) for j in jobs]
    print(f"dispatched {len(pending)} jobs across the pool\n", flush=True)
    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        status = "OK" if r["rc"] == 0 else f"FAILED(rc={r['rc']})"
        print(f"[{status}] {r['name']} gpu={r['gpu']} {r['secs']}s -> {r['log']}",
              flush=True)
    print("\nall jobs finished.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help="JSON file: [{name, cmd}, ...]")
    ap.add_argument("--gpus", type=float, default=1.0, help="GPUs per job (Ray-scheduled)")
    ap.add_argument("--gpu_ids", default=None,
                    help="comma-sep GPU ids to pin to (shared-box mode); concurrency=pool")
    ap.add_argument("--max_concurrent", type=int, default=None,
                    help="cap in-flight jobs (default: len(gpu_ids))")
    args = ap.parse_args()

    with Path(args.jobs).open() as f:
        jobs = json.load(f)

    ray.init(address="auto")
    if args.gpu_ids:
        pool = [int(x) for x in args.gpu_ids.split(",")]
        cap = args.max_concurrent or len(pool)
        _dispatch_pinned(jobs, pool, cap)
    else:
        _dispatch_ray(jobs, args.gpus)


if __name__ == "__main__":
    main()
