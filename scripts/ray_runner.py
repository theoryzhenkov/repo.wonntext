#!/usr/bin/env python
"""Dispatch training jobs across the sophon 8xH100 box via Ray.

Each job gets 1 GPU (Ray sets CUDA_VISIBLE_DEVICES); jobs queue when all 8 are
busy. Logs go to /nvme/theo/logs/<name>.log.

Usage:
  python scripts/ray_runner.py jobs.json          # run a batch
  python scripts/ray_runner.py --gpus 1 jobs.json

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


@ray.remote(num_gpus=1)
def run_job(name: str, cmd: str) -> dict:
    Path(LOGDIR).mkdir(parents=True, exist_ok=True)
    logpath = Path(LOGDIR) / f"{name}.log"
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    wrapped = (
        f"source /nvme/theo/env.sh && cd {REPO} && "
        f"PATH=/nvme/theo/venv/bin:$PATH {cmd}"
    )
    t0 = time.time()
    with logpath.open("w") as f:
        f.write(f"# job={name} gpu={gpu}\n# cmd={cmd}\n\n")
        f.flush()
        rc = subprocess.run(
            ["bash", "-lc", wrapped], stdout=f, stderr=subprocess.STDOUT
        ).returncode
    return {
        "name": name,
        "rc": rc,
        "gpu": gpu,
        "secs": round(time.time() - t0, 1),
        "log": str(logpath),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", help="JSON file: [{name, cmd}, ...]")
    ap.add_argument("--gpus", type=float, default=1.0, help="GPUs per job")
    args = ap.parse_args()

    with open(args.jobs) as f:
        jobs = json.load(f)

    ray.init(address="auto")
    runner = run_job.options(num_gpus=args.gpus)
    pending = [runner.remote(j["name"], j["cmd"]) for j in jobs]
    print(f"dispatched {len(pending)} jobs across the pool\n", flush=True)

    while pending:
        done, pending = ray.wait(pending, num_returns=1)
        r = ray.get(done[0])
        status = "OK" if r["rc"] == 0 else f"FAILED(rc={r['rc']})"
        print(
            f"[{status}] {r['name']} gpu={r['gpu']} {r['secs']}s -> {r['log']}",
            flush=True,
        )

    print("\nall jobs finished.")


if __name__ == "__main__":
    main()
