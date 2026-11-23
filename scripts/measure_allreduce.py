"""Measure the all-reduce bandwidth between the GPUs of this machine (NCCL), for the predictor.

The analytic predictor charges the DDP gradient sync with an interconnect bandwidth β taken from
a table (PCIe = 16 GB/s). This measures it: each process all-reduces fp32 tensors of growing size
and reports, per size, the time and two bandwidths in the usual nccl-tests convention:

* ``algbw`` = bytes / time (what one rank sees);
* ``busbw`` = algbw · 2(n−1)/n, the bandwidth of the links in a ring all-reduce, comparable across
  GPU counts (for n = 2 it equals algbw).

    torchrun --nproc_per_node=2 scripts/measure_allreduce.py > allreduce.log

Run under torchrun (one process per GPU). Rank 0 prints the table and, after it, the same rows
as CSV between ``# csv`` markers (the caller keeps stdout; nothing is written to a path).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import time

SIZES_MB = (1, 4, 16, 64, 128, 256, 512)


def bus_bandwidth(algbw: float, world_size: int) -> float:
    """Ring all-reduce link bandwidth from the algorithm bandwidth (nccl-tests convention)."""
    return algbw * 2 * (world_size - 1) / world_size if world_size > 1 else algbw


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    args = p.parse_args(argv)

    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank % torch.cuda.device_count())
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    rows = []
    for mb in SIZES_MB:
        x = torch.ones(mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
        for _ in range(args.warmup):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / args.iters
        algbw = x.numel() * 4 / dt / 1e9
        rows.append({"size_mb": mb, "time_ms": round(dt * 1e3, 3), "algbw_gbs": round(algbw, 3),
                     "busbw_gbs": round(bus_bandwidth(algbw, world), 3), "world_size": world})
    if dist.get_rank() == 0:
        print(f"{'MB':>6} {'ms':>9} {'algbw GB/s':>11} {'busbw GB/s':>11}")
        for r in rows:
            print(f"{r['size_mb']:>6} {r['time_ms']:>9} {r['algbw_gbs']:>11} {r['busbw_gbs']:>11}")
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
        print("# csv\n" + buf.getvalue().strip() + "\n# csv", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
