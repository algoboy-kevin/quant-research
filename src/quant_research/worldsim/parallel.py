"""Parallel per-world execution for worldsim sweeps (public toolkit).

A *job* is pure data describing ONE world-run:

    dict(cfg=WorldConfig, arm="churn.width_ticks=2" | <arm instance>, seed=int,
         collect=bool, stride=int (optional, default 1))

`arm` is normally a CLI spec string resolved by `reference_arms.build` inside the
worker. An already-built toolkit arm (a frozen dataclass of primitives defined in an
importable toolkit module) is also accepted, for callers that hold built arms whose
params cannot round-trip through the dotted spec grammar (e.g. 0.25 contains a '.').

`run_worlds(jobs, max_workers)` returns one plain-data result dict per job, in input
order. max_workers=1 runs sequentially in-process (no pool, no pickling); otherwise
a spawn-context ProcessPoolExecutor is used and results are re-ordered to input
order. Nothing defined in a study module is ever pickled.

Result dict: pnl_nano, gross_nano, fees_nano, rebates_nano (list[int], per market),
drawdown_nano, concentration, hhi, fill_count, rejections (scalars), and
equity_rows (list of {ts_ms, equity_nano}, every `stride`-th frame; [] unless collect).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import is_dataclass
import multiprocessing

from .contracts import Collect, WorldConfig

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_REQUIRED = ("cfg", "arm", "seed", "collect")


def _validate(job, i: int) -> None:
    if not isinstance(job, dict):
        raise TypeError(f"job {i}: expected dict payload, got {type(job).__name__}")
    missing = [k for k in _REQUIRED if k not in job]
    if missing:
        raise ValueError(f"job {i}: missing keys {missing}")
    extra = set(job) - set(_REQUIRED) - {"stride"}
    if extra:
        raise ValueError(f"job {i}: unknown keys {sorted(extra)}")
    if not isinstance(job["cfg"], WorldConfig):
        raise TypeError(f"job {i}: cfg must be a WorldConfig, got {type(job['cfg']).__name__}")
    arm = job["arm"]
    if not (isinstance(arm, str) or (is_dataclass(arm) and not isinstance(arm, type))):
        raise TypeError(f"job {i}: arm must be a spec string or arm dataclass instance, "
                        f"got {type(arm).__name__}")
    if type(job["seed"]) is not int or job["seed"] < 0:
        raise ValueError(f"job {i}: seed must be a nonnegative int")
    if type(job["collect"]) is not bool:
        raise TypeError(f"job {i}: collect must be bool")
    stride = job.get("stride", 1)
    if type(stride) is not int or stride <= 0:
        raise ValueError(f"job {i}: stride must be a positive int")


def run_world(job: dict) -> dict:
    """Run one world; module-level so spawn workers can import it."""
    import numpy as np

    from .engine import Engine
    from .fees import Fees
    from .reference_arms import build
    from .worlds import World

    arm = build(job["arm"])[2] if isinstance(job["arm"], str) else job["arm"]
    collect = Collect.ALL if job["collect"] else Collect.NONE
    res = Engine(fees=Fees(), collect=collect).run(World(job["cfg"], job["seed"]), arm)
    pnl = np.asarray(res.pnl, dtype=np.int64)
    ledger = res.ledger
    gross = np.asarray(ledger.gross, dtype=np.int64)
    fees = np.asarray(ledger.fees_paid, dtype=np.int64)
    rebates = np.asarray(ledger.rebates_received, dtype=np.int64)
    absolute = np.abs(pnl.astype(float))
    weights = absolute / absolute.sum() if absolute.sum() else np.zeros_like(absolute)
    equity_rows = []
    if job["collect"]:
        equity_rows = [dict(ts_ms=int(r["ts_ms"]), equity_nano=int(r["equity_nano"]))
                       for r in res.steps[::job.get("stride", 1)]]
    return dict(
        pnl_nano=pnl.tolist(), gross_nano=gross.tolist(), fees_nano=fees.tolist(),
        rebates_nano=rebates.tolist(), drawdown_nano=int(res.drawdown),
        concentration=float(weights.max()), hhi=float(np.sum(weights**2)),
        fill_count=int(res.fill_count), rejections=int(res.rejections),
        equity_rows=equity_rows)


def run_worlds(jobs, max_workers: int = 1, on_done=None) -> list[dict]:
    """on_done(index, result), if given, is called as each job finishes (completion
    order under a pool, input order sequentially) - for progress output only."""
    jobs = list(jobs)
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive int")
    for i, job in enumerate(jobs):
        _validate(job, i)
    if max_workers == 1 or len(jobs) <= 1:
        out = []
        for i, j in enumerate(jobs):
            out.append(run_world(j))
            if on_done:
                on_done(i, out[-1])
        return out
    # Children inherit the environment: pin BLAS threads before they import numpy.
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(max_workers, len(jobs)),
                                 mp_context=ctx) as pool:
            futures = {pool.submit(run_world, j): i for i, j in enumerate(jobs)}
            out = [None] * len(jobs)
            for f in as_completed(futures):
                i = futures[f]
                out[i] = f.result()          # slot by input index: order is deterministic
                if on_done:
                    on_done(i, out[i])
            return out
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
