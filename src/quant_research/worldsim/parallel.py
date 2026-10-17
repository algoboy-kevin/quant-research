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


def _summary(pnl, gross, fees, rebates, drawdown, fill_count, rejections, equity_rows) -> dict:
    import numpy as np

    pnl = np.asarray(pnl, dtype=np.int64)
    gross = np.asarray(gross, dtype=np.int64)
    fees = np.asarray(fees, dtype=np.int64)
    rebates = np.asarray(rebates, dtype=np.int64)
    absolute = np.abs(pnl.astype(float))
    weights = absolute / absolute.sum() if absolute.sum() else np.zeros_like(absolute)
    return dict(
        pnl_nano=pnl.tolist(), gross_nano=gross.tolist(), fees_nano=fees.tolist(),
        rebates_nano=rebates.tolist(), drawdown_nano=int(drawdown),
        concentration=float(weights.max()), hhi=float(np.sum(weights**2)),
        fill_count=int(fill_count), rejections=int(rejections),
        equity_rows=equity_rows)


def run_world(job: dict) -> dict:
    """Run one world unbatched (reference path); module-level so spawn workers can import it."""
    from .engine import Engine
    from .fees import Fees
    from .reference_arms import build
    from .worlds import World

    arm = build(job["arm"])[2] if isinstance(job["arm"], str) else job["arm"]
    collect = Collect.ALL if job["collect"] else Collect.NONE
    res = Engine(fees=Fees(), collect=collect).run(World(job["cfg"], job["seed"]), arm)
    ledger = res.ledger
    equity_rows = []
    if job["collect"]:
        equity_rows = [dict(ts_ms=int(r["ts_ms"]), equity_nano=int(r["equity_nano"]))
                       for r in res.steps[::job.get("stride", 1)]]
    return _summary(res.pnl, ledger.gross, ledger.fees_paid, ledger.rebates_received,
                    res.drawdown, res.fill_count, res.rejections, equity_rows)


def run_batch_task(task: dict) -> dict:
    """Run one batch task: K worlds of one config x A arms over shared frames.

    task: cfg, seeds [K], arms [A] (spec string or arm instance), strides
    {(arm_index, world_index): [stride, ...]} for worlds whose equity series is wanted.
    Returns {(arm_index, world_index): summary dict with equity_rows={stride: rows}}.
    """
    from .batch import run_batch
    from .fees import Fees
    from .reference_arms import build

    arms = [build(a)[2] if isinstance(a, str) else a for a in task["arms"]]
    K = len(task["seeds"])
    record = {}
    for (ai, k), _ in task["strides"].items():
        record.setdefault(ai, [False] * K)[k] = True
    results = run_batch(task["cfg"], task["seeds"], arms, fees=Fees(), record=record)
    out = {}
    for ai, per_world in enumerate(results):
        for k, r in enumerate(per_world):
            rows = {}
            for stride in task["strides"].get((ai, k), ()):
                rows[stride] = [dict(ts_ms=int(t), equity_nano=int(e))
                                for t, e in zip(r.ts_ms[::stride], r.equity_nano[::stride])]
            out[(ai, k)] = _summary(r.pnl, r.gross, r.fees_paid, r.rebates_received,
                                    r.drawdown, r.fill_count, r.rejections, rows)
    return out


def _execute(task: dict):
    return run_world(task["job"]) if task["kind"] == "solo" else run_batch_task(task)


# Relative cost model (ms per frame, measured on a loaded 8-core box; only the ratios
# matter) used to choose batch sizes that balance tasks across workers. It never affects
# results. A task of K worlds x A arms pays a fixed per-frame numpy-call overhead plus a
# per-world term (RNG draws, shared by the arms) and a per-world-per-arm term.
_F0, _F1, _G, _P = 0.3, 0.5, 0.058, 0.009
_K_CANDIDATES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _task_cost(cfg, narms: int, k: int) -> float:
    scale = max(1.0, cfg.markets / 10)
    return cfg.steps * (_F0 + _F1 * narms + k * (_G + _P * narms * scale))


def _makespan(costs: list[float], workers: int) -> float:
    loads = [0.0] * workers
    for c in sorted(costs, reverse=True):
        i = loads.index(min(loads))
        loads[i] += c
    return max(loads)


def _sizes(nseeds: int, strategy, cap: int) -> list[int]:
    """Split nseeds into near-equal chunks: ("k", k) -> ceil(nseeds / k) chunks of at most
    k; ("parts", p) -> min(p, nseeds) chunks. Chunks never exceed `cap` worlds."""
    kind, v = strategy
    parts = -(-nseeds // v) if kind == "k" else min(v, nseeds)
    parts = max(parts, -(-nseeds // cap))
    q, r = divmod(nseeds, parts)
    return [q + 1] * r + [q] * (parts - r)


_STARTUP = 3300.0     # model units: spawning a worker + importing numpy/polars (~1.5 s)


def _choose(bundles, workers: int, batch_size):
    """bundles: [(cfg, narms, nseeds)] -> (chunking strategy, workers to use).

    Picks the (workers, chunking) pair with the smallest modeled makespan including
    process start-up, so small sweeps stay in-process instead of paying for a pool."""
    if batch_size is not None:
        return ("k", batch_size), workers
    candidates = [("k", k) for k in _K_CANDIDATES] + [("parts", p) for p in range(2, 2 * workers + 1)]
    best = None
    for w in range(1, workers + 1):
        strategies = [("k", 64)] if w == 1 else candidates
        for strategy in strategies:
            costs = []
            for cfg, narms, nseeds in bundles:
                costs += [_task_cost(cfg, narms, sz)
                          for sz in _sizes(nseeds, strategy, _cap(cfg))]
            span = _makespan(costs, w) + (_STARTUP if w > 1 and len(costs) > 1 else 0.0)
            if best is None or span < best[0] - 1e-9:
                best = (span, strategy, w)
    return best[1], best[2]


def _cap(cfg) -> int:
    return max(1, min(64, 8192 // cfg.markets))


def _plan(jobs, workers: int, batch_size):
    """Group jobs into tasks. Returns (tasks, owners): owners[t] maps a task's output key
    to the job indices it serves."""
    from .batch import is_batchable
    from .reference_arms import build

    built: dict = {}

    def resolve(arm):
        if isinstance(arm, str):
            if arm not in built:
                built[arm] = build(arm)[2]        # unknown names raise KeyError here
            return built[arm]
        return arm

    groups: dict = {}        # cfg -> {arm_key: {seed: [job index]}}
    solo = []
    for i, job in enumerate(jobs):
        inst = resolve(job["arm"])
        key = job["arm"]
        try:
            hash((job["cfg"], key))
            ok = is_batchable(inst, job["cfg"])
        except TypeError:
            ok = False
        if not ok:
            solo.append(i)
            continue
        groups.setdefault(job["cfg"], {}).setdefault(key, {}).setdefault(job["seed"], []).append(i)

    bundles = []              # (cfg, [arm_key...], [seed...])
    for cfg, arms in groups.items():
        by_seeds: dict = {}
        for key, seeds in arms.items():
            by_seeds.setdefault(tuple(sorted(seeds)), []).append(key)
        for seeds, keys in by_seeds.items():
            bundles.append((cfg, keys, list(seeds)))
    strategy, workers = _choose([(cfg, len(keys), len(seeds)) for cfg, keys, seeds in bundles],
                                workers, batch_size)

    tasks, owners = [], []
    for i in solo:
        tasks.append(dict(kind="solo", job=jobs[i], cost=_task_cost(jobs[i]["cfg"], 1, 1)))
        owners.append({None: [i]})
    for cfg, keys, seeds in bundles:
        lo = 0
        for size in _sizes(len(seeds), strategy, _cap(cfg)):
            chunk = seeds[lo:lo + size]
            lo += size
            strides, own = {}, {}
            for ai, key in enumerate(keys):
                for sj, seed in enumerate(chunk):
                    idxs = groups[cfg][key][seed]
                    own[(ai, sj)] = idxs
                    want = sorted({jobs[i].get("stride", 1) for i in idxs if jobs[i]["collect"]})
                    if want:
                        strides[(ai, sj)] = want
            tasks.append(dict(kind="batch", cfg=cfg, seeds=chunk, arms=list(keys),
                              strides=strides, cost=_task_cost(cfg, len(keys), len(chunk))))
            owners.append(own)
    return tasks, owners, workers


def _distribute(jobs, task_out, own, results, on_done):
    done = []
    if None in own:                                  # solo task
        (i,) = own[None]
        results[i] = task_out
        done.append(i)
    else:
        for key, idxs in own.items():
            base = task_out[key]
            for i in idxs:
                res = dict(base)
                for f in ("pnl_nano", "gross_nano", "fees_nano", "rebates_nano"):
                    res[f] = list(base[f])
                res["equity_rows"] = (list(base["equity_rows"].get(jobs[i].get("stride", 1), []))
                                      if jobs[i]["collect"] else [])
                results[i] = res
                done.append(i)
    if on_done:
        for i in sorted(done):
            on_done(i, results[i])


def run_worlds(jobs, max_workers: int = 1, on_done=None, batch_size=None) -> list[dict]:
    """Run jobs; results are in input order and EXACTLY equal to `run_world(job)`.

    Jobs of the same WorldConfig are batched (K worlds stacked on the market axis, arms
    sharing the generated frames; see batch.py) when the arm is a toolkit reference arm;
    other arms run one world at a time. batch_size caps K (default: auto - the whole
    group for max_workers=1, a load-balanced size otherwise). Batching never changes
    results.

    on_done(index, result), if given, is called as each job finishes, for progress output
    only: a batch reports its jobs together (ascending index) when the batch completes.
    """
    jobs = list(jobs)
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be a positive int")
    if batch_size is not None and (type(batch_size) is not int or batch_size < 1):
        raise ValueError("batch_size must be a positive int")
    for i, job in enumerate(jobs):
        _validate(job, i)
    results = [None] * len(jobs)
    if not jobs:
        return results
    tasks, owners, max_workers = _plan(jobs, max_workers, batch_size)
    if max_workers == 1 or len(tasks) <= 1:
        for t, own in zip(tasks, owners):
            _distribute(jobs, _execute(t), own, results, on_done)
        return results
    # Children inherit the environment: pin BLAS threads before they import numpy.
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    try:
        ctx = multiprocessing.get_context("spawn")
        order = sorted(range(len(tasks)), key=lambda t: -tasks[t]["cost"])   # big first
        with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks)),
                                 mp_context=ctx) as pool:
            futures = {pool.submit(_execute, tasks[t]): t for t in order}
            for f in as_completed(futures):
                t = futures[f]
                _distribute(jobs, f.result(), owners[t], results, on_done)
        return results
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
