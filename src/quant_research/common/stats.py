"""Descriptive statistics helpers. Empty input returns None, never a fabricated 0."""
from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Mapping, Sequence


def mean(xs: Sequence[float]):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def median(xs: Sequence[float]):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def pctl(xs: Sequence[float], p: float):
    """Linear-interpolated percentile, p in [0, 100]."""
    v = sorted(x for x in xs if x is not None)
    if not v:
        return None
    k = (len(v) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def wmean(xs: Sequence[float], ws: Sequence[float]):
    pairs = [(x, w) for x, w in zip(xs, ws) if x is not None and w is not None]
    tw = sum(w for _, w in pairs)
    return sum(x * w for x, w in pairs) / tw if tw else None


def bootstrap_ci(xs: Sequence[float], stat: Callable = mean, *, n_boot=2000,
                 alpha=0.05, seed=0):
    """Percentile bootstrap CI over i.i.d. observations. None if fewer than 2."""
    v = [x for x in xs if x is not None]
    if len(v) < 2:
        return None
    rng = random.Random(seed)
    n = len(v)
    draws = sorted(stat([v[rng.randrange(n)] for _ in range(n)]) for _ in range(n_boot))
    return draws[int(alpha / 2 * n_boot)], draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def cluster_bootstrap_ci(groups: Mapping[object, Sequence[float]], stat: Callable = mean, *,
                         n_boot=2000, alpha=0.05, seed=0):
    """Bootstrap resampling whole clusters (games, markets, days...) with replacement.

    Use this instead of `bootstrap_ci` whenever observations inside a cluster share
    the same shock (several markets of one game, ticks of one burst): the plain
    bootstrap treats them as independent and gives intervals that are too narrow.
    """
    keys = [k for k, v in groups.items() if len(v)]
    if len(keys) < 2:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        pooled = []
        for _ in keys:
            pooled.extend(groups[keys[rng.randrange(len(keys))]])
        draws.append(stat(pooled))
    draws.sort()
    return draws[int(alpha / 2 * n_boot)], draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def cluster_ratio_ci(groups: Mapping[object, tuple[float, float]], *, n_boot=2000, alpha=0.05, seed=0):
    """Cluster bootstrap CI of a weighted mean = sum(num)/sum(den) with whole clusters resampled.

    `groups[key] = (sum of weight*value, sum of weight)`. Use for notional-weighted means
    (e.g. markout bps per coin). Returns (point, lo, hi) or None with fewer than 3 clusters.
    """
    keys = [k for k, (_, d) in groups.items() if d]
    if len(keys) < 3:
        return None
    rng = random.Random(seed)
    point = sum(groups[k][0] for k in keys) / sum(groups[k][1] for k in keys)
    draws = []
    for _ in range(n_boot):
        pick = [keys[rng.randrange(len(keys))] for _ in keys]
        den = sum(groups[k][1] for k in pick)
        draws.append(sum(groups[k][0] for k in pick) / den if den else 0.0)
    draws.sort()
    return point, draws[int(alpha / 2 * n_boot)], draws[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]


def subset_resample(values: Mapping[object, float], sizes, *, draws=5000, seed=0, exclude=()):
    """Breadth effect: how does the SUM over a random subset of N units behave as N grows?

    `values[key]` is each unit's contribution (e.g. net markout $ per coin). For every N in
    `sizes` draw `draws` random subsets (without replacement within a draw) and report
    mean / p10 / p90 of the sum and P(sum > 0). `exclude` drops units first (e.g. top-3
    contributors) to show the effect is not carried by a few names.
    """
    rng = random.Random(seed)
    pool = [v for k, v in values.items() if k not in set(exclude)]
    out = []
    for n in sizes:
        if n > len(pool):
            continue
        sums = sorted(sum(rng.sample(pool, n)) for _ in range(draws))
        out.append({"n": n, "mean": sum(sums) / draws, "p10": sums[int(0.10 * draws)], "p90": sums[int(0.90 * draws)],
                    "p_pos": sum(x > 0 for x in sums) / draws})
    return out
