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
