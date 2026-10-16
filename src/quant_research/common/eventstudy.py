"""Event-study primitives on DataFrames (vectorised with join_asof, no per-event Python loops).

Conventions
  * `targets`: one row per (event x instrument) with a time column and an instrument key.
  * `quotes`: one row per quote update with time, key and a value column (None = invalid
    state: one-sided / crossed book). Valid-only lookups skip invalid rows; the pre-state
    is STRICTLY before the event (an update at the same instant is not yet visible).
  * `move_<ms> = (value(t+h) - pre) * scale`. Mapping that to "adverse for the holder" is
    the caller's job. Horizons past `t_end` are censored (null), never extrapolated.
  * Labels are future values and must never feed back into a trigger.
"""
from __future__ import annotations

import warnings

import polars as pl

from .stats import bootstrap_ci, cluster_bootstrap_ci, mean, median, pctl


def coalesce(events: list[dict], gap_s: float, *, key=lambda e: e["t_mono"], group=lambda e: None):
    """Keep the first event of each burst (same group, <= gap_s after the KEPT event).

    Dropped events increment kept['n_coalesced_dropped']. Returns (kept, n_dropped).
    """
    kept, last, dropped = [], {}, 0
    for e in sorted(events, key=key):
        g = group(e)
        prev = last.get(g)
        if prev is not None and key(e) - key(prev) <= gap_s:
            prev["n_coalesced_dropped"] = prev.get("n_coalesced_dropped", 0) + 1
            dropped += 1
            continue
        e.setdefault("n_coalesced_dropped", 0)
        kept.append(e)
        last[g] = e
    return kept, dropped


def _ms(s: float) -> int:
    return int(round(s * 1000))


def _asof(df, right, *, left_on, key, strict=False):
    """Backward as-of join on `_tq`; never leaves `_tq` behind."""
    with warnings.catch_warnings():
        # polars cannot verify sortedness with `by`; both sides ARE sorted globally by time
        # (callers sort), which is what a per-group as-of join needs.
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        out = df.join_asof(right, left_on=left_on, right_on="_tq", by=key, strategy="backward",
                           allow_exact_matches=not strict, coalesce=True)
    return out.drop("_tq", strict=False)


def measure(targets: pl.DataFrame, quotes: pl.DataFrame, *, t: str = "t_mono", key: str = "asset_id",
            value: str = "mid", t_q: str | None = None, horizons_s=(0.25, 0.5, 1, 2, 5),
            lookbacks_s=(0.5, 1.0), react_s: float = 0.25, scale: float = 100.0,
            t_end: float | None = None) -> pl.DataFrame:
    """Append pre-state, forward moves and trailing moves to `targets`.

    Added columns: pre, pre_age_ms, pre_latest_valid, move_<ms>, label_<ms>,
    prior_<ms> (= pre - value(t-lb): the move already made), n_updates_react.
    `t_end` (default: last quote time) censors horizons that run past the capture.
    """
    t_q = t_q or t
    q_all = quotes.select(key, pl.col(t_q).alias("_tq"), pl.col(value).alias("_v")).sort("_tq")
    q_valid = q_all.filter(pl.col("_v").is_not_null())
    cnt = q_all.with_columns((pl.int_range(pl.len()).over(key) + 1).alias("_v")).select(key, "_tq", "_v")
    if t_end is None:
        t_end = q_all["_tq"].max() if q_all.height else float("-inf")

    df = targets.with_row_index("_i").sort(t)

    # state STRICTLY before the event; keep the observation time to report staleness
    df = df.with_columns(pl.col(t).alias("_t0"))
    df = _asof(df, q_valid.with_columns(pl.col("_tq").alias("_pre_t")).rename({"_v": "pre"}),
               left_on=t, key=key, strict=True)
    df = _asof(df, q_all.with_columns(pl.col("_tq").alias("_lat_t")).rename({"_v": "_lat"}),
               left_on=t, key=key, strict=True)
    df = df.with_columns(
        ((pl.col(t) - pl.col("_pre_t")) * 1000).round(3).alias("pre_age_ms"),
        (pl.col("_lat_t").is_not_null() & pl.col("_lat").is_not_null()).alias("pre_latest_valid"),
    ).drop("_pre_t", "_lat_t", "_lat", "_t0")

    for h in horizons_s:
        k = _ms(h)
        df = df.with_columns((pl.col(t) + h).alias("_th"))
        df = _asof(df, q_valid.rename({"_v": f"label_{k}"}), left_on="_th", key=key)
        censored = (pl.col("_th") > t_end) | pl.col("pre").is_null()
        df = df.with_columns(pl.when(censored).then(None).otherwise(pl.col(f"label_{k}")).alias(f"label_{k}"))
        df = df.with_columns(((pl.col(f"label_{k}") - pl.col("pre")) * scale).round(6).alias(f"move_{k}")).drop("_th")

    for lb in lookbacks_s:
        k = _ms(lb)
        df = df.with_columns((pl.col(t) - lb).alias("_tb")).sort("_tb")
        df = _asof(df, q_valid.rename({"_v": "_past"}), left_on="_tb", key=key).sort(t)
        df = df.with_columns(((pl.col("pre") - pl.col("_past")) * scale).round(6).alias(f"prior_{k}")).drop("_tb", "_past")

    # number of quote updates in (t, t+react]
    df = df.with_columns((pl.col(t) + react_s).alias("_tr")).sort("_tr")
    df = _asof(df, cnt.rename({"_v": "_c_hi"}), left_on="_tr", key=key).sort(t)
    df = _asof(df, cnt.rename({"_v": "_c_lo"}), left_on=t, key=key)
    df = df.with_columns(pl.when(pl.col("_tr") <= t_end)
                         .then(pl.col("_c_hi").fill_null(0) - pl.col("_c_lo").fill_null(0))
                         .otherwise(None).alias("n_updates_react")).drop("_tr", "_c_hi", "_c_lo")
    return df.sort("_i").drop("_i")


def placebo(quotes: pl.DataFrame, event_times, n: int, *, t: str = "t_mono", key: str = "asset_id",
            value: str = "mid", t_lo: float, t_hi: float, excl_s: float = 10.0, seed: int = 0) -> pl.DataFrame:
    """Seeded random (key, t) observation times that are >= excl_s from every event.

    Candidates are the series' own valid observations inside [t_lo, t_hi], so a placebo
    always has a real pre-state. Sampled without replacement.
    """
    cand = (quotes.filter(pl.col(value).is_not_null() & pl.col(t).is_between(t_lo, t_hi))
            .select(key, t).sort(t))
    ev = pl.DataFrame({"_e": sorted(event_times)}, schema={"_e": pl.Float64})
    if ev.height and cand.height:
        cand = (cand.join_asof(ev, left_on=t, right_on="_e", strategy="nearest")
                .filter((pl.col(t) - pl.col("_e")).abs() >= excl_s).drop("_e"))
    return cand.sample(min(n, cand.height), seed=seed).sort(t) if cand.height else cand


def summarize(df: pl.DataFrame, cols, *, by=None, cluster: str | None = None, n_boot: int = 2000,
              seed: int = 0) -> pl.DataFrame:
    """Long table of n / mean / median / p10 / p90 / 95% CI per column (and per `by` group).

    With `cluster` the CI resamples whole clusters (games, markets, days...). Prefer it
    whenever rows share a shock - a plain bootstrap over such rows is too narrow.
    """
    by = [by] if isinstance(by, str) else list(by or [])
    parts = df.partition_by(by, as_dict=True, maintain_order=True) if by else {(): df}
    out = []
    for gk, g in parts.items():
        gk = gk if isinstance(gk, tuple) else (gk,)
        for c in cols:
            sub = g.filter(pl.col(c).is_not_null())
            vals = sub[c].to_list()
            if cluster and sub.height:
                groups = {(k[0] if isinstance(k, tuple) else k): v[c].to_list()
                          for k, v in sub.partition_by(cluster, as_dict=True).items()}
                ci = cluster_bootstrap_ci(groups, n_boot=n_boot, seed=seed)
            else:
                ci = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
            out.append({**dict(zip(by, gk)), "col": c, "n": len(vals), "mean": mean(vals), "median": median(vals),
                        "p10": pctl(vals, 10), "p90": pctl(vals, 90),
                        "ci_lo": None if ci is None else ci[0], "ci_hi": None if ci is None else ci[1]})
    return pl.from_dicts(out, infer_schema_length=None) if out else pl.DataFrame()
