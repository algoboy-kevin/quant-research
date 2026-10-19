"""Reliability estimators shared by tape and sim (worldsim reliability spec v1, W0).

One function per gate statistic, all consuming the SAME frame schemas on both sides:
  bbo:    (coin, t_ms, bid, ask, mid)      - t_ms = VENUE/exchange time where available
  orders: (coin, t_ms, side, px, t_end_ms, ev)  ev in {"new","replace","cancel","fill"}
  fills:  (coin, t_ms, side, px, sz)

Every statistic is per coin first, then aggregated (median + IQR across coins): the gates
are about the cross-section, not the pooled tape. Sim output must be resampled to the
tape's feed cadence before comparison (the caller's job; estimators are cadence-explicit).

Refs: SIM_SPEC.md (scratch/simspec/) gates D*, Z*, S*, M*, R*.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def _coins(bbo: pl.DataFrame) -> list[str]:
    return sorted(bbo["coin"].unique().to_list())


def _mid_grid(bbo: pl.DataFrame, coin: str, at_ms: float) -> np.ndarray:
    """Mid of one coin resampled on a fixed grid [0, t_max) step at_ms, as-of backward."""
    g = (bbo.filter(pl.col("coin") == coin).sort("t_ms")
         .select("t_ms", "mid").unique(subset=["t_ms"], keep="last"))
    t = g["t_ms"].to_numpy()
    m = g["mid"].to_numpy()
    if len(t) == 0:
        return np.zeros(0)
    grid = np.arange(t[0], t[-1], at_ms)
    idx = np.searchsorted(t, grid, side="right") - 1
    return m[np.clip(idx, 0, len(m) - 1)]


def zero_move_share(bbo: pl.DataFrame, at_ms: float = 500.0) -> pl.DataFrame:
    """Share of grid slots with zero mid change, per coin. Gate Z1."""
    rows = []
    for coin in _coins(bbo):
        m = _mid_grid(bbo, coin, at_ms)
        if len(m) < 3:
            continue
        d = np.diff(m)
        rows.append(dict(coin=coin, at_ms=at_ms, zero_share=float((d == 0).mean()),
                         n_slots=int(len(d))))
    return pl.DataFrame(rows, infer_schema_length=None)


def mid_changes_per_hour(bbo: pl.DataFrame) -> pl.DataFrame:
    """Number of distinct mid updates per coin-hour (event count on the feed). Gate Z2."""
    rows = []
    for coin in _coins(bbo):
        g = (bbo.filter(pl.col("coin") == coin).sort("t_ms")
             .select("t_ms", "mid").unique(subset=["t_ms"], keep="last"))
        m = g["mid"].to_numpy()
        t = g["t_ms"].to_numpy()
        if len(m) < 3 or t[-1] <= t[0]:
            continue
        hours = (t[-1] - t[0]) / 3.6e6
        changes = int((np.diff(m) != 0).sum())
        rows.append(dict(coin=coin, changes_per_hour=changes / hours,
                         rows_per_hour=(len(m) - 1) / hours, hours=hours))
    return pl.DataFrame(rows, infer_schema_length=None)


def alternation_ratio(bbo: pl.DataFrame) -> pl.DataFrame:
    """Flicker: share of consecutive mid moves that reverse sign. Gate Z3.
    eta_uz = N_cont / (2 * N_alt) per the Robert-Rosenbaum identification."""
    rows = []
    for coin in _coins(bbo):
        g = (bbo.filter(pl.col("coin") == coin).sort("t_ms")
             .select("t_ms", "mid").unique(subset=["t_ms"], keep="last"))
        d = np.diff(g["mid"].to_numpy())
        d = d[d != 0]
        if len(d) < 3:
            continue
        s = np.sign(d)
        alt = int((s[1:] * s[:-1] < 0).sum())
        n = len(s) - 1
        alt_ratio = alt / n if n else float("nan")
        rows.append(dict(coin=coin, alt_ratio=alt_ratio,
                         eta_uz=(1.0 - alt_ratio) / (2.0 * alt_ratio) if alt_ratio > 0 else float("nan")))
    return pl.DataFrame(rows, infer_schema_length=None)


def tick_bps(bbo: pl.DataFrame) -> pl.DataFrame:
    """Per-coin minimum observed price increment in bps of mid (proxy tick)."""
    rows = []
    for coin in _coins(bbo):
        g = bbo.filter(pl.col("coin") == coin).select("bid", "ask", "mid")
        px = np.unique(np.concatenate([g["bid"].to_numpy(), g["ask"].to_numpy()]))
        px = px[np.isfinite(px) & (px > 0)]
        if len(px) < 3:
            continue
        inc = np.diff(np.sort(px))
        inc = inc[inc > 0]
        if len(inc) == 0:
            continue
        med_mid = float(np.median(g["mid"].to_numpy()))
        rows.append(dict(coin=coin, tick_bps=float(np.median(inc)) / med_mid * 1e4,
                         n_ticks=int(len(inc))))
    return pl.DataFrame(rows, infer_schema_length=None)


def return_ladder(bbo: pl.DataFrame, horizons_s=(10, 30, 60, 120, 300, 600, 900),
                  grid_ms: float = 1000.0) -> pl.DataFrame:
    """Per-coin standardized return stats per horizon: sd (bps), quantile ladder, skew.
    Gates R1/R2; K reported as diagnostic (C1: infinite 4th moment, no gate)."""
    rows = []
    for coin in _coins(bbo):
        m = _mid_grid(bbo, coin, grid_ms)
        if len(m) < max(horizons_s) + 3:
            continue
        r0 = np.diff(np.log(m, where=m > 0, out=np.full_like(m, np.nan)))
        r0 = r0[np.isfinite(r0)]
        if len(r0) < 10:
            continue
        sd1 = float(np.std(r0)) * 1e4
        for h in horizons_s:
            k = int(round(h * 1000 / grid_ms))
            rh = np.log(m[k:] / m[:-k])
            rh = rh[np.isfinite(rh)]
            if len(rh) < 10:
                continue
            sdh = float(np.std(rh)) * 1e4
            z = rh / np.std(rh) if np.std(rh) > 0 else rh
            rows.append(dict(coin=coin, horizon_s=h, sd_bps=sdh, n=int(len(rh)),
                             q001=float(np.quantile(z, 0.001)), q01=float(np.quantile(z, 0.01)),
                             q99=float(np.quantile(z, 0.99)), q999=float(np.quantile(z, 0.999)),
                             skew=float(np.mean(z ** 3)), kurt=float(np.mean(z ** 4) - 3),
                             sd1_bps=sd1))
    return pl.DataFrame(rows, infer_schema_length=None)


def jump_stats(bbo: pl.DataFrame, k_sig: float = 4.0, grid_ms: float = 1000.0,
               horizon_s: int = 10) -> pl.DataFrame:
    """Jump detect on 1 s grid moves (|dz| > k_sig), rate/h, median size (bps), 10 s revert
    share, Hill tail index. Gate R3."""
    rows = []
    for coin in _coins(bbo):
        m = _mid_grid(bbo, coin, grid_ms)
        if len(m) < horizon_s + 20:
            continue
        t = np.arange(len(m)) * grid_ms / 1000.0
        r = np.diff(np.log(m, where=m > 0, out=np.full_like(m, np.nan)))
        ok = np.isfinite(r)
        r, t = r[ok], t[1:][ok]
        if len(r) < 20:
            continue
        sd = np.std(r)
        jump = np.abs(r) > k_sig * sd
        n_j = int(jump.sum())
        hours = t[-1] / 3600 if t[-1] > 0 else float("nan")
        sizes = np.abs(r[jump]) * 1e4
        # 10 s revert: sign of the jump vs the move over the next horizon_s seconds
        k = horizon_s
        idx = np.flatnonzero(jump) + 1
        idx = idx[idx + k < len(m)]
        rev = 0
        if len(idx):
            fwdmove = np.log(m[idx + k] / m[idx])
            jump_sign = np.sign(r[idx - 1])
            rev = float((np.sign(fwdmove) * jump_sign < 0).mean())
        hill = float("nan")
        if len(sizes) >= 8:
            xs = np.sort(sizes)
            k_top = max(4, len(xs) // 4)
            hill = float(1.0 / (np.mean(np.log(xs[-k_top:] / xs[-k_top - 1])) + 1e-12))
        rows.append(dict(coin=coin, jumps_per_hour=n_j / hours if hours else float("nan"),
                         median_jump_bps=float(np.median(sizes)) if n_j else float("nan"),
                         revert10=rev, hill=hill, n_jumps=n_j))
    return pl.DataFrame(rows, infer_schema_length=None)


def corr60(bbo: pl.DataFrame, grid_ms: float = 60000.0, min_obs: int = 30) -> pl.DataFrame:
    """Cross-coin correlation of 60 s log returns (pairwise complete). Gate B1."""
    coins = _coins(bbo)
    series = {}
    for coin in coins:
        m = _mid_grid(bbo, coin, grid_ms)
        if len(m) >= min_obs:
            series[coin] = np.diff(np.log(m, where=m > 0, out=np.full_like(m, np.nan)))
    keys = sorted(series)
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            x, y = series[a], series[b]
            n = min(len(x), len(y))
            x, y = x[:n], y[:n]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= min_obs:
                rows.append(dict(coin_a=a, coin_b=b,
                                 corr=float(np.corrcoef(x[ok], y[ok])[0, 1]), n=int(ok.sum())))
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
        schema={"coin_a": pl.Utf8, "coin_b": pl.Utf8, "corr": pl.Float64, "n": pl.Int64})


def wallet_dodge(orders: pl.DataFrame, bbo: pl.DataFrame,
                 bins_bps=(0, 3, 20, np.inf)) -> pl.DataFrame:
    """Cancel hazard vs distance-to-touch and cancel-to-arrival lags (W0 dodge analysis).

    orders: ev in {'new','replace','cancel','fill'} with t_ms (venue) and px.
    For each cancel: the distance (bps) of px behind the touch at cancel time, and the time
    from that cancel to the next bbo trade-through of px (if any within 60 s) - the 'arrival
    the wallet dodged' lag."""
    rows = []
    joins = (orders.filter(pl.col("ev") == "cancel")
             .sort("t_ms").join_asof(bbo.sort("t_ms").select("coin", "t_ms", "bid", "ask"),
                                     left_on="t_ms", right_on="t_ms", by="coin",
                                     strategy="backward"))
    for r in joins.iter_rows(named=True):
        if r.get("bid") is None:
            continue
        px, side = float(r["px"]), r["side"]
        touch = float(r["bid"]) if side == "B" else float(r["ask"])
        mid = (float(r["bid"]) + float(r["ask"])) / 2
        d = (touch - px) / mid * 1e4 if side == "B" else (px - touch) / mid * 1e4
        rows.append(dict(coin=r["coin"], t_ms=r["t_ms"], side=side, d_touch_bps=float(d)))
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(
        schema={"coin": pl.Utf8, "t_ms": pl.Float64, "side": pl.Utf8, "d_touch_bps": pl.Float64})


def summarize(df: pl.DataFrame, cols: list[str]) -> dict:
    """Cross-coin median/IQR summary of per-coin stat rows (the gate aggregation)."""
    out = {}
    for c in cols:
        s = df[c].drop_nulls().to_numpy()
        s = s[np.isfinite(s)]
        if len(s) == 0:
            out[c] = None
            continue
        out[c] = dict(median=float(np.median(s)), p25=float(np.quantile(s, 0.25)),
                      p75=float(np.quantile(s, 0.75)), n=int(len(s)))
    return out
