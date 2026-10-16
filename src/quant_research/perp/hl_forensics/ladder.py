"""Ladder anatomy and inventory skew from a resting-order snapshot (frame in, frame out).

Input `orders`: coin, side ('B'|'A'), px (float) - e.g. an openOrders / frontendOpenOrders snapshot.
Input `mids`: coin, mid (the market mid at snapshot time).
Per coin: rung gaps (geometric: p_k/p_{k+1}-1, in bps - the spacing is multiplicative), rung-1 distance
from the mid, the ladder's own book mid and its skew vs the market mid. Resting orders are only quotes if
you checked they are post-only limits (a take-profit grid looks identical) - filter before calling this.
"""
from __future__ import annotations

import polars as pl

from quant_research.common import stats


def ladder_table(orders: pl.DataFrame, mids: pl.DataFrame, *, min_rungs: int = 3) -> pl.DataFrame:
    rows = []
    mid_of = dict(mids.select("coin", "mid").iter_rows())
    for coin, g in orders.group_by("coin", maintain_order=True):
        coin = coin[0]
        mid = mid_of.get(coin)
        if not mid:
            continue
        bids = sorted(g.filter(pl.col("side") == "B")["px"].to_list(), reverse=True)
        asks = sorted(g.filter(pl.col("side") == "A")["px"].to_list())
        gaps = [(a / b - 1) * 1e4 for a, b in zip(asks[1:], asks[:-1])] + [(a / b - 1) * 1e4 for a, b in zip(bids[:-1], bids[1:])]
        row = {"coin": coin, "mid": mid, "n_bid": len(bids), "n_ask": len(asks),
               "gap_bps": stats.median(gaps) if len(bids) >= min_rungs - 1 and len(asks) >= min_rungs - 1 and gaps else None,
               "rung1_bid_bps": (mid - bids[0]) / mid * 1e4 if bids else None,
               "rung1_ask_bps": (asks[0] - mid) / mid * 1e4 if asks else None}
        row["own_mid"] = (bids[0] + asks[0]) / 2 if bids and asks else None
        row["skew_bps"] = (row["own_mid"] - mid) / mid * 1e4 if row["own_mid"] else None
        row["inner_over_gap"] = (stats.mean([row["rung1_bid_bps"], row["rung1_ask_bps"]]) / row["gap_bps"]
                                 if row["gap_bps"] and bids and asks else None)
        rows.append(row)
    return pl.from_dicts(rows, infer_schema_length=None)


def end_positions(fills: pl.DataFrame) -> pl.DataFrame:
    """Per-coin position after the LAST fill in the frame (start_position + signed size), from fills alone."""
    s = pl.when(pl.col("side") == "B").then(pl.col("sz")).otherwise(-pl.col("sz"))
    return (fills.sort("t").group_by("coin", maintain_order=True)
            .agg((pl.col("start_position").last() + s.last()).alias("position")))


def skew_vs_inventory(ladders: pl.DataFrame, positions: pl.DataFrame) -> dict:
    """Does the ladder's own mid tilt against signed inventory notional (szi * mid)?

    Uses SIGNED notional - clearinghouseState positionValue is unsigned and silently gives ~0.
    Returns n, Pearson corr, and the OLS slope in bps of skew per $1M of signed inventory.
    """
    j = (ladders.join(positions, on="coin").with_columns((pl.col("position") * pl.col("mid")).alias("inv_usd"))
         .filter(pl.col("skew_bps").is_not_null()))
    n = j.height
    if n < 3:
        return {"n": n, "corr": None, "slope_bps_per_1m": None}
    x, y = j["inv_usd"].to_list(), j["skew_bps"].to_list()
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    corr = sxy / (sxx * syy) ** 0.5 if sxx and syy else None
    return {"n": n, "corr": corr, "slope_bps_per_1m": sxy / sxx * 1e6 if sxx else None}
