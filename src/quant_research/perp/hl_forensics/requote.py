"""Requote policy: how a quoter decides to cancel/replace a resting order (hysteresis / request control).

A *replace* is a cancel and an open of one order each, on the same (coin, side), with the same status
timestamp. For each replace we compare the order's distance from its OWN-SIDE TOUCH (best bid for a
bid, best ask for an ask) before and after: ln(d_new / d_old). A relative deadband shows up as a hard
edge with an empty hole around zero; measured from the right anchor the hole is emptier (touch vs mid).
Only replaces whose old distance is >= `min_ticks` ticks are used (below that, tick rounding fills the hole).
"""
from __future__ import annotations

import math
import warnings

import polars as pl

from .ticks import tick_size


def replace_events(orders: pl.DataFrame, bbo: pl.DataFrame, szd: dict) -> pl.DataFrame:
    g = (orders.filter(pl.col("status").is_in(["open", "canceled"]))
         .group_by("t", "coin", "side")
         .agg((pl.col("status") == "open").sum().alias("n_open"), (pl.col("status") == "canceled").sum().alias("n_cancel"),
              pl.col("px").filter(pl.col("status") == "open").first().alias("px_new"),
              pl.col("px").filter(pl.col("status") == "canceled").first().alias("px_old"))
         .filter((pl.col("n_open") == 1) & (pl.col("n_cancel") == 1)).sort("t"))
    valid = bbo.filter(pl.col("mid").is_not_null()).select("coin", pl.col("t").alias("_tb"), "bid", "ask", "mid").sort("_tb")
    g = g.with_columns((pl.col("t") - 0.001).alias("_q")).sort("_q")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        g = g.join_asof(valid, left_on="_q", right_on="_tb", by="coin", strategy="backward").drop("_q", "_tb")
    g = g.filter(pl.col("mid").is_not_null())
    bid = pl.col("side") == "B"
    touch = pl.when(bid).then(pl.col("bid")).otherwise(pl.col("ask"))
    sign = pl.when(bid).then(1.0).otherwise(-1.0)
    g = g.with_columns((sign * (touch - pl.col("px_old"))).alias("d_old_touch"),
                       (sign * (touch - pl.col("px_new"))).alias("d_new_touch"),
                       (pl.col("px_old") - pl.col("mid")).abs().alias("d_old_mid"),
                       (pl.col("px_new") - pl.col("mid")).abs().alias("d_new_mid"))
    ticks = [tick_size(p, szd.get(c, 0)) for p, c in zip(g["px_old"].to_list(), g["coin"].to_list())]
    g = g.with_columns(pl.Series("tick", ticks))
    ln = lambda a, b: pl.when((pl.col(a) > 0) & (pl.col(b) > 0)).then((pl.col(b) / pl.col(a)).log()).otherwise(None)
    return g.with_columns(ln("d_old_touch", "d_new_touch").alias("lnr_touch"), ln("d_old_mid", "d_new_mid").alias("lnr_mid"))


def band_summary(rep: pl.DataFrame, *, min_ticks: float = 8, hole: float = 0.20) -> pl.DataFrame:
    """Per anchor: n, share of replaces inside +-`hole` in log-ratio (the empty hole), and the inner edges
    of the two tails (2nd percentile of the positive / of |negative| log-ratios). A relative deadband
    keeping 0.75 <= d_cur/d_target <= 1.25 gives edges ~0.223 (grow) and ~0.288 (shrink)."""
    rows = []
    for anchor in ("touch", "mid"):
        d_old, lnr = f"d_old_{anchor}", f"lnr_{anchor}"
        x = rep.filter((pl.col(d_old) >= min_ticks * pl.col("tick")) & pl.col(lnr).is_not_null())[lnr]
        pos, neg = x.filter(x > 0), (-x.filter(x < 0))
        rows.append({"anchor": anchor, "n": len(x), "share_inside": (x.abs() < hole).mean() if len(x) else None,
                     "edge_grow": pos.quantile(0.02) if len(pos) else None,
                     "edge_shrink": neg.quantile(0.02) if len(neg) else None})
    return pl.from_dicts(rows, infer_schema_length=None)


def deadband_edges_expected(theta: float = 0.25) -> tuple[float, float]:
    """(edge_grow, edge_shrink) implied by 'keep while (1-theta) <= d_cur/d_target <= (1+theta)'.

    A replace whose old distance was BELOW the band (d_old = (1-theta) d_target) gives ln(d_new/d_old) =
    -ln(1-theta) > 0 ("grow"); one above the band gives -ln(1+theta) < 0 ("shrink", reported as |.|)."""
    return -math.log(1 - theta), math.log(1 + theta)
