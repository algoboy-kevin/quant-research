"""Fill quality by quote distance, per-coin concentration and the breadth effect.

Conventions (all in bps, `+` = good for the wallet; side-signed):
  edge_bps   s * (pre_mid - px) / pre_mid         how far inside the mid the fill printed
  dfill_bps  |px - pre_mid| / pre_mid             distance of the fill price from the mid just before the fill
  mk_<h>     s * (mid(t+h) - px) / px              markout measured FROM THE FILL PRICE (it already contains the edge)
  net_<h>    mk_<h> - fee_bps                      so never add `edge` on top of it (that double-counts)
Pre-state is the last valid bbo strictly before the fill; a bbo older than `max_age_s` is treated as missing.
"""
from __future__ import annotations

import warnings

import polars as pl

from quant_research.common import stats
from quant_research.common.eventstudy import measure

INF = float("inf")


def _ms(h):
    return int(round(h * 1000))


def fill_markouts(fills: pl.DataFrame, bbo: pl.DataFrame, orders: pl.DataFrame | None = None, *,
                  horizons_s=(1, 5, 60, 300), max_age_s: float = 60.0, t_end: float | None = None,
                  maker_only: bool = True) -> pl.DataFrame:
    tg = fills.filter(~pl.col("crossed")) if maker_only else fills
    m = measure(tg, bbo, t="t", key="coin", value="mid", horizons_s=horizons_s, lookbacks_s=(), react_s=None,
                scale=1.0, t_end=t_end, max_age_s=max_age_s, carry=("bid", "ask"))
    s = pl.when(pl.col("side") == "B").then(1.0).otherwise(-1.0)
    m = m.filter(pl.col("pre").is_not_null()).with_columns(
        (s * (pl.col("pre") - pl.col("px")) / pl.col("pre") * 1e4).alias("edge_bps"),
        ((pl.col("px") - pl.col("pre")).abs() / pl.col("pre") * 1e4).alias("dfill_bps"),
        ((pl.col("pre_ask") - pl.col("pre_bid")) / 2 / pl.col("pre") * 1e4).alias("half_bps"),
        (pl.col("fee") / pl.col("notional") * 1e4).alias("fee_bps"))
    for h in horizons_s:
        k = _ms(h)
        m = m.with_columns((s * (pl.col(f"label_{k}") - pl.col("px")) / pl.col("px") * 1e4).alias(f"mk_{k}"))
        m = m.with_columns((pl.col(f"mk_{k}") - pl.col("fee_bps")).alias(f"net_{k}"))
    if orders is not None and orders.height:
        opens = (orders.filter(pl.col("status") == "open").sort("t").unique("oid", keep="first", maintain_order=True)
                 .select("oid", "coin", pl.col("t").alias("_t_open"), pl.col("px").alias("_px_open")).sort("_t_open"))
        valid = bbo.filter(pl.col("mid").is_not_null()).select("coin", pl.col("t").alias("_tb"), pl.col("mid").alias("_mid_open")).sort("_tb")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
            opens = opens.join_asof(valid, left_on="_t_open", right_on="_tb", by="coin", strategy="backward")
        opens = opens.with_columns(((pl.col("_px_open") - pl.col("_mid_open")).abs() / pl.col("_mid_open") * 1e4).alias("dplace_bps"))
        m = m.join(opens.select("oid", "dplace_bps"), on="oid", how="left")
    return m


def bucket_table(df: pl.DataFrame, dist: str, edges, *, value: str, cluster: str = "coin", alpha: float = 0.10,
                 n_boot: int = 3000, seed: int = 0) -> pl.DataFrame:
    """Notional-weighted mean of `value` (bps) per distance bucket, with a CI that resamples whole coins."""
    rows = []
    d = df.filter(pl.col(value).is_not_null())
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = d.filter((pl.col(dist) >= lo) & (pl.col(dist) < hi))
        if not b.height:
            continue
        g = b.group_by(cluster).agg((pl.col(value) * pl.col("notional")).sum().alias("num"), pl.col("notional").sum().alias("den"))
        ci = stats.cluster_ratio_ci({r[cluster]: (r["num"], r["den"]) for r in g.iter_rows(named=True)},
                                    alpha=alpha, n_boot=n_boot, seed=seed)
        rows.append({"lo": lo, "hi": hi, "fills": b.height, "coins": g.height, "notional": b["notional"].sum(),
                     "wmean_bps": g["num"].sum() / g["den"].sum(), "net_usd": g["num"].sum() / 1e4,
                     "ci_lo": None if ci is None else ci[1], "ci_hi": None if ci is None else ci[2]})
    return pl.from_dicts(rows, infer_schema_length=None) if rows else pl.DataFrame()


def clustered_slice(df: pl.DataFrame, mask: pl.Expr, value: str, *, cluster: str = "coin", alpha=0.10, n_boot=3000, seed=0) -> dict:
    """One row of the same estimate for an arbitrary slice (e.g. far fills ex the top coin)."""
    b = df.filter(mask & pl.col(value).is_not_null())
    g = b.group_by(cluster).agg((pl.col(value) * pl.col("notional")).sum().alias("num"), pl.col("notional").sum().alias("den"))
    ci = stats.cluster_ratio_ci({r[cluster]: (r["num"], r["den"]) for r in g.iter_rows(named=True)},
                                alpha=alpha, n_boot=n_boot, seed=seed)
    return {"fills": b.height, "coins": g.height, "wmean_bps": g["num"].sum() / g["den"].sum() if g.height else None,
            "ci_lo": None if ci is None else ci[1], "ci_hi": None if ci is None else ci[2]}


def net_by_coin(df: pl.DataFrame, value: str) -> pl.DataFrame:
    return (df.filter(pl.col(value).is_not_null())
            .group_by("coin").agg(pl.len().alias("fills"), ((pl.col(value) * pl.col("notional")).sum() / 1e4).alias("net_usd"))
            .sort("net_usd", descending=True))


def concentration(by_coin: pl.DataFrame, ks=(1, 3, 5, 10)) -> list[dict]:
    tot = by_coin["net_usd"].sum()
    return [{"top": k, "net_usd": by_coin["net_usd"].head(k).sum(),
             "share_of_total": by_coin["net_usd"].head(k).sum() / tot if tot else None} for k in ks]


def breadth(by_coin: pl.DataFrame, sizes=(5, 10, 20, 40), *, exclude_top: int = 3, draws: int = 5000, seed: int = 0) -> pl.DataFrame:
    """P(sum of net markout > 0) and spread of the sum when trading N random coins, with and without the top coins."""
    vals = dict(by_coin.select("coin", "net_usd").iter_rows())
    top = list(by_coin["coin"].head(exclude_top))
    rows = [{"universe": "all", **r} for r in stats.subset_resample(vals, sizes, draws=draws, seed=seed)]
    rows += [{"universe": f"ex-top{exclude_top}", **r} for r in stats.subset_resample(vals, sizes, draws=draws, seed=seed, exclude=top)]
    schema = {"universe": pl.Utf8, "n": pl.Int64, "mean": pl.Float64, "p10": pl.Float64, "p90": pl.Float64, "p_pos": pl.Float64}
    return pl.from_dicts(rows, schema=schema) if rows else pl.DataFrame(schema=schema)   # sizes larger than the coin pool are skipped


def time_blocks(df: pl.DataFrame, mask: pl.Expr, value: str, block_s: float = 600) -> pl.DataFrame:
    """Net $ of a slice per time block (is the profit clustered in a few events?). `t` is epoch seconds."""
    return (df.filter(mask & pl.col(value).is_not_null())
            .with_columns(((pl.col("t") // block_s) * block_s).alias("block_start"))
            .group_by("block_start").agg(((pl.col(value) * pl.col("notional")).sum() / 1e4).alias("net_usd"))
            .with_columns(pl.from_epoch("block_start", time_unit="s").dt.strftime("%m-%d %H:%MZ").alias("block_utc"))
            .select("block_utc", "net_usd", "block_start").sort("net_usd", descending=True))
