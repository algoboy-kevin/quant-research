"""Generic simulation charts: PnL-over-time by N, breadth curves, sample paths.

Public toolkit: no empirical anchors or private calibration live here - callers
pass their own anchors to breadth_curves(). All renderers are deterministic,
headless (Agg), and return the output path. combined_panels() draws the three
views in ONE figure (rows = setups, cols = PnL | breadth | samples) as a
single-arm dashboard.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

EQUITY_COLS = {"setup", "arm", "markets", "world", "ts_ms", "pnl_usd"}
BREADTH_COLS = {"setup", "arm", "markets", "p_positive", "p_low", "p_high"}
SAMPLES_COLS = {"setup", "market", "ts_ms", "mid"}


def _require(frame: pl.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} frame missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ValueError(f"{name} frame is empty")


def _draw_equity(ax, sub: pl.DataFrame, ns, title: str, *,
                 quantiles=(0.1, 0.9), legend: bool = False) -> None:
    """Portfolio PnL over time for ONE (setup, arm): mean line + q_lo-q_hi band per N."""
    lo, hi = quantiles
    cmap = plt.get_cmap("viridis")
    for k, n in enumerate(ns):
        a = sub.filter(pl.col("markets") == n)
        if a.is_empty():
            continue
        band = (a.group_by("ts_ms")
                .agg(mean=pl.col("pnl_usd").mean(),
                     low=pl.col("pnl_usd").quantile(lo),
                     high=pl.col("pnl_usd").quantile(hi))
                .sort("ts_ms"))
        ts = band["ts_ms"].to_numpy() / 1000.0
        color = cmap(k / max(1, len(ns) - 1))
        ax.plot(ts, band["mean"].to_numpy(), color=color, lw=1.2, label=f"N={n}")
        ax.fill_between(ts, band["low"].to_numpy(), band["high"].to_numpy(),
                        color=color, alpha=0.12)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("portfolio PnL (USD)")
    if legend:
        ax.legend(fontsize=7)


def _draw_breadth(ax, b: pl.DataFrame, title: str, *, anchor_pts=(),
                  legend: bool = False) -> None:
    """P(sum PnL > 0) vs N with Wilson bands for ONE (setup, arm)."""
    b = b.sort("markets")
    p = b["p_positive"].to_numpy()
    yerr = np.clip([p - b["p_low"].to_numpy(), b["p_high"].to_numpy() - p], 0, None)
    ax.errorbar(b["markets"].to_numpy(), p, yerr=yerr,
                marker="o", lw=1.2, capsize=3, label="simulated")
    for pts in anchor_pts:
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, marker="x", s=60, color="red", label="empirical anchor")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("N markets")
    ax.set_ylabel("P(sum PnL > 0)")
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(fontsize=7)


def _draw_samples(ax, sub: pl.DataFrame, title: str) -> None:
    """Generated mid/value paths for ONE setup (one sample world)."""
    ax.set_title(title, fontsize=10)
    if sub.is_empty():
        return
    for mkt in sorted(sub["market"].unique().to_list()):
        p = sub.filter(pl.col("market") == mkt).sort("ts_ms")
        ax.plot(p["ts_ms"].to_numpy() / 1000.0, p["mid"].to_numpy(), lw=1.0, alpha=0.85)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("mid" if float(sub["mid"].max()) <= 1.0 else "value")
    if float(sub["mid"].max()) <= 1.0:
        ax.axhline(0.5, color="grey", lw=0.5, ls="--")
        ax.set_ylim(-0.03, 1.03)


def pnl_over_time_by_n(equity: pl.DataFrame, out: Path, *, quantiles=(0.1, 0.9)) -> Path:
    """Portfolio PnL over time per series x arm, one line per market count N.

    equity: (setup, arm, markets, world, ts_ms, pnl_usd) rows.
    Facets: rows = setups, cols = arms; each N is a mean line with a faint q_lo-q_hi band.
    """
    _require(equity, EQUITY_COLS, "equity")
    setups = sorted(equity["setup"].unique().to_list())
    arms = sorted(equity["arm"].unique().to_list())
    ns = sorted(equity["markets"].unique().to_list())
    fig, axes = plt.subplots(len(setups), len(arms),
                             figsize=(5.2 * len(arms), 3.4 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        for j, arm in enumerate(arms):
            sub = equity.filter((pl.col("setup") == setup) & (pl.col("arm") == arm))
            _draw_equity(axes[i][j], sub, ns, f"{setup} / {arm}", quantiles=quantiles,
                         legend=(i == 0 and j == len(arms) - 1))
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def breadth_curves(breadth: pl.DataFrame, out: Path, *,
                   anchors: dict | None = None, quantiles=(0.1, 0.9)) -> Path:
    """P(sum PnL > 0) vs number of markets N, per setup x arm, with Wilson bands.

    anchors: optional {setup_name_prefix: [(N, p), ...]} reference points drawn as
    red crosses on matching panels (e.g. a study's own empirical wallet curve).
    """
    _require(breadth, BREADTH_COLS, "breadth")
    setups = sorted(breadth["setup"].unique().to_list())
    arms = sorted(breadth["arm"].unique().to_list())
    fig, axes = plt.subplots(len(setups), len(arms),
                             figsize=(5.0 * len(arms), 3.4 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        for j, arm in enumerate(arms):
            b = breadth.filter((pl.col("setup") == setup) & (pl.col("arm") == arm))
            pts = [p for key, p in sorted((anchors or {}).items()) if setup.startswith(key)]
            _draw_breadth(axes[i][j], b, f"{setup} / {arm}", anchor_pts=pts,
                          legend=(i == 0 and j == 0))
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def market_samples(samples: pl.DataFrame, out: Path, *,
                   title_prefix: str = "generated markets") -> Path:
    """Sample generated mid/value paths (one world per setup) - face validity.

    samples: (setup, market, ts_ms, mid) rows.
    """
    _require(samples, SAMPLES_COLS, "samples")
    setups = sorted(samples["setup"].unique().to_list())
    fig, axes = plt.subplots(len(setups), 1, figsize=(9, 2.6 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        _draw_samples(axes[i][0], samples.filter(pl.col("setup") == setup),
                      f"{title_prefix} - {setup}")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def combined_panels(equity: pl.DataFrame, breadth: pl.DataFrame,
                    samples: pl.DataFrame, out: Path, *,
                    anchors: dict | None = None,
                    title_prefix: str = "generated markets",
                    quantiles=(0.1, 0.9)) -> Path:
    """One dashboard figure: rows = setups, cols = PnL over time | breadth | samples.

    Single-arm view (raises if equity or breadth carries more than one arm).
    Setups are the union across the three frames; a setup missing from a frame
    renders an empty labelled cell.
    """
    _require(equity, EQUITY_COLS, "equity")
    _require(breadth, BREADTH_COLS, "breadth")
    _require(samples, SAMPLES_COLS, "samples")
    for name, frame in (("equity", equity), ("breadth", breadth)):
        if frame["arm"].n_unique() > 1:
            raise ValueError("combined_panels is a single-arm dashboard; "
                             f"{name} has arms {sorted(frame['arm'].unique().to_list())}")
    setups = sorted(set(equity["setup"].unique().to_list())
                    | set(breadth["setup"].unique().to_list())
                    | set(samples["setup"].unique().to_list()))
    arm = str(equity["arm"][0])
    ns = sorted(equity["markets"].unique().to_list())
    fig, axes = plt.subplots(len(setups), 3, figsize=(16.5, 3.1 * len(setups)),
                             squeeze=False)
    for i, setup in enumerate(setups):
        _draw_equity(axes[i][0], equity.filter(pl.col("setup") == setup), ns,
                     f"{setup} / {arm}", quantiles=quantiles, legend=(i == 0))
        pts = [p for key, p in sorted((anchors or {}).items()) if setup.startswith(key)]
        _draw_breadth(axes[i][1], breadth.filter(pl.col("setup") == setup),
                      f"{setup} / {arm}", anchor_pts=pts, legend=(i == 0))
        _draw_samples(axes[i][2], samples.filter(pl.col("setup") == setup),
                      f"{title_prefix} - {setup}")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
