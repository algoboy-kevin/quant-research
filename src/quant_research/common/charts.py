"""Generic simulation charts: PnL-over-time by N, breadth curves, sample paths.

Public toolkit: no empirical anchors or private calibration live here - callers
pass their own anchors to breadth_curves(). All renderers are deterministic,
headless (Agg), and return the output path.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def pnl_over_time_by_n(equity: pl.DataFrame, out: Path, *, quantiles=(0.1, 0.9)) -> Path:
    """Portfolio PnL over time per series x arm, one line per market count N.

    equity: (setup, arm, markets, world, ts_ms, pnl_usd) rows.
    Facets: rows = setups, cols = arms; each N is a mean line with a faint q_lo-q_hi band.
    """
    required = {"setup", "arm", "markets", "world", "ts_ms", "pnl_usd"}
    missing = required - set(equity.columns)
    if missing:
        raise ValueError(f"equity frame missing columns: {sorted(missing)}")
    if equity.is_empty():
        raise ValueError("equity frame is empty")
    lo, hi = quantiles
    setups = sorted(equity["setup"].unique().to_list())
    arms = sorted(equity["arm"].unique().to_list())
    ns = sorted(equity["markets"].unique().to_list())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(len(setups), len(arms),
                             figsize=(5.2 * len(arms), 3.4 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            sub = equity.filter((pl.col("setup") == setup) & (pl.col("arm") == arm))
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
            ax.set_title(f"{setup} / {arm}", fontsize=10)
            ax.set_xlabel("t (s)")
            ax.set_ylabel("portfolio PnL (USD)")
            if i == 0 and j == len(arms) - 1:
                ax.legend(fontsize=7)
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
    required = {"setup", "arm", "markets", "p_positive", "p_low", "p_high"}
    missing = required - set(breadth.columns)
    if missing:
        raise ValueError(f"breadth frame missing columns: {sorted(missing)}")
    if breadth.is_empty():
        raise ValueError("breadth frame is empty")
    setups = sorted(breadth["setup"].unique().to_list())
    arms = sorted(breadth["arm"].unique().to_list())
    fig, axes = plt.subplots(len(setups), len(arms),
                             figsize=(5.0 * len(arms), 3.4 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        for j, arm in enumerate(arms):
            ax = axes[i][j]
            b = (breadth.filter((pl.col("setup") == setup) & (pl.col("arm") == arm))
                 .sort("markets"))
            p = b["p_positive"].to_numpy()
            yerr = np.clip([p - b["p_low"].to_numpy(), b["p_high"].to_numpy() - p], 0, None)
            ax.errorbar(b["markets"].to_numpy(), p, yerr=yerr,
                        marker="o", lw=1.2, capsize=3, label="simulated")
            for key, pts in (anchors or {}).items():
                if setup.startswith(key):
                    xs, ys = zip(*pts)
                    ax.scatter(xs, ys, marker="x", s=60, color="red", label="empirical anchor")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{setup} / {arm}", fontsize=10)
            ax.set_xlabel("N markets")
            ax.set_ylabel("P(sum PnL > 0)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
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
    required = {"setup", "market", "ts_ms", "mid"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples frame missing columns: {sorted(missing)}")
    if samples.is_empty():
        raise ValueError("samples frame is empty")
    setups = sorted(samples["setup"].unique().to_list())
    fig, axes = plt.subplots(len(setups), 1, figsize=(9, 2.6 * len(setups)), squeeze=False)
    for i, setup in enumerate(setups):
        ax = axes[i][0]
        sub = samples.filter(pl.col("setup") == setup)
        for mkt in sorted(sub["market"].unique().to_list()):
            p = sub.filter(pl.col("market") == mkt).sort("ts_ms")
            ax.plot(p["ts_ms"].to_numpy() / 1000.0, p["mid"].to_numpy(), lw=1.0, alpha=0.85)
        ax.set_title(f"{title_prefix} - {setup}", fontsize=10)
        ax.set_xlabel("t (s)")
        ax.set_ylabel("mid" if float(sub["mid"].max()) <= 1.0 else "value")
        if float(sub["mid"].max()) <= 1.0:
            ax.axhline(0.5, color="grey", lw=0.5, ls="--")
            ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
