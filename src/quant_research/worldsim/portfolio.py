"""Independent-world uncertainty and nested portfolio breadth."""
from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import polars as pl

from quant_research.common.report import Summary

from .contracts import Collect
from .engine import Engine
from .fees import Fees
from .fills import FillConfig
from .worlds import World


def statistics(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError("nonempty finite vector required")
    n = len(x)
    p = float(np.mean(x > 0))
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    sd = float(np.std(x, ddof=1)) if n > 1 else None
    return dict(mean=float(np.mean(x)), sd=sd,
                mean_se=sd / np.sqrt(n) if sd is not None else None,
                p_positive=p, p_low=max(0., center - radius), p_high=min(1., center + radius))


def breadth(pnl, seed=0):
    """Rows in dollars; input authoritative integer nanodollars [world, market]."""
    pnl = np.asarray(pnl)
    if pnl.ndim != 2 or min(pnl.shape) < 1 or not np.issubdtype(pnl.dtype, np.integer):
        raise ValueError("pnl must be nonempty integer [world,market]")
    if np.any(np.sum(np.abs(pnl.astype(float)), axis=1) >= 2**63):
        raise OverflowError("portfolio exceeds int64")
    n = pnl.shape[1]
    order = np.random.default_rng(seed).permutation(n)
    sums = np.cumsum(pnl[:, order], axis=1)
    sizes = sorted(set(range(5, min(n, 500) + 1, 5)) | {n})
    return pl.DataFrame([dict(markets=k, **statistics(sums[:, k - 1] / 1e9)) for k in sizes])


def run_portfolio(configs, arms, worlds, seed, out, fill_config=FillConfig(), fees=Fees()):
    if type(worlds) is not int or worlds <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("positive worlds and nonnegative integer seed required")
    if not configs or not arms:
        raise ValueError("at least one group and arm required")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    market_frames, world_rows, breadth_frames, summary_rows = [], [], [], []
    for config in configs:
        for arm_name, arm in arms.items():
            pnls = []
            draws, concentrations, hhis = [], [], []
            fees_usd, rebates_usd, gross_usd = [], [], []
            for w in range(worlds):
                world_seed = int(np.random.SeedSequence([seed, w]).generate_state(1, dtype=np.uint64)[0])
                result = Engine(fill_config, fees, Collect.NONE).run(World(config, world_seed), arm)
                pnl = result.pnl
                pnls.append(pnl)
                absolute = np.abs(pnl.astype(float))
                weights = absolute / absolute.sum() if absolute.sum() else np.zeros_like(absolute)
                concentration, hhi = float(weights.max()), float(np.sum(weights**2))
                draws.append(result.drawdown / 1e9)
                concentrations.append(concentration)
                hhis.append(hhi)
                ledger = result.ledger
                fees_usd.append(sum(map(int, ledger.fees_paid)) / 1e9)
                rebates_usd.append(sum(map(int, ledger.rebates_received)) / 1e9)
                gross_usd.append(sum(map(int, ledger.gross)) / 1e9)
                world_rows.append(dict(group=config.group, arm=arm_name, world=w,
                                       pnl_nano=sum(map(int, pnl)), drawdown_nano=result.drawdown,
                                       fees_nano=sum(map(int, ledger.fees_paid)),
                                       rebates_nano=sum(map(int, ledger.rebates_received)),
                                       gross_nano=sum(map(int, ledger.gross)),
                                       concentration=concentration, hhi=hhi,
                                       fill_count=result.fill_count, rejections=result.rejections))
                market_frames.append(pl.DataFrame(dict(
                    group=[config.group] * config.markets, arm=[arm_name] * config.markets,
                    world=np.full(config.markets, w, dtype=np.int64), market=np.arange(config.markets),
                    condition_id=np.arange(config.markets) // 2 if config.paired else np.arange(config.markets),
                    is_no=np.arange(config.markets) % 2 == 1 if config.paired else np.zeros(config.markets, dtype=bool),
                    pnl_nano=pnl, cash_nano=ledger.cash, inventory=ledger.inventory,
                    gross_nano=ledger.gross, fees_nano=ledger.fees_paid, rebates_nano=ledger.rebates_received)))
            panel = np.stack(pnls)
            b = breadth(panel, seed)
            breadth_frames.append(b.with_columns(pl.lit(config.group).alias("group"), pl.lit(arm_name).alias("arm")))
            totals = np.array([sum(map(int, row)) / 1e9 for row in panel])
            summary_rows.append(dict(group=config.group, arm=arm_name, **statistics(totals),
                                     mean_drawdown=float(np.mean(draws)),
                                     mean_concentration=float(np.mean(concentrations)), mean_hhi=float(np.mean(hhis)),
                                     mean_gross=float(np.mean(gross_usd)),
                                     mean_fees=float(np.mean(fees_usd)),
                                     mean_rebates=float(np.mean(rebates_usd))))
    markets = pl.concat(market_frames).with_columns((pl.col("pnl_nano") / 1e9).alias("pnl_usd"))
    world_table = pl.DataFrame(world_rows).with_columns((pl.col("pnl_nano") / 1e9).alias("pnl_usd"))
    breadth_table = pl.concat(breadth_frames)
    summary = Summary("worldsim", verdict="Generated-market simulation; illustrative defaults, not empirical evidence.", evidence="simulation")
    summary.table("Portfolio results (USD; uncertainty over worlds)", summary_rows, nd=6)
    summary.table("Breadth curve (nested contract subsets)", breadth_table, nd=6)
    summary.coverage = dict(worlds=worlds, seed=seed, markets=[c.markets for c in configs],
                            steps=[c.steps for c in configs], dt_ms=[c.dt_ms for c in configs])
    summary.caveat("Nanodollar columns reconcile exactly. USD columns are display-only.")
    summary.caveat("Default maker_rate > 0 pays maker rebates: before attributing an arm's PnL to skill, compare mean_rebates against mean (zeroing maker/taker rates is the clean ablation).")
    summary.caveat("No funding, margin, impact or calibrated edge. Carry has directional exposure; prediction settlement dominates short-horizon risk.")
    summary.caveat("95% probability intervals are Wilson intervals across independent worlds; breadth points are correlated nested subsets.")
    summary.caveat("Concentration and HHI use absolute per-market terminal PnL, not capital allocation.")
    manifest = dict(seed=seed, worlds=worlds, groups=[asdict(c) for c in configs],
                    arms={name: dict(type=type(a).__name__, parameters=asdict(a)) for name, a in arms.items()},
                    fill_config=asdict(fill_config), fees=asdict(fees),
                    versions=dict(python=platform.python_version(), numpy=np.__version__, polars=pl.__version__))
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for name, frame in (("markets", markets), ("worlds", world_table), ("breadth", breadth_table)):
        frame.write_csv(out / f"{name}.csv")
        frame.write_parquet(out / f"{name}.parquet")
    (out / "summary.md").write_text(summary.render())
    return pl.DataFrame(summary_rows)
