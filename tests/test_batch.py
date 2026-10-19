"""Batched execution must reproduce unbatched execution EXACTLY (ints equal, floats equal)."""
import dataclasses
import unittest

import numpy as np

from quant_research.worldsim.arms import ChurnArm
from quant_research.worldsim.batch import run_batch
from quant_research.worldsim.contracts import Collect, FlowEvents, WorldConfig
from quant_research.worldsim.engine import Engine
from quant_research.worldsim.fees import Fees
from quant_research.worldsim.fills import FillConfig
from quant_research.worldsim.parallel import run_world, run_worlds
from quant_research.worldsim.reference_arms import Carry, Churn, FarFill, GeometricGrid, build
from quant_research.worldsim.worlds import World, WorldBatch

PERP = dict(group="perp", perp_sigma=0.4, clusters=3, adverse_selection=0.6)
PRED = dict(group="prediction", common_weight=0.15, cluster_weight=0.35, clusters=3)
CONFIGS = {
    "perp": WorldConfig(markets=7, steps=24, dt_ms=300_000, **PERP),
    "perp_depth2": WorldConfig(markets=3, steps=14, dt_ms=1000, depth=2, **PERP),
    "pred_fast": WorldConfig(markets=6, steps=16, regime="fast", **PRED),
    "pred_paired": WorldConfig(markets=6, steps=16, regime="slow", paired=True, **PRED),
    "pred_tiny": WorldConfig(markets=1, steps=4, **PRED),
    # non-default display-layer knobs (spread / OU dislocation / wicks / toxicity)
    "perp_display": WorldConfig(markets=5, steps=20, dt_ms=60_000, spread_mean_ticks=3.0,
                                spread_dispersion=0.5, dislocation_sd_ticks=20.0,
                                dislocation_halflife_ms=90_000.0, wick_prob=0.5,
                                wick_scale_ticks=3.0, wick_max_ticks=7, informed_strength=1.0,
                                wick_toxicity_mult=0.0, **PERP),
    # flow-fix: busy multi-print bars, n>1 markets, both groups, paired mirror
    "perp_flow_busy": WorldConfig(markets=8, steps=20, dt_ms=300_000, print_rate_per_hour=600.0,
                                  clip_median=3.0, sweep_rate_per_hour=300.0, sweep_trade_frac=0.5,
                                  **PERP),
    "perp_flow_5s": WorldConfig(markets=6, steps=60, dt_ms=5_000, print_rate_per_hour=2_000.0,
                                sweep_rate_per_hour=1_000.0, sweep_trade_frac=0.5, **PERP),
    "pred_flow_busy": WorldConfig(markets=6, steps=16, regime="fast", print_rate_per_hour=40_000.0,
                                  **PRED),
    "pred_paired_flow": WorldConfig(markets=6, steps=16, regime="slow", paired=True,
                                    print_rate_per_hour=40_000.0, **PRED),
    "pred_paired_display": WorldConfig(markets=6, steps=16, regime="fast", paired=True,
                                       tick=0.01, spread_mean_ticks=8.0, dislocation_sd_ticks=4.0,
                                       wick_prob=0.4, wick_max_ticks=30, informed_strength=0.7,
                                       **PRED),
}
ARMS = [Churn(), Carry(), FarFill(), GeometricGrid(levels=3),
        Churn(levels=2, width_ticks=2, skew_ticks=1.5), ChurnArm(width_ticks=2)]
SEEDS = [3, 17, 123456789012345, 8, 9]


class FrameTests(unittest.TestCase):
    def test_world_batch_frames_equal_single_frames(self):
        for name, cfg in CONFIGS.items():
            singles = [list(World(cfg, s)) for s in SEEDS]
            for t, bf in enumerate(WorldBatch(cfg, SEEDS)):
                for k in range(len(SEEDS)):
                    sf, n = singles[k][t], cfg.markets
                    for f in dataclasses.fields(sf):
                        a, b = getattr(sf, f.name), getattr(bf, f.name)
                        if isinstance(a, np.ndarray):
                            b = b[k * n:(k + 1) * n]
                            self.assertEqual((name, t, k, f.name, a.dtype, a.tobytes()),
                                             (name, t, k, f.name, b.dtype, b.tobytes()))
                        elif isinstance(a, FlowEvents) or isinstance(b, FlowEvents):
                            self.assertTrue(a is not None and b is not None, (name, t, k))
                            E = a.size.shape[0]
                            for g in dataclasses.fields(a):
                                x, y = getattr(a, g.name), getattr(b, g.name)
                                if isinstance(x, np.ndarray):
                                    y = y[k * n:(k + 1) * n] if x.ndim == 1 else y[:E, k * n:(k + 1) * n]
                                    self.assertEqual((name, t, k, g.name, x.dtype, x.tobytes()),
                                                     (name, t, k, g.name, y.dtype, y.tobytes()))
                                else:
                                    self.assertEqual(x, y)
                            # rows beyond the solo world's own maximum are padding
                            self.assertFalse(getattr(bf.flow, "size")[E:, k * n:(k + 1) * n].any())
                        else:
                            self.assertEqual(a, b)


class EngineEquality(unittest.TestCase):
    def _check(self, cfg, fill_config, fees):
        seeds = SEEDS[:4]
        batch = run_batch(cfg, seeds, ARMS, fill_config=fill_config, fees=fees,
                          record={0: [True, False, True, False], 3: [False, True, False, True]})
        for ai, arm in enumerate(ARMS):
            for k, seed in enumerate(seeds):
                ref = Engine(fill_config, fees, Collect.ALL).run(World(cfg, seed), arm)
                got = batch[ai][k]
                lg = ref.ledger
                self.assertEqual(ref.pnl.tolist(), got.pnl.tolist())
                for name, a, b in (("gross", lg.gross, got.gross),
                                   ("fees", lg.fees_paid, got.fees_paid),
                                   ("rebates", lg.rebates_received, got.rebates_received),
                                   ("cash", lg.cash, got.cash),
                                   ("inventory", lg.inventory, got.inventory)):
                    self.assertEqual(a.tolist(), b.tolist(), (ai, k, name))
                    self.assertEqual(a.dtype, b.dtype)
                self.assertEqual(
                    (ref.drawdown, ref.rejections, ref.fill_count),
                    (got.drawdown, got.rejections, got.fill_count), (ai, k))
                if got.ts_ms is not None:
                    self.assertEqual([r["ts_ms"] for r in ref.steps], got.ts_ms)
                    self.assertEqual([r["equity_nano"] for r in ref.steps], got.equity_nano)

    def test_batched_equals_single_default_engine(self):
        for name, cfg in CONFIGS.items():
            with self.subTest(name):
                self._check(cfg, FillConfig(), Fees())

    def test_batched_equals_single_with_latency_queue_and_fees(self):
        fc = FillConfig(post_ms=250, cancel_ms=500, queue_fraction=1.5)
        for name in ("perp", "pred_fast", "pred_paired"):
            with self.subTest(name):
                self._check(CONFIGS[name], fc, Fees(maker_rate=0.0003, taker_rate=0.001))

    def test_batched_equals_single_maker_pays(self):
        # engine.py and batch.py book the maker charge separately; the Hyperliquid
        # direction (makers PAY) must not diverge between the two paths
        for name in ("perp", "pred_fast", "pred_paired"):
            with self.subTest(name):
                self._check(CONFIGS[name], FillConfig(),
                            Fees(maker_rate=0.00015, taker_rate=0.00045, maker_pays=True))

    def test_batch_is_independent_of_batch_composition(self):
        cfg = CONFIGS["perp"]
        alone = run_batch(cfg, [SEEDS[2]], [ARMS[1]])[0][0]
        among = run_batch(cfg, SEEDS, ARMS)[1][2]
        self.assertEqual(alone.pnl.tolist(), among.pnl.tolist())
        self.assertEqual((alone.drawdown, alone.fill_count), (among.drawdown, among.fill_count))

    def test_rejections_attributed_per_world(self):
        cfg = WorldConfig(markets=5, steps=60, dt_ms=100, **PERP)
        fc = FillConfig(post_ms=400)        # quotes go stale while in flight -> crossings
        batch = run_batch(cfg, SEEDS, [Churn()], fill_config=fc)[0]
        singles = [Engine(fc).run(World(cfg, s), Churn()).rejections for s in SEEDS]
        self.assertEqual(singles, [r.rejections for r in batch])


@dataclasses.dataclass(frozen=True)
class _OddArm:                      # not a toolkit reference arm -> must run unbatched
    levels: int = 1

    def decide(self, s):
        return Churn().decide(s)


class RunWorldsBatching(unittest.TestCase):
    def _jobs(self):
        jobs = []
        for name, cfg in CONFIGS.items():
            for spec in ("churn", "carry", "far_fill"):
                for s in SEEDS[:3]:
                    jobs.append(dict(cfg=cfg, arm=spec, seed=s, collect=(s % 2 == 1),
                                     stride=3))
        return jobs

    def _extras(self):
        c = CONFIGS["perp"]
        base = dict(cfg=c, arm="churn", seed=3, collect=False)
        return [dict(cfg=c, arm=_OddArm(), seed=1, collect=False),          # solo path
                dict(cfg=c, arm=build("churn")[2], seed=1, collect=True, stride=2),
                dict(base), dict(base),                                     # exact duplicates
                dict(base, collect=True, stride=5)]                         # same run, other view

    def test_every_batch_size_equals_run_world(self):
        jobs = self._jobs() + self._extras()[1:]
        expected = [run_world(j) for j in jobs]
        for batch_size in (None, 1, 2, 3, 64):
            with self.subTest(batch_size=batch_size):
                self.assertEqual(run_worlds(jobs, 1, batch_size=batch_size), expected)

    def test_unbatchable_arm_runs_solo_and_matches(self):
        jobs = self._extras()
        self.assertEqual(run_worlds(jobs, 1), [run_world(j) for j in jobs])

    def test_pool_equals_sequential_with_batches(self):
        jobs = self._jobs()[:30] + self._extras()
        expected = [run_world(j) for j in jobs]
        self.assertEqual(run_worlds(jobs, 3), expected)
        self.assertEqual(run_worlds(jobs, 3, batch_size=2), expected)

    def test_progress_reports_every_job_once(self):
        jobs = self._jobs() + self._extras()
        seen = []
        run_worlds(jobs, 1, on_done=lambda i, r: seen.append(i))
        self.assertEqual(sorted(seen), list(range(len(jobs))))

    def test_bad_batch_size(self):
        with self.assertRaises(ValueError):
            run_worlds([], 1, batch_size=0)


if __name__ == "__main__":
    unittest.main()
