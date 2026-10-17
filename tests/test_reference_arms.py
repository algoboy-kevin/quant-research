"""Reference arms: registry semantics + worldsim contract on both groups."""
import unittest

import numpy as np

from quant_research.worldsim.contracts import Collect, Frame, Snapshot, WorldConfig
from quant_research.worldsim.engine import Engine
from quant_research.worldsim.fees import Fees
from quant_research.worldsim.reference_arms import available, build, build_many, parse_spec
from quant_research.worldsim.worlds import World


class RegistryTests(unittest.TestCase):
    def test_three_styles_available(self):
        self.assertEqual(available(), ["carry", "churn", "far_fill"])

    def test_unknown_arm_raises(self):
        with self.assertRaises(KeyError):
            build("nonexistent_arm")

    def test_param_override(self):
        name, params, arm = build("churn.width_ticks=2.size=3")
        self.assertEqual(name, "churn")
        self.assertEqual(params["width_ticks"], 2)
        self.assertEqual(params["size"], 3)
        self.assertEqual(params["inventory_limit"], 12)  # default preserved
        self.assertEqual(arm.width_ticks, 2)

    def test_bad_override_raises(self):
        with self.assertRaises(ValueError):
            parse_spec("churn.width_ticks")

    def test_float_override_roundtrips(self):
        name, params = parse_spec("churn.skew_ticks=0.25.width_ticks=2")
        self.assertEqual(name, "churn")
        self.assertEqual(params, {"skew_ticks": 0.25, "width_ticks": 2})
        _, _, arm = build("churn.skew_ticks=0.25")
        self.assertEqual(arm.skew_ticks, 0.25)

    def test_build_many_all(self):
        arms = build_many(None)
        self.assertEqual(set(arms), {"carry", "churn", "far_fill"})
        for name, (params, arm) in arms.items():
            self.assertTrue(hasattr(arm, "decide"), name)


class ContractTests(unittest.TestCase):
    def test_each_arm_runs_on_tiny_worlds(self):
        import math

        for name, (params, arm) in build_many(None).items():
            for group in ("perp", "prediction"):
                cfg = WorldConfig(group=group, markets=8, steps=20)
                res = Engine(fees=Fees(), collect=Collect.NONE).run(World(cfg, 7), arm)
                self.assertEqual(res.pnl.shape, (8,), f"{name}/{group}")
                self.assertTrue(all(math.isfinite(float(x)) for x in res.pnl / 1e9),
                                f"{name}/{group}")

    def test_same_world_seed_is_deterministic(self):
        _, _, arm = build("churn")
        cfg = WorldConfig(group="perp", markets=6, steps=15)
        a = Engine(fees=Fees(), collect=Collect.NONE).run(World(cfg, 11), arm)
        b = Engine(fees=Fees(), collect=Collect.NONE).run(World(cfg, 11), arm)
        np.testing.assert_array_equal(a.pnl, b.pnl)


def _snapshot(mid_value: float, inv_q: int):
    """1-market prediction snapshot pinned at mid_value with inventory inv_q."""
    n, d = 1, 3
    tick = np.full(n, 0.001)
    mid = np.array([mid_value])
    offsets = 0.001 * np.arange(1, d + 1)
    bid = np.round(mid[:, None] - offsets, 6)
    ask = np.round(mid[:, None] + offsets, 6)
    frame = Frame(
        ts_ms=0, group="prediction", mid=mid,
        best_bid=bid[:, 0].copy(), best_ask=ask[:, 0].copy(), tick_size=tick,
        bid_px=bid, ask_px=ask,
        bid_size=np.full((n, d), 3, dtype=np.int64),
        ask_size=np.full((n, d), 3, dtype=np.int64),
        trade_price=mid.copy(), trade_size=np.zeros(n, dtype=np.int64),
        aggressor=np.zeros(n, dtype=np.int8),
        settlement_ms=1000, outcome=None,
        condition_id=np.arange(n), is_no=np.zeros(n, dtype=bool))
    return Snapshot(
        frame=frame, inventory=np.array([inv_q], dtype=np.int64),
        cash=np.zeros(n, dtype=np.int64),
        order_price=np.full((n, 2, 1), np.nan),
        order_size=np.zeros((n, 2, 1), dtype=np.int64),
        order_status=np.zeros((n, 2, 1), dtype=np.int8),
        last_fill_mid=np.array([np.nan]),
        time_to_settlement_ms=1000)


class ClipEdgeTests(unittest.TestCase):
    """The engine rounds quote prices OUTWARD to the tick and then enforces strict
    (0,1) bounds; arms must mask post-rounding prices, not pre-rounding ones.
    Inventory skew can push a pre-round price into (0, tick) which floors to 0."""

    def test_no_arm_submits_invalid_prediction_quote(self):
        from quant_research.worldsim.contracts import Intent  # noqa: F401
        from quant_research.worldsim.fills import FillConfig, Orders

        for name, (params, arm) in build_many(None).items():
            for mid_value in (0.004, 0.005, 0.996, 0.995):
                for inv_q in (-11, -10, -9, 0, 9, 10, 11):
                    snap = _snapshot(mid_value, inv_q)
                    intent = arm.decide(snap)
                    orders = Orders(1, arm.levels, FillConfig())
                    try:
                        orders.submit(snap.frame, intent)
                    except ValueError as e:
                        self.fail(f"{name} mid={mid_value} q={inv_q}: {e}")


if __name__ == "__main__":
    unittest.main()
