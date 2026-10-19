from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from quant_research.worldsim import (CarryArm, ChurnArm, Collect, Engine, Fees,
                                     FillConfig, Intent, World, WorldConfig)
from quant_research.worldsim.fills import Orders
from quant_research.worldsim.ledger import Ledger
from quant_research.worldsim.portfolio import breadth, run_portfolio, statistics
from quant_research.worldsim.worlds import belief_step


def frame(ts=0, bid=.49, ask=.51, trade=.49, qty=0, aggressor=-1):
    base = next(iter(World(WorldConfig(group="prediction", markets=1, steps=20), 1)))
    return replace(base, ts_ms=ts, mid=np.array([(bid + ask) / 2]),
                   best_bid=np.array([bid]), best_ask=np.array([ask]),
                   bid_px=np.array([[bid, bid - .001, bid - .002]]),
                   ask_px=np.array([[ask, ask + .001, ask + .002]]),
                   bid_size=np.array([[3, 4, 5]]), ask_size=np.array([[3, 4, 5]]),
                   trade_price=np.array([trade]), trade_size=np.array([qty]),
                   aggressor=np.array([aggressor], dtype=np.int8))


def intent(bids=(.49,), asks=None, size=1, action=1):
    levels = len(bids)
    a = np.full((1, 2, levels), 2, dtype=np.int8)
    p = np.zeros_like(a, dtype=float)
    a[:, 0] = action
    p[:, 0] = bids
    if asks is not None:
        a[:, 1] = action
        p[:, 1] = asks
    return Intent(a, p, np.full(a.shape, size, dtype=np.int64))


@pytest.mark.parametrize("group", ["perp", "prediction"])
def test_world_and_engine_determinism(group):
    cfg = WorldConfig(group=group, markets=8, steps=12)
    first, second = list(World(cfg, 7)), list(World(cfg, 7))
    for a, b in zip(first, second):
        for name in a.__dataclass_fields__:
            x, y = getattr(a, name), getattr(b, name)
            if isinstance(x, np.ndarray):
                assert x.tobytes() == y.tobytes()
                assert not x.flags.writeable
            else:
                assert x == y
    a = Engine(collect=Collect.ALL).run(World(cfg, 7), ChurnArm())
    b = Engine(collect=Collect.ALL).run(World(cfg, 7), ChurnArm())
    assert a.pnl.tobytes() == b.pnl.tobytes()
    assert a.fills == b.fills and a.steps == b.steps
    assert a.ledger.roundtrips == b.ledger.roundtrips


@pytest.mark.parametrize("regime", ["slow", "fast"])
def test_prediction_bounds_absorption_and_mirror(regime):
    frames = list(World(WorldConfig(group="prediction", markets=40, steps=100,
                                   regime=regime, paired=True), 19))
    for f in frames[:-1]:
        assert np.all((f.mid > 0) & (f.mid < 1))
        assert np.all((f.bid_px > 0) & (f.ask_px < 1))
        assert f.outcome is None
        np.testing.assert_allclose(f.mid[::2] + f.mid[1::2], 1, atol=0, rtol=0)
        np.testing.assert_allclose(f.bid_px[::2] + f.ask_px[1::2], 1, atol=1e-15)
        assert np.array_equal(f.aggressor[::2], -f.aggressor[1::2])
    terminal = frames[-1]
    assert set(terminal.mid) <= {0., 1.}
    assert np.array_equal(terminal.outcome[::2] + terminal.outcome[1::2], np.ones(20))
    assert not terminal.trade_size.any()


def test_belief_is_conditional_martingale_and_stays_bounded():
    rng = np.random.default_rng(5)
    z = rng.standard_normal(100000)
    for b in (.001, .1, .5, .9, .999):
        out = belief_step(b, z, .18)
        assert np.all((out > 0) & (out < 1))
        assert abs(out.mean() - b) < 5 * out.std() / np.sqrt(len(out))
        # Antithetic conditional symmetry, not merely an unconditional mean test.
        np.testing.assert_allclose(belief_step(b, z, .18) + belief_step(b, -z, .18), 2*b)


def test_factors_change_correlation():
    def increments(common, cluster):
        # latent-only view: the OU dislocation is an independent per-market display term
        c = WorldConfig(markets=50, steps=120, common_weight=common,
                        cluster_weight=cluster, clusters=1, dislocation_sd_ticks=0, tick=0.001)
        return np.diff(np.stack([f.mid for f in World(c, 8)]), axis=0)
    independent = np.corrcoef(increments(0, 0).T)
    clustered = np.corrcoef(increments(.3, .6).T)
    mask = ~np.eye(50, dtype=bool)
    assert abs(independent[mask].mean()) < .05
    assert clustered[mask].mean() > .8


def test_crossing_and_price_priority_conserve_print_volume():
    orders = Orders(1, 2)
    orders.submit(frame(), intent((.49, .48), size=2))
    assert not list(orders.fills(frame(250, trade=.495, qty=9)))
    fills = list(orders.fills(frame(500, trade=.47, qty=3)))
    assert [(int(q[0]), float(p[0])) for q, p in fills] == [(2, .49), (1, .48)]
    assert orders.size[0, 0, 1] == 1


def test_ask_crossing_only_on_buy():
    orders = Orders(1, 1)
    orders.submit(frame(), intent((.49,), (.51,), size=2))
    fills = list(orders.fills(frame(250, trade=.52, qty=1, aggressor=1)))
    assert len(fills) == 1 and fills[0][0][0] == -1 and fills[0][1][0] == .51


def test_queue_pessimism_and_shared_depth():
    front, back = Orders(1, 2), Orders(1, 2, FillConfig(queue_fraction=1))
    for o in (front, back):
        o.submit(frame(), intent((.49, .489), size=1))
    f = frame(250, trade=.489, qty=4)
    assert sum(abs(int(q[0])) for q, p in front.fills(f)) == 2
    assert sum(abs(int(q[0])) for q, p in back.fills(f)) == 1
    # 3 units ahead consumed once; remaining second level queue is 4, not 7.
    assert back.queue[0, 0, 1] == 4
    assert not list(back.fills(frame(500, trade=.489, qty=4)))
    assert sum(int(q[0]) for q, p in back.fills(frame(750, trade=.489, qty=1))) == 1


def test_activation_rejects_cross_and_recover_in_flight():
    orders = Orders(1, 1, FillConfig(post_ms=500))
    orders.submit(frame(), intent())
    f = frame(250, bid=.47, ask=.48, trade=.48, qty=5)
    orders.advance(f)
    assert not list(orders.fills(f))
    orders.advance(frame(500))
    assert orders.rejections == 1 and not orders.status.any()


def test_activation_tie_and_cancel_tie():
    orders = Orders(1, 1, FillConfig(post_ms=250, cancel_ms=250))
    orders.submit(frame(), intent(size=3))
    f = frame(250, qty=1)
    orders.advance(f)
    assert list(orders.fills(f))[0][0][0] == 1
    orders.submit(f, intent(action=2))
    assert list(orders.fills(frame(499, qty=1)))[0][0][0] == 1
    f = frame(500, qty=1)
    orders.advance(f)
    assert not list(orders.fills(f))


def test_replacement_waits_cancel_then_post():
    orders = Orders(1, 1, FillConfig(post_ms=250, cancel_ms=250))
    orders.submit(frame(), intent(size=3))
    orders.advance(frame(250))
    orders.submit(frame(250), intent((.48,), size=2))
    assert orders.status[0, 0, 0] == 3
    assert list(orders.fills(frame(499, qty=1)))[0][1][0] == .49
    orders.advance(frame(500))
    assert orders.status[0, 0, 0] == 1 and orders.price[0, 0, 0] == .48
    orders.advance(frame(750))
    assert orders.status[0, 0, 0] == 2


def test_offset_ticks_and_invalid_intents():
    orders = Orders(1, 1)
    a = intent((2,))
    orders.submit(frame(), replace(a, price_mode="offset"))
    assert orders.price[0, 0, 0] == .498
    with pytest.raises(ValueError):
        orders.submit(frame(), intent((float("nan"),)))
    with pytest.raises(ValueError):
        Orders(1, 2).submit(frame(), intent((.48, .49)))
    with pytest.raises(ValueError):
        Orders(1, 1).submit(frame(), intent((.52,), (.51,)))


def test_fee_symmetry_exact_and_native_formula():
    fees = Fees(.02, .03)
    p = np.arange(1, 1000) / 1000
    q = np.ones(len(p), dtype=np.int64)
    for maker in (True, False):
        assert np.array_equal(fees.amount(p, q, "prediction", maker=maker),
                              fees.amount(1-p, q, "prediction", maker=maker))
    assert fees.amount(np.array([.5]), np.array([2]), "prediction", maker=False)[0] == 15000000


def test_maker_pays_direction_and_ledger_identity():
    from quant_research.worldsim.reference_arms import build
    from quant_research.worldsim.worlds import World

    _, _, arm = build("churn")
    cfg = WorldConfig(group="perp", markets=8, steps=60)
    out = {}
    for maker_pays in (False, True):
        res = Engine(fees=Fees(maker_rate=0.00015, taker_rate=0.00045, maker_pays=maker_pays),
                     collect=Collect.NONE).run(World(cfg, 23), arm)
        led = res.ledger
        fees = np.asarray(led.fees_paid, dtype=np.int64)
        rebates = np.asarray(led.rebates_received, dtype=np.int64)
        gross = np.asarray(led.gross, dtype=np.int64)
        # ledger identity row-wise, both directions
        assert np.array_equal(gross, res.pnl + fees - rebates)
        out[maker_pays] = (fees, rebates, res.fill_count)
    fees0, reb0, fills = out[False]
    fees1, reb1, _ = out[True]
    assert fills > 0                            # the run must actually trade
    assert fees0.sum() == 0 and reb0.sum() > 0  # legacy: maker charge is a rebate
    assert fees1.sum() > 0 and reb1.sum() == 0  # Hyperliquid: maker pays the charge


@pytest.mark.parametrize("settle", [False, True])
def test_ledger_exact_randomized_and_roundtrips(settle):
    rng = np.random.default_rng(17)
    ledger = Ledger(12, Collect.FILLS)
    for t in range(80):
        dq = rng.integers(-3, 4, 12)
        p = rng.integers(1, 1000, 12) / 1000
        ledger.fill(t, dq, p, fee=rng.integers(0, 20, 12), rebate=rng.integers(0, 20, 12))
    mark = rng.integers(0, 2, 12) if settle else np.full(12, .5)
    final = ledger.finish(81, mark, settle=settle)
    for i in range(12):
        assert sum(r.pnl for r in ledger.roundtrips if r.market == i) == int(final[i] + ledger.fees_paid[i] - ledger.rebates_received[i])
    assert np.array_equal(final, ledger.equity(mark))
    with pytest.raises(ValueError):
        ledger.finish(82, mark)


def test_average_cost_partial_close_reversal_and_terminal_mark():
    ledger = Ledger(1, Collect.FILLS)
    for t, dq, price in [(0, 2, .2), (1, 1, .5), (2, -1, .6), (3, -4, .4)]:
        ledger.fill(t, np.array([dq]), np.array([price]))
    assert ledger.roundtrips[0].entry_cost == 300000000
    assert ledger.roundtrips[0].pnl == 300000000
    assert ledger.inventory[0] == -2
    final = ledger.finish(4, np.array([.1]))
    assert final[0] == 1100000000
    assert ledger.inventory[0] == -2  # mark is attribution, not liquidation
    assert ledger.roundtrips[-1].kind == "mark"


@pytest.mark.parametrize("outcome", [0, 1])
def test_complementary_pair_guaranteed_dollar_and_short_equivalence(outcome):
    ledger = Ledger(2, Collect.FILLS)
    ledger.fill(0, np.array([1, 1]), np.array([.3, .6]))
    cash_before = ledger.cash.sum()
    pnl = ledger.finish(1, np.array([outcome, 1-outcome]), settle=True)
    assert ledger.cash.sum() - cash_before == 1000000000
    assert sum(pnl) == 100000000
    assert not ledger.inventory.any()
    short = Ledger(2)
    short.fill(0, np.array([-1, 1]), np.array([.3, .7]))
    pnl = short.finish(1, np.array([outcome, 1-outcome]), settle=True)
    assert pnl[0] == pnl[1]


@pytest.mark.parametrize("group", ["perp", "prediction"])
def test_collect_modes_same_results_and_snapshot_readonly(group):
    cfg = WorldConfig(group=group, markets=6, steps=12)
    results = [Engine(collect=c).run(World(cfg, 7), CarryArm()) for c in Collect.__members__.values()]
    assert all(np.array_equal(results[0].pnl, r.pnl) for r in results)
    assert all(results[0].fill_count == r.fill_count for r in results)
    assert not results[0].fills and not results[1].steps and results[2].steps

    class ReadOnlyArm:
        levels = 1

        def decide(self, s):
            with pytest.raises(ValueError):
                s.inventory[0] = 999
            with pytest.raises(ValueError):
                s.frame.mid[0] = 0
            assert not hasattr(s, "belief")
            return ChurnArm().decide(s)

    Engine().run(World(cfg, 2), ReadOnlyArm())


def test_sized_taker_depth_and_no_same_frame_maker_fill():
    class Take:
        levels = 1

        def decide(self, s):
            zeros = np.zeros((1, 2, 1), dtype=np.int8)
            return Intent(zeros, zeros.astype(float), zeros.astype(np.int64),
                          taker_size=np.array([100 if s.frame.ts_ms == 0 else 0]))

    cfg = WorldConfig(group="prediction", markets=1, steps=1)
    world = World(cfg, 6)
    initial = next(iter(world))
    result = Engine(collect=Collect.ALL).run(world, Take())
    assert sum(f['signed_size'] for f in result.fills) == initial.ask_size.sum()
    assert all(not f['maker'] and f['ts_ms'] == 0 for f in result.fills)
    assert result.ledger.fees_paid.sum() > 0 and result.ledger.rebates_received.sum() == 0
    assert result.ledger.inventory.sum() == 0


def test_breadth_math_zero_is_not_positive_and_uncertainty():
    panel = np.array([[1] * 10, [-1] * 10, [0] * 10], dtype=np.int64) * 1000000000
    rows = breadth(panel, 7).to_dicts()
    assert [r['markets'] for r in rows] == [5, 10]
    assert all(r['p_positive'] == 1/3 and r['mean'] == 0 for r in rows)
    assert rows[0]['sd'] == 5 and rows[1]['sd'] == 10
    assert statistics([1])['sd'] is None
    assert statistics([0])['p_positive'] == 0


def test_independent_breadth_beats_shared_risk():
    rng = np.random.default_rng(4)
    independent = np.rint((.25 + rng.standard_normal((3000, 100))) * 1e6).astype(np.int64)
    clustered = np.rint((.25 + .8 * rng.standard_normal((3000, 1))
                         + .6 * rng.standard_normal((3000, 100))) * 1e6).astype(np.int64)
    a, b = breadth(independent).to_dicts(), breadth(clustered).to_dicts()
    assert a[-1]['p_positive'] > a[0]['p_positive']
    assert a[-1]['p_positive'] > b[-1]['p_positive'] + .2


def test_artifacts_byte_identical_and_reconcile(tmp_path):
    cfg = WorldConfig(group="prediction", markets=5, steps=4)
    outputs = [tmp_path / name for name in ('a', 'b')]
    for out in outputs:
        run_portfolio([cfg], {'churn': ChurnArm()}, 2, 7, out)
    files = sorted(p.name for p in outputs[0].iterdir())
    assert len(files) == 8
    for name in files:
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
    rows = pl.read_parquet(outputs[0] / 'markets.parquet')
    assert rows.height == 10
    assert rows.select((pl.col('gross_nano') == pl.col('pnl_nano') + pl.col('fees_nano') - pl.col('rebates_nano')).all()).item()
    with pytest.raises(FileExistsError):
        run_portfolio([cfg], {'churn': ChurnArm()}, 2, 7, outputs[0])


def test_cli_smoke(tmp_path):
    proc = subprocess.run([sys.executable, '-m', 'quant_research.worldsim', 'run',
                           '--group', 'both', '--markets', '5', '--worlds', '2',
                           '--steps', '3', '--out', str(tmp_path / 'cli')],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert 'elapsed_seconds=' in proc.stdout
    assert (tmp_path / 'cli' / 'summary.md').is_file()


@pytest.mark.parametrize('kwargs', [dict(markets=0), dict(common_weight=.8, cluster_weight=.5),
                                    dict(group='prediction', paired=True, markets=3),
                                    dict(tick=.0000001), dict(adverse_selection=2)])
def test_bad_world_config(kwargs):
    with pytest.raises(ValueError):
        WorldConfig(**kwargs)


def test_no_same_frame_fill_and_last_fill_reference():
    class Tape:
        config = WorldConfig(group="perp", markets=1, steps=2)

        def __iter__(self):
            for t in (0, 250, 500):
                yield replace(frame(t, qty=2), group="perp", settlement_ms=None)

    class PostOnce:
        levels = 1

        def decide(self, snapshot):
            if snapshot.frame.ts_ms == 0:
                assert np.isnan(snapshot.last_fill_mid[0])
                return intent(size=1)
            assert snapshot.last_fill_mid[0] == .5
            return intent(action=0)

    result = Engine(collect=Collect.ALL).run(Tape(), PostOnce())
    assert len(result.fills) == 1 and result.fills[0]['ts_ms'] == 250


def test_cancel_inflight_and_zero_latency_rejection():
    orders = Orders(1, 1, FillConfig(post_ms=500))
    orders.submit(frame(), intent())
    orders.submit(frame(250), intent(action=2))
    orders.advance(frame(500, qty=10))
    assert not orders.status.any() and not list(orders.fills(frame(500, qty=10)))
    immediate = Orders(1, 1)
    immediate.submit(frame(), intent((.51,)))
    assert immediate.rejections == 1 and not immediate.status.any()


def test_empty_cancel_pending_does_not_steal_deeper_volume():
    orders = Orders(1, 2, FillConfig(cancel_ms=500))
    orders.submit(frame(), intent((.49, .489), size=1))
    orders.submit(frame(250), intent((.49, .489), action=2))
    first = list(orders.fills(frame(300, trade=.49, qty=1)))
    assert first[0][1][0] == .49
    second = list(orders.fills(frame(400, trade=.489, qty=1)))
    assert second[0][1][0] == .489


def test_negative_perp_price_and_overflow_are_explicit():
    ledger = Ledger(1, Collect.FILLS)
    ledger.fill(0, np.array([2]), np.array([-3.]))
    assert ledger.cash[0] == 6000000000
    assert ledger.finish(1, np.array([-2.]))[0] == 2000000000
    with pytest.raises(OverflowError):
        Ledger(1).fill(0, np.array([1000001]), np.array([1.]))


def test_adverse_selection_couples_flow_to_next_move():
    def sign_alignment(adverse):
        cfg = WorldConfig(markets=200, steps=40, common_weight=0, cluster_weight=0,
                          adverse_selection=adverse, perp_sigma=.1, trade_probability=1,
                          informed_strength=1.0, dislocation_sd_ticks=0)
        frames = list(World(cfg, 11))
        direction = np.stack([f.aggressor for f in frames[1:-1]])
        moves = np.diff(np.stack([f.mid for f in frames[1:]]), axis=0)
        return np.mean(direction[moves != 0] == np.sign(moves[moves != 0]))
    assert sign_alignment(1) == 1
    assert .47 < sign_alignment(0) < .53


def test_settlement_frequency_and_slow_fast_information():
    def sample(regime):
        cfg = WorldConfig(group="prediction", markets=10000, steps=15,
                          common_weight=0, cluster_weight=0, regime=regime)
        frames = list(World(cfg, 6))
        return frames[-2].mid, frames[-1].outcome
    slow, ys = sample('slow')
    fast, yf = sample('fast')
    assert fast.std() > slow.std() * 5
    assert abs(ys.mean() - .5) < .02 and abs(yf.mean() - .5) < .02
    # Bernoulli draw is conditional on the last latent update, not forced to 50/50.
    assert yf[fast > .55].mean() > yf[fast < .45].mean()


@pytest.mark.parametrize('group', ['perp', 'prediction'])
@pytest.mark.parametrize('arm', [ChurnArm(levels=3), CarryArm(levels=3)])
def test_multilevel_arms_with_latency_and_queue(group, arm):
    cfg = WorldConfig(group=group, markets=12, steps=30, regime='slow')
    result = Engine(FillConfig(post_ms=300, cancel_ms=300, queue_fraction=.5),
                    collect=Collect.FILLS).run(World(cfg, 9), arm)
    assert np.array_equal(result.ledger.gross,
                          result.pnl + result.ledger.fees_paid - result.ledger.rebates_received)
    assert result.fill_count > 0
    assert sum(r.pnl for r in result.ledger.roundtrips) == sum(result.ledger.gross)
