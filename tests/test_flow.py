"""Event-time flow model (round 2): contract, determinism, hftbacktest-style same-price queue,
first passage / sweeps, no-stale-fill invariant, ledger identity."""
import dataclasses
from dataclasses import replace

import numpy as np
import pytest

from quant_research.worldsim import Collect, Engine, Fees, FillConfig, Intent, World, WorldConfig
from quant_research.worldsim.contracts import FlowEvents, Frame, arm_view
from quant_research.worldsim.fills import Orders
from quant_research.worldsim.reference_arms import Churn, GeometricGrid

ARM_FIELDS = ["ts_ms", "group", "mid", "best_bid", "best_ask", "tick_size", "bid_px", "ask_px",
              "bid_size", "ask_size", "trade_price", "trade_size", "aggressor", "settlement_ms",
              "outcome", "condition_id", "is_no"]
NEUTRAL = dict(kappa_p=0.0, sweep_rate_per_hour=0.0, vol_logsd=0.0, sv_logsd=0.0, spread_logsd=0.0,
               spread_keep_5s=0.0)


def test_arm_facing_frame_contract_pinned():
    names = [f.name for f in dataclasses.fields(Frame)]
    assert names[:len(ARM_FIELDS)] == ARM_FIELDS            # the 17 contract fields, same order
    assert names[len(ARM_FIELDS):] == ["flow"]               # one additive engine-internal field
    f = list(World(WorldConfig(markets=8, steps=6, depth=3, dt_ms=5000), 3))[2]
    n, d = 8, 3
    for name in ("mid", "best_bid", "best_ask", "tick_size", "trade_price", "trade_size",
                 "aggressor", "condition_id", "is_no"):
        assert getattr(f, name).shape == (n,), name
    for name in ("bid_px", "ask_px", "bid_size", "ask_size"):
        assert getattr(f, name).shape == (n, d), name
    assert f.aggressor.dtype == np.int8 and f.trade_size.dtype == np.int64
    assert isinstance(f.flow, FlowEvents) and f.flow.c0.shape == (n,) and f.flow.hs.shape == (n,)
    assert arm_view(f).flow is None


def test_arms_cannot_see_flow_through_snapshot():
    seen = []

    class Spy:
        levels = 1

        def decide(self, snap):
            seen.append(snap.frame.flow)
            a = np.full((len(snap.frame.mid), 2, 1), 2, dtype=np.int8)
            return Intent(a, np.zeros(a.shape), np.ones(a.shape, dtype=np.int64))

    Engine().run(World(WorldConfig(markets=4, steps=5, print_rate_per_hour=3000.0), 1), Spy())
    assert seen and all(x is None for x in seen)


def _same(x, y):
    for g in dataclasses.fields(x):
        u, v = getattr(x, g.name), getattr(y, g.name)
        assert (u.tobytes() == v.tobytes()) if isinstance(u, np.ndarray) else u == v, g.name


def test_flow_is_deterministic_and_seed_dependent():
    cfg = WorldConfig(markets=8, steps=10, dt_ms=300_000, print_rate_per_hour=600.0)
    a, b, c = list(World(cfg, 5)), list(World(cfg, 5)), list(World(cfg, 6))
    for x, y in zip(a[1:], b[1:]):
        _same(x.flow, y.flow)
    assert a[3].flow.mid.tobytes() != c[3].flow.mid.tobytes()


def test_flow_neutral_config_leaves_arm_facing_streams_bit_identical():
    """With every impact / heterogeneity / sweep term at zero the flow layer (its own RNG
    stream) must not move a single arm-facing array relative to the flow-off world."""
    kw = dict(markets=6, steps=8, dt_ms=5000, **NEUTRAL)
    on = list(World(WorldConfig(print_rate_per_hour=2000.0, **kw), 9))
    off = list(World(WorldConfig(print_rate_per_hour=0.0, **kw), 9))
    for x, y in zip(on, off):
        for n in ARM_FIELDS:
            u, v = getattr(x, n), getattr(y, n)
            assert (u.tobytes() == v.tobytes()) if isinstance(u, np.ndarray) else u == v
    assert all(f.flow is None for f in off)


@pytest.mark.parametrize("group", ["perp", "prediction"])
def test_poisson_event_rate_and_structure(group):
    pr, sr = 1800.0, 600.0
    cfg = WorldConfig(group=group, markets=40, steps=30, dt_ms=300_000, print_rate_per_hour=pr,
                      sweep_rate_per_hour=sr, paired=(group == "prediction"), print_spread_elast=0.0)
    frames = list(World(cfg, 2))
    counts = [(f.flow.size > 0).sum(axis=0) for f in frames[1:-1] if f.flow is not None]
    assert abs(np.mean(counts) - (pr + sr) * 300_000 / 3.6e6) < 2.0
    f = frames[5]
    live = f.flow.size > 0
    fr = f.flow.frac
    assert ((fr[live] > 0) & (fr[live] < 1)).all()
    assert (np.diff(np.where(live, fr, np.inf), axis=0)[live[1:] & live[:-1]] >= 0).all()     # time order
    assert set(np.unique(f.flow.side[live])) <= {-1, 1}
    np.testing.assert_allclose(f.flow.price, np.round(f.flow.price, 6))
    # path knots: bar-end centre equals the displayed mid of the frame
    np.testing.assert_allclose(f.flow.c1, f.mid, atol=1e-9)
    if group == "prediction":
        assert (f.flow.price[live] > 0).all() and (f.flow.price[live] < 1).all()
        np.testing.assert_array_equal(f.flow.side[:, ::2][live[:, ::2]], -f.flow.side[:, 1::2][live[:, 1::2]])
        np.testing.assert_allclose(f.flow.mid_post, f.flow.mid)          # no impact in prediction worlds


def test_event_time_path_is_continuous_and_consistent():
    """mid just after event i == path knot; start/end knots tie to consecutive frames; impact moves
    the centre by the permanent + transient jump of a sweep."""
    cfg = WorldConfig(markets=20, steps=30, dt_ms=5000, print_rate_per_hour=3000.0, sweep_rate_per_hour=2000.0,
                      sweep_trade_frac=1.0, kappa_p=0.0)
    frames = list(World(cfg, 4))
    for a, b in zip(frames[1:-1], frames[2:]):
        np.testing.assert_allclose(a.flow.c1, b.flow.c0, atol=1e-9)         # bar knots chain
    f = frames[6].flow
    live = f.size > 0
    jump = (f.mid_post - f.mid)[live]
    sweeps = f.size[live] == 1_000_000
    assert sweeps.any()
    # a trade-through sweep moves the centre by its depth in the direction of its side
    tick = cfg.tick
    d = np.round(jump[sweeps] / tick)
    assert (d != 0).all() and (np.sign(d) == f.side[live][sweeps]).all()
    # touch prints (kappa_p = 0) leave the centre where it is
    assert np.allclose(jump[~sweeps], 0.0)


def test_flow_validation():
    for bad in (dict(print_rate_per_hour=-1.0), dict(clip_median=0.0), dict(clip_sigma=-1.0),
                dict(print_depth_ticks=0.0), dict(sweep_rate_per_hour=-1.0), dict(sweep_alpha=1.0),
                dict(sweep_trade_frac=1.5), dict(lambda_perm=1.5), dict(kappa_p=-0.1),
                dict(impact_tau_ms=0.0), dict(vol_logsd=-1.0), dict(sv_logsd=-1.0),
                dict(sv_halflife_ms=0.0), dict(spread_keep_5s=1.0), dict(spread_logsd=-1.0),
                dict(intrabar_sd_mult=-1.0), dict(sweep_xm_ticks=float("nan")),
                dict(print_rate_per_hour=1e9), dict(sweep_max_ticks=0)):
        with pytest.raises(ValueError):
            WorldConfig(**bad)
    for bad in (dict(psi=1.5), dict(d1=-1.0), dict(cancel_hazard_per_s=-1.0), dict(beta=float("nan"))):
        with pytest.raises(ValueError):
            FillConfig(**bad)


# ------------------------------------------------------------------ fill semantics (hand-built flow)
N = 2


def _fe(side, price, size, touch, mid=100.0, mid_post=None, frac=0.5, hs=5.0, c0=100.0, c1=100.0, t0=0, span=1000):
    """One event per market (E = 1)."""
    arr = lambda x, dt: np.full((1, N), x, dtype=dt)
    vec = lambda x: np.full(N, x, dtype=float)
    return FlowEvents(size=arr(size, np.int64), price=arr(price, float), touch=arr(touch, float),
                      side=arr(side, np.int8), mid=arr(mid, float),
                      mid_post=arr(mid if mid_post is None else mid_post, float), tpost=arr(0.0, float),
                      frac=arr(frac, float), t0_ms=t0, span_ms=span, c0=vec(c0), c1=vec(c1), hs=vec(hs),
                      tau_ms=2000.0, tstart=vec(0.0), tend=vec(0.0))


def _empty(c0=100.0, c1=100.0, hs=5.0, span=1000):
    z = np.zeros((0, N))
    vec = lambda x: np.full(N, x, dtype=float)
    return FlowEvents(size=z.astype(np.int64), price=z, touch=z, side=z.astype(np.int8), mid=z, mid_post=z,
                      tpost=z, frac=z, t0_ms=0, span_ms=span, c0=vec(c0), c1=vec(c1), hs=vec(hs),
                      tau_ms=2000.0, tstart=vec(0.0), tend=vec(0.0))


def _frame(flow, ts=1000):
    base = list(World(WorldConfig(markets=N, steps=3, print_rate_per_hour=0.0), 1))[1]
    return replace(base, ts_ms=ts, bid_px=np.tile([[99.94, 99.93, 99.92]], (N, 1)),
                   ask_px=np.tile([[100.06, 100.07, 100.08]], (N, 1)),
                   bid_size=np.full((N, 3), 5), ask_size=np.full((N, 3), 5),
                   best_bid=np.full(N, 99.95), best_ask=np.full(N, 100.05), mid=np.full(N, 100.0),
                   tick_size=np.full(N, 0.01), flow=flow)


def _book(cfg, bids=(99.9, 99.8), asks=None, sizes=None):
    """Orders with live rungs; the first bar's frame carries an (empty) flow so the same-price
    queue is initialised (psi * D(k))."""
    L = 2
    o = Orders(N, L, cfg)
    a = np.full((N, 2, L), 2, dtype=np.int8)
    p = np.zeros((N, 2, L))
    if bids is not None:
        a[:, 0, :] = 1
        p[:, 0, :] = bids
    if asks is not None:
        a[:, 1, :] = 1
        p[:, 1, :] = asks
    o.submit(_frame(_empty(), ts=0), Intent(a, p, np.ones(a.shape, dtype=np.int64) if sizes is None else sizes))
    return o


def _fills(o, flow, ts=1000):
    out = list(o.fills(_frame(flow, ts)))
    return out


CFG = FillConfig(psi=1.0, d1=10.0, beta=0.0, cancel_hazard_per_s=0.0)


def test_same_price_queue_then_overflow_fills():
    o = _book(CFG)                                       # rung 99.9 is 5 ticks behind the touch
    assert (o.queue[:, 0, 0] == 10.0).all()              # Q0 = psi * D1 (beta = 0)
    # B(5) = 5 * 10 displayed units between touch and rung, then Q = 10: 60 burns, 61 fills 1
    assert not _fills(o, _fe(-1, 99.90, 60, 99.95))
    o = _book(CFG)
    out = _fills(o, _fe(-1, 99.90, 61, 99.95))
    assert len(out) == 1 and (out[0][0] == 1).all() and (out[0][1] == 99.9).all()
    assert (o.size[:, 0, 0] == 0).all() and (o.size[:, 0, 1] == 1).all()
    assert (o.queue[:, 0, 0] == 0.0).all()               # the queue ahead was burned


def test_queue_burns_cumulatively_and_is_kept_across_bars():
    o = _book(CFG)
    # three prints reaching rung 1, each too small to get past B(5)=50 on its own
    for v in (30, 40):
        assert not _fills(o, _fe(-1, 99.90, v, 99.95))
    assert (o.queue[:, 0, 0] == 10.0).all()              # volume spent on the displayed depth only
    # the queue position is earned: an order that is KEPT keeps its Q (no re-initialisation)
    o.queue[:, 0, 0] = 3.0
    o.advance(_frame(_empty(), ts=2000))
    assert (o.queue[:, 0, 0] == 3.0).all()
    out = _fills(o, _fe(-1, 99.90, 54, 99.95), ts=3000)  # 50 + 3 = 53 burns, 1 overflows
    assert len(out) == 1 and (out[0][0] == 1).all()


def test_cancel_hazard_decays_queue():
    o = _book(replace(CFG, cancel_hazard_per_s=np.log(2.0)))      # half-life 1 s
    _fills(o, _fe(-1, 99.0, 1, 99.95, span=1000))                  # does not reach: pure decay over 1 bar
    np.testing.assert_allclose(o.queue[:, 0, 0], 5.0)


def test_sweep_trades_through_every_rung_within_its_depth_and_not_beyond():
    o = _book(CFG)
    out = _fills(o, _fe(-1, 99.85, 1_000_000, 99.95, mid_post=99.90))       # 10 ticks deep: reaches 99.9 only
    assert len(out) == 1 and (out[0][1] == 99.9).all() and (out[0][0] == 1).all()
    assert (o.size[:, 0, 1] == 1).all()
    o = _book(CFG)
    out = _fills(o, _fe(-1, 99.70, 1_000_000, 99.95, mid_post=99.75))       # 25 ticks: both rungs, each at ITS price
    assert sorted(float(p[0]) for _, p in out) == [99.8, 99.9]
    assert (o.size[:, 0, :] == 0).all()


def test_fill_edge_invariant_pre_fill_mid_never_on_wrong_side():
    o = _book(CFG)
    _fills(o, _fe(-1, 99.70, 1_000_000, 99.95, mid=99.88, mid_post=99.7))    # mid already drifted below the rung
    o2 = _book(CFG)
    seen = []
    for q, p in o2.fills(_frame(_fe(-1, 99.70, 1_000_000, 99.95, mid=99.88, mid_post=99.7))):
        fr, pm = o2.event_info
        seen.append((p[0], pm[0]))
    assert seen and all(pm - p >= 0.05 - 1e-9 for p, pm in seen)           # centre >= rung + half spread


def test_drift_alone_never_fills_by_default_but_literal_first_passage_does():
    drift = _empty(c0=100.0, c1=99.80)                                     # touch drifts 99.95 -> 99.75 through 99.9
    o = _book(CFG)
    assert not _fills(o, drift)                                            # exogenous drift cannot trade through the order
    o = _book(replace(CFG, drift_passage=True))
    out = []
    for q, p in o.fills(_frame(drift)):
        out.append((p[0], q[0], o.event_info[0][0], o.event_info[1][0]))
    # filled at its OWN price (not the stale/new touch), at first passage 25% into the bar, centre = rung + hs
    assert {(a, b) for a, b, _, _ in out} >= {(99.9, 1)}
    px, qty, frac, pre = next(x for x in out if x[0] == 99.9)
    assert abs(frac - 0.25) < 1e-9 and abs(pre - 99.95) < 1e-9


def test_rung_inside_the_market_is_hit_by_any_print_with_no_queue():
    o = _book(CFG, bids=(99.97, 99.8))                                     # 99.97 is inside the 99.95 touch? no: above it
    assert (o.queue[:, 0, 0] == 0.0).all()                                 # inside the market: nothing ahead
    out = _fills(o, _fe(-1, 99.95, 1, 99.95))
    assert len(out) == 1 and (out[0][1] == 99.97).all() and (out[0][0] == 1).all()


def test_ask_side_and_wrong_side_events():
    o = _book(CFG, bids=None, asks=(100.1, 100.2))
    assert not _fills(o, _fe(-1, 99.0, 1_000_000, 99.95))                   # sells never touch asks
    out = _fills(o, _fe(+1, 100.15, 1_000_000, 100.05, mid_post=100.10))
    assert len(out) == 1 and (out[0][1] == 100.1).all() and (out[0][0] == -1).all()


def test_post_latency_blocks_events_before_the_post_lands():
    cfg = replace(CFG, post_ms=500, psi=0.0)
    o = Orders(N, 1, cfg)
    a = np.full((N, 2, 1), 2, dtype=np.int8)
    a[:, 0] = 1
    p = np.zeros((N, 2, 1))
    p[:, 0] = 99.96
    o.submit(_frame(_empty(), ts=0), Intent(a, p, np.ones(a.shape, dtype=np.int64)))
    o.advance(_frame(_empty(), ts=1000))
    assert not list(o.fills(_frame(_fe(-1, 99.9, 1_000_000, 99.95, frac=0.2), ts=1000)))     # t=200 ms < 500 ms
    assert len(list(o.fills(_frame(_fe(-1, 99.9, 1_000_000, 99.95, frac=0.8), ts=1000)))) == 1


# ------------------------------------------------------------------ whole worlds
@pytest.mark.parametrize("drift_passage", [False, True])
def test_no_fill_with_maker_edge_below_one_tick(drift_passage):
    """INVARIANT: in a real book price passing a resting order fills it at its own price, so no
    fill has a maker edge below -1 tick (tape share: 0.06%)."""
    cfg = WorldConfig(markets=12, steps=1500, dt_ms=5000, perp_sigma=0.03, common_weight=0.05,
                      cluster_weight=0.1, clusters=3, spread_mean_ticks=5.0)
    r = Engine(FillConfig(drift_passage=drift_passage), Fees(maker_rate=0.00002, maker_pays=True),
               Collect.FILLS).run(World(cfg, 4), GeometricGrid(grid_step=0.0008))
    assert r.fill_count > 100
    edge = np.array([np.sign(f["signed_size"]) * (f["print_mid"] - f["price"]) / cfg.tick for f in r.fills])
    assert edge.min() >= -1.0 - 1e-9


@pytest.mark.parametrize("maker_pays", [False, True])
def test_ledger_identity_and_fills_in_flow_world(maker_pays):
    cfg = WorldConfig(markets=8, steps=200, dt_ms=5000, perp_sigma=0.03, common_weight=0.05,
                      cluster_weight=0.1, clusters=3, spread_mean_ticks=5.0)
    fees = Fees(maker_rate=0.00002, maker_pays=maker_pays)
    for arm in (GeometricGrid(grid_step=0.0008), Churn()):
        r = Engine(FillConfig(), fees, Collect.FILLS).run(World(cfg, 4), arm)
        lg = r.ledger
        assert r.fill_count > 0
        assert np.array_equal(lg.gross, r.pnl + lg.fees_paid - lg.rebates_received)
        assert (lg.fees_paid > 0).any() if maker_pays else (lg.rebates_received > 0).any()
        assert all("frac" in f and 0 < f["frac"] <= 1 for f in r.fills)


def test_prediction_flow_world_settles_and_stays_in_bounds():
    cfg = WorldConfig(group="prediction", markets=6, steps=20, regime="fast", paired=True,
                      print_rate_per_hour=40_000.0, sweep_rate_per_hour=20_000.0, sweep_trade_frac=1.0)
    r = Engine(FillConfig(), Fees(), Collect.FILLS).run(World(cfg, 3), Churn())
    assert r.fill_count > 0
    for f in list(World(cfg, 3))[:-1]:
        if f.flow is not None and (f.flow.size > 0).any():
            live = f.flow.size > 0
            assert (f.flow.price[live] > 0).all() and (f.flow.price[live] < 1).all()
    assert list(World(cfg, 3))[-1].flow is None        # terminal frame: no events, only settlement
