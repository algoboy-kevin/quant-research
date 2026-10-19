"""Display-layer fill model: NegBin spread, OU dislocation, wick prints, toxicity.

Statistical tests run on generated worlds with fixed seeds (deterministic); tolerances
are several standard errors wide. The calibration gates live in
test_fillmodel_calibration.py (slow).
"""
import math

import numpy as np
import pytest

from quant_research.worldsim.contracts import WorldConfig
from quant_research.worldsim.worlds import World, WorldBatch, _display_rng, _draw_display


def _frames(cfg, seed=3):
    return list(World(cfg, seed))[1:]      # drop step 0 (no prints)


def test_negbin_spread_moments():
    cfg = WorldConfig(spread_mean_ticks=12.0, spread_dispersion=4.0)
    draws = np.concatenate([_draw_display(_display_rng(s), cfg, 20000)[0] for s in range(5)])
    mean, var = draws.mean(), draws.var()
    assert abs(mean - 12.0) < 0.1
    assert abs(var - (12.0 + 12.0 ** 2 / 4.0)) / 48.0 < 0.05       # 48 = mean + mean^2/k
    assert var > mean


def test_displayed_spread_is_even_ticks_overdispersed_and_ordered():
    cfg = WorldConfig(group="perp", markets=200, steps=60, spread_mean_ticks=12.0, spread_dispersion=4.0,
                      print_rate_per_hour=0.0)
    spreads, wide = [], 0
    for f in _frames(cfg):
        ticks = np.rint((f.best_ask - f.best_bid) / f.tick_size).astype(int)
        assert np.all(ticks % 2 == 0) and ticks.min() >= 2          # outward rounding
        spreads.append(ticks)
        assert np.all(np.diff(f.bid_px, axis=1) < 0) and np.all(np.diff(f.ask_px, axis=1) > 0)
        assert np.all(f.bid_px[:, 0] < f.mid) and np.all(f.ask_px[:, 0] > f.mid)
        np.testing.assert_allclose(f.mid - f.best_bid, f.best_ask - f.mid, atol=1e-9)
    s = np.concatenate(spreads)
    assert 12.0 <= s.mean() <= 13.5          # NegBin mean 12 + outward-rounding bias (<= ~0.5 tick+)
    assert s.var() > s.mean()
    del wide


def test_eta_ou_autocorrelation_and_sd():
    halflife, dt = 600_000.0, 300_000
    cfg = WorldConfig(group="perp", markets=500, steps=80, dt_ms=dt, perp_sigma=0.0,
                      tick=0.001, dislocation_sd_ticks=50.0, dislocation_halflife_ms=halflife,
                      common_weight=0, cluster_weight=0, print_rate_per_hour=0.0)
    eta = np.stack([f.mid for f in World(cfg, 5)]) - 100.0          # latent value is constant
    eta /= 0.001
    assert abs(eta.std() - 50.0) < 1.5                              # stationary sd
    assert abs(eta.mean()) < 1.5
    for lag in (1, 2):
        rho = np.corrcoef(eta[:-lag].ravel(), eta[lag:].ravel())[0, 1]
        assert abs(rho - 0.5 ** (lag * dt / halflife)) < 0.02, (lag, rho)


def test_latent_process_independent_of_display_knobs():
    kw = dict(group="perp", markets=6, steps=30, dislocation_sd_ticks=0.0, perp_sigma=0.4,
              tick=0.001)
    a = WorldConfig(spread_mean_ticks=2.0, wick_prob=0.0, **kw)
    b = WorldConfig(spread_mean_ticks=40.0, spread_dispersion=0.3, wick_prob=0.9,
                    wick_max_ticks=5, **kw)
    for fa, fb in zip(World(a, 9), World(b, 9)):
        assert fa.mid.tobytes() == fb.mid.tobytes()
    # eta only displaces the displayed centre by a bounded multiple of its sd
    c = WorldConfig(**{**kw, "dislocation_sd_ticks": 10.0})
    dev = np.concatenate([np.abs(fa.mid - fc.mid) / 0.001 for fa, fc in zip(World(a, 9), World(c, 9))])
    assert 1 < dev.mean() < 20 and dev.max() < 10 * 6


def test_wick_distance_distribution():
    cfg = WorldConfig(group="perp", markets=400, steps=120, trade_probability=1.0,
                      wick_prob=0.1, wick_scale_ticks=10.0, wick_max_ticks=60, depth=3)
    w = []
    for f in _frames(cfg):
        touch = np.where(f.aggressor > 0, f.best_ask, f.best_bid)
        w.append(np.rint(f.aggressor * (f.trade_price - touch) / f.tick_size))
    w = np.concatenate(w)
    assert w.min() >= 0 and w.max() <= 60
    assert abs((w >= cfg.depth + 2).mean() - 0.1) < 0.005          # ordinary pen <= depth+1
    for x in (10, 20, 30):
        expect = 0.1 * math.exp(-(x - cfg.depth - 2) / 10.0)
        assert abs((w > x).mean() - expect) < 0.004, x
    # no wicks: every print within penetration depth+1 of the touch
    f0 = _frames(WorldConfig(group="perp", markets=100, steps=40, wick_prob=0.0, depth=3))
    for f in f0:
        touch = np.where(f.aggressor > 0, f.best_ask, f.best_bid)
        assert np.all(np.abs(f.trade_price - touch) <= 4 * f.tick_size + 1e-9)


def test_wicks_sweep_every_rung_at_or_beyond_print():
    """Existing fill rule untouched: a sell print fills every maker bid >= its price."""
    from quant_research.worldsim.fills import Orders
    from quant_research.worldsim.contracts import Intent
    cfg = WorldConfig(group="perp", markets=300, steps=40, trade_probability=1.0, wick_prob=1.0,
                      wick_max_ticks=40)
    frames = list(World(cfg, 4))
    f = next(x for x in frames[1:] if (x.aggressor == -1).sum() > 20)
    o = Orders(cfg.markets, 3)
    px = np.stack([f.best_bid - 0.01 * k for k in range(3)], axis=1)
    price = np.stack((px, np.full_like(px, np.nan) + 1e9), axis=1)
    price[:, 1] = f.best_ask[:, None] + 0.01 * np.arange(1, 4)
    intent = Intent(np.ones((cfg.markets, 2, 3), dtype=np.int8), price,
                    np.ones((cfg.markets, 2, 3), dtype=np.int64))
    o.submit(f, intent)
    filled = np.zeros((cfg.markets, 3), dtype=bool)
    for dq, p in o.fills(f):
        pass
    sold = f.aggressor == -1
    for k in range(3):
        should = sold & (f.trade_price <= px[:, k])
        got = (o.size[:, 0, k] == 0)
        # size 1 per rung, print size >= 1: every swept rung (up to volume) leaves size 0
        assert not (got & ~should).any()
    del filled


def _alignment(cfg, seed=11):
    frames = list(World(cfg, seed))
    direction = np.stack([f.aggressor for f in frames[1:-1]])
    moves = np.diff(np.stack([f.mid for f in frames[1:]]), axis=0)
    nz = moves != 0
    return np.mean(direction[nz] == np.sign(moves[nz]))


def test_informed_strength_scales_toxicity():
    kw = dict(markets=300, steps=40, common_weight=0, cluster_weight=0, perp_sigma=0.3,
              trade_probability=1, dislocation_sd_ticks=0, adverse_selection=0.8)
    lo = _alignment(WorldConfig(informed_strength=0.0, wick_toxicity_mult=0, **kw))
    mid = _alignment(WorldConfig(informed_strength=0.5, wick_prob=0.0, **kw))
    hi = _alignment(WorldConfig(informed_strength=1.0, wick_prob=0.0, **kw))
    assert abs(lo - 0.5) < 0.03
    assert abs(mid - (0.5 + 0.5 * 0.4)) < 0.03        # 0.8*0.5 = 0.4 informed
    assert hi == pytest.approx(0.5 + 0.5 * 0.8, abs=0.03)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("regime", ["slow", "fast"])
def test_prediction_bounds_and_settlement_with_wide_display(paired, regime):
    cfg = WorldConfig(group="prediction", markets=40, steps=120, regime=regime, paired=paired,
                      tick=0.01, spread_mean_ticks=30.0, spread_dispersion=1.0,
                      dislocation_sd_ticks=15.0, wick_prob=0.5, wick_max_ticks=45)
    frames = list(World(cfg, 8))
    for f in frames[:-1]:
        assert np.all((f.mid > 0) & (f.mid < 1))
        assert np.all((f.bid_px > 0) & (f.ask_px < 1))
        assert np.all((f.trade_price > 0) & (f.trade_price < 1))
        assert np.all(np.diff(f.bid_px, axis=1) < 0) and np.all(np.diff(f.ask_px, axis=1) > 0)
        if paired:
            np.testing.assert_allclose(f.mid[::2] + f.mid[1::2], 1, atol=1e-12)
            np.testing.assert_allclose(f.bid_px[::2] + f.ask_px[1::2], 1, atol=1e-12)
            np.testing.assert_allclose(f.trade_price[::2] + f.trade_price[1::2], 1, atol=1e-12)
    last = frames[-1]
    assert set(last.mid) <= {0.0, 1.0}
    assert np.array_equal(last.mid, last.outcome.astype(float))
    assert not last.trade_size.any()


def test_batch_matches_world_with_display_knobs():
    cfg = WorldConfig(group="prediction", markets=8, steps=10, wick_prob=0.3, paired=True)
    singles = [list(World(cfg, s)) for s in (1, 2, 3)]
    for t, bf in enumerate(WorldBatch(cfg, [1, 2, 3])):
        for k in range(3):
            sf = singles[k][t]
            for name in ("mid", "bid_px", "ask_px", "trade_price"):
                assert getattr(sf, name).tobytes() == getattr(bf, name)[k * 8:(k + 1) * 8].tobytes()


@pytest.mark.parametrize("bad", [
    dict(spread_mean_ticks=0), dict(spread_dispersion=-1), dict(spread_mean_ticks=float("nan")),
    dict(dislocation_sd_ticks=-1), dict(dislocation_halflife_ms=0), dict(wick_prob=1.5),
    dict(wick_scale_ticks=0), dict(wick_max_ticks=0), dict(wick_max_ticks=2.5),
    dict(informed_strength=1.2), dict(wick_toxicity_mult=-1),
    dict(dislocation_halflife_ms=float("inf"))])
def test_display_knob_validation(bad):
    with pytest.raises(ValueError):
        WorldConfig(**bad)


def test_default_tick_per_group():
    assert WorldConfig(group="perp").tick == 0.01
    assert WorldConfig(group="prediction").tick == 0.001
    assert WorldConfig(group="perp", tick=0.001).tick == 0.001
