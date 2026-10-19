"""Reliability estimators: planted-statistic tests (worldsim reliability spec v1, W0)."""
import numpy as np
import polars as pl

from quant_research.perp.hl_forensics import reliability as R


def _bbo(coin, t_ms, mids):
    half = 0.5
    return pl.DataFrame({"coin": [coin] * len(t_ms), "t_ms": t_ms,
                         "bid": [m - half for m in mids], "ask": [m + half for m in mids],
                         "mid": mids})


def test_zero_move_share_exact():
    # 10 slots at 500 ms; mid changes only at slot 4 -> zero share 8/9
    t = [i * 500.0 for i in range(11)]
    m = [100.0] * 5 + [101.0] * 6
    df = _bbo("A", t, m)
    z = R.zero_move_share(df, 500.0)
    assert z["zero_share"][0] == 8 / 9
    assert z["n_slots"][0] == 9


def test_mid_changes_per_hour_and_alternation():
    # 5 changes over 2 hours, pattern + - + - (all alternate)
    t = [0.0, 3.6e6 * 0.2, 3.6e6 * 0.4, 3.6e6 * 0.6, 3.6e6 * 0.8, 3.6e6 * 1.0]
    m = [100.0, 101.0, 100.0, 101.0, 100.0, 101.0]
    df = _bbo("A", t, m)
    ch = R.mid_changes_per_hour(df)
    assert abs(ch["changes_per_hour"][0] - 5 / 1.0) < 1e-9
    al = R.alternation_ratio(df)
    assert al["alt_ratio"][0] == 1.0            # all 4 sign pairs alternate
    assert abs(al["eta_uz"][0] - 0.0) < 1e-9    # N_cont=0 -> eta = 0/(2*4) = 0


def test_tick_bps():
    t = [0.0, 1000.0, 2000.0]
    df = pl.DataFrame({"coin": ["A"] * 3, "t_ms": t,
                       "bid": [100.00, 100.50, 101.00], "ask": [100.50, 101.00, 101.50],
                       "mid": [100.25, 100.75, 101.25]})
    tk = R.tick_bps(df)
    assert abs(tk["tick_bps"][0] - 0.5 / 100.75 * 1e4) < 1e-6


def test_return_ladder_shapes_and_scale():
    rng = np.random.default_rng(1)
    n = 2000
    m = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    df = _bbo("A", [i * 1000.0 for i in range(n)], list(m))
    df = pl.concat([df, df.with_columns(pl.lit("B").alias("coin"))])   # n>1 coins
    rl = R.return_ladder(df, horizons_s=(10, 60))
    assert set(rl.columns) >= {"coin", "horizon_s", "sd_bps", "q001", "q99", "skew", "kurt"}
    assert rl["coin"].n_unique() == 2
    for h, g in rl.group_by("horizon_s"):
        sd = g["sd_bps"][0]
        assert sd > 0


def test_jump_detect_and_revert():
    rng = np.random.default_rng(3)
    t = [float(i * 1000) for i in range(400)]          # 400 s on a 1 s grid
    m = list(100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 400))))  # 10 bps noise
    m[200] = m[199] * 1.01                             # one ~100 bps jump, kept
    df = _bbo("A", t, m)
    js = R.jump_stats(df, k_sig=4.0)
    assert js["n_jumps"][0] >= 1
    assert js["median_jump_bps"][0] > 0


def test_corr60_identical_series():
    rng = np.random.default_rng(2)
    n = 120
    m = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    t = [i * 60000.0 for i in range(n)]
    df = pl.concat([_bbo("A", t, list(m)), _bbo("B", t, list(m))])
    c = R.corr60(df, min_obs=30)
    assert abs(c["corr"][0] - 1.0) < 1e-9


def test_wallet_dodge_distance():
    bbo = pl.DataFrame({"coin": ["A", "A"], "t_ms": [0.0, 1000.0],
                        "bid": [100.0, 100.0], "ask": [100.2, 100.2], "mid": [100.1, 100.1]})
    orders = pl.DataFrame({"coin": ["A", "A"], "t_ms": [500.0, 500.0],
                           "side": ["B", "A"], "px": [99.9, 100.3], "ev": ["cancel", "cancel"]})
    od = R.wallet_dodge(orders, bbo)
    d = sorted(od["d_touch_bps"].to_list())
    assert abs(d[0] - 9.99) < 0.1        # bid 9.99 bps behind bid touch 100.0
    assert abs(d[1] - 9.99) < 0.1        # ask 9.99 bps behind ask touch 100.2


def test_summarize():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    s = R.summarize(df, ["x"])["x"]
    assert s["median"] == 2.5 and s["n"] == 4 and s["p25"] < s["p75"]
