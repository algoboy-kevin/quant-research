"""Slow calibration smoke for the event-time flow model (small version of the report scripts).

Asserts the directional G1 signature (far fills mark better than near fills, far positive), the
G2 order of magnitude, and the no-stale-fill invariant on a SMALL calibrated world at the
wallet cadence (dt = 5 s). The numeric gates live in the calibration report
(scratch/worldsim-flowfix/calib2.py).
"""
import pytest

import flow_calibration as fc


@pytest.mark.slow
def test_flow_calibration_smoke_g1_direction_g2_magnitude_and_edge_invariant():
    cfg = fc.hl_config(markets=10, steps=int(2 * 3.6e6 / fc.DT_MS))
    r = fc.summarize_fills(fc.run_fills(cfg, seeds=range(4242, 4244), jobs=1))
    near, mid, far = (x[0] for x in r["mk60"][:3])
    assert r["n_b"][0] > 20 and r["n_b"][2] > 20, r["n_b"]
    assert far > near, (near, far)                       # G1 direction
    assert far > 0                                       # far fills keep a net edge after the move
    assert 1.0 < r["fills_per_coin_hour"] < 30.0, r["fills_per_coin_hour"]   # order of magnitude
    assert r["share_edge_lt_minus1tick"] == 0.0           # no stale fills
