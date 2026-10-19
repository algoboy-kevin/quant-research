"""Calibration gates for the display layer (small version of fillmodel_calibration.py).

Out-of-sample seed (the defaults were tuned on seed 5). Run the full-size report with
`python tests/fillmodel_calibration.py --worlds 14 --steps 2000`.
"""
import pytest

import fillmodel_calibration as cal


@pytest.mark.skip(reason="retired by the flow-fix model change: pre-flow single-print gates; see test_flow_calibration_smoke.py and the flow-fix calibration report")
@pytest.mark.slow
def test_hl_default_regime_gates():
    cfg = cal.hl_config(markets=20, steps=1000)
    r = cal.summarize(cfg, cal.run(cfg, worlds=6, seed=77, jobs=3))
    cal.report(cfg, r, "small calibration (tests)")
    assert 8 <= r["spread_mean_bps"] <= 20
    assert r["spread_var_ticks"] > r["spread_mean_ticks"]
    assert -2.7 <= r["near_markout"] <= -0.7, r["near_markout"]           # -1.7 +/- 1.0
    assert r["far_n"] > 500 and 0 < r["far_markout"] < 20, r["far_markout"]   # positive, target +5..+15
    assert 0.005 < r["print_frac_gt20bps"] < 0.12
