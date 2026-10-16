import polars as pl
import pytest

from quant_research.common.eventstudy import coalesce, measure, placebo, summarize

Q = pl.DataFrame({
    "asset_id": ["A", "A", "A", "A"],
    "t_mono": [1.0, 2.0, 3.0, 4.0],
    "mid": [0.50, 0.52, None, 0.60],      # t=3.0 is an invalid (one-sided) state
})


def run(rows, **kw):
    t = pl.DataFrame(rows, schema={"asset_id": pl.Utf8, "t_mono": pl.Float64}, orient="row")
    return measure(t, Q, horizons_s=(1, 2, 5), lookbacks_s=(0.5,), **kw)


def test_pre_is_strictly_before_and_moves_use_valid_values():
    r = run([("A", 2.0)]).row(0, named=True)
    assert r["pre"] == 0.50                  # the t=2.0 update is not visible at t=2.0
    assert r["pre_age_ms"] == 1000.0
    assert r["pre_latest_valid"] is True
    assert r["move_1000"] == pytest.approx(2.0)    # t+1=3.0 invalid -> latest valid is 0.52
    assert r["move_2000"] == pytest.approx(10.0)   # t+2=4.0 -> 0.60
    assert r["move_5000"] is None                  # 7.0 > last quote time: censored, not extrapolated
    assert r["prior_500"] == pytest.approx(0.0)


def test_latest_row_invalid_is_flagged_and_pre_skips_it():
    r = run([("A", 3.5)]).row(0, named=True)
    assert r["pre"] == 0.52 and r["pre_latest_valid"] is False


def test_no_history_and_unknown_asset_give_null_not_zero():
    out = run([("A", 1.0), ("B", 2.0)])
    assert out["pre"].to_list() == [None, None]
    assert out["move_1000"].to_list() == [None, None]


def test_row_order_preserved_and_react_window_counts_all_frames():
    out = run([("A", 3.5), ("A", 2.9), ("A", 2.0)])
    assert out["t_mono"].to_list() == [3.5, 2.9, 2.0]
    by_t = {r["t_mono"]: r for r in out.iter_rows(named=True)}
    assert by_t[2.9]["n_updates_react"] == 1      # the invalid t=3.0 frame still counts as an update
    assert by_t[2.0]["n_updates_react"] == 0


def test_t_end_override_censors():
    r = run([("A", 2.0)], t_end=3.5).row(0, named=True)
    assert r["move_2000"] is None and r["move_1000"] is not None


def test_coalesce_keeps_first_of_burst_per_group():
    ev = [{"t_mono": 0.0, "g": 1}, {"t_mono": 0.1, "g": 1}, {"t_mono": 0.15, "g": 2}, {"t_mono": 0.5, "g": 1}]
    kept, dropped = coalesce(ev, 0.2, group=lambda e: e["g"])
    assert [(e["t_mono"], e["g"]) for e in kept] == [(0.0, 1), (0.15, 2), (0.5, 1)]
    assert dropped == 1 and kept[0]["n_coalesced_dropped"] == 1


def test_placebo_excludes_event_neighbourhoods_and_is_seeded():
    q = pl.DataFrame({"asset_id": ["A"] * 100, "t_mono": [float(i) for i in range(100)], "mid": [0.5] * 100})
    p1 = placebo(q, [50.0], 10, t_lo=0, t_hi=99, excl_s=10, seed=1)
    assert p1.height == 10
    assert all(abs(t - 50.0) >= 10 for t in p1["t_mono"])
    assert p1.equals(placebo(q, [50.0], 10, t_lo=0, t_hi=99, excl_s=10, seed=1))


def test_summarize_cluster_and_groups():
    df = pl.DataFrame({"m": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "g": ["a", "a", "b", "b", "c", "c"], "s": ["x", "x", "x", "y", "y", "y"]})
    out = summarize(df, ["m"], by="s", cluster="g", n_boot=200)
    assert set(out["s"].to_list()) == {"x", "y"}
    assert out.filter(pl.col("s") == "x")["n"].item() == 3
