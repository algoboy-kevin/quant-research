import json
import math

import polars as pl
import pytest

from quant_research.common import stats
from quant_research.common.eventstudy import measure
from quant_research.perp.hl_forensics import fills as F
from quant_research.perp.hl_forensics import ladder as L
from quant_research.perp.hl_forensics import requote as R
from quant_research.perp.hl_forensics.wallet_capture import load_wallet_capture


# ------------------------------------------------------------------ helpers
def bbo_msg(coin, t, bid, ask):
    return {"rx": int(t * 1000), "msg": {"channel": "bbo", "data": {"coin": coin, "time": int(t * 1000),
            "bbo": [{"px": str(bid), "sz": "1", "n": 1} if bid else None, {"px": str(ask), "sz": "1", "n": 1} if ask else None]}}}


def fill_msg(t, coin, side, px, sz, tid, crossed=False, fee="0.01", oid=1, snapshot=False, start="0"):
    return {"rx": int(t * 1000), "msg": {"channel": "userFills", "data": {"isSnapshot": snapshot, "fills": [
        {"coin": coin, "px": str(px), "sz": str(sz), "side": side, "time": int(t * 1000), "startPosition": start,
         "dir": "Open Long", "closedPnl": "0", "oid": oid, "crossed": crossed, "fee": fee, "tid": tid}]}}}


def ou_msg(t, items):
    return {"rx": int(t * 1000), "msg": {"channel": "orderUpdates", "data": [
        {"order": {"coin": c, "side": s, "limitPx": str(px), "sz": "1", "oid": oid, "timestamp": int(t * 1000), "origSz": "1", "cloid": "0x1"},
         "status": st, "statusTimestamp": int(t * 1000)} for (c, s, px, oid, st) in items]}}


def write(tmp_path, msgs):
    p = tmp_path / "cap.jsonl"
    p.write_text("\n".join(json.dumps(m) for m in msgs) + "\n")
    return p


@pytest.fixture
def cap(tmp_path):
    msgs = [{"rx": 1000, "msg": {"channel": "__start__"}},
            bbo_msg("X", 100.0, 99.0, 101.0),                      # mid 100
            ou_msg(100.2, [("X", "B", 99.0, 11, "open")]),
            fill_msg(100.5, "X", "B", 99.0, 10, tid=1, oid=11),    # maker bid fill at 99 (1% from mid = 100 bps)
            fill_msg(100.5, "X", "B", 99.0, 10, tid=1, oid=11),    # duplicate tid
            fill_msg(100.6, "X", "A", 101.0, 1, tid=2, crossed=True),   # taker
            fill_msg(50.0, "X", "B", 90.0, 1, tid=3, snapshot=True),    # snapshot history
            bbo_msg("X", 160.4, 97.0, 99.0),                       # mid 98 at +60 s
            bbo_msg("X", 400.0, 97.0, 99.0)]
    return load_wallet_capture(write(tmp_path, msgs))


# ------------------------------------------------------------------ loader
def test_loader_dedupes_skips_snapshot_flags_taker(cap):
    f = cap["fills"]
    assert f.height == 2 and f["tid"].to_list() == [1, 2]
    assert f.filter(pl.col("crossed")).height == 1
    assert cap["bbo"].height == 3 and cap["orders"].height == 1
    assert cap["meta"]["channels"]["bbo"] == 3 and cap["meta"]["coins"] == 1


def test_one_sided_bbo_has_null_mid(tmp_path):
    c = load_wallet_capture(write(tmp_path, [bbo_msg("X", 1.0, 99, None), bbo_msg("X", 2.0, 99, 101)]))
    assert c["bbo"]["mid"].to_list() == [None, 100.0]


# ------------------------------------------------------------------ markouts
def test_fill_markouts_signs_and_net(cap):
    m = F.fill_markouts(cap["fills"], cap["bbo"], cap["orders"], horizons_s=(60,), t_end=400.0)
    assert m.height == 1                                    # maker only
    r = m.row(0, named=True)
    assert r["pre"] == 100.0 and r["edge_bps"] == pytest.approx(100.0)         # bought 1% below mid
    assert r["dfill_bps"] == pytest.approx(100.0) and r["half_bps"] == pytest.approx(100.0)
    assert r["mk_60000"] == pytest.approx((98 - 99) / 99 * 1e4)                # mid fell: loses vs the fill price
    assert r["net_60000"] == pytest.approx(r["mk_60000"] - r["fee_bps"])
    assert r["fee_bps"] == pytest.approx(0.01 / 990 * 1e4)
    assert r["dplace_bps"] == pytest.approx(100.0)          # order opened at 99 vs mid 100


def test_stale_pre_and_censored_horizon_are_null_not_zero(cap):
    m = F.fill_markouts(cap["fills"], cap["bbo"], horizons_s=(60, 300), max_age_s=0.3, t_end=400.0)
    assert m.height == 0                                    # pre-state 0.5 s old > 0.3 s allowed -> dropped
    m = F.fill_markouts(cap["fills"], cap["bbo"], horizons_s=(60, 300), t_end=200.0)
    assert m["mk_60000"].item() is not None and m["mk_300000"].item() is None   # +300 s runs past the capture


def test_measure_carry_and_max_age():
    q = pl.DataFrame({"k": ["a", "a"], "t": [1.0, 10.0], "v": [1.0, 2.0], "w": [5.0, 6.0]})
    t = pl.DataFrame({"k": ["a", "a"], "t": [2.0, 20.0]})
    out = measure(t, q, t="t", key="k", value="v", horizons_s=(1,), lookbacks_s=(), react_s=None, carry=("w",), max_age_s=5)
    assert out["pre"].to_list() == [1.0, None] and out["pre_w"].to_list() == [5.0, None]   # 2nd pre is 10 s old
    assert "n_updates_react" not in out.columns and "prior_500" not in out.columns


# ------------------------------------------------------------------ tables
def synth(n_coins=6, per=5):
    rows = []
    for c in range(n_coins):
        for i in range(per):
            far = i >= 3
            rows.append({"coin": f"c{c}", "t": 1000.0 * c + 700.0 * i, "notional": 1000.0, "dfill_bps": 30.0 if far else 1.0,
                         "net_60000": (20.0 if far else -1.0) + (c - 2.5)})
    return pl.DataFrame(rows)


def test_bucket_table_weighted_mean_and_cluster_ci():
    d = synth()
    t = F.bucket_table(d, "dfill_bps", [0, 3, float("inf")], value="net_60000", n_boot=500)
    assert t["fills"].to_list() == [18, 12] and t["coins"].to_list() == [6, 6]
    assert t["wmean_bps"].to_list() == [pytest.approx(-1.0), pytest.approx(20.0)]
    assert t["net_usd"][1] == pytest.approx(20.0 * 12 * 1000 / 1e4)
    assert t["ci_lo"][1] <= 20.0 <= t["ci_hi"][1]


def test_concentration_breadth_and_time_blocks():
    d = synth()
    bc = F.net_by_coin(d, "net_60000")
    assert bc["net_usd"].is_sorted(descending=True) and bc.height == 6
    c = F.concentration(bc, ks=(1, 6))
    assert c[1]["share_of_total"] == pytest.approx(1.0) and 0 < c[0]["share_of_total"] < 1
    b = F.breadth(bc, sizes=(1, 3, 5), exclude_top=1, draws=400)
    allp = b.filter(pl.col("universe") == "all")["p_pos"].to_list()
    assert allp[-1] >= allp[0]                              # more coins -> positive sum more often
    assert set(b["universe"]) == {"all", "ex-top1"}
    tb = F.time_blocks(d, pl.col("dfill_bps") >= 3, "net_60000", block_s=1000)
    assert tb["net_usd"].is_sorted(descending=True)


def test_clustered_slice_needs_three_clusters():
    d = synth(n_coins=2)
    assert F.clustered_slice(d, pl.lit(True), "net_60000")["ci_lo"] is None


# ------------------------------------------------------------------ stats
def test_cluster_ratio_ci_point_is_weighted_mean():
    g = {"a": (10.0, 1.0), "b": (30.0, 3.0), "c": (0.0, 1.0), "d": (8.0, 2.0)}
    pt, lo, hi = stats.cluster_ratio_ci(g, n_boot=300)
    assert pt == pytest.approx(48 / 7) and lo <= pt <= hi
    assert stats.cluster_ratio_ci({"a": (1, 1)}) is None


def test_subset_resample_exclude_and_sizes():
    v = {f"c{i}": x for i, x in enumerate([10, 5, 1, -1, -2, -3])}
    r = stats.subset_resample(v, (1, 6, 7), draws=300)
    assert [x["n"] for x in r] == [1, 6]                    # 7 > pool: skipped
    assert r[1]["mean"] == pytest.approx(10) and r[1]["p_pos"] == 1.0
    ex = stats.subset_resample(v, (3,), draws=300, exclude=("c0", "c1"))
    assert ex[0]["mean"] < r[1]["mean"]


# ------------------------------------------------------------------ requote
def test_replace_events_and_band_summary():
    # bid ladder rung: market bid 100 / ask 101. Old rung 20 below touch (80), new rung 25 below (75): ln(25/20)
    bbo = pl.DataFrame({"coin": ["X"], "t": [10.0], "bid": [100.0], "ask": [101.0], "bid_sz": [1.0], "ask_sz": [1.0], "mid": [100.5]})
    orders = pl.DataFrame({"t": [20.0, 20.0, 21.0], "coin": ["X"] * 3, "side": ["B", "B", "B"], "oid": [1, 2, 3],
                           "px": [80.0, 75.0, 70.0], "sz": [1.0] * 3, "orig_sz": [1.0] * 3,
                           "status": ["canceled", "open", "open"], "cloid": ["a"] * 3, "placed_t": [20.0, 20.0, 21.0]})
    rep = R.replace_events(orders, bbo, {"X": 2})
    assert rep.height == 1                                   # the lone open at t=21 is not a replace
    r = rep.row(0, named=True)
    assert r["d_old_touch"] == pytest.approx(20.0) and r["d_new_touch"] == pytest.approx(25.0)
    assert r["lnr_touch"] == pytest.approx(math.log(25 / 20))
    s = R.band_summary(rep, min_ticks=8).filter(pl.col("anchor") == "touch").row(0, named=True)
    assert s["n"] == 1 and s["share_inside"] == 0.0 and s["edge_grow"] == pytest.approx(math.log(1.25))


def test_expected_edges_for_quarter_band():
    grow, shrink = R.deadband_edges_expected(0.25)
    assert grow == pytest.approx(0.2877, abs=1e-4) and shrink == pytest.approx(0.2231, abs=1e-4)


# ------------------------------------------------------------------ ladder / skew
def test_ladder_table_geometric_gaps_and_skew():
    o = pl.DataFrame({"coin": ["X"] * 6, "side": ["B", "B", "B", "A", "A", "A"],
                      "px": [99.0, 98.0, 97.0, 101.0, 102.0, 103.0]})
    t = L.ladder_table(o, pl.DataFrame({"coin": ["X"], "mid": [100.0]}))
    r = t.row(0, named=True)
    assert r["rung1_bid_bps"] == pytest.approx(100.0) and r["rung1_ask_bps"] == pytest.approx(100.0)
    assert r["own_mid"] == 100.0 and r["skew_bps"] == pytest.approx(0.0)
    gaps = [(102 / 101 - 1) * 1e4, (103 / 102 - 1) * 1e4, (99 / 98 - 1) * 1e4, (98 / 97 - 1) * 1e4]
    assert r["gap_bps"] == pytest.approx(stats.median(gaps))             # geometric gaps p_k/p_{k+1}-1, both sides pooled
    assert r["inner_over_gap"] == pytest.approx(100.0 / stats.median(gaps))


def test_skew_uses_signed_inventory_and_end_positions():
    ladders = pl.DataFrame({"coin": ["a", "b", "c", "d"], "mid": [10.0] * 4, "skew_bps": [-30.0, -10.0, 10.0, 30.0]})
    pos = pl.DataFrame({"coin": ["a", "b", "c", "d"], "position": [3e5, 1e5, -1e5, -3e5]})   # long inventory -> quotes tilt down
    r = L.skew_vs_inventory(ladders, pos)
    assert r["n"] == 4 and r["corr"] == pytest.approx(-0.0, abs=1.0) and r["corr"] < -0.99
    assert r["slope_bps_per_1m"] < 0
    fills = pl.DataFrame({"t": [1.0, 2.0], "coin": ["a", "a"], "side": ["B", "A"], "sz": [5.0, 2.0], "start_position": [10.0, 15.0]})
    assert L.end_positions(fills)["position"].item() == pytest.approx(13.0)
