import gzip
import json

import polars as pl
import pytest

from quant_research.prediction.polymarket import api, scores, wallet
from quant_research.prediction.polymarket.book import LiveBook, TokenBook, mid_if_valid, snapshot_bbo
from quant_research.prediction.polymarket.tape import (FileStatus, iter_market_quotes, quotes_frame,
                                                       read_records, stream_gzip_lines)


# ------------------------------------------------------------------ book
def test_snapshot_bbo_ignores_zero_size_and_order():
    bids = [{"price": "0.40", "size": "5"}, {"price": "0.45", "size": "0"}, {"price": "0.42", "size": "1"}]
    asks = [{"price": "0.60", "size": "2"}, {"price": "0.55", "size": "3"}]
    assert snapshot_bbo(bids, asks) == (0.42, 0.55)
    assert snapshot_bbo([], []) == (None, None)


@pytest.mark.parametrize("b,a,expect", [(0.4, 0.6, 0.5), (None, 0.6, None), (0.6, 0.4, None),
                                         (0.0, 0.5, None), (0.5, 1.0, None), (0.5, 0.5, 0.5)])
def test_mid_if_valid(b, a, expect):
    assert mid_if_valid(b, a) == expect


def test_tokenbook_missing_side_invalidates():
    tb = TokenBook()
    tb.apply_change({"asset_id": "x", "best_bid": "0.3", "best_ask": "0.4"})
    assert tb.quote("x")[2] == pytest.approx(0.35)
    tb.apply_change({"asset_id": "x", "best_bid": "0.3", "best_ask": ""})
    assert tb.quote("x") == (0.3, None, None)


def test_livebook_both_price_change_shapes():
    lb = LiveBook()
    lb.apply_book("t", {"bids": [{"price": "0.40", "size": "10"}], "asks": [{"price": "0.60", "size": "10"}]})
    assert lb.top("t")["mid"] == pytest.approx(0.5)
    assert lb.apply_price_change({"changes": [{"price": "0.45", "size": "5", "side": "BUY"}]}, "t") == ["t"]
    assert lb.top("t")["best_bid"] == 0.45
    lb.apply_price_change({"price_changes": [{"asset_id": "t", "price": "0.45", "size": "0", "side": "BUY"}]})
    assert lb.top("t")["best_bid"] == 0.40
    # no snapshot yet for "u": its changes are ignored, never turned into a half-empty book
    assert lb.apply_price_change({"price_changes": [{"asset_id": "u", "price": "0.7", "size": "1", "side": "SELL"}]}) == []
    assert lb.top("u")["best_ask"] is None
    lb.apply_book("u", {"bids": [], "asks": [{"price": "0.7", "size": "1"}]})
    assert lb.top("u")["best_ask"] == 0.7 and lb.top("u")["mid"] is None     # one-sided -> no mid


# ------------------------------------------------------------------ tape
def _rec(i, t, arr, data):
    return json.dumps({"seq_id": i, "type": t, "timestamp": arr - 5, "received_at": arr, "data": data}) + "\n"


def _tape(path, lines):
    path.write_bytes(gzip.compress("".join(lines).encode()))
    return path


LINES = [
    _rec(1, "book_snapshot", 1000, {"asset_id": "A", "bids": [{"price": "0.40", "size": "1"}], "asks": [{"price": "0.60", "size": "1"}]}),
    _rec(2, "price_change", 1100, {"changes": [{"asset_id": "A", "best_bid": "0.45", "best_ask": "0.60"}]}),
    _rec(3, "price_change", 1200, {"changes": [{"asset_id": "A", "best_bid": "0.45", "best_ask": "1"}]}),
]


def test_complete_tape(tmp_path):
    st = FileStatus(path="x")
    rows = list(read_records(_tape(tmp_path / "e.gz", LINES), status=st))
    assert len(rows) == 3 and st.ok and st.complete_records == 3 and not st.truncated_tail


def test_truncated_tape_keeps_complete_records_and_reports_tail(tmp_path):
    p = tmp_path / "e.gz"
    raw = gzip.compress(("".join(LINES) + '{"seq_id":4,"type":"price_ch').encode())
    p.write_bytes(raw[:-8])        # drop the gzip trailer: an unfinished recorder file
    st = FileStatus(path=str(p))
    rows = list(read_records(p, status=st))
    assert len(rows) == 3
    assert not st.ok and st.truncated_tail and st.discarded_bytes > 0 and st.corrupt is None


def test_quotes_frame_schema_clock_and_invalid_mid(tmp_path):
    d = tmp_path / "market" / "123" / "m1"
    d.mkdir(parents=True)
    _tape(d / "events.gz", LINES)
    sts = []
    df = quotes_frame([d / "events.gz"], statuses=sts)
    assert df.columns == ["arrival_ms", "exchange_ms", "asset_id", "bid", "ask", "mid", "tape"]
    assert df["arrival_ms"].to_list() == [1000, 1100, 1200]        # LOCAL arrival is the clock
    assert df["exchange_ms"].to_list() == [995, 1095, 1195]
    assert df["mid"].to_list()[0] == pytest.approx(0.5) and df["mid"].to_list()[1] == pytest.approx(0.525)
    assert df["mid"].to_list()[2] is None                          # ask == 1 -> not a valid mid
    assert df["tape"].unique().to_list() == ["m1"] and sts[0]["ok"]


# ------------------------------------------------------------------ api
def test_outcomes_map_by_name_not_order():
    m = {"outcomes": '["No", "Yes"]', "clobTokenIds": '["tn", "ty"]'}
    assert api.market_tokens(m) == {"tn": "NO", "ty": "YES"}
    assert api.outcome_for_name("Up") == "YES" and api.outcome_for_name("Draw") == "UNKNOWN" and api.outcome_for_name("") is None


def test_v2_pages_follow_cursor_and_stop():
    pages = {None: {"data": [1], "pagination": {"next_cursor": "c1"}},
             "c1": {"data": [2], "pagination": {"next_cursor": "c2"}},
             "c2": {"data": [3], "pagination": {}}}
    calls = []

    def get(url, params):
        calls.append(params.get("cursor"))
        return pages[params.get("cursor")]

    pm = api.Polymarket(get=get)
    assert [p["data"] for _, p in pm.v2_pages("/v2/x", {})] == [[1], [2], [3]]
    calls.clear()
    assert len(list(pm.v2_pages("/v2/x", {}, stop=lambda p: p["data"] == [2]))) == 2
    assert calls == [None, "c1"]


# ------------------------------------------------------------------ scores
def test_parse_and_classify():
    assert scores.parse_score("12-11|1-1|Bo3")["w"] == (1, 1)
    assert scores.parse_score("3-4")["simple"] == (3, 4)
    assert scores.classify_change("0-0", "1-0") == ("score", "home")
    assert scores.classify_change("1-0", "1-1") == ("score", "away")
    assert scores.classify_change("1-0", "1-0") is None
    assert scores.classify_change("10-8|0-0|Bo3", "11-8|0-0|Bo3") == ("kill", "home")
    assert scores.classify_change("10-8|0-0|Bo3", "0-0|0-1|Bo3") == ("mapwin", "away")
    assert scores.classify_change("0-0", "weird") == ("score", None)


def _sports(rows):
    return pl.DataFrame(rows, schema={"gameId": pl.Utf8, "t_mono": pl.Float64, "t_wall": pl.Float64, "score": pl.Utf8,
                                      "status": pl.Utf8, "ended": pl.Boolean, "leagueAbbreviation": pl.Utf8}, orient="row")


def test_detect_events_heartbeats_coalescing_and_end():
    rows = [("g", 0.0, 0.0, "0-0", "live", False, "nhl"),
            ("g", 10.0, 10.0, "0-0", "live", False, "nhl"),      # heartbeat: not an event
            ("g", 20.0, 20.0, "1-0", "live", False, "nhl"),      # home goal
            ("g", 20.1, 20.1, "2-0", "live", False, "nhl"),      # within 200 ms: coalesced
            ("g", 30.0, 30.0, "2-1", "live", False, "nhl"),      # away
            ("g", 40.0, 40.0, "2-1", "final", True, "nhl"),      # ends without a score change
            ("h", 0.0, 0.0, "0-0", "live", False, "nhl"),
            ("h", 5.0, 5.0, "1-0", "final", True, "nhl")]        # score change AND end in one frame
    ev, raw = scores.detect_events(_sports(rows))
    got = [(r["game_id"], r["etype"], r["direction"], r["t_mono"], r["n_coalesced_dropped"], r["also_ended"])
           for r in ev.iter_rows(named=True)]
    assert got == [("h", "score", "home", 5.0, 0, True), ("g", "score", "home", 20.0, 1, False),
                   ("g", "score", "away", 30.0, 0, False), ("g", "end", None, 40.0, 0, True)]
    assert raw.height == 5            # includes the coalesced change, for placebo exclusion


# ------------------------------------------------------------------ wallet
def _leg(cid, tok, opp, size, price, cost, fees, value, title="m"):
    return {"condition_id": cid, "token_id": tok, "opposite_token_id": opp, "current_size": size, "avg_price": price,
            "entry_cost_usdc": cost, "entry_fees_usdc": fees, "current_value": value, "title": title, "slug": cid,
            "redeemable": False}


def test_pair_conditions_requires_reciprocal_tokens():
    live = [_leg("c1", "a", "b", 100, 0.48, 48, 0.1, 50), _leg("c1", "b", "a", 80, 0.50, 40, 0.1, 40),
            _leg("c2", "x", "y", 10, 0.5, 5, 0, 5), _leg("c2", "z", "w", 10, 0.5, 5, 0, 5),   # not reciprocal
            _leg("c3", "p", "q", 7, 0.5, 3, 0, 4)]                                              # single leg
    df = wallet.pair_conditions(wallet.live_positions(live)).sort("condition_id")
    c1, c2, c3 = df.row(0, named=True), df.row(1, named=True), df.row(2, named=True)
    assert not c3["paired"] and c3["legs"] == 1 and c3["min_payout"] == 0
    assert c1["paired"] and c1["min_payout"] == 80 and c1["residual_shares"] == 20
    assert c1["min_terminal_pnl_before_fees"] == pytest.approx(80 - 88)
    assert c1["sum_avg_price"] == pytest.approx(0.98)
    assert c1["paired_component_gross"] == pytest.approx(80 * 0.02)
    assert not c2["paired"] and c2["min_payout"] == 0 and c2["sum_avg_price"] is None


def test_live_positions_filters_zero_and_redeemable():
    rows = [_leg("c", "a", "b", 1, .5, 1, 0, 0.0), _leg("c", "b", "a", 1, .5, 1, 0, 2.0) | {"redeemable": True},
            _leg("d", "e", "f", 1, .5, 1, 0, 3.0)]
    assert wallet.live_positions(rows)["condition_id"].to_list() == ["d"]


def test_activity_summary_window_dedupe_and_completeness():
    def tr(ts, cid, tok, side, cash):
        return {"timestamp": ts, "type": "TRADE", "condition_id": cid, "token_id": tok, "side": side, "usdc_size": cash}

    rows = [tr(100000, "c1", "a", "BUY", 10), tr(100000, "c1", "a", "BUY", 10),   # exact duplicate
            tr(99000, "c1", "b", "BUY", 20), tr(50000, "c2", "x", "SELL", 5),
            {"timestamp": 95000, "type": "REWARD", "condition_id": "c1", "token_id": None, "side": None, "usdc_size": 1}]
    s = wallet.activity_summary(rows, window_s=10000)
    assert s["trades"] == 2 and s["buys"] == 2 and s["buy_cash"] == 30           # dup removed, c2 outside window
    assert s["both_outcome_conditions"] == 1 and s["both_outcome_cash"] == 30
    assert s["complete_window"] is True
    assert wallet.activity_summary(rows[:3], window_s=10000)["complete_window"] is False


def test_pnl_delta_and_drawdown():
    pts = [{"timestamp": 0, "economic_pnl": 0.0, "fees": 0.0}, {"timestamp": 3600, "economic_pnl": 10.0, "fees": 1.0},
           {"timestamp": 7200, "economic_pnl": 4.0, "fees": 2.0}]
    d = wallet.pnl_delta(pts)
    assert d["delta"]["economic_pnl"] == 4.0 and d["delta"]["fees"] == 2.0 and d["max_drawdown_usd"] == 6.0
