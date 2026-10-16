"""Wallet research: bounded v2 collection with a source ledger, and pure accounting helpers.

Rules this module encodes (each one produced a wrong number before):
  * Persist every raw page with its request URL and retrieval time (sources.jsonl).
  * Bound collection by the question. Open positions sorted by current value stop once
    the zero-value tail is reached; activity follows keyset cursors past a fixed cutoff.
    A walk stopped by max_pages is NOT complete history.
  * Only reciprocal opposite tokens of ONE condition are a complementary pair, and
    only they get the $1 minimum-payout floor. Correlated sub-markets are not legs.
  * Report cash turnover as cash turnover: profit / turnover is not a return on capital.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from quant_research.common.tables import frame_from_records

from .api import Polymarket


def iso(t) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


# ---------------------------------------------------------------- collection
class SourceLedger:
    """sources.jsonl: one line per fetched page - name, url, params, retrieval time."""

    def __init__(self, path):
        self.path = Path(path)

    def add(self, name, info):
        with self.path.open("a") as f:
            f.write(json.dumps({"name": name, **info}) + "\n")


def _save(raw: Path, ledger: SourceLedger, name: str, info: dict, page) -> None:
    (raw / f"{name}.json").write_text(json.dumps(page))
    ledger.add(name, info)


def pull_wallet(user: str, out_dir, *, api: Polymarket | None = None, activity_hours: float = 24,
                activity_max_pages: int = 40, position_pages: int = 8, workers: int = 6) -> dict:
    """Collect stats / PnL / leaderboard / open + closed positions / activity into `<out_dir>/raw`.

    Returns {'raw': path, 'incomplete': [names of walks stopped by a page cap]}.
    """
    api = api or Polymarket()
    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ledger = SourceLedger(out / "sources.jsonl")
    base = "https://data-api.polymarket.com"

    singles = {
        "v2_stats": ("/v2/user-stats", {"user": user}),
        "v2_pnl_all": ("/v2/user-pnl", {"user": user, "interval": "all", "fidelity": "1d"}),
        "v2_pnl_month": ("/v2/user-pnl", {"user": user, "interval": "1m", "fidelity": "1h"}),
        "v2_leader_month": ("/v2/leaderboard", {"user": user, "time_period": "month"}),
    }

    def one(item):
        name, (path, params) = item
        page = api.get(base + path, params)
        _save(raw, ledger, name, {"url": base + path, "params": params,
                                  "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}, page)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, singles.items()))

    incomplete = []

    def walk(name, path, params, max_pages, stop=None):
        last = None
        for i, (info, page) in enumerate(api.v2_pages(path, params, max_pages=max_pages, stop=stop)):
            _save(raw, ledger, f"{name}_{i}", info, page)
            last = page
        if last is not None and (last.get("pagination") or {}).get("next_cursor") and (stop is None or not stop(last)):
            incomplete.append(name)

    walk("v2_open", "/v2/positions", {"user": user, "limit": 1000, "status": "OPEN", "filter_amount": 0,
                                       "sort_by": "CURRENT_VALUE"}, position_pages,
         stop=lambda p: bool(p.get("data")) and p["data"][-1].get("current_value", 1) <= 0)
    walk("v2_closed", "/v2/positions", {"user": user, "limit": 1000, "status": "CLOSED",
                                         "sort_by": "TIMESTAMP"}, position_pages)

    first_page = api.get(base + "/v2/activity", {"user": user, "limit": 1000, "sort_direction": "DESC"})
    rows = first_page.get("data") or []
    cutoff = (rows[0]["timestamp"] - activity_hours * 3600) if rows else 0
    walk("v2_activity", "/v2/activity", {"user": user, "limit": 1000, "sort_direction": "DESC"},
         activity_max_pages,
         stop=lambda p: bool(p.get("data")) and min(x["timestamp"] for x in p["data"]) < cutoff)
    return {"raw": str(raw), "incomplete": incomplete}


def load_pages(raw_dir, name: str) -> list:
    """All saved pages of a paginated walk, concatenated in order (name_0, name_1, ...)."""
    raw = Path(raw_dir)
    rows = []
    for f in sorted(raw.glob(f"{name}_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
        rows += json.loads(f.read_text()).get("data", [])
    return rows


# ---------------------------------------------------------------- accounting (DataFrames)
def frame(rows) -> pl.DataFrame:
    """Rows (list of dicts) or a DataFrame -> DataFrame. Schema inferred from ALL rows."""
    return rows if isinstance(rows, pl.DataFrame) else frame_from_records(rows)


def pnl_delta(points) -> dict:
    """First/last/delta of every numeric component of a PnL series, plus max drawdown of
    `economic_pnl` on the grid it was sampled on (a USD figure, not a return)."""
    df = frame(points)
    num = [c for c, t in df.schema.items() if t.is_numeric() and c != "timestamp"]
    first, last = df.row(0, named=True), df.row(-1, named=True)
    dd = 0.0
    if "economic_pnl" in df.columns:
        dd = df.select((pl.col("economic_pnl").cum_max() - pl.col("economic_pnl")).max()).item() or 0.0
    return {"start": iso(first["timestamp"]), "end": iso(last["timestamp"]),
            "delta": {c: last[c] - first[c] for c in num}, "max_drawdown_usd": dd}


def live_positions(open_rows) -> pl.DataFrame:
    df = frame(open_rows)
    cond = pl.col("current_value") > 0
    if "redeemable" in df.columns:
        cond = cond & ~pl.col("redeemable").fill_null(False)
    return df.filter(cond)


def pair_conditions(live) -> pl.DataFrame:
    """One row per condition. `paired` needs exactly two legs that are each other's opposite
    token; only paired conditions get the $1 minimum-payout floor (min of the two sizes)."""
    g = frame(live).group_by("condition_id", maintain_order=True).agg(
        pl.col("title").first(), pl.col("slug").first(), pl.len().alias("legs"),
        pl.col("entry_cost_usdc").sum().alias("cost"), pl.col("entry_fees_usdc").sum().alias("fees"),
        pl.col("current_value").sum().alias("mark"),
        pl.col("current_size").min().alias("q_min"), pl.col("current_size").max().alias("q_max"),
        pl.col("avg_price").sum().alias("sum_avg_price"),
        pl.col("token_id").alias("_tok"), pl.col("opposite_token_id").alias("_opp"))
    g_ = lambda c, i: pl.col(c).list.get(i, null_on_oob=True)     # single-leg conditions have no 2nd leg
    paired = ((pl.col("legs") == 2) & (g_("_tok", 0) == g_("_opp", 1)) & (g_("_tok", 1) == g_("_opp", 0))).fill_null(False)
    return (g.with_columns(paired.alias("paired"))
            .with_columns(pl.when(pl.col("paired")).then(pl.col("q_min")).otherwise(0.0).alias("min_payout"))
            .with_columns((pl.col("min_payout") - pl.col("cost")).alias("min_terminal_pnl_before_fees"),
                          (pl.col("min_payout") - pl.col("cost") - pl.col("fees")).alias("min_terminal_pnl_after_entry_fees"),
                          (pl.col("q_max") - pl.col("min_payout")).alias("residual_shares"),
                          pl.when(pl.col("paired")).then(pl.col("min_payout") * (1 - pl.col("sum_avg_price")))
                          .otherwise(0.0).alias("paired_component_gross"))
            .with_columns(pl.when(pl.col("paired")).then(pl.col("sum_avg_price")).otherwise(None).alias("sum_avg_price"))
            .drop("_tok", "_opp", "q_min", "q_max").sort("mark", descending=True))


def dedup_rows(rows: list[dict]) -> list[dict]:
    return list({json.dumps(x, sort_keys=True): x for x in rows}.values())


def activity_summary(rows, *, window_s: float = 86400, categorize=None) -> dict:
    """Exact trailing-window trade summary from (possibly over-fetched) activity rows.

    `categorize(row_dict) -> str` is optional (slug rules are study-specific, so none are
    shipped). `complete_window` says whether the walk reached the window start - if False
    the counts are a lower bound.
    """
    raw = rows.to_dicts() if isinstance(rows, pl.DataFrame) else rows
    df = frame(dedup_rows(raw))
    end = df["timestamp"].max()
    start = end - window_s
    complete = df["timestamp"].min() <= start
    df = df.filter(pl.col("timestamp").is_between(start, end))
    t = df.filter(pl.col("type") == "TRADE")
    both = (t.group_by("condition_id").agg(pl.col("token_id").n_unique().alias("n"))
            .filter(pl.col("n") == 2)["condition_id"])
    cash = t["usdc_size"]
    side = lambda s: t.filter(pl.col("side") == s)
    out = {"start": iso(start), "end": iso(end), "rows": df.height, "complete_window": complete,
           "types": dict(df.group_by("type").len().iter_rows()),
           "trades": t.height, "buys": side("BUY").height, "sells": side("SELL").height,
           "buy_cash": side("BUY")["usdc_size"].sum(), "sell_cash": side("SELL")["usdc_size"].sum(),
           "median_fill_cash": cash.median(), "p90_fill_cash": cash.sort()[int(0.9 * len(cash))] if len(cash) else None,
           "max_fill_cash": cash.max(),
           "conditions": t["condition_id"].n_unique(), "both_outcome_conditions": both.len(),
           "both_outcome_cash": t.filter(pl.col("condition_id").is_in(both.implode()))["usdc_size"].sum()}
    if categorize:
        t2 = t.with_columns(pl.struct(t.columns).map_elements(categorize, return_dtype=pl.Utf8).alias("category"))
        out["categories"] = {k: {"fills": n, "cash": c} for k, n, c in
                             t2.group_by("category").agg(pl.len(), pl.col("usdc_size").sum()).iter_rows()}
    return out
