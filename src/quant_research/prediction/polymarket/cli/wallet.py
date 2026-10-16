"""pm-wallet: collect a wallet's v2 data (bounded, with a source ledger) and write a descriptive report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_research.common.report import Summary
from quant_research.common.runs import RunFolder
from quant_research.prediction.polymarket import wallet as W


def cmd_pull(a):
    r = W.pull_wallet(a.user, a.out, activity_hours=a.hours)
    print(json.dumps(r, indent=1))
    if r["incomplete"]:
        print("WARNING: walks stopped by a page cap (NOT complete history):", ", ".join(r["incomplete"]))


def cmd_report(a):
    raw = Path(a.src) / "raw"
    run = RunFolder(a.out, a.tag, args=vars(a) | {"func": None})
    stats = json.loads((raw / "v2_stats.json").read_text())["data"]
    pts = json.loads((raw / "v2_pnl_month.json").read_text())["data"]["points"]
    pnl = W.pnl_delta(pts)
    live = W.live_positions(W.load_pages(raw, "v2_open"))
    conds = W.pair_conditions(live)
    act = W.activity_summary(W.load_pages(raw, "v2_activity"), window_s=a.hours * 3600)
    run.write_json("metrics.json", {"wallet": stats.get("proxy_wallet"), "pnl_month": pnl,
                                    "live_conditions": conds.to_dicts(), "activity": act})
    run.write_frame("live_conditions", conds)
    run.write_frame("live_positions", live)
    d = pnl["delta"]
    s = Summary(f"Wallet snapshot {stats.get('proxy_wallet')}",
                verdict="Descriptive snapshot only - no strategy verdict.",
                evidence="description (API aggregates; not a replication or a profit claim)")
    s.table("PnL over the month window (USD)", [{"component": k, "change": v} for k, v in d.items()])
    s.section("Window", f"{pnl['start']} -> {pnl['end']}; max drawdown on the sample grid: ${pnl['max_drawdown_usd']:,.0f} "
                        "(a USD figure; capital was not established, so this is not a percentage).")
    s.table("Live conditions (top 15 by mark)", conds[:15],
            ["title", "paired", "mark", "cost", "min_terminal_pnl_after_entry_fees", "residual_shares"])
    s.section("Activity", "\n".join(f"- **{k}**: {v}" for k, v in act.items() if k != "types"))
    s.coverage = {"live legs": live.height, "activity rows in window": act["rows"],
                  "activity window complete": act["complete_window"]}
    s.caveat("Position PnL, trading PnL and fee components come from different API series; do not assume fees reconcile a gap.")
    s.caveat("Role (maker/taker) is not established here; a fee difference is a diagnostic, not a label.")
    run.write_text("summary.md", s.render())
    run.finish()
    print(run.dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("user"); p.add_argument("--out", required=True)
    p.add_argument("--hours", type=float, default=24); p.set_defaults(func=cmd_pull)
    r = sub.add_parser("report"); r.add_argument("--src", required=True, help="dir containing raw/ from `pull`")
    r.add_argument("--out", required=True, help="run root"); r.add_argument("--tag"); r.add_argument("--hours", type=float, default=24)
    r.set_defaults(func=cmd_report)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
