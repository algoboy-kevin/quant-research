"""Offline CLI: python -m quant_research.worldsim run --help."""
import argparse
from pathlib import Path
from time import perf_counter

from .arms import CarryArm, ChurnArm
from .contracts import WorldConfig
from .fees import Fees
from .fills import FillConfig
from .portfolio import run_portfolio


def main(argv=None):
    parser = argparse.ArgumentParser(prog="worldsim")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generated-market experiments (illustrative defaults)")
    run.add_argument("--group", choices=("perp", "prediction", "both"), default="both")
    run.add_argument("--arm", choices=("churn", "carry", "both"), default="both")
    run.add_argument("--markets", type=int, default=500)
    run.add_argument("--worlds", type=int, default=100)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--steps", type=int, default=240)
    run.add_argument("--dt-ms", type=int, default=250)
    run.add_argument("--regime", choices=("slow", "fast"), default="slow")
    run.add_argument("--common-weight", type=float, default=0.05)
    run.add_argument("--cluster-weight", type=float, default=0.15)
    run.add_argument("--clusters", type=int, default=10)
    run.add_argument("--adverse-selection", type=float, default=0.5)
    run.add_argument("--post-ms", type=int, default=0)
    run.add_argument("--cancel-ms", type=int, default=0)
    run.add_argument("--queue-fraction", type=float, default=0.0)
    run.add_argument("--maker-rate", type=float, default=0.0002)
    run.add_argument("--taker-rate", type=float, default=0.002)
    run.add_argument("--paired", action="store_true")
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.paired and args.group != "prediction":
        parser.error("--paired requires --group prediction")
    groups = ("perp", "prediction") if args.group == "both" else (args.group,)
    try:
        configs = [WorldConfig(group=g, markets=args.markets, steps=args.steps, dt_ms=args.dt_ms,
                               regime=args.regime, common_weight=args.common_weight,
                               cluster_weight=args.cluster_weight, clusters=args.clusters,
                               adverse_selection=args.adverse_selection, paired=args.paired) for g in groups]
        arms = {name: cls() for name, cls in (("churn", ChurnArm), ("carry", CarryArm))
                if args.arm in (name, "both")}
        fill = FillConfig(args.post_ms, args.cancel_ms, args.queue_fraction)
        fees = Fees(args.maker_rate, args.taker_rate)
        start = perf_counter()
        summary = run_portfolio(configs, arms, args.worlds, args.seed, args.out, fill, fees)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    elapsed = perf_counter() - start
    print(f"worldsim complete: groups={','.join(groups)} arms={','.join(arms)} "
          f"markets={args.markets} worlds={args.worlds} steps={args.steps} dt_ms={args.dt_ms}")
    for row in summary.to_dicts():
        sd = "null" if row['sd'] is None else f"{row['sd']:.6f}"
        print(f"{row['group']}/{row['arm']}: mean_usd={row['mean']:.6f} sd_usd={sd} "
              f"p_positive={row['p_positive']:.6f} mean_drawdown_usd={row['mean_drawdown']:.6f}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print(f"out={args.out.resolve()}")


if __name__ == "__main__":
    main()
