"""Charts: render smoke + input validation (Agg, deterministic)."""
import tempfile
import unittest
from pathlib import Path

import polars as pl

from quant_research.common import charts


class ChartTests(unittest.TestCase):
    def _equity(self) -> pl.DataFrame:
        rows = []
        for setup in ("a", "b"):
            for arm in ("churn", "carry"):
                for n in (5, 10):
                    for world in range(3):
                        for step in range(10):
                            rows.append(dict(setup=setup, arm=arm, markets=n, world=world,
                                             ts_ms=step * 1000,
                                             pnl_usd=float(step * (world + 1) * n)))
        return pl.DataFrame(rows)

    def _breadth(self) -> pl.DataFrame:
        rows = []
        for setup in ("a", "b"):
            for arm in ("churn",):
                for n in (5, 10):
                    rows.append(dict(setup=setup, arm=arm, markets=n,
                                     p_positive=0.7, p_low=0.6, p_high=0.8, mean=0.1))
        return pl.DataFrame(rows)

    def _samples(self) -> pl.DataFrame:
        return pl.DataFrame({"setup": ["a"] * 24, "market": [0] * 12 + [1] * 12,
                             "ts_ms": list(range(12)) * 2, "mid": [0.5] * 24})

    def test_render_all(self):
        with tempfile.TemporaryDirectory() as d:
            outs = [
                charts.pnl_over_time_by_n(self._equity(), Path(d) / "p.png"),
                charts.breadth_curves(self._breadth(), Path(d) / "b.png",
                                      anchors={"a": [(5, 0.75), (10, 0.9)]}),
                charts.market_samples(self._samples(), Path(d) / "s.png"),
            ]
            for o in outs:
                self.assertTrue(o.exists())
                self.assertGreater(o.stat().st_size, 5_000)

    def test_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                charts.pnl_over_time_by_n(pl.DataFrame({"x": [1]}), Path(d) / "p.png")
            with self.assertRaises(ValueError):
                charts.pnl_over_time_by_n(self._equity().head(0), Path(d) / "p.png")
            with self.assertRaises(ValueError):
                charts.breadth_curves(pl.DataFrame({"x": [1]}), Path(d) / "b.png")
            with self.assertRaises(ValueError):
                charts.market_samples(pl.DataFrame({"x": [1]}), Path(d) / "s.png")


if __name__ == "__main__":
    unittest.main()
