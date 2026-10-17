import unittest

from quant_research.worldsim.contracts import WorldConfig
from quant_research.worldsim.parallel import run_world, run_worlds
from quant_research.worldsim.reference_arms import build

PRED = dict(common_weight=0.15, cluster_weight=0.35, clusters=3, regime="fast")


def _jobs():
    jobs = []
    for i, (group, markets, steps, spec) in enumerate([
            ("prediction", 6, 10, "churn"), ("perp", 2, 3, "carry"),
            ("prediction", 3, 8, "far_fill"), ("perp", 5, 10, "churn.width_ticks=2"),
            ("prediction", 2, 2, "churn"), ("perp", 4, 6, "far_fill")]):
        kw = PRED if group == "prediction" else dict(perp_sigma=0.5)
        jobs.append(dict(cfg=WorldConfig(group=group, markets=markets, steps=steps,
                                         dt_ms=250, **kw),
                         arm=spec, seed=100 + i, collect=bool(i % 2), stride=2))
    return jobs


class ParallelTests(unittest.TestCase):
    def test_sequential_equals_parallel(self):
        jobs = _jobs()
        self.assertEqual(run_worlds(jobs, 1), run_worlds(jobs, 4))

    def test_input_order_under_concurrency(self):
        jobs = _jobs()  # very different sizes -> completion order differs from input order
        out = run_worlds(jobs, 4)
        self.assertEqual(len(out), len(jobs))
        for job, res in zip(jobs, out):
            self.assertEqual(len(res["pnl_nano"]), job["cfg"].markets)
            self.assertEqual(res, run_world(job))
            self.assertEqual(bool(res["equity_rows"]), job["collect"])

    def test_arm_instance_matches_spec(self):
        job = _jobs()[0]
        inst = dict(job, arm=build(job["arm"])[2])
        self.assertEqual(run_worlds([job], 1), run_worlds([inst], 1))

    def test_bad_payloads_raise(self):
        good = _jobs()[1]
        for bad, exc in [
                ("notadict", TypeError),
                ({k: v for k, v in good.items() if k != "seed"}, ValueError),
                (dict(good, extra=1), ValueError),
                (dict(good, cfg={"group": "perp"}), TypeError),
                (dict(good, arm=lambda: None), TypeError),
                (dict(good, seed=-1), ValueError),
                (dict(good, collect=1), TypeError),
                (dict(good, stride=0), ValueError)]:
            for workers in (1, 2):
                with self.assertRaises(exc):
                    run_worlds([good, bad], workers)
        with self.assertRaises(ValueError):
            run_worlds([good], 0)
        with self.assertRaises(KeyError):
            run_worlds([dict(good, arm="nonexistent")], 1)

    def test_empty(self):
        self.assertEqual(run_worlds([], 4), [])


if __name__ == "__main__":
    unittest.main()
