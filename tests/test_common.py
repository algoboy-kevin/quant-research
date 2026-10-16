import random

import polars as pl
import pytest

from quant_research.common import stats
from quant_research.common.jsonl import JsonlWriter, read_jsonl
from quant_research.common.report import Summary, md_table
from quant_research.common.runs import RunFolder
from quant_research.common.tables import frame_from_records, read_jsonl_frame


def test_stats_basic_and_empty():
    assert stats.mean([]) is None and stats.median([]) is None and stats.pctl([], 50) is None
    assert stats.mean([1, 2, None, 3]) == 2
    assert stats.pctl([1, 2, 3, 4], 50) == 2.5
    assert stats.pctl([1, 2, 3, 4], 0) == 1 and stats.pctl([1, 2, 3, 4], 100) == 4
    assert stats.wmean([1, 3], [1, 3]) == 2.5


def test_bootstrap_is_seeded_and_covers_mean():
    xs = [random.Random(1).gauss(5, 1) for _ in range(1)] + [5.0, 5.2, 4.8, 5.1, 4.9, 5.3]
    a = stats.bootstrap_ci(xs, seed=3)
    assert a == stats.bootstrap_ci(xs, seed=3)
    assert a[0] <= stats.mean(xs) <= a[1]
    assert stats.bootstrap_ci([1.0]) is None


def test_cluster_bootstrap_wider_than_naive_when_clusters_share_shock():
    groups = {g: [shock + e for e in (0.0, 0.01, -0.01, 0.02)] for g, shock in enumerate([-2, -1, 0, 1, 2, 3])}
    flat = [v for vs in groups.values() for v in vs]
    naive = stats.bootstrap_ci(flat, seed=0)
    clus = stats.cluster_bootstrap_ci(groups, seed=0)
    assert (clus[1] - clus[0]) > 0.5 * (naive[1] - naive[0])
    assert stats.cluster_bootstrap_ci({"a": [1.0]}) is None


def test_jsonl_roundtrip_gz_and_tolerance(tmp_path):
    p = tmp_path / "a.jsonl"
    with JsonlWriter(p) as w:
        w.write({"x": 1}); w.write({"x": 2})
    with open(p, "a") as f:
        f.write("not json\n")
    st = {}
    rows = list(read_jsonl(p, stats=st))
    assert [r["x"] for r in rows] == [1, 2] and st["malformed"] == 1
    assert "t_mono" in rows[0] and "t_wall" in rows[0]
    g = tmp_path / "b.jsonl.gz"
    with JsonlWriter(g, stamp=False) as w:
        w.write({"k": "v"})
    assert list(read_jsonl(g)) == [{"k": "v"}]


def test_frame_from_records_infers_from_all_rows():
    df = frame_from_records([{"a": None, "b": 1}, {"a": 2.5, "b": 2}])
    assert df.schema["a"] == pl.Float64 and df.height == 2
    assert frame_from_records([]).height == 0


def test_read_jsonl_frame_select(tmp_path):
    p = tmp_path / "s.jsonl"
    with JsonlWriter(p) as w:
        w.write({"msg": {"score": "1-0"}}); w.write({"msg": {"score": "1-1"}})
    df = read_jsonl_frame(p, select={"score": "msg.score", "t": "t_mono"})
    assert df["score"].to_list() == ["1-0", "1-1"]


def test_run_folder_and_summary(tmp_path):
    run = RunFolder(tmp_path, "t1", args={"a": 1})
    run.write_frame("tbl", pl.DataFrame({"a": [1, 2], "b": [0.5, None]}))
    assert (run.dir / "tbl.parquet").exists() and (run.dir / "tbl.csv").exists()
    s = Summary("T", verdict="V", evidence="description")
    s.table("Rows", pl.DataFrame({"a": [1], "b": [0.12345]})).caveat("c")
    s.coverage = {"n": 3}
    run.write_text("summary.md", s.render())
    run.finish()
    md = (run.dir / "summary.md").read_text()
    assert md.index("**V**") < md.index("## Rows") < md.index("## Coverage") < md.index("## Caveats")
    assert "0.123" in md
    assert (run.dir / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        RunFolder(tmp_path, "t1")


def test_md_table_handles_none_and_empty():
    assert "-" in md_table([{"a": None}])
    assert md_table([]) == "_(no rows)_\n"
