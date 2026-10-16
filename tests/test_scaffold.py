import pytest

from quant_research import scaffold


def test_scaffold_writes_study_outside_repo(tmp_path):
    scaffold.main(["polymarket", "Sports Adverse", "--root", str(tmp_path)])
    d = tmp_path / "polymarket" / "sports-adverse"
    assert {p.name for p in d.iterdir()} == {"study.py", "SPEC.md", "test_study.py"}
    compile((d / "study.py").read_text(), "study.py", "exec")      # generated code is valid Python
    scaffold.main(["polymarket", "Sports Adverse", "--root", str(tmp_path)])   # idempotent: never overwrites
    assert "TODO" in (d / "study.py").read_text()


def test_scaffold_refuses_to_write_inside_repo():
    repo = scaffold.Path(scaffold.__file__).resolve().parents[2]
    with pytest.raises(SystemExit):
        scaffold.main(["polymarket", "x", "--root", str(repo / "research")])
