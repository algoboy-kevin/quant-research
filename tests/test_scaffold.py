import pytest

from quant_research import lab, scaffold


def test_scaffold_writes_study_outside_repo(tmp_path):
    scaffold.main(["polymarket", "Sports Adverse", "--root", str(tmp_path)])
    d = tmp_path / "polymarket" / "sports-adverse"
    assert {p.name for p in d.iterdir()} == {"study.py", "SPEC.md", "test_study.py", "FINDINGS.md"}
    src = (d / "study.py").read_text()
    compile(src, "study.py", "exec")                                   # generated code is valid Python
    compile((d / "test_study.py").read_text(), "test_study.py", "exec")
    assert "uv run python polymarket/sports-adverse/study.py" in src
    scaffold.main(["polymarket", "Sports Adverse", "--root", str(tmp_path)])   # idempotent: never overwrites
    assert "TODO" in (d / "study.py").read_text()


def test_scaffold_refuses_to_write_inside_repo():
    repo = scaffold.Path(scaffold.__file__).resolve().parents[2]
    with pytest.raises(SystemExit):
        scaffold.main(["polymarket", "x", "--root", str(repo / "research")])


def test_lab_root_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_LAB", str(tmp_path))
    assert scaffold.lab_root() == tmp_path


def test_fresh_findings_template_is_flagged_until_filled(tmp_path):
    scaffold.main(["polymarket", "s1", "--root", str(tmp_path)])
    rows = lab.scan(tmp_path)
    assert len(rows) == 1 and rows[0]["problems"]            # TODO verdict/run placeholders are not acceptable
    f = tmp_path / "polymarket" / "s1" / "FINDINGS.md"
    f.write_text(f.read_text().replace("TODO one hedged sentence, no stronger than the evidence level",
                                       "Descriptive only: X moves after Y in this sample."))
    rows = lab.scan(tmp_path)
    assert rows[0]["problems"] == [] and rows[0]["verdict"].startswith("Descriptive only")


def test_check_rejects_bad_values_and_index_renders(tmp_path):
    assert lab.check({"status": "great", "evidence": "vibes", "date": "yesterday", "verdict": "v"}) == [
        "status 'great' not in ['draft', 'reviewed', 'superseded']",
        "evidence 'vibes' not in ['description', 'live', 'proxy', 'simulation']",
        "date must be YYYY-MM-DD"]
    d = tmp_path / "hyperliquid" / "ladder"
    d.mkdir(parents=True)
    (d / "FINDINGS.md").write_text("# Ladder\n\n- status: reviewed\n- evidence: description\n- date: 2026-10-16\n- verdict: ok | fine\n\n## Result\n- status: ignored\n")
    (tmp_path / "_data" / "x").mkdir(parents=True)
    (tmp_path / "_data" / "x" / "FINDINGS.md").write_text("# skipped")
    rows = lab.scan(tmp_path)
    assert [r["study"] for r in rows] == ["ladder"] and rows[0]["status"] == "reviewed"   # stops at first '## '
    md = lab.render_index(rows)
    assert "[ladder](hyperliquid/ladder/FINDINGS.md)" in md and "ok / fine" in md
    lab.main([str(tmp_path), "--check"])
    assert (tmp_path / "INDEX.md").exists()
