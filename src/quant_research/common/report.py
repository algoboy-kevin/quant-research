"""summary.md builder: verdict first, then tables, coverage and caveats."""
from __future__ import annotations


def fmt(x, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return str(x)


def md_table(rows, columns: list[str] | None = None, nd: int = 3, max_rows: int | None = None) -> str:
    """Markdown table from a polars DataFrame or a list of dicts."""
    if hasattr(rows, "to_dicts"):
        columns = columns or list(rows.columns)
        rows = rows.to_dicts()
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[:max_rows]
    if not rows:
        return "_(no rows)_\n"
    cols = columns or list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(c), nd) for c in cols) + " |")
    return "\n".join(out) + "\n"


class Summary:
    """Collects sections and renders one markdown report.

    Order is fixed: title, verdict, evidence level, sections, coverage, caveats.
    `evidence` says what kind of claim this is (description / proxy / simulation / live)
    so a descriptive result cannot be mistaken for a profit claim.
    """

    def __init__(self, title: str, *, verdict: str = "", evidence: str = ""):
        self.title, self.verdict, self.evidence = title, verdict, evidence
        self.sections: list[tuple[str, str]] = []
        self.coverage: dict = {}
        self.caveats: list[str] = []

    def section(self, heading: str, body: str) -> "Summary":
        self.sections.append((heading, body))
        return self

    def table(self, heading: str, rows, columns: list[str] | None = None, nd: int = 3, max_rows: int | None = None):
        return self.section(heading, md_table(rows, columns, nd, max_rows))

    def caveat(self, text: str) -> "Summary":
        self.caveats.append(text)
        return self

    def render(self) -> str:
        parts = [f"# {self.title}\n"]
        if self.verdict:
            parts.append(f"**{self.verdict}**\n")
        if self.evidence:
            parts.append(f"_Evidence level: {self.evidence}_\n")
        for h, b in self.sections:
            parts.append(f"## {h}\n\n{b.rstrip()}\n")
        if self.coverage:
            parts.append("## Coverage\n\n" + "\n".join(f"- **{k}**: {fmt(v)}" for k, v in self.coverage.items()) + "\n")
        if self.caveats:
            parts.append("## Caveats\n\n" + "\n".join(f"- {c}" for c in self.caveats) + "\n")
        return "\n".join(parts)


def save_fig(fig, path, *, dpi: int = 110):
    """Save a matplotlib figure headlessly and close it. Needs the `analysis` extra."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
