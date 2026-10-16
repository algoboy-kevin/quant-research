import pytest
from quant_research.perp.hl_forensics.ticks import tick_size


@pytest.mark.parametrize("px,szd,expected", [
    (65000.0, 5, 1.0),      # 5 sig figs -> integer tick
    (2500.5, 4, 0.1),       # 5 sig figs -> 1 decimal
    (150.25, 2, 0.01),      # 5 sig figs -> 2 decimals
    (1.2345, 0, 1e-4),
    (0.5, 0, 1e-5),         # <1: 5 sig figs counted after leading zeros
    (0.012345, 0, 1e-6),    # capped by 6 - szDecimals
    (0.012345, 3, 1e-3),    # szDecimals 3 -> at most 3 decimals
    # observed live (venue docs / worked table)
    (68755.0, 5, 1.0),      # JP225
    (1188.0, 4, 0.1),       # LLY
    (7773.0, 3, 0.1),       # SP500
    (336.0, 3, 0.01),       # AAPL
    (6.686, 2, 1e-4),       # COPPER
    (7.56, 1, 1e-4),        # CXMT
    (3.361, 1, 1e-4),       # NATGAS
])
def test_tick_size(px, szd, expected):
    assert tick_size(px, szd) == pytest.approx(expected)


def test_nonpositive_price_uses_decimal_cap():
    assert tick_size(0, 2) == pytest.approx(1e-4)


def test_not_naive_szdecimals():
    # the documented trap: 10 ** -szDecimals gives 1e-5 here, the real tick is 1
    assert tick_size(65000.0, 5) != 10 ** -5
