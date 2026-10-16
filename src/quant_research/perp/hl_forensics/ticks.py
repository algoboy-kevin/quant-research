"""Hyperliquid perp price tick - the single source of truth.

A price may have at most 5 significant figures AND at most (6 - szDecimals)
decimals; integer prices are always allowed. The tick therefore shrinks as the
price magnitude grows. Do NOT use 10 ** -szDecimals.
"""


def tick_size(price, sz_decimals, max_decimals=6):
    if price <= 0:
        return 10 ** (-(max_decimals - sz_decimals))
    if price >= 1:
        dec_sig = max(0, 5 - len(str(int(price))))
    else:
        k, x = 0, price
        while x < 1 and k < 12:
            x *= 10
            k += 1
        dec_sig = 5 + (k - 1)
    return 10 ** (-min(max_decimals - sz_decimals, dec_sig))
