"""Vectorized resting-order state machine; prints conserve volume across levels."""
from dataclasses import dataclass

import numpy as np

from .contracts import Frame, Intent


@dataclass(frozen=True)
class FillConfig:
    post_ms: int = 0
    cancel_ms: int = 0
    # Fraction of the displayed better-or-equal depth that is queued ahead of a new quote
    # (units of the quote's own size ~ 1/5 of one displayed rung). Calibrated: see
    # worldsim-flowfix calibration report; 0 restores absolute priority (legacy).
    queue_fraction: float = 1.0
    # --- event-time (flow) fill model: hftbacktest-style same-price queue ---
    # At post the rung's same-price queue ahead is Q0 = psi * D(k), D(j) = d1 * (j + 1)**beta
    # units of the order's own size, k = ticks behind the displayed touch. A print of volume v
    # reaching the rung first consumes the displayed depth between touch and rung,
    # B(k) = sum_{j<k} D(j), then burns Q; only the overflow fills. Q decays with a cancel
    # hazard (per second) and is KEPT while the order is kept (queue position is earned).
    psi: float = 0.75
    d1: float = 10.0
    beta: float = 0.17
    cancel_hazard_per_s: float = 0.005
    # Literal first passage on the DISPLAYED path: a rung the displayed touch drifts strictly
    # through is filled. Off by default: the exogenous drift ignores the resting order, and the
    # tape measures P(fill | touch arrives at a wallet rung) = 3.1%, not ~1 (in a real book the
    # touch stops at the order unless TRADES go through it). Trade-driven passage (sweeps, which
    # trade through every level up to their depth) is always on.
    drift_passage: bool = False

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.post_ms, self.cancel_ms)):
            raise ValueError("latencies must be nonnegative integer milliseconds")
        if not np.isfinite(self.queue_fraction) or not 0 <= self.queue_fraction <= 1e6:
            raise ValueError("queue_fraction outside [0,1e6]")
        if not all(np.isfinite(x) for x in (self.psi, self.d1, self.beta, self.cancel_hazard_per_s)):
            raise ValueError("finite queue configuration required")
        if not 0 <= self.psi <= 1 or self.d1 < 0 or self.cancel_hazard_per_s < 0 or not -2 <= self.beta <= 3:
            raise ValueError("psi in [0,1], d1 >= 0, hazard >= 0, beta in [-2,3] required")


class Orders:
    def __init__(self, markets, levels, config=FillConfig(), worlds=1):
        """`markets` slots in total; with worlds=K > 1 they are K stacked equal worlds
        (world-major), used only to attribute rejections per world."""
        if type(levels) is not int or levels < 1:
            raise ValueError("positive integer levels required")
        if type(worlds) is not int or worlds < 1 or markets % worlds:
            raise ValueError("worlds must be a positive int dividing markets")
        self.worlds = worlds
        self.config = config
        shape = (markets, 2, levels)
        self.price = np.full(shape, np.nan)
        self.size = np.zeros(shape, dtype=np.int64)
        self.status = np.zeros(shape, dtype=np.int8)
        self.due = np.zeros(shape, dtype=np.int64)
        self.blocked = np.zeros(shape, dtype=bool)
        self.queue = np.zeros(shape)
        self.replace_price = np.full(shape, np.nan)
        self.replace_size = np.zeros(shape, dtype=np.int64)
        self.rejections_by_world = np.zeros(worlds, dtype=np.int64)
        self.event_info = None
        c = self.config
        self._B = np.concatenate(([0.0], np.cumsum(c.d1 * np.arange(1, 4097) ** c.beta)))

    @property
    def rejections(self) -> int:
        return int(self.rejections_by_world.sum())

    def clear(self, mask):
        self.status[mask] = 0
        self.price[mask] = np.nan
        self.size[mask] = 0
        self.queue[mask] = 0
        self.blocked[mask] = False

    def advance(self, frame: Frame):
        t = frame.ts_ms
        # Cancellation acknowledgement, then replacement transmission.
        ack = (self.status == 3) & (self.due <= t)
        replace = ack & (self.replace_size > 0)
        self.clear(ack)
        self.price[replace] = self.replace_price[replace]
        self.size[replace] = self.replace_size[replace]
        self.status[replace] = 1
        self.due[replace] = t + self.config.post_ms
        self.replace_size[ack] = 0
        self.replace_price[ack] = np.nan
        for side in range(2):
            pending = self.status[:, side] == 1
            crossed = (frame.best_ask[:, None] <= self.price[:, side] if side == 0
                       else frame.best_bid[:, None] >= self.price[:, side])
            self.blocked[:, side] |= pending & crossed & (frame.flow is None)
        activate = (self.status == 1) & (self.due <= t)
        reject = activate & self.blocked
        if reject.any():
            self.rejections_by_world += np.count_nonzero(
                reject.reshape(self.worlds, -1), axis=1)
        self.clear(reject)
        activate &= ~reject
        for side in range(2):
            depth_price = frame.bid_px if side == 0 else frame.ask_px
            depth_size = frame.bid_size if side == 0 else frame.ask_size
            for level in range(self.price.shape[2]):
                mask = activate[:, side, level]
                if not (mask).any():
                    continue
                p = self.price[:, side, level]
                if frame.flow is not None:
                    touch = frame.best_bid if side == 0 else frame.best_ask
                    k = np.rint((touch - p if side == 0 else p - touch) / frame.tick_size)
                    kk = np.clip(k, 0, 4095).astype(np.int64)
                    ahead = np.where(k < 0, 0.0, self.config.psi * self.config.d1 * (kk + 1.0) ** self.config.beta)
                    self.queue[mask, side, level] = ahead[mask]
                    continue
                better = depth_price >= p[:, None] if side == 0 else depth_price <= p[:, None]
                ahead = np.ceil(np.sum(better * depth_size, axis=1) * self.config.queue_fraction)
                self.queue[mask, side, level] = ahead[mask].astype(np.int64)
        self.status[activate] = 2

    def submit(self, frame: Frame, intent: Intent):
        shape = self.price.shape
        for a in (intent.action, intent.price, intent.size):
            if np.shape(a) != shape:
                raise ValueError("intent arrays must have shape [N,2,L]")
        if not np.issubdtype(intent.action.dtype, np.integer) or (
                (intent.action < 0) | (intent.action > 2)).any():
            raise ValueError("invalid intent action")
        post = intent.action == 1
        if (not np.issubdtype(intent.size.dtype, np.integer)
                or (post & ((intent.size <= 0) | (intent.size > 1_000_000))).any()):
            raise ValueError("post sizes must be positive integer contracts <=1e6")
        price = np.array(intent.price, dtype=float, copy=True)
        if (post & ~np.isfinite(price)).any():
            raise ValueError("nonfinite quote")
        if intent.price_mode == "offset":
            if (post & ((price < 0) | (price != np.floor(price)))).any():
                raise ValueError("offset must be nonnegative integer ticks")
            price = frame.mid[:, None, None] + price * frame.tick_size[:, None, None] * np.array([-1, 1])[None, :, None]
        elif intent.price_mode != "absolute":
            raise ValueError("unknown price mode")
        ticks = frame.tick_size[:, None]
        price[:, 0] = np.floor(price[:, 0] / ticks + 1e-9) * ticks
        price[:, 1] = np.ceil(price[:, 1] / ticks - 1e-9) * ticks
        price = np.round(price, 6)
        if frame.group == "prediction" and (post & ((price <= 0) | (price >= 1))).any():
            raise ValueError("prediction quote outside (0,1)")
        # Validate the requested ladder, excluding cancelled/absent slots.
        candidate = np.where(post, price, self.price)
        candidate = np.where(intent.action == 2, np.nan, candidate)
        for side in range(2):
            for left in range(shape[2]):
                for right in range(left + 1, shape[2]):
                    bad = (candidate[:, side, left] <= candidate[:, side, right] if side == 0
                           else candidate[:, side, left] >= candidate[:, side, right])
                    if (bad).any():
                        raise ValueError("ladder must be distinct and best first")
        for bid in range(shape[2]):
            for ask in range(shape[2]):
                if (candidate[:, 0, bid] >= candidate[:, 1, ask]).any():
                    raise ValueError("own ladder crosses")
        cancel = intent.action == 2
        self.clear(cancel & (self.status == 1))
        live = self.status == 2
        changed = post & ((price != self.price) | (intent.size != self.size))
        replacing = live & changed
        cancelling = live & (cancel | changed)
        self.status[cancelling] = 3
        self.due[cancelling] = frame.ts_ms + self.config.cancel_ms
        self.replace_price[replacing] = price[replacing]
        self.replace_size[replacing] = intent.size[replacing]
        fresh = post & (self.status == 0)
        self.price[fresh] = price[fresh]
        self.size[fresh] = intent.size[fresh]
        self.status[fresh] = 1
        self.blocked[fresh] = False
        self.due[fresh] = frame.ts_ms + self.config.post_ms
        self.advance(frame)

    def fills(self, frame: Frame):
        """Yield (signed sizes[N], prices[N]); mutate remaining size and queue.

        flow frames (event-time model): the displayed centre follows the bar's path
        (straight segments between the event knots; each event is a knot with a pre/post
        centre). Two ways to fill, both at the rung's own price:
          * FIRST PASSAGE: the displayed touch path goes strictly through the rung (a sweep
            or drift): filled at the first passage; the pre-fill mid is the centre at that
            instant (rung + half spread), never stale.
          * PRINTS: a print of volume v reaching the rung burns the displayed depth in front
            (B(k)), then the same-price queue Q; the overflow fills.
        Legacy frames (flow None): the single tape print (trade_price/trade_size/aggressor)
        is swept against the displayed better-or-equal depth (hand-built frames, tests).
        After each yield `self.event_info` holds (frac[N], pre-fill mid[N]); only rows with a
        fill are meaningful (None on the legacy path)."""
        self.event_info = None
        if frame.flow is None:
            yield from self._sweep(np.arange(self.price.shape[0]), frame.aggressor,
                                   frame.trade_size, frame.trade_price, None)
            return
        fl = frame.flow
        N = self.price.shape[0]
        tick = frame.tick_size
        hz = self.config.cancel_hazard_per_s
        if hz > 0:
            self.queue *= 0.5 ** (hz * fl.span_ms / 1000.0 / 0.6931471805599453)
        E = fl.size.shape[0]
        hs_t = fl.hs * tick
        cur_mid, cur_u = fl.c0.astype(float), np.zeros(N)
        done = np.zeros(N, dtype=bool)
        for r in range(E + 1):
            if r < E:
                valid = fl.size[r] > 0
                nxt_mid = np.where(valid, fl.mid[r], fl.c1)
                nxt_u = np.where(valid, fl.frac[r], 1.0)
            else:
                valid = np.zeros(N, dtype=bool)
                nxt_mid, nxt_u = fl.c1.astype(float), np.ones(N)
            yield from self._passage(frame, ~done, cur_mid, cur_u, nxt_mid, nxt_u, hs_t) if \
                self.config.drift_passage else iter(())
            if r < E and valid.any():
                yield from self._event(frame, np.flatnonzero(valid), r)
                cur_mid = np.where(valid, fl.mid_post[r], cur_mid)
                cur_u = np.where(valid, fl.frac[r], cur_u)
            done |= ~valid
        self.event_info = None

    def _passage(self, frame, seg, ma, ua, mb, ub, hs_t):
        """First-passage fills over the straight displayed-centre segment (ma,ua)->(mb,ub)."""
        fl = frame.flow
        eps = 1e-9
        for side in range(2):
            sgn = 1.0 if side == 0 else -1.0          # bid: touch below rung is "through"
            live = ((self.status[:, side] >= 2) & (self.size[:, side] > 0)) & seg[:, None]
            if not live.any():
                continue
            ta = ma - sgn * hs_t                       # touch path endpoints (bid: mid - hs)
            tb = mb - sgn * hs_t
            price = self.price[:, side]
            inside = sgn * (price - ta[:, None]) > eps      # rung strictly INSIDE the displayed market
            # behind/at the touch: filled when the touch path goes strictly through it; inside
            # the market the rung IS the touch (only prints hit it) unless the far touch crosses it
            far_b = mb + sgn * hs_t
            through = live & np.where(inside, sgn * (price - far_b[:, None]) > eps,
                                      sgn * (price - tb[:, None]) > eps)
            if not through.any():
                continue
            start_in = ~inside                                   # touch was at/behind the rung at the start
            denom = np.where(np.abs(ta - tb) > eps, ta - tb, 1.0)
            lam = np.where(start_in, np.clip((ta[:, None] - price) / denom[:, None], 0.0, 1.0), 1.0)
            fr = ua[:, None] + lam * (ub - ua)[:, None]
            te = fl.t0_ms + fr * fl.span_ms
            ok = through & ((self.status[:, side] == 3) | (self.due[:, side] <= te))
            premid = np.where(start_in, price + sgn * hs_t[:, None], price)
            for level in range(self.price.shape[2]):
                m = ok[:, level]
                if not m.any():
                    continue
                qty = np.where(m, self.size[:, side, level], 0)
                self.size[:, side, level] -= qty
                self.event_info = (np.where(m, fr[:, level], 0.0), np.where(m, premid[:, level], 0.0))
                yield (qty if side == 0 else -qty), np.where(m, self.price[:, side, level], 0.0)

    def _event(self, frame, sel, r):
        """One print/sweep per selected market against the same-price queues (best price first)."""
        fl = frame.flow
        N = self.price.shape[0]
        for direction in (-1, 1):
            rows = sel[fl.side[r, sel] == direction]
            if not len(rows):
                continue
            side = 0 if direction == -1 else 1
            price = self.price[rows, side]
            size = self.size[rows, side]
            status = self.status[rows, side]
            queue = self.queue[rows, side].copy()
            te = fl.t0_ms + fl.frac[r, rows] * fl.span_ms
            active = ((status >= 2) & (size > 0)
                      & ((status == 3) | (self.due[rows, side] <= te[:, None])))
            reach = fl.price[r, rows][:, None]
            touch = fl.touch[r, rows][:, None]
            tick = frame.tick_size[rows][:, None]
            eligible = active & (price >= reach - 1e-9 if side == 0 else price <= reach + 1e-9)
            if not eligible.any():
                continue
            k = np.rint((touch - price if side == 0 else price - touch) / tick)
            kk = np.clip(k, 0, 4095).astype(np.int64)
            v = fl.size[r, rows].astype(float)
            priority = np.argsort(np.where(eligible, -price if side == 0 else price, np.inf),
                                  axis=1, kind="stable")
            sub = np.arange(len(rows))
            own = np.zeros(len(rows))
            for rank in range(price.shape[1]):
                level = priority[:, rank]
                ok = eligible[sub, level]
                avail = np.where(ok, np.maximum(v - self._B[kk[sub, level]] - own, 0.0), 0.0)
                q = queue[sub, level]
                burn = np.minimum(avail, q)
                avail = avail - burn
                queue[sub, level] = np.where(ok, q - burn, q)
                qty = np.where(ok, np.minimum(np.floor(avail), size[sub, level]), 0).astype(np.int64)
                size[sub, level] -= qty
                own += qty
                if qty.any():
                    full_qty = np.zeros(N, dtype=np.int64)
                    full_px = np.zeros(N)
                    fr, pm = np.zeros(N), np.zeros(N)
                    full_qty[rows] = qty
                    full_px[rows] = np.where(qty > 0, price[sub, level], 0.0)
                    hs_r = (fl.hs * frame.tick_size)[rows]
                    pre = fl.mid[r, rows]
                    # the pre-fill centre can never sit on the wrong side of the quote: a rung
                    # that drift left inside the market is the touch (centre = rung +/- hs)
                    pre = np.maximum(pre, price[sub, level] + hs_r) if side == 0 \
                        else np.minimum(pre, price[sub, level] - hs_r)
                    fr[rows], pm[rows] = fl.frac[r, rows], pre
                    self.event_info = (fr, pm)
                    self.size[rows, side] = size
                    yield (full_qty if side == 0 else -full_qty), full_px
            self.size[rows, side] = size
            self.queue[rows, side] = queue

    def _sweep(self, rows_sel, aggressor, trade_size, trade_price, te, only_side=None, info=None):
        """Legacy: one tape print per market row against the resting orders (best price first)."""
        rows = rows_sel
        for side in (range(2) if only_side is None else (only_side,)):
            direction = -1 if side == 0 else 1
            price = self.price[rows, side]
            size = self.size[rows, side]
            status = self.status[rows, side]
            queue = self.queue[rows, side].astype(np.int64)
            available = np.where(aggressor[rows] == direction, trade_size[rows], 0).astype(np.int64)
            tp = trade_price[rows]
            external_used = np.zeros_like(available)
            active = ((status == 2) | (status == 3)) & (size > 0)
            eligible = active & (tp[:, None] <= price if side == 0 else tp[:, None] >= price)
            priority = np.argsort(np.where(active, -price if side == 0 else price, np.inf),
                                  axis=1, kind="stable")
            sub = np.arange(len(rows))
            for rank in range(price.shape[1]):
                level = priority[:, rank]
                ok = eligible[sub, level]
                ahead = np.maximum(queue[sub, level] - external_used, 0)
                burn = np.where(ok, np.minimum(available, ahead), 0)
                available -= burn
                external_used += burn
                qty = np.where(ok, np.minimum(available, size[sub, level]), 0)
                available -= qty
                size[sub, level] -= qty
                if qty.any():
                    full_qty = np.zeros(self.price.shape[0], dtype=np.int64)
                    full_px = np.zeros(self.price.shape[0])
                    full_qty[rows] = qty
                    full_px[rows] = np.where(qty > 0, price[sub, level], 0.0)
                    self.size[rows, side] = size
                    yield (full_qty if side == 0 else -full_qty), full_px
            queue = np.where(eligible, np.maximum(queue - external_used[:, None], 0), queue)
            self.size[rows, side] = size
            self.queue[rows, side] = queue
