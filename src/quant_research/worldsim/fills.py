"""Vectorized resting-order state machine; prints conserve volume across levels."""
from dataclasses import dataclass

import numpy as np

from .contracts import Frame, Intent


@dataclass(frozen=True)
class FillConfig:
    post_ms: int = 0
    cancel_ms: int = 0
    queue_fraction: float = 0.0

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.post_ms, self.cancel_ms)):
            raise ValueError("latencies must be nonnegative integer milliseconds")
        if not np.isfinite(self.queue_fraction) or not 0 <= self.queue_fraction <= 1e6:
            raise ValueError("queue_fraction outside [0,1e6]")


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
        self.queue = np.zeros(shape, dtype=np.int64)
        self.replace_price = np.full(shape, np.nan)
        self.replace_size = np.zeros(shape, dtype=np.int64)
        self.rejections_by_world = np.zeros(worlds, dtype=np.int64)

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
            self.blocked[:, side] |= pending & crossed
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
        """Yield (signed sizes[N], prices[N]); mutate remaining size and queue."""
        rows = np.arange(self.price.shape[0])
        for side in range(2):
            direction = -1 if side == 0 else 1
            available = np.where(frame.aggressor == direction, frame.trade_size, 0).copy()
            external_used = np.zeros_like(available)
            active = ((self.status[:, side] == 2) | (self.status[:, side] == 3)) & (self.size[:, side] > 0)
            eligible = active & (frame.trade_price[:, None] <= self.price[:, side] if side == 0
                                 else frame.trade_price[:, None] >= self.price[:, side])
            priority = np.argsort(np.where(active, -self.price[:, side] if side == 0
                                           else self.price[:, side], np.inf), axis=1, kind="stable")
            for rank in range(self.price.shape[2]):
                level = priority[:, rank]
                ok = eligible[rows, level]
                ahead = np.maximum(self.queue[rows, side, level] - external_used, 0)
                burn = np.where(ok, np.minimum(available, ahead), 0)
                available -= burn
                external_used += burn
                qty = np.where(ok, np.minimum(available, self.size[rows, side, level]), 0)
                available -= qty
                self.size[rows, side, level] -= qty
                if (qty).any():
                    price = np.where(qty > 0, self.price[rows, side, level], 0.0)
                    yield qty if side == 0 else -qty, price
            self.queue[:, side] = np.where(eligible,
                                           np.maximum(self.queue[:, side] - external_used[:, None], 0),
                                           self.queue[:, side])
        # A completely filled cancel-pending slot still waits for acknowledgement
        # so its replacement cannot bypass cancel latency.
        self.clear((self.size == 0) & (self.status == 2))
