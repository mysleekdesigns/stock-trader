"""Level-2 order book representation and microstructure analytics.

Maintains a sorted, depth-limited order book and exposes derived metrics
such as spread, imbalance, and VPIN.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BookLevel:
    """A single price level in the order book."""

    price: float
    quantity: float


class OrderBook:
    """L2 order book for a single symbol.

    Bids are stored in descending price order; asks in ascending price order.
    Only the top ``max_levels`` are retained on each side.

    Parameters
    ----------
    symbol:
        Ticker symbol this book represents.
    max_levels:
        Maximum number of price levels to track per side.
    """

    def __init__(self, symbol: str, max_levels: int = 10) -> None:
        self.symbol = symbol
        self.max_levels = max_levels

        self._bids: list[BookLevel] = []
        self._asks: list[BookLevel] = []

        # Rolling trade-bucket history for VPIN
        self._buy_volume_buckets: deque[float] = deque()
        self._sell_volume_buckets: deque[float] = deque()
        self._bucket_total: deque[float] = deque()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
    ) -> None:
        """Replace the current book with a new snapshot.

        Parameters
        ----------
        bids:
            List of ``(price, quantity)`` tuples — need not be sorted.
        asks:
            List of ``(price, quantity)`` tuples — need not be sorted.
        """
        self._bids = sorted(
            [BookLevel(p, q) for p, q in bids if q > 0],
            key=lambda lv: lv.price,
            reverse=True,
        )[: self.max_levels]

        self._asks = sorted(
            [BookLevel(p, q) for p, q in asks if q > 0],
            key=lambda lv: lv.price,
        )[: self.max_levels]

        logger.debug(
            "order_book.updated",
            symbol=self.symbol,
            bid_levels=len(self._bids),
            ask_levels=len(self._asks),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_bid(self) -> float | None:
        """Highest bid price, or ``None`` if book is empty."""
        return self._bids[0].price if self._bids else None

    @property
    def best_ask(self) -> float | None:
        """Lowest ask price, or ``None`` if book is empty."""
        return self._asks[0].price if self._asks else None

    @property
    def mid_price(self) -> float | None:
        """Mid-point between best bid and best ask."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> float | None:
        """Absolute spread between best ask and best bid."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    # ------------------------------------------------------------------
    # Depth & imbalance
    # ------------------------------------------------------------------

    def get_depth(self, levels: int = 5) -> dict[str, list[dict[str, float]]]:
        """Return top-of-book depth as a dict of bid/ask level dicts.

        Parameters
        ----------
        levels:
            Number of price levels to include per side.
        """
        return {
            "bids": [
                {"price": lv.price, "quantity": lv.quantity}
                for lv in self._bids[:levels]
            ],
            "asks": [
                {"price": lv.price, "quantity": lv.quantity}
                for lv in self._asks[:levels]
            ],
        }

    def imbalance(self, levels: int = 5) -> float:
        """Order-book imbalance: ``(bid_vol - ask_vol) / total``.

        Returns a value in ``[-1, 1]``.  Positive values indicate heavier
        buying pressure; negative values indicate selling pressure.

        Parameters
        ----------
        levels:
            Number of price levels to include in the calculation.
        """
        bid_vol = sum(lv.quantity for lv in self._bids[:levels])
        ask_vol = sum(lv.quantity for lv in self._asks[:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    # ------------------------------------------------------------------
    # VPIN (Volume-synchronized Probability of Informed Trading)
    # ------------------------------------------------------------------

    def record_trade(self, price: float, quantity: float) -> None:
        """Record an executed trade for VPIN computation.

        Trades are classified as buy or sell using the tick rule relative to
        the current mid price.
        """
        mid = self.mid_price
        if mid is None:
            return

        buy_vol = quantity if price >= mid else 0.0
        sell_vol = quantity if price < mid else 0.0

        self._buy_volume_buckets.append(buy_vol)
        self._sell_volume_buckets.append(sell_vol)
        self._bucket_total.append(quantity)

    def vpin(self, window: int = 50) -> float:
        """Volume-synchronized Probability of Informed Trading.

        Estimates the fraction of trading volume attributable to informed
        traders over the last ``window`` trade buckets.

        Returns a value in ``[0, 1]`` where higher values indicate more
        informed-trader presence.

        Parameters
        ----------
        window:
            Number of recent trade observations to include.
        """
        n = min(window, len(self._buy_volume_buckets))
        if n == 0:
            return 0.0

        # Slice the most recent n buckets
        buy_vols = list(self._buy_volume_buckets)[-n:]
        sell_vols = list(self._sell_volume_buckets)[-n:]
        totals = list(self._bucket_total)[-n:]

        total_volume = sum(totals)
        if total_volume == 0:
            return 0.0

        abs_diff_sum = sum(
            abs(bv - sv) for bv, sv in zip(buy_vols, sell_vols)
        )

        vpin_value = abs_diff_sum / total_volume
        return min(max(vpin_value, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def trim_history(self, max_buckets: int = 1000) -> None:
        """Discard old trade buckets to bound memory usage."""
        while len(self._buy_volume_buckets) > max_buckets:
            self._buy_volume_buckets.popleft()
            self._sell_volume_buckets.popleft()
            self._bucket_total.popleft()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<OrderBook symbol={self.symbol!r} "
            f"bid={self.best_bid} ask={self.best_ask} "
            f"levels={len(self._bids)}/{len(self._asks)}>"
        )
