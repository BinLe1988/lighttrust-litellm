"""
Sliding-window statistics for rules engine.

Tracks spend rate, request rate, error rate, and latency over
configurable time windows (default 5 min, 1 hour, 1 day).
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List
from .models import RuleContext


@dataclass
class SlidingWindow:
    window_secs: int
    _samples: deque = field(default_factory=deque)

    def add(self, value: float = 1.0):
        self._samples.append((time.time(), value))
        self._trim()

    def _trim(self):
        cutoff = time.time() - self.window_secs
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def count(self) -> int:
        self._trim()
        return len(self._samples)

    @property
    def sum(self) -> float:
        self._trim()
        return sum(v for _, v in self._samples)

    @property
    def rate(self) -> float:
        """Events per second."""
        self._trim()
        if not self._samples:
            return 0.0
        span = self._samples[-1][0] - self._samples[0][0]
        return len(self._samples) / span if span > 0 else 0.0

    @property
    def avg(self) -> float:
        self._trim()
        if not self._samples:
            return 0.0
        return self.sum / len(self._samples)

    def reset(self):
        self._samples.clear()


class PerUserWindows:
    """Sliding windows keyed by user_id."""

    def __init__(self, windows_secs: Optional[List[int]] = None):
        self._windows_secs = windows_secs or [300, 3600, 86400]
        self._data: dict[str, dict[str, SlidingWindow]] = {}

    def _ensure(self, uid: str, kind: str) -> SlidingWindow:
        user = self._data.setdefault(uid, {})
        key = f"{kind}_{self._windows_secs[0]}"
        if key not in user:
            user[key] = SlidingWindow(self._windows_secs[0])
        return user[key]

    def add_spend(self, uid: str, amount: float):
        for secs in self._windows_secs:
            user = self._data.setdefault(uid, {})
            w = user.setdefault(f"spend_{secs}", SlidingWindow(secs))
            w.add(amount)

    def add_request(self, uid: str, is_error: bool = False):
        for secs in self._windows_secs:
            user = self._data.setdefault(uid, {})
            w = user.setdefault(f"req_{secs}", SlidingWindow(secs))
            w.add(1.0)
            if is_error:
                ew = user.setdefault(f"err_{secs}", SlidingWindow(secs))
                ew.add(1.0)

    def spend_rate_short(self, uid: str) -> float:
        w = self._data.get(uid, {}).get(f"spend_{self._windows_secs[0]}")
        return w.rate if w else 0.0

    def spend_rate_long(self, uid: str) -> float:
        w = self._data.get(uid, {}).get(f"spend_{self._windows_secs[-1]}")
        return w.rate if w else 0.0

    def request_rate(self, uid: str) -> float:
        w = self._data.get(uid, {}).get(f"req_{self._windows_secs[0]}")
        return w.rate if w else 0.0

    def error_rate(self, uid: str) -> float:
        w = self._data.get(uid, {}).get(f"err_{self._windows_secs[0]}")
        req = self._data.get(uid, {}).get(f"req_{self._windows_secs[0]}")
        if not req or req.count == 0:
            return 0.0
        return w.count / req.count if w else 0.0

    def count_short(self, uid: str) -> int:
        w = self._data.get(uid, {}).get(f"req_{self._windows_secs[0]}")
        return w.count if w else 0

    def build_context(self, uid: str, current_spend: float, max_budget: float,
                      default_budget: float = 0.0, days_elapsed: int = 1,
                      days_in_month: int = 30) -> RuleContext:
        sr_short = self.spend_rate_short(uid)
        sr_long = self.spend_rate_long(uid)
        monthly = current_spend
        predicted = monthly
        if days_elapsed > 0 and days_in_month > days_elapsed:
            daily_rate = monthly / days_elapsed
            predicted = daily_rate * days_in_month
        return RuleContext(
            user_id=uid,
            current_spend=current_spend,
            max_budget=max_budget,
            default_budget=default_budget,
            monthly_spend=monthly,
            predicted_monthly_spend=predicted,
            spend_rate=sr_short,
            spend_rate_baseline=sr_long,
            request_rate=self.request_rate(uid),
            request_rate_baseline=0.0,
            error_rate=self.error_rate(uid),
            request_count=self.count_short(uid),
            days_elapsed=days_elapsed,
            days_in_month=days_in_month,
        )
