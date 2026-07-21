from abc import ABC, abstractmethod
from typing import Optional
from ..models import PriceSnapshot


class AbstractMonitor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def fetch(self) -> PriceSnapshot:
        ...

    def default_snapshot(self, reason: str) -> PriceSnapshot:
        snap = PriceSnapshot.now(self.name)
        for m in snap.models:
            m.notes = reason
        return snap
