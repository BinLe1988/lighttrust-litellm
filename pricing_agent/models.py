from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ModelPrice:
    model_name: str
    provider: str
    input_cost_per_token: float
    output_cost_per_token: float
    cache_read_input_token_cost: Optional[float] = None
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    effective_date: Optional[str] = None
    source: Optional[str] = None
    notes: str = ""


@dataclass
class PriceSnapshot:
    provider: str
    fetched_at: str
    models: list[ModelPrice] = field(default_factory=list)

    @classmethod
    def now(cls, provider: str) -> "PriceSnapshot":
        return cls(
            provider=provider,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )


CHANGE_TYPES = [
    "price_change",
    "model_added",
    "model_deprecated",
    "caching_discount",
    "config_mismatch",
]


@dataclass
class PriceChange:
    change_type: str
    model_name: str
    field: Optional[str] = None
    old_value: Optional[float] = None
    new_value: Optional[float] = None
    description: str = ""
    impact: str = ""
    suggested_action: str = ""

    def detail(self) -> str:
        pfx = f"[{self.change_type}] {self.model_name}"
        if self.field:
            pfx += f" {self.field}"
        out = f"{pfx}: {self.description}"
        if self.impact:
            out += f"\n    impact: {self.impact}"
        if self.suggested_action:
            out += f"\n    action: {self.suggested_action}"
        return out


@dataclass
class PendingChange:
    change_id: str
    created_at: str
    provider: str
    changes: list[PriceChange] = field(default_factory=list)
    status: str = "pending"
    approved_at: Optional[str] = None
    rejected_at: Optional[str] = None
    reject_reason: str = ""

    def summary(self) -> str:
        tc = len(self.changes)
        return (
            f"[{self.change_id}] {self.provider} | "
            f"{tc} change{'s' if tc != 1 else ''} | "
            f"status={self.status}"
        )
