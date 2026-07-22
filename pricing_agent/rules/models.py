from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Condition:
    metric: str = ""
    operator: str = ">="
    value: float = 0.0
    multiplier: float = 0.0
    min_samples: int = 0
    min_days_elapsed: int = 0


@dataclass
class RuleAction:
    action: str = "notify"
    channels: list[str] = field(default_factory=lambda: ["dingtalk"])
    message: str = ""
    throttle_pct: float = 0.0
    fallback_model: str = ""
    status_code: int = 429


@dataclass
class Rule:
    name: str
    description: str = ""
    enabled: bool = True
    type: str = "budget_threshold"
    condition: Optional[Condition] = None
    actions: list[RuleAction] = field(default_factory=list)
    cooldown: int = 0
    last_fired: float = 0.0


@dataclass
class RuleContext:
    user_id: str = "default"
    team_id: str = ""
    model: str = ""
    provider: str = ""
    call_type: str = ""

    current_spend: float = 0.0
    max_budget: float = 0.0
    default_budget: float = 0.0

    daily_spend: float = 0.0
    monthly_spend: float = 0.0
    predicted_monthly_spend: float = 0.0

    spend_rate: float = 0.0
    spend_rate_baseline: float = 0.0
    request_rate: float = 0.0
    request_rate_baseline: float = 0.0
    error_rate: float = 0.0
    error_rate_baseline: float = 0.0
    avg_latency: float = 0.0

    request_count: int = 0
    error_count: int = 0
    days_in_month: int = 30
    days_elapsed: int = 1

    @property
    def spend_ratio(self) -> float:
        b = self.max_budget or self.default_budget or 1.0
        return self.current_spend / b

    @property
    def predicted_ratio(self) -> float:
        b = self.max_budget or self.default_budget or 1.0
        return self.predicted_monthly_spend / b

    @property
    def daily_spend_ratio(self) -> float:
        b = self.max_budget or self.default_budget or 1.0
        daily_alloc = b / max(self.days_in_month, 1)
        return self.daily_spend / daily_alloc if daily_alloc else 0.0


@dataclass
class RuleResult:
    rule_name: str
    rule_type: str
    triggered: bool
    actions_taken: list[str] = field(default_factory=list)
    message: str = ""
    degrade_throttle_pct: float = 0.0
    degrade_fallback_model: str = ""
    reject: bool = False
    reject_status: int = 429
    cooldown: float = 0.0
