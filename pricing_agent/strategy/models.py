"""
Strategy adjustment layer data models.

Covers the full closed-loop data flow:
  Langfuse signals → proposals → guardrails → execution → audit → rollback
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ── ① Langfuse 归因信号 ────────────────────────────────────────

ANOMALY_CATEGORIES = [
    "bug_loop",
    "model_overqualified",
    "budget_risk",
    "key_abuse",
]


@dataclass
class TeamEfficiencyMetrics:
    team_id: str
    feature: str
    total_cost: float
    total_tokens: int
    total_requests: int
    avg_tokens_per_interaction: float
    token_baseline: float
    token_deviation_pct: float
    cost_by_model: dict[str, float] = field(default_factory=dict)
    repeat_call_rate: float = 0.0
    evaluation_score: Optional[float] = None
    cost_per_quality_point: float = 0.0
    period_days: int = 7

    @property
    def cost_per_request(self) -> float:
        return self.total_cost / max(self.total_requests, 1)


@dataclass
class ModelSelectionSignal:
    team_id: str
    feature: str
    task_type: str
    current_model: str
    recommended_model: str
    current_cost_per_call: float
    recommended_cost_per_call: float
    quality_diff: Optional[float] = None
    quality_diff_description: str = ""
    cost_savings_pct: float = 0.0
    cost_savings_monthly: float = 0.0
    confidence: str = "medium"
    request_count: int = 0

    @property
    def savings_per_call(self) -> float:
        return self.current_cost_per_call - self.recommended_cost_per_call


@dataclass
class AnomalySignal:
    anomaly_type: str  # one of ANOMALY_CATEGORIES
    severity: str  # low / medium / high / critical
    team_id: str = ""
    feature: str = ""
    model: str = ""
    description: str = ""
    evidence: dict = field(default_factory=dict)
    suggested_action: str = ""
    detected_at: str = ""


# ── ② 变更提案 ─────────────────────────────────────────────────

PROPOSAL_TYPES = [
    "route_change",
    "quota_adjustment",
    "model_fallback",
    "budget_alert",
    "manual_review_required",
]

RISK_LEVELS = ["low", "medium", "high"]

APPROVAL_GATES = ["none", "team_lead", "admin"]


@dataclass
class RollbackCondition:
    metric: str = "evaluation_score"
    drop_pct: float = 5.0
    window_hours: int = 24
    min_samples: int = 10


@dataclass
class ChangeProposal:
    proposal_id: str
    proposal_type: str  # PROPOSAL_TYPES
    target: dict = field(default_factory=dict)
    current_state: str = ""
    suggested_state: str = ""
    risk_level: str = "low"
    expected_savings: float = 0.0
    expected_savings_currency: str = "USD"
    supporting_signals: list = field(default_factory=list)
    human_readable_summary: str = ""
    auto_executable: bool = False
    requires_approval: str = "none"
    created_at: str = ""
    rollback_condition: Optional[RollbackCondition] = None
    status: str = "pending"  # pending / approved / rejected / executed / rolled_back / failed

    @classmethod
    def new(
        cls,
        proposal_type: str,
        target: dict,
        current_state: str,
        suggested_state: str,
        **kw,
    ) -> "ChangeProposal":
        return cls(
            proposal_id=_generate_id(),
            proposal_type=proposal_type,
            target=target,
            current_state=current_state,
            suggested_state=suggested_state,
            created_at=datetime.now(timezone.utc).isoformat(),
            **kw,
        )

    def summary(self) -> str:
        return (
            f"[{self.proposal_id}] {self.proposal_type} | "
            f"{self.target.get('team_id', '?')}/{self.target.get('feature', '?')} | "
            f"risk={self.risk_level} auto={self.auto_executable} "
            f"approval={self.requires_approval} | status={self.status}"
        )


# ── ③ 安全护栏结果 ─────────────────────────────────────────────


@dataclass
class GuardrailResult:
    passed: bool = False
    failures: list[str] = field(default_factory=list)
    amplitude_check: Optional[dict] = None
    cooldown_check: Optional[dict] = None
    approval_required: str = "none"
    approval_hint: str = ""


# ── ④ 审计记录 ─────────────────────────────────────────────────


@dataclass
class AuditRecord:
    record_id: str
    proposal_id: str
    action: str  # proposed / approved / rejected / executed / rolled_back / failed / expired
    timestamp: str
    actor: str = "system"
    details: str = ""
    data_snapshot: dict = field(default_factory=dict)

    @classmethod
    def make(cls, proposal_id: str, action: str, **kw) -> "AuditRecord":
        return cls(
            record_id=_generate_id(),
            proposal_id=proposal_id,
            action=action,
            timestamp=datetime.now(timezone.utc).isoformat(),
            **kw,
        )


# ── ⑤ 执行结果 ─────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    success: bool
    proposal_id: str
    action_taken: str = ""
    response_data: dict = field(default_factory=dict)
    error: str = ""


# ── ⑥ 滚动监控记录 ─────────────────────────────────────────────


@dataclass
class RollbackMonitor:
    proposal_id: str
    condition: RollbackCondition
    start_time: str
    current_baseline: float = 0.0
    current_value: float = 0.0
    samples_collected: int = 0
    triggered: bool = False
    resolved: bool = False


# ── helper ──────────────────────────────────────────────────────


import hashlib
import time


def _generate_id() -> str:
    raw = f"{time.time_ns()}{id({})}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
