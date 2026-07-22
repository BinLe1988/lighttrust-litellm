"""
提案引擎。

从归因信号生成具体的配置变更提案,每个提案包含:
  - 类型 (route_change / quota_adjustment / model_fallback / budget_alert)
  - 目标 (team_id, feature, model)
  - 建议变更 + 预期收益
  - 风险等级 + 是否可自动执行
  - 回滚条件
"""

import logging
from typing import Optional
from .models import (
    ChangeProposal,
    RollbackCondition,
    ModelSelectionSignal,
    AnomalySignal,
    TeamEfficiencyMetrics,
)

logger = logging.getLogger("strategy.proposals")


class ProposalEngine:
    """Generate change proposals from aggregated signals."""

    def __init__(self):
        self._proposals: list[ChangeProposal] = []

    # ── route_change: model overqualified ──────────────────────

    def from_model_selection(
        self,
        signal: ModelSelectionSignal,
    ) -> Optional[ChangeProposal]:
        """Generate route-change proposal when a flagship model is overqualified."""
        if signal.cost_savings_pct < 20:
            return None  # not worth proposing

        risk = "low"
        approval = "none"
        if signal.request_count < 100:
            risk = "medium"
            approval = "team_lead"
        if signal.quality_diff and signal.quality_diff > 5:
            risk = "high"
            approval = "admin"
            logger.info(
                "Quality diff >5%% for %s/%s, requiring admin approval",
                signal.team_id, signal.feature,
            )

        proposal = ChangeProposal.new(
            proposal_type="route_change",
            target={
                "team_id": signal.team_id,
                "feature": signal.feature,
                "task_type": signal.task_type,
                "current_model": signal.current_model,
                "recommended_model": signal.recommended_model,
            },
            current_state=f"Default model: {signal.current_model}",
            suggested_state=f"Default model: {signal.recommended_model}",
            risk_level=risk,
            expected_savings=signal.cost_savings_monthly,
            supporting_signals=[signal],
            auto_executable=risk == "low",
            requires_approval=approval,
            rollback_condition=RollbackCondition(
                metric="evaluation_score",
                drop_pct=5.0,
                window_hours=24,
                min_samples=10,
            ),
        )
        self._proposals.append(proposal)
        logger.info("Proposal %s: route %s/%s → %s", proposal.proposal_id, signal.team_id, signal.feature, signal.recommended_model)
        return proposal

    # ── model_fallback: anomaly detected ───────────────────────

    def from_anomaly(self, signal: AnomalySignal) -> Optional[ChangeProposal]:
        """Generate proposals from anomaly signals.

        Different anomaly types → different proposal types:
          bug_loop            → quota_adjustment (rate-limit the session/team)
          model_overqualified → route_change (already handled above)
          budget_risk         → budget_alert
          key_abuse           → manual_review_required
        """
        type_map = {
            "bug_loop": "quota_adjustment",
            "budget_risk": "budget_alert",
            "key_abuse": "manual_review_required",
            "model_overqualified": "route_change",
        }
        proposal_type = type_map.get(signal.anomaly_type, "manual_review_required")

        risk_map = {"low": "low", "medium": "medium", "high": "high", "critical": "high"}
        risk = risk_map.get(signal.severity, "medium")

        auto_map = {"low": True, "medium": True, "high": False, "critical": False}
        auto_exec = auto_map.get(signal.severity, False)

        approval_map = {"low": "none", "medium": "none", "high": "team_lead", "critical": "admin"}
        approval = approval_map.get(signal.severity, "admin")

        current_state = ""
        suggested_state = ""
        if signal.anomaly_type == "bug_loop":
            current_state = f"Session {signal.evidence.get('session_id', '?')}: {signal.evidence.get('repeat_count', 0)} repeated calls"
            suggested_state = "Enforce rate limit on this session/team"
        elif signal.anomaly_type == "budget_risk":
            current_state = f"Monthly projected: ${signal.evidence.get('monthly_projected', 0):.0f}"
            suggested_state = "Review quota or switch to cheaper model"
        elif signal.anomaly_type == "key_abuse":
            current_state = "Suspicious calling pattern detected"
            suggested_state = "Escalate to manual review"

        proposal = ChangeProposal.new(
            proposal_type=proposal_type,
            target={
                "team_id": signal.team_id,
                "feature": signal.feature or "general",
                "anomaly_type": signal.anomaly_type,
            },
            current_state=current_state,
            suggested_state=suggested_state,
            risk_level=risk,
            expected_savings=0.0,
            supporting_signals=[signal],
            auto_executable=auto_exec,
            requires_approval=approval,
            rollback_condition=RollbackCondition(
                metric="request_rate",
                drop_pct=50.0,
                window_hours=2,
                min_samples=5,
            ),
        )
        self._proposals.append(proposal)
        logger.info(
            "Proposal %s (%s): %s for %s",
            proposal.proposal_id, proposal_type, signal.anomaly_type, signal.team_id,
        )
        return proposal

    # ── batch generation ───────────────────────────────────────

    def generate_all(
        self,
        efficiency_signals: Optional[list] = None,
        model_signals: Optional[list[ModelSelectionSignal]] = None,
        anomaly_signals: Optional[list[AnomalySignal]] = None,
    ) -> list[ChangeProposal]:
        """Generate all proposals from all signal types."""
        self._proposals = []

        if model_signals:
            for s in model_signals:
                self.from_model_selection(s)

        if anomaly_signals:
            for s in anomaly_signals:
                # model_overqualified already covered by model signals
                if s.anomaly_type == "model_overqualified":
                    continue
                self.from_anomaly(s)

        return self._proposals

    @property
    def proposals(self) -> list[ChangeProposal]:
        return self._proposals
