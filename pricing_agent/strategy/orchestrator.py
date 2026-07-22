"""
主编排闭环。

完整流程:
  Langfuse → 信号聚合 → 提案生成 → 安全护栏 → 解释 → 执行 → 审计 → 回滚监控

调用方式:
  orchestrator = StrategyOrchestrator()
  result = orchestrator.run(team_id="team-a")
"""

import json
import logging
import os
import sys
from typing import Optional

from .langfuse import LangfuseClient, LangfuseConfig
from .signals import SignalAggregator
from .proposals import ProposalEngine
from .guardrails import GuardrailEngine
from .explainer import explain_proposal
from .executor import Executor
from .rollback import RollbackManager
from .audit import AuditStore
from .models import (
    ChangeProposal,
    ExecutionResult,
    GuardrailResult,
    AuditRecord,
)

logger = logging.getLogger("strategy.orchestrator")


class StrategyOrchestrator:
    """Main closed-loop orchestrator."""

    def __init__(
        self,
        lf_client: Optional[LangfuseClient] = None,
        executor: Optional[Executor] = None,
        audit: Optional[AuditStore] = None,
    ):
        self._lf = lf_client or LangfuseClient()
        self._signals = SignalAggregator(self._lf)
        self._proposals = ProposalEngine()
        self._guardrails = GuardrailEngine()
        self._executor = executor or Executor()
        self._audit = audit or AuditStore()
        self._rollback = RollbackManager(audit_store=self._audit)

    # ── main loop ──────────────────────────────────────────────

    def run(
        self,
        team_id: str = "",
        feature: str = "",
        days: int = 7,
        dry_run: bool = True,
        explain: bool = True,
    ) -> dict:
        """Run one full closed-loop cycle.

        Args:
          team_id:    Target team (empty = all teams)
          feature:    Target feature (empty = all features)
          days:       Lookback window in days
          dry_run:    If True, skip actual execution (just validate)
          explain:    If True, generate LLM explanations

        Returns:
          dict with full cycle results
        """
        logger.info(
            "Starting strategy cycle: team=%s feature=%s days=%d dry=%s",
            team_id or "*", feature or "*", days, dry_run,
        )

        # Step 1: Gather signals from Langfuse
        logger.info("Step 1: Gathering signals from Langfuse...")
        efficiency = self._signals.team_efficiency(team_id, feature, days)
        model_signals = self._signals.model_selection_signals(team_id, days=min(days, 30))
        anomaly_signals = self._signals.anomaly_signals(team_id, days)

        logger.info(
            "  efficiency=%d model_signals=%d anomaly_signals=%d",
            len(efficiency), len(model_signals), len(anomaly_signals),
        )

        # Step 2: Generate proposals
        logger.info("Step 2: Generating proposals...")
        proposals = self._proposals.generate_all(
            efficiency_signals=efficiency,
            model_signals=model_signals,
            anomaly_signals=anomaly_signals,
        )
        logger.info("  proposals=%d", len(proposals))

        # Step 3 ~ 7: Process each proposal
        results = []
        for proposal in proposals:
            result = self._process_proposal(proposal, dry_run, explain)
            results.append(result)

        # Step 8: Check rollback monitors for previous executions
        rollback_decisions = self._check_rollbacks()

        return {
            "cycle_info": {
                "team_id": team_id,
                "feature": feature,
                "days": days,
                "dry_run": dry_run,
            },
            "signals": {
                "efficiency": [s.__dict__ for s in efficiency],
                "model_selection": [s.__dict__ for s in model_signals],
                "anomaly": [s.__dict__ for s in anomaly_signals],
            },
            "proposals": results,
            "rollback_decisions": rollback_decisions,
        }

    # ── single proposal pipeline ───────────────────────────────

    def _process_proposal(
        self,
        proposal: ChangeProposal,
        dry_run: bool,
        explain: bool,
    ) -> dict:
        logger.info("  Processing proposal %s (%s)...", proposal.proposal_id, proposal.proposal_type)

        # Log the proposal
        self._audit.log(AuditRecord.make(
            proposal_id=proposal.proposal_id,
            action="proposed",
            actor="system",
            details=f"Type={proposal.proposal_type} target={proposal.target}",
            data_snapshot=proposal.__dict__,
        ))

        # Guardrails
        guardrail = self._guardrails.evaluate(proposal)
        logger.info(
            "    guardrail: passed=%s failures=%s approval=%s",
            guardrail.passed, guardrail.failures, guardrail.approval_required,
        )

        if not guardrail.passed:
            self._audit.log(AuditRecord.make(
                proposal_id=proposal.proposal_id,
                action="failed",
                actor="system",
                details=f"Guardrail blocked: {guardrail.failures}",
                data_snapshot=guardrail.__dict__,
            ))

        # Explanation
        explanation = ""
        if explain:
            explanation = explain_proposal(proposal, guardrail)

        # Execution (dry run or actual)
        execution: Optional[ExecutionResult] = None
        if guardrail.passed and proposal.auto_executable:
            if dry_run:
                logger.info("    dry-run: validating without executing...")
                execution = self._executor.dry_run(proposal)
                self._audit.log(AuditRecord.make(
                    proposal_id=proposal.proposal_id,
                    action="dry_run" if dry_run else "executed",
                    actor="system",
                    details=f"Dry-run: {'success' if execution.success else 'failed'}: {execution.error or 'ok'}",
                    data_snapshot=execution.__dict__,
                ))
            else:
                logger.info("    executing...")
                execution = self._executor.execute(proposal)
                action = "executed" if execution.success else "failed"
                self._audit.log(AuditRecord.make(
                    proposal_id=proposal.proposal_id,
                    action=action,
                    actor="system",
                    details=f"Execute: {execution.action_taken}: {execution.error or 'ok'}",
                    data_snapshot=execution.__dict__,
                ))

                # Start rollback monitoring if execution succeeded
                if execution.success and proposal.rollback_condition:
                    baseline = proposal.rollback_condition.metric == "evaluation_score" and 100.0 or 100.0
                    self._rollback.start_monitoring(
                        proposal.proposal_id,
                        proposal.rollback_condition,
                        initial_baseline=baseline,
                    )
                    self._guardrails.record_execution(proposal)

        # Mark approval if needed
        if guardrail.approval_required != "none":
            self._audit.log(AuditRecord.make(
                proposal_id=proposal.proposal_id,
                action="pending_approval",
                actor="system",
                details=f"Requires {guardrail.approval_required} approval: {guardrail.approval_hint}",
                data_snapshot=guardrail.__dict__,
            ))

        return {
            "proposal": proposal.__dict__,
            "guardrail": guardrail.__dict__,
            "explanation": explanation,
            "execution": execution.__dict__ if execution else None,
        }

    # ── rollback check ─────────────────────────────────────────

    def _check_rollbacks(self) -> list[dict]:
        decisions = []
        for pid in list(self._rollback.all_monitors().keys()):
            decision = self._rollback.check(pid)
            if decision:
                logger.warning("Rollback decision: %s", decision)
                decisions.append(decision)
                # TODO: execute the actual rollback (reverse the config change)
        return decisions

    # ── manual approval workflow ────────────────────────────────

    def approve_proposal(self, proposal_id: str, actor: str = "admin") -> dict:
        """Approve a pending proposal and execute it."""
        records = self._audit.get_chain(proposal_id)
        if not records:
            return {"error": f"Proposal {proposal_id} not found"}

        # Find the proposal from recent logs
        # In real impl, you'd have a proposal store; here we scan audit
        proposed = [r for r in records if r.action == "proposed"]
        if not proposed:
            return {"error": f"No proposal record found for {proposal_id}"}

        self._audit.log(AuditRecord.make(
            proposal_id=proposal_id,
            action="approved",
            actor=actor,
            details=f"Approved by {actor}",
        ))

        # Reconstruct and execute
        data = proposed[-1].data_snapshot
        proposal = ChangeProposal(**data)
        execution = self._executor.execute(proposal)

        self._audit.log(AuditRecord.make(
            proposal_id=proposal_id,
            action="executed" if execution.success else "failed",
            actor="system",
            details=f"Post-approval execute: {execution.action_taken}: {execution.error or 'ok'}",
            data_snapshot=execution.__dict__,
        ))

        return {
            "proposal_id": proposal_id,
            "approved_by": actor,
            "execution": execution.__dict__,
        }

    def reject_proposal(self, proposal_id: str, reason: str, actor: str = "admin") -> dict:
        """Reject a pending proposal."""
        self._audit.log(AuditRecord.make(
            proposal_id=proposal_id,
            action="rejected",
            actor=actor,
            details=f"Rejected by {actor}: {reason}",
            data_snapshot={"reason": reason},
        ))
        return {"proposal_id": proposal_id, "rejected_by": actor, "reason": reason}

    # ── cleanup ────────────────────────────────────────────────

    def close(self):
        self._lf.close()
        self._executor.close()
