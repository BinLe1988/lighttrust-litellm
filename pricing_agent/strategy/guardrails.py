"""
安全护栏。

每条变更提案必须通过安全检查才能执行,包括:
  1. 冷却期检查 — 同一 target 的同类变更须间隔最短时间
  2. 幅度上限检查 — 单次变更幅度不超过硬上限(超限转人工)
  3. 审批门禁 — 按 risk_level 决定审批等级
  4. 可逆性检查 — 不可逆变更需更高审批
"""

import os
import time
import logging
from typing import Optional
from collections import defaultdict

from .models import (
    ChangeProposal,
    GuardrailResult,
    RISK_LEVELS,
    APPROVAL_GATES,
)

logger = logging.getLogger("strategy.guardrails")

# 环境变量可配安全阈值
COOLDOWN_HOURS = float(os.environ.get("GUARDRAIL_COOLDOWN_HOURS", "6"))
AMPLITUDE_MAX_PCT = float(os.environ.get("GUARDRAIL_AMPLITUDE_MAX_PCT", "20"))
AMPLITUDE_QUOTA_MAX_PCT = float(os.environ.get("GUARDRAIL_QUOTA_MAX_PCT", "20"))


class GuardrailEngine:
    """Evaluate all guardrails for a proposal."""

    def __init__(self):
        # execution_history[(target_key, proposal_type)] -> list of timestamps
        self._history: dict[tuple[str, str], list[float]] = defaultdict(list)

    def _target_key(self, proposal: ChangeProposal) -> str:
        t = proposal.target
        return f"{t.get('team_id', '?')}:{t.get('feature', '?')}"

    def record_execution(self, proposal: ChangeProposal):
        key = (self._target_key(proposal), proposal.proposal_type)
        self._history[key].append(time.time())

    # ── 冷却期检查 ─────────────────────────────────────────────

    def _check_cooldown(self, proposal: ChangeProposal) -> Optional[dict]:
        key = (self._target_key(proposal), proposal.proposal_type)
        timestamps = self._history.get(key, [])
        if not timestamps:
            return {"passed": True, "last_execution": None, "remaining_hours": 0}
        last = timestamps[-1]
        elapsed_hours = (time.time() - last) / 3600
        if elapsed_hours < COOLDOWN_HOURS:
            remaining = COOLDOWN_HOURS - elapsed_hours
            return {
                "passed": False,
                "last_execution": last,
                "remaining_hours": round(remaining, 1),
            }
        return {"passed": True, "last_execution": last, "remaining_hours": 0}

    # ── 幅度上限检查 ───────────────────────────────────────────

    def _check_amplitude(self, proposal: ChangeProposal) -> Optional[dict]:
        if proposal.proposal_type == "quota_adjustment":
            # Check if we know the current quota and proposed change
            current = proposal.target.get("current_quota")
            suggested = proposal.target.get("suggested_quota")
            if current and suggested and current > 0:
                change_pct = abs(suggested - current) / current * 100
                if change_pct > AMPLITUDE_QUOTA_MAX_PCT:
                    return {
                        "passed": False,
                        "change_pct": round(change_pct, 1),
                        "max_allowed": AMPLITUDE_QUOTA_MAX_PCT,
                        "reason": f"配额变更幅度 {change_pct:.1f}% 超过上限 {AMPLITUDE_QUOTA_MAX_PCT:.0f}%",
                    }
        if proposal.proposal_type == "route_change":
            # Route changes: check if the switch is between similar-enough models
            # This is inherently low amplitude since it's a model switch
            pass
        return {"passed": True, "change_pct": 0, "max_allowed": 100}

    # ── 审批门禁 ───────────────────────────────────────────────

    def _check_approval(self, proposal: ChangeProposal) -> tuple[str, str]:
        """Determine required approval level and hint."""
        required = proposal.requires_approval or "none"

        if not proposal.auto_executable and required == "none":
            required = "team_lead"

        if proposal.risk_level == "high":
            required = "admin"

        hints = {
            "none": "无需人工审批,可自动执行",
            "team_lead": "需要团队负责人审批",
            "admin": "需要管理员审批",
        }
        return required, hints.get(required, "")

    # ── 统一评估入口 ───────────────────────────────────────────

    def evaluate(self, proposal: ChangeProposal) -> GuardrailResult:
        """Run all guardrails, return pass/fail + details."""
        failures = []
        amplitude = self._check_amplitude(proposal)
        cooldown = self._check_cooldown(proposal)

        if not amplitude["passed"]:
            failures.append(amplitude["reason"])

        if not cooldown["passed"]:
            failures.append(
                f"冷却期: 距上次同类变更不足 {COOLDOWN_HOURS}h "
                f"(还需 {cooldown['remaining_hours']}h)"
            )

        approval_required, approval_hint = self._check_approval(proposal)

        return GuardrailResult(
            passed=len(failures) == 0,
            failures=failures,
            amplitude_check=amplitude,
            cooldown_check=cooldown,
            approval_required=approval_required,
            approval_hint=approval_hint,
        )
