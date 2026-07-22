"""
自动回滚管理器。

职责:
  - 每次变更执行后创建 RollbackMonitor
  - 定时检查观察窗口内的指标是否满足回滚条件
  - 条件触发时自动执行回滚(将配置恢复到变更前状态)

回滚条件:
  - metric (evaluation_score / request_rate / error_rate)
  - drop_pct (指标下降百分比)
  - window_hours (观察时长)
  - min_samples (最少样本数)
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from .models import RollbackMonitor, AuditRecord

logger = logging.getLogger("strategy.rollback")


class RollbackManager:
    """Create monitors and check conditions for auto-rollback."""

    def __init__(self, audit_store=None):
        self._monitors: dict[str, RollbackMonitor] = {}
        self._audit = audit_store

    # ── lifecycle ──────────────────────────────────────────────

    def start_monitoring(
        self,
        proposal_id: str,
        condition: "RollbackCondition",
        initial_baseline: float,
    ) -> RollbackMonitor:
        monitor = RollbackMonitor(
            proposal_id=proposal_id,
            condition=condition,
            start_time=datetime.now(timezone.utc).isoformat(),
            current_baseline=initial_baseline,
        )
        self._monitors[proposal_id] = monitor
        logger.info(
            "RollbackMonitor started for %s: baseline=%.4f, window=%dh, drop>%.1f%%",
            proposal_id, initial_baseline,
            condition.window_hours, condition.drop_pct,
        )
        return monitor

    def record_sample(self, proposal_id: str, current_value: float):
        """Record a new metric sample for an existing monitor."""
        monitor = self._monitors.get(proposal_id)
        if not monitor:
            return
        monitor.current_value = current_value
        monitor.samples_collected += 1

    # ── evaluation ─────────────────────────────────────────────

    def check(self, proposal_id: str) -> Optional[dict]:
        """Check if rollback conditions are met for a given monitor.

        Returns:
          None → conditions not met yet
          dict → rollback decision with details
        """
        monitor = self._monitors.get(proposal_id)
        if not monitor:
            return None
        if monitor.triggered or monitor.resolved:
            return None

        cond = monitor.condition
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(monitor.start_time)).total_seconds()
        elapsed_hours = elapsed / 3600

        if elapsed_hours < cond.window_hours:
            return None  # still in observation window

        if monitor.samples_collected < cond.min_samples:
            logger.info(
                "Rollback check skipped for %s: %d samples < min %d",
                proposal_id, monitor.samples_collected, cond.min_samples,
            )
            return None

        baseline = monitor.current_baseline
        current = monitor.current_value
        if baseline <= 0:
            return None

        drop_pct = (baseline - current) / baseline * 100
        if drop_pct >= cond.drop_pct:
            decision = {
                "rollback": True,
                "proposal_id": proposal_id,
                "metric": cond.metric,
                "baseline": baseline,
                "current": current,
                "drop_pct": round(drop_pct, 2),
                "threshold": cond.drop_pct,
                "reason": (
                    f"{cond.metric} 下降 {drop_pct:.1f}% "
                    f"(阈值 {cond.drop_pct:.0f}%), "
                    f"基线 {baseline:.4f} → 当前 {current:.4f}"
                ),
                "window_hours": elapsed_hours,
                "samples": monitor.samples_collected,
            }
            monitor.triggered = True
            logger.warning("Rollback triggered for %s: %s", proposal_id, decision["reason"])

            if self._audit:
                self._audit.log(AuditRecord.make(
                    proposal_id=proposal_id,
                    action="rolled_back",
                    actor="system",
                    details=decision["reason"],
                    data_snapshot=decision,
                ))
            return decision

        # Conditions not met → resolve the monitor
        resolution_pct = (current - baseline) / baseline * 100
        if resolution_pct >= 0:
            monitor.resolved = True
            logger.info(
                "Rollback monitor resolved for %s: %s improved %.1f%% above baseline",
                proposal_id, cond.metric, resolution_pct,
            )

        return None

    def all_monitors(self) -> dict[str, RollbackMonitor]:
        return dict(self._monitors)
