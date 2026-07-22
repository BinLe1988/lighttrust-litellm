"""
归因信号聚合层。

职责:
  接收 Langfuse 原始 traces/observations/scores,
  按团队/功能维度聚合为三类结构化信号:
    1. TeamEfficiencyMetrics  — 效率指标
    2. ModelSelectionSignal   — 模型选型合理性
    3. AnomalySignal          — 异常模式分类

异常分类标签:
  bug_loop              — 同一 session 内相似 prompt 高频重复
  model_overqualified   — 简单任务长期使用旗舰模型,质量分无显著差异
  budget_risk           — 用量趋势指向超支
  key_abuse             — 异常时段大量调用,疑似规避行为
"""

import logging
from typing import Optional
from collections import defaultdict

from .models import (
    TeamEfficiencyMetrics,
    ModelSelectionSignal,
    AnomalySignal,
    ANOMALY_CATEGORIES,
)
from .langfuse import LangfuseClient

logger = logging.getLogger("strategy.signals")

SIMPLE_MODELS = {"deepseek-v4-flash", "gpt-4o-mini", "claude-3-haiku"}
FLAGSHIP_MODELS = {"deepseek-v4-pro", "gpt-4o", "claude-3-opus", "gpt-4-turbo"}


class SignalAggregator:
    """Aggregate Langfuse data into structured signals."""

    def __init__(self, lf: LangfuseClient):
        self._lf = lf

    # ── ① 团队效率指标 ──────────────────────────────────────

    def team_efficiency(
        self,
        team_id: str,
        feature: Optional[str] = None,
        days: int = 7,
    ) -> list[TeamEfficiencyMetrics]:
        """Build efficiency metrics for a given team/feature.

        Expects Langfuse traces to have metadata.team_id and
        metadata.feature set (enforced by MetadataValidator).
        """
        traces = self._lf.traces_in_period(days)
        grouped: dict[str, dict] = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_requests": 0,
            "cost_by_model": defaultdict(float),
            "sessions": defaultdict(list),
            "scores": [],
        })

        for t in traces:
            md = t.get("metadata") or {}
            tid = md.get("team_id", "default")
            feat = md.get("feature", "general")
            if team_id and tid != team_id:
                continue
            if feature and feat != feature:
                continue
            key = f"{tid}:{feat}"
            g = grouped[key]
            g["total_cost"] += t.get("totalCost", 0) or 0
            g["total_requests"] += 1
            g["sessions"][t.get("sessionId", "none")].append(t.get("id", ""))

            # tokens from observations
            gens = self._lf.generations_for_trace(t.get("id", ""))
            for gen in gens:
                g["total_tokens"] += (gen.get("usage", {}) or {}).get("totalTokens", 0) or 0
                model = gen.get("model", "unknown")
                # cost per generation
                cost = gen.get("cost", 0) or 0
                g["cost_by_model"][model] += cost

            # scores
            scores = self._lf.scores_for_trace(t.get("id", ""))
            g["scores"].extend(scores)

        results = []
        for key, g in grouped.items():
            tid, feat = key.split(":", 1)
            total_cost = g["total_cost"]
            total_req = g["total_requests"]
            total_tok = g["total_tokens"]
            avg_tok = total_tok / max(total_req, 1)
            baseline = avg_tok * 1.0  # simple: use avg as baseline for now
            dev = ((avg_tok - baseline) / max(baseline, 1)) * 100

            # repeat call rate
            sessions = g["sessions"]
            multi_call_sessions = sum(1 for v in sessions.values() if len(v) > 1)
            repeat_rate = multi_call_sessions / max(len(sessions), 1)

            # evaluation score
            all_scores = g["scores"]
            avg_score = None
            if all_scores:
                scores_vals = [s.get("value", 0) for s in all_scores if s.get("value") is not None]
                if scores_vals:
                    avg_score = sum(scores_vals) / len(scores_vals)

            results.append(TeamEfficiencyMetrics(
                team_id=tid,
                feature=feat,
                total_cost=total_cost,
                total_tokens=total_tok,
                total_requests=total_req,
                avg_tokens_per_interaction=avg_tok,
                token_baseline=baseline,
                token_deviation_pct=round(dev, 2),
                cost_by_model=dict(g["cost_by_model"]),
                repeat_call_rate=round(repeat_rate, 4),
                evaluation_score=avg_score,
                period_days=days,
            ))
        return results

    # ── ② 模型选型合理性 ────────────────────────────────────

    def model_selection_signals(
        self,
        team_id: str,
        days: int = 30,
        min_requests: int = 50,
    ) -> list[ModelSelectionSignal]:
        """Detect cases where a simpler/cheaper model could replace a flagship one.

        Criteria:
          - Same team/feature uses both a flagship and a simple model for similar tasks
          - Quality difference (when scores available) is < 5%
          - Request count >= min_requests on the flagship model
        """
        traces = self._lf.traces_in_period(days)
        def _empty_usage():
            return {"count": 0, "cost": 0.0, "scores": []}
        usage: dict[tuple[str, str, str, str], dict] = defaultdict(_empty_usage)

        for t in traces:
            md = t.get("metadata") or {}
            tid = md.get("team_id", "default")
            feat = md.get("feature", "general")
            task = md.get("task_type", "general")
            if team_id and tid != team_id:
                continue

            gens = self._lf.generations_for_trace(t.get("id", ""))
            for gen in gens:
                model = gen.get("model", "unknown")
                key = (tid, feat, task, model)
                usage[key]["count"] += 1
                usage[key]["cost"] += gen.get("cost", 0) or 0

                scores = self._lf.scores_for_trace(t.get("id", ""))
                usage[key]["scores"].extend(scores)

        signals = []
        # For each (team, feature, task), check overqualified models
        grouped_tasks: dict[tuple[str, str, str], dict] = defaultdict(dict)
        for (tid, feat, task, model), data in usage.items():
            grouped_tasks[(tid, feat, task)][model] = data

        for (tid, feat, task), models in grouped_tasks.items():
            flagship_models = {m: d for m, d in models.items() if m.lower() in FLAGSHIP_MODELS or m.lower() in SIMPLE_MODELS}
            for m_name, m_data in flagship_models.items():
                if m_data["count"] < min_requests:
                    continue
                is_flagship = m_name.lower() in FLAGSHIP_MODELS
                # Find a counterpart in the opposite tier
                for other_name, other_data in models.items():
                    if other_name == m_name:
                        continue
                    other_is_flagship = other_name.lower() in FLAGSHIP_MODELS
                    if is_flagship == other_is_flagship:
                        continue  # same tier, skip

                    current = m_data if is_flagship else other_data
                    recommended = other_data if is_flagship else m_data
                    current_model = m_name if is_flagship else other_name
                    recommended_model = other_name if is_flagship else m_name

                    # compare scores
                    quality_diff = None
                    qd_desc = ""
                    current_quality = None
                    recommended_quality = None
                    for dset_name, dset in [("current", current), ("recommended", recommended)]:
                        sc = dset.get("scores", [])
                        vals = [s.get("value", 0) for s in sc if s.get("value") is not None]
                        if vals:
                            avg = sum(vals) / len(vals)
                            if dset_name == "current":
                                current_quality = avg
                            else:
                                recommended_quality = avg
                    if current_quality is not None and recommended_quality is not None:
                        quality_diff = current_quality - recommended_quality
                        if abs(quality_diff) < 5:  # less than 5% difference
                            qd_desc = f"质量分差异 < 5% (当前={current_quality:.1f}, 推荐={recommended_quality:.1f})"
                        elif quality_diff < 0:
                            qd_desc = f"推荐模型质量分更高 ({recommended_quality:.1f} vs {current_quality:.1f})"
                        else:
                            qd_desc = f"当前模型质量分更高 ({current_quality:.1f} vs {recommended_quality:.1f})"

                    savings_pct = 0.0
                    if recommended["cost"] > 0 and current["cost"] > 0:
                        cost_current_per_call = current["cost"] / current["count"]
                        cost_rec_per_call = recommended["cost"] / recommended["count"]
                        savings_pct = (1 - cost_rec_per_call / max(cost_current_per_call, 0.001)) * 100

                        monthly_requests_est = current["count"] / max(days, 1) * 30
                        savings_monthly = (cost_current_per_call - cost_rec_per_call) * monthly_requests_est

                        if savings_pct > 20:
                            signals.append(ModelSelectionSignal(
                                team_id=tid,
                                feature=feat,
                                task_type=task,
                                current_model=current_model,
                                recommended_model=recommended_model,
                                current_cost_per_call=cost_current_per_call,
                                recommended_cost_per_call=cost_rec_per_call,
                                quality_diff=quality_diff,
                                quality_diff_description=qd_desc,
                                cost_savings_pct=round(savings_pct, 1),
                                cost_savings_monthly=round(savings_monthly, 2),
                                confidence="high" if (quality_diff is None or abs(quality_diff or 0) < 5) else "medium",
                                request_count=current["count"],
                            ))
        return signals

    # ── ③ 异常模式检测 ──────────────────────────────────────

    def anomaly_signals(
        self,
        team_id: str,
        days: int = 7,
    ) -> list[AnomalySignal]:
        """Detect and classify anomaly patterns."""
        traces = self._lf.traces_in_period(days)
        signals = []
        now = datetime.now(timezone.utc).isoformat()

        # Session-level analysis for bug_loop
        sessions: dict[str, list[dict]] = defaultdict(list)
        for t in traces:
            sid = t.get("sessionId")
            if sid:
                sessions[sid].append(t)

        for sid, sts in sessions.items():
            if len(sts) < 3:
                continue
            # Check for similar prompts in same session
            prompts_seen = set()
            repeats = 0
            for t in sts:
                inp = t.get("input") or ""
                inp_str = str(inp)[:200]
                if inp_str in prompts_seen:
                    repeats += 1
                else:
                    prompts_seen.add(inp_str)
            if repeats >= 3:
                try:
                    tid = sts[0].get("metadata", {}).get("team_id", team_id or "unknown")
                except Exception:
                    tid = team_id or "unknown"
                signals.append(AnomalySignal(
                    anomaly_type="bug_loop",
                    severity="high",
                    team_id=tid,
                    description=f"同一 session ({sid}) 检测到 {repeats} 次相似 prompt 重复,可能为应用层死循环",
                    evidence={"session_id": sid, "repeat_count": repeats, "total_requests": len(sts)},
                    suggested_action="限流该 session 并通知开发团队排查",
                    detected_at=now,
                ))

        # Team-level: budget risk detection
        if team_id:
            efficiency = self.team_efficiency(team_id, days=days)
            for e in efficiency:
                if e.total_requests < 10:
                    continue
                daily_rate = e.total_cost / max(days, 1)
                monthly_projected = daily_rate * 30
                # Simple heuristic: if projected > $1000/month, flag
                if monthly_projected > 1000:
                    signals.append(AnomalySignal(
                        anomaly_type="budget_risk",
                        severity="medium" if monthly_projected < 5000 else "high",
                        team_id=team_id,
                        feature=e.feature,
                        description=(
                            f"团队 {tid} 功能 {e.feature} 月预测花费 ${monthly_projected:.0f}, "
                            f"日均 ${daily_rate:.2f}, 近 {days} 天花费 ${e.total_cost:.2f}"
                        ),
                        evidence={
                            "monthly_projected": round(monthly_projected, 2),
                            "daily_rate": round(daily_rate, 4),
                            "period_cost": round(e.total_cost, 2),
                            "period_days": days,
                            "total_requests": e.total_requests,
                        },
                        suggested_action="审查该功能用量趋势,考虑调整配额或切换模型",
                        detected_at=now,
                    ))

        return signals
