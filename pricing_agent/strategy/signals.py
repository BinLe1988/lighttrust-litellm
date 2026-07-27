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

优化: 使用批量 API 调用,避免 N+1 查询问题。
"""

import logging
from typing import Optional
from collections import defaultdict

from .models import (
    TeamEfficiencyMetrics,
    ModelSelectionSignal,
    AnomalySignal,
    RoutingQualitySignal,
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

    def _build_trace_index(self, days: int) -> dict[str, dict]:
        """Fetch all traces and build a trace_id → trace index."""
        traces = self._lf.traces_in_period(days)
        return {t["id"]: t for t in traces if t.get("id")}

    def _batch_generations(self, days: int) -> dict[str, list[dict]]:
        """Batch-fetch ALL generations for the period, indexed by trace_id."""
        import time
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        since = (now - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        all_gens: dict[str, list[dict]] = defaultdict(list)
        page = 1
        while True:
            data = self._lf.list_observations(
                limit=100, page=page,
                from_timestamp=since,
                observation_type="GENERATION",
            )
            items = data.get("data", [])
            if not items:
                break
            for gen in items:
                tid = gen.get("traceId")
                if tid:
                    all_gens[tid].append(gen)
            meta = data.get("meta", {})
            total = meta.get("totalPages", 1) or 1
            if page >= total:
                break
            page += 1
            time.sleep(3.0)
        return dict(all_gens)

    def _batch_scores(self, days: int) -> dict[str, list[dict]]:
        """Batch-fetch ALL scores for the period, indexed by trace_id."""
        import time
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        since = (now - __import__("datetime").timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        all_scores: dict[str, list[dict]] = defaultdict(list)
        page = 1
        while True:
            data = self._lf.list_scores(
                limit=100, page=page,
                from_timestamp=since,
            )
            items = data.get("data", [])
            if not items:
                break
            for sc in items:
                tid = sc.get("traceId")
                if tid:
                    all_scores[tid].append(sc)
            meta = data.get("meta", {})
            total = meta.get("totalPages", 1) or 1
            if page >= total:
                break
            page += 1
            time.sleep(3.0)
        return dict(all_scores)

    # ── ① 团队效率指标 ──────────────────────────────────────

    def team_efficiency(
        self,
        team_id: str,
        feature: Optional[str] = None,
        days: int = 7,
    ) -> list[TeamEfficiencyMetrics]:
        """Build efficiency metrics for a given team/feature.

        Uses batch API calls (not N+1) for performance.
        """
        traces = self._trace_index = self._build_trace_index(days)
        generations = self._batch_generations(days)
        scores = self._batch_scores(days)

        grouped: dict[str, dict] = defaultdict(lambda: {
            "total_cost": 0.0,
            "total_tokens": 0,
            "total_requests": 0,
            "cost_by_model": defaultdict(float),
            "sessions": defaultdict(list),
            "scores": [],
        })

        for tid, t in traces.items():
            md = t.get("metadata") or {}
            tid_team = md.get("team_id", "default")
            feat = md.get("feature", "general")
            if team_id and tid_team != team_id:
                continue
            if feature and feat != feature:
                continue
            key = f"{tid_team}:{feat}"
            g = grouped[key]
            g["total_cost"] += t.get("totalCost", 0) or 0
            g["total_requests"] += 1
            g["sessions"][t.get("sessionId", "none")].append(tid)

            for gen in generations.get(tid, []):
                g["total_tokens"] += (gen.get("usage", {}) or {}).get("totalTokens", 0) or 0
                model = gen.get("model", "unknown")
                g["cost_by_model"][model] += gen.get("cost", 0) or 0

            g["scores"].extend(scores.get(tid, []))

        results = []
        for key, g in grouped.items():
            tid_name, feat = key.split(":", 1)
            total_cost = g["total_cost"]
            total_req = g["total_requests"]
            total_tok = g["total_tokens"]
            avg_tok = total_tok / max(total_req, 1)
            baseline = avg_tok * 1.0
            dev = ((avg_tok - baseline) / max(baseline, 1)) * 100

            sessions = g["sessions"]
            multi_call_sessions = sum(1 for v in sessions.values() if len(v) > 1)
            repeat_rate = multi_call_sessions / max(len(sessions), 1)

            all_scores_list = g["scores"]
            avg_score = None
            if all_scores_list:
                vals = [s.get("value", 0) for s in all_scores_list if s.get("value") is not None]
                if vals:
                    avg_score = sum(vals) / len(vals)

            results.append(TeamEfficiencyMetrics(
                team_id=tid_name,
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

        Uses batch API calls for performance.
        """
        traces = self._build_trace_index(days)
        generations = self._batch_generations(days)
        scores = self._batch_scores(days)

        usage: dict[tuple[str, str, str, str], dict] = defaultdict(lambda: {"count": 0, "cost": 0.0, "scores": []})

        for tid, t in traces.items():
            md = t.get("metadata") or {}
            uid = md.get("team_id", "default")
            feat = md.get("feature", "general")
            task = md.get("task_type", "general")
            if team_id and uid != team_id:
                continue

            for gen in generations.get(tid, []):
                model = gen.get("model", "unknown")
                key = (uid, feat, task, model)
                usage[key]["count"] += 1
                usage[key]["cost"] += gen.get("cost", 0) or 0

            usage[(uid, feat, task, "scores")]["scores"].extend(scores.get(tid, []))

        signals = []
        grouped_tasks: dict[tuple[str, str, str], dict] = defaultdict(dict)
        for (uid, feat, task, model), data in usage.items():
            if model == "scores":
                continue
            grouped_tasks[(uid, feat, task)][model] = data

        for (uid, feat, task), models in grouped_tasks.items():
            for m_name, m_data in models.items():
                if m_data["count"] < min_requests:
                    continue
                is_flagship = m_name.lower() in FLAGSHIP_MODELS
                if not is_flagship:
                    continue
                for other_name, other_data in models.items():
                    if other_name == m_name:
                        continue
                    other_is_flagship = other_name.lower() in FLAGSHIP_MODELS
                    if is_flagship == other_is_flagship:
                        continue

                    current = m_data
                    recommended = other_data
                    current_quality = None
                    recommended_quality = None

                    for val in current.get("scores", []):
                        v = val.get("value")
                        if v is not None:
                            current_quality = v
                    for val in recommended.get("scores", []):
                        v = val.get("value")
                        if v is not None:
                            recommended_quality = v

                    quality_diff = None
                    qd_desc = ""
                    if current_quality is not None and recommended_quality is not None:
                        quality_diff = current_quality - recommended_quality
                        if abs(quality_diff) < 5:
                            qd_desc = f"质量分差异 < 5% (当前={current_quality:.1f}, 推荐={recommended_quality:.1f})"

                    savings_pct = 0.0
                    if recommended["cost"] > 0 and current["cost"] > 0:
                        cost_current_per_call = current["cost"] / current["count"]
                        cost_rec_per_call = recommended["cost"] / recommended["count"]
                        savings_pct = (1 - cost_rec_per_call / max(cost_current_per_call, 0.001)) * 100
                        monthly_requests_est = current["count"] / max(days, 1) * 30
                        savings_monthly = (cost_current_per_call - cost_rec_per_call) * monthly_requests_est

                        if savings_pct > 20:
                            signals.append(ModelSelectionSignal(
                                team_id=uid,
                                feature=feat,
                                task_type=task,
                                current_model=m_name,
                                recommended_model=other_name,
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

    # ── ④ Prompt Routing 质量评估 ─────────────────────────

    def routing_quality_signals(
        self,
        team_id: str = "",
        days: int = 7,
    ) -> list[RoutingQualitySignal]:
        """分析路由决策质量。

        从 Langfuse 找出含有 routing_* metadata 的 trace，
        评估:
          - 各 category 的分布
          - 路由后的成本效率
          - 路由错误模式（基于 score / error）
        """
        traces = self._build_trace_index(days)
        generations = self._batch_generations(days)
        scores = self._batch_scores(days)

        grouped: dict[str, dict] = defaultdict(lambda: {
            "routed": 0, "total": 0,
            "cost_routed": 0.0, "cost_total": 0.0,
            "errors_routed": 0, "errors_total": 0,
            "categories": defaultdict(int),
            "misroutes": defaultdict(int),
            "error_messages": [],
        })

        for tid, t in traces.items():
            md = t.get("metadata") or {}
            tid_team = md.get("team_id", "default")
            if team_id and tid_team != team_id:
                continue
            feat = md.get("feature", "general")
            key = f"{tid_team}:{feat}"
            g = grouped[key]
            g["total"] += 1

            is_routed = bool(md.get("routing_method"))
            if is_routed:
                g["routed"] += 1
                cat = md.get("routing_category", "unknown")
                g["categories"][cat] += 1

                trace_scores = scores.get(tid, [])
                for sc in trace_scores:
                    if sc.get("name", "").startswith("routing_"):
                        val = sc.get("value", 0)
                        if val is not None and val < 0.5:
                            g["misroutes"][cat] += 1
                            g["error_messages"].append(
                                f"trace={tid} category={cat} score={val}"
                            )

            for gen in generations.get(tid, []):
                cost = gen.get("cost", 0) or 0
                if is_routed:
                    g["cost_routed"] += cost
                g["cost_total"] += cost

            if t.get("status") == "ERROR":
                if is_routed:
                    g["errors_routed"] += 1
                g["errors_total"] += 1

        results = []
        for key, g in grouped.items():
            tid_name, feat = key.split(":", 1)
            if g["routed"] < 10:
                continue
            accuracy = 1.0
            total_misroutes = sum(g["misroutes"].values())
            if g["routed"] > 0:
                accuracy = max(0.0, 1.0 - total_misroutes / g["routed"])
            avg_cost = g["cost_routed"] / max(g["routed"], 1)
            avg_cost_all = g["cost_total"] / max(g["total"], 1)
            savings = max(0.0, avg_cost_all - avg_cost)

            confidence = "high"
            if g["routed"] < 50:
                confidence = "medium"
            if g["routed"] < 20:
                confidence = "low"

            results.append(RoutingQualitySignal(
                team_id=tid_name,
                feature=feat,
                period_days=days,
                total_routed=g["routed"],
                accuracy=round(accuracy, 4),
                avg_cost_per_request=round(avg_cost, 6),
                avg_cost_savings=round(savings, 6),
                misroute_by_category=dict(g["misroutes"]),
                category_distribution=dict(g["categories"]),
                top_errors=g["error_messages"][:5],
                confidence=confidence,
            ))

        return results

    # ── ③ 异常模式检测 ──────────────────────────────────────

    def anomaly_signals(
        self,
        team_id: str,
        days: int = 7,
    ) -> list[AnomalySignal]:
        """Detect and classify anomaly patterns."""
        traces = self._build_trace_index(days)
        generations = self._batch_generations(days)
        signals = []
        now_dt = __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()

        sessions: dict[str, list[dict]] = defaultdict(list)
        for tid, t in traces.items():
            md = t.get("metadata") or {}
            if team_id and md.get("team_id") != team_id:
                continue
            sid = t.get("sessionId")
            if sid:
                sessions[sid].append(t)

        for sid, sts in sessions.items():
            if len(sts) < 3:
                continue
            prompts_seen = set()
            repeats = 0
            for t in sts:
                inp = str(t.get("input") or "")[:200]
                if inp in prompts_seen:
                    repeats += 1
                else:
                    prompts_seen.add(inp)
            if repeats >= 3:
                tid_name = sts[0].get("metadata", {}).get("team_id", team_id or "unknown")
                signals.append(AnomalySignal(
                    anomaly_type="bug_loop",
                    severity="high",
                    team_id=tid_name,
                    description=f"同一 session ({sid}) 检测到 {repeats} 次相似 prompt 重复,可能为应用层死循环",
                    evidence={"session_id": sid, "repeat_count": repeats, "total_requests": len(sts)},
                    suggested_action="限流该 session 并通知开发团队排查",
                    detected_at=now_dt,
                ))

        if team_id:
            efficiency = self.team_efficiency(team_id, days=days)
            for e in efficiency:
                if e.total_requests < 10:
                    continue
                daily_rate = e.total_cost / max(days, 1)
                monthly_projected = daily_rate * 30
                if monthly_projected > 1000:
                    signals.append(AnomalySignal(
                        anomaly_type="budget_risk",
                        severity="medium" if monthly_projected < 5000 else "high",
                        team_id=team_id,
                        feature=e.feature,
                        description=(
                            f"团队 {team_id} 功能 {e.feature} 月预测花费 ${monthly_projected:.0f}, "
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
                        detected_at=now_dt,
                    ))

        return signals
