"""
轻量规则引擎。

使用方式:
  engine = RuleEngine("rules.yaml")
  results = engine.evaluate(ctx)

RuleEngine 负责:
  1. 加载 rules
  2. 匹配 type → evaluator
  3. 条件评估
  4. cooldown 检查
  5. 执行 actions
  6. 返回 RuleResult 列表
"""

import os
import time
import logging
from typing import Callable, Optional

import yaml

from .models import Rule, Condition, RuleAction, RuleContext, RuleResult
from .actions import execute_action

logger = logging.getLogger("rules.engine")

Evaluator = Callable[[Condition, RuleContext], tuple[bool, str]]


def _eval_threshold(cond: Condition, ctx: RuleContext) -> tuple[bool, str]:
    """budget_threshold / rate_limit rules."""
    actual = _resolve_metric(cond.metric, ctx)
    ops = {
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        "==": lambda a, b: a == b,
    }
    fn = ops.get(cond.operator)
    if not fn:
        return False, f"unknown operator {cond.operator}"
    fired = fn(actual, cond.value)
    msg = f"{cond.metric}={actual:.4f} {cond.operator} {cond.value} → {'fired' if fired else 'ok'}"
    return fired, msg


def _eval_prediction(cond: Condition, ctx: RuleContext) -> tuple[bool, str]:
    """budget_prediction rules."""
    if ctx.days_elapsed < cond.min_days_elapsed:
        return False, f"days_elapsed={ctx.days_elapsed} < min_days_elapsed={cond.min_days_elapsed}"
    actual = ctx.predicted_ratio
    fired = actual >= cond.value
    msg = f"predicted_ratio={actual:.4f} >= {cond.value} → {'fired' if fired else 'ok'}"
    return fired, msg


def _eval_anomaly(cond: Condition, ctx: RuleContext) -> tuple[bool, str]:
    """Anomaly detection via multiplier vs baseline."""
    metric = cond.metric
    actual = _resolve_metric(metric, ctx)
    baseline = _resolve_baseline(metric, ctx)

    if ctx.request_count < cond.min_samples:
        return False, f"samples={ctx.request_count} < min_samples={cond.min_samples}"

    if baseline <= 0:
        return False, f"baseline={baseline} <= 0, cannot detect anomaly"

    ratio = actual / baseline
    fired = ratio >= cond.multiplier
    msg = f"{metric}={actual:.4f} vs baseline={baseline:.4f} (ratio={ratio:.2f}x, threshold={cond.multiplier}x) → {'fired' if fired else 'ok'}"
    return fired, msg


def _resolve_metric(metric: str, ctx: RuleContext) -> float:
    m = {
        "spend_ratio": ctx.spend_ratio,
        "predicted_ratio": ctx.predicted_ratio,
        "daily_spend_ratio": ctx.daily_spend_ratio,
        "spend_rate": ctx.spend_rate,
        "request_rate": ctx.request_rate,
        "error_rate": ctx.error_rate,
        "current_spend": ctx.current_spend,
        "daily_spend": ctx.daily_spend,
        "monthly_spend": ctx.monthly_spend,
        "predicted_monthly_spend": ctx.predicted_monthly_spend,
            "avg_latency": ctx.avg_latency,
        "token_rate": ctx.token_rate,
        "total_tokens": float(ctx.total_tokens),
    }
    return m.get(metric, 0.0)


def _resolve_baseline(metric: str, ctx: RuleContext) -> float:
    m = {
        "spend_rate": ctx.spend_rate_baseline,
        "request_rate": ctx.request_rate_baseline,
        "error_rate": ctx.error_rate_baseline,
        "token_rate": ctx.token_rate_baseline,
    }
    return m.get(metric, 0.0)


EVALUATORS: dict[str, Evaluator] = {
    "budget_threshold": _eval_threshold,
    "budget_prediction": _eval_prediction,
    "anomaly": _eval_anomaly,
    "rate_limit": _eval_threshold,
}


def _parse_action(d: dict) -> RuleAction:
    return RuleAction(
        action=d.get("action", "notify"),
        channels=d.get("channels", ["dingtalk"]),
        message=d.get("message", ""),
        throttle_pct=float(d.get("throttle_pct", 0)),
        fallback_model=d.get("fallback_model", ""),
        status_code=int(d.get("status_code", 429)),
    )


def _parse_condition(d: dict) -> Condition:
    return Condition(
        metric=d.get("metric", ""),
        operator=d.get("operator", ">="),
        value=float(d.get("value", 0)),
        multiplier=float(d.get("multiplier", 0)),
        min_samples=int(d.get("min_samples", 0)),
        min_days_elapsed=int(d.get("min_days_elapsed", 0)),
    )


def load_rules(path: str) -> list[Rule]:
    if not os.path.isfile(path):
        logger.warning("Rules file not found: %s", path)
        return []
    with open(path) as f:
        raw = yaml.safe_load(f)
    rules = []
    for d in (raw.get("rules") or []):
        cond_raw = d.get("condition", {})
        actions_raw = d.get("actions", [])
        cond = _parse_condition(cond_raw) if cond_raw else None
        actions = [_parse_action(a) for a in actions_raw]
        rules.append(Rule(
            name=d.get("name", "unnamed"),
            description=d.get("description", ""),
            enabled=d.get("enabled", True),
            type=d.get("type", "budget_threshold"),
            condition=cond,
            actions=actions,
            cooldown=int(d.get("cooldown", 0)),
        ))
    logger.info("Loaded %d rules from %s", len(rules), path)
    return rules


class RuleEngine:
    def __init__(self, rules_path: str = "", notifier=None):
        self._rules: list[Rule] = []
        self._notifier = notifier
        if rules_path:
            self._rules = load_rules(rules_path)

    def reload(self, rules_path: str):
        self._rules = load_rules(rules_path)

    @property
    def rules(self) -> list[Rule]:
        return self._rules

    def evaluate(self, ctx: RuleContext) -> list[RuleResult]:
        results: list[RuleResult] = []
        now = time.time()
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.cooldown > 0 and (now - rule.last_fired) < rule.cooldown:
                continue
            evaluator = EVALUATORS.get(rule.type)
            if not evaluator or not rule.condition:
                continue
            fired, msg = evaluator(rule.condition, ctx)
            result = RuleResult(
                rule_name=rule.name,
                rule_type=rule.type,
                triggered=fired,
                message=msg,
                cooldown=float(rule.cooldown),
            )
            if fired:
                rule.last_fired = now
                for a in rule.actions:
                    execute_action(a, result, ctx, self._notifier)
            results.append(result)
        return results
