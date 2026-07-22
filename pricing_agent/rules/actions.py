"""Rule actions: notify, degrade, reject, log."""

import logging
from typing import Optional
from .models import RuleAction, RuleResult, RuleContext

logger = logging.getLogger("rules.actions")


def execute_action(
    action: RuleAction,
    result: RuleResult,
    ctx: RuleContext,
    notifier=None,
) -> None:
    mapping = {
        "notify": _action_notify,
        "degrade": _action_degrade,
        "reject": _action_reject,
        "log": _action_log,
    }
    fn = mapping.get(action.action)
    if fn:
        fn(action, result, ctx, notifier)


def _action_notify(
    action: RuleAction, result: RuleResult, ctx: RuleContext, notifier=None
):
    message = _render(action.message, ctx) or (
        f"[Rules] {result.rule_name} triggered for {ctx.user_id}"
    )
    if notifier and hasattr(notifier, "send"):
        for ch in action.channels:
            notifier.send(message, channel=ch)
    logger.warning("NOTIFY %s: %s", action.channels, message)
    result.actions_taken.append(f"notify:{','.join(action.channels)}")


def _action_degrade(
    action: RuleAction, result: RuleResult, ctx: RuleContext, notifier=None
):
    logger.warning(
        "DEGRADE %s: throttle=%s fallback=%s",
        ctx.user_id,
        action.throttle_pct,
        action.fallback_model,
    )
    if action.throttle_pct > 0:
        result.degrade_throttle_pct = action.throttle_pct
    if action.fallback_model:
        result.degrade_fallback_model = action.fallback_model
    result.actions_taken.append("degrade")


def _action_reject(
    action: RuleAction, result: RuleResult, ctx: RuleContext, notifier=None
):
    code = action.status_code or 429
    logger.warning("REJECT %s: %s", ctx.user_id, action.message)
    result.reject = True
    result.reject_status = code
    result.message = _render(action.message, ctx) or (
        f"Request rejected by rule {result.rule_name}"
    )
    result.actions_taken.append(f"reject:{code}")


def _action_log(
    action: RuleAction, result: RuleResult, ctx: RuleContext, notifier=None
):
    level = (action.message or "").lower()
    msg = f"[Rules] {result.rule_name} | user={ctx.user_id} model={ctx.model}"
    if level in ("warn", "warning"):
        logger.warning(msg)
    elif level in ("error", "err"):
        logger.error(msg)
    else:
        logger.info(msg)
    result.actions_taken.append("log")


def _render(template: str, ctx: RuleContext) -> str:
    try:
        return template.format(
            user_id=ctx.user_id,
            team_id=ctx.team_id,
            model=ctx.model,
            current_spend=ctx.current_spend,
            max_budget=ctx.max_budget or ctx.default_budget,
            spend_ratio=ctx.spend_ratio,
            predicted_ratio=ctx.predicted_ratio,
            predicted_monthly_spend=ctx.predicted_monthly_spend,
            spend_rate=ctx.spend_rate,
            error_rate=ctx.error_rate,
            request_rate=ctx.request_rate,
        )
    except Exception:
        return template
