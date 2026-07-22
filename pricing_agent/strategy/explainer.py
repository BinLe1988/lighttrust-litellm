"""
LLM 驱动的提案解释生成器。

职责:
  将 ChangeProposal + 支撑信号 + 安全护栏结果
  组织成一段业务负责人能看懂的自然语言说明。

设计原则:
  - LLM 只负责"翻译"结构化数据为自然语言,不参与决策逻辑
  - 确定性数据(信号值、阈值、风险等级)由上游模块提供
  - LLM 调用失败时有 fallback——基于模板生成概要
"""

import json
import logging
import os
from typing import Optional

from .models import ChangeProposal, GuardrailResult

logger = logging.getLogger("strategy.explainer")


def _call_llm(prompt: str, model: str = "deepseek/deepseek-v4-flash") -> str:
    """Call litellm to generate explanation text.

    Falls back to a template-based summary on any failure.
    """
    try:
        # Suppress noisy logging from litellm
        logging.getLogger("LiteLLM").setLevel(logging.ERROR)
        logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
        logging.getLogger("openai").setLevel(logging.ERROR)
        import litellm
        resp = litellm.completion(
            model=model,
            api_key=os.environ.get("DEEPSEEK_API_KEY") or None,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个企业 AI 成本治理系统的解释器。"
                        "你将收到一份结构化的配置变更提案数据(JSON),"
                        "请用中文生成一段业务负责人(非技术人员)能看懂的自然语言说明。\n\n"
                        "要求:\n"
                        "- 语言简洁、清晰、不带技术黑话\n"
                        "- 说清楚: 发现了什么问题 → 为什么重要 → 建议怎么做 → 预期效果\n"
                        "- 如果提案需要人工审批,在末尾明确说明需要谁审批\n"
                        "- 控制在 200 字以内\n"
                        "- 不要使用 markdown 格式,纯文本即可"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM explanation failed: %s", exc)
        return _fallback_explanation(prompt)


def _fallback_explanation(data_json: str) -> str:
    """Template-based fallback when LLM is unavailable."""
    try:
        data = json.loads(data_json) if isinstance(data_json, str) else data_json
    except Exception:
        return "（无法生成解释）"
    ptype = data.get("proposal_type", "未知")
    target = data.get("target", {})
    team = target.get("team_id", "?")
    feature = target.get("feature", "?")
    current = data.get("current_state", "")
    suggested = data.get("suggested_state", "")
    savings = data.get("expected_savings", 0)
    risk = data.get("risk_level", "低")
    approval = data.get("requires_approval", "无")

    type_names = {
        "route_change": "路由调整",
        "quota_adjustment": "配额调整",
        "model_fallback": "模型降级",
        "budget_alert": "预算预警",
        "manual_review_required": "人工审查",
    }
    cn_type = type_names.get(ptype, ptype)
    return (
        f"[{cn_type}] 团队 {team} 的功能「{feature}」当前"
        f"「{current}」建议变更为「{suggested}」。"
        f"预期月度节省 ${savings:.2f}。"
        f"风险等级: {risk}。"
        f"审批要求: {approval}。"
    )


def explain_proposal(
    proposal: ChangeProposal,
    guardrail: Optional[GuardrailResult] = None,
) -> str:
    """Generate a human-readable explanation for a proposal."""
    guardrail_info = {}
    if guardrail:
        guardrail_info = {
            "guardrail_passed": guardrail.passed,
            "guardrail_failures": guardrail.failures,
            "approval_required": guardrail.approval_required,
            "approval_hint": guardrail.approval_hint,
        }

    signals_summary = []
    for s in (proposal.supporting_signals or []):
        if hasattr(s, "__dataclass_fields__"):
            signals_summary.append({
                k: v for k, v in s.__dict__.items()
                if not k.startswith("_") and v is not None
            })

    prompt_data = json.dumps({
        "proposal_id": proposal.proposal_id,
        "proposal_type": proposal.proposal_type,
        "target": proposal.target,
        "current_state": proposal.current_state,
        "suggested_state": proposal.suggested_state,
        "risk_level": proposal.risk_level,
        "expected_savings": proposal.expected_savings,
        "expected_savings_currency": proposal.expected_savings_currency,
        "auto_executable": proposal.auto_executable,
        "requires_approval": proposal.requires_approval,
        "signals": signals_summary,
        "guardrail": guardrail_info,
    }, ensure_ascii=False, default=str)

    return _call_llm(prompt_data)
