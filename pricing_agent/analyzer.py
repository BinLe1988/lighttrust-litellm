"""
LLM-powered analysis of pricing changes.

Takes the raw PriceChange list and uses an LLM to:
1. Understand the intent behind each change (price cut, promotion, EOL, …)
2. Assess projected budget impact
3. Recommend a course of action (apply / warn / skip)
"""

import json
import os
from typing import Optional
from .models import PriceChange

ANALYSIS_PROMPT_TEMPLATE = """You are a pricing change analyst for an LLM API gateway.

Below is a list of detected pricing changes. For each change, provide:
1.  intent — what the vendor likely intends (price_cut, price_increase,
    promotion, caching_incentive, model_eol, new_model, noise).
2.  impact — how this affects a typical heavy user's monthly spend.
3.  recommendation — one of: apply, warn, skip.
4.  reasoning — brief justification.

Changes:
{changes_json}

Return a JSON array of objects:
[
  {{
    "index": 0,
    "intent": "price_cut",
    "impact": "~40% reduction in monthly cost for heavy users",
    "recommendation": "apply",
    "reasoning": "Clear price reduction from vendor. No downside to updating."
  }}
]

Output ONLY the JSON array — no markdown, no explanation.
"""

DEFAULT_ANALYSIS_MODEL = "deepseek-v4-flash"


def _changes_brief(changes: list[PriceChange]) -> str:
    lines = []
    for i, c in enumerate(changes):
        lines.append(
            f"  [{i}] type={c.change_type} model={c.model_name} field={c.field or '-'} "
            f"old={c.old_value} new={c.new_value}"
        )
        lines.append(f"       desc: {c.description}")
        if c.impact:
            lines.append(f"       current_impact: {c.impact[:120]}")
    return "\n".join(lines)


async def analyze_changes(
    changes: list[PriceChange],
    model: Optional[str] = None,
) -> list[dict]:
    if not changes:
        return []

    resolved_model = model or os.environ.get("PRICING_AGENT_LLM_MODEL") or DEFAULT_ANALYSIS_MODEL
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(changes_json=_changes_brief(changes))

    try:
        import litellm
        resp = await litellm.acompletion(
            model=f"deepseek/{resolved_model}",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0,
        )
        raw = resp.choices[0].message.content
        if not raw:
            return []
        raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(raw)
        if isinstance(data, dict):
            if "analyses" in data:
                data = data["analyses"]
            elif "changes" in data:
                data = data["changes"]
            else:
                data = [data]
        return data if isinstance(data, list) else []
    except Exception:
        return []
