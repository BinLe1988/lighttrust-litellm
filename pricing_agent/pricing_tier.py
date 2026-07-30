"""
价值定价引擎 — 按业务结果/ROI 定价。

三层模型:
  基础层 (standard)  — 按 token 计费（现有）
  价值层 (premium)   — 按质量分级，1.5x 溢价
  结果层 (enterprise) — ROI 分成，2.0x 溢价 + SLA 保障

定价公式:
  effective_cost = base_cost × tier_multiplier × sla_discount
    sla_discount = 质量达标率 ≥ min_quality_score ? 1.0 : (达标率 / min_quality_score)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("pricing_agent.pricing_tier")

DEFAULT_TIER = "standard"
VALID_TIERS = ["standard", "premium", "enterprise"]


@dataclass
class TierConfig:
    name: str
    multiplier: float
    min_quality_score: float
    max_latency_ms: int
    guaranteed_uptime: str
    support: str
    sla_compensation_rate: float  # 未达标时打折比例 (0.8 = 打8折)
    description: str = ""


TIER_CONFIGS: dict[str, TierConfig] = {
    "standard": TierConfig(
        name="standard",
        multiplier=1.0,
        min_quality_score=0.0,
        max_latency_ms=10_000,
        guaranteed_uptime="99.0%",
        support="community",
        sla_compensation_rate=1.0,
        description="按量计费，无质量保障",
    ),
    "premium": TierConfig(
        name="premium",
        multiplier=1.5,
        min_quality_score=0.7,
        max_latency_ms=2_000,
        guaranteed_uptime="99.5%",
        support="priority",
        sla_compensation_rate=0.8,
        description="质量保障级，SLA ≥7/10，低延迟优先路由",
    ),
    "enterprise": TierConfig(
        name="enterprise",
        multiplier=2.0,
        min_quality_score=0.85,
        max_latency_ms=1_000,
        guaranteed_uptime="99.9%",
        support="dedicated",
        sla_compensation_rate=0.7,
        description="企业级，ROI 对赌，99.9% 可用性保障",
    ),
}


def get_tier(tier_name: str) -> TierConfig:
    tier_name = tier_name.lower()
    if tier_name in TIER_CONFIGS:
        return TIER_CONFIGS[tier_name]
    logger.warning("Unknown tier '%s', falling back to standard", tier_name)
    return TIER_CONFIGS["standard"]


# ── 计费计算 ────────────────────────────────────────────────────


@dataclass
class PricedRequest:
    base_cost: float
    tier: str
    tier_multiplier: float
    quality_score: Optional[float]
    latency_ms: float
    sla_met: bool
    sla_discount: float
    effective_cost: float


def calculate_request_cost(
    base_cost: float,
    tier_name: str,
    quality_score: Optional[float] = None,
    latency_ms: float = 0.0,
) -> PricedRequest:
    """单次请求的定价计算。

    Args:
        base_cost: litellm response_cost（已校准）
        tier_name: 客户 tier
        quality_score: Langfuse score (0-1), None 表示无评分
        latency_ms: 请求延迟

    Returns:
        PricedRequest 含各定价因子
    """
    config = get_tier(tier_name)

    # SLA 检查
    sla_met = True
    sla_discount = 1.0

    if quality_score is not None and config.min_quality_score > 0:
        if quality_score < config.min_quality_score:
            sla_met = False
            sla_discount = max(
                quality_score / max(config.min_quality_score, 0.01),
                config.sla_compensation_rate,
            )

    if config.max_latency_ms > 0 and latency_ms > config.max_latency_ms:
        sla_met = False
        latency_ratio = max_latency_budget = config.max_latency_ms
        lat_discount = max_latency_budget / max(latency_ms, 1)
        sla_discount = min(sla_discount, max(lat_discount, config.sla_compensation_rate))

    effective_cost = base_cost * config.multiplier * sla_discount

    return PricedRequest(
        base_cost=base_cost,
        tier=tier_name,
        tier_multiplier=config.multiplier,
        quality_score=quality_score,
        latency_ms=latency_ms,
        sla_met=sla_met,
        sla_discount=round(sla_discount, 4),
        effective_cost=round(effective_cost, 6),
    )


# ── 价值报告 ──────────────────────────────────────────────────────


@dataclass
class ValueReport:
    team_id: str
    tier: str
    period_start: str
    period_end: str
    total_requests: int
    total_base_cost: float
    total_effective_cost: float
    tier_premium: float
    sla_discount_total: float
    sla_compliance_rate: float
    avg_latency_ms: float
    estimated_savings_vs_baseline: float  # 相对不路由场景的节省
    breakdown_by_feature: dict = field(default_factory=dict)


def generate_report(
    team_id: str,
    tier_name: str,
    days: int = 30,
) -> Optional[ValueReport]:
    """生成客户价值对账单。

    需要 signals.billing_signals() 提供聚合数据。
    此函数仅为数据模型示例，实际数据由 signals.py 填充。
    """
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).isoformat()
    return ValueReport(
        team_id=team_id,
        tier=tier_name,
        period_start=since,
        period_end=now.isoformat(),
        total_requests=0,
        total_base_cost=0.0,
        total_effective_cost=0.0,
        tier_premium=0.0,
        sla_discount_total=0.0,
        sla_compliance_rate=0.0,
        avg_latency_ms=0.0,
        estimated_savings_vs_baseline=0.0,
    )


# ── 报表格式化 ──────────────────────────────────────────────────


def format_report(report: ValueReport) -> str:
    """格式化为客户可见的价值对账单文本。"""
    premium_amount = report.tier_premium
    discount_amount = report.sla_discount_total
    savings = report.estimated_savings_vs_baseline
    net = report.total_effective_cost

    lines = [
        f"📊 AI 服务价值对账单",
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"  客户: {report.team_id}    计划: {report.tier}",
        f"  周期: {report.period_start[:10]} → {report.period_end[:10]}",
        f"",
        f"📈 用量统计",
        f"  请求次数:  {report.total_requests:,}",
        f"  基础成本:  ${report.total_base_cost:.2f}",
        f"",
        f"🎯 质量达标率",
        f"  SLA (≥{TIER_CONFIGS[report.tier].min_quality_score*100:.0f}分):  {report.sla_compliance_rate*100:.1f}%",
        f"  平均延迟:  {report.avg_latency_ms:.0f}ms",
    ]

    if report.tier != "standard":
        lines.extend([
            f"",
            f"💰 定价明细",
            f"  基础费用:   ${report.total_base_cost:.2f}",
            f"  Tier 溢价:  +${premium_amount:.2f} ({TIER_CONFIGS[report.tier].multiplier}x)",
        ])
        if discount_amount > 0:
            lines.append(f"  SLA 折扣:   -${discount_amount:.2f}")
        lines.append(f"  ─────────────────");
        lines.append(f"  应收:       ${net:.2f}")

    lines.extend([
        f"",
        f"📉 成本优化",
        f"  不路由场景预估:  ${report.total_base_cost + savings:.2f}",
    ])
    if savings > 0:
        lines.append(f"  实际支付:        ${net:.2f}")
        lines.append(f"  节省:            ${savings:.2f} ({savings/(report.total_base_cost + savings)*100:.0f}%)")
    else:
        lines.append(f"  实际支付:        ${net:.2f}")

    lines.append(f"━  ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━ ━")
    return "\n".join(lines)
