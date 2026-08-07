"""
缓存预热模块 — 保证模型切换后仍能享受 cache 命中价。

背景:
  DeepSeek/Anthropic 等 provider 的 prompt cache 按 (model, 精确前缀) 建条目。
  策略闭环把团队默认模型从旗舰切到轻量后, 目标模型的 cache 是冷的,
  首批请求按全价计费, 直到缓存重新建立——这会让切换瞬间失去 cache 折扣。

本模块在两处保证"切换后仍享受 cache 价":
  1. 计价侧: 模型需配置 cache_read_input_token_cost (见 proxy_server_config.yaml),
     litellm 才会对 cached tokens 按折扣计价, 否则 fallback 到全价输入价。
  2. 命中侧: 切换前用该 team/feature 的高频 prompt 前缀重放目标模型
     (max_tokens=1), 预建 cache 条目, 让真实流量一切过去就能命中。

用法:
  from pricing_agent.strategy.cache_warm import warm_before_switch
  report = warm_before_switch(target_model="deepseek-v4-flash", ...)
"""

import logging
import os
import time
from dataclasses import dataclass, field

logger = logging.getLogger("strategy.cache_warm")

# 前缀归一化: 只保留前缀前 N 字符作为 cache 键近似
PREFIX_CHARS = 200
MIN_OCCURRENCES = 3
TOP_K = 10
WARMUP_MAX_TOKENS = 1
WARMUP_TIMEOUT_S = 30


@dataclass
class CacheWarmReport:
    target_model: str
    warmup_mode: str          # live / simulated
    prefixes_warmed: list[dict] = field(default_factory=list)
    input_tokens_used: int = 0
    successes: int = 0
    failures: int = 0
    elapsed_s: float = 0.0
    error: str = ""
    ok: bool = False

    def summary(self) -> str:
        if self.error:
            return f"预热失败: {self.error}"
        return (
            f"目标 {self.target_model} 预热 {self.successes}/{len(self.prefixes_warmed)} 条前缀, "
            f"输入 {self.input_tokens_used} tokens, 耗时 {self.elapsed_s:.1f}s"
        )


def fetch_dominant_prefixes(
    lf=None,
    team_id: str = "",
    feature: str = "",
    days: int = 30,
    top_k: int = TOP_K,
    min_occurrences: int = MIN_OCCURRENCES,
    prefix_chars: int = PREFIX_CHARS,
) -> list[dict]:
    """从 Langfuse observations 提取该 team/feature 的高频 prompt 前缀。

    返回 [{"prefix": str, "count": int}], 按出现次数降序。
    lf 为 None 或数据不足时返回空列表, 由调用方决定降级为模拟。
    """
    if lf is None:
        return []
    try:
        from collections import Counter

        from pricing_agent.strategy.signals import SignalAggregator

        agg = SignalAggregator(lf)
        generations = agg._batch_generations(days)
        counter: Counter = Counter()
        for tid, gens in generations.items():
            for gen in gens:
                md = (gen.get("metadata") or {}) or {}
                req_md = (md.get("requester_metadata") or {}) or {}
                if team_id and req_md.get("team_id") != team_id:
                    continue
                if feature and req_md.get("feature") != feature:
                    continue
                inp = gen.get("input")
                if isinstance(inp, str) and inp.strip():
                    counter[inp[:prefix_chars]] += 1
        return [
            {"prefix": p, "count": c}
            for p, c in counter.most_common(top_k)
            if c >= min_occurrences
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_dominant_prefixes 失败, 返回空: %s", exc)
        return []


def warm_model_cache(
    target_model: str,
    prefixes: list[dict],
    api_key: str | None = None,
    max_tokens: int = WARMUP_MAX_TOKENS,
    live: bool = True,
) -> CacheWarmReport:
    """用高频前缀重放目标模型以预建 provider cache。

    live=False 时只返回模拟报告 (不真实调用上游), 用于演示/离线。
    """
    start = time.time()
    report = CacheWarmReport(target_model=target_model, warmup_mode="live" if live else "simulated")

    if not prefixes:
        report.error = "无可预热前缀"
        report.elapsed_s = time.time() - start
        return report

    report.prefixes_warmed = prefixes

    if not live:
        report.input_tokens_used = sum(len(p["prefix"]) // 4 for p in prefixes)
        report.successes = len(prefixes)
        report.ok = True
        report.elapsed_s = time.time() - start
        return report

    import litellm  # 延迟导入, 避免模块加载开销

    for p in prefixes:
        try:
            resp = litellm.completion(
                model=target_model,
                api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
                messages=[{"role": "user", "content": p["prefix"]}],
                max_tokens=max_tokens,
                timeout=WARMUP_TIMEOUT_S,
                cache={"no-cache": True},  # 不污染 proxy 侧语义缓存, 只建 provider 侧 prompt cache
            )
            usage = getattr(resp, "usage", None)
            tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            report.input_tokens_used += tokens
            report.successes += 1
        except Exception as exc:  # noqa: BLE001
            report.failures += 1
            logger.warning("预热前缀失败 (%s): %s", p["prefix"][:40], exc)

    report.ok = report.successes > 0
    report.elapsed_s = time.time() - start
    if report.failures and not report.successes:
        report.error = f"{report.failures} 条前缀全部预热失败 (目标模型可能不可用)"
    return report


def warm_before_switch(
    target_model: str,
    lf=None,
    team_id: str = "",
    feature: str = "",
    days: int = 30,
    prefixes: list[dict] | None = None,
    api_key: str | None = None,
    live: bool = True,
) -> CacheWarmReport:
    """切换前的标准预热入口。

    prefixes 未提供时, 先从 Langfuse 提取高频前缀;
    提取不到且 live=True 时降级为 simulated 报告(保证流程不被阻断)。
    """
    if not prefixes:
        prefixes = fetch_dominant_prefixes(
            lf=lf, team_id=team_id, feature=feature, days=days,
        )
    if not prefixes and live:
        logger.info("未获取到真实高频前缀, 降级为 simulated 预热")
        live = False
    return warm_model_cache(
        target_model=target_model,
        prefixes=prefixes,
        api_key=api_key,
        live=live,
    )
