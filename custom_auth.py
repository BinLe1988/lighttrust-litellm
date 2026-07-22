"""
Litellm 自定义校验：metadata 强制 + 超支熔断 + 规则引擎。

- MetadataValidator: 强制请求携带 metadata（pre_call_hook）
- BudgetTracker:   按 user_id 跟踪花销，超预算时熔断
- RulesEngine:     轻量规则引擎，支持预算预测 / 异常检测 / 限流降级

在 proxy_server_config.yaml 中引用:

  litellm_settings:
    callbacks:
      - custom_auth.validator
      - custom_auth.tracker
"""

# 模块级实例（litellm 的 get_instance_fn 返回模块属性，故需要实例而非类）
# 见文件末尾赋值

import asyncio
import json
import os
import random
from typing import Optional
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

# ── 可选的规则引擎 ────────────────────────────────────────────────
_RULES_ENGINE = None
_WINDOWS = None
_FIRST_SEEN: dict[str, float] = {}

try:
    from pricing_agent.rules.engine import RuleEngine
    from pricing_agent.rules.windows import PerUserWindows
    from pricing_agent.rules.models import RuleContext

    _rules_path = os.environ.get("RULES_PATH", "")
    if not _rules_path:
        _rules_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "pricing_agent",
            "rules.yaml",
        )
    _RULES_ENGINE = RuleEngine(_rules_path, notifier=None)
    _WINDOWS = PerUserWindows()
except ImportError:
    pass


# ── MetadataValidator ──────────────────────────────────────────────


def _required_fields() -> list[str]:
    raw = os.environ.get("REQUIRED_METADATA_FIELDS", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


class MetadataValidator(CustomLogger):
    """每次 LLM 调用前检查请求体是否包含 metadata。"""

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ):
        metadata = data.get("metadata")

        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": "metadata is required and must be a JSON object",
                        "type": "bad_request",
                        "param": "metadata",
                        "code": "400",
                    }
                },
            )

        for field in _required_fields():
            if field not in metadata or metadata[field] is None:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "message": f"metadata.{field} is required",
                            "type": "bad_request",
                            "param": f"metadata.{field}",
                            "code": "400",
                        }
                    },
                )

        return data


# ── 降级状态 ────────────────────────────────────────────────────────

class DegradeState:
    """多维度降级指示器，规则引擎在 async_log_success_event 中写入，
    async_pre_call_hook 从中读取并决策是否拒绝/降级本请求。"""

    def __init__(self):
        self._lock = asyncio.Lock()
        # _data[uid] = {throttle_pct, fallback_model, message, set_at, ttl}
        self._data: dict[str, dict] = {}

    async def set(self, uid: str, ttl: float = 300.0, **attrs):
        async with self._lock:
            entry = self._data.setdefault(uid, {})
            entry.update(attrs, set_at=__import__("time").time(), ttl=ttl)

    async def should_degrade(self, uid: str) -> Optional[dict]:
        async with self._lock:
            entry = self._data.get(uid)
            if not entry:
                return None
            if __import__("time").time() - entry.get("set_at", 0) > entry.get("ttl", 300):
                self._data.pop(uid, None)
                return None
            # pass a copy so caller can't mutate
            return dict(entry)

    async def clear(self, uid: str):
        async with self._lock:
            self._data.pop(uid, None)


# ── BudgetTracker（超支熔断 + 规则引擎）───────────────────────────


class BudgetTracker(CustomLogger):
    """
    按 user_id 跟踪 LLM 调用花销，超过预算时返回 429 熔断。

    集成规则引擎后额外支持:
      - 实时规则评估（pre_call）
      - 滑动窗口统计 & 规则评估（post_call）
      - 降级状态管理

    环境变量:
      USER_BUDGET_MAP      JSON {"user_id": max_budget_usd, ...}
      DEFAULT_BUDGET       默认预算（USD）
      BUDGET_PERSIST_FILE  花销持久化文件路径
      RULES_PATH           规则 YAML 路径（可选，默认 pricing_agent/rules.yaml）
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._spend: dict[str, float] = {}
        self._degrade = DegradeState()

        raw = os.environ.get("USER_BUDGET_MAP", "{}")
        self._budgets: dict[str, float] = {
            k: float(v) for k, v in json.loads(raw).items()
        }

        self._default_budget: Optional[float] = None
        default_raw = os.environ.get("DEFAULT_BUDGET", "")
        if default_raw:
            self._default_budget = float(default_raw)

        self._persist_file = os.environ.get("BUDGET_PERSIST_FILE", "")
        if self._persist_file:
            try:
                with open(self._persist_file) as f:
                    self._spend = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                self._spend = {}

    def _get_budget(self, user_id: str) -> Optional[float]:
        if user_id in self._budgets:
            return self._budgets[user_id]
        return self._default_budget

    async def _persist(self):
        if not self._persist_file:
            return
        with open(self._persist_file, "w") as f:
            json.dump(self._spend, f)

    # ── pre_call: 熔断检查 + 规则引擎 pre 评估 ──────────────────

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ):
        metadata = data.get("metadata") or {}
        user_id = metadata.get("user_id", "default")
        budget = self._get_budget(user_id)

        # 1. 基础熔断
        if budget is not None:
            async with self._lock:
                current = self._spend.get(user_id, 0.0)
            if current >= budget:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": (
                                f"Budget exceeded for user '{user_id}': "
                                f"spent ${current:.4f}, limit ${budget:.4f}"
                            ),
                            "type": "budget_exceeded",
                            "code": "429",
                        }
                    },
                )

        # 2. 规则引擎 pre 评估 — degrade 状态检查
        degrade = await self._degrade.should_degrade(user_id)
        if degrade:
            tp = degrade.get("throttle_pct", 0.0)
            fb = degrade.get("fallback_model", "")
            if tp > 0 and random.random() < tp:
                msg = degrade.get("message", "")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": msg or f"Rate limited (throttle {tp:.0%})",
                            "type": "rate_limited",
                            "code": "429",
                        }
                    },
                )
            if fb and data.get("model", "").lower() != fb.lower():
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": {
                            "message": (
                                f"Please retry with fallback model '{fb}'"
                            ),
                            "type": "fallback_required",
                            "code": "429",
                        }
                    },
                )

        return data

    # ── post_call: 记录花销 + 更新滑动窗口 + 规则引擎 post 评估 ──

    async def async_log_success_event(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
    ):
        response_cost = kwargs.get("response_cost") or 0.0
        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}
        user_id = metadata.get("user_id", "default")

        # 更新花销
        budget = self._get_budget(user_id)
        if budget is not None:
            async with self._lock:
                self._spend[user_id] = self._spend.get(user_id, 0.0) + response_cost
                await self._persist()

        # 规则引擎 post 评估（滑动窗口 + 预算预测 / 异常检测）
        if _RULES_ENGINE and _WINDOWS:
            model = litellm_params.get("model", "")
            provider = litellm_params.get("custom_llm_provider", "")
            is_error = response_cost <= 0

            _WINDOWS.add_spend(user_id, response_cost)
            _WINDOWS.add_request(user_id, is_error)

            now = __import__("time").time()
            if user_id not in _FIRST_SEEN:
                _FIRST_SEEN[user_id] = now
            days_elapsed = max(1, int((now - _FIRST_SEEN[user_id]) / 86400))
            days_in_month = 30

            async with self._lock:
                current_spend = self._spend.get(user_id, 0.0)

            ctx = _WINDOWS.build_context(
                uid=user_id,
                current_spend=current_spend,
                max_budget=budget or self._default_budget or 0,
                default_budget=self._default_budget or 0,
                days_elapsed=days_elapsed,
                days_in_month=days_in_month,
            )
            ctx.model = model
            ctx.provider = provider

            results = _RULES_ENGINE.evaluate(ctx)
            for r in results:
                if r.triggered:
                    if r.degrade_throttle_pct > 0 or r.degrade_fallback_model:
                        ttl = max(r.cooldown, 300) if r.cooldown > 0 else 3600
                        await self._degrade.set(
                            user_id,
                            ttl=ttl,
                            throttle_pct=r.degrade_throttle_pct,
                            fallback_model=r.degrade_fallback_model,
                            message=r.message,
                        )


# 模块级实例 — litellm 的 get_instance_fn 通过 getattr(module, name) 获取，必须返回实例
validator = MetadataValidator()
tracker = BudgetTracker()
