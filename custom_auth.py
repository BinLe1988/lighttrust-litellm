"""
Litellm 自定义校验：metadata 强制 + 超支熔断。

- MetadataValidator: 强制请求携带 metadata（pre_call_hook）
- BudgetTracker: 按 user_id 跟踪花销，超预算时熔断

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
from typing import Optional
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


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


# ── BudgetTracker（超支熔断）─────────────────────────────────────


class BudgetTracker(CustomLogger):
    """
    按 user_id 跟踪 LLM 调用花销，超过预算时返回 429 熔断。

    环境变量:
      USER_BUDGET_MAP     JSON 字典 {"user_id": max_budget_usd, ...}
      DEFAULT_BUDGET      所有用户的默认预算（USD，可选）
      BUDGET_PERSIST_FILE  花销持久化文件路径（可选，重启后恢复）
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._spend: dict[str, float] = {}

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
        if budget is None:
            return data

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
        return data

    async def async_log_success_event(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
    ):
        response_cost = kwargs.get("response_cost") or 0.0
        if response_cost <= 0:
            return

        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}
        user_id = metadata.get("user_id", "default")

        budget = self._get_budget(user_id)
        if budget is None:
            return

        async with self._lock:
            self._spend[user_id] = self._spend.get(user_id, 0.0) + response_cost
            await self._persist()


# 模块级实例 — litellm 的 get_instance_fn 通过 getattr(module, name) 获取，必须返回实例
validator = MetadataValidator()
tracker = BudgetTracker()
