"""
Litellm 自定义校验：强制请求携带 metadata。

改用 CustomLogger 的 pre_call_hook，避免 custom_auth 递归问题。
在 proxy_server_config.yaml 中引用:

  litellm_settings:
    callbacks: custom_auth.MetadataValidator
"""

import os
from typing import Optional
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger
from litellm import model_cost, ImageResponse, ModelResponse, EmbeddingResponse


def _required_fields() -> list[str]:
    raw = os.environ.get("REQUIRED_METADATA_FIELDS", "")
    return [f.strip() for f in raw.split(",") if f.strip()]


class MetadataValidator(CustomLogger):
    """
    在每次 LLM 调用前检查请求体是否包含 metadata。
    环境变量 REQUIRED_METADATA_FIELDS 可指定必须的子字段。
    """

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
                        "message": (
                            "metadata is required and must be a JSON object"
                        ),
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
