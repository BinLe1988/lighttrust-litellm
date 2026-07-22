import json
import os
from typing import Optional
from .base import LLMMonitor
from ..models import ModelPrice, PriceSnapshot

DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"

DEEPSEEK_EXTRACTION_PROMPT = """You are a pricing data extraction assistant.
Extract model pricing from the DeepSeek pricing page below.

Return a JSON array of objects:
  - model_name: str
  - input_cost_per_token: float (USD per token)
  - output_cost_per_token: float (USD per token)
  - cache_read_input_token_cost: float | null (USD per token, null if not shown)
  - max_input_tokens: int | null
  - max_output_tokens: int | null

Rules:
  - Convert "$X per 1M tokens" to per-token (divide by 1,000,000).
  - Include ALL models with explicit pricing.
  - Output ONLY the JSON array — no markdown, no explanation.

Pricing page:
```
{raw_text}
```"""

KNOWN_MODELS = {
    "deepseek-v4-flash": {
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "cache_read_input_token_cost": 2.8e-09,
        "max_input_tokens": 1_000_000,
    },
    "deepseek-v4-pro": {
        "input_cost_per_token": 4.35e-07,
        "output_cost_per_token": 8.7e-07,
        "cache_read_input_token_cost": 3.625e-09,
        "max_input_tokens": 1_000_000,
    },
}


class DeepSeekMonitor(LLMMonitor):
    @property
    def name(self) -> str:
        return "deepseek"

    pricing_url = DEEPSEEK_PRICING_URL

    def build_extraction_prompt(self, raw_text: str) -> str:
        return DEEPSEEK_EXTRACTION_PROMPT.format(raw_text=raw_text)

    async def fetch_raw(self) -> Optional[str]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as cli:
                resp = await cli.get(DEEPSEEK_PRICING_URL)
                if resp.status_code == 200:
                    return resp.text
        except Exception:
            pass
        return None

    async def _load_litellm_internal(self) -> Optional[list[ModelPrice]]:
        try:
            litellm_prices_path = os.path.join(
                os.path.dirname(__file__),
                "..", "..",
                "model_prices_and_context_window.json",
            )
            with open(litellm_prices_path) as f:
                db = json.load(f)
        except Exception:
            return None
        seen: set[str] = set()
        result: list[ModelPrice] = []
        for litellm_key, entry in db.items():
            provider = entry.get("litellm_provider", "")
            if provider != "deepseek":
                continue
            for kbase, info in KNOWN_MODELS.items():
                if kbase in litellm_key or kbase.replace("deepseek-", "") in litellm_key:
                    if kbase in seen:
                        continue
                    seen.add(kbase)
                    result.append(ModelPrice(
                        model_name=kbase,
                        provider="deepseek",
                        input_cost_per_token=entry.get("input_cost_per_token", info["input_cost_per_token"]),
                        output_cost_per_token=entry.get("output_cost_per_token", info["output_cost_per_token"]),
                        cache_read_input_token_cost=entry.get("cache_read_input_token_cost")
                            or entry.get("input_cost_per_token_cache_hit"),
                        max_input_tokens=entry.get("max_input_tokens"),
                        source=entry.get("source", ""),
                    ))
                    break
        return result if result else None

    def _fallback(self) -> list[ModelPrice]:
        return [
            ModelPrice(
                model_name=name,
                provider="deepseek",
                input_cost_per_token=info["input_cost_per_token"],
                output_cost_per_token=info["output_cost_per_token"],
                cache_read_input_token_cost=info.get("cache_read_input_token_cost"),
                notes="fallback: LLM + internal both failed, using known defaults",
            )
            for name, info in KNOWN_MODELS.items()
        ]
