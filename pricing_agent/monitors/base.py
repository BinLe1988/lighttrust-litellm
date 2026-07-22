import json
import os
from abc import ABC, abstractmethod
from typing import Optional
from ..models import ModelPrice, PriceSnapshot

EXTRACTION_PROMPT_TEMPLATE = """You are a pricing data extraction assistant.
Extract model pricing information from the text below.

Return a JSON array of objects. Each object must have these fields:
  - model_name: str         — model identifier
  - input_cost_per_token: float — cost per input token (USD)
  - output_cost_per_token: float — cost per output token (USD)
  - cache_read_input_token_cost: float | null — cached input cost (USD)
  - max_input_tokens: int | null
  - max_output_tokens: int | null

Rules:
  - Convert "per 1M tokens" or "/M" prices to per-token (divide by 1,000,000).
  - Do NOT include models without explicit pricing.
  - If a field is absent in the source, use null.
  - Output ONLY the JSON array — no markdown, no explanation.

Source text:
```
{raw_text}
```"""

DEFAULT_LLM_MODEL = "deepseek-v4-flash"


class AbstractMonitor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def fetch(self) -> PriceSnapshot:
        ...

    def default_snapshot(self, reason: str) -> PriceSnapshot:
        snap = PriceSnapshot.now(self.name)
        for m in snap.models:
            m.notes = reason
        return snap


class LLMMonitor(AbstractMonitor):
    """Monitor that uses an LLM to extract structured pricing from raw text."""

    pricing_url: str = ""
    llm_model: str = ""

    def get_llm_model(self) -> str:
        return (
            self.llm_model
            or os.environ.get("PRICING_AGENT_LLM_MODEL")
            or DEFAULT_LLM_MODEL
        )

    @abstractmethod
    def build_extraction_prompt(self, raw_text: str) -> str:
        ...

    async def fetch_raw(self) -> Optional[str]:
        return None

    async def _call_llm(self, prompt: str) -> Optional[list[ModelPrice]]:
        model = self.get_llm_model()
        try:
            import litellm
            resp = await litellm.acompletion(
                model=f"deepseek/{model}",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=4096,
                temperature=0,
            )
            raw = resp.choices[0].message.content
            if not raw:
                return None
            raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(raw)
            if isinstance(data, dict) and "models" in data:
                data = data["models"]
            if isinstance(data, dict):
                data = [data]
            if not isinstance(data, list):
                return None
            return [ModelPrice(
                model_name=item.get("model_name", ""),
                provider=self.name,
                input_cost_per_token=float(item["input_cost_per_token"]),
                output_cost_per_token=float(item["output_cost_per_token"]),
                cache_read_input_token_cost=(
                    float(item["cache_read_input_token_cost"])
                    if item.get("cache_read_input_token_cost") is not None
                    else None
                ),
                max_input_tokens=item.get("max_input_tokens"),
                max_output_tokens=item.get("max_output_tokens"),
                source=self.pricing_url,
            ) for item in data if item.get("model_name") and item.get("input_cost_per_token") is not None]
        except Exception:
            return None

    async def fetch(self) -> PriceSnapshot:
        snap = PriceSnapshot.now(self.name)

        raw = await self.fetch_raw()
        if raw:
            prices = await self._call_llm(self.build_extraction_prompt(raw))
            if prices:
                snap.models = prices
                return snap

        prices = await self._load_litellm_internal()
        if prices:
            snap.models = prices
            snap.models[0].notes = "fallback: LLM extraction failed, used litellm internal"
            return snap

        prices = self._fallback()
        if prices:
            snap.models = prices
        return snap

    async def _load_litellm_internal(self) -> Optional[list[ModelPrice]]:
        return None

    def _fallback(self) -> list[ModelPrice]:
        return []
