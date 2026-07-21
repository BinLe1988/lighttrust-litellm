import json
import os
from typing import Optional
from .base import AbstractMonitor
from ..models import ModelPrice, PriceSnapshot

DEEPSEEK_PRICING_URL = "https://api-docs.deepseek.com/quick_start/pricing"

KNOWN_MODELS = {
    "deepseek-v4-flash": {
        "name": "deepseek-v4-flash",
        "input_cost_per_token": 1.4e-07,
        "output_cost_per_token": 2.8e-07,
        "cache_read_input_token_cost": 2.8e-09,
        "max_input_tokens": 1_000_000,
    },
    "deepseek-v4-pro": {
        "name": "deepseek-v4-pro",
        "input_cost_per_token": 4.35e-07,
        "output_cost_per_token": 8.7e-07,
        "cache_read_input_token_cost": 3.625e-09,
        "max_input_tokens": 1_000_000,
    },
}


class DeepSeekMonitor(AbstractMonitor):
    @property
    def name(self) -> str:
        return "deepseek"

    async def fetch(self) -> PriceSnapshot:
        prices = await self._try_fetch_remote()
        if prices is None:
            prices = await self._load_litellm_internal()
        if prices is None:
            prices = self._fallback()
        snap = PriceSnapshot.now(self.name)
        snap.models = prices
        return snap

    async def _try_fetch_remote(self) -> Optional[list[ModelPrice]]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15) as cli:
                resp = await cli.get(DEEPSEEK_PRICING_URL)
                if resp.status_code != 200:
                    return None
                return self._parse_html(resp.text)
        except Exception:
            return None

    def _parse_html(self, html: str) -> Optional[list[ModelPrice]]:
        rows = self._extract_table_rows(html)
        if not rows:
            return None
        result: list[ModelPrice] = []
        for row in rows:
            cols = [c.strip() for c in row if c.strip()]
            if len(cols) < 4:
                continue
            model = self._match_model(cols[0])
            if model is None:
                continue
            try:
                input_price = self._parse_price(cols[1])
                output_price = self._parse_price(cols[2])
                cache_price = None
                if len(cols) >= 4:
                    cache_price = self._parse_price(cols[3])
            except (ValueError, IndexError):
                continue
            result.append(ModelPrice(
                model_name=model["name"],
                provider="deepseek",
                input_cost_per_token=input_price,
                output_cost_per_token=output_price,
                cache_read_input_token_cost=cache_price,
                source=DEEPSEEK_PRICING_URL,
            ))
        return result if result else None

    def _extract_table_rows(self, html: str) -> list[list[str]]:
        import re
        rows: list[list[str]] = []
        in_body = False
        for line in html.split("\n"):
            if "<tbody" in line or "<table" in line:
                in_body = True
            if "</tbody>" in line or "</table>" in line:
                in_body = False
            if not in_body:
                continue
            line = re.sub(r"<[^>]+>", "|", line)
            cells = [c.strip("| ").strip() for c in line.split("|") if c.strip("| ").strip()]
            if len(cells) >= 2:
                rows.append(cells)
        return rows

    def _match_model(self, raw: str) -> Optional[dict]:
        raw_lower = raw.lower().replace(" ", "-")
        for key, info in KNOWN_MODELS.items():
            key_short = key.replace("deepseek-", "")
            if key in raw_lower or key_short in raw_lower:
                return info
        return None

    def _parse_price(self, s: str) -> float:
        s = s.replace("$", "").replace(",", "").strip()
        if "/M" in s:
            return float(s.replace("/M", "")) / 1_000_000
        if "per million" in s.lower():
            return float(s.split()[0]) / 1_000_000
        return float(s)

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
                    name = info["name"]
                    if name in seen:
                        continue
                    seen.add(name)
                    result.append(ModelPrice(
                        model_name=name,
                        provider="deepseek",
                        input_cost_per_token=entry.get("input_cost_per_token", info["input_cost_per_token"]),
                        output_cost_per_token=entry.get("output_cost_per_token", info["output_cost_per_token"]),
                        cache_read_input_token_cost=entry.get("cache_read_input_token_cost") or entry.get("input_cost_per_token_cache_hit"),
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
                notes="fallback: remote fetch failed, using known defaults",
            )
            for name, info in KNOWN_MODELS.items()
        ]
