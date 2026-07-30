"""
Langfuse 归因查询客户端。

职责:
  1. 从 Langfuse 拉取 traces / observations / scores
  2. 按 team_id / feature 维度聚合
  3. 返回结构化数据供 signals 模块分析

设计原则:
  - 只读: 不写入 Langfuse,不对观测系统产生副作用
  - 查询范围可配: 默认最近 7 天,可通过 period_days 参数覆盖
  - 兼容 Langfuse API v2 (Langfuse 所在容器: http://localhost:3000)
"""

import os
import json
import time as _time
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode

import httpx

logger = logging.getLogger("strategy.langfuse")


LANGFUSE_BASE = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3001")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-local")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-local")


@dataclass
class LangfuseConfig:
    base_url: str = LANGFUSE_BASE
    public_key: str = LANGFUSE_PUBLIC_KEY
    secret_key: str = LANGFUSE_SECRET_KEY


def _basic_auth(cfg: LangfuseConfig) -> str:
    import base64
    raw = f"{cfg.public_key}:{cfg.secret_key}"
    return base64.b64encode(raw.encode()).decode()


class LangfuseClient:
    """Lightweight Langfuse API v2 client — read-only."""

    def __init__(self, cfg: Optional[LangfuseConfig] = None):
        self._cfg = cfg or LangfuseConfig()
        self._auth = _basic_auth(self._cfg)
        self._http = httpx.Client(
            base_url=self._cfg.base_url,
            headers={
                "Authorization": f"Basic {self._auth}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        self._last_req_time = 0.0

    def _rate_limit(self):
        """Ensure at least RATE_LIMIT_INTERVAL seconds between requests."""
        interval = 4.0  # well within 15 req / 39s
        elapsed = _time.time() - self._last_req_time
        if elapsed < interval:
            _time.sleep(interval - elapsed)
        self._last_req_time = _time.time()

    # ── generic GET helper ──────────────────────────────────────

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        self._rate_limit()
        url = f"/api/public{path}"
        if params:
            clean = {}
            for k, v in params.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    if v:
                        clean[k] = v
                else:
                    clean[k] = v
            if clean:
                qs = urlencode(clean, doseq=True)
                url = f"{url}?{qs}"
        for attempt in range(10):
            resp = self._http.get(url)
            if resp.status_code == 429:
                try:
                    body = resp.json()
                    wait = body.get("details", {}).get("retryAfterSeconds", 10)
                except Exception:
                    wait = 10
                logger.warning("Rate limited, retrying in %ds (attempt %d/10)...", wait, attempt + 1)
                _time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise httpx.HTTPStatusError(
            "Rate limited after 10 retries",
            request=resp.request,
            response=resp,
        )

    # ── traces ──────────────────────────────────────────────────

    def list_traces(
        self,
        page: int = 1,
        limit: int = 100,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        tags: Optional[list[str]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        params = dict(
            page=page,
            limit=limit,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
            tags=tags,
            userId=user_id,
            sessionId=session_id,
        )
        return self._get("/traces", params)

    # ── observations ────────────────────────────────────────────

    def list_observations(
        self,
        page: int = 1,
        limit: int = 50,
        trace_id: Optional[str] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        observation_type: Optional[str] = None,
    ) -> dict:
        params = dict(
            page=page,
            limit=limit,
            traceId=trace_id,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
            type=observation_type,
        )
        return self._get("/observations", params)

    # ── scores ──────────────────────────────────────────────────

    def list_scores(
        self,
        page: int = 1,
        limit: int = 100,
        trace_id: Optional[str] = None,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
    ) -> dict:
        params = dict(
            page=page,
            limit=limit,
            traceId=trace_id,
            userId=user_id,
            name=name,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
        )
        return self._get("/scores", params)

    # ── daily metrics (aggregated) ──────────────────────────────

    def daily_metrics(
        self,
        page: int = 1,
        limit: int = 100,
        from_timestamp: Optional[str] = None,
        to_timestamp: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> dict:
        """Langfuse daily metrics endpoint (availability depends on version)."""
        params = dict(
            page=page,
            limit=limit,
            fromTimestamp=from_timestamp,
            toTimestamp=to_timestamp,
            tags=tags,
        )
        return self._get("/metrics/daily", params)

    # ── high-level helpers ──────────────────────────────────────

    @staticmethod
    def _fmt_ts(dt: datetime) -> str:
        """格式化为 Langfuse ISO datetime（带时区偏移）。"""
        offset = dt.astimezone().strftime("%z")
        offset_formatted = f"{offset[:3]}:{offset[3:]}" if offset else "+00:00"
        return dt.strftime("%Y-%m-%dT%H:%M:%S") + offset_formatted

    def traces_in_period(self, days: int = 7, **extra) -> list[dict]:
        now = datetime.now()
        since = self._fmt_ts(now - timedelta(days=days + 1))
        till = self._fmt_ts(now + timedelta(hours=1))
        data = self.list_traces(from_timestamp=since, to_timestamp=till, limit=100, **extra)
        return data.get("data", [])

    def generations_for_trace(self, trace_id: str) -> list[dict]:
        data = self.list_observations(
            trace_id=trace_id, observation_type="GENERATION", limit=50
        )
        return data.get("data", [])

    def scores_for_trace(self, trace_id: str, name: Optional[str] = None) -> list[dict]:
        data = self.list_scores(trace_id=trace_id, name=name, limit=50)
        return data.get("data", [])

    # ── usage cost helpers ──────────────────────────────────────

    def total_cost_in_period(self, days: int = 7) -> float:
        """Quick total cost; not a substitute for Langfuse dashboard."""
        traces = self.traces_in_period(days)
        total = 0.0
        for t in traces:
            total += t.get("totalCost", 0) or 0
        return total

    def _rate_limit_delay(self):
        """Small delay between requests to avoid 429."""
        _time.sleep(0.1)

    def close(self):
        self._http.close()
