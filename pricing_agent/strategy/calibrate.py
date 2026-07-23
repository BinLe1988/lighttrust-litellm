"""
三层统计校准：官方账单 ↔ Langfuse ↔ litellm 实时花销。

数据漂移来源：
  - 模型供应商调价后 proxy_server_config.yaml 未同步更新
  - litellm response_cost 按内部定价表估算，与官方计费存在四舍五入差异
  - Langfuse 的 trace.totalCost 来自 SDK 上报，可能因网络或采样缺失

校准流程：
  1. 从官方渠道拉取账单（API 或 CSV）
  2. 从 Langfuse 聚合同周期内成本
  3. 从 BudgetTracker 持久化文件拉取实时花销
  4. 按（模型，周期）分组对比
  5. 漂移超标 → 告警 + 生成校准系数
  6. 可选：自动更新 proxy_server_config.yaml 中的 cost 系数

使用方式:
  calibrator = BillCalibrator()
  report = calibrator.compare(
      official_bills=[BillRecord(...), ...],
      days=30,
  )
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from .langfuse import LangfuseClient, LangfuseConfig

logger = logging.getLogger("strategy.calibrate")

DRIFT_WARN_PCT = float(os.environ.get("CALIBRATE_DRIFT_WARN_PCT", "5"))
DRIFT_CRIT_PCT = float(os.environ.get("CALIBRATE_DRIFT_CRIT_PCT", "15"))
AUTO_CALIBRATE = os.environ.get("CALIBRATE_AUTO", "false").lower() == "true"


# ── 数据模型 ──────────────────────────────────────────────────


@dataclass
class BillRecord:
    """单条官方账单记录（来自 API 或 CSV）。"""

    provider: str            # deepseek / openai / bedi
    model: str               # deepseek-v4-flash
    period_start: str        # 2026-07-01T00:00:00Z
    period_end: str          # 2026-07-31T23:59:59Z
    input_tokens: int = 0
    output_tokens: int = 0
    cache_input_tokens: int = 0
    total_cost: float = 0.0  # USD
    currency: str = "USD"
    line_items: int = 0      # 账单明细行数
    source: str = ""         # api / csv_upload


@dataclass
class InternalCostSlice:
    """Langfuse 或 BudgetTracker 中同一周期同一模型的聚合。"""

    source: str          # langfuse / budget_tracker
    model: str
    total_cost: float = 0.0
    total_tokens: int = 0
    request_count: int = 0
    period_start: str = ""
    period_end: str = ""


@dataclass
class DriftResult:
    """单模型单周期漂移分析结果。"""

    provider: str
    model: str
    period_start: str
    period_end: str
    official_cost: float = 0.0
    langfuse_cost: float = 0.0
    budget_cost: float = 0.0

    langfuse_drift_pct: float = 0.0     # (langfuse - official) / official
    budget_drift_pct: float = 0.0       # (budget - official) / official
    drift_severity: str = "ok"          # ok / warn / crit

    langfuse_tokens: int = 0
    official_tokens: int = 0
    effective_cost_per_token: float = 0.0  # official_cost / official_tokens

    calibration_factor: float = 1.0     # official / langfuse, 乘以 response_cost 可校准

    @property
    def description(self) -> str:
        return (
            f"{self.provider}/{self.model} | "
            f"official=${self.official_cost:.4f} "
            f"langfuse=${self.langfuse_cost:.4f} "
            f"({self.langfuse_drift_pct:+.2f}%) "
            f"budget=${self.budget_cost:.4f} "
            f"({self.budget_drift_pct:+.2f}%) "
            f"→ {self.drift_severity}"
        )


@dataclass
class CalibrationReport:
    report_id: str
    generated_at: str
    period_days: int
    drift_results: list[DriftResult] = field(default_factory=list)
    auto_calibrated: bool = False
    calibration_updates: list[dict] = field(default_factory=list)

    @property
    def summary(self) -> str:
        total = len(self.drift_results)
        crit = sum(1 for d in self.drift_results if d.drift_severity == "crit")
        warn = sum(1 for d in self.drift_results if d.drift_severity == "warn")
        ok = total - crit - warn
        return (
            f"校准报告: {total} 项 | ok={ok} warn={warn} crit={crit} "
            f"auto_calibrated={self.auto_calibrated}"
        )


# ── 官方账单获取器 ─────────────────────────────────────────────


class OfficialBillFetcher:
    """从各供应商 API 拉取账单。

    支持的供应商:
      deepseek — 通过余额/用量 API
      openai   — 通过 Usage API
      bedi     — 目前仅 CSV 导入
    """

    async def fetch(
        self,
        provider: str,
        days: int = 30,
        api_key: str = "",
    ) -> list[BillRecord]:
        fetcher = {
            "deepseek": self._fetch_deepseek,
        }.get(provider)
        if not fetcher:
            logger.warning("No bill fetcher for %s, use CSV upload", provider)
            return []
        return await fetcher(days, api_key)

    async def _fetch_deepseek(self, days: int, api_key: str) -> list[BillRecord]:
        """DeepSeek 官方用量查询 API (账单 v2)。"""
        if not api_key:
            api_key = os.environ.get("DEEPSEEK_BILL_API_KEY", "")
        if not api_key:
            logger.warning("DEEPSEEK_BILL_API_KEY not set, cannot fetch bill")
            return []
        import httpx
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=days)).isoformat()
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
        try:
            async with httpx.AsyncClient(timeout=30) as cli:
                # 假设 DeepSeek 有用量汇总 API
                resp = await cli.get(
                    "https://api.deepseek.com/billing/usage",
                    params={"start_date": since[:10], "end_date": now.strftime("%Y-%m-%d")},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                results = []
                for item in data.get("data", []):
                    results.append(BillRecord(
                        provider="deepseek",
                        model=item.get("model_name", "unknown"),
                        period_start=item.get("start_time", since),
                        period_end=item.get("end_time", now.isoformat()),
                        input_tokens=item.get("input_tokens", 0) or 0,
                        output_tokens=item.get("output_tokens", 0) or 0,
                        cache_input_tokens=item.get("cached_input_tokens", 0) or 0,
                        total_cost=item.get("total_cost", 0) or 0,
                        source="api",
                    ))
                return results
        except Exception as exc:
            logger.error("Failed to fetch DeepSeek bill: %s", exc)
            return []

    @staticmethod
    def from_csv(path: str) -> list[BillRecord]:
        """从 CSV 导入账单（通用格式）。"""
        import csv
        records = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(BillRecord(
                    provider=row.get("provider", ""),
                    model=row.get("model", ""),
                    period_start=row.get("period_start", ""),
                    period_end=row.get("period_end", ""),
                    input_tokens=int(row.get("input_tokens", 0) or 0),
                    output_tokens=int(row.get("output_tokens", 0) or 0),
                    cache_input_tokens=int(row.get("cache_input_tokens", 0) or 0),
                    total_cost=float(row.get("total_cost", 0) or 0),
                    source="csv_upload",
                ))
        return records

    @staticmethod
    def from_json(path: str) -> list[BillRecord]:
        with open(path) as f:
            raw = json.load(f)
        return [BillRecord(**item) for item in raw]


# ── 内部数据采集器 ─────────────────────────────────────────────


class InternalCollector:
    """从 Langfuse + BudgetTracker 采集同周期内部成本数据。"""

    def __init__(self, lf: Optional[LangfuseClient] = None):
        self._lf = lf

    def _batch_generations(self, days: int, provider: str = "") -> dict[str, list[dict]]:
        """批量获取所有 generations, 按 traceId 索引 (避免 N+1)。"""
        import time as _time
        now = datetime.now(timezone.utc)
        since = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S") + "Z"
        indexed: dict[str, list[dict]] = {}
        page = 1
        while True:
            data = self._lf.list_observations(
                limit=100, page=page,
                from_timestamp=since,
                observation_type="GENERATION",
            )
            items = data.get("data", [])
            if not items:
                break
            for gen in items:
                tid = gen.get("traceId")
                if not tid:
                    continue
                model = gen.get("model", "unknown")
                if provider and provider not in model:
                    continue
                indexed.setdefault(tid, []).append(gen)
            meta = data.get("meta", {})
            total_pages = meta.get("totalPages", 1) or 1
            if page >= total_pages:
                break
            page += 1
            _time.sleep(3.0)
        return indexed

    def from_langfuse(
        self,
        days: int = 30,
        provider: str = "",
    ) -> list[InternalCostSlice]:
        """从 Langfuse 聚合模型级成本（批量查询，无 N+1）。"""
        if not self._lf:
            return []

        gens_indexed = self._batch_generations(days, provider)
        slices: dict[str, InternalCostSlice] = {}

        for trace_id, gens in gens_indexed.items():
            for gen in gens:
                model = gen.get("model", "unknown")
                cost = gen.get("cost", 0) or 0
                usage = gen.get("usage", {}) or {}
                tokens = (usage.get("totalTokens", 0) or 0) if isinstance(usage, dict) else 0
                key = model
                if key not in slices:
                    slices[key] = InternalCostSlice(
                        source="langfuse", model=model,
                        period_start=(datetime.now(timezone.utc) - timedelta(days=days)).isoformat(),
                        period_end=datetime.now(timezone.utc).isoformat(),
                    )
                slices[key].total_cost += cost
                slices[key].total_tokens += tokens
                slices[key].request_count += 1

        return list(slices.values())

    @staticmethod
    def from_budget_file(path: str = "") -> list[InternalCostSlice]:
        """从 BudgetTracker 持久化文件读取花销汇总。"""
        path = path or os.environ.get("BUDGET_PERSIST_FILE", "")
        if not path:
            return []
        try:
            with open(path) as f:
                spend = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        return [
            InternalCostSlice(
                source="budget_tracker",
                model="(aggregated)",
                total_cost=float(v),
                request_count=0,
            )
            for k, v in spend.items()
        ]


# ── 漂移分析 & 校准器 ──────────────────────────────────────────


class BillCalibrator:
    """三层统计算校准核心。"""

    def __init__(
        self,
        lf_client: Optional[LangfuseClient] = None,
    ):
        self._fetcher = OfficialBillFetcher()
        self._collector = InternalCollector(lf_client)

    def compare(
        self,
        official_bills: Optional[list[BillRecord]] = None,
        days: int = 30,
        provider: str = "",
    ) -> CalibrationReport:
        """运行一次完整校准周期。

        两种模式：
          1. 传入 official_bills → 按提供的数据对比
          2. 不传 → 尝试从 API 拉取 + CSV 导入
        """
        now = datetime.now(timezone.utc)
        report = CalibrationReport(
            report_id=self._gen_id(),
            generated_at=now.isoformat(),
            period_days=days,
        )

        # 1. 获取官方账单
        bills = official_bills or []

        # 2. 采集内部数据
        langfuse_slices = self._collector.from_langfuse(days, provider)
        budget_slices = self._collector.from_budget_file()

        # 3. 按模型匹配对比
        langfuse_by_model = {s.model: s for s in langfuse_slices}
        budget_total = sum(s.total_cost for s in budget_slices)

        for bill in bills:
            model_key = bill.model
            lf_slice = langfuse_by_model.get(model_key)
            lf_cost = lf_slice.total_cost if lf_slice else 0.0

            # 漂移计算
            lf_drift = 0.0
            b_drift = 0.0
            if bill.total_cost > 0:
                lf_drift = (lf_cost - bill.total_cost) / bill.total_cost * 100
                b_drift = (budget_total - bill.total_cost) / bill.total_cost * 100

            severity = "ok"
            if abs(lf_drift) >= DRIFT_CRIT_PCT:
                severity = "crit"
            elif abs(lf_drift) >= DRIFT_WARN_PCT:
                severity = "warn"

            cal_factor = 1.0
            if lf_cost > 0 and bill.total_cost > 0:
                cal_factor = bill.total_cost / lf_cost

            dr = DriftResult(
                provider=bill.provider,
                model=model_key,
                period_start=bill.period_start,
                period_end=bill.period_end,
                official_cost=bill.total_cost,
                langfuse_cost=lf_cost,
                budget_cost=budget_total,
                langfuse_drift_pct=round(lf_drift, 2),
                budget_drift_pct=round(b_drift, 2),
                drift_severity=severity,
                langfuse_tokens=lf_slice.total_tokens if lf_slice else 0,
                official_tokens=bill.input_tokens + bill.output_tokens,
                effective_cost_per_token=bill.total_cost / max(bill.input_tokens + bill.output_tokens, 1),
                calibration_factor=round(cal_factor, 6),
            )
            report.drift_results.append(dr)

        # 4. 自动校准（可选）
        if AUTO_CALIBRATE:
            updates = self._auto_calibrate(report.drift_results)
            report.auto_calibrated = True
            report.calibration_updates = updates

        return report

    # ── 自动校准 ──────────────────────────────────────────────

    def _auto_calibrate(self, drift_results: list[DriftResult]) -> list[dict]:
        """对漂移超过 crit 阈值的模型生成校准条目。

        校准方式：
          - 修改 proxy_server_config.yaml 中的 model_cost_map 或 cost 字段
          - 或生成一个 cost_multiplier 文件供 BudgetTracker 读取
        """
        updates = []
        for dr in drift_results:
            if dr.drift_severity != "crit":
                continue
            update = {
                "provider": dr.provider,
                "model": dr.model,
                "field": "calibration_factor",
                "old_value": 1.0,
                "new_value": dr.calibration_factor,
                "reason": (
                    f"Langfuse drift {dr.langfuse_drift_pct:+.2f}% "
                    f"(official=${dr.official_cost:.4f} vs lf=${dr.langfuse_cost:.4f})"
                ),
            }
            updates.append(update)
            logger.warning(
                "Calibration for %s/%s: factor %.6f",
                dr.provider, dr.model, dr.calibration_factor,
            )

        if updates:
            self._write_calibration_file(updates)
        return updates

    def _write_calibration_file(self, updates: list[dict]):
        path = os.environ.get(
            "CALIBRATION_FILE",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "calibration_factors.json",
            ),
        )
        existing = {}
        try:
            with open(path) as f:
                existing = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        for u in updates:
            existing[f"{u['provider']}/{u['model']}"] = {
                "factor": u["new_value"],
                "reason": u["reason"],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        with open(path, "w") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        logger.info("Wrote %d calibration factors to %s", len(updates), path)

    @staticmethod
    def _gen_id() -> str:
        import hashlib, time
        return hashlib.sha256(f"cal{time.time_ns()}".encode()).hexdigest()[:12]

    @staticmethod
    def print_report(report: CalibrationReport):
        """终端友好输出。"""
        print(f"\n{'='*60}")
        print(f"  成本校准报告 [{report.report_id}]")
        print(f"  周期: {report.period_days} 天 | 自动校准: {report.auto_calibrated}")
        print(f"{'='*60}")
        if not report.drift_results:
            print("  (无账单数据可对比)")
            return
        for dr in report.drift_results:
            icon = {"ok": "✅", "warn": "⚠️", "crit": "🔥"}.get(dr.drift_severity, "❓")
            print(f"\n  {icon} {dr.description}")
            print(f"     官方 token: {dr.official_tokens:,} | Langfuse token: {dr.langfuse_tokens:,}")
            print(f"     校准系数: {dr.calibration_factor:.6f}")
        print(f"\n{report.summary}")
        if report.calibration_updates:
            print(f"\n  自动校准更新 ({len(report.calibration_updates)} 项):")
            for u in report.calibration_updates:
                print(f"    {u['provider']}/{u['model']}: {u['old_value']} → {u['new_value']}")
        print(f"\n{'='*60}\n")
