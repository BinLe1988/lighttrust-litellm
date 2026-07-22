"""
全链路审计存储。

每次策略调整(提案、审批、执行、回滚)都记录为 AuditRecord。
存储后端可配置: 默认 JSON 文件,也可用 PostgresStore。

每条审计记录包含:
  - record_id, proposal_id, action, timestamp
  - actor (system 或人工用户)
  - details (人类可读描述)
  - data_snapshot (触发该记录的 Langfuse 数据快照)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from .models import AuditRecord

logger = logging.getLogger("strategy.audit")


class AuditStore:
    """Write-audit-log store with JSON file or Postgres backends."""

    def __init__(self, path: str = ""):
        self._path = path or os.environ.get(
            "AUDIT_LOG_PATH",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "audit_log.jsonl",
            ),
        )
        self._records: list[AuditRecord] = []
        self._load()

    def _load(self):
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        self._records.append(AuditRecord(**data))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Failed to load audit log: %s", exc)

    def log(self, record: AuditRecord):
        self._records.append(record)
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record.__dict__, default=str, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed to write audit record: %s", exc)

    def get_by_proposal(self, proposal_id: str) -> list[AuditRecord]:
        return [r for r in self._records if r.proposal_id == proposal_id]

    def get_by_action(self, action: str) -> list[AuditRecord]:
        return [r for r in self._records if r.action == action]

    def recent(self, n: int = 20) -> list[AuditRecord]:
        return self._records[-n:]

    def get_chain(self, proposal_id: str) -> list[AuditRecord]:
        """Get full decision chain for a proposal sorted by time."""
        return sorted(
            self.get_by_proposal(proposal_id),
            key=lambda r: r.timestamp,
        )

    @property
    def all(self) -> list[AuditRecord]:
        return list(self._records)
