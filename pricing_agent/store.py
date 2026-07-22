"""
Store abstraction — automatic switch between JSON and PostgreSQL.

Usage:
  DATABASE_URL=postgresql://...  →  PostgresStore (async)
  no DATABASE_URL                →  JsonStore  (sync, file-backed)
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

from .models import PendingChange, PriceChange

STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".pricing_agent_pending.json",
)

# ── serialization helpers ────────────────────────────────────────


def _serialize(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


def _deserialize_change(d: dict) -> PriceChange:
    return PriceChange(
        change_type=d.get("change_type", ""),
        model_name=d.get("model_name", ""),
        field=d.get("field"),
        old_value=d.get("old_value"),
        new_value=d.get("new_value"),
        description=d.get("description", ""),
        impact=d.get("impact", ""),
        suggested_action=d.get("suggested_action", ""),
    )


def _deserialize_pending(d: dict) -> PendingChange:
    return PendingChange(
        change_id=d.get("change_id", ""),
        created_at=d.get("created_at", ""),
        provider=d.get("provider", ""),
        changes=[_deserialize_change(c) for c in d.get("changes", [])],
        status=d.get("status", "pending"),
        approved_at=d.get("approved_at"),
        rejected_at=d.get("rejected_at"),
        reject_reason=d.get("reject_reason", ""),
    )


def new_change_id(provider: str) -> str:
    return f"pc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{provider}"


# ── abstract store ───────────────────────────────────────────────


class AbstractStore(ABC):
    @abstractmethod
    async def add(self, provider: str, changes: list[PriceChange]) -> PendingChange: ...

    @abstractmethod
    async def pending(self) -> list[PendingChange]: ...

    @abstractmethod
    async def list_all(self) -> list[PendingChange]: ...

    @abstractmethod
    async def get(self, change_id: str) -> Optional[PendingChange]: ...

    @abstractmethod
    async def approve(self, change_id: str) -> bool: ...

    @abstractmethod
    async def reject(self, change_id: str, reason: str = "") -> bool: ...


# ── JSON backend (sync, wrapped with asyncio for uniform API) ───


class JsonStore(AbstractStore):
    def __init__(self, path: Optional[str] = None):
        self._path = path or STORE_PATH
        self._items: list[PendingChange] = []
        self._load()

    def _load(self):
        try:
            with open(self._path) as f:
                raw = json.load(f)
            self._items = [_deserialize_pending(d) for d in raw]
        except (FileNotFoundError, json.JSONDecodeError):
            self._items = []

    def _save(self):
        data = [_serialize(pc) for pc in self._items]
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    async def add(self, provider: str, changes: list[PriceChange]) -> PendingChange:
        pc = PendingChange(
            change_id=new_change_id(provider),
            created_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            changes=changes,
        )
        self._items.append(pc)
        self._save()
        return pc

    async def pending(self) -> list[PendingChange]:
        return [i for i in self._items if i.status == "pending"]

    async def list_all(self) -> list[PendingChange]:
        return list(self._items)

    async def get(self, change_id: str) -> Optional[PendingChange]:
        for i in self._items:
            if i.change_id == change_id:
                return i
        return None

    async def approve(self, change_id: str) -> bool:
        pc = await self.get(change_id)
        if pc is None or pc.status != "pending":
            return False
        pc.status = "approved"
        pc.approved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    async def reject(self, change_id: str, reason: str = "") -> bool:
        pc = await self.get(change_id)
        if pc is None or pc.status != "pending":
            return False
        pc.status = "rejected"
        pc.rejected_at = datetime.now(timezone.utc).isoformat()
        pc.reject_reason = reason
        self._save()
        return True


# ── PostgreSQL backend ────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pending_changes (
    change_id    TEXT PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    provider     TEXT NOT NULL,
    changes      JSONB NOT NULL DEFAULT '[]',
    status       TEXT NOT NULL DEFAULT 'pending',
    approved_at  TIMESTAMPTZ,
    rejected_at  TIMESTAMPTZ,
    reject_reason TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_pending_changes_status ON pending_changes(status);
"""


class PostgresStore(AbstractStore):
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool = None

    async def _ensure_pool(self):
        if self._pool is not None:
            return
        import asyncpg
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    def _row_to_pc(self, row) -> PendingChange:
        return PendingChange(
            change_id=row["change_id"],
            created_at=row["created_at"].isoformat() if row["created_at"] else "",
            provider=row["provider"],
            changes=[_deserialize_change(c) for c in (row["changes"] or [])],
            status=row["status"],
            approved_at=row["approved_at"].isoformat() if row.get("approved_at") else None,
            rejected_at=row["rejected_at"].isoformat() if row.get("rejected_at") else None,
            reject_reason=row.get("reject_reason", ""),
        )

    async def add(self, provider: str, changes: list[PriceChange]) -> PendingChange:
        await self._ensure_pool()
        change_id = new_change_id(provider)
        created = datetime.now(timezone.utc)
        async with self._pool.acquire() as conn:  # type: ignore
            await conn.execute(
                "INSERT INTO pending_changes (change_id, created_at, provider, changes) VALUES ($1, $2, $3, $4)",
                change_id, created, provider, json.dumps([_serialize(c) for c in changes]),
            )
        return PendingChange(
            change_id=change_id,
            created_at=created.isoformat(),
            provider=provider,
            changes=changes,
        )

    async def pending(self) -> list[PendingChange]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:  # type: ignore
            rows = await conn.fetch(
                "SELECT * FROM pending_changes WHERE status = 'pending' ORDER BY created_at DESC",
            )
            return [self._row_to_pc(r) for r in rows]

    async def list_all(self) -> list[PendingChange]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:  # type: ignore
            rows = await conn.fetch(
                "SELECT * FROM pending_changes ORDER BY created_at DESC",
            )
            return [self._row_to_pc(r) for r in rows]

    async def get(self, change_id: str) -> Optional[PendingChange]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:  # type: ignore
            row = await conn.fetchrow(
                "SELECT * FROM pending_changes WHERE change_id = $1", change_id,
            )
            return self._row_to_pc(row) if row else None

    async def approve(self, change_id: str) -> bool:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:  # type: ignore
            result = await conn.execute(
                "UPDATE pending_changes SET status = 'approved', approved_at = NOW() "
                "WHERE change_id = $1 AND status = 'pending'",
                change_id,
            )
            return result != "UPDATE 0"

    async def reject(self, change_id: str, reason: str = "") -> bool:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:  # type: ignore
            result = await conn.execute(
                "UPDATE pending_changes SET status = 'rejected', rejected_at = NOW(), reject_reason = $2 "
                "WHERE change_id = $1 AND status = 'pending'",
                change_id, reason,
            )
            return result != "UPDATE 0"


# ── factory ──────────────────────────────────────────────────────


def create_store() -> AbstractStore:
    dsn = os.environ.get("DATABASE_URL", "")
    if dsn:
        return PostgresStore(dsn)
    return JsonStore()
