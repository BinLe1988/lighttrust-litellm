import json
import os
from datetime import datetime, timezone
from typing import Optional
from .models import PendingChange, PriceChange

STORE_PATH = os.path.join(
    os.path.dirname(__file__), "..", ".pricing_agent_pending.json",
)


def _serialize(obj):
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, float):
        return obj
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


class PendingChangeStore:
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

    def add(self, provider: str, changes: list[PriceChange]) -> PendingChange:
        from datetime import datetime, timezone
        pc = PendingChange(
            change_id=f"pc_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{provider}",
            created_at=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            changes=changes,
        )
        self._items.append(pc)
        self._save()
        return pc

    def pending(self) -> list[PendingChange]:
        return [i for i in self._items if i.status == "pending"]

    def list_all(self) -> list[PendingChange]:
        return list(self._items)

    def get(self, change_id: str) -> Optional[PendingChange]:
        for i in self._items:
            if i.change_id == change_id:
                return i
        return None

    def approve(self, change_id: str) -> bool:
        pc = self.get(change_id)
        if pc is None or pc.status != "pending":
            return False
        pc.status = "approved"
        pc.approved_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return True

    def reject(self, change_id: str, reason: str = "") -> bool:
        pc = self.get(change_id)
        if pc is None or pc.status != "pending":
            return False
        pc.status = "rejected"
        pc.rejected_at = datetime.now(timezone.utc).isoformat()
        pc.reject_reason = reason
        self._save()
        return True
