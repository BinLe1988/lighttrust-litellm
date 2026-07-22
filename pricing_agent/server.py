"""
FastAPI webhook service for the Pricing Intelligence Agent.

Endpoints:
  POST /agent/check          — trigger pricing check (with optional notify/analyze)
  GET  /agent/pending        — list pending changes
  GET  /agent/change/{id}    — get change detail
  POST /agent/approve/{id}   — approve a change
  POST /agent/reject/{id}    — reject a change
  GET  /agent/status         — health + state summary

Run:
  python -m pricing_agent.agent serve
"""

import os
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn required for server mode. "
        "Install: pip install fastapi uvicorn"
    )

from .store import create_store
from .notifier import CompositeNotifier

store = create_store()
notifier = CompositeNotifier()

app = FastAPI(title="Pricing Agent")


# ── models ────────────────────────────────────────────────────────


class CheckRequest(BaseModel):
    provider: str = ""
    analyze: bool = False
    notify: bool = False
    webhook_url: str = ""


class ChangeDetail(BaseModel):
    id: str
    provider: str
    created_at: str
    status: str
    summary: str
    change_count: int


# ── startup ───────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global store
    await store.list_all()  # warm up connection


# ── endpoints ─────────────────────────────────────────────────────


@app.get("/agent/status")
async def get_status():
    all_items = await store.list_all()
    pending = await store.pending()
    return {
        "ok": True,
        "store_type": "postgres" if "PostgresStore" in type(store).__name__ else "json",
        "notifier_channels": notifier.summarize_channels(),
        "total_changes": len(all_items),
        "pending": len(pending),
        "providers": list({pc.provider for pc in all_items}),
    }


@app.post("/agent/check")
async def webhook_check(req: CheckRequest):
    from .agent import run_check

    try:
        result = await run_check(
            monitor_name=req.provider or None,
            analyze=req.analyze,
        )
        changes = result.get("changes", [])
        if changes:
            last_id = changes[-1]["change_id"]
        else:
            last_id = ""

        if req.notify and last_id:
            pc = await store.get(last_id)
            if pc:
                detail_url = os.environ.get("PRICING_AGENT_PUBLIC_URL", "")
                if not detail_url:
                    host = os.environ.get("PRICING_AGENT_HOST", "localhost")
                    port = os.environ.get("PRICING_AGENT_PORT", "4001")
                    detail_url = f"http://{host}:{port}/agent/change/{last_id}"
                await notifier.send_change_notification(pc, detail_url)

        if req.webhook_url and last_id:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as cli:
                    await cli.post(req.webhook_url, json={
                        "change_id": last_id,
                        "provider": req.provider or "all",
                        "changes": changes,
                    })
            except Exception:
                pass

        return {"ok": True, "changes": len(changes), "change_id": last_id or None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/pending")
async def list_pending(provider: str = Query("", description="filter by provider")):
    items = await store.pending()
    if provider:
        items = [i for i in items if i.provider == provider]
    return [
        ChangeDetail(
            id=pc.change_id,
            provider=pc.provider,
            created_at=pc.created_at,
            status=pc.status,
            summary=f"{len(pc.changes)} change(s)",
            change_count=len(pc.changes),
        )
        for pc in items
    ]


@app.get("/agent/change/{change_id}")
async def get_change(change_id: str):
    pc = await store.get(change_id)
    if pc is None:
        raise HTTPException(404, f"change not found: {change_id}")
    return {
        "id": pc.change_id,
        "provider": pc.provider,
        "created_at": pc.created_at,
        "status": pc.status,
        "approved_at": pc.approved_at,
        "rejected_at": pc.rejected_at,
        "reject_reason": pc.reject_reason,
        "changes": [
            {
                "type": ch.change_type,
                "model": ch.model_name,
                "field": ch.field,
                "old_value": ch.old_value,
                "new_value": ch.new_value,
                "description": ch.description,
                "impact": ch.impact,
                "suggested_action": ch.suggested_action,
            }
            for ch in pc.changes
        ],
    }


@app.post("/agent/approve/{change_id}")
async def approve_change(change_id: str, notify: bool = Query(False)):
    ok = await store.approve(change_id)
    if not ok:
        raise HTTPException(404, f"change not found or not pending: {change_id}")
    return {"status": "approved", "change_id": change_id}


@app.post("/agent/reject/{change_id}")
async def reject_change(change_id: str, reason: str = Query("")):
    ok = await store.reject(change_id, reason)
    if not ok:
        raise HTTPException(404, f"change not found or not pending: {change_id}")
    return {"status": "rejected", "change_id": change_id}
