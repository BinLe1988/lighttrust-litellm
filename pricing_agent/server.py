"""
FastAPI webhook service for the Pricing Intelligence Agent.

Provides endpoints for:
  POST /agent/check          — trigger a pricing check
  GET  /agent/pending        — list pending changes
  POST /agent/approve/{id}   — approve a change
  POST /agent/reject/{id}    — reject a change
  GET  /agent/status         — agent health & state

Run:
  uvicorn pricing_agent.server:app --host 0.0.0.0 --port 4001

Or via agent CLI:
  python -m pricing_agent.agent serve
"""

import os
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn required for server mode. "
        "Install: pip install fastapi uvicorn"
    )

from .agent import store, run_check
from .store import PendingChangeStore

app = FastAPI(title="Pricing Agent")


class CheckRequest(BaseModel):
    provider: str = ""
    webhook_url: str = ""
    analyze: bool = False


class ApproveResponse(BaseModel):
    status: str
    change_id: str


@app.get("/agent/status")
async def status():
    all_items = store.list_all()
    pending = store.pending()
    return {
        "ok": True,
        "total_changes": len(all_items),
        "pending": len(pending),
        "providers": list({pc.provider for pc in all_items}),
    }


@app.post("/agent/check")
async def webhook_check(req: CheckRequest):
    try:
        result = await run_check(
            monitor_name=req.provider or None,
            analyze=req.analyze,
            webhook_url=req.webhook_url or None,
        )
        return {"ok": True, "changes": len(result.get("changes", [])), "detail": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/pending")
async def list_pending():
    return [{"id": pc.change_id, "provider": pc.provider, "created_at": pc.created_at}
            for pc in store.pending()]


@app.post("/agent/approve/{change_id}")
async def approve(change_id: str):
    ok = store.approve(change_id)
    if not ok:
        raise HTTPException(404, f"change not found or not pending: {change_id}")
    return ApproveResponse(status="approved", change_id=change_id)


@app.post("/agent/reject/{change_id}")
async def reject(change_id: str, reason: str = ""):
    ok = store.reject(change_id, reason)
    if not ok:
        raise HTTPException(404, f"change not found or not pending: {change_id}")
    return ApproveResponse(status="rejected", change_id=change_id)
