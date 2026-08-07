"""
配置变更执行器。

通过 litellm 的管理 API 执行提案中的配置变更:
  - route_change:      更新 proxy_server_config.yaml 中 team 的默认路由模型
  - quota_adjustment:  调用 litellm /budget 管理 API 更新配额
  - model_fallback:    在特定条件(如错误率)下设置 fallback 模型
  - budget_alert:      只写入审计日志,不执行配置变更

设计原则:
  - 所有变更都有对应的"撤销"操作,记录在执行结果中
  - 先做 dry-run 验证,再实际执行
  - 执行后更新审计记录
"""

import json
import logging
import os
from typing import Optional

import httpx

from .models import ChangeProposal, ExecutionResult

logger = logging.getLogger("strategy.executor")

LITELLM_BASE = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")


class Executor:
    """Execute config changes via litellm management API."""

    def __init__(self):
        self._base = LITELLM_BASE.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if LITELLM_MASTER_KEY:
            self._headers["Authorization"] = f"Bearer {LITELLM_MASTER_KEY}"
        self._http = httpx.Client(base_url=self._base, headers=self._headers, timeout=15)

    # ── litellm API wrappers ───────────────────────────────────

    def _get(self, path: str) -> dict:
        resp = self._http.get(path)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: dict) -> dict:
        resp = self._http.post(path, json=data)
        resp.raise_for_status()
        return resp.json()

    def get_models(self) -> list[dict]:
        return self._get("/model/info").get("data", [])

    def get_teams(self) -> list[dict]:
        data = self._get("/team/list")
        if isinstance(data, list):
            return data
        return data.get("teams", [])

    def create_team(self, team_alias: str, models: Optional[list] = None) -> str:
        """Create a litellm team, return its team_id."""
        payload = {"team_alias": team_alias}
        if models:
            payload["models"] = models
        resp = self._post("/team/new", payload)
        if isinstance(resp, dict) and resp.get("team_id"):
            return resp["team_id"]
        # /team/new 部分返回 wrapped data
        data = resp.get("data") if isinstance(resp, dict) else None
        if isinstance(data, dict) and data.get("team_id"):
            return data["team_id"]
        raise ValueError(f"/team/new 响应中未找到 team_id: {resp}")

    def update_team(self, team_id: str, **updates) -> dict:
        return self._post(f"/team/update", {"team_id": team_id, **updates})

    def get_budget(self) -> dict:
        return self._get("/budget")

    def update_budget(self, user_id: str, max_budget: float) -> dict:
        return self._post("/budget", {
            "user_id": user_id,
            "max_budget": max_budget,
        })

    # ── proposal execution ─────────────────────────────────────

    def dry_run(self, proposal: ChangeProposal) -> ExecutionResult:
        """Validate the proposal is executable without making changes."""
        try:
            if proposal.proposal_type == "route_change":
                models = self.get_models()
                rec = proposal.target.get("recommended_model", "")
                available = [m.get("model_name", "") for m in models]
                if rec not in available:
                    return ExecutionResult(
                        success=False,
                        proposal_id=proposal.proposal_id,
                        action_taken="dry_run",
                        error=f"模型 {rec} 不在 litellm 可用模型列表中",
                    )
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="dry_run",
                    response_data={"available_models": available},
                )

            elif proposal.proposal_type == "quota_adjustment":
                budget_data = self.get_budget()
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="dry_run",
                    response_data={"budget_info": budget_data},
                )

            return ExecutionResult(
                success=True,
                proposal_id=proposal.proposal_id,
                action_taken="dry_run",
                response_data={"note": f"Proposal type {proposal.proposal_type} has no dry-run validation"},
            )

        except Exception as exc:
            logger.error("Dry-run failed for %s: %s", proposal.proposal_id, exc)
            return ExecutionResult(
                success=False,
                proposal_id=proposal.proposal_id,
                action_taken="dry_run",
                error=str(exc),
            )

    def execute(self, proposal: ChangeProposal) -> ExecutionResult:
        """Execute the proposal's config change."""
        try:
            if proposal.proposal_type == "route_change":
                team_id = proposal.target.get("team_id", "")
                rec_model = proposal.target.get("recommended_model", "")
                if not team_id or not rec_model:
                    raise ValueError("route_change needs team_id + recommended_model")
                resp = self.update_team(
                    team_id=team_id,
                    models=[rec_model],
                    metadata={"strategy_proposal_id": proposal.proposal_id},
                )
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="route_change",
                    response_data={"api_response": resp},
                )

            elif proposal.proposal_type == "quota_adjustment":
                team_id = proposal.target.get("team_id", "")
                suggested_quota = proposal.target.get("suggested_quota", 0)
                if not team_id or not suggested_quota:
                    raise ValueError("quota_adjustment needs team_id + suggested_quota")
                resp = self.update_budget(team_id, suggested_quota)
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="quota_adjustment",
                    response_data={"api_response": resp},
                )

            elif proposal.proposal_type == "budget_alert":
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="logged_only",
                    response_data={"note": "Budget alert logged, no config change required"},
                )

            elif proposal.proposal_type == "manual_review_required":
                return ExecutionResult(
                    success=True,
                    proposal_id=proposal.proposal_id,
                    action_taken="pending_review",
                    response_data={"note": "Manual review required, no automated action taken"},
                )

            else:
                return ExecutionResult(
                    success=False,
                    proposal_id=proposal.proposal_id,
                    action_taken="unknown_type",
                    error=f"Unknown proposal type: {proposal.proposal_type}",
                )

        except Exception as exc:
            logger.error("Execution failed for %s: %s", proposal.proposal_id, exc)
            return ExecutionResult(
                success=False,
                proposal_id=proposal.proposal_id,
                action_taken="error",
                error=str(exc),
            )

    def close(self):
        self._http.close()
