#!/usr/bin/env python3
"""
策略调整层 CLI。

用法:
  python -m pricing_agent.strategy run --team-id team-a --days 7
  python -m pricing_agent.strategy run --dry-run false --explain false
  python -m pricing_agent.strategy approve <proposal_id>
  python -m pricing_agent.strategy reject <proposal_id> --reason "..."
  python -m pricing_agent.strategy audit --recent 20
  python -m pricing_agent.strategy audit --proposal <proposal_id>
  python -m pricing_agent.strategy rollbacks
"""

import argparse
import json
import logging
import os
import sys
from typing import Optional

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("strategy.cli")


def main():
    parser = argparse.ArgumentParser(description="Strategy Adjustment Layer CLI")
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Run one full strategy cycle")
    run_p.add_argument("--team-id", default="", help="Target team ID")
    run_p.add_argument("--feature", default="", help="Target feature")
    run_p.add_argument("--days", type=int, default=7, help="Lookback days")
    run_p.add_argument("--dry-run", default="true", choices=["true", "false"], help="Skip actual execution")
    run_p.add_argument("--explain", default="true", choices=["true", "false"], help="Generate LLM explanations")
    run_p.add_argument("--json", action="store_true", help="Output as JSON")

    # approve
    approve_p = sub.add_parser("approve", help="Approve a pending proposal")
    approve_p.add_argument("proposal_id", help="Proposal ID to approve")
    approve_p.add_argument("--actor", default="admin", help="Who approved it")

    # reject
    reject_p = sub.add_parser("reject", help="Reject a pending proposal")
    reject_p.add_argument("proposal_id", help="Proposal ID to reject")
    reject_p.add_argument("--reason", default="", help="Rejection reason")
    reject_p.add_argument("--actor", default="admin", help="Who rejected it")

    # audit
    audit_p = sub.add_parser("audit", help="View audit log")
    audit_p.add_argument("--recent", type=int, default=0, help="Show N most recent records")
    audit_p.add_argument("--proposal", default="", help="Show chain for a specific proposal ID")

    # rollbacks
    sub.add_parser("rollbacks", help="Check rollback monitor status")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _run(args)
    elif args.command == "approve":
        _approve(args)
    elif args.command == "reject":
        _reject(args)
    elif args.command == "audit":
        _audit(args)
    elif args.command == "rollbacks":
        _rollbacks()


def _run(args):
    from pricing_agent.strategy.orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator()
    try:
        dry_run = args.dry_run.lower() == "true"
        explain = args.explain.lower() == "true"
        result = orch.run(
            team_id=args.team_id,
            feature=args.feature,
            days=args.days,
            dry_run=dry_run,
            explain=explain,
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        else:
            _print_results(result)
    finally:
        orch.close()


def _approve(args):
    from pricing_agent.strategy.orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator()
    try:
        result = orch.approve_proposal(args.proposal_id, actor=args.actor)
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    finally:
        orch.close()


def _reject(args):
    from pricing_agent.strategy.orchestrator import StrategyOrchestrator

    orch = StrategyOrchestrator()
    try:
        result = orch.reject_proposal(args.proposal_id, reason=args.reason, actor=args.actor)
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    finally:
        orch.close()


def _audit(args):
    from pricing_agent.strategy.audit import AuditStore

    store = AuditStore()
    if args.proposal:
        records = store.get_chain(args.proposal)
        print(f"=== Audit chain for proposal {args.proposal} ===")
        for r in records:
            print(f"  [{r.action}] {r.timestamp} | {r.actor} | {r.details[:120]}")
    elif args.recent > 0:
        records = store.recent(args.recent)
        print(f"=== Last {len(records)} audit records ===")
        for r in records:
            print(f"  [{r.action}] {r.proposal_id[:16]} | {r.timestamp} | {r.actor} | {r.details[:100]}")
    else:
        print("Use --recent N or --proposal <id>")


def _rollbacks():
    from pricing_agent.strategy.rollback import RollbackManager

    mgr = RollbackManager()
    monitors = mgr.all_monitors()
    if not monitors:
        print("No active rollback monitors.")
        return
    print(f"=== {len(monitors)} rollback monitor(s) ===")
    for pid, m in monitors.items():
        print(f"\n  Proposal: {pid}")
        print(f"    Metric: {m.condition.metric} | drop > {m.condition.drop_pct}%")
        print(f"    Window: {m.condition.window_hours}h | min_samples: {m.condition.min_samples}")
        print(f"    Baseline: {m.current_baseline:.4f} | Current: {m.current_value:.4f}")
        print(f"    Samples: {m.samples_collected} | Triggered: {m.triggered} | Resolved: {m.resolved}")


def _print_results(result: dict):
    cycle = result["cycle_info"]
    print(f"\n{'='*60}")
    print(f"  策略调整闭环 — 运行结果")
    print(f"  团队: {cycle['team_id'] or '*'}  功能: {cycle['feature'] or '*'}")
    print(f"  周期: {cycle['days']}d  {'DRY-RUN' if cycle['dry_run'] else 'LIVE'}")
    print(f"{'='*60}")

    signals = result["signals"]
    print(f"\n📊 归因信号:")
    print(f"  效率指标: {len(signals['efficiency'])} 条")
    print(f"  模型选型: {len(signals['model_selection'])} 条")
    print(f"  异常检测: {len(signals['anomaly'])} 条")

    proposals = result["proposals"]
    print(f"\n📋 变更提案 ({len(proposals)}):")
    for i, p in enumerate(proposals):
        prop = p["proposal"]
        guard = p["guardrail"]
        print(f"\n  {'─'*50}")
        print(f"  #{i+1}: {prop['proposal_type']} [{prop['proposal_id'][:12]}]")
        print(f"      目标: {prop['target']}")
        print(f"      当前: {prop['current_state']}")
        print(f"      建议: {prop['suggested_state']}")
        print(f"      风险: {prop['risk_level']} | 节省: ${prop['expected_savings']}/月")
        print(f"      自动执行: {prop['auto_executable']} | 审批: {prop['requires_approval']}")

        if guard["passed"]:
            print(f"      ✅ 护栏通过")
        else:
            print(f"      ❌ 护栏拦截: {guard['failures']}")
        print(f"      审批: {guard['approval_required']} ({guard['approval_hint']})")

        if p["explanation"]:
            print(f"\n      📝 解释: {p['explanation'][:200]}...")

        exec_ = p["execution"]
        if exec_:
            status = "✅" if exec_["success"] else "❌"
            print(f"      执行: {status} {exec_['action_taken']} {exec_.get('error', '')}")

    rollbacks = result.get("rollback_decisions", [])
    if rollbacks:
        print(f"\n🔄 回滚决策 ({len(rollbacks)}):")
        for r in rollbacks:
            print(f"  {r['reason']}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
