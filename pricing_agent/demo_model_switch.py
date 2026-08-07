#!/usr/bin/env python3
"""
模型自动切换演示脚本（旗舰模型过度配置 → 自动降级到轻量模型）。

演示闭环:
  ① 模拟数据: 某功能长期使用旗舰模型, 质量分无差异
  ② 系统生成建议: 生成 route_change 提案 (可自动执行)
  ③ 安全护栏: 冷却/幅度/审批门禁评估
  ④ 现场自动切换: 通过 litellm /team/update 真实切换到轻量模型
  ⑤ 回滚阈值展示: 展示监控配置, 不真正触发

用法:
  python3 pricing_agent/demo_model_switch.py
  python3 pricing_agent/demo_model_switch.py --json
  python3 pricing_agent/demo_model_switch.py --team-id demo-team --explain

前置条件:
  - litellm 代理运行在 :4000
  - LITELLM_MASTER_KEY 已配置 (env 或通过 .env)
"""

import argparse
import json
import logging
import os
import sys

# 保证无论从哪个目录/方式运行, pricing_agent 包都能被导入
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pricing_agent.strategy.models import ModelSelectionSignal
from pricing_agent.strategy.proposals import ProposalEngine
from pricing_agent.strategy.guardrails import GuardrailEngine
from pricing_agent.strategy.executor import Executor
from pricing_agent.strategy.rollback import RollbackManager

# ── 演示参数 ──────────────────────────────────────────────
DEMO_TEAM = "demo-team"
DEMO_FEATURE = "chat_summary"
FLAGSHIP = "deepseek-v4-pro"
LIGHT = "deepseek-v4-flash"

QUALITY_FLAGSHIP = 92.3   # 旗舰模型近30天平均质量分
QUALITY_LIGHT = 91.8      # 轻量模型质量分 (差异 0.5 < 5%)
COST_PER_CALL_FLAGSHIP = 0.0015
COST_PER_CALL_LIGHT = 0.0003
REQ_COUNT = 3200
SAVINGS_PCT = 80.0
SAVINGS_MONTHLY = 38.40

# ── 输出辅助 ──────────────────────────────────────────────
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"
C_RED = "\033[31m"
C_DIM = "\033[2m"


def _title(s: str):
    print(f"\n{C_BOLD}{'═' * 62}{C_RESET}")
    print(f"{C_BOLD}{s}{C_RESET}")
    print(f"{C_BOLD}{'═' * 62}{C_RESET}")


def _step(n: int, title: str):
    print(f"\n{C_CYAN}── 步骤 {n}: {title} ──{C_RESET}")


def _kv(k: str, v: str, color: str = ""):
    print(f"  {k:<22} {color}{v}{C_RESET}")


def _ok(s: str):
    print(f"  {C_GREEN}✔ {s}{C_RESET}")


def _warn(s: str):
    print(f"  {C_YELLOW}! {s}{C_RESET}")


# ── ① 模拟数据 ───────────────────────────────────────────
def build_simulated_signal() -> ModelSelectionSignal:
    """模拟「某功能长期用旗舰模型但质量分没差异」的检测结果。

    数值刻意满足自动执行门槛:
      - cost_savings_pct = 80% (> 20%)
      - request_count = 3200 (>= 100 → risk=low, 免审批)
      - quality_diff = 0.5 (< 5% → 无质量差异)
    """
    return ModelSelectionSignal(
        team_id=DEMO_TEAM,
        feature=DEMO_FEATURE,
        task_type="summarization",
        current_model=FLAGSHIP,
        recommended_model=LIGHT,
        current_cost_per_call=COST_PER_CALL_FLAGSHIP,
        recommended_cost_per_call=COST_PER_CALL_LIGHT,
        quality_diff=QUALITY_FLAGSHIP - QUALITY_LIGHT,
        quality_diff_description=(
            f"质量分差异 {QUALITY_FLAGSHIP - QUALITY_LIGHT:.1f} < 5% "
            f"(当前={QUALITY_FLAGSHIP:.1f}, 推荐={QUALITY_LIGHT:.1f})"
        ),
        cost_savings_pct=SAVINGS_PCT,
        cost_savings_monthly=SAVINGS_MONTHLY,
        confidence="high",
        request_count=REQ_COUNT,
    )


# ── 主流程 ────────────────────────────────────────────────
def run_demo(team_id: str, explain: bool, json_out: bool) -> dict:
    # ① 模拟数据
    signal = build_simulated_signal()

    # ② 生成建议
    engine = ProposalEngine()
    proposal = engine.from_model_selection(signal)

    # ③ 安全护栏
    guardrails = GuardrailEngine()
    guard = guardrails.evaluate(proposal)

    # ④ 执行器 (真实调用 litellm API)
    executor = Executor()
    rollback = RollbackManager()
    exec_result = None
    team_before = None
    team_after = None
    monitor_state = None

    try:
        # ---- 确保 demo 团队存在且处于旗舰模型初始状态 ----
        teams = executor.get_teams()
        existing = next(
            (t for t in teams if t.get("team_id") == team_id or t.get("team_alias") == team_id),
            None,
        )
        if existing:
            target_id = existing["team_id"]
            executor.update_team(target_id, models=[FLAGSHIP])
        else:
            target_id = executor.create_team(team_alias=team_id, models=[FLAGSHIP])

        proposal.target["team_id"] = target_id
        team_before = {"team_id": target_id, "models": [FLAGSHIP]}

        # ---- dry-run 校验 ----
        dry = executor.dry_run(proposal)

        # ---- 真实执行切换 ----
        if guard.passed and proposal.auto_executable and dry.success:
            exec_result = executor.execute(proposal)
        else:
            from types import SimpleNamespace
            exec_result = SimpleNamespace(
                success=False,
                action_taken="blocked",
                error=(
                    f"guardrail.passed={guard.passed} "
                    f"auto={proposal.auto_executable} dry={dry.success}"
                ),
                response_data={},
            )

        # ---- 回读团队确认 ----
        teams_after = executor.get_teams()
        after = next((t for t in teams_after if t.get("team_id") == target_id), None)
        team_after = {"team_id": target_id, "models": (after or {}).get("models") or []}

        # ---- 布防回滚监控 ----
        if exec_result.success and proposal.rollback_condition:
            rollback.start_monitoring(
                proposal.proposal_id,
                proposal.rollback_condition,
                initial_baseline=QUALITY_FLAGSHIP,
            )
            # 记录一个健康样本 → 监控中, 未触发
            rollback.record_sample(proposal.proposal_id, QUALITY_LIGHT)
            check = rollback.check(proposal.proposal_id)
            m = rollback.all_monitors().get(proposal.proposal_id)
            monitor_state = {
                "armed": True,
                "triggered": bool(check),
                "metric": proposal.rollback_condition.metric,
                "drop_pct": proposal.rollback_condition.drop_pct,
                "window_hours": proposal.rollback_condition.window_hours,
                "min_samples": proposal.rollback_condition.min_samples,
                "baseline": QUALITY_FLAGSHIP,
                "latest_sample": QUALITY_LIGHT,
            }
    finally:
        executor.close()

    # ⑤ 解释 (可选)
    explanation = ""
    if explain and exec_result.success:
        try:
            from pricing_agent.strategy.explainer import explain_proposal
            explanation = explain_proposal(proposal, guard)
        except Exception as e:  # noqa: BLE001
            explanation = f"（解释生成失败: {e}）"

    # ── 输出 ──
    result = {
        "step1_signal": signal.__dict__,
        "step2_proposal": proposal.__dict__,
        "step3_guardrail": guard.__dict__,
        "step4_execution": exec_result.__dict__,
        "team_before": team_before,
        "team_after": team_after,
        "step5_rollback": monitor_state,
        "explanation": explanation,
    }

    if json_out:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return result

    # ── 终端渲染 ──
    _title("  模型自动切换演示 — 旗舰模型过度配置自动降级")

    _step(1, "模拟数据: 某功能长期用旗舰模型, 质量分无差异")
    _kv("团队 / 功能", f"{signal.team_id} / {signal.feature}")
    _kv("任务类型", signal.task_type)
    _kv("当前模型", f"{C_RED}{signal.current_model} (旗舰){C_RESET}")
    _kv("质量分", f"旗舰={QUALITY_FLAGSHIP:.1f} vs 轻量={QUALITY_LIGHT:.1f} → 差异 {signal.quality_diff:.1f} (< 5%)")
    _kv("单次成本", f"旗舰=${COST_PER_CALL_FLAGSHIP:.4f} vs 轻量=${COST_PER_CALL_LIGHT:.4f}")
    _kv("请求量 / 周期", f"{signal.request_count} 次 / 30 天")
    _ok("检测结论: 简单任务长期使用旗舰模型, 质量无显著差异 → model_overqualified")

    _step(2, "系统生成建议")
    _kv("提案类型", f"{C_BOLD}{proposal.proposal_type}{C_RESET} (route_change)")
    _kv("提案 ID", proposal.proposal_id)
    _kv("当前状态", proposal.current_state)
    _kv("建议状态", f"{C_GREEN}{proposal.suggested_state}{C_RESET}")
    _kv("风险等级", f"{proposal.risk_level} | 审批: {proposal.requires_approval or 'none'}")
    _kv("自动执行", f"{C_GREEN}{proposal.auto_executable}{C_RESET}")
    _kv("预期节省", f"${proposal.expected_savings:.2f} / 月")
    _kv("支撑信号数", str(len(proposal.supporting_signals)))

    _step(3, "安全护栏评估")
    if guard.passed:
        _ok("护栏全部通过")
    else:
        _warn(f"护栏拦截: {guard.failures}")
    _kv("冷却期检查", json.dumps(guard.cooldown_check or {}, ensure_ascii=False))
    _kv("幅度上限检查", json.dumps(guard.amplitude_check or {}, ensure_ascii=False))
    _kv("审批要求", f"{guard.approval_required} ({guard.approval_hint})")

    _step(4, "现场自动切换 (真实执行)")
    _kv("目标团队", team_before["team_id"] if team_before else team_id)
    _kv("切换前", f"models = {team_before['models'] if team_before else '?'}")
    _kv("切换动作", f"POST /team/update → models = [{LIGHT}]")
    if exec_result.success:
        _ok(f"执行成功: {exec_result.action_taken}")
        _kv("切换后", f"models = {C_GREEN}{team_after['models']}{C_RESET}")
        _ok("已自动切换为轻量模型")
    else:
        _warn(f"执行未成功: {exec_result.error}")
        _kv("切换后", f"models = {team_after['models'] if team_after else '?'}")

    _step(5, "回滚阈值配置 (仅展示, 不触发)")
    if proposal.rollback_condition:
        rc = proposal.rollback_condition
        _kv("监控指标", rc.metric)
        _kv("触发阈值", f"质量分下降 ≥ {rc.drop_pct:.0f}%")
        _kv("观察窗口", f"{rc.window_hours}h")
        _kv("最少样本", f"{rc.min_samples} 个")
        _kv("基线", f"{QUALITY_FLAGSHIP:.1f} → 触发线 {QUALITY_FLAGSHIP * (1 - rc.drop_pct / 100):.1f}")
    if monitor_state and monitor_state["armed"]:
        _ok(f"回滚监控已布防 (baseline={monitor_state['baseline']:.1f}, 最新样本={monitor_state['latest_sample']:.1f})")
        _ok("健康样本: 降幅 0.5% < 5% → 监控中, 未触发")
        _warn(
            f"若 24h 内质量分跌破 {QUALITY_FLAGSHIP * (1 - 5 / 100):.1f} "
            f"且样本 ≥ 10 个, 将自动回滚到 {FLAGSHIP} —— 演示到此为止, 不触发"
        )
    else:
        _warn("回滚监控未布防")

    if explanation:
        print(f"\n{C_DIM}📝 系统解释: {explanation}{C_RESET}")

    print(f"\n{C_BOLD}{'═' * 62}{C_RESET}\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="模型自动切换演示")
    parser.add_argument("--team-id", default=DEMO_TEAM, help="演示团队名/ID")
    parser.add_argument("--explain", action="store_true", help="生成中文自然语言解释 (需 DEEPSEEK_API_KEY)")
    parser.add_argument("--json", dest="json_out", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    try:
        run_demo(team_id=args.team_id, explain=args.explain, json_out=args.json_out)
    except Exception as exc:  # noqa: BLE001
        print(f"{C_RED}演示失败: {exc}{C_RESET}", file=sys.stderr)
        print(
            f"{C_YELLOW}请确认 litellm 代理已启动: "
            f"`litellm --config proxy_server_config.yaml --port 4000` "
            f"且 LITELLM_MASTER_KEY 已配置。{C_RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
