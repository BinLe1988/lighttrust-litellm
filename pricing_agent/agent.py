#!/usr/bin/env python3
"""
Pricing Intelligence Agent — CLI tool for litellm.

Usage:
  python -m pricing_agent.agent check           fetch & diff vendor prices
  python -m pricing_agent.agent check --model M  use specific LLM for extraction
  python -m pricing_agent.agent check --analyze  also run LLM impact analysis
  python -m pricing_agent.agent review           review pending changes
  python -m pricing_agent.agent apply            apply approved changes
  python -m pricing_agent.agent status           show current state
  python -m pricing_agent.agent daemon           run periodic checks in background
  python -m pricing_agent.agent serve            start FastAPI webhook server
"""

import asyncio
import os
import sys
from typing import Optional

from .models import PendingChange, PriceChange
from .monitors.registry import get_monitor, list_monitors
from .differ import diff_snapshot, CONFIG_PATH
from .store import AbstractStore, create_store
from .reporter import format_changes, format_pending
from .notifier import CompositeNotifier


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m"


store: AbstractStore = create_store()
notifier = CompositeNotifier()


async def run_check(
    monitor_name: Optional[str] = None,
    analyze: bool = False,
    webhook_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Shared check logic used by CLI, server, and daemon modes."""
    monitors = [monitor_name] if monitor_name else list_monitors()
    result: dict = {"changes": [], "analyses": [], "notified": False}

    for name in monitors:
        mon = get_monitor(name)
        if llm_model:
            mon.llm_model = llm_model

        snap = await mon.fetch()
        if not snap.models:
            continue

        changes = diff_snapshot(snap)
        if not changes:
            continue

        pc = await store.add(provider=name, changes=changes)
        result["changes"].append({
            "change_id": pc.change_id,
            "provider": name,
            "count": len(changes),
            "changes": [ch.detail() for ch in changes],
        })

        if analyze:
            from .analyzer import analyze_changes
            analyses = await analyze_changes(changes, model=llm_model)
            if analyses:
                result["analyses"].extend(analyses)
                for analysis in analyses:
                    rec = analysis.get("recommendation", "")
                    if rec == "apply":
                        await store.approve(pc.change_id)
                    elif rec == "skip":
                        await store.reject(pc.change_id, "LLM recommended skip")

        if notifier.enabled:
            detail_url = os.environ.get(
                "PRICING_AGENT_PUBLIC_URL",
                f"http://localhost:{os.environ.get('PRICING_AGENT_PORT', '4001')}",
            )
            await notifier.send_change_notification(
                pc, f"{detail_url}/agent/change/{pc.change_id}",
            )
            result["notified"] = True

        if webhook_url:
            try:
                import httpx
                async with httpx.AsyncClient(timeout=10) as cli:
                    await cli.post(webhook_url, json={
                        "change_id": pc.change_id,
                        "provider": name,
                        "changes": [ch.detail() for ch in changes],
                    })
                result["notified"] = True
            except Exception:
                pass

    return result


async def cmd_check(
    monitor_name: Optional[str] = None,
    analyze: bool = False,
    llm_model: Optional[str] = None,
):
    print(_cyan("\n=== Pricing Intelligence: check ==="))

    monitors = [monitor_name] if monitor_name else list_monitors()

    for name in monitors:
        print(f"\n  fetching {name} pricing...", end=" ", flush=True)
        try:
            mon = get_monitor(name)
            if llm_model:
                mon.llm_model = llm_model
            snap = await mon.fetch()
            print(
                _green(f"OK ({len(snap.models)} model(s))")
                if snap.models
                else _yellow("no models returned")
            )
        except Exception as e:
            print(_red(f"ERROR: {e}"))
            continue

        for mp in snap.models:
            print(
                f"    {mp.model_name:24s}"
                f"  in ${mp.input_cost_per_token:.2e}"
                f"  out ${mp.output_cost_per_token:.2e}"
                f"  cache {f'${mp.cache_read_input_token_cost:.2e}' if mp.cache_read_input_token_cost else '—':>12s}"
            )

        print(f"  diffing {name} against config...", end=" ", flush=True)
        try:
            changes = diff_snapshot(snap)
            print(
                _green(f"{len(changes)} change(s) detected")
                if changes
                else _green("no changes")
            )
        except Exception as e:
            print(_red(f"ERROR: {e}"))
            continue

        if not changes:
            continue

        print(format_changes(changes))
        pc = await store.add(provider=name, changes=changes)
        print(
            f"  saved as {_yellow(pc.change_id)} "
            f"({_cyan('python -m pricing_agent.agent review')} to review)"
        )

        if analyze:
            print(f"  running LLM analysis...", end=" ", flush=True)
            from .analyzer import analyze_changes
            analyses = await analyze_changes(changes, model=llm_model)
            if analyses:
                print(_green(f"{len(analyses)} analysis(es)"))
                for a in analyses:
                    rec = a.get("recommendation", "?")
                    intent = a.get("intent", "?")
                    print(
                        f"    [{_yellow(rec):>8s}] intent={intent:20s} "
                        f"{a.get('reasoning', '')[:100]}"
                    )
                    if rec == "apply":
                        await store.approve(pc.change_id)
                        print(f"      → {_green('auto-approved')}")
                    elif rec == "skip":
                        await store.reject(pc.change_id, "LLM recommended skip")
                        print(f"      → {_yellow('auto-skipped')}")
            else:
                print(_yellow("no analysis returned"))

    print()


async def cmd_review():
    items = await store.pending()
    if not items:
        print(_green("No pending changes.\n"))
        return

    for pc in items:
        print(format_pending(pc))
        for i, ch in enumerate(pc.changes, 1):
            print(f"  {i}. {ch.description}")
            if ch.impact:
                print(f"     impact: {ch.impact[:120]}")
            if ch.suggested_action:
                print(f"     action: {ch.suggested_action[:120]}")
            print()

        while True:
            try:
                inp = input(
                    f"  Approve all {len(pc.changes)} change(s)? "
                    f"[{_green('y')}/{_red('n')}/v(erbose)] "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if inp == "y":
                if await store.approve(pc.change_id):
                    print(_green(f"  ✓ {pc.change_id} approved\n"))
                break
            elif inp == "n":
                reason = input("  reason (optional): ").strip()
                if await store.reject(pc.change_id, reason):
                    print(_red(f"  ✗ {pc.change_id} rejected\n"))
                break
            elif inp == "v":
                print(format_changes(pc.changes))
                continue
            else:
                print("  enter y / n / v")


async def cmd_apply():
    all_items = await store.list_all()
    items = [i for i in all_items if i.status == "approved"]
    if not items:
        print(_yellow("No approved changes to apply.\n"))
        return

    import yaml

    config_path = CONFIG_PATH
    if not os.path.exists(config_path):
        print(_red(f"config not found: {config_path}"))
        return

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    applied = 0
    for pc in items:
        print(f"  applying {pc.change_id}...", end=" ", flush=True)
        try:
            _apply_changes(config, pc.changes)
            pc.status = "applied"
            applied += 1
            print(_green("done"))
        except Exception as e:
            print(_red(f"ERROR: {e}"))

    if applied:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(
            _green(f"\n  ✓ {applied} change(s) applied to {config_path}\n")
        )
    else:
        print(_yellow("  nothing to apply\n"))


def _apply_changes(config: dict, changes: list[PriceChange]):
    model_list = config.setdefault("model_list", [])
    for ch in changes:
        if ch.change_type == "model_added":
            model_list.append({
                "model_name": ch.model_name,
                "litellm_params": {
                    "model": f"deepseek/{ch.model_name}",
                    "api_key": "os.environ/DEEPSEEK_API_KEY",
                },
            })
            continue

        entry = _find_model_entry(model_list, ch.model_name)
        if entry is None:
            continue
        lp = entry.setdefault("litellm_params", {})
        field_map = {
            "input_cost_per_token": "input_cost_per_token",
            "output_cost_per_token": "output_cost_per_token",
            "cache_read_input_token_cost": "cache_read_input_token_cost",
        }
        target = field_map.get(ch.field or "")
        if target and ch.new_value is not None:
            lp[target] = ch.new_value


def _find_model_entry(model_list: list[dict], name: str) -> Optional[dict]:
    for entry in model_list:
        if entry.get("model_name") == name:
            return entry
        lp = entry.get("litellm_params", {})
        if lp.get("model") == name:
            return entry
    return None


async def cmd_status():
    all_items = await store.list_all()
    if not all_items:
        print(_yellow("No change records.\n"))
        return

    for pc in all_items:
        status_color = {
            "pending": _yellow,
            "approved": _green,
            "rejected": _red,
            "applied": _green,
        }.get(pc.status, lambda s: s)
        print(f"  {status_color(pc.summary())}")
        for ch in pc.changes:
            print(f"    · {ch.description[:100]}")
    async_store_type = "postgres" if "PostgresStore" in type(store).__name__ else "json"
    print(f"  store: {async_store_type}  |  notifier: {notifier.summarize_channels()}\n")


async def cmd_daemon(
    interval: int = 3600,
    analyze: bool = False,
    webhook_url: Optional[str] = None,
    llm_model: Optional[str] = None,
):
    channels = notifier.summarize_channels()
    print(_cyan(f"\n=== Daemon mode — checking every {interval}s ==="))
    print(f"  store: {'postgres' if 'PostgresStore' in type(store).__name__ else 'json'}")
    print(f"  notifier: {channels}\n")
    while True:
        try:
            await cmd_check(analyze=analyze, llm_model=llm_model)
            if webhook_url:
                await run_check(webhook_url=webhook_url, llm_model=llm_model)
        except Exception as e:
            print(_red(f"check error: {e}"))
        print(f"  next check in {interval}s...\n")
        await asyncio.sleep(interval)


def cmd_serve():
    import uvicorn
    port = int(os.environ.get("PRICING_AGENT_PORT", "4001"))
    host = os.environ.get("PRICING_AGENT_HOST", "0.0.0.0")
    print(_cyan(f"\n=== Starting server on {host}:{port} ===\n"))
    uvicorn.run("pricing_agent.server:app", host=host, port=port, reload=False)


def _parse_flags(rest: list[str]) -> dict:
    kwargs: dict = {}
    it = iter(rest)
    for arg in it:
        if arg == "--analyze":
            kwargs["analyze"] = True
        elif arg == "--model" or arg == "--llm-model":
            kwargs["llm_model"] = next(it, None)
        elif arg == "--webhook" or arg == "--webhook-url":
            kwargs["webhook_url"] = next(it, None)
        elif arg == "--interval":
            try:
                kwargs["interval"] = int(next(it, "3600"))
            except ValueError:
                kwargs["interval"] = 3600
        elif not arg.startswith("--"):
            kwargs.setdefault("monitor_name", arg)
    return kwargs


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    rest = args[1:]
    flags = _parse_flags(rest)

    try:
        if cmd == "check":
            asyncio.run(cmd_check(**flags))
        elif cmd == "review":
            asyncio.run(cmd_review())
        elif cmd == "apply":
            asyncio.run(cmd_apply())
        elif cmd == "status":
            asyncio.run(cmd_status())
        elif cmd == "daemon":
            asyncio.run(cmd_daemon(**flags))
        elif cmd == "serve":
            cmd_serve()
        else:
            print(f"unknown command: {cmd}\n{__doc__}")
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":
    main()
