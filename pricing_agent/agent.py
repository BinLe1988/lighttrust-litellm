#!/usr/bin/env python3
"""
Pricing Intelligence Agent — CLI tool for litellm.

Usage:
  python -m pricing_agent.agent check          fetch & diff vendor prices
  python -m pricing_agent.agent review          review pending changes
  python -m pricing_agent.agent apply           apply approved changes
  python -m pricing_agent.agent status          show current state
"""

import asyncio
import sys
from typing import Optional

from .models import PendingChange, PriceChange
from .monitors.registry import get_monitor, list_monitors
from .differ import diff_snapshot, CONFIG_PATH
from .store import PendingChangeStore
from .reporter import format_changes, format_pending


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _cyan(s: str) -> str:
    return f"\033[96m{s}\033[0m"


store = PendingChangeStore()


async def cmd_check(monitor_name: Optional[str] = None):
    print(_cyan("\n=== Pricing Intelligence: check ==="))

    monitors = [monitor_name] if monitor_name else list_monitors()
    all_changes: list[PriceChange] = []

    for name in monitors:
        print(f"\n  fetching {name} pricing...", end=" ", flush=True)
        try:
            mon = get_monitor(name)
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

        if changes:
            print(format_changes(changes))
            pc = store.add(provider=name, changes=changes)
            print(
                f"  saved as {_yellow(pc.change_id)} "
                f"({_cyan('python -m pricing_agent.agent review')} to review)"
            )
            all_changes.extend(changes)

    if not all_changes:
        print(_green("\n  No changes detected — everything is up to date.\n"))
    else:
        print(
            f"\n  {_yellow(f'{len(all_changes)} change(s) pending review.')}\n"
        )


def cmd_review():
    items = store.pending()
    if not items:
        print(_green("No pending changes.\n"))
        return

    for pc in items:
        print(format_pending(pc))
        for i, ch in enumerate(pc.changes, 1):
            print(f"  {i}. {ch.description}")
            if ch.impact:
                print(f"     impact: {ch.impact[:120]}...")
            if ch.suggested_action:
                print(f"     action: {ch.suggested_action[:120]}...")
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
                if store.approve(pc.change_id):
                    print(_green(f"  ✓ {pc.change_id} approved\n"))
                break
            elif inp == "n":
                reason = input("  reason (optional): ").strip()
                if store.reject(pc.change_id, reason):
                    print(_red(f"  ✗ {pc.change_id} rejected\n"))
                break
            elif inp == "v":
                print(format_changes(pc.changes))
                continue
            else:
                print("  enter y / n / v")


def cmd_apply():
    items = [i for i in store.list_all() if i.status == "approved"]
    if not items:
        print(_yellow("No approved changes to apply.\n"))
        return

    import os
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
        store._save()
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


def cmd_status():
    all_items = store.list_all()
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
    print()


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return

    cmd = args[0]
    rest = args[1:]

    try:
        if cmd == "check":
            asyncio.run(cmd_check(rest[0] if rest else None))
        elif cmd == "review":
            cmd_review()
        elif cmd == "apply":
            cmd_apply()
        elif cmd == "status":
            cmd_status()
        else:
            print(f"unknown command: {cmd}\n{__doc__}")
    except KeyboardInterrupt:
        print("\ninterrupted")


if __name__ == "__main__":
    main()
