from .models import PriceChange, PendingChange


def format_changes(changes: list[PriceChange]) -> str:
    lines = [f"total {len(changes)} change(s) detected\n"]
    for i, c in enumerate(changes, 1):
        lines.append(f"  {i}. {c.detail()}\n")
    return "".join(lines)


def format_pending(pc: PendingChange) -> str:
    header = (
        f"\n{'='*60}\n"
        f"  {pc.summary()}\n"
        f"  created: {pc.created_at}\n"
        f"{'='*60}\n"
    )
    body = format_changes(pc.changes)
    return header + body
