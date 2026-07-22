"""
Multi-channel notification for pricing changes.

Supported channels (configured via env vars):
  DINGTALK_WEBHOOK_URL  — 钉钉群机器人
  SLACK_WEBHOOK_URL     — Slack Incoming Webhook
  SMTP_*                — 邮件 (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
                          NOTIFY_EMAIL_FROM, NOTIFY_EMAIL_TO)

All channels are optional — only those with environment variables are used.
"""

import json
import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from typing import Optional
from .models import PendingChange


class Notifier(ABC):
    @abstractmethod
    async def send_change_notification(self, pc: PendingChange, detail_url: str = "") -> bool:
        ...


# ── 钉钉 ──────────────────────────────────────────────────────────


class DingTalkNotifier(Notifier):
    def __init__(self, webhook_url: str):
        self._url = webhook_url

    async def send_change_notification(self, pc: PendingChange, detail_url: str = "") -> bool:
        title = f"定价变更提醒 - {pc.provider}"
        lines = [f"### {title}", f"**变更 ID**: {pc.change_id}", ""]
        for ch in pc.changes:
            lines.append(f"- {ch.description}")
            if ch.impact:
                lines.append(f"  > {ch.impact}")
        if detail_url:
            lines.append(f"\n[查看详情]({detail_url})")

        payload = {
            "msgtype": "markdown",
            "markdown": {"title": title, "text": "\n".join(lines)},
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as cli:
                resp = await cli.post(self._url, json=payload)
            return resp.status_code == 200
        except Exception:
            return False


# ── Slack ─────────────────────────────────────────────────────────


class SlackNotifier(Notifier):
    def __init__(self, webhook_url: str):
        self._url = webhook_url

    async def send_change_notification(self, pc: PendingChange, detail_url: str = "") -> bool:
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text",
                         "text": f"💰 定价变更 — {pc.provider}"},
            },
            {"type": "context", "elements": [
                {"type": "mrkdwn", "text": f"`{pc.change_id}` | {pc.created_at}"},
            ]},
            {"type": "divider"},
        ]
        for ch in pc.changes:
            text = ch.description
            if ch.impact:
                text += f"\n• *影响*: {ch.impact}"
            if ch.suggested_action:
                text += f"\n• *建议*: {ch.suggested_action}"
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

        if detail_url:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"<{detail_url}|查看详情>"},
            })

        payload = {"blocks": blocks}
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as cli:
                resp = await cli.post(self._url, json=payload)
            return resp.status_code == 200
        except Exception:
            return False


# ── 邮件 ──────────────────────────────────────────────────────────


class EmailNotifier(Notifier):
    def __init__(self):
        self._host = os.environ.get("SMTP_HOST", "")
        self._port = int(os.environ.get("SMTP_PORT", "587"))
        self._user = os.environ.get("SMTP_USER", "")
        self._pass = os.environ.get("SMTP_PASS", "")
        self._from = os.environ.get("NOTIFY_EMAIL_FROM", self._user)
        self._to_raw = os.environ.get("NOTIFY_EMAIL_TO", "")
        self._to = [addr.strip() for addr in self._to_raw.split(",") if addr.strip()]

    async def send_change_notification(self, pc: PendingChange, detail_url: str = "") -> bool:
        if not self._host or not self._to:
            return False
        body = f"定价变更报告 — {pc.provider}\n{'=' * 40}\n\n"
        body += f"变更 ID: {pc.change_id}\n"
        body += f"时间: {pc.created_at}\n\n"
        for ch in pc.changes:
            body += f"- {ch.description}\n"
            if ch.impact:
                body += f"  影响: {ch.impact}\n"
            if ch.suggested_action:
                body += f"  建议: {ch.suggested_action}\n"
        body += f"\n详情: {detail_url}\n" if detail_url else ""
        body += f"\n---\n请通过 {detail_url or 'CLI review'} 审批或拒绝。\n"

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = f"[Pricing Agent] {pc.provider} 定价变更"
        msg["From"] = self._from
        msg["To"] = ", ".join(self._to)

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._send_sync, msg)
            return True
        except Exception:
            return False

    def _send_sync(self, msg: MIMEText):
        use_tls = self._port == 587
        if use_tls:
            with smtplib.SMTP(self._host, self._port) as s:
                s.starttls()
                if self._user:
                    s.login(self._user, self._pass)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(self._host, self._port) as s:
                if self._user:
                    s.login(self._user, self._pass)
                s.send_message(msg)


# ── composite notifier ────────────────────────────────────────────


class CompositeNotifier(Notifier):
    """Runs all configured channel notifiers."""

    def __init__(self):
        self._channels: list[Notifier] = []
        dingtalk = os.environ.get("DINGTALK_WEBHOOK_URL")
        slack = os.environ.get("SLACK_WEBHOOK_URL")
        if dingtalk:
            self._channels.append(DingTalkNotifier(dingtalk))
        if slack:
            self._channels.append(SlackNotifier(slack))
        if os.environ.get("SMTP_HOST") and os.environ.get("NOTIFY_EMAIL_TO"):
            self._channels.append(EmailNotifier())

    @property
    def enabled(self) -> bool:
        return len(self._channels) > 0

    async def send_change_notification(self, pc: PendingChange, detail_url: str = "") -> bool:
        if not self._channels:
            return False
        results = []
        for ch in self._channels:
            try:
                ok = await ch.send_change_notification(pc, detail_url)
                results.append(ok)
            except Exception:
                results.append(False)
        return any(results)

    def summarize_channels(self) -> str:
        names = []
        for ch in self._channels:
            if isinstance(ch, DingTalkNotifier):
                names.append("dingtalk")
            elif isinstance(ch, SlackNotifier):
                names.append("slack")
            elif isinstance(ch, EmailNotifier):
                names.append("email")
        return ", ".join(names) if names else "none"
