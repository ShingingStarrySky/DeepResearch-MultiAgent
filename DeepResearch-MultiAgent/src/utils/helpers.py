from __future__ import annotations

from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel


def format_time(dt: Optional[datetime] = None) -> str:
    dt = dt or datetime.now()
    return dt.strftime("%H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def format_token_report(token_data: dict) -> str:
    console = Console(record=True, width=80)

    table = Table(title="Token 消耗报告", show_header=True, header_style="bold cyan")
    table.add_column("项目", style="dim", width=30)
    table.add_column("数值", justify="right")

    table.add_row("会话总Token", f"{token_data.get('total_tokens', 0):,}")
    table.add_row("总调用次数", f"{token_data.get('total_calls', 0):,}")
    table.add_row("Prompt Tokens", f"{token_data.get('prompt_tokens', 0):,}")
    table.add_row("Completion Tokens", f"{token_data.get('completion_tokens', 0):,}")
    table.add_row("", "")

    by_model = token_data.get("by_model", {})
    for model, usage in by_model.items():
        table.add_row(f"  [{model}]", f"{usage.get('total', 0):,} tokens ({usage.get('calls', 0)} calls)")

    table.add_row("", "")
    by_agent = token_data.get("by_agent", {})
    for agent, usage in by_agent.items():
        table.add_row(f"  [{agent}]", f"{usage.get('total', 0):,} tokens")

    console.print(table)
    return console.export_text()


def format_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}m"
    elif n >= 1_000:
        return f"{n:,}"
    return str(n)
