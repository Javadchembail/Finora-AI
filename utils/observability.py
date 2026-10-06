from __future__ import annotations

import os

import logfire


_configured = False


def configure_logfire() -> bool:
    """
    Configure Logfire once for Finora AI.

    Local development:
        Uses LOGFIRE_TOKEN if available.

    Streamlit Cloud:
        Uses LOGFIRE_TOKEN from Streamlit Secrets/environment.
    """
    global _configured

    if _configured:
        return True

    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        return False

    try:
        logfire.configure(
            token=token,
            service_name="finora-ai",
        )
        _configured = True
        return True
    except Exception as exc:
        print(f"Finora Logfire configuration warning: {exc}")
        return False


def log_event(message: str, **attributes) -> None:
    """
    Record a lightweight Finora event without logging financial data.
    """
    if not _configured:
        return

    try:
        logfire.info(message, **attributes)
    except Exception as exc:
        print(f"Finora Logfire event warning: {exc}")