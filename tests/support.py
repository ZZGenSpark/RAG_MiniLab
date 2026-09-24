"""Shared paths for tests that still use the six-section expense fixture."""

from __future__ import annotations

from pathlib import Path

EXPENSE_POLICY_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "expense-policy.md"
