"""Small helpers shared by the pipeline."""

from __future__ import annotations

import subprocess


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def weighted_mean(parts: list[tuple[float, float]]) -> float:
    """Average values by non-negative weights. Weights are normalized."""
    total = 0.0
    acc = 0.0
    for value, weight in parts:
        if weight < 0:
            raise ValueError("weights must be non-negative")
        acc += float(value) * float(weight)
        total += float(weight)
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return acc / total


def run_command(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def truncate(text: str, limit: int = 400) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
