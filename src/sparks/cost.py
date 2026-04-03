"""Cost tracking and model routing."""

from __future__ import annotations

from pydantic import BaseModel

# $/1M tokens (input, output)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
}

# Tool → model mapping
# Set SPARKS_ALL_OPUS=1 to force Opus for all tools (ablation test)
import os as _os
_OPUS = "claude-opus-4-20250514"
_SONNET = "claude-sonnet-4-20250514"
_HAIKU = "claude-haiku-4-5-20251001"
_ALL_OPUS = _os.environ.get("SPARKS_ALL_OPUS", "") == "1"

MODEL_ROUTING: dict[str, str] = {
    "observe": _OPUS if _ALL_OPUS else _HAIKU,
    "recognize_patterns": _OPUS if _ALL_OPUS else _HAIKU,
    "form_patterns": _OPUS if _ALL_OPUS else _HAIKU,
    "abstract": _OPUS if _ALL_OPUS else _SONNET,
    "analogize": _OPUS if _ALL_OPUS else _SONNET,
    "model": _OPUS if _ALL_OPUS else _HAIKU,
    "synthesize": _OPUS if _ALL_OPUS else _SONNET,
    # Infrastructure
    "lens_generate": _OPUS if _ALL_OPUS else _SONNET,
    "quick_scan": _OPUS if _ALL_OPUS else _HAIKU,
    "convergence": _OPUS if _ALL_OPUS else _SONNET,
}

TOOL_PRIORITY: dict[str, int] = {
    "observe": 1,
    "recognize_patterns": 1,
    "abstract": 1,
    "synthesize": 1,
    "form_patterns": 2,
    "analogize": 2,
    "model": 2,
    "imagine": 3,
    "empathize": 3,
    "shift_dimension": 3,
    "play": 3,
    "transform": 3,
}


class DepthBudget(BaseModel):
    max_cost: float
    max_rounds: int
    max_priority: int  # tools with priority <= this are active

DEPTH_BUDGETS: dict[str, DepthBudget] = {
    "quick": DepthBudget(max_cost=0.50, max_rounds=1, max_priority=1),
    "standard": DepthBudget(max_cost=5.00, max_rounds=3, max_priority=2),
    "deep": DepthBudget(max_cost=20.00, max_rounds=5, max_priority=3),
}


class CallRecord(BaseModel):
    tool: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostTracker:
    def __init__(self, budget: DepthBudget):
        self.budget = budget
        self.total_cost = 0.0
        self.total_calls = 0
        self.breakdown: dict[str, float] = {}
        self.records: list[CallRecord] = []

    def record(self, tool: str, model: str, input_tokens: int, output_tokens: int):
        costs = MODEL_COSTS.get(model, (3.0, 15.0))
        cost = (input_tokens / 1_000_000 * costs[0]) + (output_tokens / 1_000_000 * costs[1])
        self.total_cost += cost
        self.total_calls += 1
        self.breakdown[tool] = self.breakdown.get(tool, 0) + cost
        self.records.append(CallRecord(
            tool=tool, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens, cost=cost,
        ))

    def remaining(self) -> float:
        return self.budget.max_cost - self.total_cost

    def can_afford(self, model: str, estimated_tokens: int = 10000) -> bool:
        costs = MODEL_COSTS.get(model, (3.0, 15.0))
        est = estimated_tokens / 1_000_000 * (costs[0] + costs[1])
        return self.remaining() >= est

    def select_model(self, tool: str) -> str:
        preferred = MODEL_ROUTING.get(tool, "claude-sonnet-4-20250514")
        if not self.can_afford(preferred):
            return "claude-haiku-4-5-20251001"
        return preferred


def get_active_tools(depth: str) -> list[str]:
    budget = DEPTH_BUDGETS[depth]
    return [t for t, p in TOOL_PRIORITY.items() if p <= budget.max_priority]
