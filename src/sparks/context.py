"""Context assembly — build the right context for each tool."""

from __future__ import annotations

from sparks.state import CognitiveState


# What each tool needs as full data vs. summary
TOOL_DATA_NEEDS: dict[str, dict[str, list[str]]] = {
    "observe": {"full": ["lens"], "summary": ["patterns", "principles"]},
    "imagine": {"full": ["principles"], "summary": ["patterns", "contradictions"]},
    "recognize_patterns": {"full": ["observations", "patterns"], "summary": ["principles"]},
    "form_patterns": {"full": ["patterns"], "summary": ["observations"]},
    "abstract": {"full": ["patterns"], "summary": ["principles", "model_results"]},
    "analogize": {"full": ["principles"], "summary": ["patterns"]},
    "body_think": {"full": ["observations"], "summary": ["patterns"]},
    "empathize": {"full": ["patterns"], "summary": ["principles", "observations"]},
    "shift_dimension": {"full": ["observations", "patterns"], "summary": ["principles"]},
    "model": {"full": ["principles", "analogies"], "summary": ["patterns"]},
    "play": {"full": ["principles", "contradictions"], "summary": ["patterns"]},
    "transform": {"full": ["principles", "patterns"], "summary": ["observations"]},
    "synthesize": {"full": ["principles", "contradictions", "analogies", "model_results"], "summary": ["observations", "patterns"]},
}


def state_summary(state: CognitiveState) -> str:
    """Compact summary of entire state — shared across all tools."""
    principles_brief = "\n".join(
        f"  - [{p.confidence:.0%}] {p.statement}"
        for p in sorted(state.principles, key=lambda x: -x.confidence)[:5]
    ) or "  None yet."

    contradictions_brief = "\n".join(
        f"  - {c.insight_a} vs {c.insight_b}"
        for c in state.contradictions if not c.resolved
    ) or "  None."

    return f"""## Cognitive State — Round {state.round} ({state.phase.value})

### Signals
- convergence: {state.signals.convergence}
- contradiction: {state.signals.contradiction}
- diminishing_returns: {state.signals.diminishing_returns}
- anomaly: {state.signals.anomaly}

### Progress
- Observations: {len(state.observations)}
- Patterns: {len(state.patterns)} (recurring: {sum(1 for p in state.patterns if p.type == 'recurring')}, absent: {sum(1 for p in state.patterns if p.type == 'absent')})
- Principles: {len(state.principles)}
- Contradictions: {len(state.contradictions)} ({sum(1 for c in state.contradictions if not c.resolved)} unresolved)

### Current Principles
{principles_brief}

### Active Contradictions
{contradictions_brief}
"""


def full_view(state: CognitiveState, field: str, max_items: int = 30) -> str:
    """Full view of a specific state field."""
    items = getattr(state, field, [])
    if not items:
        return f"## {field}\nNo data yet."

    if isinstance(items, list) and items:
        parts = []
        for item in items[:max_items]:
            if hasattr(item, "model_dump"):
                d = item.model_dump(exclude={"timestamp", "evidence"})
                parts.append(str(d))
            else:
                parts.append(str(item))
        result = "\n".join(parts)
        if len(items) > max_items:
            result += f"\n... and {len(items) - max_items} more"
        return f"## {field}\n{result}"

    return f"## {field}\n{items}"


def tool_context(tool_name: str, state: CognitiveState) -> str:
    """Assemble the right context for a specific tool."""
    needs = TOOL_DATA_NEEDS.get(tool_name, {"full": [], "summary": []})

    parts = [state_summary(state)]

    for field in needs.get("full", []):
        if field == "lens" and state.lens:
            parts.append(f"## Lens\n{state.lens.model_dump_json(indent=2)}")
        else:
            parts.append(full_view(state, field))

    for field in needs.get("summary", []):
        items = getattr(state, field, [])
        count = len(items) if isinstance(items, list) else 0
        parts.append(f"## {field} (summary): {count} items")

    return "\n\n---\n\n".join(parts)
