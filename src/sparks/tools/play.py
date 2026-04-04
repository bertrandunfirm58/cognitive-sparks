"""Tool #11: Play — break the rules and see what happens."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.llm import llm_structured
from sparks.state import CognitiveState, PlayDiscovery
from sparks.tools.base import BaseTool


class PlayResultBatch(BaseModel):
    discoveries: list[dict]  # [{rule_broken, what_happened, useful, surprise_level, insight}]


class PlayTool(BaseTool):
    """
    "Play is the laboratory of the possible."
     — Sparks of Genius

    Serious play: break ONE constraint at a time.
    What if gravity didn't exist? What if time ran backwards?
    What if the dominant pattern were forbidden?

    Play ≠ random. Play = systematic rule-breaking with observation.
    """
    name = "play"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if we have principles to play with."""
        return len(state.principles) >= 2

    def run(self, state: CognitiveState, **kwargs):
        principles_text = "\n".join(
            f"{i+1}. [{p.confidence:.0%}] {p.statement}"
            for i, p in enumerate(state.principles)
        )

        patterns_text = "\n".join(
            f"- [{p.type}] {p.description}"
            for p in state.patterns[:10]
        )

        contradictions_text = "\n".join(
            f"- {c.insight_a} vs {c.insight_b}"
            for c in state.contradictions if not c.resolved
        ) or "None."

        prompt = f"""You are a PLAYER. Your job is to BREAK RULES — systematically.

## Current Principles (the "rules")
{principles_text}

## Current Patterns
{patterns_text}

## Unresolved Contradictions
{contradictions_text}

## Instructions — Systematic Play

For EACH principle, perform one act of play:

1. **REMOVE it**: Delete this principle. What happens to the data?
   Does anything IMPROVE? Does a contradiction resolve?

2. **INVERT it**: The exact opposite is true. What world does that create?
   Is that world internally consistent? (If yes, your principle is weak.)

3. **EXAGGERATE it**: Multiply its effect 10x. What absurdity emerges?
   The absurdity often points to the principle's hidden boundary.

4. **COMBINE two rules**: Take two principles and COLLIDE them.
   What new rule emerges from their intersection?

5. **FORBIDDEN move**: What's the one thing the data "obviously" can't
   support? Try it anyway. Sometimes the forbidden move is the key.

For each discovery:
- rule_broken: which rule you broke and how
- what_happened: what emerged from the breaking
- useful: true if this revealed something genuinely new, false if dead end
- surprise_level: 0.0-1.0 how unexpected was the result
- insight: the takeaway (even from failures)

Generate 4-6 play experiments. Failed experiments are valuable too."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("play"),
            schema=PlayResultBatch,
            tool="play",
            tracker=self.tracker,
        )

        for d in result.discoveries:
            discovery = PlayDiscovery(
                id=f"play_{uuid.uuid4().hex[:8]}",
                constraint_broken=d.get("rule_broken", ""),
                discovery=d.get("insight", ""),
                useful=d.get("useful", None),
            )
            state.play_discoveries = getattr(state, 'play_discoveries', [])
            state.play_discoveries.append(discovery)
            self.emit("play_discovery", discovery.id, state.round)
