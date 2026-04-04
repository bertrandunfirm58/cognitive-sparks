"""Tool #8: Empathize — become the actors inside the data."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, PerspectiveInsight
from sparks.tools.base import BaseTool


class PerspectiveBatch(BaseModel):
    perspectives: list[dict]  # [{actor, sees, feels, would_do, reveals, confidence}]


class EmpathizeTool(BaseTool):
    """
    "To understand is to stand under — to see from inside."
     — Sparks of Genius

    Don't look AT the data — look FROM INSIDE it.
    Become the actors, the variables, the constraints.
    What does the world look like from their position?
    """
    name = "empathize"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if patterns exist to empathize with."""
        return len(state.patterns) >= 3

    def run(self, state: CognitiveState, **kwargs):
        patterns_text = "\n".join(
            f"- [{p.type}] {p.description}"
            for p in state.patterns[:15]
        )

        principles_text = "\n".join(
            f"- {p.statement}"
            for p in state.principles[:5]
        ) or "None yet."

        prompt = f"""You are an EMPATHIZER. Your job is to BECOME the actors in the data.

## Patterns Found
{patterns_text}

## Principles So Far
{principles_text}

## Goal: {state.goal}

## Instructions

Identify 3-5 key ACTORS, AGENTS, or FORCES in this data. Then become each one.

For each actor:
1. **Who are you?** Name the actor/agent/force
2. **What do you see?** Describe the world from their perspective
3. **What do you feel?** Their pressures, fears, goals, constraints
4. **What would you do next?** Given their position, what's rational?
5. **What does this reveal?** What insight is invisible from the outside
   but obvious from the inside?

Examples of actors to empathize with:
- A data point at the extreme (what does it feel like to be an outlier?)
- A pattern that's dying (what killed you? what's replacing you?)
- The absence (what would exist if you were present?)
- The constraint (what are you holding back? what breaks if you release?)

For each perspective:
- actor: who you became
- sees: what the world looks like from inside
- feels: pressures, goals, constraints
- would_do: rational next action from this perspective
- reveals: insight invisible from outside
- confidence: 0.0-1.0"""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("empathize"),
            schema=PerspectiveBatch,
            tool="empathize",
            tracker=self.tracker,
        )

        for p_data in result.perspectives:
            insight = PerspectiveInsight(
                id=f"emp_{uuid.uuid4().hex[:8]}",
                perspective=p_data.get("actor", "unknown"),
                interpretation=p_data.get("reveals", ""),
                differs_from_default=p_data.get("sees", ""),
            )
            state.perspective_insights = getattr(state, 'perspective_insights', [])
            state.perspective_insights.append(insight)
            self.emit("empathy_insight", insight.id, state.round)
