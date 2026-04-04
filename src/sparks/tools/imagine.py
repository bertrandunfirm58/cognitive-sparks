"""Tool #2: Imagine — mental simulation, 'what if' scenarios."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Hypothesis
from sparks.tools.base import BaseTool


class ScenarioBatch(BaseModel):
    scenarios: list[dict]  # [{hypothesis, mechanism, observable_if_true, observable_if_false, probability}]


class ImagineTool(BaseTool):
    """
    "Imaging is not mere daydreaming. It is the deliberate construction
     of a mental model that can be manipulated and tested."
     — Sparks of Genius

    Run mental movies. Imagine what WOULD happen if a principle were
    pushed to extremes, inverted, or applied to a new domain.
    """
    name = "imagine"

    def should_run(self, state: CognitiveState) -> bool:
        """Run after principles exist — imagination needs raw material."""
        return len(state.principles) >= 2

    def run(self, state: CognitiveState, **kwargs):
        if not state.principles:
            return

        principles_text = "\n".join(
            f"- [{p.confidence:.0%}] {p.statement}"
            for p in state.principles
        )

        ctx = tool_context("imagine", state)

        prompt = f"""You are a mental simulator. Your job is to IMAGINE — run 'what if' movies.

## Discovered Principles
{principles_text}

## Context
{ctx}

## Instructions
For each principle, run THREE types of mental simulation:

1. **Extreme**: Push it to its logical extreme. What happens if this principle
   operates at 10x intensity? What breaks? What new behavior emerges?

2. **Inversion**: What if the OPPOSITE were true? What would the data look like?
   Does the inversion reveal hidden assumptions?

3. **Transfer**: Apply this principle to a completely different domain.
   Does it still work? Where does it break?

For each scenario:
- hypothesis: the 'what if' statement
- mechanism: how it would play out step by step
- observable_if_true: what evidence would we see if this scenario is real
- observable_if_false: what evidence would disprove it
- probability: 0.0-1.0 how likely is this scenario

Generate 4-6 scenarios total. Prioritize the ones that are most SURPRISING
yet still plausible — these reveal the deepest structure."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("imagine"),
            schema=ScenarioBatch,
            tool="imagine",
            tracker=self.tracker,
        )

        for s in result.scenarios:
            hyp = Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                statement=s.get("hypothesis", ""),
                probability=float(s.get("probability", 0.5)),
            )
            state.hypotheses = getattr(state, 'hypotheses', [])
            state.hypotheses.append(hyp)
            self.emit("hypothesis_generated", hyp.id, state.round)
