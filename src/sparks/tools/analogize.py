"""Tool #6: Analogize — find structural correspondence, not surface similarity."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Analogy
from sparks.tools.base import BaseTool


class AnalogyBatch(BaseModel):
    analogies: list[dict]  # [{current, past_match, structural_mapping, prediction, confidence}]


class AnalogizerTool(BaseTool):
    """
    "Analogy ≠ resemblance. Analogy discovers functional correspondence
     between internal relationships." — Sparks of Genius

    Baseball ↔ orange: resemblance (both round) ❌
    Baseball ↔ sun: analogy (both arc through the sky) ✅
    """
    name = "analogize"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if principles exist to analogize from."""
        return len(state.principles) >= 1

    def run(self, state: CognitiveState, **kwargs):
        if not state.principles:
            return

        principles_text = "\n".join(
            f"- [{p.confidence:.0%}] {p.statement}"
            for p in state.principles
        )

        ctx = tool_context("analogize", state)

        prompt = f"""You are a structural analogy engine.

## CRITICAL DISTINCTION
- RESEMBLANCE: "Both are about technology" → surface, useless
- ANALOGY: "Both exhibit the same structural mechanism" → deep, powerful

Like a baseball and the sun — both arc through the sky, rise and fall.
NOT like a baseball and an orange — both are round (mere resemblance).

## Discovered Principles
{principles_text}

## Context
{ctx}

## Task
You MUST produce at least 2 structural analogies per principle. This is not optional.
If you have 5 principles, produce at minimum 10 analogies.

For each principle, find 2-3 STRUCTURAL analogies from DIFFERENT domains:
- Physics (phase transitions, critical states, resonance, entropy)
- Biology (homeostasis, evolution, predator-prey dynamics, immune response)
- Information theory (signal/noise, compression, channel capacity)
- History (empire cycles, technology adoption curves, social contagion)
- Engineering (feedback loops, governor mechanisms, load balancing)

The analogy must share the same MECHANISM, not just topic.

### ANALOGY VALIDATION
For each analogy, answer these two questions:
1. Does the analogous system make a prediction about the current data that is NOT already stated in the principle? If no, the analogy is decorative — discard it.
2. Can you identify where the analogy BREAKS DOWN? Every analogy has limits. Stating the breakdown boundary makes the analogy more useful, not less.

For each analogy:
- current: the principle being matched
- past_match: the analogous case from another domain (be specific — name the phenomenon)
- structural_mapping: what maps to what (A→X, B→Y, therefore C→?)
- prediction: a NEW prediction the analogy generates about the current data
- confidence: 0.0-1.0

Filter ruthlessly: if it's resemblance, not analogy, discard it.
But do NOT filter so aggressively that you return zero — that means the tool failed."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("analogize"),
            schema=AnalogyBatch,
            tool="analogize",
            tracker=self.tracker,
        )

        for a_data in result.analogies:
            mapping = a_data.get("structural_mapping", "")
            if isinstance(mapping, dict):
                mapping = "; ".join(f"{k} → {v}" for k, v in mapping.items())
            analogy = Analogy(
                id=f"ana_{uuid.uuid4().hex[:8]}",
                current=str(a_data.get("current", "")),
                past_match=str(a_data.get("past_match", "")),
                structural_mapping=str(mapping),
                prediction=str(a_data.get("prediction", "")),
                confidence=float(a_data.get("confidence", 0.5)),
            )
            state.analogies.append(analogy)
            self.emit("analogy_found", analogy.id, state.round)
