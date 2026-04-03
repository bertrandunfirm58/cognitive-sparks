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
For each principle, find 1-2 STRUCTURAL analogies from other domains:
- History, science, nature, economics, art, engineering...
- The analogy must share the same MECHANISM, not just topic
- From the analogy, derive a PREDICTION about the current data

For each analogy:
- current: the principle being matched
- past_match: the analogous case from another domain
- structural_mapping: what maps to what (A→X, B→Y, therefore C→?)
- prediction: what the analogy suggests will happen / is true
- confidence: 0.0-1.0

Filter ruthlessly: if it's resemblance, not analogy, discard it."""

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
