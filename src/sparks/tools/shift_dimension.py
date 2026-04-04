"""Tool #9: Dimensional Thinking — change the axis, change the truth."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Observation
from sparks.tools.base import BaseTool


class DimensionShiftBatch(BaseModel):
    shifts: list[dict]  # [{dimension_from, dimension_to, what_changed, new_pattern, confidence}]


class ShiftDimensionTool(BaseTool):
    """
    "The map is not the territory — change the projection,
     and hidden continents appear." — after Korzybski

    Same data, different axis. Time→frequency. Linear→log.
    Micro→macro. Static→dynamic. Each shift reveals what
    the default view hides.
    """
    name = "shift_dimension"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if observations and patterns exist."""
        return len(state.observations) >= 5 and len(state.patterns) >= 2

    def run(self, state: CognitiveState, **kwargs):
        obs_text = "\n".join(
            f"- [{o.channel}] {o.content}"
            for o in state.observations[:25]
        )

        patterns_text = "\n".join(
            f"- [{p.type}] {p.description}"
            for p in state.patterns[:10]
        )

        prompt = f"""You are a DIMENSIONAL THINKER. Same data, different axis.

## Current Observations (default view)
{obs_text}

## Current Patterns
{patterns_text}

## Goal: {state.goal}

## Instructions — Shift the Axis

The current observations are from ONE perspective. Now ROTATE:

1. **Time shift**: If you compress time 10x, what rhythm appears?
   If you expand time 100x, what trend is invisible at normal speed?

2. **Scale shift**: Zoom OUT — what emerges at the macro level that's
   invisible at the micro? Zoom IN — what detail is lost in the aggregate?

3. **Inversion shift**: Flip figure and ground. What's currently background
   becomes foreground. The ABSENCE becomes the subject.

4. **Frequency shift**: Convert events to frequencies. What oscillates?
   What has a period? What's aperiodic (and therefore signal)?

5. **Network shift**: Instead of items, see CONNECTIONS. Who links to whom?
   Where are the bridges? Where are the islands?

6. **Entropy shift**: Where is disorder increasing? Where is order emerging?
   Information theory: where is surprise concentrated?

For each shift:
- dimension_from: the current viewing axis
- dimension_to: the new axis after shift
- what_changed: what became visible/invisible after the shift
- new_pattern: the pattern that only appears in this dimension
- confidence: 0.0-1.0

Perform 4-6 shifts. The best shifts make you say "how did I miss that?" """

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("shift_dimension"),
            schema=DimensionShiftBatch,
            tool="shift_dimension",
            tracker=self.tracker,
        )

        for s in result.shifts:
            obs = Observation(
                id=f"dim_{uuid.uuid4().hex[:8]}",
                channel=f"dimension_{s.get('dimension_to', 'shifted')}",
                content=f"[{s.get('dimension_from','?')}→{s.get('dimension_to','?')}] {s.get('new_pattern', '')}",
                lens_used="dimensional_thinking",
                confidence=float(s.get("confidence", 0.6)),
            )
            state.observations.append(obs)
            self.emit("dimension_shifted", obs.id, state.round)
