"""Tool #1: Observe — see with all senses, guided by the lens."""

from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.data import DataStore
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Observation
from sparks.tools.base import BaseTool


class ObservationBatch(BaseModel):
    observations: list[dict]  # [{channel, content, confidence}]


class ObserveTool(BaseTool):
    name = "observe"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if no observations yet, or if anomaly detected (re-observe)."""
        if len(state.observations) == 0:
            return True
        if state.signals.anomaly_potential.fired:
            return True  # model failed → re-observe
        return False

    def run(self, state: CognitiveState, data: Optional[DataStore] = None, **kwargs):
        if not data or not state.lens:
            return

        chunks = data.chunks(max_chars=200000)

        # Get adaptive hint if available
        tool_hint = ""
        if hasattr(state, '_tool_hints') and "observe" in state._tool_hints:
            tool_hint = state._tool_hints["observe"].get("hint", "")

        # Predictive coding: in Phase 2+, tell observer what was found before
        prediction_hint = ""
        if hasattr(state, '_predictions') and state._predictions:
            prediction_hint = "\n## Previous Round Findings (look for SURPRISES vs these)\n" + \
                "\n".join(f"- {p}" for p in state._predictions[:5]) + \
                "\n\nFocus on what CONTRADICTS or is MISSING from the above. Surprises > confirmations."

        for i, chunk in enumerate(chunks):
            prompt = f"""You are an expert observer. Your job is to OBSERVE, not interpret.
{f'{chr(10)}## Adaptive Guidance{chr(10)}{tool_hint}' if tool_hint else ''}

## Your Lens
Domain: {state.lens.domain}
{state.lens.domain_description}

## Observation Channels (observe through each)
{chr(10).join(f'- **{ch.name}** (priority {ch.priority}): {ch.description}' for ch in state.lens.channels)}

## Focus Questions
{chr(10).join(f'- {q}' for q in state.lens.focus_questions)}

## Anomaly Criteria (flag if seen)
{chr(10).join(f'- {a}' for a in state.lens.anomaly_criteria)}

## Absence Criteria (flag if NOT seen)
{chr(10).join(f'- {a}' for a in state.lens.absence_criteria)}

## Data (chunk {i+1}/{len(chunks)})
{chunk}

{prediction_hint}

## Instructions
Observe through EACH channel. Record what you see — facts, not interpretations.
For each observation:
- channel: which observation channel
- content: what you observed (factual)
- confidence: 0.0-1.0 how clear/certain this observation is

Also check: is anything ABSENT that should be there? (absence = strongest signal)
No threshold filters. Everything matters. The mundane can be significant."""

            result = llm_structured(
                prompt,
                model=self.tracker.select_model("observe"),
                schema=ObservationBatch,
                tool="observe",
                tracker=self.tracker,
                max_tokens=16000,
            )

            for obs_data in result.observations:
                content = obs_data.get("content", "")
                # Provenance: find source file:line for this observation
                source_refs = data.find_source(content[:80]) if hasattr(data, 'find_source') else []
                obs = Observation(
                    id=f"obs_{uuid.uuid4().hex[:8]}",
                    channel=obs_data.get("channel", "general"),
                    content=content,
                    lens_used=state.lens.domain,
                    confidence=obs_data.get("confidence", 0.5),
                    source_refs=source_refs,
                )
                state.observations.append(obs)
                self.emit("observation_added", obs.id, state.round)
