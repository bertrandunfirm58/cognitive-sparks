"""Tool #7: Body Thinking — feel the data physically."""

from __future__ import annotations

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.data import DataStore
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Observation
from sparks.tools.base import BaseTool

import uuid


class BodyInsightBatch(BaseModel):
    insights: list[dict]  # [{sense, what_felt, why_it_matters, confidence}]


class BodyThinkTool(BaseTool):
    """
    "The body knows things the mind has not yet formulated."
     — Sparks of Genius

    Don't analyze the data — FEEL it. What's heavy? What's sharp?
    What makes you flinch? Where's the rhythm? Where's the friction?
    Proprioception for data: sensing scale, weight, texture, momentum.
    """
    name = "body_think"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if we have observations to physically engage with."""
        return len(state.observations) >= 5

    def run(self, state: CognitiveState, data: DataStore = None, **kwargs):
        # Sample some raw data and observations
        obs_sample = "\n".join(
            f"- [{o.channel}] {o.content}"
            for o in state.observations[:30]
        )

        data_sample = ""
        if data:
            samples = data.sample(ratio=0.15, min_n=2, max_n=4)
            data_sample = "\n\n".join(
                f"[{s['file']}]: {s['content'][:3000]}" for s in samples
            )

        prompt = f"""You are a BODY THINKER. Don't analyze — FEEL the data.

## Raw Data Sample
{data_sample if data_sample else "(use observations below)"}

## Observations So Far
{obs_sample}

## Instructions — Engage Your Senses

Imagine holding this data in your hands. Feel it.

1. **Weight**: What's HEAVY in this data? What pulls you down?
   What topics/themes have gravitational mass?

2. **Texture**: What's rough vs smooth? Where does the data flow
   easily, and where does it catch and snag?

3. **Temperature**: What's HOT (active, volatile, energetic)?
   What's COLD (static, stable, dormant)?

4. **Rhythm**: What beats? What cycles? Tap the tempo of the data.
   Where does the rhythm break?

5. **Tension**: Where does it PULL? Where are opposing forces creating
   strain? Where is something about to snap?

6. **Scale**: What's huge vs tiny? What feels outsized for its context?
   What's surprisingly small?

For each body insight:
- sense: which sense (weight/texture/temperature/rhythm/tension/scale)
- what_felt: the physical sensation — describe it viscerally
- why_it_matters: what this bodily sense reveals about the data's structure
- confidence: 0.0-1.0

The mundane mind skips what the body catches. Trust the flinch."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("body_think"),
            schema=BodyInsightBatch,
            tool="body_think",
            tracker=self.tracker,
        )

        for insight in result.insights:
            obs = Observation(
                id=f"body_{uuid.uuid4().hex[:8]}",
                channel=f"body_{insight.get('sense', 'general')}",
                content=f"{insight.get('what_felt', '')} → {insight.get('why_it_matters', '')}",
                lens_used="body_thinking",
                confidence=float(insight.get("confidence", 0.6)),
            )
            state.observations.append(obs)
            self.emit("body_insight", obs.id, state.round)
