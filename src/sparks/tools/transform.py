"""Tool #12: Transform — convert one form into another to reveal hidden structure."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.llm import llm_structured
from sparks.state import CognitiveState, Observation, Pattern
from sparks.tools.base import BaseTool


class TransformBatch(BaseModel):
    transformations: list[dict]  # [{from_form, to_form, method, result, revealed, confidence}]


class TransformTool(BaseTool):
    """
    "Translation between forms — words to images, images to models,
     models to equations — is itself a creative act that reveals
     structure invisible in the original form."
     — Sparks of Genius

    The structure that hides in text appears in a table.
    The pattern invisible in a table appears in a graph.
    The rhythm absent from a graph appears in music notation.
    Every transformation is a new lens.
    """
    name = "transform"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if patterns and principles exist to transform."""
        return len(state.patterns) >= 3 and len(state.principles) >= 1

    def run(self, state: CognitiveState, **kwargs):
        principles_text = "\n".join(
            f"- {p.statement}" for p in state.principles
        )

        patterns_text = "\n".join(
            f"- [{p.type}] {p.description} (conf: {p.confidence:.0%})"
            for p in state.patterns[:15]
        )

        obs_text = "\n".join(
            f"- [{o.channel}] {o.content}"
            for o in state.observations[:15]
        )

        prompt = f"""You are a TRANSFORMER. Convert information between forms to reveal hidden structure.

## Current Principles
{principles_text}

## Current Patterns
{patterns_text}

## Sample Observations
{obs_text}

## Instructions — Transform Between Representations

Each representation reveals different structure. Perform these transformations:

1. **Narrative → Equation**: Express a principle as a mathematical relationship.
   Even a rough formula reveals hidden variables and dependencies.
   Example: "More X leads to less Y" → Y = k/X (inverse relationship)

2. **Static → Dynamic**: Take a snapshot pattern and describe its MOVIE.
   How did it get here? Where is it going? What's the trajectory?

3. **Parts → Whole**: Take individual patterns and describe the SYSTEM
   they form together. Draw the system diagram in words.

4. **Positive → Negative**: Describe everything in terms of what's NOT there.
   The photographic negative often reveals the hidden subject.

5. **Concrete → Metaphor**: Express a principle as a vivid metaphor.
   Good metaphors compress enormous complexity into graspable form.

6. **Hierarchy → Network**: If the data looks hierarchical, redraw it
   as a flat network. If it looks like a network, impose hierarchy.

For each transformation:
- from_form: original representation
- to_form: new representation
- method: how you transformed it
- result: the transformed content
- revealed: what became visible ONLY after transformation
- confidence: 0.0-1.0

Generate 4-6 transformations. The best ones make invisible structure visible."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("transform"),
            schema=TransformBatch,
            tool="transform",
            tracker=self.tracker,
        )

        for t in result.transformations:
            revealed = t.get("revealed", "")
            if revealed:
                # Add as new observation from transformation lens
                obs = Observation(
                    id=f"tfm_{uuid.uuid4().hex[:8]}",
                    channel=f"transform_{t.get('to_form', 'new')}",
                    content=f"[{t.get('from_form','?')}→{t.get('to_form','?')}] {revealed}",
                    lens_used="transformation",
                    confidence=float(t.get("confidence", 0.6)),
                )
                state.observations.append(obs)
                self.emit("transformation_revealed", obs.id, state.round)
