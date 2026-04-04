"""Tool #4 & #5: Recognize Patterns + Form Patterns."""

from __future__ import annotations

import re
import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Contradiction, Pattern, Phase
from sparks.tools.base import BaseTool


class PatternBatch(BaseModel):
    patterns: list[dict]  # [{type, description, confidence, evidence_refs}]


class RecognizePatternsTool(BaseTool):
    name = "recognize_patterns"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if observations exist and patterns need updating."""
        if len(state.observations) <= 3:
            return False
        # Phase 1: run if no patterns yet. Phase 2+: always re-run on new observations.
        return len(state.patterns) == 0 or state.phase != Phase.SEQUENTIAL

    def run(self, state: CognitiveState, **kwargs):
        if not state.observations:
            return

        obs_text = "\n".join(
            f"[{o.channel}] {o.content}" for o in state.observations[:60]
        )

        prompt = f"""You are a pattern recognition expert. Analyze these observations and find patterns.

## Context
{tool_context("recognize_patterns", state)}

## Observations
{obs_text}

## Find THREE types of patterns:

### 1. Recurring Patterns (type: "recurring")
What structures, themes, or relationships appear repeatedly across different observations?

### 2. Absent Patterns (type: "absent")
What SHOULD be present but is conspicuously missing? What's expected but not observed?
These are often the strongest signals.

### 3. Interference Patterns (type: "interference")
Where do two patterns overlap or conflict? What emerges from their intersection?

For each pattern:
- type: "recurring" | "absent" | "interference"
- description: clear description of the pattern
- confidence: 0.0-1.0
- evidence_refs: list of observation channels that support this

## MINIMUM REQUIREMENTS
- At least 8 patterns total
- At least 2 MUST be type "absent"
- At least 2 MUST be type "interference" (contradictions)

If you find zero contradictions, your analysis is almost certainly biased. Any real-world dataset spanning multiple months WILL contain contradictory signals. Zero contradictions = confirmation bias.

### MANDATORY CONTRADICTION-SEEKING PASS
After identifying recurring patterns, perform this step for EACH recurring pattern:
- Actively search the observations for COUNTER-EXAMPLES: cases where the pattern did NOT hold
- If a pattern has zero counter-examples across all observations, lower your confidence — it may be an artifact of source selection bias
- Express contradictions using "vs" between the two sides (e.g., "Signal A suggests X vs Signal B suggests not-X")

### NOTABLE ABSENCES CHECK
Answer this question explicitly: "What patterns would you EXPECT to see in this type of data over this time period that are NOT present?" Missing expected patterns often reveal source data bias or blind spots more important than any pattern found.

Absent patterns and contradictions are the MOST valuable outputs. A clean, contradiction-free analysis is a failed analysis."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("recognize_patterns"),
            schema=PatternBatch,
            tool="recognize_patterns",
            tracker=self.tracker,
            max_tokens=8192,
        )

        for p_data in result.patterns:
            pat = Pattern(
                id=f"pat_{uuid.uuid4().hex[:8]}",
                type=p_data.get("type", "recurring"),
                description=p_data.get("description", ""),
                confidence=p_data.get("confidence", 0.5),
            )
            state.patterns.append(pat)
            # Detect contradictions from interference patterns
            if pat.type == "interference":
                parts = re.split(r'\s+(?:vs\.?|versus|VS)\s+', pat.description, maxsplit=1)
                if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                    state.contradictions.append(Contradiction(
                        id=f"con_{uuid.uuid4().hex[:8]}",
                        insight_a=parts[0].strip(),
                        insight_b=parts[1].strip(),
                    ))
                    self.emit("contradiction_found", pat.id, state.round)

            event_type = "pattern_absent_found" if pat.type == "absent" else "pattern_added"
            self.emit(event_type, pat.id, state.round)


class FormPatternsTool(BaseTool):
    name = "form_patterns"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if enough individual patterns exist to combine."""
        return len(state.patterns) >= 3

    def run(self, state: CognitiveState, **kwargs):
        if len(state.patterns) < 3:
            return

        patterns_text = "\n".join(
            f"[{p.type}] {p.description} (confidence: {p.confidence:.0%})"
            for p in state.patterns
        )

        prompt = f"""You are a pattern composer. You have individual patterns — now combine them.

## Existing Patterns
{patterns_text}

## Your Task
Like African music: each pattern is simple, but JUXTAPOSITION creates complexity.

1. Take 2-3 existing patterns and combine them
2. What NEW pattern emerges from their interaction? (moiré effect)
3. What does the combination reveal that individual patterns don't?

Create 3-5 compound patterns. Each should be:
- type: "interference" (born from combining others)
- description: what the compound pattern reveals
- confidence: your confidence in this combination
- evidence_refs: IDs of patterns combined

These compound patterns should be MORE INSIGHTFUL than any individual pattern."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("form_patterns"),
            schema=PatternBatch,
            tool="form_patterns",
            tracker=self.tracker,
        )

        for p_data in result.patterns:
            pat = Pattern(
                id=f"cpat_{uuid.uuid4().hex[:8]}",
                type="interference",
                description=p_data.get("description", ""),
                confidence=p_data.get("confidence", 0.5),
                related_patterns=p_data.get("evidence_refs", []),
            )
            state.patterns.append(pat)
            self.emit("pattern_added", pat.id, state.round)
