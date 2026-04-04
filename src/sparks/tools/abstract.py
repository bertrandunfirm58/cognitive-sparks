"""Tool #3: Abstract — the Picasso Bull method. The intellectual core."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, Principle, Evidence, Phase
from sparks.tools.base import BaseTool


class RemainingPrinciple(BaseModel):
    statement: str
    confidence: float = 0.7
    supporting_patterns: list[str] = []
    reason_kept: str = ""


class AbstractionRound(BaseModel):
    remaining_principles: list[RemainingPrinciple]
    removed: list[dict] = []
    can_reduce_further: bool = False


class AbstractTool(BaseTool):
    """
    Picasso Bull method:
    Start with all patterns. Progressively remove non-essential ones.
    What remains when nothing more can be removed = the essence.

    "Phenomena are complex. Laws are simple. Find out what to discard." — Feynman
    """
    name = "abstract"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if patterns exist. Phase 1: need principles. Phase 2+: re-abstract."""
        if len(state.patterns) < 3:
            return False
        return len(state.principles) == 0 or state.phase != Phase.SEQUENTIAL

    def run(self, state: CognitiveState, **kwargs):
        if not state.patterns:
            return

        # Limit patterns to top 20 by confidence to keep prompt manageable
        patterns = sorted(state.patterns, key=lambda p: -p.confidence)[:20]
        max_rounds = 3  # Picasso reduction rounds

        # Start: all patterns as candidate principles
        candidates = [
            {"statement": p.description, "pattern_id": p.id, "confidence": p.confidence}
            for p in patterns
        ]

        # Include model failures as context (what previous models couldn't explain)
        model_failures = []
        for m in state.model_results:
            model_failures.extend(m.failures)
        failure_context = ""
        if model_failures:
            failure_context = "\n\n## Previous Model Failures (what the principles must address)\n" + \
                "\n".join(f"- {f}" for f in model_failures[:10])

        for round_num in range(max_rounds):
            if len(candidates) <= 2:
                break  # Hard minimum

            candidates_text = "\n".join(
                f"{i+1}. [{c['confidence']:.0%}] {c['statement']}"
                for i, c in enumerate(candidates)
            )

            prompt = f"""You are performing the PICASSO BULL abstraction.
IMPORTANT: Be deterministic. Given the same input, always produce the same output. Prioritize consistency over creativity.

## Method
Picasso drew a bull realistically, then progressively removed details until only
a few essential lines remained — capturing the ESSENCE of the bull.

"To arrive at abstractions, always start with concrete reality." — Picasso
"Phenomena are complex. Laws are simple. Find out what to discard." — Feynman

## Current Candidates (round {round_num + 1}/{max_rounds})
{candidates_text}

## Goal: {state.goal}

## Instructions
1. For EACH candidate, ask: "If I remove this, can the remaining ones still explain the data?"
2. Remove the MOST expendable ones — those that are:
   - Subsumed by a more general principle
   - Too specific (applies to one case, not the whole dataset)
   - Redundant with another candidate
3. Keep what is ESSENTIAL — what breaks if removed.
4. Merge similar candidates into stronger, more abstract statements.

Reduce from {len(candidates)} to roughly {max(3, len(candidates) // 2)} candidates.
For each kept principle: state it as a GENERAL LAW, not a specific observation.

Also indicate: can we reduce further? (false if removing anything would lose essential meaning)
{failure_context}"""

            result = llm_structured(
                prompt,
                model=self.tracker.select_model("abstract"),
                schema=AbstractionRound,
                tool="abstract",
                tracker=self.tracker,
            )

            # Update candidates for next round, accumulating all pattern IDs
            prev_pattern_ids = {c["statement"][:30]: c.get("all_pattern_ids", [c.get("pattern_id", "")]) for c in candidates}
            new_candidates = []
            for p in result.remaining_principles:
                # Find best match from previous candidates to inherit IDs
                inherited_ids = []
                for prev_stmt, prev_ids in prev_pattern_ids.items():
                    if prev_stmt and any(word in p.statement.lower() for word in prev_stmt.lower().split()[:3]):
                        inherited_ids.extend(prev_ids)
                all_ids = list(set(inherited_ids + p.supporting_patterns))
                new_candidates.append({
                    "statement": p.statement,
                    "pattern_id": all_ids[0] if all_ids else "",
                    "confidence": p.confidence,
                    "supporting_patterns": all_ids,
                    "all_pattern_ids": all_ids,
                })
            candidates = new_candidates

            if not result.can_reduce_further:
                break

        # Final: convert candidates to Principles
        state.principles = []
        for i, c in enumerate(candidates):
            principle = Principle(
                id=f"prin_{uuid.uuid4().hex[:8]}",
                statement=c["statement"],
                supporting_patterns=c.get("supporting_patterns", []),
                confidence=c.get("confidence", 0.7),
                abstraction_level=max_rounds - len(candidates) + 1,
                round_extracted=state.round,
            )
            state.principles.append(principle)
            self.emit("principle_extracted", principle.id, state.round)
