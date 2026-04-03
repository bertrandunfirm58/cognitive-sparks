"""Tool #13: Synthesize — all tools resonate at once."""

from __future__ import annotations

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.llm import llm_structured
from sparks.state import CognitiveState, SynthesisOutput
from sparks.tools.base import BaseTool


class SynthesisResult(BaseModel):
    final_principles: list[dict]  # [{statement, confidence, evidence_summary, counter_evidence}]
    overall_confidence: float
    coverage_estimate: float
    limitations: list[str]
    key_insight: str


class SynthesizeTool(BaseTool):
    """
    "Transformation is sequential, but synthesis is simultaneous.
     Memory, knowledge, imagination, feeling — all at once, as a whole,
     understood through the body." — Sparks of Genius

    This is not summarization. This is integration.
    """
    name = "synthesize"

    def should_run(self, state: CognitiveState) -> bool:
        """Always run — synthesis is the final integration."""
        return True

    def run(self, state: CognitiveState, **kwargs) -> SynthesisOutput:
        ctx = tool_context("synthesize", state)

        principles_text = "\n".join(
            f"- [{p.confidence:.0%}] {p.statement}\n  Patterns: {', '.join(p.supporting_patterns[:3])}"
            for p in state.principles
        )

        analogies_text = "\n".join(
            f"- {a.current} ↔ {a.past_match}\n  Mapping: {a.structural_mapping}"
            for a in state.analogies[:5]
        ) or "None found."

        contradictions_text = "\n".join(
            f"- {c.insight_a} vs {c.insight_b}"
            for c in state.contradictions if not c.resolved
        ) or "None."

        model_text = ""
        if state.model_results:
            latest = state.model_results[-1]
            model_text = f"""
Model accuracy: {latest.accuracy or 'unknown'}
Failures: {chr(10).join('- ' + f for f in latest.failures[:5])}
Insights: {chr(10).join('- ' + i for i in latest.insights[:5])}"""

        prompt = f"""You are performing SYNTHESIS — the final integration.

This is NOT summarization. This is the moment where all thinking tools
converge simultaneously, like African polyrhythmic music where each
instrument plays its simple pattern but the combination creates complexity.

## Goal
{state.goal}

## What the thinking tools found:

### Principles (from abstraction)
{principles_text}

### Analogies (structural matches from other domains)
{analogies_text}

### Unresolved Contradictions
{contradictions_text}

### Model Test Results
{model_text}

### Full Context
{ctx}

## Your Task
Synthesize everything into FINAL PRINCIPLES:

1. For each principle:
   - Refine the statement to its most powerful, general form
   - Assess confidence based on ALL evidence (patterns, analogies, model tests)
   - Note supporting evidence AND counter-evidence

2. Overall assessment:
   - coverage_estimate: how much of the original data do these principles explain? (0.0-1.0)
   - overall_confidence: how confident are you in this set of principles? (0.0-1.0)
   - key_insight: the single most important thing learned
   - limitations: what couldn't be captured / remains uncertain

The principles should be:
- GENERAL (applicable beyond just this dataset)
- ESSENTIAL (removing any one would lose important understanding)
- EVIDENCE-BASED (rooted in observed patterns, not speculation)"""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("synthesize"),
            schema=SynthesisResult,
            tool="synthesize",
            tracker=self.tracker,
            max_tokens=4096,
        )

        # Update state principles with refined versions
        for fp in result.final_principles:
            for p in state.principles:
                if _similar_enough(p.statement, fp.get("statement", "")):
                    p.statement = fp["statement"]
                    p.confidence = fp.get("confidence", p.confidence)
                    break

        output = SynthesisOutput(
            principles=state.principles,
            convergence_score=1.0 if state.signals.convergence else 0.0,
            coverage=result.coverage_estimate,
            contradictions=state.contradictions,
            analogies=state.analogies,
            model_accuracy=state.model_results[-1].accuracy if state.model_results else None,
            rounds_completed=state.round + 1,
            tools_used=state.signals.active_tools,
            total_cost=state.signals.total_cost,
            confidence=result.overall_confidence,
            limitations=result.limitations,
            thinking_process={
                "observations": len(state.observations),
                "patterns": len(state.patterns),
                "principles_before_synthesis": len(state.principles),
                "analogies": len(state.analogies),
                "contradictions": len(state.contradictions),
                "model_tests": len(state.model_results),
                "key_insight": result.key_insight,
            },
        )

        self.emit("synthesis_complete", "", state.round)
        return output


def _similar_enough(a: str, b: str) -> bool:
    """Quick check if two statements refer to the same principle."""
    a_words = set(a.lower().split())
    b_words = set(b.lower().split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words) / min(len(a_words), len(b_words))
    return overlap > 0.3
