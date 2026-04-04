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

        # Collect outputs from ALL tools (not just core pipeline)
        hypotheses_text = ""
        hypotheses = getattr(state, 'hypotheses', [])
        if hypotheses:
            hypotheses_text = "\n".join(
                f"- [{h.probability:.0%}] {h.statement}" for h in hypotheses[:8]
            )

        perspectives_text = ""
        perspectives = getattr(state, 'perspective_insights', [])
        if perspectives:
            perspectives_text = "\n".join(
                f"- **{p.perspective}**: {p.interpretation}" for p in perspectives[:8]
            )

        play_text = ""
        discoveries = getattr(state, 'play_discoveries', [])
        if discoveries:
            play_text = "\n".join(
                f"- [{'useful' if d.useful else 'dead end'}] Broke: {d.constraint_broken[:50]}... → {d.discovery[:80]}..."
                for d in discoveries[:6]
            )

        # Body/dimension/transform observations (stored as observations with special channels)
        special_obs = [o for o in state.observations if o.lens_used in
                       ("body_thinking", "dimensional_thinking", "transformation")]
        special_text = ""
        if special_obs:
            special_text = "\n".join(
                f"- [{o.lens_used}] {o.content}" for o in special_obs[:10]
            )

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

### Hypotheses (from imagination — "what if" scenarios)
{hypotheses_text or "None."}

### Empathy Insights (from inside the data's actors)
{perspectives_text or "None."}

### Play Discoveries (from breaking rules)
{play_text or "None."}

### Body/Dimension/Transform Observations
{special_text or "None."}

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
            max_tokens=8192,
        )

        # Update state principles with refined versions, keep unmatched new ones
        matched_indices = set()
        for fp in result.final_principles:
            found = False
            for i, p in enumerate(state.principles):
                if i not in matched_indices and _similar_enough(p.statement, fp.get("statement", "")):
                    p.statement = fp["statement"]
                    p.confidence = fp.get("confidence", p.confidence)
                    matched_indices.add(i)
                    found = True
                    break
            if not found and fp.get("statement"):
                # New principle from synthesis — add it
                from sparks.state import Principle
                import uuid
                state.principles.append(Principle(
                    id=f"syn_{uuid.uuid4().hex[:8]}",
                    statement=fp["statement"],
                    confidence=fp.get("confidence", 0.7),
                    round_extracted=state.round,
                ))

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
    """Check if two statements refer to the same principle using TF-IDF similarity."""
    from sparks.similarity import semantic_similarity
    return semantic_similarity(a, b) > 0.4
