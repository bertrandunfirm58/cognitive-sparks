"""Tool #10: Model — build a cardboard model and see what breaks."""

from __future__ import annotations

import uuid

from pydantic import BaseModel

from sparks.context import tool_context
from sparks.data import DataStore
from sparks.llm import llm_structured
from sparks.state import CognitiveState, ModelResult
from sparks.tools.base import BaseTool


class ModelOutput(BaseModel):
    accuracy_estimate: float
    explained: list[str]
    failures: list[str]
    insights: list[str]


class ModelTool(BaseTool):
    """
    "The most important thing about building a model is that it reveals
     where your understanding is lacking." — Sparks of Genius

    Cardboard model: quick, rough. The value isn't accuracy — it's
    discovering what you CAN'T explain.
    """
    name = "model"

    def should_run(self, state: CognitiveState) -> bool:
        """Run if principles exist to test."""
        return len(state.principles) >= 1

    def run(self, state: CognitiveState, data: DataStore = None, **kwargs):
        if not state.principles:
            return

        principles_text = "\n".join(
            f"- {p.statement}" for p in state.principles
        )

        # Use observations as test data (already processed, no raw data re-read needed)
        sample_text = ""
        if state.observations:
            import random
            sample_obs = random.sample(state.observations, min(15, len(state.observations)))
            sample_text = "\n".join(
                f"[{o.channel}] {o.content}" for o in sample_obs
            )
        elif data:
            # Fallback to raw data only if no observations
            sample = data.sample(ratio=0.2, min_n=2, max_n=5)
            sample_text = "\n\n".join(
                f"[{s['file']}]: {s['content'][:5000]}" for s in sample
            )

        prompt = f"""You are building a CARDBOARD MODEL — quick, rough, meant to break.

## Extracted Principles
{principles_text}

## Sample Data to Test Against
{sample_text if sample_text else "(no sample available)"}

## Task
Use the extracted principles as a model to EXPLAIN the sample data.

For each sample item, try to explain it using ONLY the principles above.

### MANDATORY TESTING PROTOCOL
Before reporting results, perform these three tests for EACH principle:

**Test A — Testable Predictions:** Generate 2-3 specific, falsifiable predictions from the principle. Check each prediction against observations NOT already cited as evidence for that principle. A principle that only explains its own evidence is circular.

**Test B — Historical Stress Test:** For each principle, identify at least one historical analogue period (e.g., 2008 GFC, 2018 volmageddon, 2020 COVID crash, 2022 UK gilt crisis, dot-com bubble). Would the principle have held during that period? If you cannot identify a historical test, flag the principle as "untested."

**Test C — Novelty Check:** Is this principle well-documented in existing literature (market microstructure, behavioral finance, etc.) or genuinely novel? Novel principles require HIGHER evidence bars, not lower.

### CONFIDENCE CAP RULE
No principle may receive >85% confidence unless it passes at least ONE empirical test (A or B). Principles that fail both tests must be capped at 60% regardless of how intuitive they seem.

Report:
1. accuracy_estimate: What fraction of the data can the principles explain? (0.0-1.0)
2. explained: List what the principles successfully explain
3. failures: List what the principles CANNOT explain — this is the most valuable output!
   "What breaks?" reveals where understanding is lacking.
4. insights: What did you learn from the three tests?
   Which principles survived stress testing? Which broke?
   Which are novel vs. well-known? Which predictions failed?

A 50% accuracy cardboard model that reveals its blind spots
is more valuable than a 70% model that hides them."""

        result = llm_structured(
            prompt,
            model=self.tracker.select_model("model"),
            schema=ModelOutput,
            tool="model",
            tracker=self.tracker,
        )

        model_result = ModelResult(
            id=f"mod_{uuid.uuid4().hex[:8]}",
            fidelity="cardboard",
            accuracy=result.accuracy_estimate,
            failures=result.failures,
            insights=result.insights,
        )
        state.model_results.append(model_result)

        if result.failures:
            self.emit("model_failed", model_result.id, state.round)
        self.emit("model_built", model_result.id, state.round)
