"""Phase F: Evolution Loop — principles and code evolve together.

Inspired by AutoAgent (kevinrgu/autoagent):
  program.md (goal) + agent.py (current) → benchmark → modify → re-benchmark → keep/rollback

Sparks evolution:
  data → principles → validate → modify principles → re-validate → keep/rollback
  = Understanding evolves, not just code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Optional

from rich.console import Console

from sparks.cost import CostTracker, DEPTH_BUDGETS
from sparks.data import DataStore
from sparks.events import EventBus
from sparks.state import CognitiveState, SynthesisOutput

console = Console()


class EvolutionResult:
    """Result of one evolution cycle."""
    def __init__(
        self,
        generation: int,
        score_before: float,
        score_after: float,
        kept: bool,
        principles_before: list[str],
        principles_after: list[str],
        mutation_description: str,
    ):
        self.generation = generation
        self.score_before = score_before
        self.score_after = score_after
        self.kept = kept
        self.principles_before = principles_before
        self.principles_after = principles_after
        self.mutation_description = mutation_description

    def improved(self) -> bool:
        return self.score_after > self.score_before


class EvolutionLoop:
    """
    The Sparks evolution loop:

    1. Run analysis → extract principles → score
    2. Mutate (re-run with different lens/depth/forgetting strategy)
    3. Re-score
    4. Keep if better, rollback if worse
    5. Repeat

    Score = coverage × confidence × convergence_bonus
    """

    def __init__(self, goal: str, data_path: str, max_generations: int = 3):
        self.goal = goal
        self.data_path = data_path
        self.max_generations = max_generations
        self.history: list[EvolutionResult] = []
        self.best_output: Optional[SynthesisOutput] = None
        self.best_score: float = 0.0

    def score(self, output: SynthesisOutput) -> float:
        """Score a set of principles. Higher = better."""
        if not output.principles:
            return 0.0

        # Coverage: how much of the data do principles explain
        coverage = output.coverage

        # Confidence: average principle confidence
        confidence = sum(p.confidence for p in output.principles) / len(output.principles)

        # Convergence bonus: if converged across rounds, +20%
        convergence_bonus = 1.2 if output.convergence_score > 0.5 else 1.0

        # Penalty for too many or too few principles
        n = len(output.principles)
        count_penalty = 1.0
        if n < 2:
            count_penalty = 0.7  # Too few
        elif n > 7:
            count_penalty = 0.8  # Too many (not abstract enough)

        # Model accuracy bonus
        model_bonus = 1.0
        if output.model_accuracy and output.model_accuracy > 0.5:
            model_bonus = 1.1

        score = coverage * confidence * convergence_bonus * count_penalty * model_bonus
        return round(score, 4)

    def run(self) -> SynthesisOutput:
        """Run the full evolution loop."""
        from sparks.engine import run as engine_run

        console.print(f"\n[bold cyan]🧬 Evolution Loop[/] — {self.max_generations} generations")
        console.print(f"   Goal: {self.goal}")

        # Generation 0: baseline
        console.print(f"\n[bold]Generation 0[/] (baseline, standard mode)")
        output = engine_run(self.goal, self.data_path, depth="standard")
        current_score = self.score(output)
        self.best_output = output
        self.best_score = current_score

        console.print(f"   Score: {current_score:.4f}")

        # Evolution generations
        for gen in range(1, self.max_generations + 1):
            console.print(f"\n[bold]Generation {gen}[/] — mutating...")

            # Mutate: re-run with deep mode (more tools, more rounds)
            mutation = self._select_mutation(gen)
            console.print(f"   Mutation: {mutation['description']}")

            mutated_output = engine_run(
                self.goal,
                self.data_path,
                depth=mutation["depth"],
            )
            new_score = self.score(mutated_output)

            # Compare
            improved = new_score > current_score
            result = EvolutionResult(
                generation=gen,
                score_before=current_score,
                score_after=new_score,
                kept=improved,
                principles_before=[p.statement for p in self.best_output.principles],
                principles_after=[p.statement for p in mutated_output.principles],
                mutation_description=mutation["description"],
            )
            self.history.append(result)

            if improved:
                console.print(f"   Score: {current_score:.4f} → [green]{new_score:.4f} ↑ KEPT[/]")
                self.best_output = mutated_output
                self.best_score = new_score
                current_score = new_score
            else:
                console.print(f"   Score: {current_score:.4f} → [red]{new_score:.4f} ↓ ROLLBACK[/]")

        console.print(f"\n[bold green]🏆 Best score: {self.best_score:.4f}[/]")
        console.print(f"   Generations: {len(self.history)}")
        console.print(f"   Improvements: {sum(1 for r in self.history if r.kept)}")

        return self.best_output

    def _select_mutation(self, generation: int) -> dict:
        """Select mutation strategy for this generation."""
        mutations = [
            {"depth": "deep", "description": "Deep mode — all 13 tools, more rounds"},
            {"depth": "standard", "description": "Re-run standard — different lens bootstrap"},
            {"depth": "quick", "description": "Quick sanity check — do core principles hold?"},
        ]
        return mutations[generation % len(mutations)]

    def save_history(self, path: str = "output/evolution_log.json"):
        """Save evolution history to file."""
        log = []
        for r in self.history:
            log.append({
                "generation": r.generation,
                "score_before": r.score_before,
                "score_after": r.score_after,
                "kept": r.kept,
                "mutation": r.mutation_description,
                "principles_before": r.principles_before,
                "principles_after": r.principles_after,
            })

        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(log, indent=2, ensure_ascii=False))
        console.print(f"[dim]Evolution log saved to {path}[/]")
