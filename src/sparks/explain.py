"""Explainability Layer — why did this tool fire?

For each tool firing in the autonomic cascade, generates a structured
explanation of the neural circuit dynamics that led to the selection.

This replaces "magic black-box" with transparent, auditable reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sparks.circuit import NeuralCircuit, SENSORY, SIGNALS, MODES


@dataclass
class FiringExplanation:
    """Why a specific tool was selected to fire."""
    tool: str
    activation: float
    mode: str                              # sympathetic / parasympathetic / balanced
    top_drivers: list[tuple[str, float]]   # [(source, contribution), ...] sorted desc
    hunger_levels: dict[str, float]        # obs/pat/prin hunger
    signal_levels: dict[str, float]        # convergence/contradiction/...
    modulators: dict[str, float]           # DA/NE/ACh
    runner_up: str | None = None           # Second-highest tool
    runner_up_activation: float = 0.0
    suppressed: list[str] = field(default_factory=list)  # Tools suppressed (refractory/limit)
    summary: str = ""                      # One-line natural language

    def format_log(self, firing_num: int) -> str:
        """Rich-compatible log line for console output."""
        lines = []
        # Header
        mode_icon = {"sympathetic": "🔴", "parasympathetic": "🔵", "balanced": "⚪"}.get(self.mode, "⚪")
        lines.append(
            f"   {mode_icon} [{firing_num:2d}] [bold]{self.tool}[/] "
            f"(act={self.activation:.2f}, {self.mode})"
        )
        # Why
        drivers_str = ", ".join(
            f"{src}→{self.tool} ({contrib:+.2f})"
            for src, contrib in self.top_drivers[:4]
        )
        lines.append(f"         [dim]why: {drivers_str}[/]")

        # Hunger + signals in one line
        active_hungers = [
            f"{k.replace('_hunger', '')}={v:.1f}"
            for k, v in self.hunger_levels.items() if v > 0.3
        ]
        active_signals = [
            f"{k}={v:.2f}"
            for k, v in self.signal_levels.items() if v > 0.3
        ]
        context_parts = []
        if active_hungers:
            context_parts.append(f"hunger({', '.join(active_hungers)})")
        if active_signals:
            context_parts.append(f"signals({', '.join(active_signals)})")
        context_parts.append(
            f"DA={self.modulators['dopamine']:+.2f} "
            f"NE={self.modulators['norepinephrine']:.2f} "
            f"ACh={self.modulators['acetylcholine']:.2f}"
        )
        lines.append(f"         [dim]{' | '.join(context_parts)}[/]")

        if self.runner_up:
            lines.append(
                f"         [dim]runner-up: {self.runner_up} ({self.runner_up_activation:.2f})[/]"
            )

        return "\n".join(lines)

    def format_summary(self) -> str:
        """Short one-line explanation."""
        return self.summary


def explain_firing(
    circuit: NeuralCircuit,
    winner: str,
    candidates: dict[str, float],
    suppressed_tools: list[str] | None = None,
) -> FiringExplanation:
    """Generate explanation for why `winner` was selected.

    Computes the contribution of each incoming connection to the winner's
    activation, sorted by magnitude. This shows exactly which signals
    and modes drove the selection.
    """
    # 1. Compute per-connection contribution to winner
    contributions: list[tuple[str, float]] = []
    for conn in circuit.connections:
        if conn.target == winner and conn.source in circuit.populations:
            source_rate = circuit.populations[conn.source].rate
            contrib = source_rate * conn.effective_weight
            if abs(contrib) > 0.01:  # Skip negligible
                contributions.append((conn.source, contrib))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    # 2. Hunger levels
    hungers = {
        name: circuit.populations[name].rate
        for name in ("obs_hunger", "pat_hunger", "prin_hunger")
        if name in circuit.populations
    }

    # 3. Signal levels
    signals = {
        name: circuit.populations[name].rate
        for name in SIGNALS
        if name in circuit.populations
    }

    # 4. Modulators
    modulators = {
        "dopamine": circuit.dopamine,
        "norepinephrine": circuit.norepinephrine,
        "acetylcholine": circuit.acetylcholine,
    }

    # 5. Runner-up
    runner_up = None
    runner_up_act = 0.0
    for name, rate in sorted(candidates.items(), key=lambda x: -x[1]):
        if name != winner:
            runner_up = name
            runner_up_act = rate
            break

    # 6. Generate natural language summary
    summary = _generate_summary(winner, contributions, hungers, signals, circuit.get_mode())

    return FiringExplanation(
        tool=winner,
        activation=candidates.get(winner, 0.0),
        mode=circuit.get_mode(),
        top_drivers=contributions,
        hunger_levels=hungers,
        signal_levels=signals,
        modulators=modulators,
        runner_up=runner_up,
        runner_up_activation=runner_up_act,
        suppressed=suppressed_tools or [],
        summary=summary,
    )


def _generate_summary(
    tool: str,
    contributions: list[tuple[str, float]],
    hungers: dict[str, float],
    signals: dict[str, float],
    mode: str,
) -> str:
    """Generate a natural language explanation of why this tool fired."""
    parts = []

    # Top driver
    if contributions:
        top_src, top_val = contributions[0]
        if top_src in ("obs_hunger", "pat_hunger", "prin_hunger"):
            hunger_name = top_src.replace("_hunger", "")
            parts.append(f"high {hunger_name} hunger ({hungers.get(top_src, 0):.1f})")
        elif top_src in SIGNALS:
            parts.append(f"{top_src} signal strong ({signals.get(top_src, 0):.2f})")
        elif top_src in MODES:
            parts.append(f"{top_src} mode active")
        elif top_src.endswith("_count"):
            count_name = top_src.replace("_count", "")
            parts.append(f"existing {count_name} data drives next step")
        else:
            parts.append(f"connection from {top_src}")

    # Mode context
    if mode == "sympathetic":
        parts.append("exploration phase")
    elif mode == "parasympathetic":
        parts.append("integration phase")

    # Signal context
    if signals.get("contradiction", 0) > 0.4:
        parts.append("contradictions detected")
    if signals.get("anomaly", 0) > 0.4:
        parts.append("anomaly active")

    return f"{tool} fired: {'; '.join(parts)}" if parts else f"{tool} fired (default activation)"


@dataclass
class CascadeTrace:
    """Full trace of a cascade execution for post-hoc analysis."""
    explanations: list[FiringExplanation] = field(default_factory=list)
    consolidation_events: list[int] = field(default_factory=list)
    termination_reason: str = ""

    def add(self, explanation: FiringExplanation):
        self.explanations.append(explanation)

    def to_dict(self) -> dict:
        """Serialize for JSON export."""
        return {
            "firings": [
                {
                    "step": i + 1,
                    "tool": e.tool,
                    "activation": round(e.activation, 3),
                    "mode": e.mode,
                    "top_drivers": [
                        {"source": s, "contribution": round(c, 3)}
                        for s, c in e.top_drivers[:5]
                    ],
                    "hunger": {k: round(v, 2) for k, v in e.hunger_levels.items()},
                    "signals": {k: round(v, 2) for k, v in e.signal_levels.items() if v > 0.1},
                    "modulators": {k: round(v, 3) for k, v in e.modulators.items()},
                    "runner_up": e.runner_up,
                    "summary": e.summary,
                }
                for i, e in enumerate(self.explanations)
            ],
            "consolidation_at": self.consolidation_events,
            "termination": self.termination_reason,
            "total_firings": len(self.explanations),
        }

    def format_report(self) -> str:
        """Human-readable full trace."""
        lines = ["═══ Cascade Trace ═══", ""]
        for i, e in enumerate(self.explanations):
            lines.append(f"[{i+1}] {e.summary}")
            drivers = ", ".join(f"{s}({c:+.2f})" for s, c in e.top_drivers[:3])
            lines.append(f"    drivers: {drivers}")
            if e.runner_up:
                lines.append(f"    runner-up: {e.runner_up} ({e.runner_up_activation:.2f})")
            lines.append("")
        if self.consolidation_events:
            lines.append(f"Consolidation at steps: {self.consolidation_events}")
        lines.append(f"Terminated: {self.termination_reason}")
        return "\n".join(lines)
