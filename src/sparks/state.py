"""Cognitive State — the shared blackboard for all 13 thinking tools."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Phase ──


class Phase(str, Enum):
    SEQUENTIAL = "sequential"
    ITERATIVE = "iterative"
    INTEGRATED = "integrated"


# ── Evidence ──


class Evidence(BaseModel):
    source: str
    snippet: str
    confidence: float = 0.5


# ── Observation Channel ──


class ObservationChannel(BaseModel):
    name: str
    data_type: str  # "text", "numeric", "categorical", "temporal"
    description: str
    priority: int = 3  # 1-5


# ── Lens ──


class Lens(BaseModel):
    domain: str
    domain_description: str = ""
    channels: list[ObservationChannel] = []
    focus_questions: list[str] = []
    anomaly_criteria: list[str] = []
    absence_criteria: list[str] = []
    version: int = 0
    confidence: float = 0.3
    source: str = "bootstrap"  # "bootstrap" | "user" | "evolved"


# ── Tool Outputs ──


class Observation(BaseModel):
    id: str
    channel: str
    content: str
    lens_used: str = ""
    evidence: list[Evidence] = []
    confidence: float = 0.5
    timestamp: datetime = Field(default_factory=datetime.now)


class Pattern(BaseModel):
    id: str
    type: str  # "recurring" | "absent" | "interference"
    description: str
    evidence: list[Evidence] = []
    confidence: float = 0.5
    related_patterns: list[str] = []


class Principle(BaseModel):
    id: str
    statement: str
    supporting_patterns: list[str] = []
    counter_evidence: list[Evidence] = []
    abstraction_level: int = 1
    confidence: float = 0.5
    round_extracted: int = 0


class Analogy(BaseModel):
    id: str
    current: str
    past_match: str
    structural_mapping: str
    prediction: str = ""
    confidence: float = 0.5


class Contradiction(BaseModel):
    id: str
    insight_a: str
    insight_b: str
    possible_conditions: list[str] = []
    resolved: bool = False
    resolution: Optional[str] = None


class ModelResult(BaseModel):
    id: str
    fidelity: str = "cardboard"  # "cardboard" | "wood" | "steel"
    accuracy: Optional[float] = None
    failures: list[str] = []
    insights: list[str] = []


class PlayDiscovery(BaseModel):
    id: str
    constraint_broken: str
    discovery: str
    useful: Optional[bool] = None


class PerspectiveInsight(BaseModel):
    id: str
    perspective: str
    interpretation: str
    differs_from_default: str = ""


class Hypothesis(BaseModel):
    id: str
    statement: str
    probability: float = 0.5
    evidence_for: list[Evidence] = []
    evidence_against: list[Evidence] = []


# ── Snapshot ──


class StateSnapshot(BaseModel):
    round: int
    principle_count: int
    principle_ids: list[str]
    principle_statements: list[str]
    pattern_count: int
    contradiction_count: int
    summary: str = ""


# ── Nervous System Signals ──


class SignalPotential(BaseModel):
    """Biological signal: accumulates excitation/inhibition → fires at threshold."""
    value: float = 0.0
    threshold: float = 0.6
    fired: bool = False
    refractory: int = 0  # rounds of cooldown after firing

    def contribute(self, amount: float):
        """Excitation (+) or inhibition (-) from a tool."""
        if self.refractory > 0:
            return  # absolute refractory period
        self.value = max(-1.0, min(1.0, self.value + amount))

    def check_fire(self) -> bool:
        """Threshold firing: all-or-none."""
        if self.refractory > 0:
            self.refractory -= 1
            self.fired = False
            return False
        if self.value >= self.threshold:
            self.fired = True
            self.value = 0.0  # reset after firing
            self.refractory = 1  # 1-round refractory
            return True
        self.fired = False
        return False

    def decay(self, rate: float = 0.1):
        """Natural decay toward resting state."""
        self.value *= (1.0 - rate)


class Neuromodulators(BaseModel):
    """4 modulators that control learning dynamics."""
    dopamine: float = 0.0       # reward prediction error (-1 to 1)
    norepinephrine: float = 0.5  # explore(high) vs exploit(low)
    serotonin: float = 0.5      # long-term(high) vs short-term(low)
    acetylcholine: float = 0.5  # learning rate / precision


class AutonomicMode(BaseModel):
    """Sympathetic (explore/diverge) vs Parasympathetic (integrate/converge)."""
    mode: str = "balanced"  # "sympathetic" | "parasympathetic" | "balanced"
    vagal_tone: float = 0.5  # flexibility of mode switching (0-1)


class NervousSignals(BaseModel):
    # ── Biological signal potentials (replace bool flags) ──
    convergence_potential: SignalPotential = Field(default_factory=lambda: SignalPotential(threshold=0.6))
    contradiction_potential: SignalPotential = Field(default_factory=lambda: SignalPotential(threshold=0.4))
    diminishing_potential: SignalPotential = Field(default_factory=lambda: SignalPotential(threshold=0.5))
    anomaly_potential: SignalPotential = Field(default_factory=lambda: SignalPotential(threshold=0.5))
    sufficient_potential: SignalPotential = Field(default_factory=lambda: SignalPotential(threshold=0.7))

    # ── Convenience: still expose bool for backward compat ──
    @property
    def convergence(self) -> bool:
        return self.convergence_potential.fired

    @property
    def contradiction(self) -> bool:
        return self.contradiction_potential.fired

    @property
    def diminishing_returns(self) -> bool:
        return self.diminishing_potential.fired

    @property
    def anomaly(self) -> bool:
        return self.anomaly_potential.fired

    @property
    def sufficient_depth(self) -> bool:
        return self.sufficient_potential.fired

    # ── Neuromodulators ──
    modulators: Neuromodulators = Field(default_factory=Neuromodulators)

    # ── Autonomic mode ──
    autonomic: AutonomicMode = Field(default_factory=AutonomicMode)

    # ── Tool tracking ──
    active_tools: list[str] = []
    tool_rewards: dict[str, float] = {}  # tool → cumulative reward
    tool_activity: dict[str, int] = {}   # tool → run count this round
    habituation: dict[str, float] = {}   # pattern_type → weight (decays with repetition)

    # ── Hebbian plasticity: tool→tool connection strengths ──
    # "fire together, wire together"
    synapses: dict[str, float] = {}  # "toolA→toolB" → strength (0-2, default 1.0)
    last_fired: list[str] = []  # order of tool execution this round (for STDP)

    # ── Population coding: which tools contributed to each signal ──
    signal_contributors: dict[str, list[str]] = {}  # "convergence" → ["abstract", "model"]

    # ── Sleep/consolidation state ──
    consolidation_needed: bool = False

    # ── Oscillation groups ──
    rhythm_groups: dict[str, list[str]] = {
        "sensory": ["observe", "empathize", "shift_dimension"],   # theta-like: intake
        "processing": ["recognize_patterns", "form_patterns", "transform"],  # beta-like: analysis
        "integration": ["abstract", "analogize", "model", "synthesize"],  # gamma-like: binding
        "exploration": ["play", "imagine"],  # alpha-like: open/creative
    }

    # ── Meta ──
    round: int = 0
    phase: Phase = Phase.SEQUENTIAL
    total_cost: float = 0.0
    prediction_errors: list[str] = []  # observations that violated predictions


# ── Synthesis Output ──


class SynthesisOutput(BaseModel):
    principles: list[Principle] = []
    convergence_score: float = 0.0
    coverage: float = 0.0
    contradictions: list[Contradiction] = []
    analogies: list[Analogy] = []
    model_accuracy: Optional[float] = None
    rounds_completed: int = 0
    tools_used: list[str] = []
    total_cost: float = 0.0
    confidence: float = 0.0
    limitations: list[str] = []
    thinking_process: dict = {}


# ── Main State ──


class CognitiveState(BaseModel):
    goal: str
    phase: Phase = Phase.SEQUENTIAL
    round: int = 0
    depth: str = "standard"

    # Lens
    lens: Optional[Lens] = None

    # Nervous system
    signals: NervousSignals = Field(default_factory=NervousSignals)

    # Layer 1: Tool outputs
    observations: list[Observation] = []
    patterns: list[Pattern] = []
    principles: list[Principle] = []
    analogies: list[Analogy] = []
    contradictions: list[Contradiction] = []
    model_results: list[ModelResult] = []

    # Layer 2
    forgotten_rounds: list[int] = []

    # History
    snapshots: dict[int, StateSnapshot] = {}

    def take_snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            round=self.round,
            principle_count=len(self.principles),
            principle_ids=[p.id for p in self.principles],
            principle_statements=[p.statement for p in self.principles],
            pattern_count=len(self.patterns),
            contradiction_count=len(self.contradictions),
        )

    def clean_slate(self):
        """Strategic forgetting: clear derived, keep observations."""
        snapshot = self.take_snapshot()
        self.snapshots[self.round] = snapshot
        self.patterns = []
        self.principles = []
        self.analogies = []
        self.contradictions = []
        self.model_results = []
        self.forgotten_rounds.append(self.round)
        self.round += 1
