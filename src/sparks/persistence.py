"""Persistence — synapses, knowledge base, and learning survive across sessions.

"Fire together, wire together" only matters if the wiring persists.
A brain that forgets its connections every night isn't learning.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

SPARKS_HOME = Path.home() / ".sparks"


# ─── Persistent Synapses ───


class SynapseStore:
    """Tool-to-tool connection strengths that persist across sessions."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else SPARKS_HOME / "synapses.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, float] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (json.JSONDecodeError, Exception):
                self.data = {}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, key: str, default: float = 1.0) -> float:
        return self.data.get(key, default)

    def set(self, key: str, value: float):
        self.data[key] = value

    def merge_from_session(self, session_synapses: dict[str, float], learning_rate: float = 0.3):
        """Merge session synapses into persistent store.

        Uses exponential moving average: persistent = (1-lr)*persistent + lr*session
        """
        for key, session_val in session_synapses.items():
            persistent_val = self.data.get(key, 1.0)
            self.data[key] = (1 - learning_rate) * persistent_val + learning_rate * session_val
        self.save()

    def load_into_session(self) -> dict[str, float]:
        """Load persistent synapses as starting point for a new session."""
        return dict(self.data)


# ─── Persistent Knowledge Base ───


class KnowledgeEntry(BaseModel):
    domain: str
    goal: str
    principles: list[dict]  # [{statement, confidence}]
    analogies: list[dict] = []
    patterns_summary: str = ""
    timestamp: str = ""
    score: float = 0.0


class KnowledgeBase:
    """Past analysis results that enable cross-session analogy and learning.

    Every completed analysis adds to the KB.
    Future analyses can draw on past principles for structural analogy.
    = The framework gets smarter with every use.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else SPARKS_HOME / "knowledge.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[KnowledgeEntry] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                for line in self.path.read_text().strip().split("\n"):
                    if line.strip():
                        self.entries.append(KnowledgeEntry.model_validate_json(line))
            except Exception:
                self.entries = []

    def save_run(self, domain: str, goal: str, principles: list, analogies: list = None,
                 patterns_summary: str = "", score: float = 0.0):
        """Save a completed analysis to the knowledge base."""
        entry = KnowledgeEntry(
            domain=domain,
            goal=goal,
            principles=[{"statement": p.statement, "confidence": p.confidence} for p in principles],
            analogies=[{"current": a.current, "past_match": a.past_match,
                        "structural_mapping": a.structural_mapping} for a in (analogies or [])],
            patterns_summary=patterns_summary,
            timestamp=datetime.now().isoformat(),
            score=score,
        )
        self.entries.append(entry)

        with open(self.path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def search(self, query: str, top_k: int = 3) -> list[KnowledgeEntry]:
        """Simple keyword search for relevant past analyses."""
        if not self.entries:
            return []

        query_words = set(query.lower().split())
        scored = []
        for entry in self.entries:
            entry_words = set()
            entry_words.update(entry.domain.lower().split())
            entry_words.update(entry.goal.lower().split())
            for p in entry.principles:
                entry_words.update(p["statement"].lower().split())

            overlap = len(query_words & entry_words)
            if overlap > 0:
                scored.append((overlap, entry))

        scored.sort(key=lambda x: -x[0])
        return [entry for _, entry in scored[:top_k]]

    def get_past_principles(self, top_k: int = 10) -> list[str]:
        """Get all past principles for cross-domain analogy."""
        principles = []
        for entry in sorted(self.entries, key=lambda e: -e.score)[:20]:
            for p in entry.principles:
                principles.append(f"[{entry.domain}] {p['statement']}")
                if len(principles) >= top_k:
                    return principles
        return principles

    @property
    def total_entries(self) -> int:
        return len(self.entries)

    @property
    def domains_seen(self) -> list[str]:
        return list({e.domain for e in self.entries})


# ─── Session Manager ───


class SessionMemory:
    """Manages persistence across sessions."""

    def __init__(self):
        self.synapses = SynapseStore()
        self.kb = KnowledgeBase()

    def start_session(self) -> dict[str, float]:
        """Load persistent state at session start."""
        return self.synapses.load_into_session()

    def end_session(self, state, output=None):
        """Save session results to persistent stores."""
        from sparks.state import CognitiveState, SynthesisOutput

        # Merge synapses
        if hasattr(state, 'signals') and state.signals.synapses:
            self.synapses.merge_from_session(state.signals.synapses)

        # Save to knowledge base
        if output and hasattr(output, 'principles') and output.principles:
            domain = state.lens.domain if state.lens else "unknown"
            score = output.confidence * output.coverage if output.confidence and output.coverage else 0
            self.kb.save_run(
                domain=domain,
                goal=state.goal,
                principles=output.principles,
                analogies=output.analogies if hasattr(output, 'analogies') else [],
                patterns_summary=f"{len(state.patterns)} patterns, {len(state.observations)} observations",
                score=score,
            )

    def get_past_context(self, goal: str) -> str:
        """Get relevant past knowledge for the analogize tool."""
        past_principles = self.kb.get_past_principles(top_k=5)
        if not past_principles:
            return ""
        return "## Past Analyses (from knowledge base)\n" + "\n".join(f"- {p}" for p in past_principles)
