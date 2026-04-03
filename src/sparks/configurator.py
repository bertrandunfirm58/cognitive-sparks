"""Adaptive Tool & Model Routing — auto-configure pipeline based on data.

Not just "what to observe" (lens) but "what tools to use" and "which models" too.
The system thinks about what it needs, not just what it sees.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from sparks.cost import CostTracker
from sparks.lens import DataProfile
from sparks.llm import llm_structured

# ── Domain-specific configurations ──

DOMAIN_CONFIGS: dict[str, dict] = {
    "academic_papers": {
        "tool_hints": {
            "observe": {"subtask": "semantic", "hint": "Read for claims, methods, and gaps"},
            "recognize_patterns": {"subtask": "semantic", "hint": "Look for methodological trends and citation patterns"},
            "form_patterns": {"hint": "Combine findings across papers, not within"},
            "abstract": {"hint": "Seek universal principles, not field-specific findings"},
            "analogize": {"hint": "Match to other scientific domains, history, philosophy"},
            "model": {"hint": "Test if principles predict which papers succeed"},
            "empathize": {"perspectives": ["author", "reviewer", "practitioner"]},
            "shift_dimension": {"hint": "Time axis (evolution), citation network, methodology clusters"},
            "play": {"hint": "Invert the mainstream assumption"},
        },
        "model_overrides": {
            "abstract": "opus",      # Academic abstraction needs deepest thinking
            "analogize": "opus",     # Cross-domain analogy needs breadth
        },
        "tool_boost": ["empathize", "shift_dimension", "play"],  # Activate in standard+
        "external_suggestions": [
            "Semantic Scholar API for citation network analysis",
            "arXiv for latest preprints in related fields",
        ],
    },
    "technical_blog": {
        "tool_hints": {
            "observe": {"subtask": "semantic", "hint": "Focus on architecture decisions and their rationale"},
            "recognize_patterns": {"subtask": "semantic", "hint": "Find recurring design patterns across posts"},
            "form_patterns": {"hint": "Look for how patterns interact and conflict"},
            "abstract": {"hint": "Extract design principles, not implementation details"},
            "analogize": {"hint": "Match to classical engineering, military strategy, biology"},
            "model": {"hint": "Test if principles predict system behavior"},
            "empathize": {"perspectives": ["architect", "new developer", "user"]},
        },
        "model_overrides": {},
        "tool_boost": ["empathize"],
        "external_suggestions": [
            "GitHub repos mentioned in posts for source verification",
        ],
    },
    "source_code": {
        "tool_hints": {
            "observe": {"subtask": "structural", "hint": "Map module boundaries, dependency flows, naming"},
            "recognize_patterns": {"subtask": "mixed", "hint": "Algorithmic patterns + semantic intent patterns"},
            "form_patterns": {"hint": "How do architectural patterns combine across modules?"},
            "abstract": {"hint": "What design philosophy drives the code organization?"},
            "analogize": {"hint": "Match to known design patterns (GoF, SOLID, etc.)"},
            "model": {"hint": "Can principles predict where bugs cluster?"},
            "transform": {"hint": "Code → architecture diagram, dependency graph"},
            "play": {"hint": "What breaks if you remove this module?"},
        },
        "model_overrides": {
            "recognize_patterns": "sonnet",  # Code patterns need understanding
        },
        "tool_boost": ["transform", "play"],
        "external_suggestions": [
            "git log for change frequency (complexity hotspots)",
            "Test coverage data for reliability mapping",
        ],
    },
    "financial_time_series": {
        "tool_hints": {
            "observe": {"subtask": "quantitative", "hint": "Price, volume, momentum, correlation"},
            "recognize_patterns": {"subtask": "statistical", "hint": "Recurring cycles, regime changes, anomalies"},
            "form_patterns": {"hint": "Cross-asset, cross-timeframe pattern combinations"},
            "abstract": {"hint": "Market laws, not trading rules"},
            "analogize": {"hint": "Historical market cycles, macro events, behavioral economics"},
            "model": {"hint": "Backtest: do principles predict direction?"},
            "empathize": {"perspectives": ["institutional trader", "retail investor", "market maker"]},
            "shift_dimension": {"hint": "Daily↔Weekly↔Monthly fractal, sector rotation"},
            "imagine": {"hint": "3-month scenarios with variable manipulation"},
        },
        "model_overrides": {
            "abstract": "opus",
            "synthesize": "opus",
        },
        "tool_boost": ["empathize", "shift_dimension", "imagine"],
        "external_suggestions": [
            "VIX data for market sentiment channel",
            "Sector ETF data for rotation pattern detection",
            "FRED data for macro indicators (rates, CPI, unemployment)",
        ],
    },
    "conversation_data": {
        "tool_hints": {
            "observe": {"subtask": "semantic", "hint": "Content, sentiment, frequency, escalation patterns"},
            "recognize_patterns": {"subtask": "mixed", "hint": "Frequency analysis + sentiment patterns"},
            "abstract": {"hint": "What drives customer behavior? Root causes, not symptoms"},
            "empathize": {"perspectives": ["customer", "support agent", "product manager"]},
            "shift_dimension": {"hint": "Time-of-day, category, sentiment axis"},
            "transform": {"hint": "Text → sentiment score → heatmap"},
        },
        "model_overrides": {
            "empathize": "opus",  # Empathy needs deepest model
        },
        "tool_boost": ["empathize", "shift_dimension", "transform"],
        "external_suggestions": [
            "Sentiment analysis API for emotion channel",
            "Time-bucketed classification for peak patterns",
        ],
    },
    "general": {
        "tool_hints": {
            "observe": {"hint": "Broad scan, look for structure and outliers"},
            "recognize_patterns": {"hint": "Start with frequency, then meaning"},
            "abstract": {"hint": "What would a one-sentence summary miss?"},
            "analogize": {"hint": "What known domain is this most like?"},
        },
        "model_overrides": {},
        "tool_boost": [],
        "external_suggestions": [],
    },
}


# ── Adaptive Configuration ──


class ToolConfig(BaseModel):
    enabled: bool = True
    subtask: str = "default"
    hint: str = ""
    perspectives: list[str] = []


class AdaptiveConfig(BaseModel):
    domain: str
    tool_configs: dict[str, ToolConfig] = {}
    model_overrides: dict[str, str] = {}
    external_suggestions: list[str] = []
    nervous_hints: list[str] = []  # Extra signals for the nervous system


class DynamicDomainConfig(BaseModel):
    """LLM-generated domain configuration for unknown data types."""
    domain_name: str
    tool_hints: dict[str, dict] = {}  # tool_name → {subtask, hint, perspectives}
    model_suggestions: dict[str, str] = {}  # tool_name → "haiku"/"sonnet"/"opus"
    tool_boost: list[str] = []  # extra tools to activate in standard+
    external_suggestions: list[str] = []
    reasoning: str = ""  # why this configuration


def _generate_dynamic_config(
    profile: DataProfile,
    goal: str,
    tracker: Optional[CostTracker] = None,
) -> dict:
    """When no template matches, LLM generates config from scratch."""

    prompt = f"""You are configuring an AI analysis pipeline for unfamiliar data.

## Data Profile
- Items: {profile.total_items}
- Tokens: ~{profile.total_tokens_est}
- Types: {profile.data_types}
- Topics: {', '.join(profile.sample_topics)}
- Languages: {', '.join(profile.languages)}
- Avg item length: {profile.avg_length} chars

## Goal
{goal}

## Available Tools
1. observe — scan data through focused channels
2. recognize_patterns — find recurring, absent, interference patterns
3. form_patterns — combine patterns into compound insights
4. abstract — reduce N patterns to K core principles (Picasso Bull method)
5. analogize — find structural correspondence from other domains
6. model — build cardboard model, see what breaks
7. empathize — analyze from multiple stakeholder perspectives
8. shift_dimension — view data at different scales/axes
9. play — break rules, invert assumptions, explore boundaries
10. transform — convert between modalities (text→graph, etc.)
11. imagine — generate future scenarios
12. synthesize — integrate all findings

## Your Task
Design the optimal configuration:

1. **tool_hints**: For each of the 12 tools, provide:
   - subtask: "semantic" | "statistical" | "structural" | "mixed" | "default"
   - hint: specific guidance for this tool on THIS data
   - perspectives: (for empathize only) list of viewpoints

2. **model_suggestions**: Which tools need expensive models vs cheap?
   - "opus" for tools needing deepest reasoning
   - "sonnet" for tools needing good judgment
   - "haiku" for bulk processing

3. **tool_boost**: Which tools beyond the core 4 (observe, patterns, abstract, synthesize) should be activated for standard depth?

4. **external_suggestions**: What additional data sources would improve analysis?

5. **domain_name**: A descriptive name for this data domain

6. **reasoning**: Why this configuration?

Think carefully about what THIS specific data needs. Don't give generic advice."""

    result = llm_structured(
        prompt,
        model="claude-sonnet-4-20250514",
        schema=DynamicDomainConfig,
        tool="dynamic_config",
        tracker=tracker,
    )

    # Convert to same format as DOMAIN_CONFIGS
    return {
        "tool_hints": result.tool_hints,
        "model_overrides": result.model_suggestions,
        "tool_boost": result.tool_boost,
        "external_suggestions": result.external_suggestions,
        "_dynamic": True,
        "_reasoning": result.reasoning,
        "_domain_name": result.domain_name,
    }


class ToolConfigurator:
    """Auto-configure the pipeline based on data profile and domain.

    Templates (6 domains) are accelerators — speed up known domains.
    Unknown domains get LLM-generated config from scratch.
    The data always decides, not the templates.
    """

    def configure(
        self,
        profile: DataProfile,
        domain: str,
        goal: str,
        depth: str = "standard",
        tracker: Optional[CostTracker] = None,
    ) -> AdaptiveConfig:

        # Templates = accelerator for known domains
        # Unknown = LLM generates from scratch
        if domain == "general":
            domain_config = _generate_dynamic_config(profile, goal, tracker)
            domain = domain_config.get("_domain_name", "auto_detected")
        else:
            domain_config = DOMAIN_CONFIGS.get(domain, DOMAIN_CONFIGS["general"])

        # 1. Tool configurations from domain template
        tool_configs = {}
        for tool_name, hints in domain_config.get("tool_hints", {}).items():
            tool_configs[tool_name] = ToolConfig(
                enabled=True,
                subtask=hints.get("subtask", "default"),
                hint=hints.get("hint", ""),
                perspectives=hints.get("perspectives", []),
            )

        # 2. Boost tools for standard/deep
        if depth in ("standard", "deep"):
            for tool_name in domain_config.get("tool_boost", []):
                if tool_name not in tool_configs:
                    tool_configs[tool_name] = ToolConfig(enabled=True)

        # 3. Model overrides from domain
        model_overrides = dict(domain_config.get("model_overrides", {}))

        # 4. Data size adjustments
        if profile.total_tokens_est > 100000:
            # Large data: use cheaper models for bulk work
            model_overrides.setdefault("observe", "haiku")
            model_overrides.setdefault("recognize_patterns", "haiku")

        if profile.total_tokens_est < 10000:
            # Small data: can afford better models everywhere
            model_overrides.setdefault("observe", "sonnet")
            model_overrides.setdefault("recognize_patterns", "sonnet")

        # 5. Nervous system hints based on data characteristics
        nervous_hints = []
        if profile.total_items < 5:
            nervous_hints.append("Small dataset — convergence may be unreliable, lower threshold")
        if profile.total_tokens_est > 200000:
            nervous_hints.append("Large dataset — aggressive context compression needed")
        if len(profile.sample_topics) > 10:
            nervous_hints.append("High topic diversity — patterns may be fragmented")

        # 6. External suggestions
        external = list(domain_config.get("external_suggestions", []))

        return AdaptiveConfig(
            domain=domain,
            tool_configs=tool_configs,
            model_overrides=model_overrides,
            external_suggestions=external,
            nervous_hints=nervous_hints,
        )


def apply_config(config: AdaptiveConfig, routing: dict[str, str]) -> dict[str, str]:
    """Apply adaptive config overrides to the model routing table."""
    updated = dict(routing)

    # Model name mapping (short → full)
    model_map = {
        "opus": "claude-opus-4-20250514",
        "sonnet": "claude-sonnet-4-20250514",
        "haiku": "claude-haiku-4-5-20251001",
    }

    for tool, model_short in config.model_overrides.items():
        full_name = model_map.get(model_short, model_short)
        updated[tool] = full_name

    return updated
