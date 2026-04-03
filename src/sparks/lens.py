"""Lens bootstrapping — Steps 0-2: scan, sense domain, generate lens."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from sparks.cost import CostTracker
from sparks.data import DataStore
from sparks.llm import llm_call, llm_structured
from sparks.state import Lens, ObservationChannel


# ── Step 0: Quick Scan ──


class DataProfile(BaseModel):
    total_items: int
    total_tokens_est: int
    data_types: list[str]
    file_extensions: list[str]
    sample_topics: list[str]
    avg_length: int
    languages: list[str]


def quick_scan(data: DataStore, tracker: Optional[CostTracker] = None) -> DataProfile:
    """Step 0: Profile data without LLM (mostly algorithmic)."""
    sample = data.sample(ratio=0.3, min_n=2, max_n=10)
    sample_text = "\n\n".join(s["content"][:2000] for s in sample)

    # Detect file types
    extensions = list({s["file"].rsplit(".", 1)[-1] if "." in s["file"] else "txt" for s in data.items})

    # Quick topic extraction via LLM (cheap)
    topic_prompt = f"""Analyze this text sample and extract:
1. Main language(s) used (e.g., "English", "Korean", "mixed")
2. 5-7 dominant topics/themes as short phrases
3. What type of content this is (e.g., "academic papers", "blog posts", "source code")

Sample:
{sample_text[:4000]}

Respond as JSON: {{"languages": [...], "topics": [...], "content_type": "..."}}"""

    model = "claude-haiku-4-5-20251001"
    result = llm_call(topic_prompt, model=model, tool="quick_scan", tracker=tracker)

    # Parse (best effort)
    import json
    try:
        parsed = json.loads(result.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError:
        parsed = {"languages": ["unknown"], "topics": ["general"], "content_type": "documents"}

    return DataProfile(
        total_items=data.total_items,
        total_tokens_est=data.estimated_tokens(),
        data_types=[parsed.get("content_type", "documents")],
        file_extensions=extensions,
        sample_topics=parsed.get("topics", [])[:7],
        avg_length=data.total_chars() // max(data.total_items, 1),
        languages=parsed.get("languages", ["unknown"]),
    )


# ── Step 1: Domain Sense ──


KNOWN_DOMAINS = {
    "academic_papers": {
        "signals": ["abstract", "introduction", "methodology", "references", "conclusion"],
        "channels": [
            ("core_claims", "text", "Core claims and contributions of each piece", 5),
            ("methodology", "text", "Methods, experiments, evaluation approaches", 4),
            ("temporal_evolution", "text", "How approaches changed over time", 3),
            ("disagreements", "text", "Conflicting claims between sources", 4),
            ("unspoken_assumptions", "text", "Assumptions not explicitly stated", 5),
        ],
    },
    "technical_blog": {
        "signals": ["architecture", "pattern", "system", "design", "framework", "implementation"],
        "channels": [
            ("architecture_decisions", "text", "Key architectural choices and why", 5),
            ("design_patterns", "text", "Recurring design patterns across sources", 5),
            ("trade_offs", "text", "Explicit and implicit trade-offs made", 4),
            ("innovations", "text", "Novel approaches not seen elsewhere", 4),
            ("gaps_and_limitations", "text", "What's missing or acknowledged as weak", 3),
        ],
    },
    "source_code": {
        "signals": ["import", "class", "function", "def", "return", "module"],
        "channels": [
            ("structure", "text", "Code organization, module boundaries", 5),
            ("patterns", "text", "Design patterns, idioms, conventions", 5),
            ("dependencies", "text", "External and internal dependencies", 3),
            ("complexity_hotspots", "text", "Complex or dense areas of code", 4),
            ("naming_conventions", "text", "Naming patterns that reveal intent", 3),
        ],
    },
    "general": {
        "signals": [],
        "channels": [
            ("main_content", "text", "Primary content and themes", 5),
            ("structure", "text", "How information is organized", 3),
            ("relationships", "text", "Connections between items", 4),
            ("outliers", "text", "Items that don't fit the main pattern", 4),
            ("gaps", "text", "What seems to be missing", 3),
        ],
    },
}


def sense_domain(profile: DataProfile) -> str:
    """Step 1: Classify domain from profile (rule-based)."""
    # Check file extensions first
    code_exts = {"py", "ts", "js", "go", "rs", "java", "c", "cpp"}
    if any(ext in code_exts for ext in profile.file_extensions):
        return "source_code"

    # Check topics for signals
    topics_lower = " ".join(profile.sample_topics).lower()
    for domain, info in KNOWN_DOMAINS.items():
        if domain == "general":
            continue
        hits = sum(1 for s in info["signals"] if s in topics_lower)
        if hits >= 2:
            return domain

    # Check content type
    for dtype in profile.data_types:
        dtype_lower = dtype.lower()
        if "paper" in dtype_lower or "academic" in dtype_lower:
            return "academic_papers"
        if "blog" in dtype_lower or "article" in dtype_lower or "technical" in dtype_lower:
            return "technical_blog"
        if "code" in dtype_lower:
            return "source_code"

    return "general"


# ── Step 2: Lens Generation ──


class LensOutput(BaseModel):
    domain_description: str
    focus_questions: list[str]
    anomaly_criteria: list[str]
    absence_criteria: list[str]
    additional_channels: list[dict] = []


def generate_lens(
    profile: DataProfile,
    domain: str,
    goal: str,
    tracker: Optional[CostTracker] = None,
) -> Lens:
    """Step 2: Generate observation lens from profile + domain."""
    template = KNOWN_DOMAINS.get(domain, KNOWN_DOMAINS["general"])

    prompt = f"""You are preparing a focused observation lens for data analysis.

Goal: {goal}
Domain: {domain}
Data: {profile.total_items} items, ~{profile.total_tokens_est} tokens
Topics found: {', '.join(profile.sample_topics)}
Languages: {', '.join(profile.languages)}

Base observation channels already defined:
{chr(10).join(f'- {ch[0]}: {ch[2]}' for ch in template['channels'])}

Generate:
1. A brief description of this domain and what matters
2. 5 focus questions — what should we look for to achieve the goal?
3. 3 anomaly criteria — what would be unusual/surprising in this data?
4. 3 absence criteria — what SHOULD be there but might be missing?
5. Any additional observation channels beyond the base ones (0-2 max)

Think about what a domain expert would focus on vs. what a newcomer would miss."""

    result = llm_structured(
        prompt,
        model="claude-sonnet-4-20250514",
        schema=LensOutput,
        tool="lens_generate",
        tracker=tracker,
    )

    channels = [
        ObservationChannel(name=ch[0], data_type=ch[1], description=ch[2], priority=ch[3])
        for ch in template["channels"]
    ]
    for extra in result.additional_channels:
        channels.append(ObservationChannel(
            name=extra.get("name", "extra"),
            data_type="text",
            description=extra.get("description", ""),
            priority=extra.get("priority", 3),
        ))

    return Lens(
        domain=domain,
        domain_description=result.domain_description,
        channels=channels,
        focus_questions=result.focus_questions,
        anomaly_criteria=result.anomaly_criteria,
        absence_criteria=result.absence_criteria,
        version=0,
        confidence=0.3,
        source="bootstrap",
    )


def bootstrap_lens(
    data: DataStore,
    goal: str,
    tracker: Optional[CostTracker] = None,
) -> Lens:
    """Full lens bootstrapping pipeline: scan → sense → generate."""
    profile = quick_scan(data, tracker=tracker)
    domain = sense_domain(profile)
    return generate_lens(profile, domain, goal, tracker=tracker)
