"""Research utilities — evidence chains, reproducibility, export, benchmarking.

Makes Sparks output usable in academic papers:
- Evidence chains: principle → pattern → observation → source:line
- Reproducibility: deterministic runs with --seed
- Export: LaTeX tables, Jupyter notebooks
- Benchmarking: N runs → mean±std, reproducibility score
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sparks.state import CognitiveState, SynthesisOutput


# ── Evidence Chain ──


def build_evidence_chains(state: CognitiveState) -> list[dict]:
    """Build full provenance chain: principle → patterns → observations → source.

    Returns list of dicts, one per principle:
    {
        "principle": "...",
        "confidence": 0.85,
        "patterns": [{"description": "...", "observations": [{"content": "...", "source": "file:42"}]}],
        "chain_depth": 3,  # how many layers of evidence
        "source_files": ["file1.txt:42", "file2.txt:103"],
    }
    """
    # Index observations and patterns by ID
    obs_by_id = {o.id: o for o in state.observations}
    pat_by_id = {p.id: p for p in state.patterns}

    chains = []
    for prin in state.principles:
        chain = {
            "principle": prin.statement,
            "confidence": prin.confidence,
            "round": prin.round_extracted,
            "patterns": [],
            "source_files": set(),
        }

        # Find supporting patterns
        for pat_id in prin.supporting_patterns:
            pat = pat_by_id.get(pat_id)
            if not pat:
                # Try fuzzy match by description fragment
                for p in state.patterns:
                    if pat_id in p.description[:50]:
                        pat = p
                        break
            if not pat:
                chain["patterns"].append({"id": pat_id, "description": pat_id, "observations": []})
                continue

            pat_entry = {"id": pat.id, "description": pat.description, "observations": []}

            # Find observations that support this pattern (by content overlap)
            pat_words = set(pat.description.lower().split()[:5])
            for obs in state.observations:
                obs_words = set(obs.content.lower().split()[:10])
                if len(pat_words & obs_words) >= 2:
                    obs_entry = {
                        "id": obs.id,
                        "channel": obs.channel,
                        "content": obs.content[:200],
                        "confidence": obs.confidence,
                        "sources": obs.source_refs,
                    }
                    pat_entry["observations"].append(obs_entry)
                    chain["source_files"].update(obs.source_refs)

            chain["patterns"].append(pat_entry)

        chain["source_files"] = sorted(chain["source_files"])
        chain["chain_depth"] = 1 + (1 if chain["patterns"] else 0) + \
            (1 if any(p.get("observations") for p in chain["patterns"]) else 0)
        chains.append(chain)

    return chains


# ── Reproducibility ──


def set_seed(seed: int):
    """Set random seed for reproducible runs."""
    random.seed(seed)
    # Note: LLM outputs are not fully deterministic even with temperature=0,
    # but setting seed ensures our internal randomness is controlled.


# ── Export: LaTeX ──


def to_latex_table(result: SynthesisOutput, caption: str = "Extracted Principles") -> str:
    """Export principles as LaTeX table."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        f"\\caption{{{caption}}}",
        r"\begin{tabular}{clc}",
        r"\toprule",
        r"\# & Principle & Confidence \\",
        r"\midrule",
    ]

    for i, p in enumerate(result.principles, 1):
        # Escape LaTeX special chars
        stmt = p.statement.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        # Truncate for table width
        if len(stmt) > 100:
            stmt = stmt[:97] + "..."
        conf_pct = f"{p.confidence:.0%}"
        lines.append(f"{i} & {stmt} & {conf_pct} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\label{{tab:principles}}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def to_latex_evidence(chains: list[dict]) -> str:
    """Export evidence chains as LaTeX appendix."""
    lines = [
        r"\section*{Appendix: Evidence Chains}",
        "",
    ]

    for i, chain in enumerate(chains, 1):
        stmt = chain["principle"].replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
        lines.append(f"\\subsection*{{Principle {i}: {stmt[:80]}...}}")
        lines.append(f"Confidence: {chain['confidence']:.0%}, Chain depth: {chain['chain_depth']}")
        lines.append("")

        if chain["source_files"]:
            lines.append(f"Source files: {', '.join(chain['source_files'][:5])}")
            lines.append("")

        lines.append(r"\begin{itemize}")
        for pat in chain["patterns"][:5]:
            desc = pat["description"][:80].replace("&", r"\&").replace("_", r"\_")
            lines.append(f"  \\item Pattern: {desc}")
            if pat.get("observations"):
                lines.append(r"  \begin{itemize}")
                for obs in pat["observations"][:3]:
                    content = obs["content"][:60].replace("&", r"\&").replace("_", r"\_")
                    src = obs.get("sources", [])
                    src_str = f" (source: {', '.join(src[:2])})" if src else ""
                    lines.append(f"    \\item {content}{src_str}")
                lines.append(r"  \end{itemize}")
        lines.append(r"\end{itemize}")
        lines.append("")

    return "\n".join(lines)


def to_latex_full(result: SynthesisOutput, state: CognitiveState,
                  goal: str, caption: str = "Extracted Principles") -> str:
    """Full LaTeX export: table + evidence chains."""
    chains = build_evidence_chains(state)
    parts = [
        to_latex_table(result, caption),
        "",
        to_latex_evidence(chains),
    ]
    return "\n\n".join(parts)


# ── Export: Jupyter Notebook ──


def to_notebook(result: SynthesisOutput, state: CognitiveState, goal: str) -> dict:
    """Export as Jupyter notebook (.ipynb) JSON."""
    chains = build_evidence_chains(state)

    cells = [
        _nb_markdown(f"# Sparks Analysis: {goal}"),
        _nb_markdown(
            f"**Confidence**: {result.confidence:.0%} | "
            f"**Coverage**: {result.coverage:.0%} | "
            f"**Cost**: ${result.total_cost:.2f} | "
            f"**Rounds**: {result.rounds_completed}"
        ),
        _nb_markdown("## Principles"),
    ]

    for i, p in enumerate(result.principles, 1):
        cells.append(_nb_markdown(
            f"### {i}. [{p.confidence:.0%}] {p.statement}\n\n"
            f"Supporting patterns: {', '.join(p.supporting_patterns[:3])}"
        ))

    # Evidence chains
    cells.append(_nb_markdown("## Evidence Chains"))
    for i, chain in enumerate(chains, 1):
        md = f"### Principle {i}\n"
        md += f"Chain depth: {chain['chain_depth']} | Sources: {len(chain['source_files'])}\n\n"
        for pat in chain["patterns"][:5]:
            md += f"- **Pattern**: {pat['description'][:80]}\n"
            for obs in pat.get("observations", [])[:3]:
                src = f" `{obs['sources'][0]}`" if obs.get("sources") else ""
                md += f"  - {obs['content'][:60]}{src}\n"
        cells.append(_nb_markdown(md))

    # Analogies
    if result.analogies:
        cells.append(_nb_markdown("## Analogies"))
        for a in result.analogies:
            cells.append(_nb_markdown(
                f"**{a.current}** <-> **{a.past_match}**\n\n"
                f"Mapping: {a.structural_mapping}\n\n"
                f"Prediction: {a.prediction}"
            ))

    # Thinking process code cell
    cells.append(_nb_code(
        "# Thinking process summary\n"
        f"thinking = {json.dumps(result.thinking_process, indent=2, ensure_ascii=False)}\n"
        "thinking"
    ))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def _nb_markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}


def _nb_code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": [source], "outputs": [], "execution_count": None}


# ── Benchmarking ──


def benchmark(
    goal: str,
    data_path: str,
    n_runs: int = 5,
    depth: str = "standard",
    seeds: list[int] = None,
) -> dict:
    """Run Sparks N times, compute statistics.

    Returns:
    {
        "runs": [...],
        "mean_principles": 5.2,
        "std_principles": 1.1,
        "mean_confidence": 0.78,
        "std_confidence": 0.05,
        "reproducibility": 0.80,  # fraction of principles found in >50% of runs
        "total_cost": 15.30,
        "mean_time": 120.5,
    }
    """
    from sparks.autonomic import run_autonomic
    from sparks.similarity import principle_convergence

    if seeds is None:
        seeds = list(range(42, 42 + n_runs))

    runs = []
    all_principles = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Run {i+1}/{n_runs} (seed={seed})")
        print(f"{'='*60}")

        set_seed(seed)
        start = time.time()

        try:
            result = run_autonomic(goal=goal, data_path=data_path, depth=depth)
            elapsed = time.time() - start

            run_data = {
                "seed": seed,
                "n_principles": len(result.principles),
                "confidence": result.confidence,
                "coverage": result.coverage,
                "cost": result.total_cost,
                "time": elapsed,
                "principles": [p.statement for p in result.principles],
            }
            runs.append(run_data)
            all_principles.append([p.statement for p in result.principles])
        except Exception as e:
            runs.append({"seed": seed, "error": str(e)})

    # Compute statistics
    successful = [r for r in runs if "error" not in r]
    if not successful:
        return {"runs": runs, "error": "All runs failed"}

    n_prins = [r["n_principles"] for r in successful]
    confs = [r["confidence"] for r in successful]
    covs = [r["coverage"] for r in successful]
    costs = [r["cost"] for r in successful]
    times = [r["time"] for r in successful]

    # Reproducibility: for each principle in run 1, how many other runs found it?
    reproducibility = 0.0
    if len(all_principles) >= 2:
        reference = all_principles[0]
        match_counts = []
        for p in reference:
            count = 0
            for other_run in all_principles[1:]:
                _, pairs = principle_convergence([p], other_run)
                if pairs:
                    count += 1
            match_counts.append(count / (len(all_principles) - 1))
        reproducibility = sum(match_counts) / len(match_counts) if match_counts else 0

    return {
        "runs": runs,
        "n_runs": len(successful),
        "mean_principles": _mean(n_prins),
        "std_principles": _std(n_prins),
        "mean_confidence": _mean(confs),
        "std_confidence": _std(confs),
        "mean_coverage": _mean(covs),
        "std_coverage": _std(covs),
        "reproducibility": reproducibility,
        "total_cost": sum(costs),
        "mean_cost": _mean(costs),
        "mean_time": _mean(times),
    }


def format_benchmark(stats: dict) -> str:
    """Format benchmark results as markdown table."""
    lines = [
        "## Benchmark Results\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Runs | {stats['n_runs']} |",
        f"| Principles | {stats['mean_principles']:.1f} +/- {stats['std_principles']:.1f} |",
        f"| Confidence | {stats['mean_confidence']:.0%} +/- {stats['std_confidence']:.0%} |",
        f"| Coverage | {stats['mean_coverage']:.0%} +/- {stats['std_coverage']:.0%} |",
        f"| Reproducibility | {stats['reproducibility']:.0%} |",
        f"| Mean cost | ${stats['mean_cost']:.2f} |",
        f"| Total cost | ${stats['total_cost']:.2f} |",
        f"| Mean time | {stats['mean_time']:.0f}s |",
    ]
    return "\n".join(lines)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
