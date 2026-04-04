"""Multi-domain benchmark suite.

Runs Sparks + baselines across 3+ domains, comparing:
- Sparks (autonomic, standard depth)
- Single CoT prompt (chain-of-thought baseline)
- Sparks without nervous system (sequential ablation)

Usage:
    python benchmarks/run_all.py
    python benchmarks/run_all.py --domains science code
    python benchmarks/run_all.py --depth quick  # cheaper run
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparks.research import set_seed
from sparks.llm import llm_call
from sparks.cost import CostTracker, DEPTH_BUDGETS

BENCHMARK_DIR = Path(__file__).parent
DOMAINS = {
    "science": {
        "data": str(BENCHMARK_DIR / "science" / "data"),
        "goal": "Extract the fundamental principles governing these biological processes",
        "expected_themes": ["energy", "adaptation", "selection", "constraint", "efficiency"],
    },
    "code": {
        "data": str(BENCHMARK_DIR / "code" / "data"),
        "goal": "Extract the core software engineering principles from these patterns",
        "expected_themes": ["modularity", "coupling", "abstraction", "trade-off", "complexity"],
    },
    "creative": {
        "data": str(BENCHMARK_DIR / "creative" / "data"),
        "goal": "Extract the fundamental principles of effective storytelling",
        "expected_themes": ["conflict", "character", "structure", "tension", "meaning"],
    },
}


def run_sparks(domain_name: str, config: dict, depth: str = "standard") -> dict:
    """Run Sparks autonomic on a domain."""
    set_seed(42)
    try:
        from sparks.autonomic import run_autonomic
        start = time.time()
        result = run_autonomic(goal=config["goal"], data_path=config["data"], depth=depth)
        elapsed = time.time() - start
        return {
            "method": "sparks",
            "domain": domain_name,
            "n_principles": len(result.principles),
            "confidence": result.confidence,
            "coverage": result.coverage,
            "cost": result.total_cost,
            "time": elapsed,
            "principles": [
                {"statement": p.statement, "confidence": p.confidence}
                for p in result.principles
            ],
            "tools_used": result.tools_used,
            "trace": result.thinking_process.get("cascade_trace"),
        }
    except Exception as e:
        return {"method": "sparks", "domain": domain_name, "error": str(e)}


def run_sparks_no_circuit(domain_name: str, config: dict, depth: str = "standard") -> dict:
    """Run Sparks without neural circuit (sequential)."""
    set_seed(42)
    try:
        from sparks.engine import run as engine_run
        start = time.time()
        result = engine_run(
            goal=config["goal"], data_path=config["data"],
            depth=depth, nervous_system=False,
        )
        elapsed = time.time() - start
        return {
            "method": "sparks_no_circuit",
            "domain": domain_name,
            "n_principles": len(result.principles),
            "confidence": result.confidence,
            "coverage": result.coverage,
            "cost": result.total_cost,
            "time": elapsed,
            "principles": [
                {"statement": p.statement, "confidence": p.confidence}
                for p in result.principles
            ],
        }
    except Exception as e:
        return {"method": "sparks_no_circuit", "domain": domain_name, "error": str(e)}


def run_cot_baseline(domain_name: str, config: dict) -> dict:
    """Run chain-of-thought baseline (single prompt with step-by-step)."""
    try:
        from sparks.data import DataStore
        data = DataStore(config["data"])
        data_text = data.all_text(max_chars=80000)
        tracker = CostTracker(DEPTH_BUDGETS["standard"])

        cot_prompt = f"""Analyze this data step by step and extract the core principles.

Think through this carefully:
1. First, observe the key facts and patterns in the data
2. Then, identify recurring themes and contradictions
3. Abstract these into general principles
4. For each principle, rate your confidence (0-100%)

Data:
{data_text}

Goal: {config["goal"]}

Think step by step, then provide numbered principles with confidence scores."""

        model = "claude-opus-4-20250514" if os.environ.get("SPARKS_ALL_OPUS") else "claude-sonnet-4-20250514"
        start = time.time()
        result = llm_call(prompt=cot_prompt, model=model, tool="cot_baseline", tracker=tracker)
        elapsed = time.time() - start

        # Count principles (rough: numbered lines)
        n_principles = sum(
            1 for line in result.split("\n")
            if line.strip() and line.strip()[0].isdigit() and "." in line[:5]
        )

        return {
            "method": "cot_baseline",
            "domain": domain_name,
            "n_principles": max(n_principles, 1),
            "cost": tracker.total_cost,
            "time": elapsed,
            "raw_output": result[:3000],
        }
    except Exception as e:
        return {"method": "cot_baseline", "domain": domain_name, "error": str(e)}


def theme_coverage(principles: list[dict], expected_themes: list[str]) -> float:
    """What fraction of expected themes appear in the principles?"""
    if not principles:
        return 0.0
    all_text = " ".join(p["statement"].lower() for p in principles)
    hits = sum(1 for theme in expected_themes if theme in all_text)
    return hits / len(expected_themes)


def run_all(domains: list[str] | None = None, depth: str = "standard"):
    """Run full benchmark suite."""
    target_domains = domains or list(DOMAINS.keys())
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for domain_name in target_domains:
        if domain_name not in DOMAINS:
            print(f"Unknown domain: {domain_name}, skipping")
            continue

        config = DOMAINS[domain_name]
        print(f"\n{'='*60}")
        print(f"DOMAIN: {domain_name.upper()}")
        print(f"Goal: {config['goal']}")
        print(f"{'='*60}")

        domain_results = {}

        # 1. Sparks (autonomic)
        print(f"\n--- Sparks (autonomic, {depth}) ---")
        r = run_sparks(domain_name, config, depth)
        domain_results["sparks"] = r
        if "error" not in r:
            tc = theme_coverage(r["principles"], config["expected_themes"])
            r["theme_coverage"] = tc
            print(f"  {r['n_principles']} principles, {r['confidence']:.0%} conf, "
                  f"theme coverage: {tc:.0%}, ${r['cost']:.2f}, {r['time']:.0f}s")
        else:
            print(f"  ERROR: {r['error']}")

        # 2. CoT baseline
        print(f"\n--- CoT Baseline ---")
        r = run_cot_baseline(domain_name, config)
        domain_results["cot"] = r
        if "error" not in r:
            print(f"  ~{r['n_principles']} principles, ${r['cost']:.2f}, {r['time']:.0f}s")
        else:
            print(f"  ERROR: {r['error']}")

        # 3. Sparks no circuit (if running standard+)
        if depth != "quick":
            print(f"\n--- Sparks (no circuit) ---")
            r = run_sparks_no_circuit(domain_name, config, depth)
            domain_results["no_circuit"] = r
            if "error" not in r:
                tc = theme_coverage(r["principles"], config["expected_themes"])
                r["theme_coverage"] = tc
                print(f"  {r['n_principles']} principles, {r['confidence']:.0%} conf, "
                      f"theme coverage: {tc:.0%}, ${r['cost']:.2f}, {r['time']:.0f}s")
            else:
                print(f"  ERROR: {r['error']}")

        results[domain_name] = domain_results

    # ══════ Summary ══════
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Domain':<12} {'Method':<18} {'Principles':<12} {'Confidence':<12} {'Themes':<10} {'Cost':<8}")
    print("-" * 72)

    total_cost = 0
    for domain_name, domain_results in results.items():
        for method, r in domain_results.items():
            if "error" in r:
                print(f"{domain_name:<12} {method:<18} ERROR")
                continue
            conf = f"{r.get('confidence', 0):.0%}" if "confidence" in r else "N/A"
            themes = f"{r.get('theme_coverage', 0):.0%}" if "theme_coverage" in r else "N/A"
            cost = r.get("cost", 0)
            total_cost += cost
            print(f"{domain_name:<12} {method:<18} {r['n_principles']:<12} {conf:<12} {themes:<10} ${cost:.2f}")

    print(f"\nTotal cost: ${total_cost:.2f}")

    # Save
    out_dir = BENCHMARK_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"benchmark_{timestamp}.json"
    save_path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    print(f"Results saved to {save_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="*", default=None, help="Domains to test (default: all)")
    parser.add_argument("--depth", default="standard", help="Depth: quick/standard/deep")
    args = parser.parse_args()

    run_all(domains=args.domains, depth=args.depth)
