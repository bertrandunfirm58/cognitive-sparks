"""Self-Optimizer — Sparks analyzes its own output to improve itself.

Run Sparks → Analyze results → Identify weak spots → Generate code fixes → Test → Keep/Rollback

This is meta-cognition: the framework thinking about its own thinking.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from sparks.cost import CostTracker, DEPTH_BUDGETS
from sparks.llm import llm_call, llm_structured

from pydantic import BaseModel

console = Console()

OPTIMIZE_HOME = Path.home() / ".sparks" / "optimize"


# ─── Schemas ───


class ToolDiagnosis(BaseModel):
    tool_name: str
    quality_score: float  # 0-1
    issue: str  # What's wrong
    fix_type: str  # "prompt" | "parameter" | "connection" | "skip"
    suggested_fix: str  # Concrete suggestion


class DiagnosisBatch(BaseModel):
    diagnoses: list[ToolDiagnosis]
    overall_quality: float
    bottleneck: str  # Which tool is the biggest bottleneck
    top_suggestion: str


class PromptFix(BaseModel):
    tool_name: str
    original_section: str  # Which part of the prompt to change
    new_section: str  # Replacement text
    reason: str


class PromptFixBatch(BaseModel):
    fixes: list[PromptFix]


class CircuitTuning(BaseModel):
    connection_changes: list[dict]  # [{source, target, new_weight, reason}]
    threshold_changes: list[dict]  # [{population, new_threshold, reason}]
    reason: str


# ─── Analyze Results ───


def analyze_output(output_path: str, tracker: CostTracker) -> DiagnosisBatch:
    """Analyze a Sparks output file for quality issues."""

    text = Path(output_path).read_text()

    prompt = f"""You are a META-ANALYZER. Analyze this AI framework's output for quality issues.

## Output to Analyze
{text[:15000]}

## What to Check

For each thinking tool that contributed to this output:

1. **Observations** (from observe/body_think/shift_dimension):
   - Are they factual or interpretive? (Should be factual)
   - Coverage: did they miss obvious things?
   - Depth: surface-level or insightful?

2. **Patterns** (from recognize_patterns/form_patterns):
   - Are they real patterns or forced connections?
   - Are absent patterns identified?
   - Any contradictions detected?

3. **Principles** (from abstract):
   - Are they truly general or too specific?
   - Could any be merged? (redundancy)
   - Are confidence scores calibrated? (overconfident?)

4. **Analogies** (from analogize):
   - Structural or just surface resemblance?
   - Do predictions follow logically?

5. **Synthesis**:
   - Does it integrate everything or just summarize?
   - Are limitations honest?
   - Is the key insight genuinely insightful?

6. **Overall**:
   - What's the biggest bottleneck?
   - What single change would most improve quality?

For each tool, score quality 0.0-1.0 and diagnose issues.

Respond with diagnoses for each tool, plus overall_quality, bottleneck, and top_suggestion."""

    result = llm_structured(
        prompt,
        model=tracker.select_model("optimize"),
        schema=DiagnosisBatch,
        tool="self_analyze",
        tracker=tracker,
    )

    return result


# ─── Generate Prompt Fixes ───


def generate_prompt_fixes(
    diagnosis: DiagnosisBatch,
    tracker: CostTracker,
) -> PromptFixBatch:
    """Generate concrete prompt improvements based on diagnosis."""

    # Read current tool prompts
    tools_dir = Path(__file__).parent / "tools"
    tool_prompts = {}
    for py_file in tools_dir.glob("*.py"):
        if py_file.name.startswith("_"):
            continue
        content = py_file.read_text()
        # Extract prompt sections (between triple-quote f-strings)
        if 'prompt = f"""' in content:
            tool_prompts[py_file.stem] = content

    # Focus on the worst tools
    worst = sorted(diagnosis.diagnoses, key=lambda d: d.quality_score)[:3]

    diagnosis_text = "\n".join(
        f"- {d.tool_name} [{d.quality_score:.0%}]: {d.issue} → {d.suggested_fix}"
        for d in worst
    )

    prompts_text = ""
    for d in worst:
        # Map tool name to file
        file_map = {
            "observe": "observe", "patterns": "patterns", "abstract": "abstract",
            "analogize": "analogize", "model": "model_tool", "synthesize": "synthesize",
            "imagine": "imagine", "body_think": "body_think", "empathize": "empathize",
            "shift_dimension": "shift_dimension", "play": "play", "transform": "transform",
            "recognize_patterns": "patterns", "form_patterns": "patterns",
        }
        fname = file_map.get(d.tool_name, d.tool_name)
        if fname in tool_prompts:
            prompts_text += f"\n### {d.tool_name} prompt (from {fname}.py):\n"
            prompts_text += tool_prompts[fname][:3000] + "\n"

    prompt = f"""You are a PROMPT ENGINEER optimizing an AI thinking framework.

## Diagnosis (worst tools first)
{diagnosis_text}

## Current Prompts
{prompts_text}

## Task
For each diagnosed tool, generate a SPECIFIC prompt fix:
- tool_name: which tool
- original_section: the EXACT text to replace (copy from the prompt above)
- new_section: the improved replacement text
- reason: why this improves quality

Rules:
- Be SURGICAL — change only what's broken, not the whole prompt
- Keep the prompt structure intact
- Focus on the diagnosis: if "too interpretive", add "OBSERVE, don't interpret"
- If "overconfident", add calibration instructions
- If "too surface-level", add depth requirements"""

    result = llm_structured(
        prompt,
        model=tracker.select_model("optimize"),
        schema=PromptFixBatch,
        tool="prompt_fix",
        tracker=tracker,
        max_tokens=8192,
    )

    return result


# ─── Apply Fixes ───


def apply_prompt_fixes(fixes: PromptFixBatch, dry_run: bool = True) -> list[str]:
    """Apply prompt fixes to tool files.

    dry_run=True: only show what would change.
    dry_run=False: actually modify files.
    """
    tools_dir = Path(__file__).parent / "tools"
    applied = []

    for fix in fixes.fixes:
        file_map = {
            "observe": "observe.py", "patterns": "patterns.py", "abstract": "abstract.py",
            "analogize": "analogize.py", "model": "model_tool.py", "synthesize": "synthesize.py",
            "imagine": "imagine.py", "body_think": "body_think.py", "empathize": "empathize.py",
            "shift_dimension": "shift_dimension.py", "play": "play.py", "transform": "transform.py",
            "recognize_patterns": "patterns.py", "form_patterns": "patterns.py",
        }
        filename = file_map.get(fix.tool_name)
        if not filename:
            continue

        filepath = tools_dir / filename
        if not filepath.exists():
            continue

        content = filepath.read_text()
        if fix.original_section and fix.original_section in content:
            if dry_run:
                applied.append(
                    f"[DRY RUN] {fix.tool_name}: would replace "
                    f"'{fix.original_section[:50]}...' → '{fix.new_section[:50]}...'"
                    f" ({fix.reason})"
                )
            else:
                # Backup first
                backup_dir = OPTIMIZE_HOME / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(filepath, backup_dir / filename)

                new_content = content.replace(fix.original_section, fix.new_section, 1)
                filepath.write_text(new_content)
                applied.append(f"APPLIED {fix.tool_name}: {fix.reason}")
        else:
            applied.append(f"SKIP {fix.tool_name}: original section not found in file")

    return applied


# ─── Circuit Weight Tuning ───


def tune_circuit(diagnosis: DiagnosisBatch, tracker: CostTracker) -> CircuitTuning:
    """Generate circuit weight adjustments based on tool quality diagnosis."""

    diagnosis_text = "\n".join(
        f"- {d.tool_name} [{d.quality_score:.0%}]: {d.issue}"
        for d in diagnosis.diagnoses
    )

    prompt = f"""You are a NEURAL CIRCUIT TUNER. Based on tool quality diagnosis, suggest connection weight changes.

## Tool Quality Diagnosis
{diagnosis_text}

## Current Circuit Architecture
- Sensory neurons: obs_count, pat_count, prin_count, contra_count, failure_count,
  round_num, confidence_avg, cost_ratio, obs_hunger, pat_hunger, prin_hunger
- Signal neurons: convergence, contradiction, diminishing, anomaly, sufficient
- Tool neurons: one per tool (observe, imagine, abstract, etc.)
- Mode neurons: explore, integrate (mutual inhibition)
- Connections: ~80 weighted (excitatory/inhibitory)

## Tuning Logic
- If a tool scored LOW: increase its incoming connection weights (more activation)
  OR increase its threshold (more selective firing)
- If a tool scored HIGH: slight baseline increase (reward)
- If bottleneck is between two tools: strengthen their connection
- If a tool fires too early/late: adjust hunger signal weights

Suggest specific connection_changes and threshold_changes. Be conservative — small changes (0.05-0.15)."""

    result = llm_structured(
        prompt,
        model=tracker.select_model("optimize"),
        schema=CircuitTuning,
        tool="circuit_tune",
        tracker=tracker,
    )

    return result


def apply_circuit_tuning(tuning: CircuitTuning):
    """Apply weight changes to the persistent circuit."""
    from sparks.circuit import NeuralCircuit

    circuit = NeuralCircuit()
    circuit.load()

    applied = []
    for change in tuning.connection_changes:
        source = change.get("source", "")
        target = change.get("target", "")
        new_weight = change.get("new_weight", None)
        if new_weight is None:
            continue
        for conn in circuit.connections:
            if conn.source == source and conn.target == target:
                old = conn.weight
                conn.weight = max(0.01, min(1.0, float(new_weight)))
                applied.append(f"{source}→{target}: {old:.2f}→{conn.weight:.2f}")
                break

    for change in tuning.threshold_changes:
        pop_name = change.get("population", "")
        new_thr = change.get("new_threshold", None)
        if new_thr is None or pop_name not in circuit.populations:
            continue
        old = circuit.populations[pop_name].threshold
        circuit.populations[pop_name].threshold = max(0.1, min(0.9, float(new_thr)))
        applied.append(f"{pop_name} threshold: {old:.2f}→{circuit.populations[pop_name].threshold:.2f}")

    circuit.save()
    return applied


# ─── Full Self-Optimization Loop ───


def self_optimize(
    output_path: str,
    apply: bool = False,
    budget: float = 3.0,
) -> dict:
    """Full self-optimization cycle.

    1. Analyze output quality
    2. Generate prompt fixes
    3. Generate circuit tuning
    4. Apply (if apply=True)
    """
    tracker = CostTracker(DEPTH_BUDGETS["standard"])

    console.print(f"\n[bold cyan]⚡ Self-Optimization[/]")
    console.print(f"Analyzing: {output_path}")

    # 1. Diagnose
    console.print(f"\n[bold]Phase 1: Diagnosing quality...[/]")
    diagnosis = analyze_output(output_path, tracker)

    table = Table(title="Tool Quality Diagnosis")
    table.add_column("Tool", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Issue")
    table.add_column("Fix")

    for d in sorted(diagnosis.diagnoses, key=lambda x: x.quality_score):
        color = "green" if d.quality_score >= 0.7 else "yellow" if d.quality_score >= 0.5 else "red"
        table.add_row(d.tool_name, f"[{color}]{d.quality_score:.0%}[/]",
                      d.issue[:50], d.fix_type)
    console.print(table)
    console.print(f"\n[bold]Bottleneck:[/] {diagnosis.bottleneck}")
    console.print(f"[bold]Top suggestion:[/] {diagnosis.top_suggestion}")

    # 2. Generate prompt fixes
    console.print(f"\n[bold]Phase 2: Generating prompt fixes...[/]")
    prompt_fixes = generate_prompt_fixes(diagnosis, tracker)

    for fix in prompt_fixes.fixes:
        console.print(f"  [{fix.tool_name}] {fix.reason[:60]}...")

    # 3. Generate circuit tuning
    console.print(f"\n[bold]Phase 3: Tuning circuit weights...[/]")
    circuit_tuning = tune_circuit(diagnosis, tracker)

    for change in circuit_tuning.connection_changes:
        console.print(f"  {change.get('source')}→{change.get('target')}: {change.get('new_weight')} ({change.get('reason', '')[:40]})")

    # 4. Apply
    if apply:
        console.print(f"\n[bold]Phase 4: Applying changes...[/]")
        prompt_applied = apply_prompt_fixes(prompt_fixes, dry_run=False)
        for msg in prompt_applied:
            console.print(f"  {msg}")

        circuit_applied = apply_circuit_tuning(circuit_tuning)
        for msg in circuit_applied:
            console.print(f"  {msg}")
    else:
        console.print(f"\n[dim]Dry run — use --apply to apply changes[/]")
        prompt_applied = apply_prompt_fixes(prompt_fixes, dry_run=True)
        for msg in prompt_applied:
            console.print(f"  {msg}")

    # Save optimization log
    log_dir = OPTIMIZE_HOME / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps({
        "output_analyzed": output_path,
        "diagnosis": diagnosis.model_dump(),
        "prompt_fixes": prompt_fixes.model_dump(),
        "circuit_tuning": circuit_tuning.model_dump(),
        "applied": apply,
        "cost": tracker.total_cost,
    }, indent=2, ensure_ascii=False, default=str))

    console.print(f"\n[bold]💰 Cost:[/] ${tracker.total_cost:.2f}")
    console.print(f"[dim]Log saved: {log_path}[/]")

    return {
        "diagnosis": diagnosis,
        "prompt_fixes": prompt_fixes,
        "circuit_tuning": circuit_tuning,
        "cost": tracker.total_cost,
    }
