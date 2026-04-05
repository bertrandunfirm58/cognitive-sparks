"""CLI — sparks run --goal "..." --data ./path/ --depth standard"""

from __future__ import annotations

from pathlib import Path

import typer
from typing import Optional
from rich.console import Console

app = typer.Typer(
    name="sparks",
    help="13 cognitive primitives that teach AI to think — not just compute.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    goal: str = typer.Option(..., "--goal", "-g", help="What to find in the data"),
    data: str = typer.Option(..., "--data", "-d", help="Path to data directory or file"),
    depth: str = typer.Option("standard", "--depth", help="quick / standard / deep"),
    output: str = typer.Option("", "--output", "-o", help="Output file path (default: stdout + ./output/)"),
    no_nervous: bool = typer.Option(False, "--no-nervous", help="Ablation: disable biological nervous system"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility"),
    reset_weights: bool = typer.Option(False, "--reset-weights", help="Reset circuit weights to initial values"),
):
    """Analyze data using 13 cognitive thinking tools."""
    from sparks.engine import run as engine_run
    from sparks.output import format_output

    if seed is not None:
        from sparks.research import set_seed
        set_seed(seed)
        console.print(f"[dim]Seed: {seed}[/]")

    if reset_weights:
        from sparks.circuit import NeuralCircuit
        c = NeuralCircuit()
        c.reset()
        c.save()
        console.print("[yellow]Circuit weights reset to initial values[/]")

    if depth not in ("quick", "standard", "deep"):
        console.print(f"[red]Invalid depth: {depth}. Use quick/standard/deep.[/]")
        raise typer.Exit(1)

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Cognitive Analysis")
    console.print(f"[bold]Goal:[/] {goal}")
    console.print(f"[bold]Depth:[/] {depth}")

    if not no_nervous:
        # Autonomic mode: circuit-driven cascade (no TOOL_ORDER)
        from sparks.autonomic import run_autonomic
        result = run_autonomic(goal=goal, data_path=str(data_path), depth=depth)
    else:
        result = engine_run(goal=goal, data_path=str(data_path), depth=depth, nervous_system=False)
    md = format_output(result, goal)

    # Save to file
    out_path = output or "output/results.md"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md)
    console.print(f"\n[bold green]📄 Results saved to {out_file}[/]")

    # Print key results
    console.print(f"\n{'='*60}")
    for i, p in enumerate(result.principles, 1):
        console.print(f"[bold]{i}. [{p.confidence:.0%}][/] {p.statement}")
    console.print(f"{'='*60}")


@app.command()
def evolve(
    goal: str = typer.Option(..., "--goal", "-g", help="What to find in the data"),
    data: str = typer.Option(..., "--data", "-d", help="Path to data directory or file"),
    generations: int = typer.Option(3, "--generations", "-n", help="Evolution generations"),
    output: str = typer.Option("", "--output", "-o", help="Output file path"),
):
    """Evolve principles through multiple generations (AutoAgent-style)."""
    from sparks.evolution import EvolutionLoop
    from sparks.output import format_output

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Evolution Mode")

    loop = EvolutionLoop(goal=goal, data_path=str(data_path), max_generations=generations)
    result = loop.run()
    loop.save_history()

    md = format_output(result, goal)
    out_path = output or "output/evolved_results.md"
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(md)
    console.print(f"\n[bold green]📄 Evolved results saved to {out_file}[/]")

    console.print(f"\n{'='*60}")
    for i, p in enumerate(result.principles, 1):
        console.print(f"[bold]{i}. [{p.confidence:.0%}][/] {p.statement}")
    console.print(f"{'='*60}")


@app.command()
def loop(
    principles: str = typer.Option(..., "--principles", "-p", help="Path to .md output or store name"),
    data: str = typer.Option("", "--data", "-d", help="New data directory for validation"),
    cycles: int = typer.Option(1, "--cycles", "-n", help="B→C validation cycles"),
    budget: float = typer.Option(10.0, "--budget", "-b", help="Max cost in dollars"),
    predict: str = typer.Option("", "--predict", help="Situation to generate predictions for"),
    outcomes: str = typer.Option("", "--outcomes", help="Actual outcomes for feedback"),
):
    """Full loop: Validate → Evolve → Predict → Feedback."""
    from sparks.loop import run_loop

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Full Loop (B→F)")

    data_dirs = [data] if data else []
    run_loop(
        principles_source=principles,
        data_dirs=data_dirs,
        cycles=cycles,
        budget=budget,
        predict_input=predict,
        outcomes=outcomes,
    )


@app.command()
def bench(
    goal: str = typer.Option(..., "--goal", "-g", help="What to find in the data"),
    data: str = typer.Option(..., "--data", "-d", help="Path to data directory"),
    runs: int = typer.Option(5, "--runs", "-n", help="Number of runs"),
    depth: str = typer.Option("standard", "--depth", help="quick/standard/deep"),
    output: str = typer.Option("", "--output", "-o", help="Output path for results"),
):
    """Benchmark: run N times, compute reproducibility statistics."""
    from sparks.research import benchmark, format_benchmark
    import json

    console.print(f"\n[bold cyan]⚡ Sparks[/] — Benchmark ({runs} runs)")
    stats = benchmark(goal=goal, data_path=data, n_runs=runs, depth=depth)

    md = format_benchmark(stats)
    console.print(md)

    out_path = output or "output/benchmark.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
    console.print(f"\n[bold green]Saved to {out_path}[/]")


@app.command()
def export(
    output_md: str = typer.Option(..., "--input", "-i", help="Path to Sparks output .md"),
    fmt: str = typer.Option("latex", "--format", "-f", help="latex / notebook"),
    out: str = typer.Option("", "--output", "-o", help="Output file path"),
):
    """Export results as LaTeX table or Jupyter notebook."""
    # Note: for full evidence chains, we need the state (not just output.md)
    # This exports from the output.md with limited evidence
    from sparks.loop import PrincipleStore

    store = PrincipleStore("export_temp")
    n = store.load_from_output(output_md)

    if fmt == "latex":
        from sparks.state import SynthesisOutput, Principle
        result = SynthesisOutput(
            principles=[Principle(id=f"p{i}", statement=p["statement"],
                                  confidence=p["confidence"])
                        for i, p in enumerate(store.principles)],
        )
        from sparks.research import to_latex_table
        latex = to_latex_table(result)
        out_path = out or "output/principles.tex"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(latex)
        console.print(f"[bold green]LaTeX saved to {out_path}[/]")
        console.print(latex)
    elif fmt == "notebook":
        console.print("[yellow]Notebook export requires full state (run with --export-notebook flag)[/]")
    else:
        console.print(f"[red]Unknown format: {fmt}. Use latex or notebook.[/]")


@app.command()
def optimize(
    output: str = typer.Option(..., "--output", "-o", help="Path to analysis output (.md) to optimize from"),
    apply: bool = typer.Option(False, "--apply", help="Actually apply changes (default: dry run)"),
):
    """Self-optimize: analyze output quality, fix prompts, tune circuit."""
    from sparks.self_optimize import self_optimize
    self_optimize(output_path=output, apply=apply)


@app.command()
def wiki_ingest(
    data: str = typer.Option(..., "--data", "-d", help="Data path to ingest"),
    goal: str = typer.Option("Extract and organize knowledge", "--goal", "-g", help="Analysis goal"),
    wiki: str = typer.Option("", "--wiki", "-w", help="Wiki path (default: ~/.sparks/wiki/default)"),
    depth: str = typer.Option("quick", "--depth", help="Analysis depth"),
    raw: bool = typer.Option(False, "--raw", help="Ingest as raw text (no Sparks analysis)"),
):
    """Ingest data into the wiki knowledge base."""
    from sparks.wiki import Wiki

    wiki_path = wiki or str(Path.home() / ".sparks" / "wiki" / "default")
    w = Wiki(wiki_path)

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]⚡ Wiki Ingest[/]")
    console.print(f"Wiki: {wiki_path}")
    console.print(f"Data: {data}")

    if raw:
        text = data_path.read_text() if data_path.is_file() else "\n".join(
            f.read_text() for f in sorted(data_path.glob("*")) if f.is_file()
        )
        result = w.ingest_text(text, source=str(data_path), goal=goal)
    else:
        result = w.ingest(data_path=str(data_path), goal=goal, depth=depth)

    console.print(f"\n[green]✓ Updated: {result.get('updated', 0)} pages[/]")
    console.print(f"[green]✓ Created: {result.get('created', 0)} pages[/]")
    stats = w.stats()
    console.print(f"[dim]Total: {stats['pages']} pages, {stats['total_chars']:,} chars[/]")


@app.command()
def wiki_query(
    question: str = typer.Option(..., "--question", "-q", help="Question to ask the wiki"),
    wiki: str = typer.Option("", "--wiki", "-w", help="Wiki path"),
    save: bool = typer.Option(False, "--save", help="Save answer as wiki page"),
):
    """Query the wiki knowledge base."""
    from sparks.wiki import Wiki

    wiki_path = wiki or str(Path.home() / ".sparks" / "wiki" / "default")
    w = Wiki(wiki_path)

    console.print(f"\n[bold cyan]⚡ Wiki Query[/]")
    result = w.query(question, file_result=save)

    console.print(f"\n{result.answer}")
    if result.sources:
        console.print(f"\n[dim]Sources: {', '.join(result.sources)}[/]")
    if result.filed_as:
        console.print(f"[green]Filed as: {result.filed_as}[/]")


@app.command()
def wiki_lint(
    wiki: str = typer.Option("", "--wiki", "-w", help="Wiki path"),
):
    """Health check the wiki knowledge base."""
    from sparks.wiki import Wiki

    wiki_path = wiki or str(Path.home() / ".sparks" / "wiki" / "default")
    w = Wiki(wiki_path)

    console.print(f"\n[bold cyan]⚡ Wiki Lint[/]")
    result = w.lint()

    if result.contradictions:
        console.print(f"\n[red]Contradictions ({len(result.contradictions)}):[/]")
        for c in result.contradictions:
            console.print(f"  {c.get('page_a', '?')} vs {c.get('page_b', '?')}: {c.get('issue', '?')}")
    if result.orphan_pages:
        console.print(f"\n[yellow]Orphan pages: {', '.join(result.orphan_pages)}[/]")
    if result.stale_pages:
        console.print(f"\n[yellow]Stale pages: {', '.join(result.stale_pages)}[/]")
    if result.missing_links:
        console.print(f"\n[yellow]Missing links:[/]")
        for m in result.missing_links:
            console.print(f"  {m['from_page']} → [[{m['broken_link']}]]")

    for s in result.suggestions:
        console.print(f"\n[dim]💡 {s}[/]")


@app.command()
def wiki_stats(
    wiki: str = typer.Option("", "--wiki", "-w", help="Wiki path"),
):
    """Show wiki statistics."""
    from sparks.wiki import Wiki

    wiki_path = wiki or str(Path.home() / ".sparks" / "wiki" / "default")
    w = Wiki(wiki_path)
    stats = w.stats()

    console.print(f"\n[bold cyan]⚡ Wiki Stats[/]")
    console.print(f"Path: {stats['wiki_path']}")
    console.print(f"Pages: {stats['pages']}")
    console.print(f"Characters: {stats['total_chars']:,}")
    console.print(f"Categories: {stats['categories']}")

    pages = w.list_pages()
    if pages:
        console.print(f"\nPages:")
        for p in pages:
            console.print(f"  - {p}")


@app.command()
def meta(
    data: str = typer.Option(..., "--data", "-d", help="Benchmark data path"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Max improvement iterations"),
    depth: str = typer.Option("quick", "--depth", help="Benchmark depth"),
    apply: bool = typer.Option(False, "--apply", help="Actually apply changes (default: dry run)"),
    risk: str = typer.Option("low", "--risk", help="Max risk level: low/medium/high"),
):
    """Self-improvement loop: analyze own code → patch → benchmark → keep/rollback."""
    from sparks.meta import meta_loop

    data_path = Path(data)
    if not data_path.exists():
        console.print(f"[red]Data path not found: {data}[/]")
        raise typer.Exit(1)

    meta_loop(
        benchmark_data=str(data_path),
        max_iterations=iterations,
        depth=depth,
        apply=apply,
        max_risk=risk,
    )


@app.command()
def trace(
    path: str = typer.Option("", "--path", "-p", help="Path to trace JSON (default: latest)"),
):
    """View the explainability trace of a cascade run."""
    import json

    if path:
        trace_path = Path(path)
    else:
        # Find latest trace
        trace_dir = Path.home() / ".sparks" / "traces"
        if not trace_dir.exists():
            console.print("[red]No traces found. Run 'sparks run' first.[/]")
            raise typer.Exit(1)
        traces = sorted(trace_dir.glob("trace_*.json"), key=lambda p: p.stat().st_mtime)
        if not traces:
            console.print("[red]No traces found.[/]")
            raise typer.Exit(1)
        trace_path = traces[-1]

    data = json.loads(trace_path.read_text())
    console.print(f"\n[bold cyan]⚡ Cascade Trace[/] ({trace_path.name})")
    console.print(f"Total firings: {data['total_firings']} | Terminated: {data['termination']}\n")

    for f in data["firings"]:
        mode_icon = {"sympathetic": "🔴", "parasympathetic": "🔵", "balanced": "⚪"}.get(f.get("mode", ""), "⚪")
        console.print(f"{mode_icon} [{f['step']:2d}] [bold]{f['tool']}[/] (act={f['activation']:.2f}, {f['mode']})")
        console.print(f"     {f['summary']}")
        drivers = ", ".join(f"{d['source']}({d['contribution']:+.2f})" for d in f["top_drivers"][:4])
        console.print(f"     [dim]drivers: {drivers}[/]")
        if f.get("runner_up"):
            console.print(f"     [dim]runner-up: {f['runner_up']}[/]")
        console.print()

    if data.get("consolidation_at"):
        console.print(f"[dim]Consolidation at steps: {data['consolidation_at']}[/]")


@app.command()
def info():
    """Show framework info."""
    from sparks import __version__
    console.print(f"[bold cyan]⚡ Sparks[/] v{__version__}")
    console.print("13 cognitive primitives for deep understanding.")
    console.print("\nBased on: Sparks of Genius (Root-Bernstein, 1999)")
    console.print("Architecture: 3-Layer Cognitive Harness")
    console.print("  Layer 0: Nervous System (sense, don't command)")
    console.print("  Layer 1: 13 Thinking Tools (observe → synthesize)")
    console.print("  Layer 2: AI Augmentation (forget, hold contradictions)")


if __name__ == "__main__":
    app()
