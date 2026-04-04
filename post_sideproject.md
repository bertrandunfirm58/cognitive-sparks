# I taught AI the 13 thinking tools that Einstein and Picasso used — it independently discovered laws I spent months extracting manually

**TL;DR**: I made an open-source framework where AI uses the same 13 thinking tools that history's greatest minds used. It has a biological nervous system instead of a CEO orchestrator. I fed it 15 months of raw data and it independently discovered the same core laws that took me months to extract manually — plus found 4 additional laws I missed.

GitHub: [cognitive-sparks](https://github.com/PROVE1352/cognitive-sparks)

---

## The Problem

Every AI agent framework teaches agents **what to do** — call tools, manage state, follow workflows.

Nobody teaches agents **how to think**.

LangGraph, CrewAI, AutoGPT — they're all variations of "give the AI a to-do list." The AI executes steps. It doesn't *understand*. It doesn't *abstract*. It doesn't look at data and go "huh, that's structurally identical to how epidemics spread."

## The Inspiration

There's a book called **"Sparks of Genius"** (Root-Bernstein, 1999) that studied how Einstein, Picasso, da Vinci, Feynman, Nabokov, and other polymaths actually think. They found **13 cognitive operations** that every creative genius uses — across all domains:

1. **Observing** — see what others miss
2. **Imaging** — run mental simulations
3. **Abstracting** — strip away noise until only essence remains
4. **Recognizing Patterns** — find what repeats
5. **Forming Patterns** — combine patterns into new ones
6. **Analogizing** — structural correspondence, not surface similarity
7. **Body Thinking** — feel the data physically
8. **Empathizing** — become the actors inside the data
9. **Dimensional Thinking** — change the axis, change the truth
10. **Modeling** — build it rough, see what breaks
11. **Playing** — break rules systematically
12. **Transforming** — convert between representations
13. **Synthesizing** — everything resonates at once

A physicist uses the same 13 tools as a sculptor. The *content* differs. The *thinking* is identical.

**My thesis**: If these 13 operations are truly universal, we can implement them as AI primitives. Feed data in, get deep understanding out — not just summaries, but actual *principles*.

## The Architecture: A Nervous System, Not a CEO

Here's where it gets interesting. Most agent frameworks use a "CEO pattern" — one orchestrator decides what to do next. That's how corporations work, not how intelligence works.

Sparks uses a **biological nervous system** with 17 neuroscience principles:

```
NOT THIS:                          THIS:
  CEO (orchestrator)                 Nervous System
    ↓ commands                       ↕ senses
  Worker 1 → Worker 2              Tool ←→ Tool ←→ Tool
    ↓ commands                       each senses state
  Worker 3 → Worker 4              each decides locally
```

**No conductor. The musicians hear each other and self-coordinate.**

The 17 principles include:
- **Threshold firing**: Signals accumulate → threshold → fire → refractory period (not boolean flags)
- **Hebbian plasticity**: "Fire together, wire together" — tool chains that work get strengthened
- **Lateral inhibition**: Tools compete; the most relevant ones suppress the rest
- **Habituation**: Repeated patterns auto-dampen (stops the AI from fixating)
- **Predictive coding**: Round 1 = predictions. Round 2 only processes *surprises*
- **Autonomic modes**: Sympathetic (explore/diverge) ↔ Parasympathetic (integrate/converge)
- **Neuromodulation**: Dopamine/norepinephrine/acetylcholine adjust learning dynamics

Each tool has its own `should_run()` — like a ganglion making autonomous decisions based on local conditions. No central brain commands them.

## The Abstraction Method: Picasso's Bull

The core of the framework is the **Picasso Bull method** for abstraction:

Picasso drew a bull realistically, then made 11 progressively simpler versions. Each one removed details until only a few essential lines remained — capturing the *essence* of a bull.

Sparks does the same thing with data:

```
Round 1: 263 observations → 27 patterns → 7 principles
Round 2: Strategic forgetting → Re-extract → Compare with Round 1
Convergence: What survives both rounds independently = real principle
```

"Phenomena are complex. Laws are simple. Find out what to discard." — Feynman

## The Pipeline

```
Data → Lens Bootstrap → 13 Tools (Phase 1) → Sleep/Consolidation → 13 Tools (Phase 2) → Convergence
```

**Phase 1** (Training Wheels): All 13 tools run sequentially. Observe → Body Think → Shift Dimension → Patterns → Empathize → Abstract → Analogize → Imagine → Model → Play → Transform → Synthesize.

**Consolidation** (Sleep): Prune noise, merge duplicates, strengthen successful tool chains. Like actual sleep consolidation in neuroscience.

**Phase 2** (Eyes Fresh): Strategic forgetting. The system forgets its derived principles and re-extracts from observations. What it finds again independently = real signal.

**Convergence**: Round 1 principles vs Round 2 principles. Structural comparison. If both rounds found the same thing independently, it's real.

## Depth Modes

| Mode | Tools | Rounds | Budget |
|---|---|---|---|
| `quick` | 4 core tools | 1 | ~$0.15 |
| `standard` | 7 tools | 2 + convergence | ~$1-5 |
| `deep` | **13 tools** | up to 5 | ~$5-20 |

## The Validation: A-Test

I had 15 months of densely analyzed market observation data (640K chars, 263 observations across 6 analysis files). Over those months, I had manually extracted what I called "3 Core Laws" — fundamental principles governing market behavior.

I fed the raw observation data to Sparks and asked: *"Extract the fundamental laws governing this data."*

**Sparks had zero knowledge of my 3 Laws. No hints. Just raw data.**

### Results

| Metric | Value |
|---|---|
| Observations extracted | 263 |
| Patterns found | 27 |
| Principles extracted | 7 |
| Structural analogies | 8 |
| Unresolved contradictions | 2 |
| Model accuracy (cardboard) | 58% |
| Cost | $6.06 |

**The top 3 principles structurally matched my manually-extracted 3 Laws:**

- My "Cognitive Constraint" law (one thing at a time, reversal at extremes) → Sparks Principle 1: "Markets process one macro variable at a time in sequential 4-8 week dominance windows" (98% confidence)

- My "Physical Constraint" law (tangible controls intangible) → Sparks Principle 3: "Extreme-low volatility states serve as ignition states enabling rapid phase transitions" (94% confidence)

- My "Information Asymmetry" law (early knowers → late knowers) → Sparks Principle 2: "Three Core Laws activate sequentially as bottleneck migrates through system" (71% confidence)

**Plus 4 additional principles I hadn't formalized:**
- Dual-timescale dominance (institutional 6-8 weeks vs retail 2-3 weeks)
- Regime-dependent processing speed (fear = slow, FOMO = fast)
- Narrative compression within dominance windows
- 48-hour reinterpretation cycles

The framework independently discovered what took me months — and found structure I missed.

## What Each Tool Actually Does

Here's what the 6 new tools (beyond the standard observe/pattern/abstract pipeline) bring:

**Body Think** — "Feel" the data. What's heavy? What's hot? Where's the rhythm? Where's the tension? Sounds weird for AI, but it produces observations that pure analysis misses. It caught "gravitational mass" of certain themes that pattern recognition overlooked.

**Empathize** — Become the actors. "You are a dying pattern. What killed you?" "You are the constraint. What are you holding back?" Generates insights invisible from the outside.

**Shift Dimension** — Same data, different axis. Time→frequency. Micro→macro. Positive→negative. Each shift reveals what the default view hides.

**Imagine** — "What if this principle operated at 10x? What if the opposite were true?" Mental simulation that stress-tests principles.

**Play** — Systematic rule-breaking. Remove a principle — does anything improve? Invert it — is the inverted world consistent? If yes, your principle is weak.

**Transform** — Convert between representations. Narrative→equation. Static→dynamic. Parts→whole. Every transformation is a new lens.

## Technical Details

```
cognitive-sparks/
├── state.py          # CognitiveState + biological signals
├── nervous.py        # 17-principle nervous system
├── engine.py         # Pipeline orchestration
├── lens.py           # Auto domain detection + lens generation
├── configurator.py   # Adaptive tool/model routing
├── persistence.py    # Cross-session learning (synapses persist!)
├── evolution.py      # AutoAgent-style mutate→test→keep
├── llm.py            # Claude Code CLI or Anthropic API
├── tools/            # All 13 tools
│   ├── observe.py, imagine.py, abstract.py, patterns.py,
│   ├── analogize.py, body_think.py, empathize.py,
│   ├── shift_dimension.py, model_tool.py, play.py,
│   ├── transform.py, synthesize.py
└── cli.py            # sparks run / evolve / info
```

**Backends**: Works with Claude Code CLI (free with subscription) or Anthropic API directly. No API key needed for CLI mode.

**Cross-session learning**: Synapses (tool→tool connection strengths) persist across sessions via JSON. The framework literally gets smarter with every use.

**Adaptive routing**: Different models for different tools. Observe uses cheaper models (bulk work). Abstract/Synthesize use expensive models (deep reasoning). All configurable.

## Usage

```bash
pip install -e .

# Quick analysis
sparks run --goal "Find the core principles" --data ./your-data/ --depth quick

# Full 13-tool deep analysis
sparks run --goal "What governs this system?" --data ./your-data/ --depth deep

# Evolution mode (mutate config → test → keep best)
sparks evolve --goal "..." --data ./your-data/ --generations 5
```

## What I Learned Building This

1. **The nervous system metaphor is surprisingly powerful.** Threshold firing, habituation, and lateral inhibition solve real orchestration problems that "CEO pattern" frameworks struggle with.

2. **Predictive coding is underrated.** Round 1 extracts. Round 2 only processes *surprises*. This is how real brains work and it consistently produces deeper principles than two identical passes.

3. **The "useless" tools are the most valuable.** Body thinking and empathy sound ridiculous for AI. But they consistently catch things that pattern recognition and abstraction miss — because they force a different *mode* of engagement with the data.

4. **Abstraction is the bottleneck, not observation.** Every AI can observe. Very few can abstract well. The Picasso Bull method (start concrete, progressively remove) works dramatically better than "summarize the key points."

5. **Sleep matters.** The consolidation step between rounds — pruning noise, strengthening connections, resetting habituation — consistently improves Round 2 output. Just like real sleep consolidation.

## Limitations (honest)

- **Cost**: Deep mode with Opus runs $5-20. Not cheap.
- **Speed**: 13 tools × Opus = 30-60 minutes for a deep analysis.
- **Small data**: The nervous system's advantages compound over multiple rounds. On tiny datasets (3-5 files), a simple pipeline does fine.
- **No embeddings yet**: Convergence detection uses word overlap, not semantic similarity.
- **Single-LLM**: Currently Claude-only. Multi-model support planned.

## What's Next

- Convergence Level 2-3 (embedding-based + LLM judgment)
- Large dataset validation (50+ files)
- Benchmark against GPT-Researcher, STORM
- Multi-model support (GPT-4, Gemini)

---

**GitHub**: [PROVE1352/cognitive-sparks](https://github.com/PROVE1352/cognitive-sparks)

Built with Claude Code. The framework that thinks about thinking.

Happy to answer questions about the architecture, the neuroscience mapping, or the A-test results.
