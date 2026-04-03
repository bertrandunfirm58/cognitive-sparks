# ⚡ Sparks

**13 cognitive primitives that teach AI to think — not just compute.**

> Give it a goal and data. It observes, abstracts, analogizes, plays, and synthesizes — like a Renaissance polymath.

Based on [Sparks of Genius](https://en.wikipedia.org/wiki/Sparks_of_Genius) (Root-Bernstein, 1999) — the 13 thinking tools used by Einstein, Picasso, da Vinci, Feynman, and Nabokov.

---

## What is this?

Every AI agent framework teaches agents **what to do** (call tools, manage state).  
Sparks teaches agents **how to think** (observe, abstract, analogize, synthesize).

```
Input:  Goal + Data (any format)
Output: Core Principles + Evidence + Confidence + Analogies
```

### The 3-Layer Architecture

```
                         Goal + Data
                             │
                             ▼
╔═══════════════════════════════════════════════════════╗
║  LAYER 0: NERVOUS SYSTEM                              ║
║  17 biological principles                             ║
║  Senses state. Doesn't command.                       ║
║                                                       ║
║  signals: convergence | contradiction | anomaly        ║
║  ──────── accumulate → threshold → fire → reset ───── ║
║  modulators: dopamine | norepinephrine | acetylcholine ║
║  modes: sympathetic (explore) ↔ parasympathetic (integrate) ║
╠═══════════════════════════════════════════════════════╣
║  LAYER 1: 13 THINKING TOOLS (each runs autonomously)  ║
║                                                       ║
║  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐   ║
║  │ observe │ │ patterns│ │abstract │ │analogize │   ║
║  └────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘   ║
║       │           │           │            │          ║
║  ┌────┴────┐ ┌────┴────┐ ┌────┴────┐ ┌────┴─────┐   ║
║  │  model  │ │synthesize│ │  play  │ │ imagine  │   ║
║  └─────────┘ └─────────┘ └─────────┘ └──────────┘   ║
║                                                       ║
║  No arrows = no fixed order. Tools sense and decide.  ║
╠═══════════════════════════════════════════════════════╣
║  LAYER 2: AI AUGMENTATION                             ║
║  strategic forgetting | contradiction holding          ║
║  probabilistic thinking | predictive coding            ║
╚═══════════════════════════════════════════════════════╝
                             │
                             ▼
                 Core Principles + Evidence
                 + Confidence + Analogies
```

### How It's Different

```
Orchestra (LangGraph/CrewAI):     Conductor commands → musicians play in order
Harness (Claude Code):            Case protects instrument + stage manages logistics  
Nervous System (Sparks):          Musicians hear each other → self-coordinate
                                  No conductor. Resonance, not command.
```

---

## Quick Start

```bash
pip install -e .

# Analyze any data
sparks run --goal "Find the core principles" --data ./my-data/ --depth standard

# Evolution mode (AutoAgent-style: mutate → test → keep/rollback)
sparks evolve --goal "Find the core principles" --data ./my-data/ --generations 3
```

### Depth Modes

| Mode | Tools | Rounds | Cost (est.) |
|---|---|---|---|
| `quick` | 4 core | 1 | ~$0.15 |
| `standard` | 7 tools | 2 + convergence | ~$1-3 |
| `deep` | 13 tools | up to 5 | ~$5-15 |

---

## How It Works

### Phase 1: Sequential Learning (training wheels)
```
Data → Lens Bootstrap → Observe → Patterns → Abstract → Analogize → Model → Synthesize
```
The system learns to observe with a focused lens, finds patterns, abstracts them into principles, tests those principles against structural analogies, and builds a cardboard model.

### Phase 2: Iterative Deepening (Picasso Bull)
```
Round 1 results → Sleep/Consolidation → Strategic Forgetting → Re-extract → Compare
```
Like Picasso drawing the bull: start realistic, progressively remove non-essential details until only the essence remains. The system forgets what it found and re-discovers independently. What survives both rounds = the real principle.

### Phase 3: Convergence
```
Round 1 principles vs Round 2 principles → Structural comparison → Converged?
```
If both rounds found the same thing independently, it's real. If not, another round.

---

## The Nervous System (17 Biological Principles)

Not a CEO. Not an orchestrator. A **nervous system** — inspired by real neuroscience.

| Principle | Biology | Implementation |
|---|---|---|
| Threshold firing | Neurons accumulate → threshold → fire → reset | `SignalPotential`: signals accumulate, fire at threshold, refractory period |
| Excitation/Inhibition | EPSP + IPSP → net signal | Tools contribute +/- to each signal |
| Habituation | Repeated stimulus → decreased response | Repeated pattern types auto-dampen |
| Sensitization | Novel/strong stimulus → increased response | Anomaly boosts channel sensitivity |
| Reflex | Spinal cord bypasses brain | Budget exhaustion → immediate stop (no LLM) |
| Proprioception | Body senses own position | System tracks own activity, cost, momentum |
| Homeostasis | Balance excitation/inhibition | Overactive tools auto-inhibited |
| Hebbian plasticity | "Fire together, wire together" | Tool→tool connections strengthen with success |
| Distributed control | Starfish: no brain, 5 arms coordinate | Each tool has `should_run()` local rule |
| Signal summation | Spatial + temporal summation | Multiple tools contribute to one signal over time |
| Predictive coding | Brain predicts → only errors propagate up | Round 1 = prediction; Round 2 processes only surprises |
| Lateral inhibition | Retina sharpens edges via competition | Tools compete; most relevant win, rest suppressed |
| Thalamic gating | TRN gates sensory input to cortex | Observations filtered before entering state |
| Glial orchestration | Astrocytes = slow coordination channel | (Planned) Slow "mood" channel alongside fast events |
| Neuromodulation | Dopamine/NE/ACh/5-HT control learning | Reward signals adjust tool priority and exploration |
| Myelination | Frequently-used pathways speed up | Successful tool chains get better models |
| Autonomic modes | Sympathetic (fight) ↔ Parasympathetic (rest) | Anomaly → explore mode; Convergence → integrate mode |

---

## Ablation Test: Does the Nervous System Help?

Tested with Claude Opus 4.6, same data, same goal:

| Metric | Nervous ON | Nervous OFF |
|---|---|---|
| Confidence | 65% | 82% |
| Coverage | 75% | 75% |
| Cost | **$2.28** | $2.46 |
| Analogies | 4 | 6 |
| Contradictions | 0 | 1 |

**Honest result**: On this small dataset (3 files, 2 rounds), the simple pipeline produced slightly better principles. The nervous system's advantages (feedback loops, synapse learning, consolidation, cost savings) need more data and rounds to compound.

The nervous system is an investment in **long-term learning**, not a short-term boost. Like a real nervous system — you don't notice it working until it's gone.

---

## Architecture

```
sparks/
├── state.py          # CognitiveState + 17 biological signal types
├── nervous.py        # 17-principle nervous system
├── engine.py         # Pipeline orchestration
├── lens.py           # Observation lens bootstrapping
├── configurator.py   # Adaptive tool/model routing per domain
├── persistence.py    # Cross-session learning (synapses + knowledge base)
├── evolution.py      # AutoAgent-style evolution loop
├── llm.py            # Claude CLI or Anthropic API backend
├── tools/
│   ├── observe.py    # Tool #1: Observe with lens
│   ├── patterns.py   # Tool #4-5: Recognize + form patterns
│   ├── abstract.py   # Tool #3: Picasso Bull abstraction
│   ├── analogize.py  # Tool #6: Structural analogy
│   ├── model_tool.py # Tool #10: Cardboard model testing
│   └── synthesize.py # Tool #13: Final integration
└── cli.py            # sparks run / evolve / info
```

---

## Backends

### Claude Code CLI (free with subscription)
```bash
# Default — uses your Claude Code subscription, no API key needed
sparks run --goal "..." --data ./path/
```

### Anthropic API (direct)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SPARKS_BACKEND=api
sparks run --goal "..." --data ./path/
```

---

## Theoretical Foundation

### From the book
> "Phenomena are complex. Laws are simple. Find out what to discard." — Feynman

The 13 thinking tools (observing, imaging, abstracting, pattern recognition, pattern forming, analogizing, body thinking, empathizing, dimensional thinking, modeling, playing, transforming, synthesizing) are the atomic operations of creative thought — used identically by scientists, artists, and engineers across all domains.

### Our thesis
> If these 13 operations are truly universal, then applying them systematically to any dataset should extract the same depth of understanding that human experts achieve through years of experience.

### What makes this different
- **LangGraph/CrewAI**: "How to orchestrate tools" → We: "How to think with tools"
- **AutoGPT/autoresearch**: "Faster experiments" → We: "Deeper understanding"  
- **GPT-Researcher/STORM**: "Better search" → We: "Better abstraction"
- **Brain-inspired AI**: Mimics neurons for computation → We: Mimics nervous system for orchestration

---

## Status

- [x] 7/13 thinking tools implemented
- [x] 17 biological nervous system principles
- [x] Lens bootstrapping (auto domain detection)
- [x] Adaptive tool/model routing per domain
- [x] Picasso Bull abstraction method
- [x] Evolution loop (AutoAgent-style)
- [x] Cross-session learning (persistent synapses + knowledge base)
- [x] Claude Code CLI + API backends
- [ ] Remaining 6 tools (imagine, empathize, shift_dimension, play, transform, execute_and_feel)
- [ ] Convergence Level 2-3 (embeddings + LLM judgment)
- [ ] Large dataset validation (50+ files)
- [ ] Benchmark vs GPT-Researcher

---

## Origin

```
Sparks of Genius (1999) — 13 thinking tools of creative people
  +
Claude Code harness leak (2026.03.31) — the model isn't the product, the harness is
  =
"What if we taught AI to think like da Vinci, Feynman, and Picasso — simultaneously?"
```

---

*Built with [Claude Code](https://claude.ai/code). The framework that thinks about thinking.*
