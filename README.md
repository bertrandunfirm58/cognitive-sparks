# Sparks

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
                             |
                             v
+=========================================================+
|  LAYER 0: NEURAL CIRCUIT                                 |
|  ~30 neural populations, ~80+ learned connections        |
|  Rate-coded LIF neurons + STDP + neuromodulation         |
|                                                          |
|  sensory -> [weights] -> signal -> [weights] -> tool     |
|     ^                                          |         |
|     +---- state encodes as sensory input <-----+         |
|                                                          |
|  No fixed order. Tool execution ORDER EMERGES            |
|  from connection weights and activation dynamics.        |
+=========================================================+
|  LAYER 1: 13 THINKING TOOLS (fire when activated)        |
|                                                          |
|  observe  imagine  body_think  empathize                 |
|  shift_dimension  recognize_patterns  form_patterns      |
|  abstract  analogize  model  play  transform             |
|  synthesize                                              |
|                                                          |
|  Each tool fires when its neural population crosses      |
|  threshold. No conductor. No TOOL_ORDER.                 |
+=========================================================+
|  LAYER 2: AI AUGMENTATION                                |
|  strategic forgetting | contradiction holding            |
|  predictive coding | cross-session learning              |
+=========================================================+
                             |
                             v
                 Core Principles + Evidence
                 + Confidence + Analogies
```

### How It's Different

```
Orchestra (LangGraph/CrewAI):     Conductor commands -> musicians play in order
Harness (Claude Code):            Case protects instrument + stage manages logistics
Nervous System (Sparks):          No conductor. No fixed order.
                                  Tool execution emerges from neural dynamics.
                                  Connection weights learned through STDP.
```

---

## Quick Start

```bash
git clone https://github.com/PROVE1352/cognitive-sparks.git
cd cognitive-sparks
pip install -e .

# Analyze data (autonomic cascade — circuit decides tool order)
sparks run --goal "Find the core principles" --data ./my-data/ --depth standard

# Full loop: validate -> evolve -> predict -> feedback
sparks loop --principles output/results.md --data ./new-data/ --cycles 3

# Ablation: disable nervous system (sequential pipeline)
sparks run --goal "..." --data ./my-data/ --no-nervous
```

### Depth Modes

| Mode | Tools | Execution | Budget |
|---|---|---|---|
| `quick` | 4 core | 1 cascade | ~$0.15 |
| `standard` | 7 tools | cascade + consolidation + re-cascade | ~$2-6 |
| `deep` | **13 tools** | full cascade + consolidation | ~$5-15 |

---

## How It Works: Autonomic Cascade

Unlike traditional agent frameworks that loop through tools in a fixed order, Sparks uses a **pulse-driven cascade**:

```
1. State encoded as sensory input (obs_count, hunger signals, etc.)
2. Neural circuit runs ~5 ticks (signals propagate through connections)
3. Tool population with HIGHEST activation above threshold -> FIRES
4. Tool executes (LLM call) -> state mutates
5. New state -> new sensory input -> circuit re-evaluates
6. Repeat until: no tool above threshold OR sufficient_depth fires OR budget exhausted
```

**No hardcoded tool order.** The sequence `observe -> patterns -> abstract -> analogize` emerges naturally because:
- Empty state -> `obs_hunger=1.0` drives observe neuron highest
- After observe -> `obs_count` rises, drives recognize_patterns
- After patterns -> `pat_count` rises, drives abstract
- After principles -> synthesize gets strongest input

This sequence is learned and evolves via STDP across sessions.

### Consolidation (Sleep)

When no tool crosses threshold (cascade exhausts), the system "sleeps":
- Prune low-confidence observations
- Merge duplicate patterns
- Reset tool populations to baseline
- Boost norepinephrine (wake up refreshed, explore mode)
- Clear derived state, keep observations
- Re-ignite cascade from fresh perspective

---

## The Neural Circuit

Not if-else rules pretending to be neurons. Actual neural dynamics.

### Neuron Model: Rate-coded Leaky Integrate-and-Fire

```
tau * dr/dt = -(r - baseline) + I_total * gain + noise
if r > threshold: fire, enter refractory period
```

### Architecture

| Layer | Populations | Role |
|---|---|---|
| Sensory | 11 (obs_count, pat_count, hunger signals, etc.) | Encode state as neural input |
| Signal | 5 (convergence, contradiction, diminishing, anomaly, sufficient) | Integrate evidence |
| Tool | 13 (one per thinking tool) | Activation -> tool fires |
| Mode | 2 (explore / integrate, mutual inhibition) | Global behavioral mode |

### Learning

| Mechanism | Biology | Implementation |
|---|---|---|
| **STDP** | Spike-timing dependent plasticity | Co-firing tools strengthen connections; timing matters |
| **Reward modulation** | Dopamine gates plasticity | Tool success/failure modulates STDP via dopamine signal |
| **Homeostatic plasticity** | Neurons maintain target firing rate | Over-active populations shrink incoming weights |
| **Neuromodulation** | DA/NE/ACh adjust gain globally | Dopamine (reward), NE (arousal), ACh (learning rate) |
| **Hunger signals** | Absence = signal (like hunger) | Missing data types drive relevant tools (obs_hunger -> observe) |
| **Persistence** | Long-term memory | Weights + baselines saved to JSON, evolve across sessions |

### Emergent Behavior (no rules, just weights)

Tested with empty -> populated state transitions:

| State | Highest tool | Activation | Mode |
|---|---|---|---|
| Empty (no data) | **observe** | 0.59 | balanced |
| 20 observations | **recognize_patterns** | 0.52 | balanced |
| + 8 patterns | **abstract** | 0.55 | balanced |
| + 4 principles | **analogize** | 0.43 | parasympathetic |

The pipeline `observe -> patterns -> abstract -> analogize -> synthesize` emerges from dynamics alone.

---

## The 13 Thinking Tools

| # | Tool | Inspiration | What it does |
|---|---|---|---|
| 1 | **observe** | Observation with focused lens | Extract observations through domain-specific channels |
| 2 | **imagine** | Mental simulation | "What if X at 10x? What if opposite?" |
| 3 | **abstract** | Picasso Bull reduction | Progressively strip non-essential until essence remains |
| 4 | **recognize_patterns** | Pattern recognition | Find recurring, absent, interference patterns |
| 5 | **form_patterns** | Pattern composition | Combine patterns -> emergence (moire effect) |
| 6 | **analogize** | Structural analogy | Find deep structural correspondence across domains |
| 7 | **body_think** | Body thinking | Feel the data: weight, texture, rhythm, tension |
| 8 | **empathize** | Empathy | Become the actors inside the data |
| 9 | **shift_dimension** | Dimensional thinking | Same data, different axis: time, scale, frequency |
| 10 | **model** | Cardboard model | Build rough model, see what breaks |
| 11 | **play** | Systematic rule-breaking | Remove/invert/exaggerate principles |
| 12 | **transform** | Representation change | Narrative->equation, static->dynamic |
| 13 | **synthesize** | Simultaneous integration | All tools resonate at once |

---

## Full Loop (Phase B-F)

Phase A (above) extracts principles. The full loop validates and evolves them:

```
Phase A: Raw Data -> 13 Tools -> Principles (sparks run)
Phase B: Principles + New Data -> Validation scores
Phase C: Weak principles refined/dropped, strong ones kept
Phase D: Validated principles -> Predictions (sparks loop --predict)
Phase E: Predictions vs outcomes -> Feedback (sparks loop --outcomes)
Phase F: Auto-cycle B->E on each new data batch
```

```bash
# Validate + evolve principles against new data
sparks loop -p output/results.md -d ./new-data/ --cycles 3

# Generate predictions
sparks loop -p output/results.md --predict "Next week: Fed meeting + tariff deadline"

# Feedback: compare predictions to reality
sparks loop -p output/results.md --predict "..." --outcomes "What actually happened: ..."
```

Principles are persistent (`~/.sparks/loop/`) and evolve across sessions.

---

## Validation: A-Test

Fed Sparks 15 months of market observation data (640K chars) to test if it independently discovers the same core laws that human experts extracted manually over months.

| Mode | Principles | Avg Confidence | Coverage | Cost |
|---|---|---|---|---|
| standard (7 tools) | 7 | 80% | 68% | $6.06 |
| **deep (13 tools)** | **12** | **91%** | **85%** | **$9.02** |

The deep analysis found 12 principles. The top 3 structurally matched manually-extracted laws, plus 9 additional principles the manual analysis hadn't formalized. The 6 new tools (imagine, body_think, empathize, shift_dimension, play, transform) contributed 5 of those additional principles.

---

## Architecture

```
sparks/
├── circuit.py        # Neural circuit (~30 LIF populations, STDP, neuromodulation)
├── autonomic.py      # Pulse-driven cascade execution (no TOOL_ORDER)
├── engine.py         # Legacy sequential pipeline (--no-nervous fallback)
├── state.py          # CognitiveState + biological signal types
├── nervous.py        # Signal computation (if-else fallback, kept for ablation)
├── loop.py           # Full loop: validate -> evolve -> predict -> feedback
├── lens.py           # Observation lens bootstrapping
├── configurator.py   # Adaptive tool/model routing per domain
├── persistence.py    # Cross-session learning (synapses + knowledge base)
├── evolution.py      # AutoAgent-style evolution loop
├── llm.py            # Claude Code CLI or Anthropic API backend
├── cost.py           # Model routing + budget tracking
├── context.py        # Per-tool context assembly
├── data.py           # Data loading + chunking
├── events.py         # Event bus
├── output.py         # Markdown output formatter
├── tools/
│   ├── observe.py        # #1: Observe with lens
│   ├── imagine.py        # #2: Mental simulation
│   ├── abstract.py       # #3: Picasso Bull abstraction
│   ├── patterns.py       # #4-5: Recognize + form patterns
│   ├── analogize.py      # #6: Structural analogy
│   ├── body_think.py     # #7: Feel the data physically
│   ├── empathize.py      # #8: Become the actors inside
│   ├── shift_dimension.py# #9: Change the axis
│   ├── model_tool.py     # #10: Cardboard model testing
│   ├── play.py           # #11: Break rules systematically
│   ├── transform.py      # #12: Convert between forms
│   └── synthesize.py     # #13: Final integration
└── cli.py            # sparks run / loop / evolve / info
```

---

## Backends

### Claude Code CLI (free with subscription)
```bash
sparks run --goal "..." --data ./path/
```

### Anthropic API
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export SPARKS_BACKEND=api
sparks run --goal "..." --data ./path/
```

---

## Theoretical Foundation

> "Phenomena are complex. Laws are simple. Find out what to discard." — Feynman

The 13 thinking tools are the atomic operations of creative thought — used identically by scientists, artists, and engineers across all domains. If these operations are truly universal, applying them systematically to any dataset should extract the same depth of understanding that human experts achieve through years of experience.

### What makes this different
- **LangGraph/CrewAI**: "How to orchestrate tools" -> We: "How to think with tools"
- **AutoGPT/autoresearch**: "Faster experiments" -> We: "Deeper understanding"
- **GPT-Researcher/STORM**: "Better search" -> We: "Better abstraction"
- **Brain-inspired AI**: Mimics neurons for computation -> We: Mimics nervous system for orchestration

---

## Status

- [x] 13/13 thinking tools
- [x] Neural circuit (rate-coded LIF, STDP, neuromodulation)
- [x] Autonomic cascade (emergent tool ordering)
- [x] Full loop (validate -> evolve -> predict -> feedback)
- [x] Lens bootstrapping + adaptive routing
- [x] Cross-session learning (persistent weights)
- [x] Claude Code CLI + API backends
- [x] A-Test validation (12 principles, 91% avg confidence)
- [ ] Embedding-based convergence detection
- [ ] Multi-model support (GPT-4, Gemini)
- [ ] Large dataset benchmarks

---

*Built with [Claude Code](https://claude.ai/code). The framework that thinks about thinking.*
