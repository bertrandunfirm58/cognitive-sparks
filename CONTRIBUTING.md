# Contributing to Sparks

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/sparks.git
cd sparks
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running

```bash
# With Claude Code CLI (free, no API key)
sparks run --goal "test" --data ./demo/claude_code_posts/ --depth quick

# With Anthropic API
export ANTHROPIC_API_KEY=sk-ant-...
export SPARKS_BACKEND=api
sparks run --goal "test" --data ./demo/claude_code_posts/ --depth standard
```

## Project Structure

```
src/sparks/
├── state.py          # Core data models (start here)
├── nervous.py        # 17-principle nervous system
├── engine.py         # Pipeline orchestration
├── tools/            # 7 thinking tools (add new ones here)
│   ├── base.py       # BaseTool interface
│   ├── observe.py    
│   ├── patterns.py   
│   ├── abstract.py   # Picasso Bull method
│   ├── analogize.py  
│   ├── model_tool.py 
│   └── synthesize.py 
├── lens.py           # Observation lens bootstrapping
├── configurator.py   # Adaptive domain routing
├── persistence.py    # Cross-session learning
├── evolution.py      # Evolution loop
├── llm.py            # LLM backends
└── cli.py            # CLI entry point
```

## How to Add a New Thinking Tool

The easiest way to contribute — 6 tools are still unimplemented:

1. **imagine.py** — Generate future scenarios from principles
2. **empathize.py** — Analyze from multiple stakeholder perspectives
3. **shift_dimension.py** — View data at different scales/axes
4. **play.py** — Break rules, invert assumptions, explore boundaries
5. **transform.py** — Convert between modalities (text→graph, etc.)
6. **execute_and_feel.py** — Prototype execution + gut feeling detection

### Steps:

```python
# 1. Create src/sparks/tools/your_tool.py

from sparks.tools.base import BaseTool
from sparks.state import CognitiveState

class YourTool(BaseTool):
    name = "your_tool"
    
    def should_run(self, state: CognitiveState) -> bool:
        """Local rule: when should this tool activate?
        Like a ganglion — decides based on local state, not central command."""
        return True  # your condition
    
    def run(self, state: CognitiveState, **kwargs):
        # 1. Read what you need from state
        # 2. Call LLM with structured output
        # 3. Write results to state
        # 4. Emit events
        pass

# 2. Register in src/sparks/tools/__init__.py
# 3. Add to TOOL_PRIORITY in src/sparks/cost.py
# 4. Add to TOOL_DATA_NEEDS in src/sparks/context.py
```

## How to Add a New Domain Template

Edit `src/sparks/configurator.py` → `DOMAIN_CONFIGS`:

```python
"your_domain": {
    "tool_hints": {
        "observe": {"hint": "What to focus on in this domain"},
        "abstract": {"hint": "What kind of principles to extract"},
        ...
    },
    "model_overrides": {"abstract": "opus"},
    "tool_boost": ["empathize", "play"],
    "external_suggestions": ["Useful API or data source"],
}
```

## Guidelines

- **Keep tools autonomous.** Each tool decides when to run via `should_run()`. No central ordering.
- **Structured output.** All LLM calls use `llm_structured()` with Pydantic schemas.
- **Cost awareness.** Track every LLM call through `CostTracker`.
- **Biological metaphors matter.** The nervous system isn't decoration — it's architecture. New features should respect signal accumulation, threshold firing, and distributed control.
- **Absence is signal.** Don't filter out "boring" data. What's missing is often what matters most.

## What We Need Help With

- [ ] **6 missing tools** — See above
- [ ] **More domain templates** — Legal, medical, scientific, financial...
- [ ] **Convergence L2-3** — Embedding-based and LLM-judged convergence
- [ ] **Benchmarks** — Sparks vs GPT-Researcher vs STORM on same datasets
- [ ] **Multi-model support** — litellm integration for GPT-4, Gemini, local models
- [ ] **Visualization** — Dashboard showing nervous system state in real-time
- [ ] **Documentation** — Detailed guides for each thinking tool

## Philosophy

Read [Sparks of Genius](https://en.wikipedia.org/wiki/Sparks_of_Genius) by Root-Bernstein (1999) if you want to understand the theoretical foundation. The 13 thinking tools aren't arbitrary — they're derived from studying Einstein, Picasso, da Vinci, Feynman, and other creative minds.

The nervous system architecture is inspired by real neuroscience, not metaphor. If you're adding a feature, ask: "How does the real nervous system handle this?"

---

*Built with [Claude Code](https://claude.ai/code).*
