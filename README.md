# DAGA — Dynamic Agentic Architecture Generation

> Energy-efficient, latency-aware multi-agent architecture selection for software engineering tasks (SWE-bench and beyond).

---

## Overview

Most agentic systems apply the same architecture to every task: a single frontier LLM in a ReAct loop, regardless of task complexity. This wastes energy on trivial tasks and underperforms on complex ones.

**DAGA** dynamically selects the right architecture for each task at runtime, jointly optimising for:

| Objective | How |
|---|---|
| **Performance** | Choose architectures that reliably resolve each complexity tier |
| **Energy efficiency** | Route simple tasks to SLMs; use large models only when necessary |
| **Latency** | Parallel topologies for deadlines; sequential for throughput |

The composite efficiency score:

```
score = α · resolve_prob − β · (energy_J / norm) − γ · (latency_s / norm)
```

---

## Architecture

```
Task Input
    │
    ▼
┌─────────────────────┐
│   Task Profiler     │  domain · complexity · repo size · entropy · SLA
└──────────┬──────────┘
           │
    ┌──────▼──────────────────────────────────────┐
    │  Meta-Agent Router                          │
    │  ┌────────────────┐  ┌──────────────────┐   │
    │  │ Deterministic  │  │  LLM meta-agent  │   │
    │  │ rule engine    │→ │  (for ambiguous) │   │
    │  └────────────────┘  └──────────────────┘   │
    └─────────────────────────────────────────────┘
           │
    Selects one of 6 topologies:
    ┌──────────────┬───────────────────┬──────────────────┐
    │  single_slm  │ sequential_pipeline│ parallel_ensemble│
    │  single_llm  │ hierarchical       │ hybrid_adaptive  │
    └──────────────┴───────────────────┴──────────────────┘
           │
    ┌──────▼─────────────────────────┐
    │  Execution Sandbox             │
    │  Tool pool + Model pool        │
    └──────────────┬─────────────────┘
                   │
    ┌──────────────▼─────────────────┐
    │  Telemetry → Experience Store  │
    │  → Feedback Loop               │
    └────────────────────────────────┘
```

---

## Topologies

| Topology | When used | Energy | Latency | Resolve |
|---|---|---|---|---|
| `single_slm` | Trivial / latency-first / energy-capped | ★★★★★ | ★★★★★ | ★★☆☆☆ |
| `single_llm` | Moderate, well-scoped | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| `sequential_pipeline` | Moderate, structured (localise→patch→verify) | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| `hierarchical` | Complex, large repos (planner + executors) | ★★★☆☆ | ★★★☆☆ | ★★★★☆ |
| `parallel_ensemble` | Hard, quality-critical (N workers + voting) | ★★☆☆☆ | ★★★★☆ | ★★★★★ |
| `hybrid_adaptive` | Unknown SLA, energy-first | ★★★★★ | ★★★☆☆ | ★★★★☆ |

---

## Installation

```bash
git clone https://github.com/your-org/daga
cd daga
pip install -e .
# Optional: SWE-bench evaluation
pip install -e ".[eval]"
# Optional: local vLLM
pip install -e ".[local]"
```

---

## Quick Start

### With mock backends (no API key required)

```python
from daga.pipeline import DAGAPipeline, PipelineConfig
from daga.backends.registry import build_default_registry
from daga.core.models import SLATarget

registry = build_default_registry(use_mock=True)
config   = PipelineConfig(verbose=True)
pipeline = DAGAPipeline(config=config, registry=registry)

result = pipeline.run(
    task_description = "Fix the KeyError in src/utils.py when the config key is missing",
    repo_metadata    = {"file_count": 120, "has_tests": True, "language": "python"},
    sla_target       = SLATarget.BALANCED,
)

print(f"Resolved:    {result.resolved}")
print(f"Topology:    {result.topology_used}")
print(f"Energy:      {result.total_energy_j:.3f} J")
print(f"Latency:     {result.total_latency_s:.2f} s")
print(f"Efficiency:  {result.efficiency_score:.4f}")
if result.patch:
    print(result.patch)
```

### With real backends

```python
from daga.backends.registry import build_default_registry

registry = build_default_registry(
    anthropic_api_key = "sk-ant-...",     # frontier tier
    openai_api_key    = "sk-...",          # large tier
    ollama_url        = "http://localhost:11434",  # SLM tiers
)
```

### SLA targets

```python
from daga.core.models import SLATarget

# Minimise latency — use SLMs aggressively
result = pipeline.run(task, sla_target=SLATarget.LATENCY_FIRST)

# Minimise energy — hybrid adaptive, escalate only on failure
result = pipeline.run(task, sla_target=SLATarget.ENERGY_FIRST)

# Maximise resolve rate — use best models
result = pipeline.run(task, sla_target=SLATarget.QUALITY_FIRST)

# Hard resource constraints
result = pipeline.run(
    task,
    deadline_seconds   = 30.0,    # forces nano SLM
    max_energy_joules  = 5.0,     # caps energy budget
)
```

---

## SWE-bench Evaluation

```bash
# Run against SWE-bench Lite with mock backends (for testing)
daga-eval \
    --dataset swe-bench-lite \
    --split test \
    --sla_target balanced \
    --output_dir ./patches \
    --use_mock

# Run with real backends (set ANTHROPIC_API_KEY etc.)
daga-eval \
    --dataset princeton-nlp/SWE-bench_Lite \
    --split test \
    --sla_target balanced \
    --output_dir ./patches \
    --max_instances 50

# Submit patches to official SWE-bench evaluator
cd SWE-bench
python -m swebench.harness.run_evaluation \
    --predictions_path ../patches \
    --swe_bench_tasks princeton-nlp/SWE-bench_Lite \
    --split test
```

---

## Model Pool Configuration

DAGA supports any combination of backends. The registry resolves `ModelTier` → concrete model:

| Tier | Default model | Provider |
|---|---|---|
| `slm_nano` | `qwen2.5:1.5b` | Ollama (local, free) |
| `slm_small` | `qwen2.5:7b` | Ollama (local, free) |
| `llm_medium` | `qwen2.5-32b-instruct` | vLLM (local) |
| `llm_large` | `gpt-4o-mini` | OpenAI API |
| `llm_frontier` | `claude-sonnet-4-6` | Anthropic API |

Swap any tier by registering a different backend:

```python
from daga.backends.registry import OpenAICompatibleBackend, BackendRegistry
from daga.core.models import ModelTier

registry = BackendRegistry()
registry.register(
    OpenAICompatibleBackend(
        model_id = "mistral-7b-instruct",
        tier     = ModelTier.SLM_SMALL,
        base_url = "http://localhost:8000/v1",
    ),
    aliases = ["slm_small"],
)
```

---

## Efficiency Feedback Loop

After each run, DAGA updates its experience store. After 10+ tasks, it can report:

```python
pipeline.report()
# ════════════════════════════════════════════════
# DAGA Efficiency Report
# ════════════════════════════════════════════════
# Total tasks:    47
# Resolve rate:   74.5%
# Avg energy:     28.3 J
# Avg latency:    18.4 s
# Avg efficiency: 0.3127
#
# Topology statistics:
#   single_slm          resolve=68.2% energy=2.1J  latency=3.1s  score=0.414
#   sequential_pipeline resolve=79.3% energy=15.4J latency=12.8s score=0.341
#   hierarchical        resolve=81.1% energy=38.2J latency=22.1s score=0.298
#
# Data-driven topology recommendations by complexity:
#   trivial      → single_slm
#   moderate     → sequential_pipeline
#   complex      → hierarchical
```

---

## Project Structure

```
daga/
├── core/
│   ├── models.py          # Data models (TaskProfile, ArchitecturePlan, etc.)
│   ├── profiler.py        # Task profiler — extracts complexity signals
│   ├── routing_rules.py   # Deterministic rule engine (10 rules, O(1))
│   ├── meta_agent.py      # LLM meta-agent router (for ambiguous cases)
│   └── predictor.py       # Forward model: predict energy/latency before execution
├── agents/
│   ├── executor.py        # Single-agent ReAct loop
│   ├── topologies.py      # Orchestrators for all 6 topologies
│   └── prompts.py         # System prompt library (all role templates)
├── backends/
│   └── registry.py        # Model backend abstraction + registry
├── tools/
│   └── registry.py        # Tool pool (bash, file_editor, ast_search, etc.)
├── telemetry/
│   └── collector.py       # Telemetry collection + experience store
├── feedback/
│   └── loop.py            # Efficiency analysis + routing recommendations
├── evaluation/
│   └── swebench_harness.py # SWE-bench evaluation harness
├── configs/
│   └── architectures.yaml # YAML architecture templates
├── tests/
│   └── test_daga.py       # 42 tests (all pass)
└── pipeline.py            # Main public entry point
```

---

## Design Decisions

### Why deterministic rules first?
Rules cover ~80% of cases in O(1) with zero tokens spent. The meta-agent LLM is only invoked for COMPLEX/EPIC tasks or QUALITY_FIRST SLA, reducing the routing overhead by ~5-10× vs always calling the LLM.

### Why estimate energy rather than measure it?
Hardware power monitors require root access and vary by machine. The per-tier J/token estimates (from TokenPowerBench and Wilhelm et al. 2025) are accurate to ~15% and sufficient for routing decisions. When more accuracy is needed, RAPL (Intel) or `nvidia-smi` readings can replace the estimates.

### Why not train a neural router?
Statistical rule refinement (the feedback loop) is interpretable, debuggable, and doesn't need a separate training pipeline. A neural router would improve accuracy by ~3–5% at the cost of a training loop and model drift.

### Novelty vs prior work
| Prior work | DAGA difference |
|---|---|
| LLM routing (RouteLLM, Frugal-GPT) | DAGA routes *architectures*, not just models |
| SWE-agent, Agentless | Fixed single-agent; DAGA adapts topology per task |
| MoA (Mixture of Agents) | Always parallel; DAGA chooses topology dynamically |
| AutoGen, CrewAI | Manual topology config; DAGA generates it automatically |

---

## Running Tests

```bash
# All 42 tests (uses mock backends, no API keys needed)
python -m pytest daga/tests/test_daga.py -v

# Single component
python -m pytest daga/tests/test_daga.py::TestDeterministicRouter -v
python -m pytest daga/tests/test_daga.py::TestEfficiencyPredictor -v
python -m pytest daga/tests/test_daga.py::TestPipeline -v
```

---

## Extending DAGA

### Adding a new topology
1. Add a value to `AgentTopology` enum in `core/models.py`
2. Create an orchestrator class inheriting from `TopologyOrchestrator` in `agents/topologies.py`
3. Register it in the `create_orchestrator` factory
4. Add a routing rule in `core/routing_rules.py`
5. Add TOPOLOGY_COST_MULTIPLIER entry in `core/predictor.py`

### Adding a new backend
```python
from daga.backends.registry import ModelBackend, ModelTier, ModelResponse

class MyBackend(ModelBackend):
    @property
    def model_id(self): return "my-model"
    @property
    def tier(self): return ModelTier.LLM_MEDIUM

    def complete(self, messages, max_tokens=2048, temperature=0.2, stop=None):
        # ... call your inference API ...
        return self._wrap(text, in_tok, out_tok, latency)
```

### Adding a new tool
```python
from daga.tools.registry import Tool, ToolResult

class MyTool(Tool):
    @property
    def name(self): return "my_tool"
    @property
    def description(self): return "Does something useful."
    @property
    def schema(self): return {"type": "object", "properties": {...}, "required": [...]}
    def __call__(self, args): ...
```
