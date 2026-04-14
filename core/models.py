"""
DAGA — Dynamic Agentic Architecture Generation
Core data models, enumerations, and shared types.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────

class TaskDomain(str, Enum):
    SOFTWARE_ENGINEERING = "software_engineering"
    CODE_GENERATION       = "code_generation"
    BUG_FIX               = "bug_fix"
    REFACTOR              = "refactor"
    TEST_WRITING          = "test_writing"
    DOCUMENTATION         = "documentation"
    ANALYSIS              = "analysis"
    UNKNOWN               = "unknown"


class TaskComplexity(str, Enum):
    """Coarse complexity bucket used by the architecture generator."""
    TRIVIAL  = "trivial"    # < 5 files, clear root cause
    SIMPLE   = "simple"     # single-file patch, known pattern
    MODERATE = "moderate"   # multi-file, some cross-module reasoning
    COMPLEX  = "complex"    # architectural change, many modules
    EPIC     = "epic"       # cross-repo, novel algorithm, long horizon


class AgentTopology(str, Enum):
    """Architecture templates the meta-agent chooses between."""
    SINGLE_SLM         = "single_slm"          # one small model, minimal tools
    SINGLE_LLM         = "single_llm"          # one large model, full tools
    SEQUENTIAL_PIPELINE = "sequential_pipeline" # localise → draft → verify
    HIERARCHICAL       = "hierarchical"         # planner LLM + executor SLMs
    PARALLEL_ENSEMBLE  = "parallel_ensemble"    # N agents, voting/merging
    HYBRID_ADAPTIVE    = "hybrid_adaptive"      # starts SLM, escalates on fail


class ModelTier(str, Enum):
    SLM_NANO    = "slm_nano"     # ~1–3 B params  (e.g. Qwen2.5-1.5B)
    SLM_SMALL   = "slm_small"   # ~7 B params     (e.g. Qwen2.5-7B, Mistral-7B)
    LLM_MEDIUM  = "llm_medium"  # ~14–32 B params  (e.g. Qwen2.5-32B)
    LLM_LARGE   = "llm_large"   # ~70 B params     (e.g. Llama-3.1-70B)
    LLM_FRONTIER = "llm_frontier" # >70 B / API     (e.g. Claude, GPT-5)


class SLATarget(str, Enum):
    """User-declared priority for this task."""
    LATENCY_FIRST  = "latency_first"   # minimise TTFT + total time
    ENERGY_FIRST   = "energy_first"    # minimise joules consumed
    QUALITY_FIRST  = "quality_first"   # maximise resolve rate
    BALANCED       = "balanced"        # default Pareto objective


# ──────────────────────────────────────────────
# Task profile (output of TaskProfiler)
# ──────────────────────────────────────────────

@dataclass
class TaskProfile:
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_input: str = ""
    domain: TaskDomain = TaskDomain.UNKNOWN
    complexity: TaskComplexity = TaskComplexity.SIMPLE

    # Code-specific signals
    repo_file_count: int = 0
    affected_files_estimate: int = 1
    token_count: int = 0
    has_tests: bool = False
    has_type_hints: bool = False
    language: str = "python"

    # NLP signals (computed by profiler)
    entropy: float = 0.0          # token-level entropy of task description
    named_entity_count: int = 0
    embedding_variance: float = 0.0
    description_length: int = 0

    # Contextual
    sla_target: SLATarget = SLATarget.BALANCED
    deadline_seconds: Optional[float] = None
    max_energy_joules: Optional[float] = None

    created_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
# Architecture plan (output of meta-agent)
# ──────────────────────────────────────────────

@dataclass
class AgentRole:
    role_id: str
    role_name: str              # "localiser", "patcher", "verifier", etc.
    model_tier: ModelTier
    model_id: str               # resolved backend model name
    tools: List[str]            # tool IDs this agent is allowed to use
    system_prompt_template: str
    max_tokens: int = 4096
    temperature: float = 0.2
    max_iterations: int = 20   # per-role iteration cap; verifier should be much lower
    # For parallel topologies
    parallel_group: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ArchitecturePlan:
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    topology: AgentTopology = AgentTopology.SINGLE_SLM
    roles: List[AgentRole] = field(default_factory=list)

    # Predicted efficiency metrics (from generator's forward model)
    predicted_latency_s: float = 0.0
    predicted_energy_j: float = 0.0
    predicted_resolve_prob: float = 0.0
    efficiency_score: float = 0.0      # composite

    # Routing meta
    routing_source: str = "deterministic"  # "deterministic" | "meta_llm" | "hybrid"
    reasoning: str = ""
    generated_spec: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ArchitectureEdge:
    from_role_id: str
    to_role_id: str
    edge_type: str = "delegates"


@dataclass
class ArchitectureBudget:
    max_latency_s: Optional[float] = None
    max_energy_j: Optional[float] = None
    max_tokens: Optional[int] = None


@dataclass
class GeneratedArchitectureSpec:
    topology: AgentTopology = AgentTopology.SINGLE_SLM
    agents: List[AgentRole] = field(default_factory=list)
    edges: List[ArchitectureEdge] = field(default_factory=list)
    fallbacks: List[Dict[str, Any]] = field(default_factory=list)
    budgets: ArchitectureBudget = field(default_factory=ArchitectureBudget)
    reasoning: str = ""
    predicted_latency_s: float = 0.0
    predicted_energy_j: float = 0.0
    predicted_resolve_prob: float = 0.0

    def to_plan(
        self,
        task_id: str,
        source: str,
        raw_spec: Optional[Dict[str, Any]] = None,
    ) -> ArchitecturePlan:
        return ArchitecturePlan(
            task_id=task_id,
            topology=self.topology,
            roles=self.agents,
            predicted_latency_s=self.predicted_latency_s,
            predicted_energy_j=self.predicted_energy_j,
            predicted_resolve_prob=self.predicted_resolve_prob,
            routing_source=source,
            reasoning=self.reasoning,
            generated_spec=raw_spec,
        )


# ──────────────────────────────────────────────
# Execution trace
# ──────────────────────────────────────────────

@dataclass
class StepTrace:
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role_id: str = ""
    model_id: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    energy_j: float = 0.0          # measured or estimated
    success: bool = True
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ExecutionTrace:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    plan_id: str = ""
    task_id: str = ""
    steps: List[StepTrace] = field(default_factory=list)
    final_patch: Optional[str] = None
    resolved: bool = False
    total_latency_s: float = 0.0
    total_energy_j: float = 0.0
    total_tokens: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def finish(self, resolved: bool, patch: Optional[str] = None) -> None:
        self.resolved = resolved
        self.final_patch = patch
        self.finished_at = time.time()
        self.total_latency_s  = sum(s.latency_s  for s in self.steps)
        self.total_energy_j   = sum(s.energy_j   for s in self.steps)
        self.total_tokens     = sum(s.input_tokens + s.output_tokens for s in self.steps)


# ──────────────────────────────────────────────
# Experience record (persisted to experience store)
# ──────────────────────────────────────────────

@dataclass
class ExperienceRecord:
    record_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_profile: Optional[TaskProfile] = None
    architecture_plan: Optional[ArchitecturePlan] = None
    execution_trace: Optional[ExecutionTrace] = None

    # Outcomes
    resolved: bool = False
    latency_s: float = 0.0
    energy_j: float = 0.0
    tokens_used: int = 0

    # Derived efficiency
    efficiency_score: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def compute_efficiency(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        energy_norm: float = 1000.0,
        latency_norm: float = 60.0,
    ) -> float:
        """
        Composite score: alpha * resolved - beta * (energy/norm) - gamma * (latency/norm)
        Designed so higher is better; negative energy/latency terms penalise waste.
        """
        perf    = 1.0 if self.resolved else 0.0
        e_term  = self.energy_j  / energy_norm
        l_term  = self.latency_s / latency_norm
        self.efficiency_score = alpha * perf - beta * e_term - gamma * l_term
        return self.efficiency_score
