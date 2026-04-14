"""
DAGA — Main Pipeline
The top-level orchestrator that ties profiling → routing → execution → telemetry.

Usage:
    from daga.pipeline import DAGAPipeline, PipelineConfig
    from daga.backends.registry import build_default_registry

    registry = build_default_registry(use_mock=True)
    pipeline = DAGAPipeline(config=PipelineConfig(), registry=registry)
    result = pipeline.run(task_description="Fix the bug in utils.py ...")
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from daga.agents.topologies import create_orchestrator
from daga.backends.registry import BackendRegistry, build_default_registry
from daga.core.meta_agent import MetaAgentGenerator
from daga.core.models import (
    ArchitecturePlan,
    ExecutionTrace,
    ModelTier,
    SLATarget,
    TaskProfile,
)
from daga.core.profiler import TaskProfiler
from daga.core.routing_rules import DeterministicRouter
from daga.feedback.loop import FeedbackLoop
from daga.telemetry.collector import ExperienceStore, TelemetryCollector
from daga.tools.registry import ToolRegistry, build_default_tool_registry
from daga.telemetry.logging import configure_logging, get_logger, log_kv


logger = get_logger("daga.pipeline")


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Routing
    always_use_meta_llm:    bool  = False
    meta_llm_tier:          ModelTier = ModelTier.LLM_MEDIUM
    uncertainty_threshold:  float = 0.65

    # Efficiency weights
    alpha: float = 0.5   # performance
    beta:  float = 0.3   # energy
    gamma: float = 0.2   # latency

    # Sandbox
    workdir: str = "/tmp/daga_sandbox"

    # Experience store
    experience_store_path: str = "/tmp/daga_experience.jsonl"

    # Misc
    verbose: bool = False
    max_iterations_per_role: int = 20


# ──────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────

@dataclass
class PipelineResult:
    task_id: str = ""
    resolved: bool = False
    patch: Optional[str] = None
    topology_used: str = ""
    routing_source: str = ""
    reasoning: str = ""
    total_latency_s: float = 0.0
    total_energy_j: float = 0.0
    total_tokens: int = 0
    efficiency_score: float = 0.0
    trace: Optional[ExecutionTrace] = None
    plan: Optional[ArchitecturePlan] = None
    profile: Optional[TaskProfile] = None


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

class DAGAPipeline:
    """
    Full DAGA pipeline.

    Phases:
      1. Profile the task
      2. Route to architecture (deterministic rules + optional meta-LLM)
      3. Resolve model IDs from the backend registry
      4. Execute via the appropriate topology orchestrator
      5. Collect telemetry, persist experience, run feedback loop
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        registry: Optional[BackendRegistry] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.config = config or PipelineConfig()

        # Make logging usable out-of-the-box (handlers only set up once).
        configure_logging()

        self.registry     = registry or build_default_registry(use_mock=True)
        self.tool_registry= tool_registry or build_default_tool_registry(self.config.workdir)

        self.profiler     = TaskProfiler()
        self.det_router   = DeterministicRouter()
        self.meta_router  = MetaAgentGenerator(
            registry              = self.registry,
            tool_registry         = self.tool_registry,
            det_router            = self.det_router,
            meta_model_tier       = self.config.meta_llm_tier,
            uncertainty_threshold = self.config.uncertainty_threshold,
            always_use_llm        = self.config.always_use_meta_llm,
        )
        self.exp_store    = ExperienceStore(self.config.experience_store_path)
        self.telemetry    = TelemetryCollector(
            alpha        = self.config.alpha,
            beta         = self.config.beta,
            gamma        = self.config.gamma,
        )
        self.feedback     = FeedbackLoop(self.exp_store)

    def _resolve_model_ids(self, plan: ArchitecturePlan) -> ArchitecturePlan:
        """
        Replace tier placeholders in plan.roles[*].model_id with concrete model IDs
        from the registry.
        """
        for role in plan.roles:
            if role.model_id == role.model_tier.value:
                # Still using tier value as placeholder → resolve
                backends = self.registry.get_by_tier(role.model_tier)
                if backends:
                    role.model_id = backends[0].model_id
                else:
                    # Fallback: pick any available backend
                    all_ids = self.registry.list_all()
                    if all_ids:
                        role.model_id = all_ids[0]
        return plan

    def run(
        self,
        task_description: str,
        repo_metadata: Optional[Dict[str, Any]] = None,
        sla_target: SLATarget = SLATarget.BALANCED,
        deadline_seconds: Optional[float] = None,
        max_energy_joules: Optional[float] = None,
    ) -> PipelineResult:
        t_pipeline_start = time.perf_counter()
        log_kv(logger, 20, "pipeline.start", stage="pipeline", sla_target=sla_target.value)

        # ── Phase 1: Profile ──────────────────────────────────
        profile = self.profiler.profile(
            task_description   = task_description,
            repo_metadata      = repo_metadata,
            sla_target         = sla_target,
            deadline_seconds   = deadline_seconds,
            max_energy_joules  = max_energy_joules,
        )
        log_kv(
            logger,
            20,
            "pipeline.profiled",
            stage="profile",
            task_id=profile.task_id,
            domain=profile.domain.value,
            complexity=profile.complexity.value,
            repo_file_count=profile.repo_file_count,
            affected_files_est=profile.affected_files_estimate,
            token_count=profile.token_count,
            has_tests=profile.has_tests,
        )
        if self.config.verbose:
            print(f"\n[DAGA] task_id={profile.task_id} "
                  f"domain={profile.domain.value} complexity={profile.complexity.value}")

        # ── Phase 2: Route ────────────────────────────────────
        experience_summary = self.feedback.experience_summary_for_meta_agent(profile)
        plan = self.meta_router.generate(profile, experience_summary)
        plan = self._resolve_model_ids(plan)

        log_kv(
            logger,
            20,
            "pipeline.routed",
            stage="route",
            task_id=profile.task_id,
            plan_id=plan.plan_id,
            topology=plan.topology.value,
            routing_source=plan.routing_source,
            predicted_latency_s=plan.predicted_latency_s,
            predicted_energy_j=plan.predicted_energy_j,
            predicted_resolve_prob=plan.predicted_resolve_prob,
        )

        if self.config.verbose:
            print(f"[DAGA] topology={plan.topology.value} "
                  f"source={plan.routing_source} roles={[r.role_name for r in plan.roles]}")

        # ── Phase 3: Execute ──────────────────────────────────
        orchestrator = create_orchestrator(
            plan           = plan,
            registry       = self.registry,
            tool_registry  = self.tool_registry,
            verbose        = self.config.verbose,
        )
        log_kv(
            logger,
            20,
            "pipeline.execute.start",
            stage="execute",
            task_id=profile.task_id,
            plan_id=plan.plan_id,
            topology=plan.topology.value,
        )
        trace = orchestrator.execute(profile)
        log_kv(
            logger,
            20,
            "pipeline.execute.finish",
            stage="execute",
            task_id=profile.task_id,
            plan_id=plan.plan_id,
            trace_id=trace.trace_id,
            resolved=trace.resolved,
            total_latency_s=trace.total_latency_s,
            total_energy_j=trace.total_energy_j,
            total_tokens=trace.total_tokens,
            has_patch=bool(trace.final_patch),
        )

        # ── Phase 4: Telemetry ────────────────────────────────
        exp_record = self.telemetry.collect(profile, plan, trace)
        self.exp_store.save(exp_record)
        log_kv(
            logger,
            20,
            "pipeline.telemetry.saved",
            stage="telemetry",
            task_id=profile.task_id,
            plan_id=plan.plan_id,
            trace_id=trace.trace_id,
            efficiency_score=exp_record.efficiency_score,
        )

        if self.config.verbose:
            print(f"[DAGA] resolved={trace.resolved} "
                  f"energy={trace.total_energy_j:.4f}J "
                  f"latency={trace.total_latency_s:.2f}s "
                  f"efficiency={exp_record.efficiency_score:.4f}")

        return PipelineResult(
            task_id         = profile.task_id,
            resolved        = trace.resolved,
            patch           = trace.final_patch,
            topology_used   = plan.topology.value,
            routing_source  = plan.routing_source,
            reasoning       = plan.reasoning,
            total_latency_s = trace.total_latency_s,
            total_energy_j  = trace.total_energy_j,
            total_tokens    = trace.total_tokens,
            efficiency_score= exp_record.efficiency_score,
            trace           = trace,
            plan            = plan,
            profile         = profile,
        )

    def report(self) -> None:
        """Print the accumulated efficiency report."""
        self.feedback.print_report()
