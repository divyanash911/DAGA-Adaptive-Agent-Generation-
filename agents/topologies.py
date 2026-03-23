"""
DAGA — Topology Orchestrators
Each class drives the full execution of one architecture topology,
coordinating multiple AgentExecutors and collecting a unified ExecutionTrace.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from daga.agents.executor import AgentExecutor, AgentExecutorResult
from daga.backends.registry import BackendRegistry
from daga.core.models import (
    AgentTopology,
    ArchitecturePlan,
    ExecutionTrace,
    StepTrace,
    TaskProfile,
)
from daga.tools.registry import ToolRegistry
from daga.telemetry.logging import get_logger, log_kv


logger = get_logger("daga.topology")


# ──────────────────────────────────────────────
# Base orchestrator
# ──────────────────────────────────────────────

class TopologyOrchestrator(ABC):
    def __init__(
        self,
        plan: ArchitecturePlan,
        registry: BackendRegistry,
        tool_registry: ToolRegistry,
        verbose: bool = False,
    ) -> None:
        self.plan     = plan
        self.registry = registry
        self.tools    = tool_registry
        self.verbose  = verbose

    @abstractmethod
    def execute(self, profile: TaskProfile) -> ExecutionTrace: ...

    def _make_executor(self, role_id: str, max_iter: int = 20) -> AgentExecutor:
        role = next(r for r in self.plan.roles if r.role_id == role_id)
        return AgentExecutor(
            role          = role,
            registry      = self.registry,
            tool_registry = self.tools,
            max_iterations= max_iter,
            verbose       = self.verbose,
        )

    def _trace_from_result(self, res: AgentExecutorResult) -> List[StepTrace]:
        return res.steps


# ──────────────────────────────────────────────
# 1. Single-agent (SLM or LLM)
# ──────────────────────────────────────────────

class SingleAgentOrchestrator(TopologyOrchestrator):
    """One agent does everything."""

    def execute(self, profile: TaskProfile) -> ExecutionTrace:
        trace = ExecutionTrace(plan_id=self.plan.plan_id, task_id=profile.task_id)

        log_kv(
            logger,
            20,
            "topology.single.start",
            stage="execute",
            task_id=profile.task_id,
            plan_id=self.plan.plan_id,
            topology=self.plan.topology.value,
            model_id=self.plan.roles[0].model_id if self.plan.roles else None,
        )

        role = self.plan.roles[0]
        executor = self._make_executor(role.role_id)
        result   = executor.run(profile.raw_input)

        trace.steps.extend(result.steps)
        trace.finish(result.success, result.final_patch)
        log_kv(
            logger,
            20,
            "topology.single.finish",
            stage="execute",
            task_id=profile.task_id,
            plan_id=self.plan.plan_id,
            trace_id=trace.trace_id,
            resolved=trace.resolved,
            has_patch=bool(trace.final_patch),
        )
        return trace


# ──────────────────────────────────────────────
# 2. Sequential pipeline
# ──────────────────────────────────────────────

class SequentialPipelineOrchestrator(TopologyOrchestrator):
    """
    Runs roles in declaration order, passing output of each as context to the next.
    Roles: localiser → patcher → verifier
    """

    def execute(self, profile: TaskProfile) -> ExecutionTrace:
        trace = ExecutionTrace(plan_id=self.plan.plan_id, task_id=profile.task_id)
        context = profile.raw_input
        last_patch: Optional[str] = None
        resolved = False

        log_kv(
            logger,
            20,
            "topology.pipeline.start",
            stage="execute",
            task_id=profile.task_id,
            plan_id=self.plan.plan_id,
            topology=self.plan.topology.value,
            roles=[r.role_name for r in self.plan.roles],
        )

        for role in self.plan.roles:
            log_kv(
                logger,
                20,
                "topology.pipeline.role.start",
                stage="execute",
                task_id=profile.task_id,
                plan_id=self.plan.plan_id,
                role_id=role.role_id,
                role_name=role.role_name,
                model_id=role.model_id,
            )
            executor = self._make_executor(role.role_id)
            result   = executor.run(profile.raw_input, extra_context=context)
            trace.steps.extend(result.steps)

            log_kv(
                logger,
                20,
                "topology.pipeline.role.finish",
                stage="execute",
                task_id=profile.task_id,
                plan_id=self.plan.plan_id,
                role_id=role.role_id,
                role_name=role.role_name,
                success=result.success,
                has_patch=bool(result.final_patch),
            )

            if result.final_patch:
                last_patch = result.final_patch

            # Pass along role output as context for next stage
            if result.final_output:
                context = (
                    f"Previous stage ({role.role_name}) output:\n"
                    f"{result.final_output}\n\n"
                    f"Original task:\n{profile.raw_input}"
                )

            if not result.success and role.role_name == "verifier":
                # Verifier failed — try re-running patcher (one retry)
                if self.verbose:
                    print("[pipeline] Verifier failed, retrying patcher…")
                log_kv(logger, 30, "topology.pipeline.verifier_failed", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id)
                patcher_roles = [r for r in self.plan.roles if r.role_name == "patcher"]
                if patcher_roles:
                    retry_exec = self._make_executor(patcher_roles[0].role_id, max_iter=10)
                    retry_res  = retry_exec.run(
                        profile.raw_input,
                        extra_context=f"Previous patch failed verification:\n{context}",
                    )
                    trace.steps.extend(retry_res.steps)
                    if retry_res.final_patch:
                        last_patch = retry_res.final_patch
                        resolved = True

        resolved = resolved or (last_patch is not None)
        trace.finish(resolved, last_patch)
        log_kv(
            logger,
            20,
            "topology.pipeline.finish",
            stage="execute",
            task_id=profile.task_id,
            plan_id=self.plan.plan_id,
            trace_id=trace.trace_id,
            resolved=trace.resolved,
            has_patch=bool(trace.final_patch),
        )
        return trace


# ──────────────────────────────────────────────
# 3. Hierarchical (planner + executors)
# ──────────────────────────────────────────────

class HierarchicalOrchestrator(TopologyOrchestrator):
    """
    Planner decomposes the task into sub-tasks.
    Each sub-task is executed sequentially by executor agents.
    Results are combined into a final patch.
    """

    def execute(self, profile: TaskProfile) -> ExecutionTrace:
        trace = ExecutionTrace(plan_id=self.plan.plan_id, task_id=profile.task_id)

        log_kv(logger, 20, "topology.hierarchical.start", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id)

        planner_roles  = [r for r in self.plan.roles if r.role_name == "planner"]
        executor_roles = [r for r in self.plan.roles if r.role_name != "planner"]

        if not planner_roles or not executor_roles:
            # Degenerate: run all roles sequentially
            return SequentialPipelineOrchestrator(
                self.plan, self.registry, self.tools, self.verbose
            ).execute(profile)

        # Step 1: Planning
        planner_exec = self._make_executor(planner_roles[0].role_id, max_iter=5)
        plan_result  = planner_exec.run(profile.raw_input)
        trace.steps.extend(plan_result.steps)

        log_kv(
            logger,
            20,
            "topology.hierarchical.planning.finish",
            stage="execute",
            task_id=profile.task_id,
            plan_id=self.plan.plan_id,
            has_plan_output=bool(plan_result.final_output),
        )

        plan_json = plan_result.final_output or ""
        try:
            plan_data  = json.loads(plan_json) if plan_json.startswith("{") else {}
            sub_tasks  = plan_data.get("sub_tasks", [])
        except Exception:
            sub_tasks = []

        if not sub_tasks:
            sub_tasks = [{"id": "1", "description": profile.raw_input}]

        log_kv(logger, 20, "topology.hierarchical.subtasks", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, n_subtasks=len(sub_tasks))

        # Step 2: Execute sub-tasks
        all_patches: List[str] = []
        exec_role = executor_roles[0]

        for sub in sub_tasks:
            log_kv(logger, 20, "topology.hierarchical.subtask.start", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, subtask_id=sub.get("id"))
            exec_executor = self._make_executor(exec_role.role_id)
            sub_desc = sub.get("description", str(sub))
            sub_result = exec_executor.run(
                sub_desc,
                extra_context=(
                    f"Part of larger task:\n{profile.raw_input}\n\n"
                    f"Plan:\n{plan_json}"
                ),
            )
            trace.steps.extend(sub_result.steps)
            if sub_result.final_patch:
                all_patches.append(sub_result.final_patch)
            log_kv(logger, 20, "topology.hierarchical.subtask.finish", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, subtask_id=sub.get("id"), success=sub_result.success, has_patch=bool(sub_result.final_patch))

        final_patch = "\n".join(all_patches) if all_patches else None
        trace.finish(bool(final_patch), final_patch)
        log_kv(logger, 20, "topology.hierarchical.finish", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, trace_id=trace.trace_id, resolved=trace.resolved, has_patch=bool(trace.final_patch))
        return trace


# ──────────────────────────────────────────────
# 4. Parallel ensemble + voting
# ──────────────────────────────────────────────

class ParallelEnsembleOrchestrator(TopologyOrchestrator):
    """
    Runs N workers in parallel, collects their patches, and selects
    the best one via a lightweight voting / scoring step.
    """

    def execute(self, profile: TaskProfile) -> ExecutionTrace:
        trace = ExecutionTrace(plan_id=self.plan.plan_id, task_id=profile.task_id)

        log_kv(logger, 20, "topology.ensemble.start", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id)

        workers = [r for r in self.plan.roles if r.parallel_group is not None]
        if not workers:
            workers = self.plan.roles

        def _run_worker(role_id: str) -> AgentExecutorResult:
            executor = self._make_executor(role_id)
            return executor.run(profile.raw_input)

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = {pool.submit(_run_worker, r.role_id): r for r in workers}
            results: List[AgentExecutorResult] = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    if self.verbose:
                        print(f"[ensemble] worker error: {exc}")
                    log_kv(logger, 40, "topology.ensemble.worker_error", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, error=str(exc))

        for res in results:
            trace.steps.extend(res.steps)

        # Voting: prefer patches from workers that also passed tests (success=True)
        successful = [r for r in results if r.success and r.final_patch]
        if successful:
            # Pick patch with most "votes" (identical patches) or first successful
            patch_votes: Dict[str, int] = {}
            for r in successful:
                p = r.final_patch or ""
                patch_votes[p] = patch_votes.get(p, 0) + 1
            best_patch = max(patch_votes, key=lambda p: patch_votes[p])
            log_kv(logger, 20, "topology.ensemble.vote", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, n_success=len(successful), n_unique=len(patch_votes), best_votes=patch_votes.get(best_patch))
            trace.finish(True, best_patch)
        else:
            # No success — return first patch if any
            any_patch = next((r.final_patch for r in results if r.final_patch), None)
            log_kv(logger, 30, "topology.ensemble.no_success", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, has_any_patch=bool(any_patch))
            trace.finish(False, any_patch)

        log_kv(logger, 20, "topology.ensemble.finish", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, trace_id=trace.trace_id, resolved=trace.resolved, has_patch=bool(trace.final_patch))
        return trace


# ──────────────────────────────────────────────
# 5. Hybrid adaptive (SLM → escalate on failure)
# ──────────────────────────────────────────────

class HybridAdaptiveOrchestrator(TopologyOrchestrator):
    """
    Starts with the cheapest (SLM) agent.
    Escalates to the next tier if:
      - No patch produced after max_slm_iters
      - Tests fail
      - Explicit "I need more capability" signal in output
    """

    def execute(self, profile: TaskProfile) -> ExecutionTrace:
        from daga.core.models import ModelTier

        trace = ExecutionTrace(plan_id=self.plan.plan_id, task_id=profile.task_id)

        log_kv(logger, 20, "topology.hybrid.start", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id)

        tier_sequence = [
            ModelTier.SLM_NANO,
            ModelTier.SLM_SMALL,
            ModelTier.LLM_MEDIUM,
            ModelTier.LLM_LARGE,
            ModelTier.LLM_FRONTIER,
        ]

        # Find available tiers in registry
        available_tiers = [
            t for t in tier_sequence
            if self.registry.get_by_tier(t)
        ]

        # Respect energy budget
        remaining_energy = (
            profile.max_energy_joules
            if profile.max_energy_joules is not None
            else float("inf")
        )

        for tier in available_tiers:
            from daga.backends.registry import MODEL_ENERGY_PROFILE
            estimated_cost = (
                profile.token_count * MODEL_ENERGY_PROFILE[tier]["j_per_input_token"] * 3
            )
            if estimated_cost > remaining_energy:
                if self.verbose:
                    print(f"[hybrid] Skipping {tier.value}: energy budget exceeded")
                continue

            backend = self.registry.get_by_tier(tier)[0]
            from daga.core.models import AgentRole
            role = AgentRole(
                role_id            = f"adaptive_{tier.value}",
                role_name          = "solver",
                model_tier         = tier,
                model_id           = backend.model_id,
                tools              = self.plan.roles[0].tools if self.plan.roles else ["bash", "file_editor"],
                system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
                max_tokens         = 4096,
            )

            log_kv(logger, 20, "topology.hybrid.tier.start", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, tier=tier.value, model_id=backend.model_id)
            executor = AgentExecutor(
                role          = role,
                registry      = self.registry,
                tool_registry = self.tools,
                max_iterations= 15,
                verbose       = self.verbose,
            )
            result = executor.run(profile.raw_input)
            trace.steps.extend(result.steps)

            remaining_energy -= result.total_energy_j

            if result.success and result.final_patch:
                if self.verbose:
                    print(f"[hybrid] Solved at tier {tier.value}")
                trace.finish(True, result.final_patch)
                log_kv(logger, 20, "topology.hybrid.finish", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, trace_id=trace.trace_id, resolved=True, tier=tier.value)
                return trace

            if self.verbose:
                print(f"[hybrid] {tier.value} failed, escalating…")
            log_kv(logger, 30, "topology.hybrid.escalate", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, tier=tier.value)

        # All tiers failed
        trace.finish(False, None)
        log_kv(logger, 40, "topology.hybrid.failed", stage="execute", task_id=profile.task_id, plan_id=self.plan.plan_id, trace_id=trace.trace_id)
        return trace


# ──────────────────────────────────────────────
# Orchestrator factory
# ──────────────────────────────────────────────

def create_orchestrator(
    plan: ArchitecturePlan,
    registry: BackendRegistry,
    tool_registry: ToolRegistry,
    verbose: bool = False,
) -> TopologyOrchestrator:
    mapping = {
        AgentTopology.SINGLE_SLM:          SingleAgentOrchestrator,
        AgentTopology.SINGLE_LLM:          SingleAgentOrchestrator,
        AgentTopology.SEQUENTIAL_PIPELINE: SequentialPipelineOrchestrator,
        AgentTopology.HIERARCHICAL:        HierarchicalOrchestrator,
        AgentTopology.PARALLEL_ENSEMBLE:   ParallelEnsembleOrchestrator,
        AgentTopology.HYBRID_ADAPTIVE:     HybridAdaptiveOrchestrator,
    }
    cls = mapping.get(plan.topology, SingleAgentOrchestrator)
    return cls(plan, registry, tool_registry, verbose)
