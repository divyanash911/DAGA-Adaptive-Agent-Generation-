"""
DAGA — Meta-Agent Router
For cases where deterministic rules are insufficient or the efficiency
predictor shows high uncertainty, the meta-agent LLM is invoked to
reason about the best architecture plan.

The meta-agent:
  1. Receives a structured task profile and efficiency context.
  2. Returns a JSON plan (topology + roles + reasoning).
  3. Falls back gracefully to the deterministic rule result if parsing fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from daga.backends.registry import BackendRegistry, ModelBackend
from daga.core.models import (
    AgentRole,
    AgentTopology,
    ArchitecturePlan,
    ModelTier,
    TaskProfile,
)
from daga.core.routing_rules import DeterministicRouter
from daga.telemetry.logging import get_logger, log_kv


logger = get_logger("daga.router")


# ──────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────

META_AGENT_SYSTEM = """\
You are the architecture meta-agent for DAGA (Dynamic Agentic Architecture Generation).
Your job is to select the most ENERGY-EFFICIENT and LOW-LATENCY multi-agent architecture
that can still solve the given software engineering task.

OBJECTIVES (in order):
1. Maximise likelihood of resolving the task (primary)
2. Minimise total energy consumption (Joules)
3. Minimise total wall-clock latency (seconds)

AVAILABLE TOPOLOGIES:
- single_slm           : One small language model (1-7B). Fastest, cheapest.
- single_llm           : One medium/large LLM. Good for well-scoped tasks.
- sequential_pipeline  : localiser → patcher → verifier. Balanced efficiency.
- hierarchical         : planner (large) + executor(s) (small). Good for complex repos.
- parallel_ensemble    : N agents working in parallel, results voted/merged.
- hybrid_adaptive      : Starts with SLM, escalates to LLM on failure.

AVAILABLE MODEL TIERS (cheapest to most expensive):
- slm_nano    (1-3B):   ~0.00003 J/token  ~0.3s/step
- slm_small   (7B):     ~0.00018 J/token  ~1s/step
- llm_medium  (14-32B): ~0.00070 J/token  ~3s/step
- llm_large   (70B):    ~0.00190 J/token  ~8s/step
- llm_frontier (>70B):  ~0.00600 J/token  ~15s/step

RESPONSE FORMAT (JSON only, no markdown fences):
{
  "topology": "<topology_enum_value>",
  "reasoning": "<2-3 sentence justification>",
  "roles": [
    {
      "role_id": "<unique_id>",
      "role_name": "<name>",
      "model_tier": "<tier_value>",
      "tools": ["<tool1>", ...],
      "max_tokens": <int>,
      "temperature": <float>
    }
  ],
  "predicted_latency_s": <float>,
  "predicted_energy_j": <float>,
  "predicted_resolve_prob": <float>
}
"""


def _build_user_prompt(
    profile: TaskProfile,
    deterministic_plan: ArchitecturePlan,
    experience_summary: Optional[Dict[str, Any]] = None,
) -> str:
    exp = ""
    if experience_summary:
        exp = f"""
RELEVANT HISTORICAL EXPERIENCE (similar tasks):
{json.dumps(experience_summary, indent=2)}
"""
    return f"""\
TASK PROFILE:
  task_id            : {profile.task_id}
  domain             : {profile.domain.value}
  complexity         : {profile.complexity.value}
  repo_file_count    : {profile.repo_file_count}
  affected_files_est : {profile.affected_files_estimate}
  token_count        : {profile.token_count}
  has_tests          : {profile.has_tests}
  language           : {profile.language}
  entropy            : {profile.entropy:.3f}
  named_entities     : {profile.named_entity_count}
  sla_target         : {profile.sla_target.value}
  deadline_s         : {profile.deadline_seconds}
  max_energy_j       : {profile.max_energy_joules}

DETERMINISTIC RULE SUGGESTION:
  topology  : {deterministic_plan.topology.value}
  reasoning : {deterministic_plan.reasoning}
  roles     : {[r.role_name for r in deterministic_plan.roles]}
{exp}
TASK DESCRIPTION (first 600 chars):
{profile.raw_input[:600]}

Produce a JSON architecture plan. If the deterministic suggestion is good, return
a plan with topology/roles close to it. Override only when you have strong reason
to believe a different architecture will be more efficient or more likely to succeed.
"""


# ──────────────────────────────────────────────
# JSON parser
# ──────────────────────────────────────────────

def _parse_plan(
    raw: str,
    task_id: str,
    fallback: ArchitecturePlan,
) -> ArchitecturePlan:
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log_kv(logger, 30, "meta_agent.parse.failed", stage="route", task_id=task_id, reason="json_decode_error")
        return fallback

    try:
        topology = AgentTopology(data["topology"])
    except (KeyError, ValueError):
        topology = fallback.topology

    roles: List[AgentRole] = []
    for rd in data.get("roles", []):
        try:
            roles.append(AgentRole(
                role_id            = rd.get("role_id", "role"),
                role_name          = rd.get("role_name", "agent"),
                model_tier         = ModelTier(rd["model_tier"]),
                model_id           = rd.get("model_tier", "slm_small"),
                tools              = rd.get("tools", ["bash", "file_editor"]),
                system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
                max_tokens         = int(rd.get("max_tokens", 4096)),
                temperature        = float(rd.get("temperature", 0.2)),
            ))
        except Exception:
            continue

    if not roles:
        roles = fallback.roles

    log_kv(
        logger,
        20,
        "meta_agent.parse.ok",
        stage="route",
        task_id=task_id,
        topology=topology.value,
        n_roles=len(roles),
    )

    return ArchitecturePlan(
        task_id              = task_id,
        topology             = topology,
        roles                = roles,
        routing_source       = "meta_llm",
        reasoning            = data.get("reasoning", ""),
        predicted_latency_s  = float(data.get("predicted_latency_s", 0.0)),
        predicted_energy_j   = float(data.get("predicted_energy_j", 0.0)),
        predicted_resolve_prob = float(data.get("predicted_resolve_prob", 0.5)),
    )


# ──────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────

class MetaAgentRouter:
    """
    Hybrid router: deterministic rules first, meta-LLM for ambiguous cases.

    Ambiguity criteria (any → invoke LLM):
      - Predicted resolve probability from rule is below `uncertainty_threshold`
      - Task is COMPLEX or EPIC
      - SLA target is QUALITY_FIRST
      - Developer explicitly requests LLM routing
    """

    def __init__(
        self,
        registry: BackendRegistry,
        det_router: Optional[DeterministicRouter] = None,
        meta_model_tier: ModelTier = ModelTier.SLM_SMALL,
        uncertainty_threshold: float = 0.65,
        always_use_llm: bool = False,
    ) -> None:
        self._registry  = registry
        self._det_router = det_router or DeterministicRouter()
        self._meta_tier  = meta_model_tier
        self._threshold  = uncertainty_threshold
        self._always_llm = always_use_llm

    def _get_meta_backend(self) -> Optional[ModelBackend]:
        backends = self._registry.get_by_tier(self._meta_tier)
        return backends[0] if backends else None

    def _should_invoke_llm(
        self,
        profile: TaskProfile,
        det_plan: ArchitecturePlan,
    ) -> bool:
        if self._always_llm:
            return True
        from daga.core.models import TaskComplexity, SLATarget
        if profile.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EPIC):
            return True
        if profile.sla_target == SLATarget.QUALITY_FIRST:
            return True
        # No predicted prob for deterministic plans → treat as uncertain for epic
        return False

    def route(
        self,
        profile: TaskProfile,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> ArchitecturePlan:
        # Step 1: deterministic baseline
        det_plan = self._det_router.route_to_plan(profile)

        log_kv(
            logger,
            20,
            "routing.deterministic.selected",
            stage="route",
            task_id=profile.task_id,
            plan_id=det_plan.plan_id,
            topology=det_plan.topology.value,
            routing_source=det_plan.routing_source,
            reasoning=det_plan.reasoning,
        )

        invoke = self._should_invoke_llm(profile, det_plan)
        log_kv(
            logger,
            20,
            "routing.decision",
            stage="route",
            task_id=profile.task_id,
            plan_id=det_plan.plan_id,
            invoke_meta_llm=invoke,
            meta_tier=self._meta_tier.value,
            complexity=profile.complexity.value,
            sla_target=profile.sla_target.value,
            rule_topology=det_plan.topology.value,
        )

        if not invoke:
            return det_plan

        # Step 2: LLM refinement
        meta_backend = self._get_meta_backend()
        if meta_backend is None:
            log_kv(logger, 30, "routing.meta_llm.unavailable", stage="route", task_id=profile.task_id, meta_tier=self._meta_tier.value)
            return det_plan   # no backend available

        user_prompt = _build_user_prompt(profile, det_plan, experience_summary)
        messages = [
            {"role": "system", "content": META_AGENT_SYSTEM},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            log_kv(logger, 20, "routing.meta_llm.call.start", stage="route", task_id=profile.task_id, model_id=meta_backend.model_id)
            resp = meta_backend.complete(messages, max_tokens=1024, temperature=0.1)
            log_kv(
                logger,
                20,
                "routing.meta_llm.call.finish",
                stage="route",
                task_id=profile.task_id,
                model_id=meta_backend.model_id,
                latency_s=resp.latency_s,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                energy_j=resp.energy_j,
            )
            llm_plan = _parse_plan(resp.text, profile.task_id, det_plan)
            llm_plan.routing_source = "hybrid"
            return llm_plan
        except Exception as exc:
            # Graceful fallback
            log_kv(logger, 40, "routing.meta_llm.failed", stage="route", task_id=profile.task_id, error=str(exc))
            det_plan.reasoning += f" [LLM routing failed: {exc}; using deterministic]"
            return det_plan
