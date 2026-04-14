"""
DAGA — Meta-Agent Generator
Generates a task-specific agentic architecture using:
  - task profile
  - deterministic bootstrap hints
  - available model repository
  - available tool repository
  - historical experience summary

The generator emits a JSON DSL, validates it, and instantiates it as an
ArchitecturePlan consumed by the runtime orchestrators.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from daga.backends.registry import BackendRegistry, ModelBackend
from daga.core.models import (
    AgentRole,
    AgentTopology,
    ArchitectureBudget,
    ArchitectureEdge,
    ArchitecturePlan,
    GeneratedArchitectureSpec,
    ModelTier,
    TaskProfile,
)
from daga.core.routing_rules import DeterministicRouter
from daga.telemetry.logging import get_logger, log_kv
from daga.tools.registry import ToolRegistry


logger = get_logger("daga.architecture")


ROLE_PROMPT_TEMPLATES: Dict[str, str] = {
    "solver": "{SYSTEM_PROMPT_SOLVER}",
    "localiser": "{SYSTEM_PROMPT_LOCALISER}",
    "patcher": "{SYSTEM_PROMPT_PATCHER}",
    "verifier": "{SYSTEM_PROMPT_VERIFIER}",
    "planner": "{SYSTEM_PROMPT_PLANNER}",
    "executor": "{SYSTEM_PROMPT_EXECUTOR}",
}


META_AGENT_SYSTEM = """\
You are the architecture meta-agent for DAGA (Dynamic Agentic Architecture Generation).
Your job is to generate the most efficient agentic architecture for the task.

You are NOT only selecting from presets. You can synthesize a task-specific architecture
using the model repository and tool repository you are given.

OBJECTIVES (in order):
1. Maximise likelihood of resolving the task
2. Minimise total energy consumption
3. Minimise total wall-clock latency

Return JSON only, with no markdown fences.

Required schema:
{
  "topology": "<topology_enum_value>",
  "reasoning": "<2-3 sentence justification>",
  "agents": [
    {
      "id": "<unique_id>",
      "role": "<solver|planner|localiser|patcher|verifier|executor|custom>",
      "model": {
        "tier": "<model_tier_enum_value>",
        "model_id": "<optional concrete model id from model repository>"
      },
      "tools": ["<tool_name>", "..."],
      "max_tokens": <int>,
      "temperature": <float>,
      "max_iterations": <int>,
      "parallel_group": <optional int>,
      "depends_on": ["<role_id>", "..."]
    }
  ],
  "edges": [
    {
      "from": "<role_id>",
      "to": "<role_id>",
      "type": "<delegates|feeds|verifies|votes|escalates>"
    }
  ],
  "fallbacks": [
    {
      "on": "<event>",
      "action": "<retry|escalate_model_tier|handoff|stop>"
    }
  ],
  "budgets": {
    "max_latency_s": <optional float>,
    "max_energy_j": <optional float>,
    "max_tokens": <optional int>
  },
  "predicted_latency_s": <float>,
  "predicted_energy_j": <float>,
  "predicted_resolve_prob": <float>
}
"""


def _build_user_prompt(
    profile: TaskProfile,
    bootstrap_plan: ArchitecturePlan,
    model_catalog: List[Dict[str, Any]],
    tool_catalog: List[Dict[str, Any]],
    experience_summary: Optional[Dict[str, Any]] = None,
) -> str:
    exp = ""
    if experience_summary:
        exp = (
            "\nRELEVANT HISTORICAL EXPERIENCE (similar tasks):\n"
            f"{json.dumps(experience_summary, indent=2)}\n"
        )

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

BOOTSTRAP ARCHITECTURE PRIOR:
  topology  : {bootstrap_plan.topology.value}
  reasoning : {bootstrap_plan.reasoning}
  roles     : {[r.role_name for r in bootstrap_plan.roles]}

MODEL REPOSITORY:
{json.dumps(model_catalog, indent=2)}

TOOL REPOSITORY:
{json.dumps(tool_catalog, indent=2)}
{exp}
TASK DESCRIPTION (first 1200 chars):
{profile.raw_input[:1200]}

Generate a task-specific JSON architecture. Use the bootstrap prior when it already fits,
but customize roles, tool permissions, and model choices when the task benefits from it.
Prefer the smallest architecture that can plausibly solve the task.
"""


def _role_prompt_template(role_name: str) -> str:
    if role_name in ROLE_PROMPT_TEMPLATES:
        return ROLE_PROMPT_TEMPLATES[role_name]
    lowered = role_name.lower()
    if "plan" in lowered:
        return "{SYSTEM_PROMPT_PLANNER}"
    if "local" in lowered:
        return "{SYSTEM_PROMPT_LOCALISER}"
    if "patch" in lowered or "edit" in lowered:
        return "{SYSTEM_PROMPT_PATCHER}"
    if "verify" in lowered or "test" in lowered or "critic" in lowered:
        return "{SYSTEM_PROMPT_VERIFIER}"
    if "exec" in lowered or "worker" in lowered:
        return "{SYSTEM_PROMPT_EXECUTOR}"
    return "{SYSTEM_PROMPT_SOLVER}"


def _clean_json(raw: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _fallback_spec(plan: ArchitecturePlan) -> GeneratedArchitectureSpec:
    return GeneratedArchitectureSpec(
        topology=plan.topology,
        agents=plan.roles,
        reasoning=plan.reasoning,
        predicted_latency_s=plan.predicted_latency_s,
        predicted_energy_j=plan.predicted_energy_j,
        predicted_resolve_prob=plan.predicted_resolve_prob,
    )


def _parse_generated_spec(
    raw: str,
    task_id: str,
    fallback: ArchitecturePlan,
    allowed_tools: List[str],
) -> GeneratedArchitectureSpec:
    cleaned = _clean_json(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log_kv(logger, 30, "architecture.parse.failed", stage="generate", task_id=task_id, reason="json_decode_error")
        return _fallback_spec(fallback)

    try:
        topology = AgentTopology(data["topology"])
    except Exception:
        topology = fallback.topology

    fallback_roles = {role.role_id: role for role in fallback.roles}
    edges: List[ArchitectureEdge] = []
    for edge in data.get("edges", []):
        try:
            edges.append(ArchitectureEdge(
                from_role_id=str(edge["from"]),
                to_role_id=str(edge["to"]),
                edge_type=str(edge.get("type", "delegates")),
            ))
        except Exception:
            continue

    role_dependencies: Dict[str, List[str]] = {}
    for edge in edges:
        role_dependencies.setdefault(edge.to_role_id, []).append(edge.from_role_id)

    agents: List[AgentRole] = []
    for idx, agent in enumerate(data.get("agents", [])):
        try:
            role_id = str(agent.get("id", f"agent_{idx}"))
            role_name = str(agent.get("role", "solver"))

            model_info = agent.get("model", {})
            tier_value = model_info.get("tier") or agent.get("model_tier")
            model_tier = ModelTier(tier_value) if tier_value else fallback.roles[0].model_tier
            model_id = str(model_info.get("model_id") or model_tier.value)

            raw_tools = agent.get("tools", [])
            tools = [tool for tool in raw_tools if tool in allowed_tools]
            if not tools:
                fallback_role = fallback_roles.get(role_id)
                if fallback_role:
                    tools = [tool for tool in fallback_role.tools if tool in allowed_tools]
            if not tools:
                tools = [tool for tool in ["file_reader", "ast_search"] if tool in allowed_tools]
            if not tools and allowed_tools:
                tools = allowed_tools[:1]

            depends_on = [str(dep) for dep in agent.get("depends_on", role_dependencies.get(role_id, []))]

            agents.append(AgentRole(
                role_id=role_id,
                role_name=role_name,
                model_tier=model_tier,
                model_id=model_id,
                tools=tools,
                system_prompt_template=_role_prompt_template(role_name),
                max_tokens=_parse_int(agent.get("max_tokens"), 4096),
                temperature=_parse_float(agent.get("temperature"), 0.2),
                max_iterations=_parse_int(agent.get("max_iterations"), 20),
                parallel_group=agent.get("parallel_group"),
                depends_on=depends_on,
            ))
        except Exception:
            continue

    if not agents:
        return _fallback_spec(fallback)

    budgets_data = data.get("budgets", {}) if isinstance(data.get("budgets", {}), dict) else {}
    return GeneratedArchitectureSpec(
        topology=topology,
        agents=agents,
        edges=edges,
        fallbacks=data.get("fallbacks", []),
        budgets=ArchitectureBudget(
            max_latency_s=_parse_float(budgets_data.get("max_latency_s"), None) if budgets_data.get("max_latency_s") is not None else None,
            max_energy_j=_parse_float(budgets_data.get("max_energy_j"), None) if budgets_data.get("max_energy_j") is not None else None,
            max_tokens=_parse_int(budgets_data.get("max_tokens"), 0) or None,
        ),
        reasoning=str(data.get("reasoning", fallback.reasoning)),
        predicted_latency_s=_parse_float(data.get("predicted_latency_s"), fallback.predicted_latency_s),
        predicted_energy_j=_parse_float(data.get("predicted_energy_j"), fallback.predicted_energy_j),
        predicted_resolve_prob=_parse_float(data.get("predicted_resolve_prob"), fallback.predicted_resolve_prob or 0.5),
    )


class MetaAgentGenerator:
    """
    Hybrid architecture generator:
      1. derive a cheap deterministic bootstrap prior
      2. optionally invoke an LLM with model/tool repository awareness
      3. instantiate the generated JSON DSL as an ArchitecturePlan
    """

    def __init__(
        self,
        registry: BackendRegistry,
        tool_registry: Optional[ToolRegistry] = None,
        det_router: Optional[DeterministicRouter] = None,
        meta_model_tier: ModelTier = ModelTier.SLM_SMALL,
        uncertainty_threshold: float = 0.65,
        always_use_llm: bool = False,
    ) -> None:
        self._registry = registry
        self._tools = tool_registry or ToolRegistry()
        self._det_router = det_router or DeterministicRouter()
        self._meta_tier = meta_model_tier
        self._threshold = uncertainty_threshold
        self._always_llm = always_use_llm

    def _get_meta_backend(self) -> Optional[ModelBackend]:
        backends = self._registry.get_by_tier(self._meta_tier)
        return backends[0] if backends else None

    def _bootstrap_plan(self, profile: TaskProfile) -> ArchitecturePlan:
        return self._det_router.route_to_plan(profile)

    def _should_invoke_llm(
        self,
        profile: TaskProfile,
        bootstrap_plan: ArchitecturePlan,
    ) -> bool:
        if self._always_llm:
            return True

        from daga.core.models import SLATarget, TaskComplexity

        if profile.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EPIC):
            return True
        if profile.sla_target == SLATarget.QUALITY_FIRST:
            return True
        if profile.deadline_seconds is not None and profile.deadline_seconds < 30:
            return False
        if len(bootstrap_plan.roles) > 2:
            return True
        return False

    def generate(
        self,
        profile: TaskProfile,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> ArchitecturePlan:
        bootstrap_plan = self._bootstrap_plan(profile)

        log_kv(
            logger,
            20,
            "architecture.bootstrap.selected",
            stage="generate",
            task_id=profile.task_id,
            plan_id=bootstrap_plan.plan_id,
            topology=bootstrap_plan.topology.value,
            routing_source=bootstrap_plan.routing_source,
            reasoning=bootstrap_plan.reasoning,
        )

        invoke = self._should_invoke_llm(profile, bootstrap_plan)
        log_kv(
            logger,
            20,
            "architecture.generation.decision",
            stage="generate",
            task_id=profile.task_id,
            plan_id=bootstrap_plan.plan_id,
            invoke_meta_llm=invoke,
            meta_tier=self._meta_tier.value,
            complexity=profile.complexity.value,
            sla_target=profile.sla_target.value,
            bootstrap_topology=bootstrap_plan.topology.value,
        )

        if not invoke:
            bootstrap_plan.generated_spec = {
                "topology": bootstrap_plan.topology.value,
                "agents": [
                    {
                        "id": role.role_id,
                        "role": role.role_name,
                        "model": {"tier": role.model_tier.value, "model_id": role.model_id},
                        "tools": role.tools,
                    }
                    for role in bootstrap_plan.roles
                ],
            }
            return bootstrap_plan

        meta_backend = self._get_meta_backend()
        if meta_backend is None:
            log_kv(logger, 30, "architecture.meta_llm.unavailable", stage="generate", task_id=profile.task_id, meta_tier=self._meta_tier.value)
            return bootstrap_plan

        model_catalog = self._registry.catalog()
        tool_catalog = self._tools.catalog()
        user_prompt = _build_user_prompt(
            profile=profile,
            bootstrap_plan=bootstrap_plan,
            model_catalog=model_catalog,
            tool_catalog=tool_catalog,
            experience_summary=experience_summary,
        )

        messages = [
            {"role": "system", "content": META_AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            log_kv(logger, 20, "architecture.meta_llm.call.start", stage="generate", task_id=profile.task_id, model_id=meta_backend.model_id)
            resp = meta_backend.complete(messages, max_tokens=1400, temperature=0.1)
            log_kv(
                logger,
                20,
                "architecture.meta_llm.call.finish",
                stage="generate",
                task_id=profile.task_id,
                model_id=meta_backend.model_id,
                latency_s=resp.latency_s,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                energy_j=resp.energy_j,
            )
            raw_spec = json.loads(_clean_json(resp.text))
            spec = _parse_generated_spec(resp.text, profile.task_id, bootstrap_plan, self._tools.list_all())
            plan = spec.to_plan(
                task_id=profile.task_id,
                source="hybrid",
                raw_spec=raw_spec,
            )
            log_kv(
                logger,
                20,
                "architecture.instantiate.ok",
                stage="generate",
                task_id=profile.task_id,
                topology=plan.topology.value,
                n_roles=len(plan.roles),
            )
            return plan
        except Exception as exc:
            log_kv(logger, 40, "architecture.meta_llm.failed", stage="generate", task_id=profile.task_id, error=str(exc))
            bootstrap_plan.reasoning += f" [Architecture generation failed: {exc}; using bootstrap]"
            bootstrap_plan.generated_spec = {
                "topology": bootstrap_plan.topology.value,
                "agents": [
                    {
                        "id": role.role_id,
                        "role": role.role_name,
                        "model": {"tier": role.model_tier.value, "model_id": role.model_id},
                        "tools": role.tools,
                    }
                    for role in bootstrap_plan.roles
                ],
            }
            return bootstrap_plan

    def route(
        self,
        profile: TaskProfile,
        experience_summary: Optional[Dict[str, Any]] = None,
    ) -> ArchitecturePlan:
        return self.generate(profile, experience_summary)


# Backward-compatible alias while the rest of the codebase migrates.
MetaAgentRouter = MetaAgentGenerator
