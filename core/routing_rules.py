"""
DAGA — Deterministic Routing Rules
Fast rule engine that covers the majority of clear-cut cases without
calling the meta-agent LLM (saving tokens, energy, and latency).

Rule priority: higher priority rules win.
Each rule returns an ArchitecturePlan skeleton; the meta-agent fills
in detailed system prompts and final model IDs later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from daga.core.models import (
    AgentRole,
    AgentTopology,
    ArchitecturePlan,
    ModelTier,
    SLATarget,
    TaskComplexity,
    TaskDomain,
    TaskProfile,
)


# ──────────────────────────────────────────────
# Rule definition
# ──────────────────────────────────────────────

@dataclass
class RoutingRule:
    name: str
    priority: int                              # lower = higher priority
    condition: Callable[[TaskProfile], bool]
    topology: AgentTopology
    roles: List[AgentRole]
    reasoning: str = ""

    def matches(self, profile: TaskProfile) -> bool:
        try:
            return self.condition(profile)
        except Exception:
            return False


# ──────────────────────────────────────────────
# Role factory helpers
# ──────────────────────────────────────────────

def _solo_role(tier: ModelTier, model_hint: str) -> List[AgentRole]:
    return [AgentRole(
        role_id            = "solo",
        role_name          = "solver",
        model_tier         = tier,
        model_id           = model_hint,
        tools              = ["bash", "file_editor", "ast_search", "test_runner"],
        system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
        max_tokens         = 8192,
    )]


def _pipeline_roles(
    localiser_tier: ModelTier,
    patcher_tier: ModelTier,
    verifier_tier: ModelTier,
) -> List[AgentRole]:
    return [
        AgentRole(
            role_id            = "localiser",
            role_name          = "localiser",
            model_tier         = localiser_tier,
            model_id           = localiser_tier.value,
            tools              = ["ast_search", "ripgrep", "file_reader"],
            system_prompt_template = "{SYSTEM_PROMPT_LOCALISER}",
            max_tokens         = 2048,
            max_iterations     = 6,   # localiser only needs to search, not loop
        ),
        AgentRole(
            role_id            = "patcher",
            role_name          = "patcher",
            model_tier         = patcher_tier,
            model_id           = patcher_tier.value,
            tools              = ["file_editor", "bash"],
            system_prompt_template = "{SYSTEM_PROMPT_PATCHER}",
            max_tokens         = 8192,
            max_iterations     = 10,  # enough to read files + write patch
        ),
        AgentRole(
            role_id            = "verifier",
            role_name          = "verifier",
            model_tier         = verifier_tier,
            model_id           = verifier_tier.value,
            tools              = ["test_runner", "bash"],
            system_prompt_template = "{SYSTEM_PROMPT_VERIFIER}",
            max_tokens         = 2048,
            max_iterations     = 5,   # run tests + report result; should never spin
        ),
    ]


def _parallel_roles(n: int, tier: ModelTier) -> List[AgentRole]:
    return [
        AgentRole(
            role_id            = f"worker_{i}",
            role_name          = f"parallel_worker_{i}",
            model_tier         = tier,
            model_id           = tier.value,
            tools              = ["bash", "file_editor", "ast_search"],
            system_prompt_template = "{SYSTEM_PROMPT_WORKER}",
            parallel_group     = 0,
            max_tokens         = 4096,
        )
        for i in range(n)
    ]


# ──────────────────────────────────────────────
# Rule table
# ──────────────────────────────────────────────

DEFAULT_RULES: List[RoutingRule] = [

    # ── 0. Hard deadline / energy cap → smallest capable model ─────────────
    RoutingRule(
        name     = "tight_deadline_or_energy_cap",
        priority = 0,
        condition = lambda p: (
            (p.deadline_seconds is not None and p.deadline_seconds < 30)
            or (p.max_energy_joules is not None and p.max_energy_joules < 5.0)
        ),
        topology  = AgentTopology.SINGLE_SLM,
        roles     = _solo_role(ModelTier.SLM_NANO, "slm_nano"),
        reasoning = "Hard resource constraint forces nano SLM.",
    ),

    # ── 1. Latency SLA + trivial task → SLM nano ───────────────────────────
    RoutingRule(
        name     = "latency_first_trivial",
        priority = 1,
        condition = lambda p: (
            p.sla_target == SLATarget.LATENCY_FIRST
            and p.complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE)
        ),
        topology  = AgentTopology.SINGLE_SLM,
        roles     = _solo_role(ModelTier.SLM_NANO, "slm_nano"),
        reasoning = "Latency-first SLA with simple task → nano SLM.",
    ),

    # ── 2. Energy SLA + any complexity → prefer SLM, escalate as needed ────
    RoutingRule(
        name     = "energy_first",
        priority = 2,
        condition = lambda p: p.sla_target == SLATarget.ENERGY_FIRST,
        topology  = AgentTopology.HYBRID_ADAPTIVE,
        roles     = _solo_role(ModelTier.SLM_SMALL, "slm_small"),
        reasoning = "Energy-first SLA → start with SLM-7B; escalate only on failure.",
    ),

    # ── 3. Trivial task, any SLA ────────────────────────────────────────────
    RoutingRule(
        name     = "trivial_any_sla",
        priority = 3,
        condition = lambda p: p.complexity == TaskComplexity.TRIVIAL,
        topology  = AgentTopology.SINGLE_SLM,
        roles     = _solo_role(ModelTier.SLM_NANO, "slm_nano"),
        reasoning = "Trivial task → nano SLM is sufficient.",
    ),

    # ── 4. Simple bug fix, small repo ───────────────────────────────────────
    RoutingRule(
        name     = "simple_bug_fix",
        priority = 4,
        condition = lambda p: (
            p.complexity == TaskComplexity.SIMPLE
            and p.domain  == TaskDomain.BUG_FIX
            and p.repo_file_count <= 200
        ),
        topology  = AgentTopology.SINGLE_SLM,
        roles     = _solo_role(ModelTier.SLM_SMALL, "slm_small"),
        reasoning = "Simple bug in small repo → SLM-7B solo.",
    ),

    # ── 5. Simple code gen / refactor ───────────────────────────────────────
    RoutingRule(
        name     = "simple_codegen_or_refactor",
        priority = 5,
        condition = lambda p: (
            p.complexity in (TaskComplexity.SIMPLE, TaskComplexity.TRIVIAL)
            and p.domain in (TaskDomain.CODE_GENERATION, TaskDomain.REFACTOR,
                             TaskDomain.TEST_WRITING, TaskDomain.DOCUMENTATION)
        ),
        topology  = AgentTopology.SINGLE_SLM,
        roles     = _solo_role(ModelTier.SLM_SMALL, "slm_small"),
        reasoning = "Simple generation/refactor → SLM-7B.",
    ),

    # ── 6. Moderate complexity → efficient sequential pipeline ──────────────
    RoutingRule(
        name     = "moderate_pipeline",
        priority = 6,
        condition = lambda p: p.complexity == TaskComplexity.MODERATE,
        topology  = AgentTopology.SEQUENTIAL_PIPELINE,
        roles     = _pipeline_roles(
            localiser_tier  = ModelTier.SLM_SMALL,
            patcher_tier    = ModelTier.LLM_MEDIUM,
            verifier_tier   = ModelTier.SLM_SMALL,
        ),
        reasoning = "Moderate task → SLM localiser + medium LLM patcher + SLM verifier.",
    ),

    # ── 7. Complex, quality-first → hierarchical ────────────────────────────
    RoutingRule(
        name     = "complex_quality_first",
        priority = 7,
        condition = lambda p: (
            p.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EPIC)
            and p.sla_target == SLATarget.QUALITY_FIRST
        ),
        topology  = AgentTopology.HIERARCHICAL,
        roles     = [
            AgentRole(
                role_id   = "planner",
                role_name = "planner",
                model_tier= ModelTier.LLM_FRONTIER,
                model_id  = "llm_frontier",
                tools     = ["ast_search", "file_reader"],
                system_prompt_template = "{SYSTEM_PROMPT_PLANNER}",
                max_tokens= 8192,
            ),
            *_pipeline_roles(
                ModelTier.SLM_SMALL,
                ModelTier.LLM_LARGE,
                ModelTier.SLM_SMALL,
            ),
        ],
        reasoning = "Complex + quality-first → frontier planner + large LLM patcher.",
    ),

    # ── 8. Complex, balanced → parallel ensemble for robustness ────────────
    RoutingRule(
        name     = "complex_balanced_ensemble",
        priority = 8,
        condition = lambda p: (
            p.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EPIC)
            and p.sla_target == SLATarget.BALANCED
        ),
        topology  = AgentTopology.PARALLEL_ENSEMBLE,
        roles     = _parallel_roles(n=3, tier=ModelTier.LLM_MEDIUM),
        reasoning = "Complex balanced → 3 parallel medium-LLM agents + vote.",
    ),

    # ── 9. Large repo + moderate/complex → hierarchical ─────────────────────
    RoutingRule(
        name     = "large_repo",
        priority = 5,
        condition = lambda p: p.repo_file_count > 500,
        topology  = AgentTopology.HIERARCHICAL,
        roles     = [
            AgentRole(
                role_id   = "planner",
                role_name = "planner",
                model_tier= ModelTier.LLM_MEDIUM,
                model_id  = "llm_medium",
                tools     = ["ast_search", "ripgrep", "file_reader"],
                system_prompt_template = "{SYSTEM_PROMPT_PLANNER}",
                max_tokens= 4096,
            ),
            AgentRole(
                role_id   = "executor",
                role_name = "executor",
                model_tier= ModelTier.SLM_SMALL,
                model_id  = "slm_small",
                tools     = ["bash", "file_editor", "test_runner"],
                system_prompt_template = "{SYSTEM_PROMPT_EXECUTOR}",
                max_tokens= 4096,
            ),
        ],
        reasoning = "Large repo → medium LLM planner + SLM executor.",
    ),

    # ── 10. Catch-all: moderate LLM solo ────────────────────────────────────
    RoutingRule(
        name     = "catchall",
        priority = 99,
        condition = lambda _: True,
        topology  = AgentTopology.SINGLE_LLM,
        roles     = _solo_role(ModelTier.LLM_MEDIUM, "llm_medium"),
        reasoning = "No specific rule matched; defaulting to medium LLM.",
    ),
]


# ──────────────────────────────────────────────
# Rule engine
# ──────────────────────────────────────────────

class DeterministicRouter:
    """
    Evaluates rules in priority order and returns the first match.
    Used as the fast path in the meta-agent pipeline.
    """

    def __init__(self, rules: Optional[List[RoutingRule]] = None) -> None:
        self._rules = sorted(rules or DEFAULT_RULES, key=lambda r: r.priority)

    def route(self, profile: TaskProfile) -> Optional[RoutingRule]:
        for rule in self._rules:
            if rule.matches(profile):
                return rule
        return None

    def route_to_plan(self, profile: TaskProfile) -> ArchitecturePlan:
        rule = self.route(profile)
        assert rule is not None, "catchall rule must always match"

        return ArchitecturePlan(
            task_id        = profile.task_id,
            topology       = rule.topology,
            roles          = rule.roles,
            routing_source = "deterministic",
            reasoning      = rule.reasoning,
        )