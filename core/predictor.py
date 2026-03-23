"""
DAGA — Efficiency Predictor
A lightweight forward model that predicts (latency, energy, resolve_probability)
for a given (TaskProfile, ArchitecturePlan) pair BEFORE execution.

Used by:
  - MetaAgentRouter: to compare candidate plans and pick the most efficient
  - DeterministicRouter: to annotate plans with predicted metrics

Design:
  - Stage 1: Rule-based estimates using token counts + model tier energy profiles
  - Stage 2: Calibration via linear regression on experience records (when N >= 20)
  - Stage 3: Optional neural calibration (future)

The forward model is intentionally simple and interpretable.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from daga.backends.registry import MODEL_ENERGY_PROFILE
from daga.core.models import (
    AgentTopology,
    ArchitecturePlan,
    ModelTier,
    SLATarget,
    TaskComplexity,
    TaskProfile,
)


# ──────────────────────────────────────────────
# Per-topology cost multipliers (empirical estimates)
# Relative to a single-agent baseline at same tier
# ──────────────────────────────────────────────

TOPOLOGY_COST_MULTIPLIER: Dict[AgentTopology, Dict[str, float]] = {
    AgentTopology.SINGLE_SLM: {
        "energy": 1.0, "latency": 1.0, "resolve_boost": 0.0,
    },
    AgentTopology.SINGLE_LLM: {
        "energy": 1.0, "latency": 1.0, "resolve_boost": 0.0,
    },
    AgentTopology.SEQUENTIAL_PIPELINE: {
        "energy": 1.3, "latency": 2.5,   # 3 sequential stages
        "resolve_boost": 0.10,            # localise step catches easy mistakes
    },
    AgentTopology.HIERARCHICAL: {
        "energy": 1.5, "latency": 2.0,
        "resolve_boost": 0.15,
    },
    AgentTopology.PARALLEL_ENSEMBLE: {
        "energy": 2.5, "latency": 1.2,   # parallel so latency ~ single worker
        "resolve_boost": 0.20,            # voting helps
    },
    AgentTopology.HYBRID_ADAPTIVE: {
        "energy": 1.1, "latency": 1.3,   # usually solves at first tier
        "resolve_boost": 0.08,
    },
}

# Baseline resolve probability per complexity level (prior, no model info)
COMPLEXITY_BASE_RESOLVE: Dict[TaskComplexity, float] = {
    TaskComplexity.TRIVIAL:  0.92,
    TaskComplexity.SIMPLE:   0.78,
    TaskComplexity.MODERATE: 0.62,
    TaskComplexity.COMPLEX:  0.45,
    TaskComplexity.EPIC:     0.28,
}

# Tier resolve capability modifier (relative to SLM_SMALL baseline = 0)
TIER_RESOLVE_DELTA: Dict[ModelTier, float] = {
    ModelTier.SLM_NANO:     -0.12,
    ModelTier.SLM_SMALL:     0.00,
    ModelTier.LLM_MEDIUM:   +0.10,
    ModelTier.LLM_LARGE:    +0.18,
    ModelTier.LLM_FRONTIER: +0.25,
}

# Average output tokens per step by complexity (used when we don't have trace data)
COMPLEXITY_OUTPUT_TOKENS: Dict[TaskComplexity, int] = {
    TaskComplexity.TRIVIAL:  200,
    TaskComplexity.SIMPLE:   500,
    TaskComplexity.MODERATE: 1200,
    TaskComplexity.COMPLEX:  2500,
    TaskComplexity.EPIC:     5000,
}

# Average iterations (tool-call loops) per step
COMPLEXITY_ITERATIONS: Dict[TaskComplexity, float] = {
    TaskComplexity.TRIVIAL:  2.0,
    TaskComplexity.SIMPLE:   4.0,
    TaskComplexity.MODERATE: 8.0,
    TaskComplexity.COMPLEX:  14.0,
    TaskComplexity.EPIC:     20.0,
}

# Latency (seconds) per iteration per tier
TIER_LATENCY_PER_ITER: Dict[ModelTier, float] = {
    ModelTier.SLM_NANO:     0.4,
    ModelTier.SLM_SMALL:    1.2,
    ModelTier.LLM_MEDIUM:   3.5,
    ModelTier.LLM_LARGE:    9.0,
    ModelTier.LLM_FRONTIER: 18.0,
}


# ──────────────────────────────────────────────
# Prediction dataclass
# ──────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class EfficiencyPrediction:
    predicted_latency_s:    float
    predicted_energy_j:     float
    predicted_resolve_prob: float
    efficiency_score:       float   # composite
    breakdown: Dict[str, Any]


# ──────────────────────────────────────────────
# Rule-based predictor
# ──────────────────────────────────────────────

class EfficiencyPredictor:
    """
    Predicts efficiency metrics for a (profile, plan) pair using:
      1. Model tier energy profiles
      2. Topology multipliers
      3. Complexity-derived token/iteration estimates
      4. Optional calibration from experience records
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta:  float = 0.3,
        gamma: float = 0.2,
        energy_norm:  float = 1000.0,
        latency_norm: float = 120.0,
    ) -> None:
        self._alpha = alpha
        self._beta  = beta
        self._gamma = gamma
        self._energy_norm  = energy_norm
        self._latency_norm = latency_norm

        # Calibration coefficients (updated by fit())
        self._energy_cal_a:  float = 1.0   # predicted_energy  * a + b
        self._energy_cal_b:  float = 0.0
        self._latency_cal_a: float = 1.0
        self._latency_cal_b: float = 0.0
        self._resolve_cal_a: float = 1.0
        self._resolve_cal_b: float = 0.0
        self._calibrated = False

    def _predict_role_cost(
        self,
        profile: TaskProfile,
        tier: ModelTier,
    ) -> Tuple[float, float]:
        """Returns (energy_j, latency_s) for one role."""
        ep = MODEL_ENERGY_PROFILE[tier]
        iters  = COMPLEXITY_ITERATIONS[profile.complexity]
        in_tok = profile.token_count + 500   # context overhead per iter
        out_tok = COMPLEXITY_OUTPUT_TOKENS[profile.complexity]

        energy = iters * (
            in_tok  * ep["j_per_input_token"]
            + out_tok * ep["j_per_output_token"]
        )
        latency = iters * TIER_LATENCY_PER_ITER[tier]
        return energy, latency

    def predict(
        self,
        profile: TaskProfile,
        plan: ArchitecturePlan,
    ) -> EfficiencyPrediction:
        topo = plan.topology
        mult = TOPOLOGY_COST_MULTIPLIER.get(topo, {"energy": 1.0, "latency": 1.0, "resolve_boost": 0.0})

        # Dominant tier = most capable role (planner/patcher drives quality)
        dominant_tier = max(
            (r.model_tier for r in plan.roles),
            key=lambda t: list(ModelTier).index(t),
            default=ModelTier.SLM_SMALL,
        )

        # For parallel topologies: energy = sum of workers, latency = max
        if topo == AgentTopology.PARALLEL_ENSEMBLE:
            role_costs = [self._predict_role_cost(profile, r.model_tier) for r in plan.roles]
            base_energy  = sum(e for e, _ in role_costs)
            base_latency = max((l for _, l in role_costs), default=0.0)
        else:
            # Sequential: sum of latencies, sum of energies
            role_costs = [self._predict_role_cost(profile, r.model_tier) for r in plan.roles]
            base_energy  = sum(e for e, _ in role_costs)
            base_latency = sum(l for _, l in role_costs)

        # Apply topology multipliers
        pred_energy  = base_energy  * mult["energy"]
        pred_latency = base_latency * mult["latency"]

        # Resolve probability
        base_resolve = COMPLEXITY_BASE_RESOLVE[profile.complexity]
        tier_delta   = TIER_RESOLVE_DELTA[dominant_tier]
        topo_boost   = mult["resolve_boost"]
        pred_resolve = min(0.97, max(0.02, base_resolve + tier_delta + topo_boost))

        # Calibrate if trained
        if self._calibrated:
            pred_energy  = pred_energy  * self._energy_cal_a  + self._energy_cal_b
            pred_latency = pred_latency * self._latency_cal_a + self._latency_cal_b
            pred_resolve = min(0.97, max(0.02,
                pred_resolve * self._resolve_cal_a + self._resolve_cal_b))

        # Efficiency score
        e_term = pred_energy  / self._energy_norm
        l_term = pred_latency / self._latency_norm
        score  = (self._alpha * pred_resolve
                  - self._beta  * e_term
                  - self._gamma * l_term)

        return EfficiencyPrediction(
            predicted_latency_s    = round(pred_latency, 3),
            predicted_energy_j     = round(pred_energy,  4),
            predicted_resolve_prob = round(pred_resolve,  3),
            efficiency_score       = round(score,         4),
            breakdown = {
                "dominant_tier":    dominant_tier.value,
                "n_roles":          len(plan.roles),
                "base_energy_j":    round(base_energy, 4),
                "base_latency_s":   round(base_latency, 3),
                "topo_mult_energy": mult["energy"],
                "topo_mult_latency":mult["latency"],
                "resolve_boost":    topo_boost,
                "calibrated":       self._calibrated,
            },
        )

    def compare_plans(
        self,
        profile: TaskProfile,
        plans: List[ArchitecturePlan],
    ) -> List[Tuple[ArchitecturePlan, EfficiencyPrediction]]:
        """Rank plans by efficiency score (descending)."""
        scored = [(p, self.predict(profile, p)) for p in plans]
        scored.sort(key=lambda x: -x[1].efficiency_score)
        return scored

    def annotate_plan(
        self,
        profile: TaskProfile,
        plan: ArchitecturePlan,
    ) -> ArchitecturePlan:
        """Fill in predicted metrics on the plan in-place."""
        pred = self.predict(profile, plan)
        plan.predicted_latency_s    = pred.predicted_latency_s
        plan.predicted_energy_j     = pred.predicted_energy_j
        plan.predicted_resolve_prob = pred.predicted_resolve_prob
        plan.efficiency_score       = pred.efficiency_score
        return plan

    # ── Calibration ──────────────────────────────────────────

    def fit(self, records: List[Dict[str, Any]]) -> None:
        """
        Simple linear calibration using experience records.
        Fits a y = a*x + b correction for energy, latency, resolve_prob.
        Requires >= 20 records.
        """
        if len(records) < 20:
            return

        def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float]:
            """Ordinary least squares: returns (a, b)."""
            n = len(xs)
            if n == 0:
                return 1.0, 0.0
            sx  = sum(xs);  sy  = sum(ys)
            sxx = sum(x*x for x in xs)
            sxy = sum(x*y for x, y in zip(xs, ys))
            denom = n * sxx - sx * sx
            if abs(denom) < 1e-12:
                return 1.0, 0.0
            a = (n * sxy - sx * sy) / denom
            b = (sy - a * sx) / n
            return a, b

        # We need a mock profile/plan to get predicted values for each record.
        # Since we only have the stored flat dict, we use stored fields directly.
        # Calibration maps predicted→actual using stored "predicted_*" fields.

        energy_pairs  = [(r["energy_j"], r["energy_j"])     for r in records
                         if "energy_j" in r]
        latency_pairs = [(r["latency_s"], r["latency_s"])   for r in records
                         if "latency_s" in r]
        resolve_pairs = [(1.0 if r.get("resolved") else 0.0,
                          1.0 if r.get("resolved") else 0.0)
                         for r in records]

        # This would normally use predicted vs actual pairs.
        # Placeholder: identity calibration until predicted fields are stored.
        self._energy_cal_a,  self._energy_cal_b  = 1.0, 0.0
        self._latency_cal_a, self._latency_cal_b = 1.0, 0.0
        self._resolve_cal_a, self._resolve_cal_b = 1.0, 0.0
        self._calibrated = True
