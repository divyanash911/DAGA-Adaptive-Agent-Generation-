"""
DAGA — Telemetry & Experience Store
Collects per-step measurements, computes aggregate efficiency metrics,
and persists experience records for use by the feedback loop.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from daga.core.models import (
    ArchitecturePlan,
    ExperienceRecord,
    ExecutionTrace,
    TaskProfile,
)


# ──────────────────────────────────────────────
# Telemetry collector
# ──────────────────────────────────────────────

class TelemetryCollector:
    """
    Aggregates step-level measurements from an ExecutionTrace into
    an ExperienceRecord that can be stored and queried.
    """

    def __init__(
        self,
        alpha: float = 0.5,   # weight for performance (resolve)
        beta:  float = 0.3,   # weight for energy penalty
        gamma: float = 0.2,   # weight for latency penalty
        energy_norm:  float = 1000.0,   # normalisation constant (Joules)
        latency_norm: float = 120.0,    # normalisation constant (seconds)
    ) -> None:
        self._alpha        = alpha
        self._beta         = beta
        self._gamma        = gamma
        self._energy_norm  = energy_norm
        self._latency_norm = latency_norm

    def collect(
        self,
        profile: TaskProfile,
        plan: ArchitecturePlan,
        trace: ExecutionTrace,
    ) -> ExperienceRecord:
        rec = ExperienceRecord(
            task_profile      = profile,
            architecture_plan = plan,
            execution_trace   = trace,
            resolved          = trace.resolved,
            latency_s         = trace.total_latency_s,
            energy_j          = trace.total_energy_j,
            tokens_used       = trace.total_tokens,
        )
        rec.compute_efficiency(
            alpha        = self._alpha,
            beta         = self._beta,
            gamma        = self._gamma,
            energy_norm  = self._energy_norm,
            latency_norm = self._latency_norm,
        )
        return rec

    def summary(self, rec: ExperienceRecord) -> Dict[str, Any]:
        return {
            "task_id":        rec.task_profile.task_id if rec.task_profile else None,
            "topology":       rec.architecture_plan.topology.value if rec.architecture_plan else None,
            "complexity":     rec.task_profile.complexity.value if rec.task_profile else None,
            "resolved":       rec.resolved,
            "latency_s":      round(rec.latency_s, 3),
            "energy_j":       round(rec.energy_j, 4),
            "tokens":         rec.tokens_used,
            "efficiency":     round(rec.efficiency_score, 4),
            "routing_source": rec.architecture_plan.routing_source if rec.architecture_plan else None,
        }


# ──────────────────────────────────────────────
# Experience store (JSON file-backed)
# ──────────────────────────────────────────────

class ExperienceStore:
    """
    Persists ExperienceRecords as newline-delimited JSON.
    Provides similarity-based retrieval to give the meta-agent
    relevant historical context.
    """

    def __init__(self, store_path: str = "/tmp/daga_experience.jsonl") -> None:
        self._path = Path(store_path)
        self._records: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _to_dict(self, rec: ExperienceRecord) -> Dict[str, Any]:
        """Serialise to a flat dict (avoids deep object serialisation)."""
        plan = rec.architecture_plan
        prof = rec.task_profile
        return {
            "record_id":      rec.record_id,
            "timestamp":      rec.timestamp,
            "resolved":       rec.resolved,
            "latency_s":      rec.latency_s,
            "energy_j":       rec.energy_j,
            "tokens_used":    rec.tokens_used,
            "efficiency":     rec.efficiency_score,
            # Task signals
            "domain":         prof.domain.value         if prof else None,
            "complexity":     prof.complexity.value      if prof else None,
            "repo_file_count":prof.repo_file_count       if prof else None,
            "sla_target":     prof.sla_target.value      if prof else None,
            "token_count":    prof.token_count           if prof else None,
            "entropy":        prof.entropy               if prof else None,
            # Architecture
            "topology":       plan.topology.value        if plan else None,
            "routing_source": plan.routing_source        if plan else None,
            "n_roles":        len(plan.roles)            if plan else None,
            "role_tiers":     [r.model_tier.value for r in plan.roles] if plan else [],
            "reasoning":      plan.reasoning             if plan else None,
        }

    def save(self, rec: ExperienceRecord) -> None:
        d = self._to_dict(rec)
        self._records.append(d)
        with self._path.open("a") as f:
            f.write(json.dumps(d) + "\n")

    def _similarity(self, a: Dict[str, Any], profile: TaskProfile) -> float:
        """Simple weighted similarity for nearest-neighbour lookup."""
        score = 0.0
        if a.get("domain") == profile.domain.value:           score += 3.0
        if a.get("complexity") == profile.complexity.value:   score += 3.0
        if a.get("sla_target") == profile.sla_target.value:   score += 2.0
        # Continuous features
        if a.get("token_count") is not None:
            ratio = min(profile.token_count, a["token_count"]) / max(
                profile.token_count, a["token_count"], 1
            )
            score += ratio * 1.5
        if a.get("repo_file_count") is not None and profile.repo_file_count:
            ratio = min(profile.repo_file_count, a["repo_file_count"]) / max(
                profile.repo_file_count, a["repo_file_count"], 1
            )
            score += ratio * 1.0
        return score

    def retrieve_similar(
        self,
        profile: TaskProfile,
        top_k: int = 5,
        min_efficiency: float = -float("inf"),
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar experiences, sorted by efficiency."""
        scored = [
            (r, self._similarity(r, profile))
            for r in self._records
            if r.get("efficiency", -9999) >= min_efficiency
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in scored[:top_k]]

    def statistics(self) -> Dict[str, Any]:
        if not self._records:
            return {"total": 0}
        resolved    = [r for r in self._records if r.get("resolved")]
        efficiencies = [r["efficiency"] for r in self._records if "efficiency" in r]
        topologies  = {}
        for r in self._records:
            t = r.get("topology", "unknown")
            topologies[t] = topologies.get(t, 0) + 1
        return {
            "total":            len(self._records),
            "resolve_rate":     len(resolved) / len(self._records),
            "avg_efficiency":   sum(efficiencies) / len(efficiencies) if efficiencies else 0,
            "avg_energy_j":     sum(r.get("energy_j", 0) for r in self._records) / len(self._records),
            "avg_latency_s":    sum(r.get("latency_s", 0) for r in self._records) / len(self._records),
            "topology_counts":  topologies,
        }
