"""
DAGA — Efficiency-Aware Feedback Loop
Analyses accumulated experience records to:
  1. Compute per-topology efficiency statistics
  2. Update deterministic rule thresholds (e.g. complexity breakpoints)
  3. Build a summary for the meta-agent router context
  4. Log actionable insights for human review

This is intentionally simple and interpretable.
The "learning" is statistical rule refinement, not gradient-based training.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from daga.core.models import AgentTopology, ModelTier, TaskComplexity, TaskDomain
from daga.telemetry.collector import ExperienceStore


# ──────────────────────────────────────────────
# Per-topology statistics
# ──────────────────────────────────────────────

def _topology_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Group by topology and compute key stats."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        t = r.get("topology", "unknown")
        groups[t].append(r)

    stats: Dict[str, Dict[str, float]] = {}
    for topology, recs in groups.items():
        n = len(recs)
        resolved = [r for r in recs if r.get("resolved")]
        energies = [r["energy_j"]  for r in recs if "energy_j"  in r]
        latencies= [r["latency_s"] for r in recs if "latency_s" in r]
        efficiencies = [r["efficiency"] for r in recs if "efficiency" in r]
        stats[topology] = {
            "count":        n,
            "resolve_rate": len(resolved) / n if n else 0,
            "avg_energy_j": sum(energies)    / len(energies)    if energies    else 0,
            "avg_latency_s":sum(latencies)   / len(latencies)   if latencies   else 0,
            "avg_efficiency":sum(efficiencies)/len(efficiencies) if efficiencies else 0,
        }
    return stats


def _complexity_topology_matrix(
    records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    For each (complexity, topology) pair, compute average efficiency.
    Used to recommend better topologies for specific complexity levels.
    """
    matrix: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        c = r.get("complexity", "unknown")
        t = r.get("topology",   "unknown")
        if "efficiency" in r:
            matrix[c][t].append(r["efficiency"])

    result: Dict[str, Dict[str, float]] = {}
    for complexity, topologies in matrix.items():
        result[complexity] = {
            t: sum(effs) / len(effs)
            for t, effs in topologies.items()
        }
    return result


# ──────────────────────────────────────────────
# Routing recommendation
# ──────────────────────────────────────────────

def best_topology_for_complexity(
    matrix: Dict[str, Dict[str, float]],
    complexity: str,
    min_samples: int = 3,
    records: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Returns the empirically best topology for a given complexity level,
    but only if we have sufficient samples.
    """
    if complexity not in matrix:
        return None

    row = matrix[complexity]
    # Check sample count
    if records is not None:
        counts: Dict[str, int] = defaultdict(int)
        for r in records:
            if r.get("complexity") == complexity:
                counts[r.get("topology", "?")] += 1
        row = {t: eff for t, eff in row.items() if counts.get(t, 0) >= min_samples}

    if not row:
        return None
    return max(row, key=lambda t: row[t])


# ──────────────────────────────────────────────
# Main feedback loop
# ──────────────────────────────────────────────

class FeedbackLoop:
    """
    Analyses the experience store and surfaces routing recommendations
    and efficiency insights.
    """

    def __init__(
        self,
        store: ExperienceStore,
        min_records_for_update: int = 10,
    ) -> None:
        self._store = store
        self._min_records = min_records_for_update

    def analyse(self) -> Dict[str, Any]:
        """Full analysis pass. Returns a structured report."""
        records = self._store._records
        if len(records) < self._min_records:
            return {
                "status": "insufficient_data",
                "records": len(records),
                "min_required": self._min_records,
            }

        topo_stats = _topology_stats(records)
        matrix     = _complexity_topology_matrix(records)

        # Best topology per complexity from data
        complexity_recommendations = {}
        for cx in [c.value for c in TaskComplexity]:
            best = best_topology_for_complexity(matrix, cx, min_samples=3, records=records)
            if best:
                complexity_recommendations[cx] = best

        # Identify over-used expensive topologies
        waste_flags = []
        for topology, stats in topo_stats.items():
            if (
                stats["resolve_rate"] < 0.5
                and topology in ("hierarchical", "parallel_ensemble")
                and stats["avg_energy_j"] > 50
            ):
                waste_flags.append(
                    f"{topology}: low resolve rate ({stats['resolve_rate']:.1%}) "
                    f"but high energy ({stats['avg_energy_j']:.1f} J)"
                )

        return {
            "status":                    "ok",
            "total_records":             len(records),
            "topology_stats":            topo_stats,
            "complexity_matrix":         matrix,
            "complexity_recommendations":complexity_recommendations,
            "waste_flags":               waste_flags,
            "global_stats":              self._store.statistics(),
        }

    def experience_summary_for_meta_agent(
        self, profile: Any, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Build a concise experience summary to inject into the meta-agent prompt.
        """
        similar  = self._store.retrieve_similar(profile, top_k=top_k)
        analysis = self.analyse()

        cx = profile.complexity.value
        best_topo = analysis.get("complexity_recommendations", {}).get(cx)

        return {
            "similar_experiences": similar,
            "best_topology_for_complexity": best_topo,
            "global_resolve_rate": analysis.get("global_stats", {}).get("resolve_rate"),
            "avg_energy_best_topo": (
                analysis.get("topology_stats", {}).get(best_topo, {}).get("avg_energy_j")
                if best_topo else None
            ),
        }

    def print_report(self) -> None:
        report = self.analyse()
        print("=" * 60)
        print("DAGA Efficiency Report")
        print("=" * 60)
        if report["status"] == "insufficient_data":
            print(f"Not enough data ({report['records']}/{report['min_required']} records)")
            return

        g = report["global_stats"]
        print(f"Total tasks:    {report['total_records']}")
        print(f"Resolve rate:   {g.get('resolve_rate', 0):.1%}")
        print(f"Avg energy:     {g.get('avg_energy_j', 0):.2f} J")
        print(f"Avg latency:    {g.get('avg_latency_s', 0):.1f} s")
        print(f"Avg efficiency: {g.get('avg_efficiency', 0):.4f}")

        print("\nTopology statistics:")
        for t, s in sorted(report["topology_stats"].items(),
                           key=lambda x: -x[1]["avg_efficiency"]):
            print(f"  {t:30s} resolve={s['resolve_rate']:.1%} "
                  f"energy={s['avg_energy_j']:.2f}J latency={s['avg_latency_s']:.1f}s "
                  f"score={s['avg_efficiency']:.4f} (n={s['count']})")

        if report["complexity_recommendations"]:
            print("\nData-driven topology recommendations by complexity:")
            for cx, topo in report["complexity_recommendations"].items():
                print(f"  {cx:12s} → {topo}")

        if report["waste_flags"]:
            print("\nWaste flags:")
            for flag in report["waste_flags"]:
                print(f"  ⚠ {flag}")
        print("=" * 60)
