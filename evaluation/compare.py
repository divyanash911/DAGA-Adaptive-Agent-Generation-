"""
DAGA — Evaluation & Comparison Script
======================================
Compares DAGA against baseline agentic approaches on SWE-bench.

What it measures
----------------
1. Correctness  : % resolved (via SWE-bench harness or patch-presence proxy)
2. Efficiency   : tokens, energy (J), latency (s), cost ($) per resolved task
3. Comparative  : Pareto front (resolve rate vs cost), efficiency ratio vs baselines

Baseline approaches supported
------------------------------
- SWE-agent      (Princeton, 2024)    : single GPT-4 agent, fixed ReAct loop
- Agentless      (UIUC, 2024)         : no agent, localise→repair→validate pipeline
- Aider          (paul-gauthier)      : single-model, architect+editor mode
- OpenHands      (All-Hands-AI)       : single CodeAct agent
- DAGA           (this work)          : dynamic topology, multi-tier models

Inputs
------
- DAGA: daga_patches/ directory + summary.json  (produced by swebench_harness.py)
- Baselines: patch directories in the SWE-bench submission format
             OR published resolve-rate numbers + estimated cost figures

Usage
-----
# Full comparison (needs SWE-bench harness installed + Docker):
python -m daga.evaluation.compare \
    --daga_patches   ./daga_patches \
    --daga_summary   ./daga_patches/summary.json \
    --baseline_dir   ./baselines \
    --dataset        princeton-nlp/SWE-bench_Lite \
    --run_harness \
    --output_html    ./eval_report.html

# Quick comparison using published numbers (no Docker needed):
python -m daga.evaluation.compare \
    --daga_summary   ./daga_patches/summary.json \
    --use_published_baselines \
    --output_html    ./eval_report.html
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════
# Cost model  ($ per 1M tokens, approximate as of Q1 2025)
# ══════════════════════════════════════════════════════════════

# OpenRouter pricing as of March 2026 ($/M tokens)
# Free models (:free suffix) cost $0 but have rate limits.
COST_PER_1M_INPUT = {
    # Free tier (rate-limited, $0)
    "llama-3.3-70b-instruct:free":       0.00,
    "qwen3-30b-a3b:free":                0.00,
    "deepseek-r1-distill-qwen-32b:free": 0.00,
    "deepseek-r1:free":                  0.00,
    "qwen3-coder:free":                  0.00,
    # Paid — DeepSeek
    "deepseek-v3.2":                     0.25,
    "deepseek-v3.1":                     0.25,
    "deepseek-r1-0528":                  0.45,
    "deepseek-r1":                       0.45,
    "deepseek-r1-distill-qwen-32b":      0.29,
    # Paid — Qwen
    "qwen3-coder":                       0.22,   # 480B MoE
    "qwen/qwen-2.5-32b-instruct":        0.15,
    "qwen/qwen-2.5-7b-instruct":         0.10,
    # Paid — Mistral (privacy-friendly)
    "devstral-small":                    0.10,
    "devstral-medium":                   0.40,
    "codestral-2501":                    0.30,
    # Paid — Anthropic
    "claude-sonnet-4-6":                 3.00,
    "claude-haiku-4-5":                  0.80,
    # Paid — OpenAI
    "gpt-4o":                            5.00,
    "gpt-4o-mini":                       0.15,
    # Paid — Meta
    "llama-3.3-70b-instruct":            0.12,
    "default":                           0.50,
}
COST_PER_1M_OUTPUT = {
    "llama-3.3-70b-instruct:free":       0.00,
    "qwen3-30b-a3b:free":                0.00,
    "deepseek-r1-distill-qwen-32b:free": 0.00,
    "deepseek-r1:free":                  0.00,
    "qwen3-coder:free":                  0.00,
    "deepseek-v3.2":                     0.38,
    "deepseek-v3.1":                     0.38,
    "deepseek-r1-0528":                  2.15,
    "deepseek-r1":                       2.19,
    "deepseek-r1-distill-qwen-32b":      0.29,
    "qwen3-coder":                       1.00,
    "qwen/qwen-2.5-32b-instruct":        0.58,
    "qwen/qwen-2.5-7b-instruct":         0.30,
    "devstral-small":                    0.30,
    "devstral-medium":                   1.60,
    "codestral-2501":                    0.90,
    "claude-sonnet-4-6":                15.00,
    "claude-haiku-4-5":                  4.00,
    "gpt-4o":                           15.00,
    "gpt-4o-mini":                       0.60,
    "llama-3.3-70b-instruct":            0.40,
    "default":                           1.50,
}


def token_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate $ cost for one model call."""
    m = model_id.lower()
    in_rate  = next((v for k, v in COST_PER_1M_INPUT.items()  if k in m), COST_PER_1M_INPUT["default"])
    out_rate = next((v for k, v in COST_PER_1M_OUTPUT.items() if k in m), COST_PER_1M_OUTPUT["default"])
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


# ══════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════

@dataclass
class InstanceResult:
    instance_id:    str
    resolved:       bool
    tokens:         int   = 0
    energy_j:       float = 0.0
    latency_s:      float = 0.0
    cost_usd:       float = 0.0
    topology:       str   = ""
    complexity:     str   = ""
    patch_lines:    int   = 0


@dataclass
class ApproachMetrics:
    name:           str
    n_total:        int
    n_resolved:     int
    resolve_rate:   float
    avg_tokens:     float = 0.0
    avg_energy_j:   float = 0.0
    avg_latency_s:  float = 0.0
    avg_cost_usd:   float = 0.0
    # Per-resolved (efficiency normalised to successful tasks only)
    tok_per_resolved:    float = 0.0
    energy_per_resolved: float = 0.0
    cost_per_resolved:   float = 0.0
    # Topology breakdown (DAGA only)
    topology_breakdown:  Dict[str, int] = field(default_factory=dict)
    # Per-complexity resolve rate (DAGA only)
    complexity_resolve:  Dict[str, float] = field(default_factory=dict)
    # Raw instances (for Pareto scatter)
    instances: List[InstanceResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# Load DAGA results
# ══════════════════════════════════════════════════════════════

def load_daga_summary(summary_path: str) -> ApproachMetrics:
    """Parse summary.json produced by swebench_harness.py."""
    data: List[Dict[str, Any]] = json.loads(Path(summary_path).read_text())

    instances = []
    topology_breakdown: Dict[str, int] = {}
    complexity_buckets: Dict[str, List[bool]] = {}

    for r in data:
        resolved = r.get("status") == "resolved"
        topo     = r.get("topology", "unknown")
        cx       = r.get("complexity", "unknown")
        tokens   = r.get("tokens", 0)
        energy   = r.get("energy_j", 0.0)
        latency  = r.get("latency_s", 0.0)

        # Rough cost from token count (assume mixed model profile from DAGA run)
        cost = tokens * 0.0000008   # ~$0.80/Mtok blended for default topology mix

        instances.append(InstanceResult(
            instance_id = r.get("instance_id", "?"),
            resolved    = resolved,
            tokens      = tokens,
            energy_j    = energy,
            latency_s   = latency,
            cost_usd    = cost,
            topology    = topo,
            complexity  = cx,
            patch_lines = r.get("patch_lines", 0),
        ))
        topology_breakdown[topo] = topology_breakdown.get(topo, 0) + 1
        complexity_buckets.setdefault(cx, []).append(resolved)

    n_total    = len(instances)
    n_resolved = sum(1 for i in instances if i.resolved)
    resolved_i = [i for i in instances if i.resolved]

    complexity_resolve = {
        cx: sum(vals) / len(vals)
        for cx, vals in complexity_buckets.items()
    }

    return ApproachMetrics(
        name         = "DAGA",
        n_total      = n_total,
        n_resolved   = n_resolved,
        resolve_rate = n_resolved / max(n_total, 1),
        avg_tokens   = sum(i.tokens    for i in instances) / max(n_total, 1),
        avg_energy_j = sum(i.energy_j  for i in instances) / max(n_total, 1),
        avg_latency_s= sum(i.latency_s for i in instances) / max(n_total, 1),
        avg_cost_usd = sum(i.cost_usd  for i in instances) / max(n_total, 1),
        tok_per_resolved    = sum(i.tokens    for i in resolved_i) / max(n_resolved, 1),
        energy_per_resolved = sum(i.energy_j  for i in resolved_i) / max(n_resolved, 1),
        cost_per_resolved   = sum(i.cost_usd  for i in resolved_i) / max(n_resolved, 1),
        topology_breakdown  = topology_breakdown,
        complexity_resolve  = complexity_resolve,
        instances           = instances,
    )


# ══════════════════════════════════════════════════════════════
# Published baseline numbers
# (from official SWE-bench leaderboard + published papers)
# ══════════════════════════════════════════════════════════════

def load_baseline_patches(
    baseline_dir: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    use_harness: bool = False,
    use_proxy: bool = True,
) -> List[ApproachMetrics]:
    """
    Load one or more baseline patch directories and build ApproachMetrics for each.

    Expected directory layout (SWE-bench submission format):

        baseline_dir/
          swe_agent/          <- one subdir per approach, name = approach label
            astropy__astropy-12907.patch
            astropy__astropy-14182.patch
            ...
          agentless/
            astropy__astropy-12907.patch
            ...

    Each .patch file should contain a unified diff (or be empty for no-prediction).
    Token/energy/cost figures are estimated from avg literature values per approach
    when not available in a sidecar metadata.json.

    Sidecar format (optional, place in the subdir):
        metadata.json — {"avg_tokens": 40000, "avg_cost_usd": 0.30, "model": "gpt-4o"}
    """
    import re
    baseline_root = Path(baseline_dir)
    if not baseline_root.exists():
        print(f"[baseline] Directory not found: {baseline_dir}")
        return []

    # Load ground truth instance IDs from dataset if available
    all_instance_ids: List[str] = []
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(dataset, split=split)
        all_instance_ids = [row["instance_id"] for row in ds]
        print(f"[baseline] Loaded {len(all_instance_ids)} instance IDs from {dataset}")
    except Exception:
        print("[baseline] Could not load dataset for instance ID list — using patch filenames only")

    diff_pattern = re.compile(r"^(---|\+\+\+|@@)", re.MULTILINE)
    results: List[ApproachMetrics] = []

    for subdir in sorted(baseline_root.iterdir()):
        if not subdir.is_dir():
            continue
        approach_name = subdir.name

        # Load optional sidecar metadata
        meta: Dict[str, Any] = {}
        meta_path = subdir / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass

        avg_tokens   = float(meta.get("avg_tokens",   40_000))
        avg_cost     = float(meta.get("avg_cost_usd", 0.30))
        avg_energy   = float(meta.get("avg_energy_j", avg_tokens * 0.0004))
        model_id     = meta.get("model", "default")

        # Collect patch files
        patch_files  = {p.stem: p for p in subdir.glob("*.patch")}
        # Determine universe of instances
        universe = all_instance_ids if all_instance_ids else list(patch_files.keys())

        instances: List[InstanceResult] = []
        if use_harness:
            harness_results = run_swebench_harness(str(subdir), dataset, split)
        else:
            harness_results = {}

        for iid in universe:
            if use_harness and harness_results:
                resolved = harness_results.get(iid, False)
            elif use_proxy and iid in patch_files:
                text = patch_files[iid].read_text()
                resolved = bool(text.strip() and diff_pattern.search(text))
            else:
                # No patch file = no prediction = not resolved
                resolved = False

            instances.append(InstanceResult(
                instance_id = iid,
                resolved    = resolved,
                tokens      = int(avg_tokens),
                energy_j    = avg_energy,
                latency_s   = 0.0,
                cost_usd    = avg_cost,
            ))

        n_total    = len(instances)
        n_resolved = sum(1 for i in instances if i.resolved)
        resolved_i = [i for i in instances if i.resolved]

        results.append(ApproachMetrics(
            name         = approach_name,
            n_total      = n_total,
            n_resolved   = n_resolved,
            resolve_rate = n_resolved / max(n_total, 1),
            avg_tokens   = avg_tokens,
            avg_energy_j = avg_energy,
            avg_latency_s= 0.0,
            avg_cost_usd = avg_cost,
            tok_per_resolved    = avg_tokens / max(n_resolved / max(n_total, 1), 0.01),
            energy_per_resolved = avg_energy / max(n_resolved / max(n_total, 1), 0.01),
            cost_per_resolved   = avg_cost   / max(n_resolved / max(n_total, 1), 0.01),
            instances           = instances,
        ))
        print(f"[baseline] {approach_name}: {n_resolved}/{n_total} resolved "
              f"({n_resolved/max(n_total,1)*100:.1f}%)")

    return results


def published_baselines() -> List[ApproachMetrics]:
    """
    Approximate efficiency figures for known approaches on SWE-bench Lite.
    Sources:
      - SWE-bench leaderboard (swebench.com/lite) — resolve rates
      - SWE-agent paper (Yang et al. 2024) — token counts
      - Agentless paper (Xia et al. 2024) — token counts, cost
      - OpenHands / All-Hands-AI public reports
      - Aider benchmark blog posts

    Token estimates are for the full SWE-bench Lite run (300 instances).
    Cost assumes the model each approach uses by default.
    Energy is estimated using MODEL_ENERGY_PROFILE from DAGA.
    """

    # J/token for each approach's primary model
    GPT4O_J_IN  = 0.00400
    GPT4O_J_OUT = 0.00600
    SONNET_J_IN  = 0.00400
    SONNET_J_OUT = 0.00600
    HAIKU_J_IN   = 0.00120
    HAIKU_J_OUT  = 0.00190

    def _make(
        name: str,
        n_total: int,
        resolve_rate: float,
        avg_in_tokens: float,
        avg_out_tokens: float,
        j_in: float,
        j_out: float,
        in_price_1m: float,
        out_price_1m: float,
    ) -> ApproachMetrics:
        n_resolved = int(n_total * resolve_rate)
        avg_tokens = avg_in_tokens + avg_out_tokens
        avg_energy = avg_in_tokens * j_in + avg_out_tokens * j_out
        avg_cost   = (avg_in_tokens * in_price_1m + avg_out_tokens * out_price_1m) / 1_000_000
        # Instances are synthetic for baselines (no per-task breakdown available)
        return ApproachMetrics(
            name         = name,
            n_total      = n_total,
            n_resolved   = n_resolved,
            resolve_rate = resolve_rate,
            avg_tokens   = avg_tokens,
            avg_energy_j = avg_energy,
            avg_latency_s= 0.0,
            avg_cost_usd = avg_cost,
            tok_per_resolved    = avg_tokens   / max(resolve_rate, 0.01),
            energy_per_resolved = avg_energy   / max(resolve_rate, 0.01),
            cost_per_resolved   = avg_cost     / max(resolve_rate, 0.01),
        )

    return [
        # SWE-agent + GPT-4o  (Yang et al. 2024)
        # Resolve rate 18.8% on SWE-bench Lite, ~200k tokens/task average
        _make("SWE-agent (GPT-4o)", 300, 0.188,
              avg_in_tokens=160_000, avg_out_tokens=40_000,
              j_in=GPT4O_J_IN, j_out=GPT4O_J_OUT,
              in_price_1m=5.00, out_price_1m=15.00),

        # Agentless + GPT-4o  (Xia et al. 2024)
        # Resolve rate 27.3%, much cheaper (~40k tokens/task)
        _make("Agentless (GPT-4o)", 300, 0.273,
              avg_in_tokens=30_000, avg_out_tokens=10_000,
              j_in=GPT4O_J_IN, j_out=GPT4O_J_OUT,
              in_price_1m=5.00, out_price_1m=15.00),

        # Aider + Claude Sonnet  (aider.chat benchmark)
        # Resolve rate ~26%, moderate token usage
        _make("Aider (Claude Sonnet)", 300, 0.260,
              avg_in_tokens=80_000, avg_out_tokens=20_000,
              j_in=SONNET_J_IN, j_out=SONNET_J_OUT,
              in_price_1m=3.00, out_price_1m=15.00),

        # OpenHands CodeAct + GPT-4o  (All-Hands-AI 2024)
        # Resolve rate ~26.4%, higher token usage due to multi-turn
        _make("OpenHands (GPT-4o)", 300, 0.264,
              avg_in_tokens=120_000, avg_out_tokens=30_000,
              j_in=GPT4O_J_IN, j_out=GPT4O_J_OUT,
              in_price_1m=5.00, out_price_1m=15.00),

        # Agentless + Claude Haiku (cheapest published result)
        # Resolve rate ~22%, very low cost
        _make("Agentless (Haiku)", 300, 0.220,
              avg_in_tokens=30_000, avg_out_tokens=8_000,
              j_in=HAIKU_J_IN, j_out=HAIKU_J_OUT,
              in_price_1m=0.25, out_price_1m=1.25),
    ]


# ══════════════════════════════════════════════════════════════
# SWE-bench harness runner (optional, needs Docker)
# ══════════════════════════════════════════════════════════════

def run_swebench_harness(
    patches_dir: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    log_dir: str = "./harness_logs",
    num_workers: int = 4,
) -> Dict[str, bool]:
    """
    Run the official SWE-bench Docker evaluation harness.
    Returns dict of instance_id -> resolved.
    Requires: pip install swebench  and  Docker running.
    """
    print(f"[harness] Running SWE-bench evaluation on patches in {patches_dir}")
    print(f"[harness] This will pull Docker images and may take 30-90 minutes.")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    results_file = Path(log_dir) / "results.json"

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--predictions_path", patches_dir,
        "--swe_bench_tasks",  dataset,
        "--split",            split,
        "--log_dir",          log_dir,
        "--num_workers",      str(num_workers),
        "--predictions_path", patches_dir,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if proc.returncode != 0:
            print(f"[harness] ERROR: {proc.stderr[:500]}")
            return {}
        # Parse the results JSON produced by the harness
        if results_file.exists():
            raw = json.loads(results_file.read_text())
            return {k: bool(v) for k, v in raw.items()}
    except FileNotFoundError:
        print("[harness] swebench not installed. pip install swebench")
    except subprocess.TimeoutExpired:
        print("[harness] Timed out after 2h")
    return {}


# ══════════════════════════════════════════════════════════════
# Patch-presence proxy (fast, no Docker)
# ══════════════════════════════════════════════════════════════

def proxy_resolve_from_patches(patches_dir: str) -> Dict[str, bool]:
    """
    A fast proxy for resolution: a non-empty patch that looks like a real
    unified diff is treated as 'likely resolved'. This is not as accurate
    as the harness but is instant and useful for iterative development.
    """
    import re
    diff_pattern = re.compile(r"^(---|\+\+\+|@@)", re.MULTILINE)
    results = {}
    for p in Path(patches_dir).glob("*.patch"):
        text = p.read_text()
        results[p.stem] = bool(text.strip() and diff_pattern.search(text))
    return results


# ══════════════════════════════════════════════════════════════
# Metric computation helpers
# ══════════════════════════════════════════════════════════════

def efficiency_ratio(daga: ApproachMetrics, baseline: ApproachMetrics) -> Dict[str, float]:
    """
    How much more efficient is DAGA vs a baseline, per resolved task?
    Ratio > 1.0 means DAGA uses more; < 1.0 means DAGA uses less (better).
    """
    def _ratio(a: float, b: float) -> float:
        return round(a / b, 3) if b > 0 else float("inf")

    return {
        "tokens":  _ratio(daga.tok_per_resolved,    baseline.tok_per_resolved),
        "energy":  _ratio(daga.energy_per_resolved,  baseline.energy_per_resolved),
        "cost":    _ratio(daga.cost_per_resolved,    baseline.cost_per_resolved),
        "resolve": _ratio(daga.resolve_rate,         baseline.resolve_rate),
    }


def pareto_dominates(a: ApproachMetrics, b: ApproachMetrics) -> bool:
    """Does approach a Pareto-dominate b? (better resolve, lower cost, both non-worse)"""
    return (
        a.resolve_rate   >= b.resolve_rate
        and a.avg_cost_usd <= b.avg_cost_usd
        and (a.resolve_rate > b.resolve_rate or a.avg_cost_usd < b.avg_cost_usd)
    )


# ══════════════════════════════════════════════════════════════
# HTML report generator
# ══════════════════════════════════════════════════════════════

def generate_html_report(
    daga: ApproachMetrics,
    baselines: List[ApproachMetrics],
    output_path: str,
) -> None:
    all_approaches = baselines + [daga]

    # Compute ratios vs each baseline
    ratio_table = {b.name: efficiency_ratio(daga, b) for b in baselines}

    # Pareto front
    pareto_front = [a for a in all_approaches
                    if not any(pareto_dominates(b, a) for b in all_approaches if b is not a)]

    # Chart data
    chart_names   = json.dumps([a.name for a in all_approaches])
    chart_resolve = json.dumps([round(a.resolve_rate * 100, 1) for a in all_approaches])
    chart_cost    = json.dumps([round(a.avg_cost_usd, 4) for a in all_approaches])
    chart_energy  = json.dumps([round(a.avg_energy_j, 2) for a in all_approaches])
    chart_tokens  = json.dumps([round(a.avg_tokens / 1000, 1) for a in all_approaches])

    # DAGA topology breakdown
    topo_labels = json.dumps(list(daga.topology_breakdown.keys()))
    topo_counts = json.dumps(list(daga.topology_breakdown.values()))

    # DAGA complexity resolve rates
    cx_labels = json.dumps(list(daga.complexity_resolve.keys()))
    cx_rates  = json.dumps([round(v * 100, 1) for v in daga.complexity_resolve.values()])

    # Scatter points for Pareto plot (resolve% vs cost)
    scatter_data = json.dumps([
        {"x": round(a.avg_cost_usd, 4), "y": round(a.resolve_rate * 100, 1),
         "label": a.name, "pareto": a in pareto_front}
        for a in all_approaches
    ])

    def _row(a: ApproachMetrics, highlight: bool = False) -> str:
        hl = ' style="background:var(--color-background-info)"' if highlight else ""
        pr = " ★" if a in pareto_front else ""
        return (
            f"<tr{hl}>"
            f"<td><strong>{a.name}{pr}</strong></td>"
            f"<td>{a.resolve_rate*100:.1f}%</td>"
            f"<td>{a.n_resolved}/{a.n_total}</td>"
            f"<td>{a.avg_tokens/1000:.0f}k</td>"
            f"<td>{a.avg_energy_j:.1f}</td>"
            f"<td>${a.avg_cost_usd:.4f}</td>"
            f"<td>{a.tok_per_resolved/1000:.0f}k</td>"
            f"<td>${a.cost_per_resolved:.3f}</td>"
            f"</tr>"
        )

    rows = "\n".join(_row(a, a.name == "DAGA") for a in all_approaches)

    ratio_rows = ""
    for bname, ratios in ratio_table.items():
        def _cell(v: float, invert: bool = False) -> str:
            better = (v < 1.0) if not invert else (v > 1.0)
            color  = "#2e7d32" if better else "#c62828"
            arrow  = "▼" if v < 1.0 else ("▲" if v > 1.0 else "=")
            pct    = abs(1 - v) * 100
            return f'<td style="color:{color}">{arrow} {pct:.0f}% ({v:.2f}x)</td>'
        ratio_rows += (
            f"<tr><td>vs {bname}</td>"
            + _cell(ratios["tokens"])
            + _cell(ratios["energy"])
            + _cell(ratios["cost"])
            + _cell(ratios["resolve"], invert=True)
            + "</tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DAGA Evaluation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root {{
    --color-bg: #ffffff;
    --color-surface: #f8f8f6;
    --color-border: rgba(0,0,0,0.1);
    --color-text: #1a1a18;
    --color-muted: #6b6b65;
    --color-accent: #534AB7;
    --color-background-info: #E6F1FB;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --color-bg: #1a1a18;
      --color-surface: #242422;
      --color-border: rgba(255,255,255,0.12);
      --color-text: #e8e6dc;
      --color-muted: #9c9a92;
      --color-accent: #AFA9EC;
      --color-background-info: #0c1a2e;
    }}
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: var(--color-bg);
         color: var(--color-text); padding: 2rem; max-width: 1100px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; font-weight: 500; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; font-weight: 500; margin: 2rem 0 0.75rem; color: var(--color-accent); }}
  p.sub {{ font-size: 0.875rem; color: var(--color-muted); margin-bottom: 2rem; }}
  .grid-4 {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 2rem; }}
  .card {{ background: var(--color-surface); border: 0.5px solid var(--color-border);
           border-radius: 10px; padding: 1rem 1.25rem; }}
  .card .label {{ font-size: 12px; color: var(--color-muted); margin-bottom: 6px; }}
  .card .value {{ font-size: 22px; font-weight: 500; }}
  .card .sub   {{ font-size: 12px; color: var(--color-muted); margin-top: 2px; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 2rem; }}
  .chart-wrap {{ background: var(--color-surface); border: 0.5px solid var(--color-border);
                 border-radius: 10px; padding: 1.25rem; }}
  .chart-wrap h3 {{ font-size: 13px; font-weight: 500; color: var(--color-muted); margin-bottom: 12px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 1rem; }}
  th {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--color-border);
        color: var(--color-muted); font-weight: 400; font-size: 12px; }}
  td {{ padding: 8px 10px; border-bottom: 0.5px solid var(--color-border); }}
  tr:last-child td {{ border-bottom: none; }}
  .pareto-note {{ font-size: 12px; color: var(--color-muted); margin-top: 0.5rem; }}
  .tag {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 4px;
          background: var(--color-background-info); color: var(--color-accent); margin-left: 6px; }}
</style>
</head>
<body>
<h1>DAGA — SWE-bench Evaluation Report</h1>
<p class="sub">Comparing DAGA against published baselines on SWE-bench Lite.
  ★ = Pareto-optimal (not dominated on resolve rate + cost simultaneously).</p>

<div class="grid-4">
  <div class="card">
    <div class="label">Resolve rate</div>
    <div class="value">{daga.resolve_rate*100:.1f}%</div>
    <div class="sub">{daga.n_resolved} / {daga.n_total} tasks</div>
  </div>
  <div class="card">
    <div class="label">Avg cost / task</div>
    <div class="value">${daga.avg_cost_usd:.4f}</div>
    <div class="sub">all tasks, incl. unresolved</div>
  </div>
  <div class="card">
    <div class="label">Cost / resolved task</div>
    <div class="value">${daga.cost_per_resolved:.3f}</div>
    <div class="sub">normalised efficiency</div>
  </div>
  <div class="card">
    <div class="label">Avg energy / task</div>
    <div class="value">{daga.avg_energy_j:.1f} J</div>
    <div class="sub">estimated</div>
  </div>
</div>

<h2>Comparison table</h2>
<table>
<thead><tr>
  <th>Approach</th><th>Resolve %</th><th>Resolved</th>
  <th>Avg tokens</th><th>Avg energy (J)</th><th>Avg cost</th>
  <th>Tokens/resolved</th><th>Cost/resolved</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="pareto-note">★ = Pareto-optimal (no other approach is better on both resolve rate and cost)</p>

<h2>Efficiency ratio vs baselines (DAGA relative)</h2>
<table>
<thead><tr>
  <th>Comparison</th><th>Tokens/resolved</th><th>Energy/resolved</th>
  <th>Cost/resolved</th><th>Resolve rate</th>
</tr></thead>
<tbody>{ratio_rows}</tbody>
</table>
<p class="pareto-note">▼ = DAGA uses less (better for efficiency metrics); ▲ = DAGA uses more.
For resolve rate: ▲ = DAGA resolves more (better).</p>

<div class="chart-grid">
  <div class="chart-wrap">
    <h3>Resolve rate (%)</h3>
    <div style="position:relative;height:220px"><canvas id="c1"></canvas></div>
  </div>
  <div class="chart-wrap">
    <h3>Pareto front — resolve rate vs avg cost per task ($)</h3>
    <div style="position:relative;height:220px"><canvas id="c2"></canvas></div>
  </div>
  <div class="chart-wrap">
    <h3>Avg tokens per task (k)</h3>
    <div style="position:relative;height:220px"><canvas id="c3"></canvas></div>
  </div>
  <div class="chart-wrap">
    <h3>Avg energy per task (J)</h3>
    <div style="position:relative;height:220px"><canvas id="c4"></canvas></div>
  </div>
</div>

<h2>DAGA-specific: topology routing breakdown</h2>
<div class="chart-grid">
  <div class="chart-wrap">
    <h3>Tasks routed per topology</h3>
    <div style="position:relative;height:200px"><canvas id="c5"></canvas></div>
  </div>
  <div class="chart-wrap">
    <h3>Resolve rate by complexity</h3>
    <div style="position:relative;height:200px"><canvas id="c6"></canvas></div>
  </div>
</div>

<script>
const NAMES   = {chart_names};
const RESOLVE = {chart_resolve};
const COST    = {chart_cost};
const ENERGY  = {chart_energy};
const TOKENS  = {chart_tokens};
const SCATTER = {scatter_data};
const TOPO_L  = {topo_labels};
const TOPO_C  = {topo_counts};
const CX_L    = {cx_labels};
const CX_R    = {cx_rates};

const dark    = matchMedia('(prefers-color-scheme: dark)').matches;
const ACCENT  = dark ? '#AFA9EC' : '#534AB7';
const TEAL    = dark ? '#5DCAA5' : '#0F6E56';
const GRAY    = dark ? '#888780' : '#5F5E5A';
const AMBER   = dark ? '#EF9F27' : '#854F0B';
const TEXT    = dark ? '#c2c0b6' : '#3d3d3a';
const GRID    = dark ? 'rgba(255,255,255,0.07)' : 'rgba(0,0,0,0.07)';

const colors = NAMES.map(n => n === 'DAGA' ? ACCENT : GRAY);
const baseOpts = {{
  responsive: true, maintainAspectRatio: false,
  plugins: {{ legend: {{ display: false }} }},
  scales: {{
    x: {{ ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
    y: {{ ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
  }}
}};

new Chart('c1', {{
  type: 'bar',
  data: {{ labels: NAMES, datasets: [{{ data: RESOLVE, backgroundColor: colors, borderRadius: 4 }}] }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, max: 100 }} }} }}
}});

const scatterDatasets = SCATTER.map(p => ({{
  label: p.label,
  data: [{{ x: p.x, y: p.y }}],
  backgroundColor: p.label === 'DAGA' ? ACCENT : (p.pareto ? TEAL : GRAY),
  pointRadius: p.label === 'DAGA' ? 10 : 7,
  pointStyle: p.pareto ? 'star' : 'circle',
}}));
new Chart('c2', {{
  type: 'scatter',
  data: {{ datasets: scatterDatasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{ display: false }},
      tooltip: {{ callbacks: {{ label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y + '% @ $' + ctx.parsed.x }} }}
    }},
    scales: {{
      x: {{ title: {{ display: true, text: 'Avg cost / task ($)', color: TEXT, font: {{ size: 11 }} }},
            ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
      y: {{ title: {{ display: true, text: 'Resolve rate (%)', color: TEXT, font: {{ size: 11 }} }},
            ticks: {{ color: TEXT, font: {{ size: 11 }} }}, grid: {{ color: GRID }} }},
    }}
  }}
}});

new Chart('c3', {{
  type: 'bar',
  data: {{ labels: NAMES, datasets: [{{ data: TOKENS, backgroundColor: colors, borderRadius: 4 }}] }},
  options: baseOpts
}});

new Chart('c4', {{
  type: 'bar',
  data: {{ labels: NAMES, datasets: [{{ data: ENERGY, backgroundColor: colors, borderRadius: 4 }}] }},
  options: baseOpts
}});

new Chart('c5', {{
  type: 'doughnut',
  data: {{ labels: TOPO_L, datasets: [{{ data: TOPO_C,
    backgroundColor: ['#534AB7','#0F6E56','#854F0B','#993C1D','#185FA5'] }}] }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'right', labels: {{ color: TEXT, font: {{ size: 11 }}, boxWidth: 12 }} }} }} }}
}});

new Chart('c6', {{
  type: 'bar',
  data: {{ labels: CX_L, datasets: [{{ data: CX_R, backgroundColor: ACCENT, borderRadius: 4 }}] }},
  options: {{ ...baseOpts, scales: {{ ...baseOpts.scales, y: {{ ...baseOpts.scales.y, max: 100 }} }} }}
}});
</script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"[eval] Report written to {output_path}")


# ══════════════════════════════════════════════════════════════
# Print table to stdout
# ══════════════════════════════════════════════════════════════

def print_report(daga: ApproachMetrics, baselines: List[ApproachMetrics]) -> None:
    all_ap = baselines + [daga]

    w = 72
    print("=" * w)
    print("DAGA Evaluation Report — SWE-bench Lite")
    print("=" * w)

    header = f"{'Approach':<28} {'Resolve':>8} {'Tok/task':>10} {'J/task':>8} {'$/task':>8} {'$/res':>8}"
    print(header)
    print("-" * w)
    for a in all_ap:
        marker = " *" if a.name == "DAGA" else "  "
        print(
            f"{marker}{a.name:<26} "
            f"{a.resolve_rate*100:>7.1f}% "
            f"{a.avg_tokens/1000:>9.0f}k "
            f"{a.avg_energy_j:>7.1f} "
            f"${a.avg_cost_usd:>6.4f} "
            f"${a.cost_per_resolved:>6.3f}"
        )
    print("=" * w)
    print("* DAGA  |  Columns: resolve rate, avg tokens/task, avg energy,")
    print("  avg cost/task, cost per resolved task")

    print("\nEfficiency ratios (DAGA vs baselines, per resolved task):")
    print(f"  {'Baseline':<30} {'Tokens':>8} {'Energy':>8} {'Cost':>8} {'Resolve':>8}")
    print(f"  {'-'*58}")
    for b in baselines:
        r = efficiency_ratio(daga, b)
        def _fmt(v: float, invert: bool = False) -> str:
            better = (v < 1.0) if not invert else (v > 1.0)
            sym = "▼" if v < 1.0 else ("▲" if v > 1.0 else "=")
            return f"{sym}{v:.2f}x"
        print(
            f"  vs {b.name:<27}"
            f" {_fmt(r['tokens']):>8}"
            f" {_fmt(r['energy']):>8}"
            f" {_fmt(r['cost']):>8}"
            f" {_fmt(r['resolve'], invert=True):>8}"
        )
    print("\n  ▼ = DAGA uses less / resolves fewer  ▲ = DAGA uses more / resolves more")

    # Pareto
    all_ap2 = baselines + [daga]
    pareto = [a for a in all_ap2
              if not any(pareto_dominates(b, a) for b in all_ap2 if b is not a)]
    print(f"\nPareto-optimal approaches (best resolve/cost tradeoff):")
    for a in pareto:
        print(f"  ★ {a.name}  ({a.resolve_rate*100:.1f}% resolved, ${a.avg_cost_usd:.4f}/task)")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="DAGA evaluation and comparison against SWE-bench baselines",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--daga_summary",   default="./daga_patches/summary.json",
                   help="Path to summary.json from swebench_harness.py")
    p.add_argument("--daga_patches",   default="./daga_patches",
                   help="Directory of DAGA .patch files")
    p.add_argument("--baseline_dir",   default=None,
                   help="Directory containing baseline patch subdirs (optional)")
    p.add_argument("--dataset",        default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split",          default="test")
    p.add_argument("--run_harness",    action="store_true",
                   help="Run official SWE-bench Docker harness for correctness eval")
    p.add_argument("--proxy_resolve",  action="store_true",
                   help="Use patch-presence as a fast proxy for resolution (no Docker)")
    p.add_argument("--use_published_baselines", action="store_true", default=True,
                   help="Include published baseline numbers even when --baseline_dir is set")
    p.add_argument("--no_published_baselines", action="store_true", default=False,
                   help="Only use real patches from --baseline_dir, skip published numbers")
    p.add_argument("--output_html",    default="./daga_eval_report.html")
    p.add_argument("--num_workers",    type=int, default=4,
                   help="Parallel workers for Docker harness")
    args = p.parse_args()

    # ── Load DAGA results ──────────────────────────────────
    if not Path(args.daga_summary).exists():
        print(f"ERROR: {args.daga_summary} not found. Run swebench_harness.py first.")
        sys.exit(1)

    daga = load_daga_summary(args.daga_summary)

    # Optionally refine resolve flags via harness
    if args.run_harness:
        harness_results = run_swebench_harness(
            args.daga_patches, args.dataset, args.split,
            num_workers=args.num_workers,
        )
        if harness_results:
            for inst in daga.instances:
                inst.resolved = harness_results.get(inst.instance_id, inst.resolved)
            resolved = sum(1 for i in daga.instances if i.resolved)
            daga.n_resolved  = resolved
            daga.resolve_rate = resolved / max(daga.n_total, 1)
            print(f"[eval] Harness-corrected resolve rate: {daga.resolve_rate*100:.1f}%")
    elif args.proxy_resolve:
        proxy = proxy_resolve_from_patches(args.daga_patches)
        for inst in daga.instances:
            inst.resolved = proxy.get(inst.instance_id, inst.resolved)
        resolved = sum(1 for i in daga.instances if i.resolved)
        daga.n_resolved   = resolved
        daga.resolve_rate = resolved / max(daga.n_total, 1)
        print(f"[eval] Proxy resolve rate: {daga.resolve_rate*100:.1f}%")

    # ── Load baselines ─────────────────────────────────────
    baselines: List[ApproachMetrics] = []

    if args.baseline_dir:
        # Load real patches from subdirectories
        loaded = load_baseline_patches(
            args.baseline_dir,
            dataset     = args.dataset,
            split       = args.split,
            use_harness = args.run_harness,
            use_proxy   = args.proxy_resolve or (not args.run_harness),
        )
        baselines.extend(loaded)

    if (args.use_published_baselines and not args.no_published_baselines) or not baselines:
        # Merge or fall back to published numbers
        loaded_names = {b.name for b in baselines}
        for pub in published_baselines():
            if pub.name not in loaded_names:
                baselines.append(pub)

    # ── Print to console ───────────────────────────────────
    print_report(daga, baselines)

    # ── Generate HTML report ───────────────────────────────
    generate_html_report(daga, baselines, args.output_html)
    print(f"\nOpen {args.output_html} in a browser for the full visual report.")


if __name__ == "__main__":
    main()