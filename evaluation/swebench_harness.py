"""
DAGA — SWE-bench Evaluation Harness
Runs the DAGA pipeline against SWE-bench Lite / Verified instances
and produces a patch directory in the SWE-bench submission format.

Every decision, git operation, model call, tool call, routing choice,
energy reading, and timing is logged at full granularity.

Usage:
    python -m daga.evaluation.swebench_harness \
        --dataset princeton-nlp/SWE-bench_Lite \
        --split test \
        --sla_target balanced \
        --output_dir ./patches \
        --openrouter_key sk-or-v1-... \
        --cache_dir ~/daga_repo_cache \
        --log_file ./daga_run.log
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from daga.backends.registry import build_default_registry
from daga.core.models import SLATarget
from daga.pipeline import DAGAPipeline, PipelineConfig
from daga.tools.registry import build_default_tool_registry


# ══════════════════════════════════════════════════════════════
# Logging  — dual output: console (ANSI colour) + JSONL file
# ══════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _banner(msg: str, char: str = "═", width: int = 72) -> str:
    pad  = max(0, width - len(msg) - 4)
    left = pad // 2
    return f"{char * left}  {msg}  {char * (pad - left)}"


class DAGALogger:
    """
    Structured logger.  Every call prints a timestamped line to stdout
    and appends a JSON record to log_file (if set).
    """

    ANSI = {
        "reset":   "\033[0m",
        "bold":    "\033[1m",
        "dim":     "\033[2m",
        "cyan":    "\033[36m",
        "green":   "\033[32m",
        "yellow":  "\033[33m",
        "red":     "\033[31m",
        "blue":    "\033[34m",
        "magenta": "\033[35m",
        "white":   "\033[37m",
    }

    def __init__(self, log_file: Optional[str] = None, no_color: bool = False) -> None:
        self._log_file = Path(log_file) if log_file else None
        self._no_color = no_color or not sys.stdout.isatty()
        self._lock     = threading.Lock()
        if self._log_file:
            self._log_file.parent.mkdir(parents=True, exist_ok=True)

    def _c(self, color: str, text: str) -> str:
        if self._no_color:
            return text
        return f"{self.ANSI.get(color,'')}{text}{self.ANSI['reset']}"

    def _emit(self, level: str, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        ts   = _ts()
        line = f"{self._c('dim', ts)}  {tag}  {msg}"
        with self._lock:
            print(line, flush=True)
            if self._log_file:
                rec = {"ts": ts, "level": level, "tag": tag, "msg": msg}
                if data:
                    rec["data"] = data
                with self._log_file.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    def banner(self, msg: str) -> None:
        line = _banner(msg)
        print(f"\n{self._c('bold', line)}\n", flush=True)

    def section(self, msg: str) -> None:
        ts = _ts()
        print(f"\n{self._c('dim', ts)}  {self._c('cyan', '┌──')}  {self._c('cyan', msg)}\n",
              flush=True)

    def info(self, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("INFO", self._c("blue", f"[{tag}]"), msg, data)

    def success(self, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("OK", self._c("green", f"[{tag}]"), self._c("green", msg), data)

    def warn(self, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("WARN", self._c("yellow", f"[{tag}]"), self._c("yellow", msg), data)

    def error(self, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("ERR", self._c("red", f"[{tag}]"), self._c("red", msg), data)

    def decision(self, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("DECISION", self._c("magenta", "[ROUTE]"), self._c("magenta", msg), data)

    def metric(self, tag: str, msg: str, data: Optional[Dict] = None) -> None:
        self._emit("METRIC", self._c("white", f"[{tag}]"), msg, data)

    def step(self, tag: str, msg: str) -> None:
        self._emit("STEP", self._c("cyan", f"  ↳ [{tag}]"), msg)

    def kv(self, tag: str, pairs: Dict[str, Any]) -> None:
        parts = "  ".join(
            f"{self._c('dim', k)}={self._c('white', str(v))}"
            for k, v in pairs.items()
        )
        self._emit("KV", self._c("blue", f"[{tag}]"), parts)


# ══════════════════════════════════════════════════════════════
# Repo cache  — blobless clone once, copy-on-use per instance
# ══════════════════════════════════════════════════════════════

_REPO_CACHE: Dict[str, str] = {}   # safe_name -> cached_dir path


def _run_cmd(
    cmd: str,
    cwd: Optional[str],
    timeout: int,
    log: DAGALogger,
    tag: str,
    stream: bool = False,
) -> subprocess.CompletedProcess:
    """Run a shell command, optionally streaming stdout line-by-line."""
    log.step(tag, f"$ {cmd}")
    t0 = time.perf_counter()

    if stream:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        lines: List[str] = []
        try:
            for raw_line in proc.stdout:          # type: ignore[union-attr]
                stripped = raw_line.rstrip()
                if stripped:
                    log.step(tag, f"  git> {stripped}")
                    lines.append(stripped)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.error(tag, f"Timed out after {timeout}s")
            return subprocess.CompletedProcess(cmd, 1, "", "timeout")
        elapsed = time.perf_counter() - t0
        log.step(tag, f"  exit={proc.returncode}  elapsed={elapsed:.1f}s")
        return subprocess.CompletedProcess(cmd, proc.returncode, "\n".join(lines), "")
    else:
        try:
            r = subprocess.run(cmd, shell=True, cwd=cwd,
                               capture_output=True, text=True, timeout=timeout)
            elapsed = time.perf_counter() - t0
            log.step(tag, f"  exit={r.returncode}  elapsed={elapsed:.1f}s")
            if r.returncode != 0 and r.stderr:
                log.step(tag, f"  stderr: {r.stderr.strip()[:300]}")
            return r
        except subprocess.TimeoutExpired:
            log.error(tag, f"Timed out after {timeout}s")
            return subprocess.CompletedProcess(cmd, 1, "", "timeout")


def setup_repo(
    instance: Dict[str, Any],
    workdir: str,
    log: DAGALogger,
    timeout: int = 600,
    cache_dir: Optional[str] = None,
) -> bool:
    """
    Prepare repo for one SWE-bench instance.

    Steps (all logged):
      1. Check in-process cache → skip clone entirely if hit
      2. Check on-disk cache   → skip clone, do fetch only
      3. Blobless clone        → cache for future runs
      4. cp cache → workdir
      5. git checkout <base_commit>
      6. git status (transparency)
      7. pip install -e . (best-effort)
    """
    repo        = instance.get("repo", "")
    base_commit = instance.get("base_commit", "")
    repo_url    = f"https://github.com/{repo}.git"
    safe_name   = repo.replace("/", "__")
    cache_root  = Path(cache_dir or "/tmp/daga_repo_cache")
    cached_dir  = cache_root / safe_name
    wp          = Path(workdir)

    log.section(f"Repo setup  {repo}  @  {base_commit[:12]}")
    log.kv("setup", {
        "repo":        repo,
        "commit":      base_commit[:12],
        "workdir":     workdir,
        "cache_root":  str(cache_root),
        "timeout_s":   timeout,
    })

    # Clean workdir
    if wp.exists():
        log.step("setup", f"rmtree {workdir}")
        shutil.rmtree(wp)
    wp.mkdir(parents=True)

    # ── Cache check ──────────────────────────────────────────
    if safe_name in _REPO_CACHE and cached_dir.exists():
        log.success("cache", f"In-process HIT: {repo}")
    elif cached_dir.exists():
        log.success("cache", f"On-disk HIT: {cached_dir}")
        _REPO_CACHE[safe_name] = str(cached_dir)
        log.step("cache", "Fetching any new commits from origin…")
        _run_cmd(f"git -C {cached_dir} fetch --quiet origin",
                 cwd=None, timeout=60, log=log, tag="cache")
    else:
        log.info("cache", f"MISS — cloning {repo} (blobless)…")
        cache_root.mkdir(parents=True, exist_ok=True)
        clone_cmd = (
            f"GIT_TERMINAL_PROMPT=0 git clone "
            f"--filter=blob:none --progress "
            f"{repo_url} {cached_dir}"
        )
        pr = _run_cmd(clone_cmd, cwd=None, timeout=timeout,
                      log=log, tag="clone", stream=True)
        if pr.returncode != 0:
            log.warn("clone", "Blobless clone failed — retrying as full clone")
            pr2 = _run_cmd(
                f"GIT_TERMINAL_PROMPT=0 git clone --progress {repo_url} {cached_dir}",
                cwd=None, timeout=timeout, log=log, tag="clone", stream=True,
            )
            if pr2.returncode != 0:
                log.error("clone", f"Both clone strategies failed for {repo}")
                return False
        _REPO_CACHE[safe_name] = str(cached_dir)
        log.success("cache", f"Cached → {cached_dir}")

    # ── Copy to workdir ──────────────────────────────────────
    log.step("setup", f"cp {cached_dir} → {workdir}")
    t_cp = time.perf_counter()
    pr = _run_cmd(f"cp -r {cached_dir}/. {workdir}",
                  cwd=None, timeout=120, log=log, tag="setup")
    if pr.returncode != 0:
        log.error("setup", "cp failed — aborting")
        return False
    log.step("setup", f"copy done  {time.perf_counter()-t_cp:.1f}s")

    # ── Checkout ─────────────────────────────────────────────
    log.step("git", f"checkout {base_commit[:12]}")
    co = _run_cmd(f"git checkout {base_commit} -q",
                  cwd=workdir, timeout=120, log=log, tag="git")
    if co.returncode != 0:
        log.warn("git", "Clean checkout failed — retrying with --force")
        _run_cmd(f"git checkout --force {base_commit}",
                 cwd=workdir, timeout=120, log=log, tag="git")

    # Confirm HEAD
    head = _run_cmd("git rev-parse HEAD", cwd=workdir, timeout=10,
                    log=log, tag="git").stdout.strip()[:12]
    if head == base_commit[:12]:
        log.success("git", f"HEAD confirmed: {head}")
    else:
        log.warn("git", f"HEAD mismatch — got {head}, expected {base_commit[:12]}")

    # Full git status (transparency)
    st = _run_cmd("git status", cwd=workdir, timeout=10, log=log, tag="git")
    for line in st.stdout.strip().splitlines():
        log.step("git status", line)

    # git log --oneline -5
    gl = _run_cmd("git log --oneline -5", cwd=workdir, timeout=10,
                  log=log, tag="git")
    log.step("git log", f"Recent commits:\n{gl.stdout.strip()}")

    # ── Install ──────────────────────────────────────────────
    has_setup = any(
        (wp / f).exists()
        for f in ("setup.py", "pyproject.toml", "setup.cfg")
    )
    if has_setup:
        log.step("install", "pip install -e . --no-deps -q")
        inst = _run_cmd(
            "pip install -e . -q --no-deps 2>&1 | tail -5",
            cwd=workdir, timeout=120, log=log, tag="install",
        )
        if inst.returncode == 0:
            log.success("install", "Package installed OK")
        else:
            log.warn("install", "pip install non-zero (may still work)")
    else:
        log.step("install", "No setup file — skipping install")

    log.success("setup", f"Repo ready: {repo} @ {head}")
    return True


# ══════════════════════════════════════════════════════════════
# Decision / trace loggers
# ══════════════════════════════════════════════════════════════

def _log_profile(profile: Any, log: DAGALogger) -> None:
    log.section("Task profile")
    log.kv("profile", {
        "task_id":    profile.task_id,
        "domain":     profile.domain.value,
        "complexity": profile.complexity.value,
        "sla":        profile.sla_target.value,
    })
    log.kv("profile", {
        "repo_files":   profile.repo_file_count,
        "affected_est": profile.affected_files_estimate,
        "tokens":       profile.token_count,
        "entropy":      round(profile.entropy, 3),
        "ner":          profile.named_entity_count,
        "has_tests":    profile.has_tests,
        "language":     profile.language,
        "deadline_s":   profile.deadline_seconds,
        "max_energy_j": profile.max_energy_joules,
    })


def _log_plan(plan: Any, log: DAGALogger) -> None:
    log.section("Architecture plan")
    log.decision(
        f"topology={plan.topology.value}  source={plan.routing_source}",
        data={"topology": plan.topology.value, "source": plan.routing_source},
    )
    log.decision(f"Reasoning: {plan.reasoning}")
    if plan.predicted_latency_s or plan.predicted_energy_j:
        log.metric("forward-model", "Efficiency predictions", data={
            "latency_s":    round(plan.predicted_latency_s, 2),
            "energy_j":     round(plan.predicted_energy_j, 4),
            "resolve_prob": round(plan.predicted_resolve_prob, 3),
            "eff_score":    round(plan.efficiency_score, 4),
        })
    for i, r in enumerate(plan.roles, 1):
        log.kv(f"role[{i}/{len(plan.roles)}]", {
            "id":        r.role_id,
            "name":      r.role_name,
            "tier":      r.model_tier.value,
            "model":     r.model_id,
            "tools":     ",".join(r.tools),
            "max_tokens":r.max_tokens,
            "temp":      r.temperature,
            "parallel":  r.parallel_group,
        })


def _log_trace(trace: Any, log: DAGALogger) -> None:
    log.section("Execution trace")
    for i, step in enumerate(trace.steps, 1):
        sym = "✓" if step.success else "✗"
        log.metric(f"step[{i}]", f"{sym} role={step.role_id}  model={step.model_id}", data={
            "in_tok":        step.input_tokens,
            "out_tok":       step.output_tokens,
            "latency_s":     round(step.latency_s, 3),
            "energy_j":      round(step.energy_j, 5),
            "n_tool_calls":  len(step.tool_calls),
        })
        for j, tc in enumerate(step.tool_calls, 1):
            args_preview = json.dumps(tc.get("args", {}))[:140]
            log.step(f"  tool[{j}]",
                     f"tool={tc.get('tool','?')}  args={args_preview}")
        if step.error:
            log.error(f"step[{i}]", f"error: {step.error}")
    log.kv("trace-totals", {
        "steps":    len(trace.steps),
        "tokens":   trace.total_tokens,
        "energy_j": round(trace.total_energy_j, 4),
        "latency_s":round(trace.total_latency_s, 2),
        "resolved": trace.resolved,
    })


# ══════════════════════════════════════════════════════════════
# Dataset loader
# ══════════════════════════════════════════════════════════════

def load_swebench_instances(
    dataset: str,
    split: str,
    max_instances: Optional[int],
    log: DAGALogger,
) -> List[Dict[str, Any]]:
    log.info("dataset", f"Loading {dataset} / split={split}")
    if os.path.exists(dataset):
        log.info("dataset", f"Reading local JSONL: {dataset}")
        with open(dataset) as f:
            instances = [json.loads(l) for l in f if l.strip()]
        log.success("dataset", f"Loaded {len(instances)} from local file")
    else:
        try:
            from datasets import load_dataset  # type: ignore
            log.info("dataset", "Downloading from HuggingFace Hub…")
            ds = load_dataset(dataset, split=split)
            instances = list(ds)
            log.success("dataset", f"Loaded {len(instances)} from Hub")
        except ImportError:
            raise ImportError(
                "pip install datasets   OR   pass a local JSONL file as --dataset"
            )
    if max_instances:
        instances = instances[:max_instances]
        log.info("dataset", f"Capped to {len(instances)} (--max_instances={max_instances})")
    return instances


# ══════════════════════════════════════════════════════════════
# Per-instance runner
# ══════════════════════════════════════════════════════════════

def run_instance(
    instance: Dict[str, Any],
    pipeline: DAGAPipeline,
    output_dir: str,
    log: DAGALogger,
    setup_sandbox: bool = True,
    sla_target: SLATarget = SLATarget.BALANCED,
    cache_dir: Optional[str] = None,
    clone_timeout: int = 600,
) -> Dict[str, Any]:
    instance_id = instance.get("instance_id", instance.get("id", "unknown"))
    repo_name   = instance.get("repo", "")
    problem     = instance.get("problem_statement", instance.get("issue_body", ""))
    hints       = instance.get("hints_text", "")
    task_desc   = f"{problem}\n\nHints:\n{hints}" if hints else problem

    log.banner(f"Instance: {instance_id}")
    log.kv("instance", {
        "repo":        repo_name,
        "id":          instance_id,
        "problem_len": len(problem),
        "has_hints":   bool(hints),
        "sla":         sla_target.value,
    })
    log.step("problem", f"Statement (first 400 chars):\n  {problem[:400].replace(chr(10),' ')}")

    workdir = pipeline.config.workdir

    # ── 1. Sandbox ───────────────────────────────────────────
    if setup_sandbox:
        ok = setup_repo(instance, workdir, log,
                        timeout=clone_timeout, cache_dir=cache_dir)
        if not ok:
            log.error("instance", "Repo setup failed — skipping")
            return {"instance_id": instance_id, "status": "setup_failed", "patch": None}
        pipeline.tool_registry = build_default_tool_registry(workdir)
        log.info("tools", f"Tool registry rebuilt  workdir={workdir}")
    else:
        log.warn("setup", "--skip_setup: using existing workdir as-is")

    # Count actual Python files in the checked-out repo for accurate profiling
    try:
        import subprocess as _sp
        fc_result = _sp.run(
            "find . -name '*.py' | wc -l",
            shell=True, cwd=workdir, capture_output=True, text=True, timeout=10,
        )
        real_file_count = int(fc_result.stdout.strip()) if fc_result.returncode == 0 else 0
    except Exception:
        real_file_count = 0
    log.kv("setup", {"python_files_in_repo": real_file_count})

    repo_meta = {
        "file_count":     real_file_count,   # real count, not the placeholder 0
        "has_tests":      bool(instance.get("test_directives", [])),
        "has_type_hints": True,
        "language":       "python",
    }

    # ── 2. Profile ───────────────────────────────────────────
    log.section("Profiling task")
    t0 = time.perf_counter()
    profile = pipeline.profiler.profile(
        task_description = task_desc,
        repo_metadata    = repo_meta,
        sla_target       = sla_target,
    )
    log.metric("profiler", f"Done in {(time.perf_counter()-t0)*1000:.0f}ms")
    _log_profile(profile, log)

    # ── 3. Routing ───────────────────────────────────────────
    log.section("Architecture routing")
    t0 = time.perf_counter()

    # Always show what the deterministic rule picked first
    det_plan = pipeline.det_router.route_to_plan(profile)
    log.decision(f"Deterministic rule: {det_plan.reasoning}")
    log.decision(f"  → topology={det_plan.topology.value}  "
                 f"roles={[r.role_name for r in det_plan.roles]}")

    # Will meta-LLM override?
    will_llm = pipeline.meta_router._should_invoke_llm(profile, det_plan)
    log.decision(f"Meta-LLM invoked: {will_llm}"
                 + (f"  (complexity={profile.complexity.value} / sla={profile.sla_target.value})"
                    if will_llm else ""))

    # Experience context
    exp = pipeline.feedback.experience_summary_for_meta_agent(profile)
    sim = exp.get("similar_experiences", [])
    if sim:
        log.info("experience", f"{len(sim)} similar past task(s)  "
                               f"best_topo={exp.get('best_topology_for_complexity','?')}  "
                               f"global_resolve={exp.get('global_resolve_rate','?')}")
        for s in sim[:3]:
            log.step("experience",
                     f"  past: topology={s.get('topology','?')}  "
                     f"resolved={s.get('resolved','?')}  "
                     f"energy={s.get('energy_j','?')}J  "
                     f"eff={s.get('efficiency','?')}")
    else:
        log.info("experience", "No prior experience for this task type yet")

    plan = pipeline.meta_router.route(profile, exp)
    plan = pipeline._resolve_model_ids(plan)
    log.metric("router", f"Routing done in {(time.perf_counter()-t0)*1000:.0f}ms")
    _log_plan(plan, log)

    # ── 4. Execute ───────────────────────────────────────────
    log.section("Execution")
    from daga.agents.topologies import create_orchestrator
    orch = create_orchestrator(plan, pipeline.registry, pipeline.tool_registry, verbose=True)
    log.info("exec", f"Orchestrator class: {type(orch).__name__}")

    t0    = time.perf_counter()
    trace = orch.execute(profile)
    exec_t = time.perf_counter() - t0
    log.metric("exec", f"Wall time: {exec_t:.2f}s")
    _log_trace(trace, log)

    # Patch preview
    if trace.final_patch:
        patch_lines = trace.final_patch.splitlines()
        preview     = "\n".join(patch_lines[:40])
        if len(patch_lines) > 40:
            preview += f"\n  ... ({len(patch_lines)-40} more lines)"
        log.info("patch", f"Preview:\n{preview}")
    else:
        log.warn("patch", "No patch produced by any agent")

    # ── 5. Telemetry ─────────────────────────────────────────
    log.section("Telemetry & experience")
    rec = pipeline.telemetry.collect(profile, plan, trace)
    pipeline.exp_store.save(rec)
    log.metric("telemetry", "Record saved", data={
        "resolved":  trace.resolved,
        "tokens":    trace.total_tokens,
        "energy_j":  round(trace.total_energy_j, 4),
        "latency_s": round(trace.total_latency_s, 2),
        "eff":       round(rec.efficiency_score, 4),
        "topology":  plan.topology.value,
        "source":    plan.routing_source,
    })

    # Write patch file — normalise and validate before writing
    patch_path = Path(output_dir)
    patch_path.mkdir(parents=True, exist_ok=True)
    pfile = patch_path / f"{instance_id}.patch"

    final_patch = trace.final_patch or ""
    if final_patch:
        from daga.agents.executor import _normalize_patch_paths, validate_patch
        import re as _re

        final_patch = _normalize_patch_paths(final_patch)

        # ── Path existence check ─────────────────────────────────
        # Extract every patched file path from the diff header and
        # verify each one actually exists in the checked-out workdir.
        # If a path is missing, try to locate the file by name and
        # rewrite the header with the correct full path.
        def _fix_missing_paths(patch: str, wd: str) -> str:
            lines = patch.splitlines(keepends=True)
            out: list = []
            for line in lines:
                if line.startswith("--- a/") or line.startswith("+++ b/"):
                    prefix = line[:6]          # "--- a/" or "+++ b/"
                    rel    = line[6:].rstrip() # e.g. "io/fits/fitsrec.py"
                    full   = Path(wd) / rel
                    if not full.exists():
                        # Try to find it under workdir
                        fname = Path(rel).name
                        found = list(Path(wd).rglob(fname))
                        if found:
                            correct_rel = str(found[0].relative_to(wd))
                            new_prefix = "--- a/" if prefix.startswith("---") else "+++ b/"
                            line = f"{new_prefix}{correct_rel}\n"
                            log.warn("patch", f"Path corrected: {rel} → {correct_rel}")
                        else:
                            log.warn("patch", f"Path not found in repo: {rel}")
                out.append(line)
            return "".join(out)

        final_patch = _fix_missing_paths(final_patch, workdir)
        final_patch = _normalize_patch_paths(final_patch)  # re-normalise after path fix

        valid, reason = validate_patch(final_patch)
        if not valid:
            log.warn("patch", f"Patch failed validation ({reason}) — writing empty patch")
            log.warn("patch", f"Invalid patch content:\n{final_patch[:400]}")
            final_patch = ""
            trace.resolved = False
        else:
            # Verify context lines match the actual checked-out file
            from daga.agents.executor import verify_patch_context
            ctx_ok, ctx_reason = verify_patch_context(final_patch, workdir)
            if not ctx_ok:
                log.warn("patch", f"Context verification failed — hunk will likely fail in harness")
                log.warn("patch", f"{ctx_reason}")
                # Don't discard — context mismatch might still apply with -F fuzz
                # but warn loudly so we know to investigate
            n_lines = len(final_patch.splitlines())
            log.success("output", f"Patch valid: {n_lines} lines → {pfile}")
            log.step("patch", f"Full patch content:\n{final_patch}")
    else:
        log.warn("output", f"No patch produced → {pfile}")

    # Ensure patch ends with a newline — `patch` tool requires it
    if final_patch and not final_patch.endswith("\n"):
        final_patch = final_patch + "\n"
        log.step("patch", "Added trailing newline to patch")
    pfile.write_text(final_patch)

    result = {
        "instance_id":    instance_id,
        "repo":           repo_name,
        "status":         "resolved" if trace.resolved else "unresolved",
        "topology":       plan.topology.value,
        "routing_source": plan.routing_source,
        "reasoning":      plan.reasoning,
        "latency_s":      round(exec_t, 2),
        "energy_j":       round(trace.total_energy_j, 4),
        "tokens":         trace.total_tokens,
        "efficiency":     round(rec.efficiency_score, 4),
        "n_steps":        len(trace.steps),
        "patch_lines":    len(final_patch.splitlines()) if final_patch else 0,
        # SWE-bench submission fields
        "model_patch":    final_patch,
    }

    if trace.resolved:
        log.success("result",
                    f"✓ RESOLVED  topology={result['topology']}  "
                    f"energy={result['energy_j']}J  latency={result['latency_s']}s  "
                    f"eff={result['efficiency']}")
    else:
        log.warn("result",
                 f"✗ UNRESOLVED  topology={result['topology']}  "
                 f"energy={result['energy_j']}J  latency={result['latency_s']}s")

    return result


# ══════════════════════════════════════════════════════════════
# Main evaluation loop
# ══════════════════════════════════════════════════════════════

def evaluate(
    dataset:          str  = "princeton-nlp/SWE-bench_Lite",
    split:            str  = "test",
    output_dir:       str  = "./daga_patches",
    experience_store: str  = "./daga_experience.jsonl",
    sla_target:       str  = "balanced",
    max_instances:    Optional[int] = None,
    use_mock:         bool = False,
    skip_setup:       bool = False,
    openrouter_key:   Optional[str] = None,
    cache_dir:        Optional[str] = None,
    clone_timeout:    int  = 600,
    log_file:         Optional[str] = None,
    no_color:         bool = False,
    model_name:       str  = "daga",
) -> None:
    log = DAGALogger(log_file=log_file, no_color=no_color)
    sla = SLATarget(sla_target)

    log.banner("DAGA  —  Dynamic Agentic Architecture Generation")
    log.kv("run-config", {
        "dataset":        dataset,
        "split":          split,
        "sla_target":     sla_target,
        "output_dir":     output_dir,
        "experience_store": experience_store,
        "max_instances":  max_instances or "all",
        "use_mock":       use_mock,
        "skip_setup":     skip_setup,
        "cache_dir":      cache_dir or "/tmp/daga_repo_cache",
        "clone_timeout":  clone_timeout,
        "log_file":       log_file or "stdout only",
        "openrouter":     "yes" if (openrouter_key or os.environ.get("OPENROUTER_API_KEY")) else "no",
    })

    # ── Registry ─────────────────────────────────────────────
    log.section("Building backend registry")
    registry = build_default_registry(
        use_mock           = use_mock,
        openrouter_api_key = openrouter_key,
    )
    from daga.backends.registry import MODEL_ENERGY_PROFILE
    seen: set = set()
    for mid, b in sorted(registry._backends.items()):
        if b.tier not in seen:
            ep = MODEL_ENERGY_PROFILE[b.tier]
            log.kv("backend", {
                "tier":      b.tier.value,
                "model_id":  b.model_id,
                "j/in":      ep["j_per_input_token"],
                "j/out":     ep["j_per_output_token"],
            })
            seen.add(b.tier)

    # ── Pipeline ─────────────────────────────────────────────
    config   = PipelineConfig(
        experience_store_path = experience_store,
        verbose               = True,
    )
    pipeline = DAGAPipeline(config=config, registry=registry)
    log.success("pipeline", "DAGAPipeline initialised")

    # ── Dataset ──────────────────────────────────────────────
    instances = load_swebench_instances(dataset, split, max_instances, log)
    log.banner(f"Starting {len(instances)} instance(s)")

    results: List[Dict[str, Any]] = []
    t_wall_start = time.perf_counter()

    for idx, inst in enumerate(instances, 1):
        inst_id = inst.get("instance_id", f"inst_{idx}")
        log.banner(f"[{idx} / {len(instances)}]  {inst_id}")
        t_inst = time.perf_counter()

        try:
            r = run_instance(
                inst, pipeline, output_dir, log,
                setup_sandbox = not skip_setup,
                sla_target    = sla,
                cache_dir     = cache_dir,
                clone_timeout = clone_timeout,
            )
            results.append(r)
        except KeyboardInterrupt:
            log.error("run", "KeyboardInterrupt — stopping early")
            break
        except Exception as exc:
            import traceback
            log.error("run", f"Unhandled exception: {exc}")
            log.error("run", traceback.format_exc())
            results.append({"instance_id": inst_id, "status": "error", "error": str(exc)})

        elapsed = time.perf_counter() - t_inst
        log.metric("timing", f"Instance wall time: {elapsed:.1f}s")

        # Rolling progress
        n_res = sum(1 for r in results if r.get("status") == "resolved")
        log.metric("progress",
                   f"Running totals: {n_res}/{len(results)} resolved  "
                   f"({n_res/len(results)*100:.0f}%)  "
                   f"energy={sum(r.get('energy_j',0) for r in results):.2f}J")

    # ══ Final summary ════════════════════════════════════════
    total_wall = time.perf_counter() - t_wall_start
    log.banner("Run complete")

    resolved   = [r for r in results if r.get("status") == "resolved"]
    errors     = [r for r in results if r.get("status") == "error"]
    t_energy   = sum(r.get("energy_j", 0) for r in results)
    t_tokens   = sum(r.get("tokens", 0)   for r in results)
    t_latency  = sum(r.get("latency_s", 0) for r in results)

    log.kv("summary", {
        "total":           len(results),
        "resolved":        len(resolved),
        "unresolved":      len(results) - len(resolved) - len(errors),
        "errors":          len(errors),
        "resolve_rate":    f"{len(resolved)/max(len(results),1)*100:.1f}%",
        "total_energy_j":  round(t_energy, 2),
        "total_tokens":    t_tokens,
        "total_latency_s": round(t_latency, 1),
        "wall_time_s":     round(total_wall, 1),
    })

    # Topology breakdown
    log.section("Topology breakdown")
    topo_c: Dict[str, int] = {}
    topo_r: Dict[str, int] = {}
    for r in results:
        t = r.get("topology", "?")
        topo_c[t] = topo_c.get(t, 0) + 1
        if r.get("status") == "resolved":
            topo_r[t] = topo_r.get(t, 0) + 1
    for t, cnt in sorted(topo_c.items(), key=lambda x: -x[1]):
        res = topo_r.get(t, 0)
        log.kv(t, {"count": cnt, "resolved": res,
                   "rate": f"{res/cnt*100:.0f}%"})

    # Per-instance table
    log.section("Per-instance results")
    for r in results:
        sym = "✓" if r.get("status") == "resolved" else ("⚠" if r.get("status") == "error" else "✗")
        log.info(sym, (
            f"{r.get('instance_id','?'):<45}"
            f"  {r.get('topology','?'):25}"
            f"  {r.get('routing_source','?'):14}"
            f"  {r.get('energy_j',0):.3f}J"
            f"  {r.get('latency_s',0):.1f}s"
            f"  eff={r.get('efficiency',0):.4f}"
            f"  {r.get('status','?')}"
        ))

    # Feedback loop analysis
    log.section("Efficiency feedback report")
    pipeline.report()

    # Save summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))

    # ── SWE-bench predictions.jsonl ───────────────────────────
    # Read patch content directly from the .patch files on disk to
    # avoid any JSON serialisation / round-trip issues with newlines.
    predictions_path = Path(output_dir) / "predictions.jsonl"
    with predictions_path.open("w") as pf:
        for r in results:
            iid        = r["instance_id"]
            patch_file = Path(output_dir) / f"{iid}.patch"
            # Read from disk — authoritative source, already normalised
            if patch_file.exists():
                patch_text = patch_file.read_text()
            else:
                patch_text = r.get("model_patch", "")
            # Guarantee trailing newline
            if patch_text and not patch_text.endswith("\n"):
                patch_text += "\n"
            pf.write(json.dumps({
                "instance_id":        iid,
                "model_name_or_path": model_name,
                "model_patch":        patch_text,
            }) + "\n")
    log.success("output", f"Patches:         {output_dir}/")
    log.success("output", f"Summary:         {summary_path}")
    log.success("output", f"SWE-bench JSONL: {predictions_path}  ← submit this to the harness")
    if log_file:
        log.success("output", f"Full log:       {log_file}  (JSONL, machine-readable)")


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="DAGA SWE-bench evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",          default="princeton-nlp/SWE-bench_Lite")
    p.add_argument("--split",            default="test")
    p.add_argument("--output_dir",       default="./daga_patches")
    p.add_argument("--experience_store", default="./daga_experience.jsonl")
    p.add_argument("--sla_target",       default="balanced",
                   choices=["balanced","latency_first","energy_first","quality_first"])
    p.add_argument("--max_instances",    type=int, default=None)
    p.add_argument("--use_mock",         action="store_true")
    p.add_argument("--skip_setup",       action="store_true")
    p.add_argument("--openrouter_key",   default=None,
                   help="OpenRouter key (or set OPENROUTER_API_KEY env var)")
    p.add_argument("--cache_dir",        default=None,
                   help="Repo clone cache dir (persists across runs)")
    p.add_argument("--clone_timeout",    type=int, default=600)
    p.add_argument("--log_file",         default=None,
                   help="Path for structured JSONL log file")
    p.add_argument("--no_color",         action="store_true")
    p.add_argument("--model_name",       default="daga",
                   help="Model label written into predictions.jsonl (shown in harness results)")

    args = p.parse_args()
    evaluate(
        dataset          = args.dataset,
        split            = args.split,
        output_dir       = args.output_dir,
        experience_store = args.experience_store,
        sla_target       = args.sla_target,
        max_instances    = args.max_instances,
        use_mock         = args.use_mock,
        skip_setup       = args.skip_setup,
        openrouter_key   = args.openrouter_key,
        cache_dir        = args.cache_dir,
        clone_timeout    = args.clone_timeout,
        log_file         = args.log_file,
        no_color         = args.no_color,
        model_name       = args.model_name,
    )


if __name__ == "__main__":
    main()