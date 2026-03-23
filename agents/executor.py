"""
DAGA — Agent Executor
Implements a structured ReAct (Reason + Act) loop for each agent role.
Supports tool-call parsing from model output and step-level telemetry.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Tuple

from daga.backends.registry import BackendRegistry
from daga.core.models import AgentRole, StepTrace
from daga.tools.registry import ToolRegistry
from daga.telemetry.logging import get_logger, log_kv


logger = get_logger("daga.executor")


# ──────────────────────────────────────────────
# System prompt templates
# ──────────────────────────────────────────────

SYSTEM_PROMPTS: Dict[str, str] = {

    "SYSTEM_PROMPT_SOLVER": """\
You are an expert software engineer. Your task is to resolve the given issue by:
1. Understanding the problem
2. Locating the relevant code
3. Implementing a correct, minimal fix
4. Verifying it passes the tests

Use the provided tools to explore the codebase, make changes, and run tests.
When you are confident the fix is complete, output:
<FINAL_PATCH>
<unified diff here>
</FINAL_PATCH>
""",

    "SYSTEM_PROMPT_LOCALISER": """\
You are a code localisation specialist. Given an issue description, your job is to:
1. Identify the EXACT files and line numbers that need to be changed
2. Extract the relevant code context
3. Output a structured JSON localisation report:
{
  "files": [{"path": "<file>", "lines": [<start>, <end>], "reason": "<why>"}],
  "root_cause": "<brief explanation>",
  "affected_symbols": ["<function_or_class_name>"]
}
Use ast_search, ripgrep, and file_reader tools.
""",

    "SYSTEM_PROMPT_PATCHER": """\
You are a code patching specialist. You receive a localisation report and must produce
a correct minimal patch.
- Make the SMALLEST change that fixes the issue
- Preserve existing code style
- Do NOT add unnecessary comments or refactoring
When done, output:
<FINAL_PATCH>
<unified diff>
</FINAL_PATCH>
""",

    "SYSTEM_PROMPT_VERIFIER": """\
You are a test verification specialist. Given a patch, verify it is correct by:
1. Applying the patch
2. Running the relevant tests
3. Checking for regressions
Output: {"pass": true/false, "reason": "<details>"}
""",

    "SYSTEM_PROMPT_PLANNER": """\
You are a high-level planning agent. Decompose the given issue into sub-tasks.
Output a JSON execution plan:
{
  "sub_tasks": [
    {"id": "1", "description": "<task>", "files": ["<file>"], "depends_on": []}
  ],
  "strategy": "<brief approach>"
}
""",

    "SYSTEM_PROMPT_EXECUTOR": """\
You are an execution agent. Implement the sub-task assigned to you precisely.
Focus only on the described scope. When complete, output:
<TASK_RESULT>
<description of what was done>
</TASK_RESULT>
<PATCH>
<unified diff>
</PATCH>
""",

    "SYSTEM_PROMPT_WORKER": """\
You are a parallel worker agent solving a software engineering task independently.
Explore the codebase, implement a fix, and output:
<FINAL_PATCH>
<unified diff>
</FINAL_PATCH>
""",
}


def resolve_system_prompt(template: str) -> str:
    """Replace {TEMPLATE_KEY} placeholders with actual prompts."""
    for key, value in SYSTEM_PROMPTS.items():
        template = template.replace(f"{{{key}}}", value)
    return template


# ──────────────────────────────────────────────
# Tool call parsing from model output
# ──────────────────────────────────────────────

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*\})\s*</tool_call>",
    re.DOTALL,
)

# Fallback: some models emit tool calls as plain JSON (not wrapped in <tool_call>).
# We only accept JSON objects that look like {"tool": <str>, "args": <dict>},
# and we only extract objects (not arrays) to reduce false positives.
PLAIN_TOOL_CALL_PATTERN = re.compile(
    r"(?m)^(\{\s*\"tool\"\s*:\s*\"[^\"]+\"\s*,\s*\"args\"\s*:\s*\{.*?\}\s*\})\s*$",
    re.DOTALL,
)

FINAL_PATCH_PATTERN = re.compile(
    r"<FINAL_PATCH>(.*?)</FINAL_PATCH>",
    re.DOTALL,
)

TASK_RESULT_PATTERN = re.compile(
    r"<TASK_RESULT>(.*?)</TASK_RESULT>",
    re.DOTALL,
)

PATCH_PATTERN = re.compile(
    r"<PATCH>(.*?)</PATCH>",
    re.DOTALL,
)


def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []

    def _sanitize_tool_json(s: str) -> str:
        """Best-effort cleanup for common model glitches while staying conservative."""
        # Some models mistakenly append a trailing ';' inside or right after the JSON object.
        # Only strip semicolons that appear immediately before a closing brace.
        return re.sub(r";\s*(\})", r"\1", s)

    def _maybe_add(obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        tool = obj.get("tool")
        args = obj.get("args")
        if isinstance(tool, str) and isinstance(args, dict):
            calls.append({"tool": tool, "args": args})

    # Primary: explicit XML-wrapped tool calls
    for m in TOOL_CALL_PATTERN.finditer(text):
        try:
            raw = m.group(1)
            try:
                _maybe_add(json.loads(raw))
            except json.JSONDecodeError:
                fixed = _sanitize_tool_json(raw)
                _maybe_add(json.loads(fixed))
        except json.JSONDecodeError:
            log_kv(
                logger,
                30,
                "tool_call.json_decode_error",
                stage="agent",
                payload=(m.group(1)[:500] + ("..." if len(m.group(1)) > 500 else "")),
            )
            continue

    if calls:
        return calls

    # Fallback: plain JSON on its own line (common when model ignores the XML format)
    for m in PLAIN_TOOL_CALL_PATTERN.finditer(text):
        try:
            raw = m.group(1)
            try:
                _maybe_add(json.loads(raw))
            except json.JSONDecodeError:
                fixed = _sanitize_tool_json(raw)
                _maybe_add(json.loads(fixed))
        except json.JSONDecodeError:
            log_kv(
                logger,
                30,
                "tool_call.plain_json_decode_error",
                stage="agent",
                payload=(m.group(1)[:500] + ("..." if len(m.group(1)) > 500 else "")),
            )
            continue

    return calls


def _normalize_patch_paths(patch: str) -> str:
    """
    Ensure every --- / +++ line has the full repo-relative path with a/ b/ prefix.

    Common model mistakes:
      --- a/io/fits/fitsrec.py          (missing package root, e.g. astropy/)
      --- fitsrec.py                    (bare filename, no path)
      --- /tmp/daga_sandbox/astropy/... (absolute sandbox path)

    The SWE-bench harness applies with `patch -p1` from the repo root, so it needs:
      --- a/astropy/io/fits/fitsrec.py
      +++ b/astropy/io/fits/fitsrec.py
    """
    lines = patch.splitlines(keepends=True)
    fixed: List[str] = []

    for line in lines:
        if line.startswith("--- ") or line.startswith("+++ "):
            prefix = line[:4]
            rest   = line[4:].rstrip()

            # Strip absolute sandbox paths like /tmp/daga_sandbox/
            rest = re.sub(r"^/tmp/[^/]*/", "", rest)
            # Strip existing a/ or b/ prefix so we can add it cleanly
            for ab in ("a/", "b/"):
                if rest.startswith(ab):
                    rest = rest[2:]
                    break
            # Strip leading ./
            if rest.startswith("./"):
                rest = rest[2:]

            ab_prefix = "a/" if prefix.startswith("---") else "b/"
            line = f"{prefix}{ab_prefix}{rest}\n"

        fixed.append(line)

    return "".join(fixed)


def validate_patch(patch: str) -> Tuple[bool, str]:
    """
    Return (is_valid, reason).
    Checks for correct unified diff syntax AND detects truncation.
    """
    if not patch or not patch.strip():
        return False, "empty"

    # Basic structure checks
    has_minus_hdr = bool(re.search(r"^--- a/", patch, re.MULTILINE))
    has_plus_hdr  = bool(re.search(r"^\+\+\+ b/", patch, re.MULTILINE))
    has_hunk      = bool(re.search(r"^@@", patch, re.MULTILINE))
    has_changes   = bool(re.search(r"^[+\-](?!--|\+\+)", patch, re.MULTILINE))

    # Reject patches against /dev/null — model created a new file instead of editing
    if re.search(r"^--- a//dev/null|^--- /dev/null", patch, re.MULTILINE):
        return False, "patches /dev/null — model invented a new file instead of editing source"

    if not has_minus_hdr:
        return False, "missing --- a/ header"
    if not has_plus_hdr:
        return False, "missing +++ b/ header"
    if not has_hunk:
        return False, "missing @@ hunk"
    if not has_changes:
        return False, "no changed lines"

    # Truncation check: every hunk header @@ -A,B +C,D @@ declares how many lines follow.
    # Count actual context+change lines in the hunk body and compare.
    truncated = False
    for m in re.finditer(r"^@@ -\d+,(\d+) \+\d+,(\d+) @@", patch, re.MULTILINE):
        declared_old = int(m.group(1))
        declared_new = int(m.group(2))
        hunk_start   = m.end()
        # Find next hunk or end of patch
        next_m = re.search(r"^@@|^diff --git", patch[hunk_start:], re.MULTILINE)
        hunk_body = patch[hunk_start: hunk_start + next_m.start()] if next_m else patch[hunk_start:]
        hunk_lines = [l for l in hunk_body.splitlines() if l and not l.startswith("\\")]
        actual_old = sum(1 for l in hunk_lines if l.startswith(" ") or l.startswith("-"))
        actual_new = sum(1 for l in hunk_lines if l.startswith(" ") or l.startswith("+"))
        # Allow up to 2 missing context lines — legitimate when hunk is at
        # end of file (the last context lines may be omitted by the model).
        # Only flag as truncated if significantly fewer lines than declared.
        if actual_old < declared_old - 2 or actual_new < declared_new - 2:
            truncated = True
            break

    if truncated:
        return False, "patch is truncated — hunk line counts don't match (hit max_tokens mid-output)"

    # Final line truncation: if patch ends mid-word (no newline AND last char is alphanumeric)
    # this indicates the model was cut off literally in the middle of a line
    if patch and not patch.endswith("\n") and patch[-1].isalnum():
        return False, "patch ends mid-word — likely truncated by max_tokens"

    return True, "ok"


def verify_patch_context(patch: str, workdir: str) -> Tuple[bool, str]:
    """
    Verify that context lines in the patch actually match the file content
    at the checked-out commit. Returns (ok, first_mismatch_description).

    This catches "Hunk FAILED" before the harness sees the patch.
    Only checks --- a/ files that exist in workdir.
    """
    import os
    current_file: Optional[str] = None
    current_offset: int = 0        # line number of first context line in hunk
    hunk_lines: List[str] = []
    hunk_old_start: int = 0

    issues: List[str] = []

    for raw_line in patch.splitlines():
        if raw_line.startswith("--- a/"):
            current_file = raw_line[6:].strip()
            hunk_lines = []
        elif raw_line.startswith("+++ b/"):
            pass
        elif raw_line.startswith("@@ "):
            # @@ -OLD_START,OLD_COUNT +NEW_START,NEW_COUNT @@
            m = re.match(r"^@@ -(\d+)", raw_line)
            if m:
                hunk_old_start = int(m.group(1))
                hunk_lines = []
        elif raw_line.startswith(" ") or raw_line.startswith("-"):
            hunk_lines.append((raw_line[0], raw_line[1:]))

    # Re-parse more carefully: for each hunk in each file, check context lines
    if not current_file or not workdir:
        return True, "ok"

    file_path = os.path.join(workdir, current_file)
    if not os.path.exists(file_path):
        return True, "ok"   # path check handled elsewhere

    try:
        file_lines = open(file_path, errors="replace").readlines()
    except Exception:
        return True, "ok"

    # Walk hunks
    current_file = None
    for raw_line in patch.splitlines():
        if raw_line.startswith("--- a/"):
            current_file = raw_line[6:].strip()
        elif raw_line.startswith("@@ "):
            m = re.match(r"^@@ -(\d+)", raw_line)
            if m:
                hunk_old_start = int(m.group(1))
                file_idx = hunk_old_start - 1   # 0-indexed
                hunk_lines = []
        elif current_file and (raw_line.startswith(" ") or raw_line.startswith("-")):
            patch_ctx = raw_line[1:]   # strip leading space or -
            if file_idx < len(file_lines):
                actual = file_lines[file_idx].rstrip(chr(10) + chr(13))
                if actual != patch_ctx.rstrip(chr(10) + chr(13)):
                    issues.append(
                        f"{current_file}:{file_idx+1}: "
                        f"context mismatch\n"
                        f"  patch : {repr(patch_ctx[:80])}\n"
                        f"  actual: {repr(actual[:80])}"
                    )
                    if len(issues) >= 3:
                        break
            file_idx += 1

    if issues:
        return False, "Context line mismatch (hunk will fail):\n" + "\n".join(issues)
    return True, "ok"


def extract_final_patch(text: str) -> Optional[str]:
    """
    Extract, normalise, and validate a unified diff from model output.
    Returns None if no valid patch found after all attempts.
    """
    candidates: List[str] = []

    # Primary: explicit tag
    tag_m = FINAL_PATCH_PATTERN.search(text)
    if tag_m:
        # Strip leading whitespace/newline but preserve trailing newline
        candidates.append(tag_m.group(1).lstrip("\n").rstrip(" \t"))

    # Fallback: scan raw diff blocks (model sometimes omits the tag)
    for diff_m in re.finditer(
        r"((?:diff --git [^\n]+\n)?--- [^\n]+\n\+\+\+ [^\n]+\n(?:@@[^\n]+\n(?:[^-+@\n][^\n]*\n|[+-][^\n]*\n)*)+)",
        text, re.MULTILINE,
    ):
        candidates.append(diff_m.group(0).strip())

    for raw in candidates:
        normalised = _normalize_patch_paths(raw)
        ok, _ = validate_patch(normalised)
        if ok:
            return normalised
        ok2, _ = validate_patch(raw)
        if ok2:
            return raw

    return None



# ──────────────────────────────────────────────
# Single-agent ReAct executor
# ──────────────────────────────────────────────

@dataclass
class AgentExecutorResult:
    role_id: str
    steps: List[StepTrace] = field(default_factory=list)
    final_output: str = ""
    final_patch: Optional[str] = None
    success: bool = False
    total_latency_s: float = 0.0
    total_energy_j: float = 0.0
    total_tokens: int = 0


class AgentExecutor:
    """
    Drives a single agent role through a ReAct loop until:
    - The agent outputs a FINAL_PATCH or TASK_RESULT
    - Max iterations are reached
    - An error occurs
    """

    def __init__(
        self,
        role: AgentRole,
        registry: BackendRegistry,
        tool_registry: ToolRegistry,
        max_iterations: int = 20,
        verbose: bool = False,
    ) -> None:
        self._role         = role
        self._backend      = registry.get(role.model_id)
        self._tools        = tool_registry
        self._max_iter     = max_iterations
        self._verbose      = verbose

        # Build tool schemas for this role
        self._tool_schemas = tool_registry.get_schemas(role.tools)

    def _build_system(self, task_context: str) -> str:
        base = resolve_system_prompt(self._role.system_prompt_template)
        tool_desc = "\n".join(
            f"- {s['name']}: {s['description']}" for s in self._tool_schemas
        )
        return (
            f"{base}\n\nAVAILABLE TOOLS:\n{tool_desc}\n\n"
            f"CONTEXT:\n{task_context}\n\n"
            "To call a tool, output:\n"
            "<tool_call>{\"tool\": \"<name>\", \"args\": {...}}</tool_call>"
        )

    def _execute_tool_calls(
        self, calls: List[Dict[str, Any]]
    ) -> str:
        def _alias_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            """Map common model argument-key variants onto the canonical schema keys."""
            if not isinstance(args, dict):
                return {}

            # Global/common aliases (rare but cheap to support)
            if "cmd" in args and "command" not in args:
                args = {**args, "command": args.get("cmd")}
                args.pop("cmd", None)

            # Tool-specific aliases
            if tool_name == "file_reader":
                for k in ("file_path", "filepath", "filename", "file"):
                    if k in args and "path" not in args:
                        args = {**args, "path": args.get(k)}
                        args.pop(k, None)
                        break
            elif tool_name == "file_editor":
                for k in ("file_path", "filepath", "filename", "file"):
                    if k in args and "path" not in args:
                        args = {**args, "path": args.get(k)}
                        args.pop(k, None)
                        break
            elif tool_name == "ripgrep":
                # Some models use query instead of pattern
                if "query" in args and "pattern" not in args:
                    args = {**args, "pattern": args.get("query")}
                    args.pop("query", None)
            return args

        results = []
        for call in calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            try:
                tool = self._tools.get(tool_name)

                # Apply aliases before validating schema.
                args = _alias_args(tool_name, args)

                # Validate args against tool schema (helps models correct themselves).
                schema = getattr(tool, "schema", {}) or {}
                props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
                required = schema.get("required") if isinstance(schema, dict) else None
                if not isinstance(args, dict):
                    args = {}

                allowed_keys = set(props.keys()) if isinstance(props, dict) else set()
                required_keys = set(required) if isinstance(required, list) else set()
                provided_keys = set(args.keys())

                missing = sorted(list(required_keys - provided_keys))
                unknown = sorted(list(provided_keys - allowed_keys)) if allowed_keys else []

                if missing or unknown:
                    # Return an explicit tool_result so the model can immediately correct.
                    log_kv(
                        logger,
                        30,
                        "tool_call.args_mismatch",
                        stage="tool",
                        tool=tool_name,
                        missing=missing or None,
                        unknown=unknown or None,
                    )

                    schema_hint_lines: List[str] = []
                    if required_keys:
                        schema_hint_lines.append(f"Required keys: {sorted(required_keys)}")
                    if allowed_keys:
                        schema_hint_lines.append(f"Allowed keys: {sorted(allowed_keys)}")
                    if missing:
                        schema_hint_lines.append(f"Missing required: {missing}")
                    if unknown:
                        schema_hint_lines.append(f"Unknown keys: {unknown}")
                    schema_hint_lines.append("Expected args schema (JSON-schema style):")
                    schema_hint_lines.append(json.dumps(schema, ensure_ascii=False))

                    results.append(
                        f"<tool_result tool=\"{tool_name}\" success=\"false\">\n"
                        f"Invalid tool arguments. Fix and retry.\n"
                        + "\n".join(schema_hint_lines)
                        + "\n</tool_result>"
                    )
                    continue

                result = tool(args)
                
                results.append(
                    f"<tool_result tool=\"{tool_name}\" success=\"{result.success}\">"
                    f"\n{result.output}"
                    + (f"\nERROR: {result.error}" if result.error else "")
                    + f"\n</tool_result>"
                )
            except KeyError:
                results.append(f"<tool_result tool=\"{tool_name}\" success=\"false\">"
                               f"\nTool '{tool_name}' not available.</tool_result>")
        return "\n".join(results)

    def run(self, task_description: str, extra_context: str = "") -> AgentExecutorResult:
        system  = self._build_system(extra_context or task_description)
        history: List[Dict[str, str]] = [
            {"role": "system",  "content": system},
            {"role": "user",    "content": task_description},
        ]

        result = AgentExecutorResult(role_id=self._role.role_id)
        t_start = time.perf_counter()

        for iteration in range(self._max_iter):
            step = StepTrace(role_id=self._role.role_id, model_id=self._role.model_id)

            try:
                resp = self._backend.complete(
                    history,
                    max_tokens   = self._role.max_tokens,
                    temperature  = self._role.temperature,
                )
            except Exception as exc:
                step.success = False
                step.error   = str(exc)
                result.steps.append(step)
                break

            step.input_tokens  = resp.input_tokens
            step.output_tokens = resp.output_tokens
            step.latency_s     = resp.latency_s
            step.energy_j      = resp.energy_j

            if self._verbose:
                print(f"[{self._role.role_name}] iter={iteration} "
                      f"tokens={resp.output_tokens} energy={resp.energy_j:.4f}J")

            assistant_msg = resp.text
            history.append({"role": "assistant", "content": assistant_msg})

            # Check termination conditions
            patch = extract_final_patch(assistant_msg)
            if patch:
                result.final_patch  = patch
                result.final_output = assistant_msg
                result.success      = True
                result.steps.append(step)
                break

            # Execute tool calls if present
            tool_calls = extract_tool_calls(assistant_msg)
            if tool_calls:
                step.tool_calls = tool_calls
                tool_output = self._execute_tool_calls(tool_calls)
                print(f"Tool calls found: {len(tool_calls)}. Output:\n{tool_output}\n")
                history.append({"role": "user", "content": tool_output})
                print(f"Executed {len(tool_calls)} tool call(s), appended output to history for next iteration.")
            else:
                # No tool calls, no patch → check for task result
                task_result = TASK_RESULT_PATTERN.search(assistant_msg)
                if task_result:
                    result.final_output = task_result.group(1).strip()
                    patch_m = PATCH_PATTERN.search(assistant_msg)
                    if patch_m:
                        result.final_patch = patch_m.group(1).strip()
                    result.success = True
                    result.steps.append(step)
                    break
                # Otherwise prompt for continuation
                history.append({
                    "role": "user",
                    "content": (
                        "Continue. IMPORTANT: do not assume file contents or test outcomes. "
                        "If you need to read/search/edit/run tests, you MUST call a tool using exactly: "
                        "<tool_call>{\"tool\": \"<name>\", \"args\": {...}}</tool_call>. "
                        "If the fix is complete, output <FINAL_PATCH>...</FINAL_PATCH>."
                    ),
                })

            result.steps.append(step)

        result.total_latency_s = time.perf_counter() - t_start
        result.total_energy_j  = sum(s.energy_j for s in result.steps)
        result.total_tokens    = sum(s.input_tokens + s.output_tokens for s in result.steps)
        return result