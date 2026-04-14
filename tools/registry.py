"""
DAGA — Tool Pool
Concrete implementations of every tool agents can invoke.
Each tool is a callable that takes a dict of args and returns a ToolResult.

Tools are intentionally minimal and composable; agents combine them via
the action loop rather than relying on monolithic tools.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from daga.telemetry.logging import get_logger, log_kv


logger = get_logger("daga.tools")


# ──────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────

@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


# ──────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────

class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        """JSON-schema style description for function-calling."""
        ...

    @abstractmethod
    def __call__(self, args: Dict[str, Any]) -> ToolResult: ...


# ──────────────────────────────────────────────
# Bash executor
# ──────────────────────────────────────────────

class BashTool(Tool):
    """Run arbitrary shell commands in a sandboxed working directory."""

    MAX_OUTPUT = 8000

    def __init__(self, workdir: str = "/tmp/daga_sandbox", timeout: int = 60) -> None:
        self._workdir = workdir
        self._timeout = timeout
        os.makedirs(workdir, exist_ok=True)

    @property
    def name(self) -> str: return "bash"

    @property
    def description(self) -> str:
        return "Execute a shell command in the sandbox working directory."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
            },
            "required": ["command"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        cmd = args.get("command", "")
        log_kv(logger, 20, "tool.bash.start", stage="tool", tool=self.name, cmd=cmd)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self._workdir,
                capture_output=True, text=True, timeout=self._timeout,
            )
            out = (proc.stdout + proc.stderr)[:self.MAX_OUTPUT]
            log_kv(
                logger,
                20 if proc.returncode == 0 else 30,
                "tool.bash.finish",
                stage="tool",
                tool=self.name,
                cmd=cmd,
                return_code=proc.returncode,
                latency_s=time.perf_counter() - t0,
            )
            return ToolResult(
                tool_name = self.name,
                success   = proc.returncode == 0,
                output    = out,
                error     = None if proc.returncode == 0 else f"Exit {proc.returncode}",
                latency_s = time.perf_counter() - t0,
            )
        except subprocess.TimeoutExpired:
            log_kv(logger, 40, "tool.bash.timeout", stage="tool", tool=self.name, cmd=cmd, timeout_s=self._timeout)
            return ToolResult(self.name, False, "", "Timeout", latency_s=self._timeout)
        except Exception as e:
            log_kv(logger, 40, "tool.bash.error", stage="tool", tool=self.name, cmd=cmd, error=str(e))
            return ToolResult(self.name, False, "", str(e), latency_s=time.perf_counter()-t0)


# ──────────────────────────────────────────────
# File reader
# ──────────────────────────────────────────────

class FileReaderTool(Tool):
    MAX_CHARS = 40_000

    def __init__(self, workdir: str = "/tmp/daga_sandbox") -> None:
        self._workdir = workdir

    @property
    def name(self) -> str: return "file_reader"

    @property
    def description(self) -> str:
        return "Read the contents of a file relative to the repo root."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-indexed, optional."},
                "end_line":   {"type": "integer", "description": "1-indexed, optional."},
            },
            "required": ["path"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        path = Path(self._workdir) / args["path"]
        print(f"FileReaderTool: Reading file {path} with args {args}")
        try:
            log_kv(logger, 10, "tool.file_reader", stage="tool", tool=self.name, path=str(path))
            lines = path.read_text(errors="replace").splitlines(keepends=True)
            start = max(int(args.get("start_line", 1)) - 1, 0)
            end   = min(int(args.get("end_line",   len(lines))), len(lines))
            content = "".join(lines[start:end])[:self.MAX_CHARS]
            print(f"FileReaderTool: Read {len(content)} chars from {path} (lines {start+1}-{end})")
            return ToolResult(self.name, True, content)
        except Exception as e:
            return ToolResult(self.name, False, "", str(e))


# ──────────────────────────────────────────────
# File editor (write / patch hunk)
# ──────────────────────────────────────────────

class FileEditorTool(Tool):
    """
    Supports three operations:
      - write   : overwrite (or create) a file with new_content
      - replace : replace old_str with new_str (exact substring match)
      - insert  : insert content after a given line number
    """

    def __init__(self, workdir: str = "/tmp/daga_sandbox") -> None:
        self._workdir = workdir

    @property
    def name(self) -> str: return "file_editor"

    @property
    def description(self) -> str:
        return "Create or edit a file. Supports write, replace, and insert operations."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path":        {"type": "string"},
                "operation":   {"type": "string", "enum": ["write", "replace", "insert"]},
                "new_content": {"type": "string"},
                "old_str":     {"type": "string"},
                "new_str":     {"type": "string"},
                "after_line":  {"type": "integer"},
            },
            "required": ["path", "operation"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        path = Path(self._workdir) / args["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        op = args.get("operation", "write")
        try:
            if op == "write":
                path.write_text(args.get("new_content", ""))
                log_kv(logger, 20, "tool.file_editor.write", stage="tool", tool=self.name, path=str(path))
                return ToolResult(self.name, True, f"Wrote {path}")

            elif op == "replace":
                old  = args.get("old_str", "")
                new  = args.get("new_str", "")
                text = path.read_text(errors="replace")
                if old not in text:
                    log_kv(logger, 30, "tool.file_editor.replace.miss", stage="tool", tool=self.name, path=str(path))
                    return ToolResult(self.name, False, "", f"old_str not found in {path}")
                path.write_text(text.replace(old, new, 1))
                log_kv(logger, 20, "tool.file_editor.replace", stage="tool", tool=self.name, path=str(path))
                return ToolResult(self.name, True, f"Replaced in {path}")

            elif op == "insert":
                lines = path.read_text(errors="replace").splitlines(keepends=True)
                after = int(args.get("after_line", len(lines)))
                insert_text = args.get("new_content", "") + "\n"
                lines.insert(after, insert_text)
                path.write_text("".join(lines))
                log_kv(logger, 20, "tool.file_editor.insert", stage="tool", tool=self.name, path=str(path), after_line=after)
                return ToolResult(self.name, True, f"Inserted after line {after} in {path}")

            else:
                return ToolResult(self.name, False, "", f"Unknown op: {op}")
        except Exception as e:
            return ToolResult(self.name, False, "", str(e))


# ──────────────────────────────────────────────
# AST search (Python-specific, safe)
# ──────────────────────────────────────────────

class ASTSearchTool(Tool):
    """
    Parse Python files and find definitions, references, or imports.
    Falls back to regex grep if parsing fails.
    """

    def __init__(self, workdir: str = "/tmp/daga_sandbox") -> None:
        self._workdir = workdir

    @property
    def name(self) -> str: return "ast_search"

    @property
    def description(self) -> str:
        return "Search Python code using AST or regex. Find functions, classes, symbols."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query":    {"type": "string", "description": "Symbol or pattern to find."},
                "path":     {"type": "string", "description": "File or directory to search."},
                "mode":     {"type": "string", "enum": ["symbol", "regex", "imports"]},
            },
            "required": ["query"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        query  = args["query"]
        root   = Path(self._workdir) / args.get("path", "")
        mode   = args.get("mode", "symbol")
        results: List[str] = []

        py_files = list(root.rglob("*.py")) if root.is_dir() else [root]

        for fp in py_files[:200]:   # cap at 200 files
            try:
                src = fp.read_text(errors="replace")
                if mode == "regex":
                    for i, line in enumerate(src.splitlines(), 1):
                        if re.search(query, line):
                            results.append(f"{fp.relative_to(self._workdir)}:{i}: {line.rstrip()}")
                elif mode == "symbol":
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if query.lower() in node.name.lower():
                                results.append(
                                    f"{fp.relative_to(self._workdir)}:{node.lineno}: "
                                    f"{'def' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'} "
                                    f"{node.name}"
                                )
                elif mode == "imports":
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            line = ast.unparse(node)
                            if query.lower() in line.lower():
                                results.append(f"{fp.relative_to(self._workdir)}:{node.lineno}: {line}")
            except Exception:
                continue

        if not results:
            return ToolResult(self.name, True, f"No results for '{query}'")
        return ToolResult(self.name, True, "\n".join(results[:200]))


# ──────────────────────────────────────────────
# Ripgrep wrapper (fast text search)
# ──────────────────────────────────────────────

class RipgrepTool(Tool):
    """Fast regex search across the entire codebase using ripgrep (rg)."""

    def __init__(self, workdir: str = "/tmp/daga_sandbox", timeout: int = 30) -> None:
        self._workdir = workdir
        self._timeout = timeout

    @property
    def name(self) -> str: return "ripgrep"

    @property
    def description(self) -> str:
        return "Fast grep across files. Returns matching lines with file:line context."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path":    {"type": "string", "description": "File or directory (default: repo root)."},
                "flags":   {"type": "string", "description": "e.g. '-i' for case-insensitive."},
            },
            "required": ["pattern"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        pattern = args["pattern"]
        path    = args.get("path", ".")
        flags   = args.get("flags", "")
        cmd = f"rg {flags} --line-number --max-count 50 {repr(pattern)} {path}"
        t0  = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self._workdir,
                capture_output=True, text=True, timeout=self._timeout,
            )
            out = (proc.stdout + proc.stderr)[:10_000]
            return ToolResult(self.name, True, out, latency_s=time.perf_counter()-t0)
        except Exception as e:
            return ToolResult(self.name, False, "", str(e), latency_s=time.perf_counter()-t0)


# ──────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────

class TestRunnerTool(Tool):
    """Run the test suite for the repository."""

    def __init__(self, workdir: str = "/tmp/daga_sandbox", timeout: int = 120) -> None:
        self._workdir = workdir
        self._timeout = timeout

    @property
    def name(self) -> str: return "test_runner"

    @property
    def description(self) -> str:
        return "Run pytest (or the project's test command) and return pass/fail counts."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "test_path": {"type": "string", "description": "Specific test file or folder."},
                "extra_args": {"type": "string"},
            },
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        test_path  = args.get("test_path", "")
        extra_args = args.get("extra_args", "")
        cmd = f"python -m pytest {test_path} {extra_args} -x -q --tb=short 2>&1"
        t0  = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=self._workdir,
                capture_output=True, text=True, timeout=self._timeout,
            )
            out = (proc.stdout + proc.stderr)[:12_000]
            passed = "passed" in out.lower()
            return ToolResult(
                self.name, proc.returncode == 0, out,
                error=None if proc.returncode == 0 else "Tests failed",
                metadata={"return_code": proc.returncode},
                latency_s=time.perf_counter()-t0,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(self.name, False, "", "Test timeout", latency_s=self._timeout)
        except Exception as e:
            return ToolResult(self.name, False, "", str(e), latency_s=time.perf_counter()-t0)


# ──────────────────────────────────────────────
# Patch apply tool
# ──────────────────────────────────────────────

class PatchApplyTool(Tool):
    """Apply a unified diff patch string to the working directory."""

    def __init__(self, workdir: str = "/tmp/daga_sandbox") -> None:
        self._workdir = workdir

    @property
    def name(self) -> str: return "patch_apply"

    @property
    def description(self) -> str:
        return "Apply a unified diff patch to the repository."

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Unified diff text."},
            },
            "required": ["patch"],
        }

    def __call__(self, args: Dict[str, Any]) -> ToolResult:
        patch = args.get("patch", "")
        patch_file = Path(self._workdir) / "__daga_patch__.diff"
        patch_file.write_text(patch)
        t0 = time.perf_counter()
        try:
            log_kv(logger, 20, "tool.patch_apply.start", stage="tool", tool=self.name, path=str(patch_file), patch_chars=len(patch))
            proc = subprocess.run(
                f"patch -p1 < {patch_file}",
                shell=True, cwd=self._workdir,
                capture_output=True, text=True, timeout=30,
            )
            out = proc.stdout + proc.stderr
            log_kv(
                logger,
                20 if proc.returncode == 0 else 30,
                "tool.patch_apply.finish",
                stage="tool",
                tool=self.name,
                return_code=proc.returncode,
                latency_s=time.perf_counter() - t0,
            )
            return ToolResult(
                self.name, proc.returncode == 0, out,
                error=None if proc.returncode == 0 else f"Patch failed: {out[:500]}",
                latency_s=time.perf_counter()-t0,
            )
        except Exception as e:
            log_kv(logger, 40, "tool.patch_apply.error", stage="tool", tool=self.name, error=str(e))
            return ToolResult(self.name, False, "", str(e), latency_s=time.perf_counter()-t0)


# ──────────────────────────────────────────────
# Tool registry
# ──────────────────────────────────────────────

class ToolRegistry:
    """Maps tool name → Tool instance."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not registered. Available: {list(self._tools)}")
        return self._tools[name]

    def list_all(self) -> List[str]:
        return list(self._tools)

    def catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": tool.description,
                "schema": tool.schema,
            }
            for name, tool in sorted(self._tools.items())
        ]

    def get_schemas(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        selected = names or list(self._tools)
        return [
            {"name": n, "description": self._tools[n].description, "parameters": self._tools[n].schema}
            for n in selected if n in self._tools
        ]


def build_default_tool_registry(workdir: str = "/tmp/daga_sandbox") -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(BashTool(workdir))
    reg.register(FileReaderTool(workdir))
    reg.register(FileEditorTool(workdir))
    reg.register(ASTSearchTool(workdir))
    reg.register(RipgrepTool(workdir))
    reg.register(TestRunnerTool(workdir))
    reg.register(PatchApplyTool(workdir))
    return reg
