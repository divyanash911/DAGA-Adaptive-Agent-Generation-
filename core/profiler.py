"""
DAGA — Task Profiler
Extracts complexity, domain, and linguistic signals from a task description
and optional repository metadata, producing a TaskProfile.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from daga.core.models import (
    SLATarget,
    TaskComplexity,
    TaskDomain,
    TaskProfile,
)


# ──────────────────────────────────────────────
# Keyword tables
# ──────────────────────────────────────────────

_DOMAIN_KEYWORDS: Dict[TaskDomain, List[str]] = {
    TaskDomain.BUG_FIX: [
        "fix", "bug", "error", "exception", "crash", "failure", "broken",
        "regression", "traceback", "null pointer", "segfault", "issue",
    ],
    TaskDomain.CODE_GENERATION: [
        "implement", "add feature", "create", "write", "generate",
        "build", "develop", "new function", "new class",
    ],
    TaskDomain.REFACTOR: [
        "refactor", "clean up", "restructure", "rename", "reorganise",
        "optimize", "simplify", "extract", "decouple",
    ],
    TaskDomain.TEST_WRITING: [
        "test", "pytest", "unittest", "coverage", "assert", "mock",
        "integration test", "unit test",
    ],
    TaskDomain.DOCUMENTATION: [
        "document", "docstring", "readme", "comment", "annotate", "explain",
    ],
    TaskDomain.ANALYSIS: [
        "analyse", "audit", "review", "profil", "benchmark", "measure",
    ],
}

_COMPLEXITY_SIGNALS: Dict[str, int] = {
    # positive weight → raises complexity
    "cross-module":         +2,
    "multiple files":       +2,
    "architecture":         +3,
    "design pattern":       +2,
    "dependency":           +1,
    "async":                +1,
    "concurrent":           +2,
    "database migration":   +3,
    "breaking change":      +3,
    "performance":          +2,
    "security":             +2,
    "race condition":       +3,
    "deadlock":             +3,
    "memory leak":          +2,
    # negative weight → lowers complexity
    "typo":                 -2,
    "rename":               -1,
    "simple":               -1,
    "trivial":              -2,
    "one line":             -2,
    "minor":                -1,
}

_NER_PATTERNS = re.compile(
    r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,})\b'
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _token_entropy(text: str) -> float:
    """Shannon entropy of word tokens (normalised by log of vocab size)."""
    tokens = re.findall(r'\w+', text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    return entropy / max_entropy


def _detect_domain(text: str) -> TaskDomain:
    text_lower = text.lower()
    scores: Dict[TaskDomain, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=lambda d: scores[d])
    if scores[best] == 0:
        return TaskDomain.SOFTWARE_ENGINEERING
    # Prefer BUG_FIX if tied with others (most common in SWE-bench)
    if scores.get(TaskDomain.BUG_FIX, 0) == scores[best]:
        return TaskDomain.BUG_FIX
    return best


def _estimate_complexity(
    text: str,
    repo_file_count: int,
    affected_files_estimate: int,
    token_count: int,
) -> TaskComplexity:
    text_lower = text.lower()
    score = 0

    # Keyword signals
    for signal, weight in _COMPLEXITY_SIGNALS.items():
        if signal in text_lower:
            score += weight

    # Repo / scope signals
    if repo_file_count > 500:
        score += 2
    elif repo_file_count > 100:
        score += 1

    if affected_files_estimate >= 5:
        score += 3
    elif affected_files_estimate >= 3:
        score += 2
    elif affected_files_estimate >= 2:
        score += 1

    if token_count > 2000:
        score += 2
    elif token_count > 800:
        score += 1

    # Map to enum
    if score <= -2:
        return TaskComplexity.TRIVIAL
    if score <= 1:
        return TaskComplexity.SIMPLE
    if score <= 4:
        return TaskComplexity.MODERATE
    if score <= 8:
        return TaskComplexity.COMPLEX
    return TaskComplexity.EPIC


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

class TaskProfiler:
    """
    Converts a raw task description + optional repo metadata into a
    structured TaskProfile used by the architecture generator.
    """

    def profile(
        self,
        task_description: str,
        repo_metadata: Optional[Dict[str, Any]] = None,
        sla_target: SLATarget = SLATarget.BALANCED,
        deadline_seconds: Optional[float] = None,
        max_energy_joules: Optional[float] = None,
    ) -> TaskProfile:
        repo = repo_metadata or {}

        repo_file_count = repo.get("file_count", 0)
        has_tests       = bool(repo.get("has_tests", False))
        has_type_hints  = bool(repo.get("has_type_hints", False))
        language        = repo.get("language", "python")

        # Rough affected-files estimate from description
        file_patterns = re.findall(
            r'[\w/]+\.(?:py|js|ts|go|rs|java|cpp|c|h)\b',
            task_description
        )
        affected_files_estimate = max(len(set(file_patterns)), 1)

        tokens = re.findall(r'\w+', task_description)
        token_count = len(tokens)

        domain = _detect_domain(task_description)
        complexity = _estimate_complexity(
            task_description,
            repo_file_count,
            affected_files_estimate,
            token_count,
        )

        entropy = _token_entropy(task_description)
        ner_count = len(_NER_PATTERNS.findall(task_description))

        return TaskProfile(
            raw_input              = task_description,
            domain                 = domain,
            complexity             = complexity,
            repo_file_count        = repo_file_count,
            affected_files_estimate= affected_files_estimate,
            token_count            = token_count,
            has_tests              = has_tests,
            has_type_hints         = has_type_hints,
            language               = language,
            entropy                = entropy,
            named_entity_count     = ner_count,
            description_length     = len(task_description),
            sla_target             = sla_target,
            deadline_seconds       = deadline_seconds,
            max_energy_joules      = max_energy_joules,
        )
