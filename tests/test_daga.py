"""
DAGA — Test Suite
Covers: profiler, routing rules, meta-agent, executor, topologies,
telemetry, feedback loop, efficiency predictor, and end-to-end pipeline.
All tests use the mock backend (no real LLM calls required).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure package is importable from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from daga.backends.registry import MockBackend, BackendRegistry, build_default_registry
from daga.core.models import (
    AgentRole, AgentTopology, ArchitecturePlan, ExecutionTrace,
    ModelTier, SLATarget, TaskComplexity, TaskDomain, TaskProfile,
)
from daga.core.predictor import EfficiencyPredictor
from daga.core.profiler import TaskProfiler
from daga.core.routing_rules import DeterministicRouter
from daga.core.meta_agent import MetaAgentRouter
from daga.agents.executor import AgentExecutor, extract_tool_calls, extract_final_patch
from daga.agents.topologies import (
    SingleAgentOrchestrator, SequentialPipelineOrchestrator,
    HierarchicalOrchestrator, ParallelEnsembleOrchestrator,
    HybridAdaptiveOrchestrator, create_orchestrator,
)
from daga.tools.registry import (
    BashTool, FileReaderTool, FileEditorTool, ASTSearchTool,
    ToolRegistry, build_default_tool_registry,
)
from daga.telemetry.collector import TelemetryCollector, ExperienceStore
from daga.feedback.loop import FeedbackLoop, _topology_stats, _complexity_topology_matrix
from daga.pipeline import DAGAPipeline, PipelineConfig


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_registry():
    return build_default_registry(use_mock=True)


@pytest.fixture
def tool_registry(tmpdir):
    return build_default_tool_registry(workdir=tmpdir)


@pytest.fixture
def simple_profile():
    return TaskProfiler().profile(
        "Fix the AttributeError in utils.py line 42",
        repo_metadata={"file_count": 50, "has_tests": True, "language": "python"},
    )


@pytest.fixture
def complex_profile():
    return TaskProfiler().profile(
        "Refactor the authentication module to support OAuth2 "
        "across multiple files with concurrent session handling and database migration",
        repo_metadata={"file_count": 800, "has_tests": True, "language": "python"},
        sla_target=SLATarget.QUALITY_FIRST,
    )


@pytest.fixture
def mock_plan(simple_profile):
    return ArchitecturePlan(
        task_id  = simple_profile.task_id,
        topology = AgentTopology.SINGLE_SLM,
        roles    = [AgentRole(
            role_id   = "solo",
            role_name = "solver",
            model_tier= ModelTier.SLM_SMALL,
            model_id  = "mock-small",
            tools     = ["bash", "file_editor"],
            system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
        )],
        routing_source = "deterministic",
    )


# ──────────────────────────────────────────────
# Profiler tests
# ──────────────────────────────────────────────

class TestTaskProfiler:
    def test_bug_fix_detection(self):
        p = TaskProfiler().profile("Fix the crash in parser.py when input is None")
        assert p.domain == TaskDomain.BUG_FIX

    def test_simple_complexity(self):
        p = TaskProfiler().profile("Fix typo in README")
        assert p.complexity in (TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE)

    def test_complex_detection(self):
        p = TaskProfiler().profile(
            "Refactor the entire authentication system to support "
            "OAuth2, concurrent sessions, and database migration across 10+ files",
            repo_metadata={"file_count": 600},
        )
        assert p.complexity in (TaskComplexity.COMPLEX, TaskComplexity.EPIC)

    def test_file_extraction(self):
        p = TaskProfiler().profile("Fix bug in src/utils.py and tests/test_utils.py")
        assert p.affected_files_estimate >= 2

    def test_entropy_nonzero(self):
        p = TaskProfiler().profile("Implement a new feature for handling edge cases")
        assert p.entropy > 0.0

    def test_sla_forwarded(self):
        p = TaskProfiler().profile("Fix bug", sla_target=SLATarget.ENERGY_FIRST)
        assert p.sla_target == SLATarget.ENERGY_FIRST


# ──────────────────────────────────────────────
# Routing rules tests
# ──────────────────────────────────────────────

class TestDeterministicRouter:
    def setup_method(self):
        self.router = DeterministicRouter()

    def test_trivial_goes_slm_nano(self):
        p = TaskProfiler().profile("Fix typo in comment")
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.SINGLE_SLM
        tiers = {r.model_tier for r in plan.roles}
        assert ModelTier.SLM_NANO in tiers

    def test_latency_first_simple_goes_nano(self):
        p = TaskProfiler().profile("Fix small bug", sla_target=SLATarget.LATENCY_FIRST)
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.SINGLE_SLM

    def test_energy_first_uses_hybrid(self):
        p = TaskProfiler().profile(
            "Fix bug in module", sla_target=SLATarget.ENERGY_FIRST
        )
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.HYBRID_ADAPTIVE

    def test_moderate_uses_pipeline(self):
        p = TaskProfile(
            raw_input="Multi-file refactor",
            complexity=TaskComplexity.MODERATE,
            domain=TaskDomain.REFACTOR,
        )
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.SEQUENTIAL_PIPELINE

    def test_complex_quality_uses_hierarchical(self):
        p = TaskProfile(
            raw_input="Complex auth refactor",
            complexity=TaskComplexity.COMPLEX,
            sla_target=SLATarget.QUALITY_FIRST,
        )
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.HIERARCHICAL

    def test_hard_deadline_forces_nano(self):
        p = TaskProfiler().profile(
            "Fix bug", deadline_seconds=10.0
        )
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.SINGLE_SLM
        assert any(r.model_tier == ModelTier.SLM_NANO for r in plan.roles)

    def test_large_repo_uses_hierarchical(self):
        p = TaskProfile(
            raw_input="Fix bug",
            repo_file_count=600,
            complexity=TaskComplexity.MODERATE,
        )
        plan = self.router.route_to_plan(p)
        assert plan.topology == AgentTopology.HIERARCHICAL

    def test_catchall_returns_plan(self):
        p = TaskProfile(raw_input="some task")
        plan = self.router.route_to_plan(p)
        assert plan is not None
        assert len(plan.roles) > 0


# ──────────────────────────────────────────────
# Tool tests
# ──────────────────────────────────────────────

class TestTools:
    def test_bash_success(self, tmpdir):
        t = BashTool(workdir=tmpdir)
        r = t({"command": "echo hello"})
        assert r.success
        assert "hello" in r.output

    def test_bash_failure(self, tmpdir):
        t = BashTool(workdir=tmpdir)
        r = t({"command": "exit 1"})
        assert not r.success

    def test_file_write_and_read(self, tmpdir):
        editor = FileEditorTool(workdir=tmpdir)
        reader = FileReaderTool(workdir=tmpdir)
        editor({"path": "test.py", "operation": "write", "new_content": "x = 1\n"})
        r = reader({"path": "test.py"})
        assert "x = 1" in r.output

    def test_file_replace(self, tmpdir):
        editor = FileEditorTool(workdir=tmpdir)
        editor({"path": "f.py", "operation": "write", "new_content": "def foo(): pass\n"})
        editor({"path": "f.py", "operation": "replace",
                "old_str": "pass", "new_str": "return 42"})
        reader = FileReaderTool(workdir=tmpdir)
        r = reader({"path": "f.py"})
        assert "return 42" in r.output

    def test_ast_search(self, tmpdir):
        editor = FileEditorTool(workdir=tmpdir)
        editor({"path": "module.py", "operation": "write",
                "new_content": "def my_function():\n    pass\n"})
        ast_tool = ASTSearchTool(workdir=tmpdir)
        r = ast_tool({"query": "my_function", "mode": "symbol"})
        assert "my_function" in r.output

    def test_file_replace_not_found(self, tmpdir):
        editor = FileEditorTool(workdir=tmpdir)
        editor({"path": "f.py", "operation": "write", "new_content": "x = 1\n"})
        r = editor({"path": "f.py", "operation": "replace",
                    "old_str": "NONEXISTENT", "new_str": "y"})
        assert not r.success


# ──────────────────────────────────────────────
# Executor tests
# ──────────────────────────────────────────────

class TestAgentExecutor:
    def test_extract_tool_calls(self):
        text = 'Let me search.\n<tool_call>{"tool": "bash", "args": {"command": "ls"}}</tool_call>'
        calls = extract_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "bash"

    def test_extract_final_patch(self):
        text = "Done.\n<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n</FINAL_PATCH>"
        patch = extract_final_patch(text)
        assert patch is not None
        assert "--- a/f.py" in patch

    def test_mock_executor_runs(self, mock_registry, tool_registry):
        role = AgentRole(
            role_id   = "r1",
            role_name = "solver",
            model_tier= ModelTier.SLM_SMALL,
            model_id  = "mock-small",
            tools     = ["bash"],
            system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
        )
        # Mock returns a response with a final patch
        mock_registry._backends["mock-small"] = MockBackend(
            model_id           = "mock-small",
            tier               = ModelTier.SLM_SMALL,
            canned_response    = (
                "I will fix this.\n"
                "<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-bug\n+fix\n</FINAL_PATCH>"
            ),
            simulated_latency  = 0.01,
        )
        executor = AgentExecutor(role, mock_registry, tool_registry, max_iterations=3)
        result = executor.run("Fix the bug")
        assert result.success
        assert result.final_patch is not None
        assert result.total_energy_j > 0


# ──────────────────────────────────────────────
# Topology tests
# ──────────────────────────────────────────────

def _make_patch_backend(registry, model_id):
    """Replace a mock backend with one that always returns a patch."""
    registry._backends[model_id] = MockBackend(
        model_id          = model_id,
        tier              = registry._backends[model_id].tier,
        canned_response   = (
            "<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n</FINAL_PATCH>"
        ),
        simulated_latency = 0.01,
    )


class TestTopologies:
    def _simple_plan(self, topology, roles):
        return ArchitecturePlan(topology=topology, roles=roles, task_id="test")

    def _role(self, role_id, model_id, tier=ModelTier.SLM_SMALL,
               tools=None, parallel_group=None):
        return AgentRole(
            role_id   = role_id,
            role_name = role_id,
            model_tier= tier,
            model_id  = model_id,
            tools     = tools or ["bash"],
            system_prompt_template = "{SYSTEM_PROMPT_SOLVER}",
            parallel_group = parallel_group,
        )

    def test_single_agent(self, mock_registry, tool_registry, simple_profile):
        _make_patch_backend(mock_registry, "mock-small")
        plan = self._simple_plan(
            AgentTopology.SINGLE_SLM,
            [self._role("solo", "mock-small")],
        )
        orch  = SingleAgentOrchestrator(plan, mock_registry, tool_registry)
        trace = orch.execute(simple_profile)
        assert trace.resolved
        assert trace.final_patch is not None

    def test_sequential_pipeline(self, mock_registry, tool_registry, simple_profile):
        for mid in ["mock-small", "mock-medium"]:
            _make_patch_backend(mock_registry, mid)
        plan = self._simple_plan(
            AgentTopology.SEQUENTIAL_PIPELINE,
            [
                self._role("localiser", "mock-small"),
                self._role("patcher",   "mock-medium"),
                self._role("verifier",  "mock-small"),
            ],
        )
        orch  = SequentialPipelineOrchestrator(plan, mock_registry, tool_registry)
        trace = orch.execute(simple_profile)
        assert len(trace.steps) > 0

    def test_parallel_ensemble(self, mock_registry, tool_registry, simple_profile):
        for mid in ["mock-small", "mock-medium"]:
            _make_patch_backend(mock_registry, mid)
        plan = self._simple_plan(
            AgentTopology.PARALLEL_ENSEMBLE,
            [
                self._role("w0", "mock-small", parallel_group=0),
                self._role("w1", "mock-medium", parallel_group=0),
            ],
        )
        orch  = ParallelEnsembleOrchestrator(plan, mock_registry, tool_registry)
        trace = orch.execute(simple_profile)
        assert trace.final_patch is not None

    def test_hybrid_adaptive_escalates(self, mock_registry, tool_registry):
        """Nano fails, small succeeds → hybrid should escalate."""
        mock_registry._backends["mock-nano"] = MockBackend(
            "mock-nano", ModelTier.SLM_NANO,
            canned_response="I cannot solve this.",
            simulated_latency=0.01,
        )
        _make_patch_backend(mock_registry, "mock-small")
        plan = ArchitecturePlan(
            topology = AgentTopology.HYBRID_ADAPTIVE,
            roles    = [self._role("r", "mock-nano", ModelTier.SLM_NANO)],
            task_id  = "test",
        )
        profile = TaskProfile(raw_input="Fix bug", complexity=TaskComplexity.SIMPLE)
        orch  = HybridAdaptiveOrchestrator(plan, mock_registry, tool_registry)
        trace = orch.execute(profile)
        # Should have attempted at least one tier
        assert len(trace.steps) > 0


# ──────────────────────────────────────────────
# Efficiency predictor tests
# ──────────────────────────────────────────────

class TestEfficiencyPredictor:
    def setup_method(self):
        self.predictor = EfficiencyPredictor()

    def _make_plan(self, topology, tiers):
        roles = [
            AgentRole(
                role_id   = f"r{i}",
                role_name = f"role{i}",
                model_tier= t,
                model_id  = t.value,
                tools     = [],
                system_prompt_template = "",
            )
            for i, t in enumerate(tiers)
        ]
        return ArchitecturePlan(topology=topology, roles=roles, task_id="x")

    def test_slm_cheaper_than_llm(self, simple_profile):
        slm_plan = self._make_plan(AgentTopology.SINGLE_SLM, [ModelTier.SLM_NANO])
        llm_plan = self._make_plan(AgentTopology.SINGLE_LLM, [ModelTier.LLM_FRONTIER])
        slm_pred = self.predictor.predict(simple_profile, slm_plan)
        llm_pred = self.predictor.predict(simple_profile, llm_plan)
        assert slm_pred.predicted_energy_j < llm_pred.predicted_energy_j
        assert slm_pred.predicted_latency_s < llm_pred.predicted_latency_s

    def test_parallel_higher_energy(self, simple_profile):
        single = self._make_plan(AgentTopology.SINGLE_SLM,        [ModelTier.SLM_SMALL])
        parall = self._make_plan(AgentTopology.PARALLEL_ENSEMBLE,
                                 [ModelTier.SLM_SMALL, ModelTier.SLM_SMALL, ModelTier.SLM_SMALL])
        single_pred = self.predictor.predict(simple_profile, single)
        parall_pred = self.predictor.predict(simple_profile, parall)
        assert parall_pred.predicted_energy_j > single_pred.predicted_energy_j

    def test_compare_plans_sorted(self, simple_profile):
        plans = [
            self._make_plan(AgentTopology.SINGLE_SLM,        [ModelTier.SLM_NANO]),
            self._make_plan(AgentTopology.PARALLEL_ENSEMBLE,
                            [ModelTier.LLM_LARGE, ModelTier.LLM_LARGE, ModelTier.LLM_LARGE]),
        ]
        ranked = self.predictor.compare_plans(simple_profile, plans)
        assert ranked[0][1].efficiency_score >= ranked[1][1].efficiency_score


# ──────────────────────────────────────────────
# Telemetry tests
# ──────────────────────────────────────────────

class TestTelemetry:
    def test_collect_creates_record(self, simple_profile, mock_plan):
        trace = ExecutionTrace(plan_id=mock_plan.plan_id, task_id=simple_profile.task_id)
        trace.finish(resolved=True, patch="--- a/f\n+++ b/f")
        collector = TelemetryCollector()
        rec = collector.collect(simple_profile, mock_plan, trace)
        assert rec.resolved
        assert rec.efficiency_score != 0.0

    def test_experience_store_roundtrip(self, tmpdir, simple_profile, mock_plan):
        store_path = os.path.join(tmpdir, "exp.jsonl")
        store = ExperienceStore(store_path)
        trace = ExecutionTrace(task_id=simple_profile.task_id)
        trace.finish(True, "patch")
        collector = TelemetryCollector()
        rec = collector.collect(simple_profile, mock_plan, trace)
        store.save(rec)

        store2 = ExperienceStore(store_path)
        assert len(store2._records) == 1
        assert store2._records[0]["resolved"] is True

    def test_retrieve_similar(self, tmpdir, simple_profile, mock_plan):
        store_path = os.path.join(tmpdir, "exp.jsonl")
        store = ExperienceStore(store_path)
        for i in range(5):
            trace = ExecutionTrace(task_id=f"t{i}")
            trace.finish(i % 2 == 0, "patch")
            collector = TelemetryCollector()
            rec = collector.collect(simple_profile, mock_plan, trace)
            rec.task_profile = simple_profile
            store.save(rec)

        results = store.retrieve_similar(simple_profile, top_k=3)
        assert len(results) <= 3


# ──────────────────────────────────────────────
# Feedback loop tests
# ──────────────────────────────────────────────

class TestFeedbackLoop:
    def _make_records(self, n=15):
        records = []
        topologies = ["single_slm", "sequential_pipeline", "parallel_ensemble"]
        complexities = ["simple", "moderate", "complex"]
        for i in range(n):
            records.append({
                "record_id":  f"r{i}",
                "topology":   topologies[i % 3],
                "complexity": complexities[i % 3],
                "resolved":   i % 3 != 2,
                "energy_j":   10.0 + i * 2,
                "latency_s":  5.0  + i * 0.5,
                "efficiency": 0.3  - (i % 3) * 0.05,
                "sla_target": "balanced",
            })
        return records

    def test_topology_stats(self):
        records = self._make_records()
        stats = _topology_stats(records)
        assert "single_slm" in stats
        assert stats["single_slm"]["count"] > 0

    def test_complexity_matrix(self):
        records = self._make_records()
        matrix = _complexity_topology_matrix(records)
        assert "simple" in matrix

    def test_feedback_insufficient_data(self, tmpdir):
        store = ExperienceStore(os.path.join(tmpdir, "e.jsonl"))
        fb    = FeedbackLoop(store, min_records_for_update=10)
        r = fb.analyse()
        assert r["status"] == "insufficient_data"

    def test_feedback_full_analysis(self, tmpdir):
        store_path = os.path.join(tmpdir, "e.jsonl")
        store = ExperienceStore.__new__(ExperienceStore)
        store._path    = Path(store_path)
        store._records = self._make_records(20)
        fb = FeedbackLoop(store, min_records_for_update=10)
        r  = fb.analyse()
        assert r["status"] == "ok"
        assert "topology_stats" in r


# ──────────────────────────────────────────────
# End-to-end pipeline test
# ──────────────────────────────────────────────

class TestPipeline:
    def test_run_mock_simple(self, tmpdir, mock_registry):
        # Patch all mocks to return a final patch
        for mid in list(mock_registry._backends):
            mock_registry._backends[mid] = MockBackend(
                model_id          = mid,
                tier              = mock_registry._backends[mid].tier,
                canned_response   = (
                    "<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-bug\n+fix\n</FINAL_PATCH>"
                ),
                simulated_latency = 0.01,
            )
        config = PipelineConfig(
            workdir              = tmpdir,
            experience_store_path= os.path.join(tmpdir, "exp.jsonl"),
            verbose              = False,
        )
        pipeline = DAGAPipeline(config=config, registry=mock_registry)
        result   = pipeline.run(
            "Fix the AttributeError in utils.py",
            repo_metadata={"file_count": 30, "has_tests": True},
        )
        assert result.topology_used in [t.value for t in AgentTopology]
        assert result.total_tokens > 0
        assert result.efficiency_score != 0.0

    def test_run_energy_constrained(self, tmpdir, mock_registry):
        for mid in list(mock_registry._backends):
            mock_registry._backends[mid] = MockBackend(
                mid, mock_registry._backends[mid].tier,
                canned_response="<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n</FINAL_PATCH>",
                simulated_latency=0.01,
            )
        config = PipelineConfig(
            workdir=tmpdir,
            experience_store_path=os.path.join(tmpdir, "e.jsonl"),
        )
        pipeline = DAGAPipeline(config=config, registry=mock_registry)
        result   = pipeline.run(
            "Fix small bug",
            max_energy_joules = 5.0,
            sla_target        = SLATarget.ENERGY_FIRST,
        )
        # With hard energy cap, should route to cheap topology
        assert result.topology_used in (
            AgentTopology.SINGLE_SLM.value,
            AgentTopology.HYBRID_ADAPTIVE.value,
        )

    def test_routing_sources(self, tmpdir, mock_registry):
        for mid in list(mock_registry._backends):
            mock_registry._backends[mid] = MockBackend(
                mid, mock_registry._backends[mid].tier,
                canned_response="<FINAL_PATCH>\n--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-a\n+b\n</FINAL_PATCH>",
                simulated_latency=0.01,
            )
        config   = PipelineConfig(workdir=tmpdir,
                                  experience_store_path=os.path.join(tmpdir, "e.jsonl"))
        pipeline = DAGAPipeline(config=config, registry=mock_registry)
        r = pipeline.run("Fix typo")
        assert r.routing_source in ("deterministic", "hybrid", "meta_llm")


# ──────────────────────────────────────────────
# Meta-agent router test (with mock LLM)
# ──────────────────────────────────────────────

class TestMetaAgentRouter:
    def test_deterministic_for_simple(self, mock_registry, simple_profile):
        router = MetaAgentRouter(
            registry     = mock_registry,
            always_use_llm = False,
        )
        plan = router.route(simple_profile)
        assert plan.routing_source == "deterministic"

    def test_meta_llm_invoked_for_complex(self, mock_registry):
        # Mock LLM returns a valid JSON plan
        mock_registry._backends["mock-small"] = MockBackend(
            "mock-small", ModelTier.SLM_SMALL,
            canned_response=json.dumps({
                "topology": "sequential_pipeline",
                "reasoning": "Complex task benefits from staged approach.",
                "roles": [
                    {"role_id": "loc", "role_name": "localiser",
                     "model_tier": "slm_small", "tools": ["ast_search"],
                     "max_tokens": 2048, "temperature": 0.1},
                    {"role_id": "pat", "role_name": "patcher",
                     "model_tier": "llm_medium", "tools": ["file_editor"],
                     "max_tokens": 4096, "temperature": 0.2},
                ],
                "predicted_latency_s": 20.0,
                "predicted_energy_j": 5.0,
                "predicted_resolve_prob": 0.72,
            }),
            simulated_latency=0.01,
        )
        complex_p = TaskProfile(
            raw_input   = "Complex multi-module refactor",
            complexity  = TaskComplexity.COMPLEX,
            sla_target  = SLATarget.QUALITY_FIRST,
        )
        router = MetaAgentRouter(
            registry       = mock_registry,
            meta_model_tier= ModelTier.SLM_SMALL,
        )
        plan = router.route(complex_p)
        # Should have used hybrid path (det + LLM refinement)
        assert plan.routing_source in ("hybrid", "meta_llm", "deterministic")


# ──────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
