"""
Tests for DEEPWAVE.

Split into two tiers:
  1. Deterministic tests (reader, analyzer, graph structure) — no API key needed,
     always run in CI.
  2. LLM-backed tests (classifier, planner, rewriter, critic, full graph invoke) —
     require OPENAI_API_KEY, auto-skipped if it isn't set.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.reader import profiling_reader_node
from src.nodes.analyzer import kernel_analyzer_node
from src.nodes.evidence_validator import cross_validate_diagnosis
from src.nodes.severity import classify_severity
from src.state import BottleneckDiagnosis, ASTFinding
from src.graph import deepwave_graph
from config.settings import settings

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "corpus" / "synthetic"
SCENARIOS = [
    "memory_bandwidth_bound",
    "compute_bound",
    "occupancy_limited",
    "latency_bound",
]

HAS_API_KEY = bool(settings.openai_api_key)
requires_llm = pytest.mark.skipif(
    not HAS_API_KEY, reason="OPENAI_API_KEY not set — skipping LLM-backed test"
)


def _corpus_dir_exists() -> bool:
    return CORPUS_ROOT.exists() and any(CORPUS_ROOT.iterdir())


requires_corpus = pytest.mark.skipif(
    not _corpus_dir_exists(),
    reason="Synthetic corpus not found — run `python generate_corpus.py` first",
)


def _load_scenario(name: str):
    kernel = (CORPUS_ROOT / name / "kernel.hip").read_text()
    profiling = (CORPUS_ROOT / name / "rocprof.csv").read_text()
    return kernel, profiling


# Tier 1: deterministic, no API key required

def test_graph_compiles():
    """The LangGraph state machine should build without hitting the network."""
    nodes = set(deepwave_graph.get_graph().nodes.keys())
    expected = {
        "__start__", "reader", "analyzer", "classifier",
        "planner", "rewriter", "critic", "reporter", "__end__",
    }
    assert expected.issubset(nodes)


@requires_corpus
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_profiling_reader_extracts_metrics(scenario):
    kernel, profiling = _load_scenario(scenario)
    state = {"raw_kernel_code": kernel, "raw_profiling_data": profiling, "iteration_count": 0}

    result = profiling_reader_node(state)

    assert "parsed_metrics" in result
    assert isinstance(result["parsed_metrics"], dict)
    assert len(result["parsed_metrics"]) > 0
    for key, value in result["parsed_metrics"].items():
        assert isinstance(value, float), f"{key} should parse as float"


def test_profiling_reader_handles_long_format_csv():
    """Regression test: a hand-typed 'metric,value' long-format CSV (one metric per row,
    lowercase snake_case names) previously silently parsed to all zeros because the reader
    only understood rocprof's 'wide' layout with exact-case column names. This should now
    extract real values from either layout."""
    long_csv = (
        "metric,value\n"
        "valu_util,18.3\n"
        "salu_util,8.1\n"
        "mem_stalled,72.4\n"
        "max_waves_per_cu,24.0\n"
        "l2_cache_hit,31.2\n"
        "lds_bank_conflict,0.0\n"
    )
    state = {"raw_profiling_data": long_csv, "iteration_count": 0}

    result = profiling_reader_node(state)

    assert result["parsed_metrics"]["mem_stalled"] == pytest.approx(72.4)
    assert result["parsed_metrics"]["valu_util"] == pytest.approx(18.3)
    assert result["parsed_metrics"]["max_waves_per_cu"] == pytest.approx(24.0)


@requires_corpus
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_kernel_analyzer_runs_without_error(scenario):
    """Regression test for the tree-sitter >=0.22 Query/QueryCursor API change —
    this previously raised AttributeError / QueryError on every scenario."""
    kernel, profiling = _load_scenario(scenario)
    state = {"raw_kernel_code": kernel, "raw_profiling_data": profiling, "iteration_count": 0}
    state.update(profiling_reader_node(state))

    result = kernel_analyzer_node(state)

    assert "ast_findings" in result
    assert isinstance(result["ast_findings"], list)
    assert "ast_insights" in result


@requires_corpus
def test_compute_bound_scenario_flags_missing_shared_memory():
    """Sanity check that analyzer findings are meaningful, not just non-crashing:
    the compute_bound synthetic kernel has global-pointer loops with no __shared__
    usage, so it should be flagged."""
    kernel, profiling = _load_scenario("compute_bound")
    state = {"raw_kernel_code": kernel, "raw_profiling_data": profiling, "iteration_count": 0}
    state.update(profiling_reader_node(state))
    result = kernel_analyzer_node(state)

    finding_types = {f.finding_type for f in result["ast_findings"]}
    assert "missing_shared_memory" in finding_types


# Tier 2: LLM-backed, requires OPENAI_API_KEY, now groq lol

@requires_llm
@requires_corpus
def test_full_graph_invoke_memory_bandwidth_bound():
    """End-to-end smoke test: runs the full agent loop on the clearest-cut synthetic
    scenario and checks that every output artifact was actually produced."""
    kernel, profiling = _load_scenario("memory_bandwidth_bound")
    initial_state = {
        "raw_kernel_code": kernel,
        "raw_profiling_data": profiling,
        "iteration_count": 0,
        "max_iterations": settings.max_iterations,
    }

    final_state = deepwave_graph.invoke(initial_state)

    assert final_state.get("diagnosis") is not None
    assert final_state["diagnosis"].bottleneck_type == "Memory Bandwidth Bound"
    assert final_state.get("optimized_kernel_code")
    assert final_state.get("final_report")


# Evidence cross-validation- deterministic, no LLM/API key required

def _diagnosis(bottleneck_type, confidence=0.9):
    return BottleneckDiagnosis(
        bottleneck_type=bottleneck_type, confidence_score=confidence, evidence=["test evidence"]
    )


def _finding(finding_type):
    return ASTFinding(
        finding_type=finding_type, location="test_fn", description="test", severity="high"
    )


def test_cross_validation_confirms_when_ast_supports_diagnosis():
    diagnosis = _diagnosis("Memory Bandwidth Bound")
    findings = [_finding("uncoalesced_memory_access")]

    status, detail = cross_validate_diagnosis(diagnosis, findings)

    assert status == "confirmed"
    assert "uncoalesced_memory_access" in detail


def test_cross_validation_flags_conflict():
    diagnosis = _diagnosis("Compute Bound")
    findings = [_finding("uncoalesced_memory_access")]  # points to memory-bound instead

    status, detail = cross_validate_diagnosis(diagnosis, findings)

    assert status == "conflicting"
    assert "review" in detail.lower()


def test_cross_validation_metrics_only_when_no_corroborating_finding():
    diagnosis = _diagnosis("Occupancy Limited")
    findings = [_finding("loop_structure")]  # unrelated to occupancy

    status, detail = cross_validate_diagnosis(diagnosis, findings)

    assert status == "metrics_only"


def test_cross_validation_metrics_only_when_no_findings_at_all():
    diagnosis = _diagnosis("Latency Bound")

    status, detail = cross_validate_diagnosis(diagnosis, [])

    assert status == "metrics_only"



# Severity scoring: deterministic, no LLM/API key required


def test_severity_distinguishes_borderline_from_critical_at_same_bottleneck():
    """The exact case the user asked about: mem_stalled=51% and mem_stalled=95% both
    diagnose as 'Memory Bandwidth Bound', but should NOT be treated identically."""
    borderline_label, borderline_score, _ = classify_severity(
        "Memory Bandwidth Bound", {"mem_stalled": 51.0}
    )
    critical_label, critical_score, _ = classify_severity(
        "Memory Bandwidth Bound", {"mem_stalled": 95.0}
    )

    assert borderline_label != critical_label
    assert critical_score > borderline_score
    assert borderline_label == "borderline"
    assert critical_label == "critical"


def test_severity_occupancy_limited_lower_is_worse():
    """Occupancy is inverted: FEWER waves per CU is worse, not more."""
    near_threshold_label, _, _ = classify_severity(
        "Occupancy Limited", {"max_waves_per_cu": 15.0}
    )
    very_low_label, _, _ = classify_severity(
        "Occupancy Limited", {"max_waves_per_cu": 2.0}
    )

    assert near_threshold_label == "borderline"
    assert very_low_label == "critical"


def test_severity_unscored_when_metric_missing():
    label, score, _ = classify_severity("Memory Bandwidth Bound", {})
    assert label == "unscored"
    assert score == 0.0