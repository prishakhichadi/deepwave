"""DEEPWAVE LangGraph graph definition. Wires all nodes into a stateful reasoning loop
with a conditional feedback edge: if confidence is low after the first pass, the agent
re-runs the classifier and rewriter with updated context up to max_iterations times."""

from langgraph.graph import StateGraph, END
from src.state import KernelAgentState
from src.nodes.reader import profiling_reader_node
from src.nodes.analyzer import kernel_analyzer_node
from src.nodes.classifier import bottleneck_classifier_node
from src.nodes.planner import optimization_planner_node
from src.nodes.rewriter import kernel_rewriter_node
from src.nodes.reporter import report_writer_node
from src.nodes.critic import critic_node  # already exists in your codebase


# ---------------------------------------------------------------------------
# Confidence threshold — if diagnosis confidence is below this after a pass,
# the agent loops back to re-classify with enriched context.
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75
DEFAULT_MAX_ITERATIONS = 3


def should_loop(state: KernelAgentState) -> str:
    """
    Conditional edge function. Called after the critic node.
    Returns 'replan' to loop back, or 'done' to proceed to the report.

    Loop condition: diagnosis confidence is below threshold AND we haven't
    hit max_iterations yet. This prevents infinite loops on ambiguous kernels.
    """
    diagnosis = state.get("diagnosis")
    iteration = state.get("iteration_count", 1)
    max_iter = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)

    if diagnosis is None:
        print("[graph] No diagnosis available — proceeding to report.")
        return "done"

    if diagnosis.confidence_score < CONFIDENCE_THRESHOLD and iteration < max_iter:
        print(f"[graph] Low confidence ({diagnosis.confidence_score:.2f}) on iteration "
              f"{iteration}/{max_iter} — looping back to classifier.")
        return "replan"

    print(f"[graph] Confidence {diagnosis.confidence_score:.2f} sufficient or max iterations "
          f"reached ({iteration}/{max_iter}) — proceeding to report.")
    return "done"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the DEEPWAVE LangGraph agent.

    Pipeline:
        reader → analyzer → classifier → planner → rewriter → critic
                                ↑                                  |
                                └──────── (if low confidence) ─────┘
                                                                   |
                                                (if confident) → reporter → END
    """
    graph = StateGraph(KernelAgentState)

    # --- Register all nodes ---
    graph.add_node("reader",     profiling_reader_node)
    graph.add_node("analyzer",   kernel_analyzer_node)
    graph.add_node("classifier", bottleneck_classifier_node)
    graph.add_node("planner",    optimization_planner_node)
    graph.add_node("rewriter",   kernel_rewriter_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("reporter",   report_writer_node)

    # --- Linear edges (no branching) ---
    graph.add_edge("reader",     "analyzer")
    graph.add_edge("analyzer",   "classifier")
    graph.add_edge("classifier", "planner")
    graph.add_edge("planner",    "rewriter")
    graph.add_edge("rewriter",   "critic")

    # --- Conditional feedback edge from critic ---
    graph.add_conditional_edges(
        "critic",
        should_loop,
        {
            "replan": "classifier",   # Loop back — re-diagnose with richer context
            "done":   "reporter",     # Confident enough — write the final report
        }
    )

    graph.add_edge("reporter", END)

    # --- Entry point ---
    graph.set_entry_point("reader")

    return graph.compile()


# Compiled graph — import this in your runner / test harness
deepwave_graph = build_graph()