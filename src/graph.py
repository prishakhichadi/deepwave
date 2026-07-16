"""if confidence is low after first pass, agent re-runs classifier and
rewriter with updated context up to max_iterations times."""

from langgraph.graph import StateGraph, END
from src.state import KernelAgentState
from src.nodes.reader import profiling_reader_node
from src.nodes.analyzer import kernel_analyzer_node
from src.nodes.classifier import bottleneck_classifier_node
from src.nodes.planner import optimization_planner_node
from src.nodes.rewriter import kernel_rewriter_node
from src.nodes.reporter import report_writer_node
from src.nodes.critic import compiler_simulator_critic_node as critic_node
from config.settings import settings


CONFIDENCE_THRESHOLD = settings.confidence_threshold
DEFAULT_MAX_ITERATIONS = settings.max_iterations


def should_loop(state: KernelAgentState) -> str:
    """
    Conditional edge function. Called after the critic node.
    Returns 'replan' to loop back, or 'done' to proceed to the report.

    Loop condition: the critic found the rewrite syntactically/structurally invalid,
    AND we haven't hit max_iterations yet. This prevents infinite loops on
    persistently broken generations.
    """
    status = state.get("verification_status")
    iteration = state.get("iteration_count", 1)
    max_iter = state.get("max_iterations", 3)
    
    if status == "failed_retry" and iteration < max_iter:
        return "replan"
    return "done"


def build_graph() -> StateGraph:
    """
    Constructs and compiles the DEEPWAVE LangGraph agent.

    Pipeline:
        reader → analyzer → classifier → planner → rewriter → critic
                                                        ↑         |
                                                        └─ retry ─┘
                                                       (if critic rejects the rewrite)
                                                                  |
                                                (if valid) → reporter → END
    """
    graph = StateGraph(KernelAgentState)


    graph.add_node("reader",     profiling_reader_node)
    graph.add_node("analyzer",   kernel_analyzer_node)
    graph.add_node("classifier", bottleneck_classifier_node)
    graph.add_node("planner",    optimization_planner_node)
    graph.add_node("rewriter",   kernel_rewriter_node)
    graph.add_node("critic",     critic_node)
    graph.add_node("reporter",   report_writer_node)

    # linear edges (no branching)
    graph.add_edge("reader",     "analyzer")
    graph.add_edge("analyzer",   "classifier")
    graph.add_edge("classifier", "planner")
    graph.add_edge("planner",    "rewriter")
    graph.add_edge("rewriter",   "critic")


    graph.add_conditional_edges(
        "critic",
        should_loop,
        {
            "replan": "rewriter",     # Loop back- regenerate the code using critic feedback
            "done":   "reporter",     # Valid- write the final report
        }
    )

    graph.add_edge("reporter", END)


    graph.set_entry_point("reader")

    return graph.compile()


deepwave_graph = build_graph()