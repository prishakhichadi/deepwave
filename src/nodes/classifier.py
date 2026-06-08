"""Reviews normalized metrics + AST findings and outputs a structured diagnosis classifying the
primary performance bottleneck. Uses LangChain with_structured_output for schema-validated JSON.
Prompt is grounded in AMD MI300X / CDNA3 hardware thresholds, not generic GPU rules."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from src.state import KernelAgentState, BottleneckDiagnosis


# ---------------------------------------------------------------------------
# AMD MI300X hardware reference thresholds used to guide the LLM classifier.
# These are grounded in AMD CDNA3 architecture specs and omniperf guidance docs.
# Having them in the prompt (not just the system message) makes reasoning explicit.
# ---------------------------------------------------------------------------
AMD_THRESHOLD_GUIDE = """
AMD MI300X Hardware Threshold Reference (CDNA3 Architecture):
--------------------------------------------------------------
METRIC              | THRESHOLD    | INTERPRETATION
--------------------|--------------|------------------------------------------
valu_util           | > 80%        | Compute Bound — VALU pipes are saturated
valu_util           | < 40%        | Not compute bound
mem_stalled         | > 50%        | Memory Bandwidth Bound — stalls dominate
mem_stalled         | > 30%        | Likely memory pressure worth addressing
max_waves_per_cu    | < 16 waves   | Occupancy Limited (MI300X max = 32 waves/CU)
max_waves_per_cu    | 16-24 waves  | Moderate occupancy — possible register spill
l2_cache_hit        | < 50%        | High L2 miss rate — memory access pattern issue
lds_bank_conflict   | > 5%         | LDS bank conflict — shared memory layout issue
salu_util           | > 30%        | High scalar activity — possible control flow issue

Classification Priority (if multiple signals present):
1. Memory Bandwidth Bound — if mem_stalled > 50% regardless of valu_util
2. Occupancy Limited — if max_waves_per_cu < 16 (blocks everything else)
3. Compute Bound — if valu_util > 80% and mem_stalled < 30%
4. Latency Bound — if valu_util < 40%, mem_stalled < 30%, and low occupancy
"""


def bottleneck_classifier_node(state: KernelAgentState) -> Dict:
    """
    Ingests parsed hardware metrics + AST findings, then uses a structured
    LLM call to emit a schema-validated BottleneckDiagnosis. Temperature=0
    enforces deterministic hardware reasoning.
    """
    # Temperature 0 — hardware diagnosis must be data-driven, not creative
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    structured_llm = llm.with_structured_output(BottleneckDiagnosis)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a Senior AMD GPU Performance Engineer specializing in CDNA3 architecture "
            "(MI300X). Your job is to diagnose the primary execution bottleneck from hardware "
            "telemetry metrics and kernel structure findings.\n\n"
            "Follow the classification priority strictly:\n"
            "1. If mem_stalled > 50% → Memory Bandwidth Bound (regardless of compute utilization)\n"
            "2. If max_waves_per_cu < 16 → Occupancy Limited (this gates everything else)\n"
            "3. If valu_util > 80% and mem_stalled < 30% → Compute Bound\n"
            "4. Otherwise → Latency Bound\n\n"
            "Your evidence list must include the SPECIFIC metric values that drove your decision "
            "(e.g. 'mem_stalled = 72.3%, threshold = 50%') — not generic statements.\n\n"
            "If a secondary bottleneck is also clearly present, note it in secondary_bottleneck.\n\n"
            f"{AMD_THRESHOLD_GUIDE}"
        ),
        (
            "human",
            "--- HARDWARE METRICS ---\n"
            "{metrics}\n\n"
            "--- AST STRUCTURAL FINDINGS ---\n"
            "{ast_insights}\n\n"
            "Diagnose the primary bottleneck."
        )
    ])

    chain = prompt | structured_llm

    # Pull AST insights from the richer findings list if available, else fall back
    ast_insights = state.get("ast_insights", ["No AST analysis available."])
    ast_text = "\n".join(ast_insights)

    diagnosis: BottleneckDiagnosis = chain.invoke({
        "metrics": _format_metrics(state["parsed_metrics"]),
        "ast_insights": ast_text,
    })

    print(f"[bottleneck_classifier] Diagnosis: {diagnosis.bottleneck_type} "
          f"(confidence: {diagnosis.confidence_score:.2f})")
    if diagnosis.secondary_bottleneck:
        print(f"  Secondary: {diagnosis.secondary_bottleneck}")

    return {"diagnosis": diagnosis}


def _format_metrics(metrics: Dict[str, float]) -> str:
    """Formats the metric dict as a readable table for LLM prompt injection."""
    lines = ["Metric                    | Value"]
    lines.append("-" * 40)
    for key, value in metrics.items():
        if value > 0.0:
            lines.append(f"{key:25s} | {value:.4f}")
    return "\n".join(lines) if len(lines) > 2 else "No non-zero metrics available."