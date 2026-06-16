"""Cross-references the bottleneck diagnosis, optimization plan, and AST structural findings
to surgically rewrite the .hip kernel. The plan node's output is the primary driver —
this prevents the LLM from applying generic optimizations unrelated to the actual bottleneck."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from src.state import KernelAgentState, OptimizedKernelOutput


def kernel_rewriter_node(state: KernelAgentState) -> Dict:
    """
    Surgically refactors the HIP/CUDA kernel using the structured optimization plan,
    AST findings, and empirical bottleneck diagnosis. Every modification must be
    tied to a specific hardware constraint — no speculative changes allowed.
    """
    diagnosis = state.get("diagnosis")
    plan = state.get("optimization_plan")

    if diagnosis is None:
        raise ValueError("kernel_rewriter_node requires diagnosis in state.")
    if plan is None:
        raise ValueError("kernel_rewriter_node requires optimization_plan in state. "
                         "Run optimization_planner_node first.")

    # Temperature 0.2 — allow creative strategy selection (loop unrolling vs tiling)
    # but keep code generation grounded in the plan
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    structured_writer = llm.with_structured_output(OptimizedKernelOutput)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a world-class AMD GPU kernel engineer.\n"
            "Rewrite the provided HIP kernel following the APPROVED OPTIMIZATION PLAN exactly.\n\n"
            "STRICT RULES:\n"
            "1. Only modify code in the TARGET SCOPES listed in the plan.\n"
            "2. Apply ONLY the approved strategy — do not add unrelated optimizations.\n"
            "3. Every modification must appear in line_by_line_annotations with a hardware "
            "justification that cites specific AMD MI300X facts (cache line = 128 bytes, "
            "wavefront = 64 threads, LDS = 64KB/CU, etc.).\n"
            "4. Preserve all kernel function signatures and __global__ qualifiers.\n"
            "5. Add __restrict__ to pointer parameters if not already present.\n"
            "6. In theoretical_improvement, give a concrete estimate: "
            "e.g. '2-4x memory bandwidth improvement due to 128-byte coalesced loads replacing "
            "strided access pattern. Expected mem_stalled reduction from ~72% to ~25%'.\n\n"
            "AMD MI300X Reference:\n"
            "- Cache line: 128 bytes\n"
            "- Wavefront width: 64 threads\n"
            "- LDS per CU: 64 KB\n"
            "- Max waves per CU: 32\n"
            "- L2 cache: 256 MB (shared across all XCDs)\n"
            "- VGPR file: 512 per wavefront\n"
        ),
        (
            "human",
            "--- BOTTLENECK DIAGNOSIS ---\n"
            "Type: {b_type}\n"
            "Evidence:\n{b_evidence}\n\n"
            "--- APPROVED OPTIMIZATION PLAN ---\n"
            "Strategy: {strategy_name}\n"
            "Target Scopes: {target_scopes}\n"
            "Rationale: {rationale}\n"
            "AMD-Specific Hints: {amd_hints}\n\n"
            "--- AST STRUCTURAL FINDINGS ---\n"
            "{ast_insights}\n\n"
            "--- ORIGINAL SOURCE CODE ---\n"
            "{source_code}\n\n"
            "Apply the strategy. Return the full rewritten kernel."
            "--- CRITIC FEEDBACK FROM PREVIOUS PASS ---\n"
            "{critic_feedback}\n\n"
        )
    ])

    chain = prompt | structured_writer

    ast_insights = state.get("ast_insights", ["No AST findings available."])

    rewritten: OptimizedKernelOutput = chain.invoke({
        "b_type": diagnosis.bottleneck_type,
        "b_evidence": "\n".join(f"  - {e}" for e in diagnosis.evidence),
        "strategy_name": plan.strategy_name,
        "target_scopes": ", ".join(plan.target_scopes),
        "rationale": plan.rationale,
        "amd_hints": "\n".join(f"  - {h}" for h in plan.amd_specific_hints) or "None specified",
        "ast_insights": "\n".join(ast_insights),
        "source_code": state["raw_kernel_code"],
        "critic_feedback": state.get("critic_feedback", "No prior critic feedback."),
    })

    annotation_count = len(rewritten.line_by_line_annotations)
    print(f"[kernel_rewriter] Rewrite complete. {annotation_count} annotated modification(s).")
    print(f"  Theoretical improvement: {rewritten.theoretical_improvement}")

    return {
        "optimized_kernel_code": rewritten.optimized_code,
        "annotations": rewritten.line_by_line_annotations,
        "theoretical_improvement": rewritten.theoretical_improvement,
    }