"""cross-references the performance bottleneck with the structural tokens identified by your tree-sitter node,
rewriting the .hip file with high-performance kernel adjustments."""


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from typing import Dict
from src.state import KernelAgentState, OptimizedKernelOutput

def kernel_rewriter_node(state: KernelAgentState) -> Dict:
    """
    Surgically refactors the unoptimized HIP/CUDA code using the structural 
    AST observations and the empirical bottleneck diagnosis.
    """
    # Initialize the code generation LLM. We bump temperature slightly to 0.2
    # to allow the model some flexibility in applying creative refactoring strategies
    # (e.g., choosing loop unrolling vs shared memory pre-fetching).
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    # Force the output to follow our strict OptimizedKernelOutput Pydantic layout.
    # This ensures we always get code paired cleanly with line-by-line annotations.
    structured_writer = llm.with_structured_output(OptimizedKernelOutput)
    
    # Build a prompt that forces the LLM to justify its edits based on hardware truths.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a world-class GPU compiler and kernel refactoring expert.\n"
            "Your job is to rewrite the user's unoptimized HIP kernel code to solve a "
            "proven hardware performance bottleneck.\n\n"
            "CRITICAL RULES:\n"
            "1. Only modify segments relevant to fixing the active bottleneck.\n"
            "2. For every code modification made, add an entry to 'line_by_line_annotations' "
            "justifying how that change addresses physical constraints (e.g., cache alignments, memory coalescing)."
        ),
        (
            "human",
            "--- TARGET BOTTLENECK DIAGNOSIS ---\n"
            "Type: {b_type}\n"
            "Evidence: {b_evidence}\n\n"
            "--- AST STRUCTURAL INSIGHTS ---\n"
            "{ast_insights}\n\n"
            "--- ORIGINAL SOURCE CODE ---\n"
            "{source_code}"
        )
    ])
    
    # Pack parameters from our multi-node state history into the prompt payload
    rewriter_chain = prompt | structured_writer
    rewritten_artifact = rewriter_chain.invoke({
        "b_type": state["diagnosis"].bottleneck_type,
        "b_evidence": state["diagnosis"].evidence,
        "ast_insights": "\n".join(state["ast_insights"]),
        "source_code": state["raw_kernel_code"]
    })
    
    # Return the optimized code block and annotations to the state dictionary
    return {
        "optimized_kernel_code": rewritten_artifact.optimized_code,
        "annotations": rewritten_artifact.line_by_line_annotations
    }