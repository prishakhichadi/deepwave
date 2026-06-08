"""parses raw C++ HIP code files (.hip or .cu) into a structured format using tree-sitter-cpp rather than
treating the code like raw text. builds an AST so that agent can map a hardware flaw directly to specific loop
scopes or global pointer arguments."""

from tree_sitter import Language, Parser
import tree_sitter_cpp
from typing import Dict
from src.state import KernelAgentState

def kernel_analyzer_node(state: KernelAgentState) -> Dict:
    """
    Parses the target HIP kernel using tree-sitter to map out code structural tokens,
    allowing downstream LLM agents to accurately pinpoint instruction segments.
    """
    # Compile and bind standard tree-sitter C++ grammar rules
    CPP_LANGUAGE = Language(tree_sitter_cpp.language())
    parser = Parser(CPP_LANGUAGE)
    
    code_bytes = bytes(state["raw_kernel_code"], "utf8")
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    
    insights = []
    
    # Query to look for __global__ kernel hooks and pointer declarations
    query_string = """
        (primitive_type) @type
        (pointer_declarator) @global_ptr
    """
    
    try:
        query = CPP_LANGUAGE.query(query_string)
        captures = query.captures(root_node)
        
        if captures:
            insights.append(f"AST Analysis: Found {len(captures)} distinct base type/pointer handles.")
        else:
            insights.append("AST Analysis: No plain structural base pointers identified.")
            
    except Exception as e:
        insights.append(f"AST Analysis Warning: Query parse failure: {e}")
        
    return {"ast_insights": insights}