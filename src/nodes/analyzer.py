from tree_sitter import Language, Parser, Node, Query, QueryCursor
import tree_sitter_cpp
import re
from typing import Dict, List, Tuple
from src.state import KernelAgentState, ASTFinding


def _run_query(lang: Language, query_str: str, root: Node) -> List[Tuple[Node, str]]:


    query = Query(lang, query_str)
    cursor = QueryCursor(query)
    captures_by_name = cursor.captures(root)  # {capture_name: [nodes]}
    flat: List[Tuple[Node, str]] = []
    for name, nodes in captures_by_name.items():
        for node in nodes:
            flat.append((node, name))
    return flat


def kernel_analyzer_node(state: KernelAgentState) -> Dict:

    CPP_LANGUAGE = Language(tree_sitter_cpp.language())
    parser = Parser(CPP_LANGUAGE)

    code = state["raw_kernel_code"]
    code_bytes = bytes(code, "utf8")
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    code_lines = code.splitlines()

    findings: List[ASTFinding] = []

    findings += _detect_thread_divergence(root_node, code_lines, CPP_LANGUAGE)
    findings += _detect_uncoalesced_access(root_node, code_lines, CPP_LANGUAGE)
    findings += _detect_missing_shared_memory(root_node, code_lines, CPP_LANGUAGE)
    findings += _detect_scalar_ops_in_kernel(root_node, code_lines, CPP_LANGUAGE)
    findings += _detect_pointer_aliasing(root_node, code_lines, CPP_LANGUAGE)
    findings += _detect_loop_structures(root_node, code_lines, CPP_LANGUAGE)

    insights = _findings_to_insights(findings, root_node, CPP_LANGUAGE)

    print(f"[kernel_analyzer] {len(findings)} structural findings across {len(insights)} insight lines.")
    for f in findings:
        print(f"  [{f.severity.upper()}] {f.finding_type} @ {f.location}: {f.description}")

    return {
        "ast_findings": findings,
        "ast_insights": insights,
    }



def _detect_thread_divergence(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []
    captures = _run_query(lang, """
        (if_statement
            condition: (condition_clause
                (binary_expression
                    left: (field_expression
                        field: (field_identifier) @field_name)))) @if_stmt
    """, root)

    seen_scopes = set()
    for node, tag in captures:
        if tag == "field_name":
            field_text = node.text.decode("utf8") if node.text else ""
            if field_text in ("x", "y", "z") and "threadIdx" in _get_parent_text(node, lines):
                scope = _get_enclosing_function(node, lines)
                if scope not in seen_scopes:
                    seen_scopes.add(scope)
                    findings.append(ASTFinding(
                        finding_type="thread_divergence",
                        location=scope,
                        description=(
                            f"Conditional branch on threadIdx.{field_text} detected. "
                            "On AMD GCN/RDNA, threads in the same wavefront (64 lanes) that take "
                            "different branches are serialized — both paths execute with masking. "
                            "Consider restructuring to eliminate intra-wavefront divergence."
                        ),
                        severity="high"
                    ))
    return findings



def _get_thread_derived_vars(root: Node, lang: Language) -> set:
    """
    Finds simple local variables whose initializer references threadIdx/blockIdx,
    e.g. `int i = blockIdx.x * blockDim.x + threadIdx.x;`. Real kernels almost never
    subscript arrays with the raw `threadIdx.x` expression inline — they compute an
    index variable once and reuse it, so tracking that one hop of indirection is
    necessary to catch real strided-access patterns instead of only literal ones.
    """
    thread_vars = set()

    query = Query(lang, """
        (init_declarator
            declarator: (identifier) @varname
            value: (_) @initval)
    """)
    cursor = QueryCursor(query)
    matches = cursor.matches(root)
    for _, captures in matches:
        varname_nodes = captures.get("varname", [])
        initval_nodes = captures.get("initval", [])
        if not varname_nodes or not initval_nodes:
            continue
        varname = varname_nodes[0].text.decode("utf8") if varname_nodes[0].text else ""
        initval_text = initval_nodes[0].text.decode("utf8") if initval_nodes[0].text else ""
        if "threadIdx" in initval_text or "blockIdx" in initval_text:
            thread_vars.add(varname)
    return thread_vars


def _is_thread_referencing(index_text: str, thread_vars: set) -> bool:
    if "threadIdx" in index_text or "blockIdx" in index_text:
        return True
    tokens = re.findall(r"[A-Za-z_]\w*", index_text)
    return any(t in thread_vars for t in tokens)


def _detect_uncoalesced_access(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []
    captures_dict: dict = {}

    for node, tag in _run_query(lang, """
        (subscript_expression
            (subscript_argument_list) @index_expr) @array_access
    """, root):
        captures_dict.setdefault(tag, []).append(node)

    index_nodes = captures_dict.get("index_expr", [])
    thread_vars = _get_thread_derived_vars(root, lang)
    strided_accesses = 0

    for node in index_nodes:
        index_text = node.text.decode("utf8") if node.text else ""

        if _is_thread_referencing(index_text, thread_vars):
            if any(op in index_text for op in ["* ", " *", "/ ", " /"]):
                if not _is_simple_linear(index_text, thread_vars):
                    strided_accesses += 1

    if strided_accesses > 0:
        findings.append(ASTFinding(
            finding_type="uncoalesced_memory_access",
            location="global memory access pattern",
            description=(
                f"Detected {strided_accesses} potentially strided/non-linear global memory "
                "access pattern(s). AMD MI300X achieves peak bandwidth only when consecutive "
                "threads access consecutive memory addresses (128-byte cache line coalescing). "
                "Strided accesses waste bandwidth — consider AoS→SoA layout transformation "
                "or transposing the access pattern."
            ),
            severity="high"
        ))
    return findings



# Detection Pass 3 — Missing Shared Memory (LDS)

def _detect_missing_shared_memory(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []


    full_source = "\n".join(lines)
    has_shared = "__shared__" in full_source or "__local" in full_source

    if has_shared:
        return findings  


    loops = [n for n, _ in _run_query(lang, "(for_statement) @for_loop", root)]
    ptrs = [n for n, _ in _run_query(lang, "(pointer_declarator) @ptr", root)]

    if loops and ptrs:
        findings.append(ASTFinding(
            finding_type="missing_shared_memory",
            location="kernel loop body",
            description=(
                f"Kernel contains {len(loops)} loop(s) accessing {len(ptrs)} global pointer(s) "
                "with no __shared__ (LDS) memory declarations. AMD MI300X has 64KB LDS per CU — "
                "caching repeatedly accessed data in LDS can reduce global memory traffic by "
                "orders of magnitude for stencil, reduction, and tiled matrix patterns."
            ),
            severity="high"
        ))
    return findings



# Detection Pass 4 — Scalar Operations Inside Kernel

def _detect_scalar_ops_in_kernel(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []
    full_source = "\n".join(lines)


    scalar_indicators = ["% blockDim", "/ blockDim", "% gridDim", "/ gridDim"]
    hits = [s for s in scalar_indicators if s in full_source]

    if hits:
        findings.append(ASTFinding(
            finding_type="scalar_operation_in_kernel",
            location="kernel arithmetic",
            description=(
                f"Detected potential scalar-path arithmetic: {hits}. "
                "Integer division and modulo by non-power-of-2 values typically emit expensive "
                "SALU sequences or reciprocal multiply on AMD. Prefer bit-masking (& (N-1)) "
                "when the divisor is a power of 2, or precompute indices in registers."
            ),
            severity="medium"
        ))
    return findings



# Detection Pass 5 — Pointer Aliasing Risk

def _detect_pointer_aliasing(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []
    full_source = "\n".join(lines)

    ptrs = _run_query(lang, "(pointer_declarator) @ptr", root)
    ptr_count = len(ptrs)

    has_restrict = "__restrict__" in full_source or "__restrict" in full_source

    if ptr_count >= 2 and not has_restrict:
        findings.append(ASTFinding(
            finding_type="pointer_aliasing_risk",
            location="kernel function signature",
            description=(
                f"Kernel has {ptr_count} pointer parameter(s) with no __restrict__ qualifiers. "
                "Without __restrict__, the compiler must assume any two pointers may alias, "
                "preventing vectorization of loads and stores. Adding __restrict__ to all "
                "non-aliasing pointer parameters enables the compiler to emit wider vector "
                "memory instructions on AMD GCN/CDNA."
            ),
            severity="medium"
        ))
    return findings


# Detection Pass 6 — Loop Structure Analysis

def _detect_loop_structures(root: Node, lines: List[str], lang: Language) -> List[ASTFinding]:
    findings = []
    loops = [n for n, _ in _run_query(lang, "(for_statement) @loop", root)]

    if not loops:
        return findings

    max_depth = _compute_max_loop_depth(root)

    findings.append(ASTFinding(
        finding_type="loop_structure",
        location="kernel body",
        description=(
            f"Found {len(loops)} for-loop(s), maximum nesting depth: {max_depth}. "
            + (
                "Deeply nested loops (depth ≥ 3) limit the compiler's ability to unroll and "
                "pipeline. Consider flattening inner loops or using #pragma unroll on the "
                "innermost loop to expose instruction-level parallelism to the AMD compiler."
                if max_depth >= 2 else
                "Loop structure looks amenable to unrolling — consider adding #pragma unroll "
                "with an explicit count to guide the AMD compiler."
            )
        ),
        severity="medium" if max_depth >= 2 else "low"
    ))
    return findings


# Helpers


def _findings_to_insights(findings: List[ASTFinding], root: Node, lang: Language) -> List[str]:
    if not findings:
        return ["AST Analysis: No significant GPU anti-patterns detected in kernel structure."]

    lines = [f"AST Analysis: {len(findings)} structural finding(s) detected:"]
    for i, f in enumerate(findings, 1):
        lines.append(
            f"  [{i}] [{f.severity.upper()}] {f.finding_type} @ {f.location} — {f.description}"
        )
    return lines


def _get_enclosing_function(node: Node, lines: List[str]) -> str:

    current = node.parent
    while current is not None:
        if current.type in ("function_definition", "function_declarator"):
            # Try to get the function name from its declarator child
            for child in current.children:
                if child.type == "function_declarator":
                    for subchild in child.children:
                        if subchild.type in ("identifier", "qualified_identifier"):
                            return subchild.text.decode("utf8") if subchild.text else "unknown_function"
            return "unknown_function"
        current = current.parent
    return "global_scope"


def _get_parent_text(node: Node, lines: List[str]) -> str:

    if node.parent and node.parent.parent:
        try:
            start = node.parent.parent.start_point[0]
            end = node.parent.parent.end_point[0]
            return "\n".join(lines[start:end + 1])
        except Exception:
            pass
    return ""


def _is_simple_linear(index_text: str, thread_vars: set = frozenset()) -> bool:

    # Strip known linear components
    linear_tokens = ["threadIdx.x", "threadIdx.y", "blockIdx.x", "blockIdx.y",
                     "blockDim.x", "blockDim.y", "+", "-", " "]
    remaining = index_text
    for token in linear_tokens:
        remaining = remaining.replace(token, "")
    # A traced thread-derived variable used on its own (e.g. `i + 1`) is still linear —
    # only flag it once it's multiplied/divided by something else, which the caller
    # already checks for before calling this helper.
    for var in thread_vars:
        remaining = re.sub(rf"\b{re.escape(var)}\b", "", remaining)
    remaining = remaining.strip()
    # If only digits remain, it's a simple linear expression
    return remaining.isdigit() or remaining == ""


def _compute_max_loop_depth(root: Node, current_depth: int = 0) -> int:
    """Recursively computes the maximum for-loop nesting depth in the AST."""
    max_depth = current_depth
    for child in root.children:
        child_depth = current_depth + (1 if child.type == "for_statement" else 0)
        max_depth = max(max_depth, _compute_max_loop_depth(child, child_depth))
    return max_depth