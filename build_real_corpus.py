import json
import re
import textwrap
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_cpp

REPO = "ROCm/rocm-examples"
COMMIT = "41dd7463e65e230af913db75d48a1d6c0dcff6bc"  
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
LICENSE = "MIT"
LICENSE_URL = f"https://github.com/{REPO}/blob/{COMMIT}/LICENSE.md"

OUTPUT_DIR = Path("corpus/real")


@dataclass
class SourceKernel:
    name: str                   
    repo_path: str                
    kernel_functions: List[str]   
    note: str                   


SOURCES: List[SourceKernel] = [
    SourceKernel(
        name="saxpy",
        repo_path="HIP-Basic/saxpy/main.hip",
        kernel_functions=["saxpy_kernel"],
        note="Simple elementwise AXPY — a clean bandwidth-bound baseline case.",
    ),
    SourceKernel(
        name="matrix_multiplication_naive",
        repo_path="HIP-Basic/matrix_multiplication/main.hip",
        kernel_functions=["matrix_multiplication_kernel"],
        note="Textbook naive (non-tiled) matmul — no shared memory reuse.",
    ),
    SourceKernel(
        name="shared_memory_transpose",
        repo_path="HIP-Basic/shared_memory/main.hip",
        kernel_functions=["matrix_transpose_kernel"],
        note="Already uses __shared__ tiling — a useful 'well-optimized' contrast example.",
    ),
    SourceKernel(
        name="histogram",
        repo_path="Applications/histogram/main.hip",
        kernel_functions=None,   
        note="Atomic-heavy histogram accumulation — scalar/contention-bound candidate.",
    ),
    SourceKernel(
        name="reduction_naive",
        repo_path="Tutorials/reduction/example/v1.hip",
        kernel_functions=None,
        note=(
            "AMD's own documented reduction tutorial, step 1 of 10. Uses `tid % (2*i) == 0` "
            "— a textbook thread-divergent branch — plus strided shared-memory access. "
            "Real, official, deliberately-naive starting point."
        ),
    ),
    SourceKernel(
        name="matmul_naive_tiling_series",
        repo_path="Programming-Guide/Tutorials/Performance-Optimization/tiling_matrix_multiply/tiling_matrix_multiply.hip",
        kernel_functions=["matrix_multiply_naive", "matrix_multiply_lds_tiling", "matrix_multiply_register_tiling"],
        note=(
            "Three kernels in one file: naive, LDS-tiled, and register-tiled matmul. "
            "A real, official before/after/after-again optimization progression."
        ),
    ),
]


def fetch(repo_path: str) -> str:
    url = f"{RAW_BASE}/{repo_path}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        return resp.read().decode("utf-8")


def extract_kernel_functions(source: str, wanted_names: Optional[List[str]]) -> List[tuple]:
    
    
    CPP_LANGUAGE = Language(tree_sitter_cpp.language())
    parser = Parser(CPP_LANGUAGE)
    tree = parser.parse(bytes(source, "utf8"))

    query = Query(CPP_LANGUAGE, "(function_definition) @fn")
    cursor = QueryCursor(query)
    results = []

    captures = cursor.captures(tree.root_node)
    for fn_node in captures.get("fn", []):
       
       
        emit_node = fn_node
        if fn_node.parent is not None and fn_node.parent.type == "template_declaration":
            emit_node = fn_node.parent

        fn_text = fn_node.text.decode("utf8") if fn_node.text else ""
        if "__global__" not in fn_text[:fn_text.find("(") if "(" in fn_text else 40]:
          
          
            if "__global__" not in fn_text.split("{")[0]:
                continue
        name_match = re.search(r"__global__\s+(?:void|[\w:<>,\s]+?)\s+(\w+)\s*\(", fn_text)
        if not name_match:
            continue
        fn_name = name_match.group(1)
        if wanted_names is not None and fn_name not in wanted_names:
            continue
        emit_text = emit_node.text.decode("utf8") if emit_node.text else fn_text
        results.append((fn_name, emit_text))

    return results


def build_kernel_file(fn_name: str, fn_source: str, src: SourceKernel) -> str:
    header = textwrap.dedent(f"""\
        // Sourced from {REPO} @ {COMMIT[:10]}
        // Path:    {src.repo_path}
        // License: {LICENSE} — Copyright (c) Advanced Micro Devices, Inc.
        // Fetched: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
        // Note:    {src.note}
        // Profiling status: NOT YET PROFILED — see corpus/real/README.md
        #include <hip/hip_runtime.h>

        """)
    return header + fn_source.rstrip() + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []

    for src in SOURCES:
        print(f"[{src.name}] fetching {src.repo_path} ...")
        try:
            source = fetch(src.repo_path)
        except Exception as e:
            print(f"  SKIPPED — fetch failed: {e}")
            continue

        extracted = extract_kernel_functions(source, src.kernel_functions)
        if not extracted:
            print(f"  WARNING — no matching __global__ functions found, skipping.")
            continue

        for fn_name, fn_source in extracted:
            
            out_name = fn_name if len(extracted) > 1 else src.name
            out_dir = OUTPUT_DIR / out_name
            out_dir.mkdir(parents=True, exist_ok=True)

            kernel_text = build_kernel_file(fn_name, fn_source, src)
            (out_dir / "kernel.hip").write_text(kernel_text, encoding="utf-8")

            manifest.append({
                "name": out_name,
                "kernel_function": fn_name,
                "source_repo": REPO,
                "source_commit": COMMIT,
                "source_path": src.repo_path,
                "source_url": f"https://github.com/{REPO}/blob/{COMMIT}/{src.repo_path}",
                "license": LICENSE,
                "license_url": LICENSE_URL,
                "fetched": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                "note": src.note,
                "profiling_status": "not_yet_profiled",
                "profiling_csv": None,
            })
            print(f"  -> corpus/real/{out_name}/kernel.hip")

    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone. {len(manifest)} real kernel(s) written to {OUTPUT_DIR}/")
    print("None of these have profiling data yet — see corpus/real/README.md for how to get it.")


if __name__ == "__main__":
    main()