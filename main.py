import argparse
from pathlib import Path
from src.graph import deepwave_graph
from config.settings import settings


def load_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return p.read_text(encoding="utf-8")


def run(kernel_path: str, profiling_path: str, output_path: str = "report.md") -> None:
    print("\n" + "=" * 60)
    print("  DEEPWAVE — GPU Kernel Optimization Agent")
    print("=" * 60 + "\n")

    settings.require_api_key() 

    kernel_code    = load_file(kernel_path)
    profiling_data = load_file(profiling_path)

    initial_state = {
        "raw_kernel_code":      kernel_code,
        "raw_profiling_data":   profiling_data,
        "parsed_metrics":       {},
        "ast_insights":         [],
        "ast_findings":         [],
        "diagnosis":            None,
        "optimization_plan":    None,
        "optimized_kernel_code": None,
        "annotations":          None,
        "theoretical_improvement": None,
        "final_report":         None,
        "iteration_count":      0,
        "max_iterations":       3,
    }

    print(f"Kernel:   {kernel_path}")
    print(f"Profiling: {profiling_path}\n")

    final_state = deepwave_graph.invoke(initial_state)

    report = final_state.get("final_report", "No report generated.")

   
    Path(output_path).write_text(report, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"  Report saved to: {output_path}")
    print("=" * 60)
    print(report[:2000]) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEEPWAVE GPU Kernel Optimizer")
    parser.add_argument("--kernel",   required=True, help="Path to .hip or .cu kernel file")
    parser.add_argument("--profiling", required=True, help="Path to rocprof/omniperf CSV file")
    parser.add_argument("--output",   default="report.md", help="Output markdown report path")
    args = parser.parse_args()

    run(args.kernel, args.profiling, args.output)