#supports wide and long both now

import csv
import io
from typing import Dict
from src.state import KernelAgentState



METRIC_ALIASES: Dict[str, str] = {
    # VALU (Vector ALU) utilization
    "VALUUtilization":          "valu_util",
    "VALU_Utilization":         "valu_util",
    "SQ_VALU_UTIL":             "valu_util",
    "valu_util":                "valu_util",

    # SALU (Scalar ALU) utilization
    "SALUUtilization":          "salu_util",
    "SALU_Utilization":         "salu_util",
    "SQ_SALU_UTIL":             "salu_util",
    "salu_util":                "salu_util",

    # Memory stall / bandwidth pressure
    "MemUnitStalled":           "mem_stalled",
    "MemUnit_Stalled":          "mem_stalled",
    "MEM_UNIT_STALLED":         "mem_stalled",
    "TCP_READ_LATENCY_sum":     "mem_stalled",   
    "mem_stalled":              "mem_stalled",

    # Occupancy
    "MaxWavesPerCU":            "max_waves_per_cu",
    "Max_Waves_Per_CU":         "max_waves_per_cu",
    "SQ_WAVES":                 "max_waves_per_cu",
    "max_waves_per_cu":         "max_waves_per_cu",

    # L2 cache hit rate — useful for memory bottleneck confirmation
    "L2CacheHit":               "l2_cache_hit",
    "L2_Cache_Hit":             "l2_cache_hit",
    "TCP_TCC_READ_REQ_sum":     "l2_cache_hit",  
    "l2_cache_hit":             "l2_cache_hit",

    # Global memory bandwidth (GB/s) when available
    "FetchSize":                "fetch_size_kb",
    "WriteSize":                "write_size_kb",
    "fetch_size_kb":            "fetch_size_kb",
    "write_size_kb":            "write_size_kb",

    # LDS (shared memory) bank conflict indicator
    "LDSBankConflict":          "lds_bank_conflict",
    "LDS_Bank_Conflict":        "lds_bank_conflict",
    "lds_bank_conflict":        "lds_bank_conflict",

    # VGPR (vector register) usage — proxy for register pressure / spill risk
    "VGPRCount":                "vgpr_count",
    "ArchVGPRs":                "vgpr_count",          
    "vgpr_count":               "vgpr_count",
    "registers_per_thread":     "vgpr_count",           
    
    "sm__warps_active.avg.pct_of_peak_sustained_active": "max_waves_per_cu",   
    "achieved_occupancy":                                "max_waves_per_cu",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "mem_stalled",     
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed": "mem_stalled",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":  "valu_util",        
    "l2_cache_hit_rate":                                 "l2_cache_hit",
    "lts__t_sector_hit_rate.pct":                        "l2_cache_hit",     
    "shared_ld_bank_conflict":                            "lds_bank_conflict", 
    "l1tex__data_bank_conflicts_pipe_lsu.sum":            "lds_bank_conflict",
}


_ALIAS_LOOKUP: Dict[str, str] = {k.lower(): v for k, v in METRIC_ALIASES.items()}


_METRIC_DEFAULTS: Dict[str, float] = {
    "valu_util":        0.0,
    "salu_util":        0.0,
    "mem_stalled":      0.0,
    "max_waves_per_cu": 0.0,
    "l2_cache_hit":     0.0,
    "fetch_size_kb":    0.0,
    "write_size_kb":    0.0,
    "lds_bank_conflict":0.0,
    "vgpr_count":       0.0,
}


def parse_metrics_csv(raw_csv: str, label: str = "profiling_reader") -> Dict[str, float]:
   
   
    metrics: Dict[str, float] = dict(_METRIC_DEFAULTS) 

    clean_csv = (raw_csv or "").strip()
    if not clean_csv:
        print(f"[Warning] {label}: empty profiling data, returning defaults.")
        return metrics

    matched_any = False

    try:
        f = io.StringIO(clean_csv)
        reader = csv.DictReader(f)
        fieldnames = [c.strip().lower() for c in (reader.fieldnames or []) if c]

        is_long_format = {"metric", "value"}.issubset(set(fieldnames))

        for row in reader:
            clean_row = {
                (k.strip() if k else k): (v.strip() if v else v)
                for k, v in row.items()
                if k is not None and v is not None
            }

            if is_long_format:

                metric_name = clean_row.get("metric") or clean_row.get("Metric")
                raw_value = clean_row.get("value") or clean_row.get("Value")
                if metric_name is None or raw_value is None:
                    continue
                normalized_key = _ALIAS_LOOKUP.get(metric_name.strip().lower())
                if normalized_key is None:
                    continue
                try:
                    value = float(raw_value)
                    metrics[normalized_key] = max(metrics[normalized_key], value)
                    matched_any = True
                except ValueError:
                    pass
            else:

                for col_name, raw_value in clean_row.items():
                    if col_name is None:
                        continue
                    normalized_key = _ALIAS_LOOKUP.get(col_name.strip().lower())
                    if normalized_key is None:
                        continue
                    try:
                        value = float(raw_value)
                        metrics[normalized_key] = max(metrics[normalized_key], value)
                        matched_any = True
                    except (ValueError, TypeError):
                        pass

        if not matched_any:
            print(
                f"[Warning] {label}: no recognized metric columns/rows found "
                f"(saw columns: {fieldnames}). Falling back to all-zero metrics."
            )

    except Exception as e:
        print(f"[Warning] {label}: CSV parse failure — {e}")

    _log_parsed_metrics(metrics, label)
    return metrics


def profiling_reader_node(state: KernelAgentState) -> Dict:
    
    metrics = parse_metrics_csv(state["raw_profiling_data"], label="profiling_reader_node")

    return {
        "parsed_metrics": metrics,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def _log_parsed_metrics(metrics: Dict[str, float], label: str = "profiling_reader") -> None:
    print(f"[{label}] Extracted metrics:")
    for key, value in metrics.items():
        if value > 0.0:
            print(f"  {key:25s} = {value:.4f}")