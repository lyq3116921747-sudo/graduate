from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.ablation import build_ablation_config
from Agent.agent import TASK_MODE_TREATMENT, SepsisMedGemmaAgent
from Agent.config import AgentConfig
from Agent.llm_presets import (
    BASE_TEXT_PRESET_NAMES,
    MEDICAL_TREATMENT_PRESET_NAMES,
    list_text_backend_presets,
    normalize_allowed_text_backend_preset_name,
    normalize_text_backend_preset_name,
    resolve_text_backend_preset,
)
from Agent.utils import clean_text


DEFAULT_CURETEST_DIR = PROJECT_ROOT / "data" / "CureTest"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "miniTest" / "curetest_treatment_matrix"
DEFAULT_RESULTS_CSV = "results.csv"
DEFAULT_RESULTS_JSON = "results.json"
DEFAULT_SUMMARY_CSV = "summary.csv"
DEFAULT_SUMMARY_JSON = "summary.json"
DEFAULT_USER_QUERY = "请基于病例给出脓毒症治疗建议。"
DEFAULT_MODELS = "qwen_plus,deepseek,qwen_max"
DEFAULT_MEDICAL_MODELS = "baichuan,antangel"
CSV_FILENAME = "sample_patients.csv"
ARCHITECTURE_FULL_STACK_BASE = "full_stack_base"
ARCHITECTURE_MEDICAL_TREATMENT_GENERATOR = "medical_treatment_generator"
UNIFIED_TEXT_BACKEND_FIELDS = (
    "default_text_backend",
    "reflection_backend",
    "code_writer_backend",
    "diagnosis_backend",
    "treatment_backend",
)
TREATMENT_TOOL_NAMES = (
    "get_case_digest",
    "read_case_sections",
    "search_case_text",
    "run_sofa_assessment",
    "search_local_guidelines",
    "run_generated_probe",
)
TOOL_ABLATION_NONE = "none"
TOOL_ABLATION_GROUPED = "grouped"
TOOL_ABLATION_GROUPED_VARIANTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("tool_full", ()),
    ("tool_no_case_access", ("get_case_digest", "read_case_sections", "search_case_text")),
    ("tool_no_sofa", ("run_sofa_assessment",)),
    ("tool_no_guidelines", ("search_local_guidelines",)),
)
TOOL_CALL_RESULT_FIELDS = {
    tool_name: f"tool_calls_{tool_name}"
    for tool_name in TREATMENT_TOOL_NAMES
}
TOOL_CALL_SUMMARY_FIELDS = {
    tool_name: f"avg_calls_{tool_name}"
    for tool_name in TREATMENT_TOOL_NAMES
}
RESULT_FIELDS = [
    "model_preset",
    "profile_name",
    "architecture",
    "base_text_preset",
    "medical_treatment_preset",
    "treatment_generation_mode",
    "input_type",
    "case_identifier",
    "source_path",
    "row_index",
    "status",
    "answer",
    "error",
    "treatment_react_status",
    "treatment_react_rounds",
    "treatment_tool_call_count",
    "elapsed_seconds",
    "round_limit_failed",
    "tool_call_counts",
    *[TOOL_CALL_RESULT_FIELDS[tool_name] for tool_name in TREATMENT_TOOL_NAMES],
    "warnings",
    "model_routes",
]
SUMMARY_FIELDS = [
    "group",
    "model_preset",
    "architecture",
    "base_text_preset",
    "medical_treatment_preset",
    "treatment_generation_mode",
    "input_type",
    "total",
    "success_count",
    "success_rate",
    "incomplete_count",
    "error_count",
    "round_limit_failure_count",
    "round_limit_failure_rate",
    "avg_treatment_react_rounds",
    "avg_treatment_tool_call_count",
    "avg_elapsed_seconds",
    *[TOOL_CALL_SUMMARY_FIELDS[tool_name] for tool_name in TREATMENT_TOOL_NAMES],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 CureTest 上批量运行 treatment 模式多模型 Agent。")
    parser.add_argument("--curetest-dir", default=str(DEFAULT_CURETEST_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--architecture",
        choices=[ARCHITECTURE_FULL_STACK_BASE, ARCHITECTURE_MEDICAL_TREATMENT_GENERATOR],
        default=ARCHITECTURE_FULL_STACK_BASE,
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--medical-models", default=DEFAULT_MEDICAL_MODELS)
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--enable-generated-code", action="store_true")
    parser.add_argument("--enable-treatment-round-reminders", action="store_true")
    parser.add_argument(
        "--tool-ablation",
        choices=[TOOL_ABLATION_NONE, TOOL_ABLATION_GROUPED],
        default=TOOL_ABLATION_NONE,
    )
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def parse_model_presets(raw_models: str, *, allowed_presets: Optional[Sequence[str]] = None) -> List[str]:
    presets: List[str] = []
    for raw_item in raw_models.split(","):
        item = clean_text(raw_item)
        if not item:
            continue
        if allowed_presets is None:
            normalized = normalize_text_backend_preset_name(item)
        else:
            normalized = normalize_allowed_text_backend_preset_name(item, allowed_presets=allowed_presets)
        if normalized not in presets:
            presets.append(normalized)
    if not presets:
        available = ", ".join(sorted(allowed_presets or list_text_backend_presets()))
        raise ValueError(f"至少需要一个有效模型 preset；可选值: {available}")
    return presets


def build_treatment_agent_config_for_preset(
    *,
    model_preset: str,
    mode: str,
    max_new_tokens: Optional[int],
    enable_generated_code: bool = False,
    enable_treatment_round_reminders: bool = False,
) -> AgentConfig:
    normalized_model_name = normalize_allowed_text_backend_preset_name(model_preset, allowed_presets=BASE_TEXT_PRESET_NAMES)
    backend_config = resolve_text_backend_preset(normalized_model_name)
    config = build_ablation_config("with_code_probe" if enable_generated_code else "default", AgentConfig())
    resolved_generation_max_new_tokens = max_new_tokens
    if resolved_generation_max_new_tokens is None:
        resolved_generation_max_new_tokens = backend_config.default_max_new_tokens or config.generation_max_new_tokens

    replacement_fields = {field_name: backend_config for field_name in UNIFIED_TEXT_BACKEND_FIELDS}
    profile_prefix = (
        "curetest_treatment_fullstack_with_code_probe"
        if enable_generated_code
        else "curetest_treatment_fullstack"
    )
    config = replace(
        config,
        profile_name=f"{profile_prefix}/{normalized_model_name}",
        case_input_mode=mode,
        generation_max_new_tokens=resolved_generation_max_new_tokens,
        enable_treatment_round_reminders=enable_treatment_round_reminders,
        http_timeout_seconds=max(config.http_timeout_seconds, float(backend_config.default_timeout_seconds or 0.0)),
        sepsis3_extraction_max_new_tokens=max(
            config.sepsis3_extraction_max_new_tokens,
            int(backend_config.sepsis3_extraction_max_new_tokens or 0),
        ),
        sepsis3_extraction_timeout_seconds=max(
            config.sepsis3_extraction_timeout_seconds,
            float(backend_config.sepsis3_extraction_timeout_seconds or 0.0),
        ),
        **replacement_fields,
    )
    return config


def build_medical_treatment_generator_config(
    *,
    base_model_preset: str,
    medical_treatment_preset: str,
    mode: str,
    max_new_tokens: Optional[int],
    enable_generated_code: bool = False,
    enable_treatment_round_reminders: bool = False,
) -> AgentConfig:
    normalized_base_name = normalize_allowed_text_backend_preset_name(
        base_model_preset,
        allowed_presets=BASE_TEXT_PRESET_NAMES,
    )
    normalized_medical_name = normalize_allowed_text_backend_preset_name(
        medical_treatment_preset,
        allowed_presets=MEDICAL_TREATMENT_PRESET_NAMES,
    )
    base_backend = resolve_text_backend_preset(normalized_base_name)
    medical_backend = resolve_text_backend_preset(normalized_medical_name)
    config = build_ablation_config("with_code_probe" if enable_generated_code else "default", AgentConfig())
    resolved_generation_max_new_tokens = max_new_tokens
    if resolved_generation_max_new_tokens is None:
        resolved_generation_max_new_tokens = base_backend.default_max_new_tokens or config.generation_max_new_tokens

    replacement_fields = {field_name: base_backend for field_name in UNIFIED_TEXT_BACKEND_FIELDS}
    profile_prefix = (
        "curetest_treatment_medical_generator_with_code_probe"
        if enable_generated_code
        else "curetest_treatment_medical_generator"
    )
    config = replace(
        config,
        profile_name=f"{profile_prefix}/{normalized_base_name}__{normalized_medical_name}",
        case_input_mode=mode,
        generation_max_new_tokens=resolved_generation_max_new_tokens,
        enable_treatment_round_reminders=enable_treatment_round_reminders,
        http_timeout_seconds=max(config.http_timeout_seconds, float(base_backend.default_timeout_seconds or 0.0)),
        sepsis3_extraction_max_new_tokens=max(
            config.sepsis3_extraction_max_new_tokens,
            int(base_backend.sepsis3_extraction_max_new_tokens or 0),
        ),
        sepsis3_extraction_timeout_seconds=max(
            config.sepsis3_extraction_timeout_seconds,
            float(base_backend.sepsis3_extraction_timeout_seconds or 0.0),
        ),
        medical_treatment_backend=medical_backend,
        **replacement_fields,
    )
    return config


def _read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def discover_curetest_json_samples(curetest_dir: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for json_path in sorted(curetest_dir.expanduser().resolve().glob("zy@*.json")):
        samples.append(
            {
                "input_type": "json",
                "case_identifier": f"json::{json_path.name}",
                "source_path": str(json_path),
                "row_index": "",
                "payload": None,
            }
        )
    return samples


def discover_curetest_csv_samples(curetest_dir: Path) -> List[Dict[str, Any]]:
    csv_path = curetest_dir.expanduser().resolve() / CSV_FILENAME
    rows = _read_csv_rows(csv_path)
    samples: List[Dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        row_case_id = clean_text(row.get("case_id", "")) or f"row_{row_index}"
        samples.append(
            {
                "input_type": "csv",
                "case_identifier": f"csv::{row_case_id}",
                "source_path": str(csv_path),
                "row_index": str(row_index),
                "payload": dict(row),
            }
        )
    return samples


def discover_curetest_samples(curetest_dir: Path) -> List[Dict[str, Any]]:
    resolved_dir = curetest_dir.expanduser().resolve()
    samples = discover_curetest_json_samples(resolved_dir) + discover_curetest_csv_samples(resolved_dir)
    return samples


def _read_existing_result_keys(csv_path: Path) -> Set[Tuple[str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = list(RESULT_FIELDS)
        actual_header = list(reader.fieldnames or [])
        if actual_header != expected_header:
            raise ValueError(
                "Existing results.csv schema does not match the current RESULT_FIELDS; "
                "please rerun with --no-resume to rebuild the results."
            )
        return {
            (clean_text(row.get("profile_name", "")), clean_text(row.get("case_identifier", "")))
            for row in reader
            if clean_text(row.get("profile_name", "")) and clean_text(row.get("case_identifier", ""))
        }


def _append_result_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def _load_result_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_json_cell(raw_value: Any, default: Any) -> Any:
    text = clean_text(raw_value)
    if not text:
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def _normalize_int(value: Any) -> int:
    text = clean_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except Exception:
        return 0


def _normalize_float(value: Any) -> float:
    text = clean_text(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    return text in {"1", "true", "yes", "y"}


def _empty_tool_call_counts() -> Dict[str, int]:
    return {tool_name: 0 for tool_name in TREATMENT_TOOL_NAMES}


def _extract_tool_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    name = clean_text(payload.get("name", ""))
    if name:
        return name
    name = clean_text(payload.get("tool_name", ""))
    if name:
        return name
    function_payload = payload.get("function", {})
    if isinstance(function_payload, dict):
        return clean_text(function_payload.get("name", ""))
    return ""


def _collect_tool_call_counts(result: Dict[str, Any]) -> Dict[str, int]:
    counts = _empty_tool_call_counts()
    treatment = result.get("treatment", {}) if isinstance(result.get("treatment", {}), dict) else {}
    tool_calls = treatment.get("tool_calls", []) if isinstance(treatment.get("tool_calls", []), list) else []

    for item in tool_calls:
        tool_name = _extract_tool_name(item)
        if tool_name in counts:
            counts[tool_name] += 1

    if any(counts.values()):
        return counts

    meta = result.get("meta", {}) if isinstance(result.get("meta", {}), dict) else {}
    react_trace = meta.get("react_trace", []) if isinstance(meta.get("react_trace", []), list) else []
    for trace_entry in react_trace:
        if not isinstance(trace_entry, dict):
            continue
        tool_outputs = trace_entry.get("tool_outputs", []) if isinstance(trace_entry.get("tool_outputs", []), list) else []
        for output in tool_outputs:
            tool_name = clean_text(output.get("tool_name", "")) if isinstance(output, dict) else ""
            if tool_name in counts:
                counts[tool_name] += 1
    return counts


def _is_round_limit_failure(*, treatment_status: str, treatment_error: str) -> bool:
    marker = "treatment_react_round_limit_reached"
    combined = "\n".join([clean_text(treatment_status), clean_text(treatment_error)])
    return marker in combined


def build_result_row(
    *,
    model_preset: str,
    profile_name: str,
    architecture: str = "",
    base_text_preset: str = "",
    medical_treatment_preset: str = "",
    treatment_generation_mode: str = "",
    sample: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
    elapsed_seconds: float = 0.0,
) -> Dict[str, Any]:
    resolved_error = clean_text(error)
    if resolved_error:
        row = {
            "model_preset": clean_text(model_preset),
            "profile_name": clean_text(profile_name),
            "architecture": clean_text(architecture),
            "base_text_preset": clean_text(base_text_preset),
            "medical_treatment_preset": clean_text(medical_treatment_preset),
            "treatment_generation_mode": clean_text(treatment_generation_mode) or "fullstack_base",
            "input_type": clean_text(sample.get("input_type", "")),
            "case_identifier": clean_text(sample.get("case_identifier", "")),
            "source_path": clean_text(sample.get("source_path", "")),
            "row_index": clean_text(sample.get("row_index", "")),
            "status": "error",
            "answer": "",
            "error": resolved_error,
            "treatment_react_status": "",
            "treatment_react_rounds": 0,
            "treatment_tool_call_count": 0,
            "elapsed_seconds": max(0.0, float(elapsed_seconds)),
            "round_limit_failed": False,
            "tool_call_counts": "{}",
            "warnings": "[]",
            "model_routes": "{}",
        }
        for tool_name, field_name in TOOL_CALL_RESULT_FIELDS.items():
            del tool_name
            row[field_name] = 0
        return row

    payload = result if isinstance(result, dict) else {}
    meta = payload.get("meta", {}) if isinstance(payload.get("meta", {}), dict) else {}
    treatment = payload.get("treatment", {}) if isinstance(payload.get("treatment", {}), dict) else {}
    final_status = clean_text(treatment.get("status", "")) or "success"
    treatment_status = clean_text(meta.get("treatment_react_status", "")) or clean_text(treatment.get("react_status", "")) or final_status
    treatment_error = (
        clean_text(meta.get("treatment_react_error", ""))
        or clean_text(treatment.get("react_error", ""))
        or clean_text(treatment.get("error", ""))
    )
    final_error = clean_text(treatment.get("error", ""))
    warnings = meta.get("warnings", []) if isinstance(meta.get("warnings", []), list) else []
    model_routes = meta.get("model_routes", {}) if isinstance(meta.get("model_routes", {}), dict) else {}
    tool_call_counts = _collect_tool_call_counts(payload)
    round_limit_failed = _is_round_limit_failure(
        treatment_status=treatment_status,
        treatment_error=treatment_error,
    )

    row = {
        "model_preset": clean_text(model_preset),
        "profile_name": clean_text(meta.get("profile_name", "")) or clean_text(profile_name),
        "architecture": clean_text(architecture),
        "base_text_preset": clean_text(meta.get("base_text_preset", "")) or clean_text(base_text_preset),
        "medical_treatment_preset": clean_text(meta.get("medical_treatment_preset", "")) or clean_text(medical_treatment_preset),
        "treatment_generation_mode": clean_text(meta.get("treatment_generation_mode", "")) or clean_text(treatment.get("treatment_generation_mode", "")) or clean_text(treatment_generation_mode) or "fullstack_base",
        "input_type": clean_text(sample.get("input_type", "")),
        "case_identifier": clean_text(sample.get("case_identifier", "")),
        "source_path": clean_text(sample.get("source_path", "")),
        "row_index": clean_text(sample.get("row_index", "")),
        "status": final_status,
        "answer": clean_text(payload.get("answer", "")),
        "error": final_error,
        "treatment_react_status": treatment_status,
        "treatment_react_rounds": _normalize_int(meta.get("treatment_react_rounds", 0)),
        "treatment_tool_call_count": _normalize_int(meta.get("treatment_tool_call_count", 0)),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "round_limit_failed": round_limit_failed,
        "tool_call_counts": json.dumps(tool_call_counts, ensure_ascii=False, sort_keys=True),
        "warnings": json.dumps(warnings, ensure_ascii=False),
        "model_routes": json.dumps(model_routes, ensure_ascii=False, sort_keys=True),
    }
    for tool_name, field_name in TOOL_CALL_RESULT_FIELDS.items():
        row[field_name] = tool_call_counts.get(tool_name, 0)
    return row


def write_results_json(results_csv_path: Path, results_json_path: Path) -> List[Dict[str, Any]]:
    csv_rows = _load_result_rows(results_csv_path)
    parsed_rows: List[Dict[str, Any]] = []
    for row in csv_rows:
        parsed = dict(row)
        parsed["treatment_react_rounds"] = _normalize_int(parsed.get("treatment_react_rounds", 0))
        parsed["treatment_tool_call_count"] = _normalize_int(parsed.get("treatment_tool_call_count", 0))
        parsed["elapsed_seconds"] = _normalize_float(parsed.get("elapsed_seconds", 0.0))
        parsed["round_limit_failed"] = _normalize_bool(parsed.get("round_limit_failed", False))
        parsed["tool_call_counts"] = _parse_json_cell(parsed.get("tool_call_counts", ""), _empty_tool_call_counts())
        for tool_name, field_name in TOOL_CALL_RESULT_FIELDS.items():
            parsed[field_name] = _normalize_int(parsed.get(field_name, 0))
            if tool_name not in parsed["tool_call_counts"]:
                parsed["tool_call_counts"][tool_name] = parsed[field_name]
        parsed["warnings"] = _parse_json_cell(parsed.get("warnings", ""), [])
        parsed["model_routes"] = _parse_json_cell(parsed.get("model_routes", ""), {})
        parsed_rows.append(parsed)
    payload = {"results": parsed_rows}
    results_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return parsed_rows


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    first_row = rows[0] if rows else {}
    success_count = sum(1 for row in rows if clean_text(row.get("status", "")) == "success")
    incomplete_count = sum(1 for row in rows if clean_text(row.get("status", "")) == "incomplete")
    error_count = sum(1 for row in rows if clean_text(row.get("status", "")) == "error")
    round_limit_failure_count = sum(1 for row in rows if _normalize_bool(row.get("round_limit_failed", False)))
    avg_rounds = (
        sum(_normalize_int(row.get("treatment_react_rounds", 0)) for row in rows) / total
        if total
        else 0.0
    )
    avg_tool_calls = (
        sum(_normalize_int(row.get("treatment_tool_call_count", 0)) for row in rows) / total
        if total
        else 0.0
    )
    avg_elapsed_seconds = (
        sum(_normalize_float(row.get("elapsed_seconds", 0.0)) for row in rows) / total
        if total
        else 0.0
    )
    summary = {
        "architecture": clean_text(first_row.get("architecture", "")),
        "base_text_preset": clean_text(first_row.get("base_text_preset", "")),
        "medical_treatment_preset": clean_text(first_row.get("medical_treatment_preset", "")),
        "treatment_generation_mode": clean_text(first_row.get("treatment_generation_mode", "")),
        "total": total,
        "success_count": success_count,
        "success_rate": (success_count / total) if total else 0.0,
        "incomplete_count": incomplete_count,
        "error_count": error_count,
        "round_limit_failure_count": round_limit_failure_count,
        "round_limit_failure_rate": (round_limit_failure_count / total) if total else 0.0,
        "avg_treatment_react_rounds": avg_rounds,
        "avg_treatment_tool_call_count": avg_tool_calls,
        "avg_elapsed_seconds": avg_elapsed_seconds,
    }
    for tool_name, field_name in TOOL_CALL_SUMMARY_FIELDS.items():
        source_field = TOOL_CALL_RESULT_FIELDS[tool_name]
        summary[field_name] = (
            sum(_normalize_int(row.get(source_field, 0)) for row in rows) / total
            if total
            else 0.0
        )
    return summary


def build_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_model_and_input: Dict[str, Dict[str, Any]] = {}
    grouped_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        model_preset = clean_text(row.get("model_preset", ""))
        input_type = clean_text(row.get("input_type", ""))
        grouped_rows.setdefault((model_preset, input_type), []).append(row)

    for (model_preset, input_type), group_rows in sorted(grouped_rows.items()):
        by_model_and_input[f"{model_preset}/{input_type}"] = summarize_rows(group_rows)

    json_rows = [row for row in rows if clean_text(row.get("input_type", "")) == "json"]
    csv_rows = [row for row in rows if clean_text(row.get("input_type", "")) == "csv"]
    json_overall = summarize_rows(json_rows)
    csv_overall = summarize_rows(csv_rows)
    overall_all_samples = summarize_rows(rows)
    for metrics in (json_overall, csv_overall, overall_all_samples):
        metrics["architecture"] = ""
        metrics["base_text_preset"] = ""
        metrics["medical_treatment_preset"] = ""
        metrics["treatment_generation_mode"] = ""
    return {
        "by_model_and_input": by_model_and_input,
        "json_overall": json_overall,
        "csv_overall": csv_overall,
        "overall_all_samples": overall_all_samples,
    }


def write_summary_files(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / DEFAULT_SUMMARY_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for group_name, metrics in sorted(summary.get("by_model_and_input", {}).items()):
            model_preset, _, input_type = group_name.partition("/")
            writer.writerow(
                {
                    "group": group_name,
                    "model_preset": model_preset,
                    "input_type": input_type,
                    **metrics,
                }
            )

        for overall_group in ("json_overall", "csv_overall", "overall_all_samples"):
            writer.writerow(
                {
                    "group": overall_group,
                    "model_preset": "",
                    "input_type": overall_group.replace("_overall", "") if overall_group.endswith("_overall") else "all",
                    **summary.get(overall_group, {}),
                }
            )


def _prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return
    for filename in (DEFAULT_RESULTS_CSV, DEFAULT_RESULTS_JSON, DEFAULT_SUMMARY_CSV, DEFAULT_SUMMARY_JSON):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _run_sample(agent: Any, *, sample: Dict[str, Any], user_query: str) -> Dict[str, Any]:
    if clean_text(sample.get("input_type", "")) == "json":
        return agent.run(
            user_query=user_query,
            case_json_path=clean_text(sample.get("source_path", "")),
            task_mode=TASK_MODE_TREATMENT,
        )
    return agent.run(
        user_query=user_query,
        case_payload=dict(sample.get("payload", {}) or {}),
        task_mode=TASK_MODE_TREATMENT,
    )


def _append_tool_ablation_suffix(base_label: str, suffix: str) -> str:
    return f"{clean_text(base_label)}__{clean_text(suffix)}"


def _validate_tool_ablation_mode(
    *,
    architecture: str,
    tool_ablation_mode: str,
    enable_generated_code: bool,
) -> str:
    normalized_mode = clean_text(tool_ablation_mode) or TOOL_ABLATION_NONE
    if normalized_mode == TOOL_ABLATION_NONE:
        return normalized_mode
    if normalized_mode != TOOL_ABLATION_GROUPED:
        raise ValueError(f"Unsupported tool ablation mode: {tool_ablation_mode}")
    if architecture != ARCHITECTURE_FULL_STACK_BASE:
        raise ValueError("--tool-ablation grouped only supports --architecture full_stack_base.")
    if enable_generated_code:
        raise ValueError("--tool-ablation grouped does not support --enable-generated-code.")
    return normalized_mode


def _build_experiment_specs(
    *,
    architecture: str,
    model_presets: Sequence[str],
    medical_model_presets: Sequence[str],
    mode: str,
    max_new_tokens: Optional[int],
    enable_generated_code: bool,
    enable_treatment_round_reminders: bool,
    tool_ablation_mode: str = TOOL_ABLATION_NONE,
) -> List[Dict[str, Any]]:
    normalized_tool_ablation_mode = _validate_tool_ablation_mode(
        architecture=architecture,
        tool_ablation_mode=tool_ablation_mode,
        enable_generated_code=enable_generated_code,
    )
    specs: List[Dict[str, Any]] = []
    if architecture == ARCHITECTURE_FULL_STACK_BASE:
        for model_preset in model_presets:
            normalized_model_name = normalize_allowed_text_backend_preset_name(
                model_preset,
                allowed_presets=BASE_TEXT_PRESET_NAMES,
            )
            base_config = build_treatment_agent_config_for_preset(
                model_preset=normalized_model_name,
                mode=mode,
                max_new_tokens=max_new_tokens,
                enable_generated_code=enable_generated_code,
                enable_treatment_round_reminders=enable_treatment_round_reminders,
            )
            if normalized_tool_ablation_mode == TOOL_ABLATION_NONE:
                specs.append(
                    {
                        "architecture": architecture,
                        "model_preset": normalized_model_name,
                        "base_text_preset": normalized_model_name,
                        "medical_treatment_preset": "",
                        "treatment_generation_mode": "fullstack_base",
                        "config": base_config,
                    }
                )
                continue
            for variant_name, disabled_tools in TOOL_ABLATION_GROUPED_VARIANTS:
                specs.append(
                    {
                        "architecture": architecture,
                        "model_preset": _append_tool_ablation_suffix(normalized_model_name, variant_name),
                        "base_text_preset": normalized_model_name,
                        "medical_treatment_preset": "",
                        "treatment_generation_mode": "fullstack_base",
                        "config": replace(
                            base_config,
                            profile_name=_append_tool_ablation_suffix(base_config.profile_name, variant_name),
                            disabled_treatment_tools=tuple(disabled_tools),
                        ),
                    }
                )
        return specs

    for model_preset in model_presets:
        normalized_model_name = normalize_allowed_text_backend_preset_name(
            model_preset,
            allowed_presets=BASE_TEXT_PRESET_NAMES,
        )
        for medical_preset in medical_model_presets:
            normalized_medical_name = normalize_allowed_text_backend_preset_name(
                medical_preset,
                allowed_presets=MEDICAL_TREATMENT_PRESET_NAMES,
            )
            config = build_medical_treatment_generator_config(
                base_model_preset=normalized_model_name,
                medical_treatment_preset=normalized_medical_name,
                mode=mode,
                max_new_tokens=max_new_tokens,
                enable_generated_code=enable_generated_code,
                enable_treatment_round_reminders=enable_treatment_round_reminders,
            )
            specs.append(
                {
                    "architecture": architecture,
                    "model_preset": f"{normalized_model_name}__{normalized_medical_name}",
                    "base_text_preset": normalized_model_name,
                    "medical_treatment_preset": normalized_medical_name,
                    "treatment_generation_mode": "medical_backend",
                    "config": config,
                }
            )
    return specs


def run_curetest_treatment_matrix(
    *,
    curetest_dir: Path,
    output_dir: Path,
    model_presets: List[str],
    architecture: str,
    medical_model_presets: Optional[List[str]],
    mode: str,
    limit: Optional[int],
    resume: bool,
    max_new_tokens: Optional[int],
    enable_generated_code: bool = False,
    enable_treatment_round_reminders: bool = False,
    tool_ablation_mode: str = TOOL_ABLATION_NONE,
    user_query: str = DEFAULT_USER_QUERY,
) -> Dict[str, Any]:
    resolved_output_dir = output_dir.expanduser().resolve()
    _prepare_output_dir(resolved_output_dir, resume=resume)
    results_csv_path = resolved_output_dir / DEFAULT_RESULTS_CSV
    results_json_path = resolved_output_dir / DEFAULT_RESULTS_JSON

    samples = discover_curetest_samples(curetest_dir)
    if limit is not None:
        samples = samples[: max(0, limit)]

    existing_keys = _read_existing_result_keys(results_csv_path) if resume else set()
    experiment_specs = _build_experiment_specs(
        architecture=architecture,
        model_presets=model_presets,
        medical_model_presets=medical_model_presets or [],
        mode=mode,
        max_new_tokens=max_new_tokens,
        enable_generated_code=enable_generated_code,
        enable_treatment_round_reminders=enable_treatment_round_reminders,
        tool_ablation_mode=tool_ablation_mode,
    )

    for spec in experiment_specs:
        config = spec["config"]
        agent = SepsisMedGemmaAgent(config)
        for sample in samples:
            resume_key = (clean_text(config.profile_name), clean_text(sample.get("case_identifier", "")))
            if resume_key in existing_keys:
                continue
            started_at = time.perf_counter()
            try:
                result = _run_sample(agent, sample=sample, user_query=user_query)
                elapsed_seconds = time.perf_counter() - started_at
                row = build_result_row(
                    model_preset=spec["model_preset"],
                    profile_name=config.profile_name,
                    architecture=spec["architecture"],
                    base_text_preset=spec["base_text_preset"],
                    medical_treatment_preset=spec["medical_treatment_preset"],
                    treatment_generation_mode=spec["treatment_generation_mode"],
                    sample=sample,
                    result=result,
                    elapsed_seconds=elapsed_seconds,
                )
            except Exception as exc:
                elapsed_seconds = time.perf_counter() - started_at
                row = build_result_row(
                    model_preset=spec["model_preset"],
                    profile_name=config.profile_name,
                    architecture=spec["architecture"],
                    base_text_preset=spec["base_text_preset"],
                    medical_treatment_preset=spec["medical_treatment_preset"],
                    treatment_generation_mode=spec["treatment_generation_mode"],
                    sample=sample,
                    error=str(exc),
                    elapsed_seconds=elapsed_seconds,
                )
            _append_result_row(results_csv_path, row)
            existing_keys.add(resume_key)

    parsed_rows = write_results_json(results_csv_path, results_json_path)
    summary = build_summary(parsed_rows)
    write_summary_files(resolved_output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    run_id = clean_text(args.run_id) or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root).expanduser().resolve() / run_id
    summary = run_curetest_treatment_matrix(
        curetest_dir=Path(args.curetest_dir).expanduser().resolve(),
        output_dir=output_dir,
        model_presets=parse_model_presets(args.models, allowed_presets=BASE_TEXT_PRESET_NAMES),
        architecture=args.architecture,
        medical_model_presets=parse_model_presets(
            args.medical_models,
            allowed_presets=MEDICAL_TREATMENT_PRESET_NAMES,
        ),
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
        enable_generated_code=args.enable_generated_code,
        enable_treatment_round_reminders=args.enable_treatment_round_reminders,
        tool_ablation_mode=args.tool_ablation,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
