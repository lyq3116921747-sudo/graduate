from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.config import AgentConfig
from Agent.llm_presets import (
    list_text_backend_presets,
    normalize_text_backend_preset_name,
    resolve_text_backend_preset,
)
from Agent.modeling import build_text_backend
from Agent.utils import (
    build_case_payload_from_csv_row,
    case_to_context,
    clean_text,
    load_case_payload,
    truncate_text,
)


DEFAULT_CURETEST_DIR = PROJECT_ROOT / "data" / "CureTest"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "miniTest" / "curetest_treatment_matrix" / "curetest_direct_baseline"
DEFAULT_RESULTS_CSV = "results.csv"
DEFAULT_RESULTS_JSON = "results.json"
DEFAULT_SUMMARY_CSV = "summary.csv"
DEFAULT_SUMMARY_JSON = "summary.json"
DEFAULT_RUN_LOG = "run_log.jsonl"
DEFAULT_MODELS = "qwen_plus,deepseek,qwen_max"
DEFAULT_USER_QUERY = "请基于病例给出脓毒症治疗建议。"
ARCHITECTURE_DIRECT_TREATMENT_BASELINE = "direct_treatment_baseline"
DIRECT_TREATMENT_GENERATION_MODE = "direct_llm_baseline"
CSV_FILENAME = "sample_patients.csv"
TREATMENT_TOOL_NAMES = (
    "get_case_digest",
    "read_case_sections",
    "search_case_text",
    "run_sofa_assessment",
    "search_local_guidelines",
    "run_generated_probe",
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
BASELINE_SYSTEM_PROMPT = (
    "你是谨慎的临床治疗建议助手。"
    "你只能基于给定病例文本提出治疗建议，不得编造不存在的检查结果、病情进展或指南结论。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct LLM treatment baseline on CureTest without tools or React."
    )
    parser.add_argument("--curetest-dir", default=str(DEFAULT_CURETEST_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--user-query", default=DEFAULT_USER_QUERY)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def parse_model_presets(raw_models: str) -> List[str]:
    presets: List[str] = []
    available = tuple(list_text_backend_presets().keys())
    for raw_item in raw_models.split(","):
        item = clean_text(raw_item)
        if not item:
            continue
        normalized = normalize_text_backend_preset_name(item)
        if normalized not in available:
            supported = ", ".join(sorted(available))
            raise ValueError(f"Unsupported model preset: {item}; available: {supported}")
        if normalized not in presets:
            presets.append(normalized)
    if not presets:
        supported = ", ".join(sorted(available))
        raise ValueError(f"At least one model preset is required; available: {supported}")
    return presets


def resolve_direct_treatment_budget(
    *,
    model_preset: str,
    max_new_tokens: Optional[int],
) -> Tuple[Any, int, float]:
    base_config = AgentConfig()
    backend_config = resolve_text_backend_preset(model_preset)
    resolved_max_new_tokens = max_new_tokens
    if resolved_max_new_tokens is None:
        resolved_max_new_tokens = backend_config.default_max_new_tokens or base_config.generation_max_new_tokens
    resolved_timeout_seconds = max(
        base_config.http_timeout_seconds,
        float(backend_config.default_timeout_seconds or 0.0),
    )
    return backend_config, int(resolved_max_new_tokens), resolved_timeout_seconds


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
    return discover_curetest_json_samples(resolved_dir) + discover_curetest_csv_samples(resolved_dir)


def _read_existing_result_keys(csv_path: Path) -> Set[Tuple[str, str]]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = list(RESULT_FIELDS)
        actual_header = list(reader.fieldnames or [])
        if actual_header != expected_header:
            raise ValueError(
                "Existing results.csv schema does not match the direct treatment baseline RESULT_FIELDS; "
                "please rerun with --no-resume to rebuild the outputs."
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


def _append_run_log(log_path: Path, payload: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


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


def build_model_result_preset_name(model_preset: str) -> str:
    return f"direct_baseline__{normalize_text_backend_preset_name(model_preset)}"


def build_model_profile_name(model_preset: str) -> str:
    return f"curetest_treatment_direct_baseline/{normalize_text_backend_preset_name(model_preset)}"


def build_direct_treatment_prompt(*, case_context: str, user_query: str) -> str:
    del user_query
    return (
        "请直接根据下面的病例文本给出最终治疗建议。\n"
        "要求：\n"
        "1. 仅依据病例文本作答，不得编造不存在的化验、影像、诊断或指南结论。\n"
        "2. 不要调用工具，不要写思维过程，不要写免责声明。\n"
        "3. 使用中文，尽量给出 4-8 条编号治疗建议。\n"
        "4. 建议尽量覆盖：初始复苏与液体管理、抗感染策略、循环与呼吸支持、监测复评、病因或源控制。\n"
        "5. 如果信息不足，请明确写“建议评估/复查/监测”，不要虚构结果。\n"
        "6. 输出格式尽量为：\n"
        "治疗建议：\n"
        "1. ...\n"
        "2. ...\n\n"
        f"病例文本：\n{truncate_text(case_context, 5200)}"
    )


# Align the direct baseline with the agent's no-tool output principles.
BASELINE_SYSTEM_PROMPT = (
    "你是脓毒症治疗建议助手。"
    "请基于已提供的病例信息直接形成最终治疗建议。"
    "不要臆造病例事实；信息不足时请明确指出缺口。"
    "最终回答只输出“治疗建议：”一节，使用编号分点。"
)


def build_direct_treatment_prompt(*, case_context: str, user_query: str) -> str:
    return (
        f"用户请求：{clean_text(user_query) or '请给出治疗建议。'}\n\n"
        "当前没有可用工具。请仅基于以下已提供的病例信息直接作答。\n\n"
        f"病例信息：\n{truncate_text(case_context, 5200)}\n\n"
        "请直接输出最终“治疗建议：”一节，使用编号分点。\n"
        "不要复述分析过程，不要输出工具说明，不要编造未提供的检查结果、病情进展或指南结论；"
        "若信息不足，请明确指出仍需补充的评估、复查或监测内容。\n"
    )


def normalize_direct_treatment_answer(raw_text: str) -> str:
    normalized = clean_text(raw_text)
    if not normalized:
        return ""
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[a-zA-Z0-9_]*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()
    match = re.search(r"(治疗建议[:：].*)", normalized, flags=re.DOTALL)
    if match:
        normalized = clean_text(match.group(1))
    if not normalized:
        return ""
    if not re.match(r"^治疗建议[:：]", normalized):
        if re.match(r"^(?:[-*]\s+|\d+[.、])", normalized):
            normalized = f"治疗建议：\n{normalized}"
    return normalized


def run_direct_treatment_prediction(
    *,
    text_model: Any,
    case_context: str,
    user_query: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt = build_direct_treatment_prompt(case_context=case_context, user_query=user_query)
    attempts: List[Dict[str, Any]] = []
    final_error = ""

    for attempt_index in range(2):
        user_prompt = prompt
        if attempt_index == 1:
            previous = clean_text(attempts[0].get("raw", "")) if attempts else ""
            user_prompt = (
                "上一轮输出为空或不可用。请重新直接输出最终治疗建议。\n"
                "只保留“治疗建议”正文，不要附加解释。\n\n"
                f"上一轮输出：\n{truncate_text(previous, 1200)}\n\n{prompt}"
            )
        raw_text = ""
        error = ""
        try:
            raw_text = text_model.generate(
                system_prompt=BASELINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_length=None,
            )
        except Exception as exc:
            error = clean_text(exc)
        answer = normalize_direct_treatment_answer(raw_text)
        attempts.append(
            {
                "raw": clean_text(raw_text),
                "normalized_answer": answer,
                "error": error,
            }
        )
        if answer:
            return {
                "ok": True,
                "answer": answer,
                "attempts": attempts,
                "error": "",
                "parsed_on_attempt": attempt_index + 1,
            }
        final_error = error or "empty_response"

    return {
        "ok": False,
        "answer": "",
        "attempts": attempts,
        "error": final_error,
        "parsed_on_attempt": 0,
    }


def run_direct_treatment_prediction(
    *,
    text_model: Any,
    case_context: str,
    user_query: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt = build_direct_treatment_prompt(case_context=case_context, user_query=user_query)
    attempts: List[Dict[str, Any]] = []
    final_error = ""

    for attempt_index in range(2):
        user_prompt = prompt
        if attempt_index == 1:
            previous = clean_text(attempts[0].get("raw", "")) if attempts else ""
            user_prompt = (
                "上一轮输出为空或不可用。请直接基于相同病例信息重新输出最终“治疗建议：”一节。\n"
                "不要补充分析过程，不要输出工具说明；若信息不足，请直接指出缺口。\n\n"
                f"上一轮输出：\n{truncate_text(previous, 1200)}\n\n{prompt}"
            )
        raw_text = ""
        error = ""
        try:
            raw_text = text_model.generate(
                system_prompt=BASELINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_length=None,
            )
        except Exception as exc:
            error = clean_text(exc)
        answer = normalize_direct_treatment_answer(raw_text)
        attempts.append(
            {
                "raw": clean_text(raw_text),
                "normalized_answer": answer,
                "error": error,
            }
        )
        if answer:
            return {
                "ok": True,
                "answer": answer,
                "attempts": attempts,
                "error": "",
                "parsed_on_attempt": attempt_index + 1,
            }
        final_error = error or "empty_response"

    return {
        "ok": False,
        "answer": "",
        "attempts": attempts,
        "error": final_error,
        "parsed_on_attempt": 0,
    }


BASELINE_SYSTEM_PROMPT = (
    "你是脓毒症治疗建议助手。"
    "请根据给定病例信息直接输出最终治疗建议。"
    "最终回答只输出“治疗建议：”一节，使用编号分点。"
)


def build_direct_treatment_prompt(*, case_context: str, user_query: str) -> str:
    return (
        f"用户请求：{clean_text(user_query) or '请给出治疗建议。'}\n\n"
        f"病例信息：\n{truncate_text(case_context, 5200)}\n\n"
        "请直接输出最终“治疗建议：”一节。"
    )


def run_direct_treatment_prediction(
    *,
    text_model: Any,
    case_context: str,
    user_query: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt = build_direct_treatment_prompt(case_context=case_context, user_query=user_query)
    attempts: List[Dict[str, Any]] = []
    final_error = ""

    for attempt_index in range(2):
        user_prompt = prompt if attempt_index == 0 else (
            "请重新直接输出最终“治疗建议：”一节。\n\n"
            f"病例信息：\n{truncate_text(case_context, 5200)}"
        )
        raw_text = ""
        error = ""
        try:
            raw_text = text_model.generate(
                system_prompt=BASELINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_length=None,
            )
        except Exception as exc:
            error = clean_text(exc)
        answer = normalize_direct_treatment_answer(raw_text)
        attempts.append(
            {
                "raw": clean_text(raw_text),
                "normalized_answer": answer,
                "error": error,
            }
        )
        if answer:
            return {
                "ok": True,
                "answer": answer,
                "attempts": attempts,
                "error": "",
                "parsed_on_attempt": attempt_index + 1,
            }
        final_error = error or "empty_response"

    return {
        "ok": False,
        "answer": "",
        "attempts": attempts,
        "error": final_error,
        "parsed_on_attempt": 0,
    }


def build_case_context_from_csv_row(row: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    payload = build_case_payload_from_csv_row(dict(row), mode=mode)
    return {
        "payload": payload,
        "case_context": case_to_context(
            payload,
            max_field_chars=1400,
            lab_selection="default",
            task_mode="treatment",
        ),
    }


def build_case_context_from_json_path(case_json_path: Path, *, mode: str) -> Dict[str, Any]:
    payload = load_case_payload(case_json_path, mode=mode)
    return {
        "payload": payload,
        "case_context": case_to_context(
            payload,
            max_field_chars=1400,
            lab_selection="default",
            task_mode="treatment",
        ),
    }


def prepare_case_context(sample: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    if clean_text(sample.get("input_type", "")) == "json":
        return build_case_context_from_json_path(Path(clean_text(sample.get("source_path", ""))), mode=mode)
    return build_case_context_from_csv_row(dict(sample.get("payload", {}) or {}), mode=mode)


def describe_text_model(text_model: Any, *, fallback: str) -> str:
    describe_fn = getattr(text_model, "describe", None)
    if callable(describe_fn):
        try:
            described = clean_text(describe_fn())
            if described:
                return described
        except Exception:
            pass
    return fallback


def build_result_row(
    *,
    model_preset: str,
    profile_name: str,
    base_text_preset: str,
    sample: Dict[str, Any],
    answer: str,
    error: str,
    elapsed_seconds: float,
    model_route: str,
) -> Dict[str, Any]:
    resolved_answer = clean_text(answer)
    resolved_error = clean_text(error)
    success = bool(resolved_answer) and not resolved_error
    row = {
        "model_preset": clean_text(model_preset),
        "profile_name": clean_text(profile_name),
        "architecture": ARCHITECTURE_DIRECT_TREATMENT_BASELINE,
        "base_text_preset": clean_text(base_text_preset),
        "medical_treatment_preset": "",
        "treatment_generation_mode": DIRECT_TREATMENT_GENERATION_MODE,
        "input_type": clean_text(sample.get("input_type", "")),
        "case_identifier": clean_text(sample.get("case_identifier", "")),
        "source_path": clean_text(sample.get("source_path", "")),
        "row_index": clean_text(sample.get("row_index", "")),
        "status": "success" if success else "error",
        "answer": resolved_answer,
        "error": resolved_error,
        "treatment_react_status": "not_applicable" if success else "",
        "treatment_react_rounds": 0,
        "treatment_tool_call_count": 0,
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "round_limit_failed": False,
        "tool_call_counts": json.dumps(_empty_tool_call_counts(), ensure_ascii=False, sort_keys=True),
        "warnings": json.dumps([], ensure_ascii=False),
        "model_routes": json.dumps({"direct_treatment": clean_text(model_route)}, ensure_ascii=False, sort_keys=True),
    }
    for tool_name, field_name in TOOL_CALL_RESULT_FIELDS.items():
        del tool_name
        row[field_name] = 0
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
    for filename in (DEFAULT_RESULTS_CSV, DEFAULT_RESULTS_JSON, DEFAULT_SUMMARY_CSV, DEFAULT_SUMMARY_JSON, DEFAULT_RUN_LOG):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def evaluate_direct_treatment_baseline(
    *,
    curetest_dir: Path = DEFAULT_CURETEST_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_presets: Sequence[str],
    mode: str = "strict_eval",
    limit: Optional[int] = None,
    resume: bool = True,
    max_new_tokens: Optional[int] = None,
    user_query: str = DEFAULT_USER_QUERY,
    text_model_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_output_dir = output_dir.expanduser().resolve()
    _prepare_output_dir(resolved_output_dir, resume=resume)
    results_csv_path = resolved_output_dir / DEFAULT_RESULTS_CSV
    results_json_path = resolved_output_dir / DEFAULT_RESULTS_JSON
    run_log_path = resolved_output_dir / DEFAULT_RUN_LOG

    samples = discover_curetest_samples(curetest_dir)
    if limit is not None:
        samples = samples[: max(0, limit)]

    existing_keys = _read_existing_result_keys(results_csv_path) if resume else set()
    overrides = dict(text_model_overrides or {})

    for raw_model_preset in model_presets:
        model_preset = normalize_text_backend_preset_name(raw_model_preset)
        backend_config, effective_max_new_tokens, effective_timeout_seconds = resolve_direct_treatment_budget(
            model_preset=model_preset,
            max_new_tokens=max_new_tokens,
        )
        built_text_model = None
        text_model = overrides.get(model_preset)
        if text_model is None:
            base_config = AgentConfig()
            built_text_model = build_text_backend(
                backend_config=backend_config,
                hf_cache_root=base_config.hf_cache_root,
                attn_implementation=base_config.attn_implementation,
                local_files_only=base_config.local_files_only,
                max_length=base_config.generation_max_length,
                timeout_seconds=effective_timeout_seconds,
            )
            text_model = built_text_model

        profile_name = build_model_profile_name(model_preset)
        result_model_preset = build_model_result_preset_name(model_preset)
        model_route = describe_text_model(
            text_model,
            fallback=f"direct_treatment::{backend_config.provider}:{backend_config.model}",
        )

        try:
            for sample in samples:
                resume_key = (profile_name, clean_text(sample.get("case_identifier", "")))
                if resume_key in existing_keys:
                    continue
                started_at = time.perf_counter()
                prediction: Optional[Dict[str, Any]] = None
                case_context = ""
                try:
                    prepared = prepare_case_context(sample, mode=mode)
                    case_context = clean_text(prepared.get("case_context", ""))
                    prediction = run_direct_treatment_prediction(
                        text_model=text_model,
                        case_context=case_context,
                        user_query=user_query,
                        max_new_tokens=effective_max_new_tokens,
                    )
                    elapsed_seconds = time.perf_counter() - started_at
                    row = build_result_row(
                        model_preset=result_model_preset,
                        profile_name=profile_name,
                        base_text_preset=model_preset,
                        sample=sample,
                        answer=clean_text(prediction.get("answer", "")),
                        error="" if prediction.get("ok") else clean_text(prediction.get("error", "")),
                        elapsed_seconds=elapsed_seconds,
                        model_route=model_route,
                    )
                except Exception as exc:
                    elapsed_seconds = time.perf_counter() - started_at
                    row = build_result_row(
                        model_preset=result_model_preset,
                        profile_name=profile_name,
                        base_text_preset=model_preset,
                        sample=sample,
                        answer="",
                        error=clean_text(exc),
                        elapsed_seconds=elapsed_seconds,
                        model_route=model_route,
                    )
                _append_result_row(results_csv_path, row)
                _append_run_log(
                    run_log_path,
                    {
                        "model_preset": result_model_preset,
                        "base_text_preset": model_preset,
                        "profile_name": profile_name,
                        "case_identifier": clean_text(sample.get("case_identifier", "")),
                        "input_type": clean_text(sample.get("input_type", "")),
                        "source_path": clean_text(sample.get("source_path", "")),
                        "row_index": clean_text(sample.get("row_index", "")),
                        "status": clean_text(row.get("status", "")),
                        "error": clean_text(row.get("error", "")),
                        "answer_chars": len(clean_text(row.get("answer", ""))),
                        "case_context_chars": len(case_context),
                        "attempts": prediction.get("attempts", []) if isinstance(prediction, dict) else [],
                    },
                )
                existing_keys.add(resume_key)
        finally:
            if built_text_model is not None:
                built_text_model.close()

    parsed_rows = write_results_json(results_csv_path, results_json_path)
    summary = build_summary(parsed_rows)
    write_summary_files(resolved_output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_direct_treatment_baseline(
        curetest_dir=Path(args.curetest_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        model_presets=parse_model_presets(args.models),
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
        user_query=args.user_query,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
