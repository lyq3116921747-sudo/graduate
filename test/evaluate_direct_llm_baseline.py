from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.cli import DEFAULT_QUERY
from Agent.config import AgentConfig
from Agent.llm_presets import list_text_backend_presets, normalize_text_backend_preset_name, resolve_text_backend_preset
from Agent.modeling import build_text_backend
from Agent.utils import (
    build_case_payload_from_csv_row,
    case_to_context,
    clean_text,
    extract_json_object,
    load_case_payload,
    truncate_text,
)
try:
    from .evaluate_agent_on_testset import (
        DEFAULT_RESULTS_CSV,
        DEFAULT_RESULTS_JSON,
        DEFAULT_RUN_LOG,
        RESULT_FIELDS,
        build_result_row,
        build_summary,
        write_summary_files,
    )
except ImportError:
    from evaluate_agent_on_testset import (
        DEFAULT_RESULTS_CSV,
        DEFAULT_RESULTS_JSON,
        DEFAULT_RUN_LOG,
        RESULT_FIELDS,
        build_result_row,
        build_summary,
        write_summary_files,
    )


DEFAULT_CSV_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_CSV_NON_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_non_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_JSON_MANIFEST = PROJECT_ROOT / "miniTest" / "json_sample_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "miniTest" / "direct_llm_baseline"

BASELINE_SYSTEM_PROMPT = (
    "你是严格按 Sepsis-3 进行诊断的临床推理助手。"
    "只能根据给定病例文本做判断，不得编造缺失事实。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 direct LLM baseline 在 miniTest 上的表现。")
    parser.add_argument("--model-preset", choices=sorted(list_text_backend_presets()), default="qwen_plus")
    parser.add_argument("--csv-sepsis", default=str(DEFAULT_CSV_SEPSIS))
    parser.add_argument("--csv-non-sepsis", default=str(DEFAULT_CSV_NON_SEPSIS))
    parser.add_argument("--json-manifest", default=str(DEFAULT_JSON_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def resolve_direct_baseline_budget(*, model_preset: str, max_new_tokens: Optional[int]) -> tuple[Any, int, float]:
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


def _read_existing_case_ids(csv_path: Path) -> Set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {clean_text(row.get("case_identifier", "")) for row in csv.DictReader(handle) if clean_text(row.get("case_identifier", ""))}


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


def normalize_baseline_label(value: Any) -> str:
    normalized = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    if normalized in {"sepsis", "脓毒症"}:
        return "sepsis"
    if normalized in {"non_sepsis", "nonsepsis", "非脓毒症"}:
        return "non_sepsis"
    if normalized in {"insufficient_evidence", "insufficient", "证据不足"}:
        return "insufficient_evidence"
    return ""


def build_direct_baseline_prompt(case_context: str) -> str:
    schema_text = '{"label":"sepsis|non_sepsis|insufficient_evidence","reason":"简要依据"}'
    return (
        "请直接根据病例文本输出最终诊断标签，不要输出推理过程，不要使用 Markdown。\n"
        "只返回一个 JSON 对象，格式如下：\n"
        f"{schema_text}\n"
        "要求：\n"
        "1. label 只能是 sepsis、non_sepsis、insufficient_evidence 三者之一。\n"
        "2. reason 用一句中文简要说明核心依据。\n"
        "3. 如果证据不足以支持明确判断，返回 insufficient_evidence。\n"
        f"4. 任务背景：{DEFAULT_QUERY}\n\n"
        f"病例文本：\n{truncate_text(case_context, 3600)}"
    )


def parse_direct_baseline_response(raw_text: str) -> Dict[str, str]:
    payload = extract_json_object(raw_text) or {}
    label = normalize_baseline_label(payload.get("label", ""))
    reason = clean_text(payload.get("reason", ""))
    if not label:
        return {}
    return {"label": label, "reason": reason}


def run_direct_baseline_prediction(
    *,
    text_model: Any,
    case_context: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompt = build_direct_baseline_prompt(case_context)
    attempts: List[Dict[str, str]] = []
    final_payload: Dict[str, str] = {}
    final_error = ""
    parse_mode = "empty"

    for attempt_index in range(2):
        user_prompt = prompt
        if attempt_index == 1:
            previous = clean_text(attempts[0]["raw"]) if attempts else ""
            user_prompt = (
                "上一轮输出不是合法 JSON。现在重新输出一个更短且完整的 JSON。"
                "只能包含 label 和 reason 两个字段。\n\n"
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
            error = str(exc)
        parsed = parse_direct_baseline_response(raw_text)
        attempts.append({"raw": clean_text(raw_text), "error": clean_text(error)})
        if parsed:
            final_payload = parsed
            parse_mode = "json"
            final_error = ""
            break
        parse_mode = "empty" if not clean_text(raw_text) else "non_json"
        final_error = clean_text(error) or ("empty_response" if not clean_text(raw_text) else "unparseable_response")

    return {
        "ok": bool(final_payload),
        "label": clean_text(final_payload.get("label", "")),
        "reason": clean_text(final_payload.get("reason", "")),
        "raw": clean_text(attempts[-1]["raw"]) if attempts else "",
        "attempts": attempts,
        "parse_mode": parse_mode,
        "error": final_error,
    }


def build_baseline_result(prediction: Dict[str, Any]) -> Dict[str, Any]:
    label = clean_text(prediction.get("label", "")) or "extraction_failed"
    reason = clean_text(prediction.get("reason", "")) or clean_text(prediction.get("error", ""))
    stage_errors: List[Dict[str, str]] = []
    if clean_text(prediction.get("error", "")):
        stage_errors.append({"stage": "direct_baseline", "error": clean_text(prediction.get("error", ""))})
    answer = clean_text(prediction.get("raw", ""))
    if not answer:
        answer = f"label={label}; reason={reason}"
    return {
        "answer": answer,
        "meta": {
            "sepsis3_final_label": label,
            "sepsis3_judgement": {"sepsis3_reason": reason},
            "stage_errors": stage_errors,
            "direct_baseline_parse_mode": clean_text(prediction.get("parse_mode", "")),
            "direct_baseline_attempts": prediction.get("attempts", []),
        },
    }


def build_case_context_from_csv_row(row: Dict[str, Any], *, mode: str) -> Dict[str, Any]:
    payload = build_case_payload_from_csv_row(dict(row), mode=mode)
    return {
        "payload": payload,
        "case_context": case_to_context(payload, max_field_chars=1400, lab_selection="sepsis_priority"),
    }


def build_case_context_from_json_path(case_json_path: Path, *, mode: str) -> Dict[str, Any]:
    payload = load_case_payload(case_json_path, mode=mode)
    return {
        "payload": payload,
        "case_context": case_to_context(payload, max_field_chars=1400, lab_selection="sepsis_priority"),
    }


def run_csv_baseline_evaluation(
    *,
    text_model: Any,
    csv_path: Path,
    ground_truth: str,
    sample_group: str,
    mode: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int],
    model_profile: str,
    max_new_tokens: int,
) -> int:
    processed = 0
    for row_index, row in enumerate(_read_csv_rows(csv_path)):
        case_identifier = f"csv::{csv_path.name}::{row_index}"
        if case_identifier in resume_case_ids:
            continue
        if limit is not None and processed >= limit:
            break
        prepared = build_case_context_from_csv_row(row, mode=mode)
        prediction = run_direct_baseline_prediction(
            text_model=text_model,
            case_context=prepared["case_context"],
            max_new_tokens=max_new_tokens,
        )
        result = build_baseline_result(prediction)
        result_row = build_result_row(
            case_identifier=case_identifier,
            sample_type="csv",
            sample_group=sample_group,
            ground_truth=ground_truth,
            source_path=str(csv_path),
            row_index=str(row_index),
            bucket="",
            filename=clean_text(row.get("流水号", "")) or clean_text(row.get("EMPI", "")),
            model_profile=model_profile,
            result=result,
        )
        _append_result_row(results_path, result_row)
        _append_run_log(
            run_log_path,
            {
                "case_identifier": case_identifier,
                "sample_type": "csv",
                "sample_group": sample_group,
                "ground_truth": ground_truth,
                "raw_classification": result_row["raw_classification"],
                "predicted_label": result_row["predicted_label"],
            },
        )
        processed += 1
    return processed


def run_json_baseline_evaluation(
    *,
    text_model: Any,
    manifest_path: Path,
    mode: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int],
    model_profile: str,
    max_new_tokens: int,
) -> int:
    processed = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bucket = clean_text(row.get("bucket", ""))
            ground_truth = clean_text(row.get("label", ""))
            filename = clean_text(row.get("filename", ""))
            source_path = clean_text(row.get("copied_path", "")) or clean_text(row.get("source_path", ""))
            case_identifier = f"json::{bucket}::{ground_truth}::{filename}"
            if case_identifier in resume_case_ids:
                continue
            if limit is not None and processed >= limit:
                break
            prepared = build_case_context_from_json_path(Path(source_path), mode=mode)
            prediction = run_direct_baseline_prediction(
                text_model=text_model,
                case_context=prepared["case_context"],
                max_new_tokens=max_new_tokens,
            )
            result = build_baseline_result(prediction)
            sample_group = f"json_{bucket.replace('json_completeness_', '')}_{ground_truth}"
            result_row = build_result_row(
                case_identifier=case_identifier,
                sample_type="json",
                sample_group=sample_group,
                ground_truth=ground_truth,
                source_path=source_path,
                row_index="",
                bucket=bucket,
                filename=filename,
                model_profile=model_profile,
                result=result,
            )
            _append_result_row(results_path, result_row)
            _append_run_log(
                run_log_path,
                {
                    "case_identifier": case_identifier,
                    "sample_type": "json",
                    "sample_group": sample_group,
                    "ground_truth": ground_truth,
                    "raw_classification": result_row["raw_classification"],
                    "predicted_label": result_row["predicted_label"],
                },
            )
            processed += 1
    return processed


def evaluate_direct_llm_baseline(
    *,
    model_preset: str,
    csv_sepsis_path: Path = DEFAULT_CSV_SEPSIS,
    csv_non_sepsis_path: Path = DEFAULT_CSV_NON_SEPSIS,
    json_manifest_path: Path = DEFAULT_JSON_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "strict_eval",
    limit: Optional[int] = None,
    resume: bool = True,
    max_new_tokens: Optional[int] = None,
    model_profile: Optional[str] = None,
    text_model: Optional[Any] = None,
) -> Dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_results_path = output_dir / DEFAULT_RESULTS_CSV
    json_results_path = output_dir / DEFAULT_RESULTS_JSON
    run_log_path = output_dir / DEFAULT_RUN_LOG

    if not resume:
        for path in (csv_results_path, json_results_path, run_log_path):
            if path.exists():
                path.unlink()

    resume_csv_ids = _read_existing_case_ids(csv_results_path) if resume else set()
    resume_json_ids = _read_existing_case_ids(json_results_path) if resume else set()
    backend_config, effective_max_new_tokens, effective_timeout_seconds = resolve_direct_baseline_budget(
        model_preset=model_preset,
        max_new_tokens=max_new_tokens,
    )

    built_text_model = None
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

    model_profile = model_profile or f"direct_baseline/{normalize_text_backend_preset_name(model_preset)}"

    try:
        run_csv_baseline_evaluation(
            text_model=text_model,
            csv_path=csv_sepsis_path,
            ground_truth="sepsis",
            sample_group="csv_sepsis",
            mode=mode,
            results_path=csv_results_path,
            run_log_path=run_log_path,
            resume_case_ids=resume_csv_ids,
            limit=limit,
            model_profile=model_profile,
            max_new_tokens=effective_max_new_tokens,
        )
        run_csv_baseline_evaluation(
            text_model=text_model,
            csv_path=csv_non_sepsis_path,
            ground_truth="non_sepsis",
            sample_group="csv_non_sepsis",
            mode=mode,
            results_path=csv_results_path,
            run_log_path=run_log_path,
            resume_case_ids=resume_csv_ids,
            limit=limit,
            model_profile=model_profile,
            max_new_tokens=effective_max_new_tokens,
        )
        run_json_baseline_evaluation(
            text_model=text_model,
            manifest_path=json_manifest_path,
            mode=mode,
            results_path=json_results_path,
            run_log_path=run_log_path,
            resume_case_ids=resume_json_ids,
            limit=limit,
            model_profile=model_profile,
            max_new_tokens=effective_max_new_tokens,
        )
    finally:
        if built_text_model is not None:
            built_text_model.close()

    csv_rows = _load_result_rows(csv_results_path)
    json_rows = _load_result_rows(json_results_path)
    summary = build_summary(csv_rows, json_rows)
    write_summary_files(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_direct_llm_baseline(
        model_preset=args.model_preset,
        csv_sepsis_path=Path(args.csv_sepsis).expanduser().resolve(),
        csv_non_sepsis_path=Path(args.csv_non_sepsis).expanduser().resolve(),
        json_manifest_path=Path(args.json_manifest).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
