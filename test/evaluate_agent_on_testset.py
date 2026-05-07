from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.ablation import build_ablation_config
from Agent.agent import SepsisMedGemmaAgent
from Agent.cli import DEFAULT_QUERY
from Agent.config import AgentConfig
from Agent.utils import build_case_payload_from_csv_row, clean_text


DEFAULT_CSV_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_CSV_NON_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_non_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_JSON_MANIFEST = PROJECT_ROOT / "miniTest" / "json_sample_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "miniTest" / "eval_agent_run"
DEFAULT_RESULTS_CSV = "csv_results.csv"
DEFAULT_RESULTS_JSON = "json_results.csv"
DEFAULT_SUMMARY_JSON = "summary.json"
DEFAULT_SUMMARY_CSV = "summary.csv"
DEFAULT_RUN_LOG = "run_log.jsonl"

RESULT_FIELDS = [
    "case_identifier",
    "sample_type",
    "sample_group",
    "ground_truth",
    "predicted_label",
    "raw_classification",
    "strict_score",
    "soft_score",
    "is_correct_strict",
    "is_abstain",
    "case_identifier_short",
    "source_path",
    "row_index",
    "bucket",
    "filename",
    "answer",
    "sepsis3_reason",
    "stage_errors",
    "model_profile",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评测 Agent 在 data/test 测试集上的表现。")
    parser.add_argument("--csv-sepsis", default=str(DEFAULT_CSV_SEPSIS))
    parser.add_argument("--csv-non-sepsis", default=str(DEFAULT_CSV_NON_SEPSIS))
    parser.add_argument("--json-manifest", default=str(DEFAULT_JSON_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profile", default="default")
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    config = build_ablation_config(args.profile, AgentConfig())
    config = replace(config, case_input_mode=args.mode)
    if args.text_model is not None:
        config = replace(
            config,
            default_text_backend=replace(config.default_text_backend, model=args.text_model),
        )
    if args.max_new_tokens is not None:
        config = replace(config, generation_max_new_tokens=args.max_new_tokens)
    return config


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


def normalize_prediction(raw_label: str) -> str:
    normalized = clean_text(raw_label)
    if normalized == "sepsis":
        return "sepsis"
    if normalized in {"non_sepsis", "insufficient_evidence"}:
        return "non_sepsis"
    if normalized == "extraction_failed":
        return "abstain"
    return "abstain"


def score_prediction(ground_truth: str, predicted_label: str) -> Dict[str, Any]:
    is_abstain = predicted_label == "abstain"
    strict_score = 1 if predicted_label == ground_truth else 0
    if predicted_label == ground_truth:
        soft_score = 1.0
    elif is_abstain:
        soft_score = 0.5
    else:
        soft_score = 0.0
    return {
        "strict_score": strict_score,
        "soft_score": soft_score,
        "is_correct_strict": strict_score == 1,
        "is_abstain": is_abstain,
    }


def build_result_row(
    *,
    case_identifier: str,
    sample_type: str,
    sample_group: str,
    ground_truth: str,
    source_path: str,
    row_index: str,
    bucket: str,
    filename: str,
    model_profile: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    raw_classification = clean_text(result.get("meta", {}).get("sepsis3_final_label", ""))
    predicted_label = normalize_prediction(raw_classification)
    scoring = score_prediction(ground_truth, predicted_label)
    stage_errors = result.get("meta", {}).get("stage_errors", [])
    sepsis3_reason = clean_text(result.get("meta", {}).get("sepsis3_judgement", {}).get("sepsis3_reason", "")) or clean_text(
        result.get("meta", {}).get("sepsis3_failure_mode", "")
    )
    return {
        "case_identifier": case_identifier,
        "sample_type": sample_type,
        "sample_group": sample_group,
        "ground_truth": ground_truth,
        "predicted_label": predicted_label,
        "raw_classification": raw_classification,
        "strict_score": scoring["strict_score"],
        "soft_score": f"{scoring['soft_score']:.1f}",
        "is_correct_strict": int(scoring["is_correct_strict"]),
        "is_abstain": int(scoring["is_abstain"]),
        "case_identifier_short": filename or case_identifier,
        "source_path": source_path,
        "row_index": row_index,
        "bucket": bucket,
        "filename": filename,
        "answer": clean_text(result.get("answer", "")),
        "sepsis3_reason": sepsis3_reason,
        "stage_errors": json.dumps(stage_errors, ensure_ascii=False),
        "model_profile": model_profile,
    }


def run_csv_evaluation(
    *,
    agent: Any,
    csv_path: Path,
    ground_truth: str,
    sample_group: str,
    mode: str,
    user_query: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int] = None,
    model_profile: str,
) -> int:
    processed = 0
    rows = _read_csv_rows(csv_path)
    for row_index, row in enumerate(rows):
        case_identifier = f"csv::{csv_path.name}::{row_index}"
        if case_identifier in resume_case_ids:
            continue
        if limit is not None and processed >= limit:
            break
        payload = build_case_payload_from_csv_row(dict(row), mode=mode)
        result = agent.run(user_query=user_query, case_payload=payload)
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


def run_json_evaluation(
    *,
    agent: Any,
    manifest_path: Path,
    user_query: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int] = None,
    model_profile: str,
) -> int:
    processed = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bucket = clean_text(row.get("bucket", ""))
            ground_truth = clean_text(row.get("label", ""))
            filename = clean_text(row.get("filename", ""))
            source_path = resolve_json_case_path_from_manifest_row(row)
            case_identifier = f"json::{bucket}::{ground_truth}::{filename}"
            if case_identifier in resume_case_ids:
                continue
            if limit is not None and processed >= limit:
                break
            result = agent.run(user_query=user_query, case_json_path=source_path)
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


def _load_result_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_json_case_path_from_manifest_row(row: Dict[str, Any]) -> str:
    candidate = clean_text(row.get("copied_path", "")) or clean_text(row.get("source_path", ""))
    if candidate and Path(candidate).exists():
        return candidate

    bucket = clean_text(row.get("bucket", ""))
    ground_truth = clean_text(row.get("label", ""))
    filename = clean_text(row.get("filename", ""))
    if bucket and ground_truth and filename:
        local_fallback = PROJECT_ROOT / "miniTest" / "json" / bucket / ground_truth / filename
        if local_fallback.exists():
            return str(local_fallback)
    return candidate


def summarize_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct_strict = sum(int(clean_text(row.get("strict_score", "0")) or "0") for row in rows)
    score_soft_sum = sum(float(clean_text(row.get("soft_score", "0")) or "0") for row in rows)
    abstain_count = sum(1 for row in rows if clean_text(row.get("predicted_label", "")) == "abstain")
    return {
        "total": total,
        "correct_strict": correct_strict,
        "accuracy_strict": (correct_strict / total) if total else 0.0,
        "score_soft_sum": score_soft_sum,
        "accuracy_soft": (score_soft_sum / total) if total else 0.0,
        "abstain_count": abstain_count,
        "abstain_rate": (abstain_count / total) if total else 0.0,
    }


def build_summary(csv_rows: Sequence[Dict[str, Any]], json_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    csv_sepsis_rows = [row for row in csv_rows if clean_text(row.get("sample_group", "")) == "csv_sepsis"]
    csv_non_rows = [row for row in csv_rows if clean_text(row.get("sample_group", "")) == "csv_non_sepsis"]
    json_by_bucket: Dict[str, Dict[str, Any]] = {}
    for row in json_rows:
        key = clean_text(row.get("sample_group", ""))
        json_by_bucket.setdefault(key, {"rows": []})
        json_by_bucket[key]["rows"].append(row)

    json_bucket_summary = {key: summarize_rows(value["rows"]) for key, value in sorted(json_by_bucket.items())}
    return {
        "policy": "insufficient_evidence_as_non_sepsis",
        "csv_sepsis": summarize_rows(csv_sepsis_rows),
        "csv_non_sepsis": summarize_rows(csv_non_rows),
        "json_overall": summarize_rows(json_rows),
        "json_by_bucket": json_bucket_summary,
        "overall_all_samples": summarize_rows(list(csv_rows) + list(json_rows)),
    }


def write_summary_files(output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_SUMMARY_JSON).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (output_dir / DEFAULT_SUMMARY_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["group", "total", "correct_strict", "accuracy_strict", "score_soft_sum", "accuracy_soft", "abstain_count", "abstain_rate"],
        )
        writer.writeheader()

        def write_group(group_name: str, metrics: Dict[str, Any]) -> None:
            writer.writerow({"group": group_name, **metrics})

        write_group("csv_sepsis", summary["csv_sepsis"])
        write_group("csv_non_sepsis", summary["csv_non_sepsis"])
        write_group("json_overall", summary["json_overall"])
        for key, metrics in sorted(summary["json_by_bucket"].items()):
            write_group(key, metrics)
        write_group("overall_all_samples", summary["overall_all_samples"])


def evaluate_agent_on_testset(
    *,
    agent: Any,
    csv_sepsis_path: Path = DEFAULT_CSV_SEPSIS,
    csv_non_sepsis_path: Path = DEFAULT_CSV_NON_SEPSIS,
    json_manifest_path: Path = DEFAULT_JSON_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "strict_eval",
    user_query: str = DEFAULT_QUERY,
    limit: Optional[int] = None,
    resume: bool = True,
    model_profile: str = "default",
) -> Dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_results_path = output_dir / DEFAULT_RESULTS_CSV
    json_results_path = output_dir / DEFAULT_RESULTS_JSON
    run_log_path = output_dir / DEFAULT_RUN_LOG

    resume_csv_ids = _read_existing_case_ids(csv_results_path) if resume else set()
    resume_json_ids = _read_existing_case_ids(json_results_path) if resume else set()
    if not resume:
        for path in (csv_results_path, json_results_path, run_log_path):
            if path.exists():
                path.unlink()

    run_csv_evaluation(
        agent=agent,
        csv_path=csv_sepsis_path,
        ground_truth="sepsis",
        sample_group="csv_sepsis",
        mode=mode,
        user_query=user_query,
        results_path=csv_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_csv_ids,
        limit=limit,
        model_profile=model_profile,
    )
    run_csv_evaluation(
        agent=agent,
        csv_path=csv_non_sepsis_path,
        ground_truth="non_sepsis",
        sample_group="csv_non_sepsis",
        mode=mode,
        user_query=user_query,
        results_path=csv_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_csv_ids,
        limit=limit,
        model_profile=model_profile,
    )
    run_json_evaluation(
        agent=agent,
        manifest_path=json_manifest_path,
        user_query=user_query,
        results_path=json_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_json_ids,
        limit=limit,
        model_profile=model_profile,
    )

    csv_rows = _load_result_rows(csv_results_path)
    json_rows = _load_result_rows(json_results_path)
    summary = build_summary(csv_rows, json_rows)
    write_summary_files(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    agent = SepsisMedGemmaAgent(config)
    summary = evaluate_agent_on_testset(
        agent=agent,
        csv_sepsis_path=Path(args.csv_sepsis).expanduser().resolve(),
        csv_non_sepsis_path=Path(args.csv_non_sepsis).expanduser().resolve(),
        json_manifest_path=Path(args.json_manifest).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        mode=args.mode,
        user_query=DEFAULT_QUERY,
        limit=args.limit,
        resume=args.resume,
        model_profile=config.profile_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
