from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.ablation import build_ablation_config
from Agent.agent import SepsisMedGemmaAgent
from Agent.config import AgentConfig
from Agent.llm_presets import list_text_backend_presets, normalize_text_backend_preset_name, resolve_text_backend_preset
from Agent.utils import clean_text
try:
    from .evaluate_agent_on_testset import evaluate_agent_on_testset
    from .evaluate_direct_llm_baseline import (
        DEFAULT_CSV_NON_SEPSIS,
        DEFAULT_CSV_SEPSIS,
        DEFAULT_JSON_MANIFEST,
        evaluate_direct_llm_baseline,
    )
except ImportError:
    from evaluate_agent_on_testset import evaluate_agent_on_testset
    from evaluate_direct_llm_baseline import (
        DEFAULT_CSV_NON_SEPSIS,
        DEFAULT_CSV_SEPSIS,
        DEFAULT_JSON_MANIFEST,
        evaluate_direct_llm_baseline,
    )


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "miniTest" / "llm_experiments"
MATRIX_SUMMARY_JSON = "matrix_summary.json"
MATRIX_SUMMARY_CSV = "matrix_summary.csv"


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_type: str
    model_preset: str

    @property
    def normalized_preset(self) -> str:
        return normalize_text_backend_preset_name(self.model_preset)

    @property
    def name(self) -> str:
        return f"{self.experiment_type}/{self.normalized_preset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 miniTest 上批量运行多 LLM Agent 与 direct baseline 实验。")
    parser.add_argument("--csv-sepsis", default=str(DEFAULT_CSV_SEPSIS))
    parser.add_argument("--csv-non-sepsis", default=str(DEFAULT_CSV_NON_SEPSIS))
    parser.add_argument("--json-manifest", default=str(DEFAULT_JSON_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--models",
        default="deepseek,qwen_plus,qwen_max,deepseek_cot,qwen_plus_cot,qwen_max_cot,baichuan,antangel",
    )
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--skip-agent-pipeline", action="store_true")
    parser.add_argument("--skip-direct-baseline", action="store_true")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def parse_model_presets(raw_models: str) -> List[str]:
    presets: List[str] = []
    for raw_item in raw_models.split(","):
        item = clean_text(raw_item)
        if not item:
            continue
        normalized = normalize_text_backend_preset_name(item)
        if normalized not in presets:
            presets.append(normalized)
    if not presets:
        available = ", ".join(sorted(list_text_backend_presets()))
        raise ValueError(f"至少需要一个有效模型 preset；可选值: {available}")
    return presets


def build_experiment_specs(*, model_presets: Iterable[str], include_agent_pipeline: bool, include_direct_baseline: bool) -> List[ExperimentSpec]:
    specs: List[ExperimentSpec] = []
    if include_direct_baseline:
        for preset in model_presets:
            specs.append(ExperimentSpec(experiment_type="direct_baseline", model_preset=preset))
    if include_agent_pipeline:
        for preset in model_presets:
            specs.append(ExperimentSpec(experiment_type="agent_pipeline", model_preset=preset))
    return specs


def build_agent_config_for_preset(*, model_preset: str, mode: str, max_new_tokens: Optional[int]) -> AgentConfig:
    backend_config = resolve_text_backend_preset(model_preset)
    config = build_ablation_config("default", AgentConfig())
    resolved_generation_max_new_tokens = max_new_tokens
    if resolved_generation_max_new_tokens is None:
        resolved_generation_max_new_tokens = backend_config.default_max_new_tokens or config.generation_max_new_tokens
    config = replace(
        config,
        profile_name=f"agent_pipeline/{normalize_text_backend_preset_name(model_preset)}",
        case_input_mode=mode,
        diagnosis_backend=backend_config,
        generation_max_new_tokens=resolved_generation_max_new_tokens,
        http_timeout_seconds=max(config.http_timeout_seconds, float(backend_config.default_timeout_seconds or 0.0)),
        sepsis3_extraction_max_new_tokens=max(
            config.sepsis3_extraction_max_new_tokens,
            int(backend_config.sepsis3_extraction_max_new_tokens or 0),
        ),
        sepsis3_extraction_timeout_seconds=max(
            config.sepsis3_extraction_timeout_seconds,
            float(backend_config.sepsis3_extraction_timeout_seconds or 0.0),
        ),
    )
    return config


def summarize_experiment_row(
    *,
    spec: ExperimentSpec,
    output_dir: Path,
    summary: Optional[Dict[str, Any]],
    status: str,
    error: str,
) -> Dict[str, Any]:
    overall = (summary or {}).get("overall_all_samples", {})
    return {
        "experiment_name": spec.name,
        "experiment_type": spec.experiment_type,
        "model_preset": spec.normalized_preset,
        "status": status,
        "error": clean_text(error),
        "output_dir": str(output_dir),
        "total": overall.get("total", ""),
        "correct_strict": overall.get("correct_strict", ""),
        "accuracy_strict": overall.get("accuracy_strict", ""),
        "score_soft_sum": overall.get("score_soft_sum", ""),
        "accuracy_soft": overall.get("accuracy_soft", ""),
        "abstain_count": overall.get("abstain_count", ""),
        "abstain_rate": overall.get("abstain_rate", ""),
    }


def write_matrix_summary(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {"experiments": rows}
    (output_dir / MATRIX_SUMMARY_JSON).write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / MATRIX_SUMMARY_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "experiment_name",
            "experiment_type",
            "model_preset",
            "status",
            "error",
            "output_dir",
            "total",
            "correct_strict",
            "accuracy_strict",
            "score_soft_sum",
            "accuracy_soft",
            "abstain_count",
            "abstain_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_single_experiment(
    *,
    spec: ExperimentSpec,
    csv_sepsis_path: Path,
    csv_non_sepsis_path: Path,
    json_manifest_path: Path,
    output_root: Path,
    mode: str,
    limit: Optional[int],
    resume: bool,
    max_new_tokens: Optional[int],
) -> Dict[str, Any]:
    output_dir = output_root / spec.experiment_type / spec.normalized_preset
    try:
        if spec.experiment_type == "agent_pipeline":
            config = build_agent_config_for_preset(model_preset=spec.model_preset, mode=mode, max_new_tokens=max_new_tokens)
            agent = SepsisMedGemmaAgent(config)
            summary = evaluate_agent_on_testset(
                agent=agent,
                csv_sepsis_path=csv_sepsis_path,
                csv_non_sepsis_path=csv_non_sepsis_path,
                json_manifest_path=json_manifest_path,
                output_dir=output_dir,
                mode=mode,
                limit=limit,
                resume=resume,
                model_profile=config.profile_name,
            )
        elif spec.experiment_type == "direct_baseline":
            summary = evaluate_direct_llm_baseline(
                model_preset=spec.model_preset,
                csv_sepsis_path=csv_sepsis_path,
                csv_non_sepsis_path=csv_non_sepsis_path,
                json_manifest_path=json_manifest_path,
                output_dir=output_dir,
                mode=mode,
                limit=limit,
                resume=resume,
                max_new_tokens=max_new_tokens,
                model_profile=f"direct_baseline/{spec.normalized_preset}",
            )
        else:
            raise ValueError(f"未知 experiment_type: {spec.experiment_type}")
        return summarize_experiment_row(spec=spec, output_dir=output_dir, summary=summary, status="success", error="")
    except Exception as exc:
        return summarize_experiment_row(spec=spec, output_dir=output_dir, summary=None, status="failed", error=str(exc))


def run_minitest_llm_matrix(
    *,
    csv_sepsis_path: Path,
    csv_non_sepsis_path: Path,
    json_manifest_path: Path,
    output_root: Path,
    model_presets: List[str],
    mode: str,
    limit: Optional[int],
    resume: bool,
    max_new_tokens: Optional[int],
    include_agent_pipeline: bool,
    include_direct_baseline: bool,
) -> List[Dict[str, Any]]:
    specs = build_experiment_specs(
        model_presets=model_presets,
        include_agent_pipeline=include_agent_pipeline,
        include_direct_baseline=include_direct_baseline,
    )
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        rows.append(
            run_single_experiment(
                spec=spec,
                csv_sepsis_path=csv_sepsis_path,
                csv_non_sepsis_path=csv_non_sepsis_path,
                json_manifest_path=json_manifest_path,
                output_root=output_root,
                mode=mode,
                limit=limit,
                resume=resume,
                max_new_tokens=max_new_tokens,
            )
        )
    write_matrix_summary(output_root, rows)
    return rows


def main() -> None:
    args = parse_args()
    run_id = clean_text(args.run_id) or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve() / run_id
    model_presets = parse_model_presets(args.models)
    if args.skip_agent_pipeline and args.skip_direct_baseline:
        raise ValueError("不能同时跳过 agent pipeline 和 direct baseline。")
    rows = run_minitest_llm_matrix(
        csv_sepsis_path=Path(args.csv_sepsis).expanduser().resolve(),
        csv_non_sepsis_path=Path(args.csv_non_sepsis).expanduser().resolve(),
        json_manifest_path=Path(args.json_manifest).expanduser().resolve(),
        output_root=output_root,
        model_presets=model_presets,
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
        include_agent_pipeline=not args.skip_agent_pipeline,
        include_direct_baseline=not args.skip_direct_baseline,
    )
    print(json.dumps({"experiments": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
