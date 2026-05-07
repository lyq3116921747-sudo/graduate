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
from Agent.llm_presets import (
    normalize_allowed_text_backend_preset_name,
    resolve_text_backend_preset,
)
from Agent.utils import clean_text

try:
    from .evaluate_agent_on_testset import (
        DEFAULT_CSV_NON_SEPSIS,
        DEFAULT_CSV_SEPSIS,
        DEFAULT_JSON_MANIFEST,
        evaluate_agent_on_testset,
    )
except ImportError:
    from evaluate_agent_on_testset import (
        DEFAULT_CSV_NON_SEPSIS,
        DEFAULT_CSV_SEPSIS,
        DEFAULT_JSON_MANIFEST,
        evaluate_agent_on_testset,
    )


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "miniTest" / "diagnosis_ablations"
MATRIX_SUMMARY_JSON = "matrix_summary.json"
MATRIX_SUMMARY_CSV = "matrix_summary.csv"
ALLOWED_MODEL_PRESETS = ("deepseek", "antangel")
ALLOWED_GROUPS = ("full", "no_reflection", "no_traditional", "traditional_only_no_llm")


@dataclass(frozen=True)
class DiagnosisAblationSpec:
    group: str
    model_preset: Optional[str] = None

    @property
    def normalized_group(self) -> str:
        if self.group not in ALLOWED_GROUPS:
            available = ", ".join(ALLOWED_GROUPS)
            raise ValueError(f"Unknown diagnosis ablation group: {self.group}; available: {available}")
        return self.group

    @property
    def normalized_preset(self) -> str:
        if self.model_preset is None:
            return ""
        return normalize_allowed_text_backend_preset_name(self.model_preset, allowed_presets=ALLOWED_MODEL_PRESETS)

    @property
    def name(self) -> str:
        if self.normalized_group == "traditional_only_no_llm":
            return self.normalized_group
        return f"{self.normalized_preset}/{self.normalized_group}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnosis-only ablation experiments on miniTest.")
    parser.add_argument("--csv-sepsis", default=str(DEFAULT_CSV_SEPSIS))
    parser.add_argument("--csv-non-sepsis", default=str(DEFAULT_CSV_NON_SEPSIS))
    parser.add_argument("--json-manifest", default=str(DEFAULT_JSON_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--models", default="deepseek,antangel")
    parser.add_argument("--groups", default="full,no_reflection,no_traditional,traditional_only_no_llm")
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def parse_model_presets(raw_models: str) -> List[str]:
    presets: List[str] = []
    for raw_item in raw_models.split(","):
        item = clean_text(raw_item)
        if not item:
            continue
        normalized = normalize_allowed_text_backend_preset_name(item, allowed_presets=ALLOWED_MODEL_PRESETS)
        if normalized not in presets:
            presets.append(normalized)
    if not presets:
        available = ", ".join(ALLOWED_MODEL_PRESETS)
        raise ValueError(f"At least one diagnosis ablation model is required; available: {available}")
    return presets


def parse_groups(raw_groups: str) -> List[str]:
    groups: List[str] = []
    for raw_item in raw_groups.split(","):
        item = clean_text(raw_item)
        if not item:
            continue
        if item not in ALLOWED_GROUPS:
            available = ", ".join(ALLOWED_GROUPS)
            raise ValueError(f"Unknown diagnosis ablation group: {item}; available: {available}")
        if item not in groups:
            groups.append(item)
    if not groups:
        available = ", ".join(ALLOWED_GROUPS)
        raise ValueError(f"At least one diagnosis ablation group is required; available: {available}")
    return groups


def build_experiment_specs(*, model_presets: Iterable[str], groups: Iterable[str]) -> List[DiagnosisAblationSpec]:
    specs: List[DiagnosisAblationSpec] = []
    normalized_groups = [group for group in groups if group != "traditional_only_no_llm"]
    include_traditional_only = "traditional_only_no_llm" in set(groups)
    for preset in model_presets:
        normalized_preset = normalize_allowed_text_backend_preset_name(preset, allowed_presets=ALLOWED_MODEL_PRESETS)
        for group in normalized_groups:
            specs.append(DiagnosisAblationSpec(group=group, model_preset=normalized_preset))
    if include_traditional_only:
        specs.append(DiagnosisAblationSpec(group="traditional_only_no_llm", model_preset=None))
    return specs


def build_agent_config_for_diagnosis_ablation(
    *,
    model_preset: Optional[str],
    group: str,
    mode: str,
    max_new_tokens: Optional[int],
) -> AgentConfig:
    profile_name = {
        "full": "default",
        "no_reflection": "no_reflection",
        "no_traditional": "no_traditional",
        "traditional_only_no_llm": "traditional_only_no_llm",
    }[group]
    config = build_ablation_config(profile_name, AgentConfig())
    config = replace(config, case_input_mode=mode)

    if model_preset is None:
        return replace(
            config,
            profile_name=f"diagnosis_ablation/{group}",
        )

    backend_config = resolve_text_backend_preset(model_preset)
    resolved_generation_max_new_tokens = max_new_tokens
    if resolved_generation_max_new_tokens is None:
        resolved_generation_max_new_tokens = backend_config.default_max_new_tokens or config.generation_max_new_tokens
    return replace(
        config,
        profile_name=f"diagnosis_ablation/{normalize_allowed_text_backend_preset_name(model_preset, allowed_presets=ALLOWED_MODEL_PRESETS)}/{group}",
        default_text_backend=backend_config,
        diagnosis_backend=backend_config,
        generation_max_new_tokens=int(resolved_generation_max_new_tokens),
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


def summarize_experiment_row(
    *,
    spec: DiagnosisAblationSpec,
    output_dir: Path,
    summary: Optional[Dict[str, Any]],
    status: str,
    error: str,
) -> Dict[str, Any]:
    overall = (summary or {}).get("overall_all_samples", {})
    return {
        "experiment_name": spec.name,
        "group": spec.normalized_group,
        "model_preset": spec.normalized_preset or "traditional",
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
            "group",
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


def _output_dir_for_spec(output_root: Path, spec: DiagnosisAblationSpec) -> Path:
    if spec.normalized_group == "traditional_only_no_llm":
        return output_root / spec.normalized_group
    return output_root / spec.normalized_preset / spec.normalized_group


def run_single_experiment(
    *,
    spec: DiagnosisAblationSpec,
    csv_sepsis_path: Path,
    csv_non_sepsis_path: Path,
    json_manifest_path: Path,
    output_root: Path,
    mode: str,
    limit: Optional[int],
    resume: bool,
    max_new_tokens: Optional[int],
) -> Dict[str, Any]:
    output_dir = _output_dir_for_spec(output_root, spec)
    try:
        config = build_agent_config_for_diagnosis_ablation(
            model_preset=spec.model_preset,
            group=spec.normalized_group,
            mode=mode,
            max_new_tokens=max_new_tokens,
        )
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
        return summarize_experiment_row(spec=spec, output_dir=output_dir, summary=summary, status="success", error="")
    except Exception as exc:
        return summarize_experiment_row(spec=spec, output_dir=output_dir, summary=None, status="failed", error=str(exc))


def run_diagnosis_ablation_matrix(
    *,
    csv_sepsis_path: Path,
    csv_non_sepsis_path: Path,
    json_manifest_path: Path,
    output_root: Path,
    model_presets: List[str],
    groups: List[str],
    mode: str,
    limit: Optional[int],
    resume: bool,
    max_new_tokens: Optional[int],
) -> List[Dict[str, Any]]:
    specs = build_experiment_specs(model_presets=model_presets, groups=groups)
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
    rows = run_diagnosis_ablation_matrix(
        csv_sepsis_path=Path(args.csv_sepsis).expanduser().resolve(),
        csv_non_sepsis_path=Path(args.csv_non_sepsis).expanduser().resolve(),
        json_manifest_path=Path(args.json_manifest).expanduser().resolve(),
        output_root=output_root,
        model_presets=parse_model_presets(args.models),
        groups=parse_groups(args.groups),
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps({"experiments": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
