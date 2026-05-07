from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

from .ablation import build_ablation_config
from .agent import SUPPORTED_TASK_MODES, TASK_MODE_DIAGNOSIS, SepsisMedGemmaAgent
from .config import AgentConfig
from .llm_presets import (
    BASE_TEXT_PRESET_NAMES,
    MEDICAL_TREATMENT_PRESET_NAMES,
    normalize_allowed_text_backend_preset_name,
    resolve_text_backend_preset,
)
from .utils import build_case_payload_from_csv_row

DEFAULT_QUERY = "请根据病例，基于 Sepsis-3 标准判断患者是否存在脓毒症，并给出依据。"


def _parse_base_text_preset(value: str) -> str:
    return normalize_allowed_text_backend_preset_name(value, allowed_presets=BASE_TEXT_PRESET_NAMES)


def _parse_medical_treatment_preset(value: str) -> str:
    return normalize_allowed_text_backend_preset_name(value, allowed_presets=MEDICAL_TREATMENT_PRESET_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行脓毒症 Agent。")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--case-json", help="病例 JSON 路径。")
    input_group.add_argument("--case-csv", help="病例 CSV 路径。")
    parser.add_argument("--row-index", type=int, default=None, help="使用 --case-csv 时指定 0-based 行号。")
    parser.add_argument("--user-query", default=DEFAULT_QUERY)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=AgentConfig().case_input_mode)
    parser.add_argument("--task-mode", choices=sorted(SUPPORTED_TASK_MODES), default=TASK_MODE_DIAGNOSIS)
    parser.add_argument(
        "--base-text-preset",
        type=_parse_base_text_preset,
        default=None,
        help="统一覆盖通用文本基座，只允许 qwen_plus / deepseek / qwen_max。",
    )
    parser.add_argument(
        "--medical-treatment-preset",
        type=_parse_medical_treatment_preset,
        default=None,
        help="仅用于最终治疗建议生成的医学模型，只允许 baichuan / antangel。",
    )
    parser.add_argument(
        "--text-model",
        default=None,
        help="兼容旧接口：仅覆盖 default_text_backend 的模型名，例如 qwen-plus / qwen-max。",
    )
    parser.add_argument(
        "--vision-model",
        default=None,
        help="覆盖默认影像模型名，例如 qwen-vl-max-latest。",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--rag-top-k", type=int, default=None)
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument("--disable-imaging", action="store_true")
    parser.add_argument("--enable-generated-code", action="store_true")
    parser.add_argument("--enable-treatment-round-reminders", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> AgentConfig:
    config = build_ablation_config(args.profile, AgentConfig())
    config = replace(config, case_input_mode=args.mode)

    if args.base_text_preset is not None:
        base_backend = resolve_text_backend_preset(args.base_text_preset)
        config = replace(
            config,
            default_text_backend=base_backend,
            diagnosis_backend=base_backend,
            code_writer_backend=base_backend,
            reflection_backend=base_backend,
            treatment_backend=base_backend,
        )

    if args.medical_treatment_preset is not None:
        config = replace(
            config,
            medical_treatment_backend=resolve_text_backend_preset(args.medical_treatment_preset),
        )

    if args.text_model is not None:
        config = replace(
            config,
            default_text_backend=replace(config.default_text_backend, model=args.text_model),
        )
    if args.vision_model is not None:
        config = replace(
            config,
            default_vision_backend=replace(config.default_vision_backend, model=args.vision_model),
        )
    if args.max_new_tokens is not None:
        config = replace(config, generation_max_new_tokens=args.max_new_tokens)
    if args.rag_top_k is not None:
        config = replace(config, rag_top_k=args.rag_top_k)
    if args.disable_rag:
        config = replace(config, enable_local_rag=False)
    if args.disable_imaging:
        config = replace(config, enable_imaging=False)
    if args.enable_generated_code:
        config = replace(config, enable_generated_code=True)
    if args.enable_treatment_round_reminders:
        config = replace(config, enable_treatment_round_reminders=True)
    if args.disable_reflection:
        config = replace(config, enable_reflection=False)
    return config


def load_csv_row(csv_path: Path, row_index: int) -> Dict[str, Any]:
    if row_index < 0:
        raise ValueError("--row-index 必须是非负整数。")

    import csv

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for current_index, row in enumerate(reader):
            if current_index == row_index:
                return dict(row)
    raise IndexError(f"CSV 行号超出范围: {row_index}")


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    agent = SepsisMedGemmaAgent(config)

    preserve_fields = ["西医入院诊断", "西医出院诊断"] if args.task_mode == "treatment" else None
    if args.case_json:
        result = agent.run(
            user_query=args.user_query,
            case_json_path=args.case_json,
            task_mode=args.task_mode,
        )
    else:
        if args.row_index is None:
            raise ValueError("使用 --case-csv 时必须提供 --row-index。")
        row = load_csv_row(Path(args.case_csv), args.row_index)
        payload = build_case_payload_from_csv_row(
            row,
            mode=args.mode,
            preserve_fields=preserve_fields,
        )
        result = agent.run(
            user_query=args.user_query,
            case_payload=payload,
            task_mode=args.task_mode,
        )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
