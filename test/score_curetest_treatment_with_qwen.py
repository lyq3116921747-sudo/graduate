from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.config import AgentConfig
from Agent.utils import (
    build_case_payload_from_csv_row,
    case_to_context,
    clean_text,
    extract_json_object,
    load_case_payload,
    truncate_text,
)


DEFAULT_OUTPUT_DIRNAME = "general_treatment_scores"
DEFAULT_JUDGES = "gemini_flash,gpt_mini"
DEFAULT_JUDGE_MAX_NEW_TOKENS = 1024
DEFAULT_CASE_MODE = AgentConfig().case_input_mode

# Do not load API keys from Python example files in the submission version.
# Secrets must be supplied through environment variables, for example:
#   TREATMENT_JUDGE_API_KEY / TREATMENT_JUDGE_BASE_URL
# or OPENAI_API_KEY / OPENAI_BASE_URL.
HEADER_JSON_ENV_CANDIDATES = (
    "TREATMENT_JUDGE_HEADERS_JSON",
    "OPENAI_DEFAULT_HEADERS_JSON",
)


LEGACY_ENV_ALIASES = {
    "QW_KEY": "DASHSCOPE_API_KEY",
    "AntAngel_KEY": "ANTANGEL_API_KEY",
    "BAI_CHUAN_KEY": "BAICHUAN_API_KEY",
    "DeepSeek_KEY": "DEEPSEEK_API_KEY",
    "OPEN_AI_KEY": "OPENAI_API_KEY",
}

API_KEY_ENV_CANDIDATES = (
    "TREATMENT_JUDGE_API_KEY",
    "OPENAI_API_KEY",
    "CHAT_CLOUDAPI_API_KEY",
    "CLOUDAPI_API_KEY",
)

BASE_URL_ENV_CANDIDATES = (
    "TREATMENT_JUDGE_BASE_URL",
    "OPENAI_BASE_URL",
)

EXPLICIT_JUDGE_API_KEY_ENV_CANDIDATES = ("TREATMENT_JUDGE_API_KEY",)
EXPLICIT_JUDGE_BASE_URL_ENV_CANDIDATES = ("TREATMENT_JUDGE_BASE_URL",)
FALLBACK_API_KEY_ENV_CANDIDATES = tuple(
    env_name for env_name in API_KEY_ENV_CANDIDATES if env_name not in EXPLICIT_JUDGE_API_KEY_ENV_CANDIDATES
)
FALLBACK_BASE_URL_ENV_CANDIDATES = tuple(
    env_name for env_name in BASE_URL_ENV_CANDIDATES if env_name not in EXPLICIT_JUDGE_BASE_URL_ENV_CANDIDATES
)

DEFAULT_MODEL_PRESET_RESULT_OVERRIDES = {
    "qwen_plus": PROJECT_ROOT
    / "miniTest"
    / "curetest_treatment_matrix"
    / "curetest_fullstack_qwen_plus_round_reminder"
    / "results.csv",
}

SCORE_DIMENSIONS = (
    "clinical_reasonableness",
    "safety",
    "evidence_grounding",
    "completeness",
    "actionability",
)

DIMENSION_LABELS = {
    "clinical_reasonableness": "临床合理性",
    "safety": "安全性",
    "evidence_grounding": "证据依据性",
    "completeness": "完整性",
    "actionability": "可执行性",
}

DIMENSION_GUIDANCE = {
    "clinical_reasonableness": "治疗方向是否基本正确，是否符合常规临床逻辑，是否存在明显不恰当建议。",
    "safety": "是否存在潜在危险建议，是否忽略禁忌、延误救治、过度治疗或明显风险。",
    "evidence_grounding": "是否基于病例中已有证据提出建议，是否避免虚构病情、药物、检查结果或指南结论。",
    "completeness": "是否覆盖主要处理环节，包括病因处理、支持治疗、监测复评和必要后续计划。",
    "actionability": "建议是否具体、清晰、可执行，而不是停留在空泛表述。",
}

JUDGE_ALIASES = {
    "gemini_flash": "gemini_flash",
    "gemini-flash": "gemini_flash",
    "gemini_2_5_flash": "gemini_flash",
    "gemini-2.5-flash": "gemini_flash",
    "gemini25flash": "gemini_flash",
    "gpt_mini": "gpt_mini",
    "gpt-mini": "gpt_mini",
    "gpt5mini": "gpt_mini",
    "gpt_5_mini": "gpt_mini",
    "gpt-5-mini": "gpt_mini",
}

JUDGE_SPECS = {
    "gemini_flash": {
        "display_name": "Gemini Flash",
        "model": "gemini-2.5-flash",
    },
    "gpt_mini": {
        "display_name": "GPT-mini",
        "model": "gpt-5-mini",
    },
}

SCORES_CSV_FIELDS = [
    "result_key",
    "judge_model",
    "judge_backend",
    "judge_status",
    "attempt_count",
    "parsed_on_attempt",
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
    "result_status",
    "result_error",
    "treatment_react_status",
    "tool_call_counts",
    "answer_is_empty",
    "answer_is_artifact",
    "case_context_chars",
    "answer_chars",
    *SCORE_DIMENSIONS,
    "total_score",
    "percentage_score",
    "overall_grade",
    "strengths_json",
    "weaknesses_json",
    "safety_concerns_json",
    "short_comment",
    "judge_error",
]

SUMMARY_FIELDS = [
    "input_scope",
    "judge_model",
    "model_preset",
    "profile_name",
    "architecture",
    "base_text_preset",
    "medical_treatment_preset",
    "treatment_generation_mode",
    "input_type",
    "total_cases",
    "scored_cases",
    "parse_error_count",
    "backend_error_count",
    "case_load_error_count",
    "artifact_answer_count",
    *[f"mean_{dimension}" for dimension in SCORE_DIMENSIONS],
    "mean_total_score",
    "mean_percentage_score",
]


@dataclass(frozen=True)
class JudgeSpec:
    normalized_name: str
    display_name: str
    model_name: str


@dataclass
class ScoringResult:
    judge_status: str
    scores: Dict[str, float]
    total_score: float
    percentage_score: float
    overall_grade: str
    strengths: List[str]
    weaknesses: List[str]
    safety_concerns: List[str]
    short_comment: str
    attempts: List[Dict[str, Any]]
    judge_error: str
    parsed_on_attempt: int


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def load_project_env_file(project_root: Path = PROJECT_ROOT) -> None:
    env_file = (project_root / ".env").resolve()
    if not env_file.exists():
        return

    parsed_values: Dict[str, str] = {}
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw_value = line.partition("=")
        normalized_key = key.strip()
        if not normalized_key:
            continue
        parsed_values[normalized_key] = _parse_env_value(raw_value)

    for key, value in parsed_values.items():
        if value and not clean_text(os.getenv(key, "")):
            os.environ[key] = value

    for legacy_key, standard_key in LEGACY_ENV_ALIASES.items():
        if clean_text(os.getenv(standard_key, "")):
            continue
        alias_value = clean_text(os.getenv(legacy_key, "")) or clean_text(parsed_values.get(legacy_key, ""))
        if alias_value:
            os.environ[standard_key] = alias_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 Gemini Flash 与 GPT-mini 作为评审器，对 CureTest 治疗建议进行结构化评分。"
    )
    parser.add_argument("--results-csv", required=True, help="待评分的治疗实验 results.csv 路径。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="评分输出目录；默认在 results.csv 同级创建 general_treatment_scores。",
    )
    parser.add_argument(
        "--judges",
        default=DEFAULT_JUDGES,
        help="评审模型列表，逗号分隔；默认 gemini_flash,gpt_mini。",
    )
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default=DEFAULT_CASE_MODE)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_JUDGE_MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="仅评分前 N 条样本，便于调试。")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def normalize_judge_name(name: str) -> str:
    normalized = clean_text(name).lower().replace(" ", "_")
    normalized = JUDGE_ALIASES.get(normalized, normalized)
    if normalized not in JUDGE_SPECS:
        raise ValueError(f"Unsupported judge model: {name}")
    return normalized


def parse_judge_names(raw_judges: str) -> List[str]:
    judges: List[str] = []
    for item in raw_judges.split(","):
        name = clean_text(item)
        if not name:
            continue
        normalized = normalize_judge_name(name)
        if normalized not in judges:
            judges.append(normalized)
    if not judges:
        raise ValueError("At least one judge model is required.")
    return judges


def build_judge_spec(judge_name: str) -> JudgeSpec:
    normalized = normalize_judge_name(judge_name)
    payload = JUDGE_SPECS[normalized]
    return JudgeSpec(
        normalized_name=normalized,
        display_name=str(payload["display_name"]),
        model_name=str(payload["model"]),
    )


def read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_read_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    return read_csv_rows(csv_path)


def build_case_identity(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        clean_text(row.get("case_identifier", "")),
        clean_text(row.get("source_path", "")),
        clean_text(row.get("row_index", "")),
    )


def apply_model_result_overrides(source_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not source_rows:
        return []

    overrides_cache: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {}
    for model_preset, override_path in DEFAULT_MODEL_PRESET_RESULT_OVERRIDES.items():
        resolved = Path(override_path).expanduser().resolve()
        if not resolved.exists():
            continue
        override_rows = read_csv_rows(resolved)
        overrides_cache[model_preset] = {
            build_case_identity(row): dict(row)
            for row in override_rows
            if clean_text(row.get("model_preset", "")) == model_preset
        }

    if not overrides_cache:
        return [dict(row) for row in source_rows]

    merged_rows: List[Dict[str, Any]] = []
    for row in source_rows:
        model_preset = clean_text(row.get("model_preset", ""))
        override_map = overrides_cache.get(model_preset)
        if not override_map:
            merged_rows.append(dict(row))
            continue
        override_row = override_map.get(build_case_identity(row))
        merged_rows.append(dict(override_row) if override_row is not None else dict(row))
    return merged_rows


def resolve_output_dir(results_csv: Path, output_dir: Optional[str]) -> Path:
    if clean_text(output_dir):
        return Path(str(output_dir)).expanduser().resolve()
    return (results_csv.parent / DEFAULT_OUTPUT_DIRNAME).resolve()


def append_csv_row(csv_path: Path, fieldnames: Sequence[str], row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_jsonl(jsonl_path: Path, payload: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_existing_result_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {clean_text(row.get("result_key", "")) for row in csv.DictReader(handle) if clean_text(row.get("result_key", ""))}


def safe_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def reset_output_files(output_dir: Path) -> None:
    for filename in ("scores.csv", "scores.jsonl", "summary.csv", "summary.json"):
        path = output_dir / filename
        if path.exists():
            path.unlink()


def normalize_float(value: Any) -> float:
    text = clean_text(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except Exception:
        return 0.0


def normalize_int(value: Any) -> int:
    text = clean_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except Exception:
        return 0


def build_result_key(row: Dict[str, Any], judge_model: str) -> str:
    stable_payload = {
        "judge_model": normalize_judge_name(judge_model),
        "model_preset": clean_text(row.get("model_preset", "")),
        "profile_name": clean_text(row.get("profile_name", "")),
        "architecture": clean_text(row.get("architecture", "")),
        "base_text_preset": clean_text(row.get("base_text_preset", "")),
        "medical_treatment_preset": clean_text(row.get("medical_treatment_preset", "")),
        "treatment_generation_mode": clean_text(row.get("treatment_generation_mode", "")),
        "input_type": clean_text(row.get("input_type", "")),
        "case_identifier": clean_text(row.get("case_identifier", "")),
        "source_path": clean_text(row.get("source_path", "")),
        "row_index": clean_text(row.get("row_index", "")),
    }
    digest = hashlib.sha1(safe_json_dumps(stable_payload).encode("utf-8")).hexdigest()
    return f"{stable_payload['model_preset']}::{stable_payload['case_identifier']}::{stable_payload['judge_model']}::{digest[:12]}"


def normalize_string_list(value: Any, *, max_items: int = 6) -> List[str]:
    items: List[str] = []
    if isinstance(value, list):
        raw_items = value
    elif isinstance(value, str):
        raw_items = [part for part in value.replace("；", "\n").replace(";", "\n").splitlines()]
    elif value is None:
        raw_items = []
    else:
        raw_items = [str(value)]
    for item in raw_items:
        text = clean_text(item)
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def coerce_dimension_score(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = clean_text(value)
        if not text:
            return None
        try:
            score = float(text)
        except Exception:
            return None
    if not math.isfinite(score):
        return None
    if score < 0.0 or score > 5.0:
        return None
    return round(score, 3)


def derive_grade(percentage_score: float) -> str:
    if percentage_score >= 85.0:
        return "excellent"
    if percentage_score >= 70.0:
        return "good"
    if percentage_score >= 50.0:
        return "fair"
    return "poor"


def normalize_grade(value: Any, percentage_score: float) -> str:
    normalized = clean_text(value).lower().replace(" ", "_")
    if normalized in {"poor", "fair", "good", "excellent"}:
        return normalized
    return derive_grade(percentage_score)


def extract_json_object_with_repair(text: str) -> Optional[Dict[str, Any]]:
    normalized = clean_text(text)
    if not normalized:
        return None
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[a-zA-Z0-9_]*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()

    start = normalized.find("{")
    if start < 0:
        return None

    raw_candidate = normalized[start:].strip()
    candidates: List[str] = [raw_candidate]

    repaired = raw_candidate
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    brace_diff = repaired.count("{") - repaired.count("}")
    if brace_diff > 0:
        repaired += "}" * brace_diff
    bracket_diff = repaired.count("[") - repaired.count("]")
    if bracket_diff > 0:
        repaired += "]" * bracket_diff
    if repaired not in candidates:
        candidates.append(repaired)

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                payload = parser(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
    return None


def parse_scoring_payload(raw_text: str) -> Optional[Dict[str, Any]]:
    payload = extract_json_object_with_repair(raw_text)
    if not isinstance(payload, dict):
        return None
    raw_scores = payload.get("scores", {})

    scores: Dict[str, float] = {}
    if isinstance(raw_scores, dict):
        for dimension in SCORE_DIMENSIONS:
            coerced = coerce_dimension_score(raw_scores.get(dimension))
            if coerced is None:
                return None
            scores[dimension] = coerced
    elif isinstance(raw_scores, (list, tuple)) and len(raw_scores) == len(SCORE_DIMENSIONS):
        for dimension, raw_value in zip(SCORE_DIMENSIONS, raw_scores):
            coerced = coerce_dimension_score(raw_value)
            if coerced is None:
                return None
            scores[dimension] = coerced
    else:
        return None

    total_score = round(sum(scores.values()), 3)
    percentage_score = round((total_score / (5.0 * len(SCORE_DIMENSIONS))) * 100.0, 3)
    normalized_payload = {
        "scores": scores,
        "total_score": total_score,
        "percentage_score": percentage_score,
        "overall_grade": normalize_grade(payload.get("overall_grade", ""), percentage_score),
        "strengths": normalize_string_list(payload.get("strengths", [])),
        "weaknesses": normalize_string_list(payload.get("weaknesses", [])),
        "safety_concerns": normalize_string_list(payload.get("safety_concerns", [])),
        "short_comment": clean_text(payload.get("short_comment", "")),
    }
    return normalized_payload


def looks_like_tool_artifact(answer_text: str) -> bool:
    normalized = clean_text(answer_text)
    if not normalized:
        return False
    artifact_markers = (
        '"query":',
        '"matches":',
        '"tool_name":',
        '"tool_calls":',
        '"arguments":',
        "search_local_guidelines",
        "run_sofa_assessment",
        "read_case_sections",
    )
    if normalized.startswith("{") and any(marker in normalized for marker in artifact_markers):
        return True
    if normalized.startswith("治疗建议：{") and any(marker in normalized for marker in artifact_markers):
        return True
    if normalized.startswith("[") and any(marker in normalized for marker in artifact_markers):
        return True
    return False


def build_answer_meta(row: Dict[str, Any], answer_text: str) -> str:
    result_status = clean_text(row.get("status", "")) or "unknown"
    react_status = clean_text(row.get("treatment_react_status", ""))
    error_text = clean_text(row.get("error", ""))
    tool_counts = clean_text(row.get("tool_call_counts", ""))
    flags: List[str] = [f"结果状态={result_status}"]
    if react_status:
        flags.append(f"React状态={react_status}")
    if error_text:
        flags.append(f"错误={truncate_text(error_text, 220)}")
    if tool_counts:
        flags.append(f"工具调用统计={truncate_text(tool_counts, 220)}")
    if not clean_text(answer_text):
        flags.append("候选答案为空")
    if looks_like_tool_artifact(answer_text):
        flags.append("候选答案疑似工具调用残片")
    return "；".join(flags)


class CaseContextLoader:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self._csv_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._context_cache: Dict[Tuple[str, str, str], str] = {}

    def _read_csv_rows(self, csv_path: Path) -> List[Dict[str, Any]]:
        cache_key = str(csv_path.resolve())
        if cache_key not in self._csv_cache:
            self._csv_cache[cache_key] = read_csv_rows(csv_path)
        return self._csv_cache[cache_key]

    def load_case_context(self, row: Dict[str, Any]) -> str:
        input_type = clean_text(row.get("input_type", "")).lower()
        source_path_text = clean_text(row.get("source_path", ""))
        row_index_text = clean_text(row.get("row_index", ""))
        cache_key = (input_type, source_path_text, row_index_text)
        if cache_key in self._context_cache:
            return self._context_cache[cache_key]

        source_path = Path(source_path_text).expanduser()
        if input_type == "json" or source_path.suffix.lower() == ".json":
            payload = load_case_payload(source_path, mode=self.mode)
        else:
            rows = self._read_csv_rows(source_path)
            row_index = normalize_int(row_index_text)
            if row_index < 0 or row_index >= len(rows):
                raise IndexError(f"CSV row_index out of range: {row_index} for {source_path}")
            payload = build_case_payload_from_csv_row(dict(rows[row_index]), mode=self.mode)

        context = case_to_context(
            payload,
            max_field_chars=1400,
            lab_selection="default",
            task_mode="treatment",
        )
        self._context_cache[cache_key] = context
        return context


def _first_env_value(env_names: Sequence[str]) -> str:
    for env_name in env_names:
        value = clean_text(os.getenv(env_name, ""))
        if value:
            return value
    return ""


def _load_headers_from_env() -> Dict[str, str]:
    for env_name in HEADER_JSON_ENV_CANDIDATES:
        raw_value = clean_text(os.getenv(env_name, ""))
        if not raw_value:
            continue
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return {clean_text(k): clean_text(v) for k, v in parsed.items() if clean_text(k)}
    return {}


def load_reference_api_settings() -> Tuple[str, str, Dict[str, str]]:
    load_project_env_file()
    api_key = _first_env_value(EXPLICIT_JUDGE_API_KEY_ENV_CANDIDATES)
    base_url = _first_env_value(EXPLICIT_JUDGE_BASE_URL_ENV_CANDIDATES)
    headers = _load_headers_from_env()

    api_key = api_key or _first_env_value(FALLBACK_API_KEY_ENV_CANDIDATES)
    base_url = base_url or _first_env_value(FALLBACK_BASE_URL_ENV_CANDIDATES)

    if not api_key:
        raise RuntimeError(
            "Treatment judge API key not found. Please set TREATMENT_JUDGE_API_KEY or OPENAI_API_KEY; do not hard-code keys in source files."
        )
    if not base_url:
        raise RuntimeError(
            "Treatment judge base URL not found. Please set TREATMENT_JUDGE_BASE_URL or OPENAI_BASE_URL."
        )
    if not headers:
        # Compatibility header for proxy-style OpenAI endpoints; it is not a secret.
        headers = {"x-foo": "true"}
    return api_key, base_url, headers


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return clean_text(message_content)
    if isinstance(message_content, list):
        parts: List[str] = []
        for item in message_content:
            if isinstance(item, dict):
                text = clean_text(item.get("text", ""))
                if text:
                    parts.append(text)
                continue
            text = clean_text(getattr(item, "text", ""))
            if text:
                parts.append(text)
        return "\n".join(parts)
    return clean_text(message_content)


class GenericTreatmentScoringAgent:
    def __init__(
        self,
        *,
        judge_name: str,
        max_new_tokens: int,
    ) -> None:
        self.spec = build_judge_spec(judge_name)
        self.api_key, self.base_url, self.default_headers = load_reference_api_settings()
        self.request_timeout_seconds = max(90.0, float(AgentConfig().http_timeout_seconds))
        self.max_new_tokens = int(max_new_tokens or DEFAULT_JUDGE_MAX_NEW_TOKENS)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=self.default_headers,
            timeout=self.request_timeout_seconds,
        )
        self.judge_backend = f"openai_compatible::{self.spec.model_name}@{self.base_url}"

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def _build_user_prompt(self, *, case_context: str, answer_text: str, answer_meta: str) -> str:
        schema = {
            "scores": {dimension: 0 for dimension in SCORE_DIMENSIONS},
            "total_score": 0,
            "percentage_score": 0,
            "overall_grade": "poor|fair|good|excellent",
            "strengths": ["优点1", "优点2"],
            "weaknesses": ["不足1", "不足2"],
            "safety_concerns": ["安全问题1"],
            "short_comment": "一句话总评",
        }
        rubric_lines = [
            f"- {DIMENSION_LABELS[dimension]}（{dimension}）：0-5分。{DIMENSION_GUIDANCE[dimension]}"
            for dimension in SCORE_DIMENSIONS
        ]
        return (
            "请你作为严格的临床建议质量评审器，对候选治疗建议进行结构化评分。\n"
            "评分任务面向一般临床建议质量评价，而不是仅针对单一疾病场景。\n"
            "评分必须仅依据给定病例摘要与候选治疗建议，不得脑补病例中不存在的事实。\n"
            "如果候选答案为空、报错、明显是工具调用残片、或未形成真正可执行的医疗建议，应给予较低分，并在 weaknesses 或 safety_concerns 中写明原因。\n"
            "评分维度如下：\n"
            + "\n".join(rubric_lines)
            + "\n总分为 25 分，百分制为 total_score / 25 * 100。\n"
            "overall_grade 建议按照百分制划分：<50 poor，50-69 fair，70-84 good，>=85 excellent。\n"
            "请只输出一个 JSON 对象，不要输出 Markdown，不要解释评分过程。\n"
            f"JSON 模板如下：\n{safe_json_dumps(schema)}\n\n"
            f"病例摘要：\n{truncate_text(case_context, 5200)}\n\n"
            f"候选治疗建议元信息：\n{answer_meta}\n\n"
            f"候选治疗建议：\n{truncate_text(answer_text or '【空回答】', 4200)}"
        )

    def _build_compact_user_prompt(self, *, case_context: str, answer_text: str, answer_meta: str) -> str:
        compact_schema = {
            "scores": {
                "clinical_reasonableness": 0,
                "safety": 0,
                "evidence_grounding": 0,
                "completeness": 0,
                "actionability": 0,
            },
            "overall_grade": "poor|fair|good|excellent",
            "short_comment": "不超过30字",
        }
        return (
            "请输出极简 JSON，不要代码块，不要 Markdown，不要任何额外说明。\n"
            "每个分数字段只能是 0 到 5 的数字。\n"
            "如果建议内容不完整或不可执行，也要给出分数。\n"
            f"只允许输出这一种结构：\n{safe_json_dumps(compact_schema)}\n\n"
            f"病例摘要：\n{truncate_text(case_context, 2600)}\n\n"
            f"候选治疗建议元信息：\n{truncate_text(answer_meta, 400)}\n\n"
            f"候选治疗建议：\n{truncate_text(answer_text or '【空回答】', 1600)}"
        )

    def _build_score_only_prompt(self, *, case_context: str, answer_text: str, answer_meta: str) -> str:
        return (
            "请只输出一行 JSON，不要代码块，不要解释。\n"
            "JSON 必须只包含 scores、overall_grade、short_comment 三个字段。\n"
            "scores 必须是长度为 5 的数字数组，顺序固定为 clinical_reasonableness、safety、evidence_grounding、completeness、actionability。\n"
            "short_comment 不超过 20 个字。\n"
            "输出示例：{\"scores\":[3,2,4,3,2],\"overall_grade\":\"fair\",\"short_comment\":\"简短评价\"}\n"
            f"病例摘要：{truncate_text(case_context, 1800)}\n"
            f"候选治疗建议元信息：{truncate_text(answer_meta, 220)}\n"
            f"候选治疗建议：{truncate_text(answer_text or '【空回答】', 1000)}"
        )

    def _generate(self, *, system_prompt: str, user_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.spec.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=self.max_new_tokens,
        )
        if not completion.choices:
            return ""
        message = completion.choices[0].message
        return extract_message_text(getattr(message, "content", ""))

    def score(
        self,
        *,
        case_context: str,
        answer_text: str,
        answer_meta: str,
    ) -> ScoringResult:
        attempts: List[Dict[str, Any]] = []
        final_payload: Optional[Dict[str, Any]] = None
        judge_error = ""
        parsed_on_attempt = 0
        full_prompt = self._build_user_prompt(case_context=case_context, answer_text=answer_text, answer_meta=answer_meta)
        compact_prompt = self._build_compact_user_prompt(
            case_context=case_context,
            answer_text=answer_text,
            answer_meta=answer_meta,
        )
        score_only_prompt = self._build_score_only_prompt(
            case_context=case_context,
            answer_text=answer_text,
            answer_meta=answer_meta,
        )
        system_prompt = (
            "你是严格的临床治疗建议质量评审器。"
            "你必须返回一个合法 JSON 对象，并且严格依据病例证据评分。"
        )

        prompts = [full_prompt, compact_prompt, score_only_prompt]
        for attempt_index, base_prompt in enumerate(prompts):
            user_prompt = base_prompt
            if attempt_index > 0 and attempts:
                previous_raw = clean_text(attempts[-1].get("raw_text", ""))
                retry_prefix = (
                    "上一轮输出未能被解析为完整 JSON。\n"
                    "这一次请务必只输出一个完整 JSON 对象，不要代码块，不要解释，不要额外前后缀。\n\n"
                )
                if previous_raw:
                    user_prompt = retry_prefix + f"上一轮输出：\n{truncate_text(previous_raw, 800)}\n\n" + base_prompt
                else:
                    user_prompt = retry_prefix + base_prompt
            raw_text = ""
            error_text = ""
            try:
                raw_text = self._generate(system_prompt=system_prompt, user_prompt=user_prompt)
            except Exception as exc:
                error_text = clean_text(exc)

            parsed_payload = parse_scoring_payload(raw_text)
            attempts.append(
                {
                    "attempt_index": attempt_index + 1,
                    "raw_text": clean_text(raw_text),
                    "error": error_text,
                    "parsed": bool(parsed_payload),
                }
            )
            if parsed_payload is not None:
                final_payload = parsed_payload
                parsed_on_attempt = attempt_index + 1
                judge_error = ""
                break
            judge_error = error_text or ("empty_response" if not clean_text(raw_text) else "unparseable_response")

        if final_payload is None:
            judge_status = "backend_error" if any(clean_text(item.get("error", "")) for item in attempts) else "parse_error"
            return ScoringResult(
                judge_status=judge_status,
                scores={dimension: 0.0 for dimension in SCORE_DIMENSIONS},
                total_score=0.0,
                percentage_score=0.0,
                overall_grade="poor",
                strengths=[],
                weaknesses=[],
                safety_concerns=[],
                short_comment="",
                attempts=attempts,
                judge_error=judge_error,
                parsed_on_attempt=0,
            )

        return ScoringResult(
            judge_status="scored",
            scores=dict(final_payload["scores"]),
            total_score=float(final_payload["total_score"]),
            percentage_score=float(final_payload["percentage_score"]),
            overall_grade=clean_text(final_payload["overall_grade"]) or derive_grade(float(final_payload["percentage_score"])),
            strengths=list(final_payload["strengths"]),
            weaknesses=list(final_payload["weaknesses"]),
            safety_concerns=list(final_payload["safety_concerns"]),
            short_comment=clean_text(final_payload["short_comment"]),
            attempts=attempts,
            judge_error="",
            parsed_on_attempt=parsed_on_attempt,
        )


def build_score_csv_row(
    *,
    result_key: str,
    judge_name: str,
    judge_backend: str,
    source_row: Dict[str, Any],
    scoring_result: ScoringResult,
    case_context: str,
    answer_text: str,
    answer_is_artifact: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "result_key": result_key,
        "judge_model": normalize_judge_name(judge_name),
        "judge_backend": judge_backend,
        "judge_status": scoring_result.judge_status,
        "attempt_count": len(scoring_result.attempts),
        "parsed_on_attempt": scoring_result.parsed_on_attempt,
        "model_preset": clean_text(source_row.get("model_preset", "")),
        "profile_name": clean_text(source_row.get("profile_name", "")),
        "architecture": clean_text(source_row.get("architecture", "")),
        "base_text_preset": clean_text(source_row.get("base_text_preset", "")),
        "medical_treatment_preset": clean_text(source_row.get("medical_treatment_preset", "")),
        "treatment_generation_mode": clean_text(source_row.get("treatment_generation_mode", "")),
        "input_type": clean_text(source_row.get("input_type", "")),
        "case_identifier": clean_text(source_row.get("case_identifier", "")),
        "source_path": clean_text(source_row.get("source_path", "")),
        "row_index": clean_text(source_row.get("row_index", "")),
        "result_status": clean_text(source_row.get("status", "")),
        "result_error": clean_text(source_row.get("error", "")),
        "treatment_react_status": clean_text(source_row.get("treatment_react_status", "")),
        "tool_call_counts": clean_text(source_row.get("tool_call_counts", "")),
        "answer_is_empty": not bool(clean_text(answer_text)),
        "answer_is_artifact": answer_is_artifact,
        "case_context_chars": len(clean_text(case_context)),
        "answer_chars": len(clean_text(answer_text)),
        "total_score": round(scoring_result.total_score, 3),
        "percentage_score": round(scoring_result.percentage_score, 3),
        "overall_grade": clean_text(scoring_result.overall_grade),
        "strengths_json": safe_json_dumps(scoring_result.strengths),
        "weaknesses_json": safe_json_dumps(scoring_result.weaknesses),
        "safety_concerns_json": safe_json_dumps(scoring_result.safety_concerns),
        "short_comment": clean_text(scoring_result.short_comment),
        "judge_error": clean_text(scoring_result.judge_error),
    }
    for dimension in SCORE_DIMENSIONS:
        row[dimension] = round(scoring_result.scores.get(dimension, 0.0), 3)
    return row


def build_score_jsonl_record(
    *,
    score_row: Dict[str, Any],
    scoring_result: ScoringResult,
    case_context: str,
    answer_text: str,
) -> Dict[str, Any]:
    return {
        "meta": {
            key: score_row.get(key, "")
            for key in (
                "result_key",
                "judge_model",
                "judge_backend",
                "judge_status",
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
                "result_status",
                "result_error",
                "treatment_react_status",
                "tool_call_counts",
                "answer_is_empty",
                "answer_is_artifact",
            )
        },
        "score": {
            "scores": {dimension: score_row.get(dimension, 0.0) for dimension in SCORE_DIMENSIONS},
            "total_score": score_row.get("total_score", 0.0),
            "percentage_score": score_row.get("percentage_score", 0.0),
            "overall_grade": score_row.get("overall_grade", ""),
            "strengths": scoring_result.strengths,
            "weaknesses": scoring_result.weaknesses,
            "safety_concerns": scoring_result.safety_concerns,
            "short_comment": scoring_result.short_comment,
        },
        "attempts": scoring_result.attempts,
        "artifacts": {
            "case_context_preview": truncate_text(case_context, 1800),
            "answer_preview": truncate_text(answer_text, 1800),
        },
    }


def build_case_group_key(row: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        clean_text(row.get("model_preset", "")),
        clean_text(row.get("profile_name", "")),
        clean_text(row.get("architecture", "")),
        clean_text(row.get("base_text_preset", "")),
        clean_text(row.get("medical_treatment_preset", "")),
        clean_text(row.get("treatment_generation_mode", "")),
        clean_text(row.get("input_type", "")),
        clean_text(row.get("case_identifier", "")),
        clean_text(row.get("source_path", "")),
        clean_text(row.get("row_index", "")),
    )


def compute_summary_row(
    *,
    rows: Sequence[Dict[str, Any]],
    input_scope: str,
    input_type: str,
    judge_model: str,
) -> Dict[str, Any]:
    first_row = rows[0]
    total_cases = len(rows)
    scored_rows = [row for row in rows if clean_text(row.get("judge_status", "")) == "scored"]
    parse_error_count = sum(1 for row in rows if clean_text(row.get("judge_status", "")) == "parse_error")
    backend_error_count = sum(1 for row in rows if clean_text(row.get("judge_status", "")) == "backend_error")
    case_load_error_count = sum(1 for row in rows if clean_text(row.get("judge_status", "")) == "case_load_error")
    artifact_answer_count = sum(1 for row in rows if clean_text(row.get("answer_is_artifact", "")).lower() in {"true", "1"})

    summary: Dict[str, Any] = {
        "input_scope": input_scope,
        "judge_model": judge_model,
        "model_preset": clean_text(first_row.get("model_preset", "")),
        "profile_name": clean_text(first_row.get("profile_name", "")),
        "architecture": clean_text(first_row.get("architecture", "")),
        "base_text_preset": clean_text(first_row.get("base_text_preset", "")),
        "medical_treatment_preset": clean_text(first_row.get("medical_treatment_preset", "")),
        "treatment_generation_mode": clean_text(first_row.get("treatment_generation_mode", "")),
        "input_type": input_type,
        "total_cases": total_cases,
        "scored_cases": len(scored_rows),
        "parse_error_count": parse_error_count,
        "backend_error_count": backend_error_count,
        "case_load_error_count": case_load_error_count,
        "artifact_answer_count": artifact_answer_count,
    }
    for dimension in SCORE_DIMENSIONS:
        values = [normalize_float(row.get(dimension, 0.0)) for row in scored_rows]
        summary[f"mean_{dimension}"] = round(sum(values) / len(values), 4) if values else ""
    total_scores = [normalize_float(row.get("total_score", 0.0)) for row in scored_rows]
    percentage_scores = [normalize_float(row.get("percentage_score", 0.0)) for row in scored_rows]
    summary["mean_total_score"] = round(sum(total_scores) / len(total_scores), 4) if total_scores else ""
    summary["mean_percentage_score"] = round(sum(percentage_scores) / len(percentage_scores), 4) if percentage_scores else ""
    return summary


def build_summary_rows(score_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped_rows: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped_rows[
            (
                clean_text(row.get("model_preset", "")),
                clean_text(row.get("profile_name", "")),
                clean_text(row.get("architecture", "")),
                clean_text(row.get("base_text_preset", "")),
                clean_text(row.get("medical_treatment_preset", "")),
                clean_text(row.get("treatment_generation_mode", "")),
                clean_text(row.get("judge_model", "")),
                clean_text(row.get("input_type", "")),
            )
        ].append(row)

    summaries: List[Dict[str, Any]] = []
    overall_grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)

    for key, rows in grouped_rows.items():
        judge_model = key[-2]
        input_type = key[-1]
        summaries.append(
            compute_summary_row(
                rows=rows,
                input_scope="by_input_type",
                input_type=input_type,
                judge_model=judge_model,
            )
        )
        overall_grouped[key[:-1]].extend(rows)

    for key, rows in overall_grouped.items():
        judge_model = key[-1]
        summaries.append(
            compute_summary_row(
                rows=rows,
                input_scope="all",
                input_type="all",
                judge_model=judge_model,
            )
        )

    case_level_rows: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        case_level_rows[build_case_group_key(row)].append(row)

    judge_average_rows: List[Dict[str, Any]] = []
    for _, rows in case_level_rows.items():
        first_row = rows[0]
        scored = [row for row in rows if clean_text(row.get("judge_status", "")) == "scored"]
        judge_average_row = dict(first_row)
        judge_average_row["judge_model"] = "judge_average"
        if scored:
            judge_average_row["judge_status"] = "scored"
            for dimension in SCORE_DIMENSIONS:
                values = [normalize_float(row.get(dimension, 0.0)) for row in scored]
                judge_average_row[dimension] = round(sum(values) / len(values), 4)
            judge_average_row["total_score"] = round(
                sum(normalize_float(row.get("total_score", 0.0)) for row in scored) / len(scored),
                4,
            )
            judge_average_row["percentage_score"] = round(
                sum(normalize_float(row.get("percentage_score", 0.0)) for row in scored) / len(scored),
                4,
            )
            judge_average_row["judge_error"] = ""
        else:
            judge_average_row["judge_status"] = "parse_error"
            for dimension in SCORE_DIMENSIONS:
                judge_average_row[dimension] = ""
            judge_average_row["total_score"] = ""
            judge_average_row["percentage_score"] = ""
            judge_average_row["judge_error"] = "all_judges_failed"
        judge_average_rows.append(judge_average_row)

    judge_average_grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    judge_average_overall: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in judge_average_rows:
        judge_average_grouped[
            (
                clean_text(row.get("model_preset", "")),
                clean_text(row.get("profile_name", "")),
                clean_text(row.get("architecture", "")),
                clean_text(row.get("base_text_preset", "")),
                clean_text(row.get("medical_treatment_preset", "")),
                clean_text(row.get("treatment_generation_mode", "")),
                clean_text(row.get("input_type", "")),
            )
        ].append(row)
        judge_average_overall[
            (
                clean_text(row.get("model_preset", "")),
                clean_text(row.get("profile_name", "")),
                clean_text(row.get("architecture", "")),
                clean_text(row.get("base_text_preset", "")),
                clean_text(row.get("medical_treatment_preset", "")),
                clean_text(row.get("treatment_generation_mode", "")),
            )
        ].append(row)

    for _, rows in judge_average_grouped.items():
        summaries.append(
            compute_summary_row(
                rows=rows,
                input_scope="by_input_type",
                input_type=clean_text(rows[0].get("input_type", "")),
                judge_model="judge_average",
            )
        )
    for _, rows in judge_average_overall.items():
        summaries.append(
            compute_summary_row(
                rows=rows,
                input_scope="all",
                input_type="all",
                judge_model="judge_average",
            )
        )

    summaries.sort(
        key=lambda row: (
            clean_text(row.get("model_preset", "")),
            clean_text(row.get("judge_model", "")),
            clean_text(row.get("input_scope", "")),
            clean_text(row.get("input_type", "")),
        )
    )
    return summaries


def write_summary_files(
    *,
    output_dir: Path,
    results_csv: Path,
    judges: Sequence[str],
    score_rows: Sequence[Dict[str, Any]],
) -> None:
    summary_rows = build_summary_rows(score_rows)
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path = output_dir / "summary.json"

    with summary_csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "results_csv": str(results_csv),
        "judges": [normalize_judge_name(item) for item in judges],
        "score_dimensions": list(SCORE_DIMENSIONS),
        "total_score_rows": len(score_rows),
        "summary_rows": summary_rows,
    }
    summary_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    results_csv = Path(args.results_csv).expanduser().resolve()
    if not results_csv.exists():
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    output_dir = resolve_output_dir(results_csv, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        reset_output_files(output_dir)

    judges = parse_judge_names(args.judges)
    score_csv_path = output_dir / "scores.csv"
    score_jsonl_path = output_dir / "scores.jsonl"

    source_rows = apply_model_result_overrides(read_csv_rows(results_csv))
    if args.limit is not None:
        source_rows = source_rows[: max(0, int(args.limit))]

    existing_result_keys = load_existing_result_keys(score_csv_path) if args.resume else set()
    case_loader = CaseContextLoader(mode=args.mode)
    scoring_agents = {
        judge: GenericTreatmentScoringAgent(judge_name=judge, max_new_tokens=int(args.max_new_tokens))
        for judge in judges
    }

    try:
        for source_row in source_rows:
            answer_text = clean_text(source_row.get("answer", ""))
            answer_is_artifact = looks_like_tool_artifact(answer_text)
            for judge_name, scorer in scoring_agents.items():
                result_key = build_result_key(source_row, judge_name)
                if result_key in existing_result_keys:
                    continue

                try:
                    case_context = case_loader.load_case_context(source_row)
                except Exception as exc:
                    empty_result = ScoringResult(
                        judge_status="case_load_error",
                        scores={dimension: 0.0 for dimension in SCORE_DIMENSIONS},
                        total_score=0.0,
                        percentage_score=0.0,
                        overall_grade="poor",
                        strengths=[],
                        weaknesses=[],
                        safety_concerns=[],
                        short_comment="",
                        attempts=[],
                        judge_error=clean_text(exc),
                        parsed_on_attempt=0,
                    )
                    score_row = build_score_csv_row(
                        result_key=result_key,
                        judge_name=judge_name,
                        judge_backend=scorer.judge_backend,
                        source_row=source_row,
                        scoring_result=empty_result,
                        case_context="",
                        answer_text=answer_text,
                        answer_is_artifact=answer_is_artifact,
                    )
                    append_csv_row(score_csv_path, SCORES_CSV_FIELDS, score_row)
                    append_jsonl(
                        score_jsonl_path,
                        build_score_jsonl_record(
                            score_row=score_row,
                            scoring_result=empty_result,
                            case_context="",
                            answer_text=answer_text,
                        ),
                    )
                    existing_result_keys.add(result_key)
                    continue

                answer_meta = build_answer_meta(source_row, answer_text)
                scoring_result = scorer.score(
                    case_context=case_context,
                    answer_text=answer_text,
                    answer_meta=answer_meta,
                )
                score_row = build_score_csv_row(
                    result_key=result_key,
                    judge_name=judge_name,
                    judge_backend=scorer.judge_backend,
                    source_row=source_row,
                    scoring_result=scoring_result,
                    case_context=case_context,
                    answer_text=answer_text,
                    answer_is_artifact=answer_is_artifact,
                )
                append_csv_row(score_csv_path, SCORES_CSV_FIELDS, score_row)
                append_jsonl(
                    score_jsonl_path,
                    build_score_jsonl_record(
                        score_row=score_row,
                        scoring_result=scoring_result,
                        case_context=case_context,
                        answer_text=answer_text,
                    ),
                )
                existing_result_keys.add(result_key)
    finally:
        for scorer in scoring_agents.values():
            scorer.close()

    final_score_rows = maybe_read_csv_rows(score_csv_path)
    write_summary_files(
        output_dir=output_dir,
        results_csv=results_csv,
        judges=judges,
        score_rows=final_score_rows,
    )

    payload = {
        "results_csv": str(results_csv),
        "output_dir": str(output_dir),
        "judges": judges,
        "scored_rows": len(final_score_rows),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
