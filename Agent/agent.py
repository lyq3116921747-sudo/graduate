from __future__ import annotations

import copy
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence

from .config import AgentConfig
from .imaging import ImagingStudyReportGenerator, collect_candidate_study_paths
from .llm_presets import identify_text_backend_preset_name
from .modeling import build_text_backend
from .rag import PDFKnowledgeBase, split_text_to_chunks
from .summary import StructuredSummary, build_structured_summary, render_summary_prompt
from .tools import LLMGeneratedPythonTool, SOFATool, Sepsis3Judge
from .utils import (
    build_case_payload_from_csv_row,
    case_to_context,
    clean_text,
    ensure_list,
    extract_json_object,
    get_case_auxiliary_exam_text,
    load_case_payload,
    normalize_text,
    normalize_case_payload,
    apply_case_input_mode,
    safe_path_text,
)

REFLECTION_SYSTEM_PROMPT = (
    "你是医疗输出反思器。"
    "检查答案是否与病例、影像、RAG 和工具证据匹配，并保持与当前任务模式一致的输出结构。"
    "diagnosis 模式输出“诊断结果 / 诊断依据”，treatment 模式只输出“治疗建议”，diagnosis_and_treatment 模式输出三段。"
    "只返回 JSON，字段包括 pass、issues、revised_answer。"
)

SEPSIS3_EXTRACTION_SYSTEM_PROMPT = (
    "你是 Sepsis-3 指标抽取器。"
    "你的唯一任务是从病例自由文本中抽取感染证据、SOFA 计算指标、休克线索、慢性混杂因素和竞争病因。"
    "禁止直接判断患者是否为脓毒症。"
    "必须只返回一个合法 JSON 对象。"
)
SEPSIS3_EXTRACTION_MAX_NEW_TOKENS = 512
SEPSIS3_EXTRACTION_RESPONSE_FORMAT = {"type": "json_object"}
SEPSIS3_PROVIDER_TOKEN_FLOORS = {
    "antangel": 12288,
    "baichuan": 8192,
}
SEPSIS3_PROVIDER_TIMEOUT_FLOORS = {
    "antangel": 240.0,
    "baichuan": 180.0,
}

TASK_MODE_DIAGNOSIS = "diagnosis"
TASK_MODE_TREATMENT = "treatment"
TASK_MODE_DIAGNOSIS_AND_TREATMENT = "diagnosis_and_treatment"
SUPPORTED_TASK_MODES = {
    TASK_MODE_DIAGNOSIS,
    TASK_MODE_TREATMENT,
    TASK_MODE_DIAGNOSIS_AND_TREATMENT,
}
TREATMENT_DIAGNOSIS_CONTEXT_FIELDS = (
    "西医入院诊断",
    "西医出院诊断",
)

TREATMENT_REACT_SYSTEM_PROMPT = (
    "你是脓毒症治疗建议智能体。"
    "本阶段只负责为最终治疗建议收集必要信息，不直接撰写最终治疗建议。"
    "收集信息必须围绕五个评分维度：临床合理性、安全性、证据依据性、完整性、可执行性。"
    "优先补齐感染源/病原学、器官功能障碍、SOFA或严重程度、安全约束、指南证据和可执行治疗动作所需证据。"
    "不要臆造病例信息；没有证据的具体数值只能标记为需补充或需复查。"
    "当证据足够时停止调用工具，并用简短中文说明“证据已足够，可进入最终定稿”，不要输出治疗建议正文。"
    "禁止输出病例关键信息汇总、SOFA计算过程或最终治疗建议段落。"
)

TREATMENT_FINALIZATION_SYSTEM_PROMPT = (
    "你是治疗建议定稿器。"
    "你只能基于已经提供的评分维度证据记忆、病例事实、工具证据和指南证据，整理出最终可执行的治疗建议。"
    "输出必须对齐五个评分维度：临床合理性、安全性、证据依据性、完整性、可执行性。"
    "不要输出病例摘要、分析过程、SOFA计算过程、工具名、section 名称、JSON 或工具日志。"
    "只输出“治疗建议：”一节，使用编号分点；每条必须是治疗动作或明确的补充/监测计划。"
    "不得编造病例中没有的具体数值；证据不足时写需补充、需复查或需动态评估。"
)


MEDICAL_TREATMENT_GENERATION_SYSTEM_PROMPT = (
    "你是医疗专用治疗建议生成器。"
    "你只能根据已经收集好的病例信息、SOFA 关键指标、影像和指南证据生成最终治疗建议。"
    "不调用任何工具，不能编造检查结果或用药。"
    "只输出“治疗建议：”一节，使用编号分点，必要时可指出信息不足与待补充项。"
)


def _build_treatment_round_reminder(round_index: int, max_rounds: int) -> str:
    remaining_rounds = max(0, max_rounds - round_index + 1)
    lines = [
        (
            f"当前治疗工具调用进度：第 {round_index}/{max_rounds} 轮；"
            f"含本轮在内还剩 {remaining_rounds} 轮。"
            "若现有证据足够，请停止调用工具并简短说明“证据已足够，可进入最终定稿”，不要输出治疗建议正文。"
        )
    ]
    if remaining_rounds == 3:
        lines.append("还剩 3 轮，请开始收束，避免重复检索；除非存在关键缺口，否则停止工具调用并进入定稿。")
    if remaining_rounds == 1:
        lines.append("这是最后一轮，请不要再调用工具；只输出证据收集摘要，不要写最终治疗建议。")
    return "\n".join(lines)


def _validate_task_mode(value: Any) -> str:
    normalized = clean_text(value).lower() or TASK_MODE_DIAGNOSIS
    if normalized not in SUPPORTED_TASK_MODES:
        available = ", ".join(sorted(SUPPORTED_TASK_MODES))
        raise ValueError(f"Unsupported task_mode: {value}; available: {available}")
    return normalized


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if text in {"1", "true", "yes", "y", "是", "需要"}:
        return True
    if text in {"0", "false", "no", "n", "否", "不需要"}:
        return False
    return default


def _normalize_string_list(value: Any) -> List[str]:
    items = ensure_list(value)
    result: List[str] = []
    for item in items:
        text = clean_text(item)
        if text and text not in result:
            result.append(text)
    return result


def _split_numbered_section(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?:^|\n)\s*(?:[-*]|\d+[.、]|[一二三四五六七八九十]+、)\s*", normalized)
    items = [clean_text(part) for part in parts if clean_text(part)]
    if items:
        return items
    return [normalized]

def _parse_binary_label_from_text(text: str) -> Optional[int]:
    normalized = clean_text(text)
    if not normalized:
        return None
    match = re.search(r"FINAL_LABEL\s*[:：]\s*([01])", normalized, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if any(token in normalized for token in ("非脓毒症", "不是脓毒症", "不支持脓毒症", "排除脓毒症")):
        return 0
    if any(token in normalized for token in ("考虑脓毒症", "符合脓毒症", "支持脓毒症", "脓毒症")):
        return 1
    return None


def _parse_assessment_response(text: str) -> Dict[str, Any]:
    payload = extract_json_object(text)
    if payload:
        result = {
            "binary_label": None,
            "diagnosis_result": clean_text(payload.get("diagnosis_result") or payload.get("诊断结果") or payload.get("诊断结论") or ""),
            "diagnostic_basis": _normalize_string_list(payload.get("diagnostic_basis") or payload.get("诊断依据") or []),
            "treatment_advice": _normalize_string_list(payload.get("treatment_advice") or payload.get("治疗建议") or []),
            "parse_mode": "json",
        }
        if "binary_label" in payload:
            try:
                result["binary_label"] = int(payload["binary_label"])
            except Exception:
                result["binary_label"] = _parse_binary_label_from_text(text)
        else:
            result["binary_label"] = _parse_binary_label_from_text(text)
        return result

    normalized = normalize_text(text)
    diagnosis_match = re.search(r"诊断(?:结果|结论)[:：]\s*(.*?)(?=\n\s*诊断依据[:：]|\Z)", normalized, flags=re.DOTALL)
    basis_match = re.search(r"诊断依据[:：]\s*(.*?)(?=\n\s*治疗建议[:：]|\Z)", normalized, flags=re.DOTALL)
    advice_match = re.search(r"治疗建议[:：]\s*(.*)\Z", normalized, flags=re.DOTALL)
    return {
        "binary_label": _parse_binary_label_from_text(normalized),
        "diagnosis_result": clean_text(diagnosis_match.group(1) if diagnosis_match else ""),
        "diagnostic_basis": _split_numbered_section(basis_match.group(1) if basis_match else ""),
        "treatment_advice": _split_numbered_section(advice_match.group(1) if advice_match else ""),
        "parse_mode": "labeled" if (diagnosis_match or basis_match or advice_match) else "fallback",
    }


def _split_paragraphs(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]

TREATMENT_TOOL_MARKERS = (
    '"query":',
    '"matches":',
    '"tool_name":',
    '"tool_calls":',
    '"arguments":',
    "search_local_guidelines",
    "run_sofa_assessment",
    "read_case_sections",
    "get_case_digest",
    "search_case_text",
    "run_generated_probe",
)

TREATMENT_SECTION_MARKERS = (
    "case_overview",
    "structured_summary",
    "diagnosis_context",
    "chief_complaint",
    "present_illness",
    "physical_exam",
    "lab_and_auxiliary",
    "clinical_course",
    "full_context",
    "available_sections",
    "section_names",
    "resolved_section_names",
)

TREATMENT_DOMAIN_MARKERS = (
    "抗感染",
    "抗菌",
    "感染源",
    "培养",
    "引流",
    "液体复苏",
    "补液",
    "晶体液",
    "循环",
    "升压",
    "去甲肾上腺素",
    "MAP",
    "乳酸",
    "氧疗",
    "呼吸支持",
    "器官支持",
    "尿量",
    "监测",
    "复查",
    "复评",
    "动态评估",
)

TREATMENT_META_SUMMARY_MARKERS = (
    "我已经收集",
    "已经收集了足够",
    "让我整理",
    "让我综合",
    "病例关键信息",
    "关键信息总结",
    "关键发现",
    "病例信息总结",
    "现在给出最终治疗建议",
    "SOFA评分如下",
    "SOFA评分：",
)

TREATMENT_ACTION_MARKERS = (
    "给予",
    "启动",
    "使用",
    "继续",
    "调整",
    "停用",
    "留取",
    "完善",
    "完成",
    "监测",
    "复查",
    "复评",
    "评估",
    "控制",
    "引流",
    "清创",
    "手术",
    "补液",
    "复苏",
    "维持",
    "支持",
    "纠正",
    "会诊",
    "培养",
    "降阶梯",
    "升级",
    "覆盖",
    "目标",
)

TREATMENT_CORE_DOMAIN_MARKERS = {
    "anti_infective": ("抗感染", "抗菌", "抗生素", "抗真菌", "培养", "药敏", "降阶梯", "覆盖"),
    "source_control": ("感染源", "引流", "清创", "手术", "ERCP", "PTCD", "导管", "穿刺", "控制感染源"),
    "circulation": ("液体复苏", "补液", "晶体液", "循环", "升压", "去甲肾上腺素", "MAP", "乳酸", "尿量"),
    "organ_support": ("器官支持", "氧疗", "呼吸支持", "机械通气", "肾脏", "CRRT", "凝血", "血小板", "肝功能", "血糖", "营养"),
    "monitoring": ("监测", "复查", "复评", "动态评估", "生命体征", "SOFA", "PCT", "CRP", "乳酸", "培养"),
}

TREATMENT_NUMERIC_CLAIM_KEYWORDS = (
    "PCT",
    "CRP",
    "肌酐",
    "血小板",
    "乳酸",
    "总胆红素",
    "胆红素",
    "PaO2",
    "FiO2",
    "氧合指数",
    "白细胞",
    "WBC",
    "NEUT",
    "INR",
    "APTT",
    "PT",
    "D-二聚体",
    "白蛋白",
)

TREATMENT_SCORECARD_FIELDS = (
    "clinical_reasonableness_evidence",
    "safety_evidence",
    "evidence_grounding",
    "unsupported_or_uncertain_claims",
    "completeness_targets",
    "actionability_targets",
)


def _treatment_action_like(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in TREATMENT_ACTION_MARKERS)


def _treatment_core_domain_hits(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    hits: set[str] = set()
    for domain_name, markers in TREATMENT_CORE_DOMAIN_MARKERS.items():
        if any(marker.lower() in normalized for marker in markers):
            hits.add(domain_name)
    return hits


def _extract_supported_number_tokens(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {
        match.group(0).strip()
        for match in re.finditer(r"\d+(?:\.\d+)?", normalized)
        if match.group(0).strip()
    }


def _find_unsupported_numeric_claims(answer_text: str, evidence_text: str) -> List[str]:
    normalized_answer = normalize_text(answer_text)
    if not normalized_answer:
        return []
    evidence_numbers = _extract_supported_number_tokens(evidence_text)
    unsupported: List[str] = []
    for keyword in TREATMENT_NUMERIC_CLAIM_KEYWORDS:
        pattern = rf"{re.escape(keyword)}[^\n，。；;]{{0,24}}?(\d+(?:\.\d+)?)"
        for match in re.finditer(pattern, normalized_answer, flags=re.IGNORECASE):
            local = normalized_answer[max(0, match.start() - 12) : min(len(normalized_answer), match.end() + 12)]
            if any(
                token in local
                for token in (
                    "目标",
                    "阈值",
                    "安全阈值",
                    "维持",
                    "达到",
                    "控制在",
                    "升至",
                    "降至",
                    "不低于",
                    "不少于",
                    "不超过",
                    "不高于",
                    "至少",
                    "以上",
                    "以下",
                    "大于",
                    "小于",
                    "≥",
                    "≤",
                    ">",
                    "<",
                )
            ):
                continue
            number_text = clean_text(match.group(1))
            if number_text and number_text not in evidence_numbers:
                unsupported.append(f"{keyword}{number_text}")
    return _normalize_string_list(unsupported)


def _render_treatment_answer(raw_text: str, treatment_advice: Sequence[str]) -> str:
    normalized_advice = _normalize_string_list(list(treatment_advice))
    if normalized_advice:
        return _render_labeled_section("治疗建议", normalized_advice)
    normalized_raw = clean_text(raw_text)
    if not normalized_raw:
        return ""
    return normalized_raw if normalized_raw.startswith("治疗建议") else f"治疗建议：{normalized_raw}"


def _treatment_quality_flags(raw_text: str, treatment_advice: Sequence[str], evidence_text: str = "") -> List[str]:
    normalized = normalize_text(raw_text)
    lowered = normalized.lower()
    advice = _normalize_string_list(list(treatment_advice))
    flags: List[str] = []
    if not normalized or not advice:
        flags.append("empty_or_unparsed")
    if any(marker in lowered for marker in TREATMENT_TOOL_MARKERS):
        flags.append("tool_artifact")
    if any(marker in lowered for marker in TREATMENT_SECTION_MARKERS):
        flags.append("section_reference")
    if (
        re.search(r"(?:需要|还需|先|随后|接下来|下一步).{0,20}(?:调用|运行).{0,12}(?:工具|SOFA|read_|search_|run_)", normalized, flags=re.IGNORECASE)
        or re.search(r"(?:需要|还需|先|随后|接下来|下一步).{0,20}(?:读取|查看).{0,12}(?:病例|字段|section|原始)", normalized, flags=re.IGNORECASE)
        or re.search(r"(?:需要|还需|先|随后|接下来|下一步).{0,20}(?:检索|搜索).{0,12}(?:指南|文献|知识库)", normalized, flags=re.IGNORECASE)
    ):
        flags.append("tool_plan")
    domain_hits = sum(1 for marker in TREATMENT_DOMAIN_MARKERS if marker.lower() in lowered)
    if advice and len(advice) < 3 and domain_hits < 3:
        flags.append("too_brief")
    if any(marker in normalized for marker in TREATMENT_META_SUMMARY_MARKERS):
        flags.append("meta_summary_as_answer")
    first_items = advice[: min(3, len(advice))]
    if first_items and sum(1 for item in first_items if _treatment_action_like(item)) < max(1, len(first_items) - 1):
        flags.append("meta_summary_as_answer")
    if advice and len([item for item in advice if _treatment_action_like(item)]) < 5:
        flags.append("insufficient_actions")
    core_domain_hits = _treatment_core_domain_hits(normalized)
    if advice and len(core_domain_hits) < 4:
        flags.append("missing_core_domains")
    if evidence_text and _find_unsupported_numeric_claims(normalized, evidence_text):
        flags.append("unsupported_numeric_claim")
    return _normalize_string_list(flags)


def _treatment_needs_finalization(raw_text: str, treatment_advice: Sequence[str], final_error: str = "") -> bool:
    flags = set(_treatment_quality_flags(raw_text, treatment_advice))
    if clean_text(final_error) == "treatment_react_round_limit_reached":
        return True
    return bool(
        flags
        & {
            "empty_or_unparsed",
            "tool_artifact",
            "tool_plan",
            "section_reference",
            "too_brief",
            "meta_summary_as_answer",
            "insufficient_actions",
            "missing_core_domains",
            "unsupported_numeric_claim",
        }
    )

def _parse_treatment_stage_response(text: str) -> Dict[str, Any]:
    payload = extract_json_object(text)
    if payload:
        treatment_advice = _normalize_string_list(
            payload.get("treatment_advice")
            or payload.get("治疗建议")
            or payload.get("treatment")
            or payload.get("advice")
            or payload.get("management")
            or ""
        )
        return {"treatment_advice": treatment_advice, "parse_mode": "json"}

    normalized = normalize_text(text)
    if not normalized:
        return {"treatment_advice": [], "parse_mode": "empty"}

    treatment_match = re.search(r"治疗建议[:：]\s*(.*)\Z", normalized, flags=re.DOTALL)
    if treatment_match:
        return {
            "treatment_advice": _split_numbered_section(treatment_match.group(1)),
            "parse_mode": "labeled",
        }

    paragraphs = _split_paragraphs(normalized)
    if paragraphs:
        paragraph_text = "\n".join(paragraphs)
        paragraph_advice = _split_numbered_section(paragraph_text)
        paragraph_lowered = paragraph_text.lower()
        if (
            any(marker in paragraph_lowered for marker in TREATMENT_TOOL_MARKERS)
            or any(marker in paragraph_lowered for marker in TREATMENT_SECTION_MARKERS)
            or any(marker in paragraph_text for marker in TREATMENT_META_SUMMARY_MARKERS)
            or not any(_treatment_action_like(item) for item in paragraph_advice)
        ):
            return {"treatment_advice": [], "parse_mode": "paragraph_split_invalid"}
        return {
            "treatment_advice": paragraph_advice,
            "parse_mode": "paragraph_split",
        }

    return {"treatment_advice": _split_numbered_section(normalized), "parse_mode": "fallback_full"}

def _truncate_for_prompt(text: str, max_chars: int = 1800) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 1].rstrip() + "…"


def _render_labeled_section(title: str, items: List[str]) -> str:
    normalized_items = _normalize_string_list(items)
    if not normalized_items:
        return f"{title}："
    if len(normalized_items) == 1:
        return f"{title}：{normalized_items[0]}"
    return f"{title}：\n" + "\n".join(f"{idx}. {item}" for idx, item in enumerate(normalized_items, start=1))


CLINICAL_TOOL_PROMPT_TOKENS = (
    "SOFA",
    "血压",
    "MAP",
    "乳酸",
    "肌酐",
    "血小板",
    "总胆红素",
    "GCS",
    "PaO2",
    "FiO2",
    "白细胞",
    "PCT",
    "CRP",
    "培养",
    "感染",
    "发热",
    "寒战",
)
TOOL_PROMPT_NOISE_TOKENS = (
    "top_level_keys",
    "case_path",
    "candidate_dir",
    "_exists",
    "_dcm_count",
    "script_path",
    "imaging_entry_count",
)


def _tool_claim_is_prompt_worthy(claim: str) -> bool:
    normalized = normalize_text(claim)
    if not normalized:
        return False
    if any(token in normalized for token in TOOL_PROMPT_NOISE_TOKENS):
        return False
    return any(token in normalized for token in CLINICAL_TOOL_PROMPT_TOKENS)


def _build_clinical_tool_text(tools_result: Dict[str, Any]) -> str:
    lines: List[str] = []
    sofa_payload = tools_result.get("sofa", {})
    if isinstance(sofa_payload, dict):
        sofa_text = normalize_text(sofa_payload.get("text", ""))
        if sofa_text:
            lines.append(sofa_text)

    for key in ("generated_code", "static_code"):
        payload = tools_result.get(key, {})
        if not isinstance(payload, dict):
            continue
        combined_text = "\n".join(
            part for part in [clean_text(payload.get("stdout", "")), clean_text(payload.get("stderr", ""))] if part
        )
        for claim in normalize_text(combined_text).splitlines():
            if not _tool_claim_is_prompt_worthy(claim):
                continue
            cleaned_claim = clean_text(claim)
            if cleaned_claim and cleaned_claim not in lines:
                lines.append(cleaned_claim)
    return "\n".join(lines).strip()


def _make_stage_error(stage: str, error: str, *, raw_text: str = "") -> Dict[str, str]:
    payload = {
        "stage": clean_text(stage),
        "error": clean_text(error),
    }
    if clean_text(raw_text):
        payload["raw_text"] = clean_text(raw_text)
    return payload


def _freeze_backend_cache_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze_backend_cache_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_backend_cache_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_backend_cache_value(item) for item in value))
    return value


def _make_backend_cache_key(backend_config: Any) -> Any:
    dataclass_fields = getattr(backend_config, "__dataclass_fields__", None)
    if dataclass_fields:
        return tuple(
            (field_name, _freeze_backend_cache_value(getattr(backend_config, field_name)))
            for field_name in dataclass_fields
        )
    return _freeze_backend_cache_value(backend_config)

def _looks_like_csv_row_payload(case_payload: Dict[str, Any]) -> bool:
    if not isinstance(case_payload, dict):
        return False
    if "EMPI" in case_payload or "group_id" in case_payload:
        return True
    return (
        "住院年龄" in case_payload
        and "门诊/住院号" in case_payload
        and "唯一号" not in case_payload
        and "病程记录" not in case_payload
        and "医嘱记录" not in case_payload
    )


class SepsisMedGemmaAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.config.ensure_runtime_dirs()
        self._text_backend_cache: Dict[Any, Any] = {}
        self.rag = PDFKnowledgeBase(self.config)
        self.sofa_tool = SOFATool()
        self.sepsis3_judge = Sepsis3Judge()
        self.imaging = ImagingStudyReportGenerator(self.config)
        self._generated_code_tool: Optional[LLMGeneratedPythonTool] = None

    def _get_text_backend(self, backend_config: Any) -> Any:
        cache_key = _make_backend_cache_key(backend_config)
        if cache_key not in self._text_backend_cache:
            self._text_backend_cache[cache_key] = build_text_backend(
                backend_config=backend_config,
                hf_cache_root=self.config.hf_cache_root,
                attn_implementation=self.config.attn_implementation,
                local_files_only=self.config.local_files_only,
                max_length=self.config.generation_max_length,
                timeout_seconds=self.config.http_timeout_seconds,
            )
        return self._text_backend_cache[cache_key]

    def _generate_text(
        self,
        *,
        backend_config: Any,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        timeout_seconds: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        backend = self._get_text_backend(backend_config)
        return backend.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_length=self.config.generation_max_length,
            request_timeout_seconds=timeout_seconds,
            response_format=response_format,
        )

    def _get_generated_code_tool(self) -> LLMGeneratedPythonTool:
        if self._generated_code_tool is None:
            self._generated_code_tool = LLMGeneratedPythonTool(
                runtime_root=self.config.runtime_root / "code_exec",
                text_model=self._get_text_backend(self.config.resolve_text_backend("code_writer")),
                max_new_tokens=self.config.code_writer_max_new_tokens,
            )
        return self._generated_code_tool

    def _preserved_case_fields_for_task_mode(self, task_mode: str) -> List[str]:
        if _validate_task_mode(task_mode) == TASK_MODE_TREATMENT:
            return list(TREATMENT_DIAGNOSIS_CONTEXT_FIELDS)
        return []

    def _prepare_case_payload(self, case_payload: Dict[str, Any], task_mode: str = TASK_MODE_DIAGNOSIS) -> Dict[str, Any]:
        preserve_fields = self._preserved_case_fields_for_task_mode(task_mode)
        if _looks_like_csv_row_payload(case_payload):
            return build_case_payload_from_csv_row(
                case_payload,
                mode=self.config.case_input_mode,
                preserve_fields=preserve_fields,
            )
        return apply_case_input_mode(
            case_payload,
            mode=self.config.case_input_mode,
            preserve_fields=preserve_fields,
        )

    def _build_sepsis3_plan(self) -> Dict[str, Any]:
        return {
            "task_category": "sepsis3_diagnosis",
            "intent": "按 Sepsis-3 判断是否脓毒症",
            "diagnosis_mode": "sepsis3_rule_based",
            "use_rag": False,
            "use_web_search": False,
            "use_data_tools": bool(self.config.enable_diagnosis_sofa_tool),
            "use_generated_code": False,
            "use_imaging": False,
            "use_llm_extraction": bool(self.config.enable_diagnosis_llm_extraction),
            "use_sofa_tool": bool(self.config.enable_diagnosis_sofa_tool),
            "use_case_backfill": bool(self.config.enable_diagnosis_case_backfill),
            "use_reflection": bool(self.config.enable_reflection),
            "rag_queries": [],
            "web_queries": [],
            "generated_code_goal": "",
        }

    def _build_disabled_sepsis3_extraction_result(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "error": "",
            "parse_mode": "disabled",
            "payload": {},
            "attempts": [],
            "raw": "",
        }

    def _build_treatment_case_diagnosis_assessment(self, case_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        payload = normalize_case_payload(case_payload)
        diagnostic_basis: List[str] = []
        admission_diagnosis = normalize_text(payload.get("西医入院诊断", ""))
        discharge_diagnosis = normalize_text(payload.get("西医出院诊断", ""))
        if admission_diagnosis:
            diagnostic_basis.append(f"西医入院诊断：{admission_diagnosis}")
        if discharge_diagnosis:
            diagnostic_basis.append(f"西医出院诊断：{discharge_diagnosis}")
        if not diagnostic_basis:
            return None
        return {
            "diagnosis_result": "病历已有临床诊断信息（非 Sepsis-3 自动判定）",
            "diagnostic_basis": diagnostic_basis,
        }

    def _build_sepsis3_extraction_prompt(
        self,
        *,
        case_context: str,
        sofa_result: Dict[str, Any],
        compact_no_traditional: bool = False,
    ) -> str:
        schema_text = (
            '{"infection_status":"present|absent|uncertain","infection_evidence":["短语"],'
            '"infection_source":["短语"],"reported_sofa":null,'
            '"respiration":{"pao2":null,"fio2":null,"pf_ratio":null},'
            '"coagulation":{"platelets":null},"liver":{"bilirubin":null,"unit":""},'
            '"cardiovascular":{"sbp":null,"dbp":null,"map":null,"vasopressor_used":null,'
            '"vasopressor_name":"","low_bp_text":false,"shock_text":false},'
            '"cns":{"gcs":null,"new_mental_status_change":null},'
            '"renal":{"creatinine":null,"unit":"","oliguria":null},"lactate":null,'
            '"acute_change_flags":{"respiration":[],"coagulation":[],"liver":[],"cardiovascular":[],'
            '"cns":[],"renal":[],"general":[]},"chronic_confounders":[],'
            '"competing_noninfectious_causes":[],"shock_terms":[],"uncertainty_notes":[],'
            '"evidence_quotes":["原句摘录"]}'
        )
        sofa_text = _truncate_for_prompt(sofa_result.get("text", ""), 700)
        sofa_prompt = ""
        if sofa_text:
            sofa_prompt = (
                "\n\nSOFA 工具辅助抽取（仅用于帮助定位数值，不是最终判定）：\n"
                + sofa_text
            )
        if compact_no_traditional:
            return (
                "请只抽取 Sepsis-3 诊断所需结构化指标，并且只返回一个完整 JSON 对象。\n"
                "不要 Markdown，不要解释，不要前后缀文本。\n"
                "关键要求：\n"
                "1. 缺失信息必须用 null、[] 或空字符串，禁止编造。\n"
                "2. infection_evidence、infection_source、chronic_confounders、competing_noninfectious_causes、shock_terms、uncertainty_notes 最多各 2 条。\n"
                "3. evidence_quotes 最多 1 条。\n"
                "4. acute_change_flags 的每个器官桶最多 1 条短语。\n"
                "5. 普通列表项尽量不超过 16 个中文字符，evidence_quotes 尽量不超过 40 个中文字符。\n"
                "6. 优先保留 infection_status、reported_sofa、各器官数值字段、lactate 和最关键感染证据。\n"
                "7. 如果某个长列表不确定，直接返回空数组，不要继续展开。\n"
                "8. 宁可少填，也必须闭合所有括号、引号和数组，保证 JSON 完整可解析。\n\n"
                f"返回 JSON schema 示例：\n{schema_text}\n\n"
                f"病例文本：\n{_truncate_for_prompt(case_context, 3200)}"
            )
        if not sofa_text:
            return (
                "Extract only the structured Sepsis-3 evidence fields and return exactly one JSON object.\n"
                "Do not make the final sepsis diagnosis directly.\n"
                "Hard requirements:\n"
                "1. Output JSON only. No markdown and no extra explanation.\n"
                "2. Do not invent facts. Use null, [] or empty strings for missing values.\n"
                "3. Keep chronic confounders and competing non-infectious causes separate.\n"
                "4. Keep infection_evidence, infection_source, shock_terms and uncertainty_notes concise.\n\n"
                f"JSON schema example:\n{schema_text}\n\n"
                f"Case text:\n{_truncate_for_prompt(case_context, 3200)}"
            )
        return (
            "请只抽取 Sepsis-3 诊断所需结构化指标，不要直接判断是否脓毒症。\n"
            "硬性要求：\n"
            "1. 只返回一个 JSON 对象，不要 Markdown，不要解释。\n"
            "2. 不能编造；缺失信息用 null、[] 或空字符串。\n"
            "3. “脓毒症/脓毒性休克”诊断词只能放入 evidence_quotes 或 uncertainty_notes，不能当最终结论。\n"
            "4. 慢性基础病与竞争病因要单独抽取。\n"
            "5. infection_evidence、infection_source、shock_terms、uncertainty_notes 每项最多 4 条，每条尽量不超过 18 个字。\n"
            "6. evidence_quotes 最多 2 条，只保留最关键原句。\n"
            "7. 如果输出过长，优先保留数值字段和最关键的感染证据。\n\n"
            f"返回 JSON schema 示例：\n{schema_text}\n\n"
            f"病例文本：\n{_truncate_for_prompt(case_context, 3200)}\n\n"
            "SOFA 工具辅助抽取（仅用于帮助定位数值，不是最终判定）：\n"
            f"{_truncate_for_prompt(sofa_result.get('text', ''), 700)}"
        )

    def _build_sepsis3_retry_prompt(
        self,
        *,
        previous_raw: str,
        base_prompt: str,
        compact_no_traditional: bool = False,
    ) -> str:
        if compact_no_traditional:
            return (
                "上一轮 JSON 不完整或格式错误。现在请直接重写一个更短、完整、可解析的 JSON。\n"
                "不要延续上一轮长输出，不要补充解释。\n"
                "优先保留 infection_status、reported_sofa、respiration/coagulation/liver/cardiovascular/cns/renal、lactate、最少量 infection_evidence。\n"
                "如果某个长列表不确定，返回空数组。\n"
                "务必闭合所有括号、引号和数组。\n\n"
                f"上一轮原始输出：\n{_truncate_for_prompt(previous_raw, 1200)}\n\n{base_prompt}"
            )
        return (
            "上一轮输出不是合法 JSON，可能是截断或格式错误。"
            "现在请重新输出一个更短但完整的 JSON：优先保留 infection_status、infection_evidence、infection_source、reported_sofa、"
            "respiration/coagulation/liver/cardiovascular/cns/renal/lactate、acute_change_flags、chronic_confounders、competing_noninfectious_causes、shock_terms、uncertainty_notes、evidence_quotes。\n\n"
            f"上一轮原始输出：\n{_truncate_for_prompt(previous_raw, 1200)}\n\n{base_prompt}"
        )

    def _run_sepsis3_extraction(
        self,
        *,
        case_context: str,
        sofa_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        diagnosis_backend = self.config.resolve_text_backend("diagnosis")
        backend_description = diagnosis_backend.describe().lower()
        preset_generation_floor = int(getattr(diagnosis_backend, "default_max_new_tokens", 0) or 0)
        preset_sepsis3_floor = int(getattr(diagnosis_backend, "sepsis3_extraction_max_new_tokens", 0) or 0)
        preset_sepsis3_timeout_floor = float(getattr(diagnosis_backend, "sepsis3_extraction_timeout_seconds", 0.0) or 0.0)
        sepsis3_max_new_tokens = max(
            self.config.generation_max_new_tokens,
            getattr(self.config, "sepsis3_extraction_max_new_tokens", SEPSIS3_EXTRACTION_MAX_NEW_TOKENS),
            preset_generation_floor,
            preset_sepsis3_floor,
        )
        sepsis3_timeout_seconds = getattr(
            self.config,
            "sepsis3_extraction_timeout_seconds",
            self.config.http_timeout_seconds,
        )
        sepsis3_timeout_seconds = max(sepsis3_timeout_seconds, preset_sepsis3_timeout_floor)
        for provider_key, token_floor in SEPSIS3_PROVIDER_TOKEN_FLOORS.items():
            if provider_key in backend_description:
                sepsis3_max_new_tokens = max(sepsis3_max_new_tokens, token_floor)
                sepsis3_timeout_seconds = max(
                    sepsis3_timeout_seconds,
                    SEPSIS3_PROVIDER_TIMEOUT_FLOORS.get(provider_key, sepsis3_timeout_seconds),
                )

        compact_no_traditional = (not self.config.enable_diagnosis_sofa_tool) and ("deepseek" in backend_description)
        prompt = self._build_sepsis3_extraction_prompt(
            case_context=case_context,
            sofa_result=sofa_result,
            compact_no_traditional=compact_no_traditional,
        )
        attempts: List[Dict[str, str]] = []
        extraction_payload: Dict[str, Any] = {}
        parse_mode = "empty"
        final_error = ""

        for attempt_index in range(2):
            user_prompt = prompt
            if attempt_index == 1:
                previous = clean_text(attempts[0]["raw"]) if attempts else ""
                user_prompt = self._build_sepsis3_retry_prompt(
                    previous_raw=previous,
                    base_prompt=prompt,
                    compact_no_traditional=compact_no_traditional,
                )
            raw = ""
            error = ""
            try:
                raw = self._generate_text(
                    backend_config=diagnosis_backend,
                    system_prompt=SEPSIS3_EXTRACTION_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_new_tokens=sepsis3_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    timeout_seconds=sepsis3_timeout_seconds,
                    response_format=SEPSIS3_EXTRACTION_RESPONSE_FORMAT,
                )
            except Exception as exc:
                error = str(exc)
                raw = ""

            parsed = extract_json_object(raw)
            attempts.append({"raw": clean_text(raw), "error": clean_text(error)})
            if parsed:
                extraction_payload = parsed
                parse_mode = "json"
                final_error = ""
                break
            parse_mode = "empty" if not clean_text(raw) else "non_json"
            final_error = clean_text(error) or ("empty_200_response" if not clean_text(raw) else "unparseable_response")

        return {
            "ok": bool(extraction_payload),
            "error": final_error,
            "parse_mode": parse_mode,
            "payload": extraction_payload,
            "attempts": attempts,
            "raw": clean_text(attempts[-1]["raw"]) if attempts else "",
        }

    def _assess_with_sepsis3(
        self,
        *,
        extraction_result: Dict[str, Any],
        sofa_result: Dict[str, Any],
        case_payload: Dict[str, Any],
        llm_extraction_enabled: bool = True,
    ) -> Dict[str, Any]:
        stage_errors: List[Dict[str, str]] = []
        extraction_error = clean_text(extraction_result.get("error", ""))
        if extraction_error:
            stage_errors.append(
                _make_stage_error(
                    "sepsis3_extraction",
                    extraction_error,
                    raw_text=extraction_result.get("raw", ""),
                )
            )

        if extraction_error and llm_extraction_enabled and not extraction_result.get("payload"):
            return {
                "binary_label": None,
                "classification": "extraction_failed",
                "diagnosis_result": "Sepsis-3 指标抽取失败，未完成自动判断",
                "diagnostic_basis": [
                    f"模型/接口错误：{extraction_error}",
                    "当前未获得可用的 Sepsis-3 结构化抽取结果，因此不输出自动诊断结论。",
                ],
                "treatment_advice": [],
                "sepsis3_reason": extraction_error,
                "sepsis3_judgement": {},
                "sepsis3_extraction": self.sepsis3_judge.normalize_extraction_payload(extraction_result.get("payload", {})),
                "sepsis3_extraction_raw": clean_text(extraction_result.get("raw", "")),
                "sepsis3_extraction_attempts": extraction_result.get("attempts", []),
                "sepsis3_extraction_parse_mode": clean_text(extraction_result.get("parse_mode", "")),
                "stage_errors": stage_errors,
            }

        judged = self.sepsis3_judge.evaluate(
            extraction_result.get("payload", {}),
            sofa_result,
            case_payload=case_payload,
            enable_case_backfill=self.config.enable_diagnosis_case_backfill,
            enable_sofa_backfill=self.config.enable_diagnosis_sofa_tool,
        )
        return {
            "binary_label": judged.get("binary_label"),
            "classification": clean_text(judged.get("classification", "")),
            "diagnosis_result": clean_text(judged.get("diagnosis_result", "")),
            "diagnostic_basis": _normalize_string_list(judged.get("diagnostic_basis")),
            "treatment_advice": [],
            "sepsis3_reason": clean_text(judged.get("sepsis3_reason", "")),
            "sepsis3_judgement": judged,
            "sepsis3_extraction": self.sepsis3_judge.normalize_extraction_payload(extraction_result.get("payload", {})),
            "sepsis3_extraction_raw": clean_text(extraction_result.get("raw", "")),
            "sepsis3_extraction_attempts": extraction_result.get("attempts", []),
            "sepsis3_extraction_parse_mode": clean_text(extraction_result.get("parse_mode", "")),
            "stage_errors": stage_errors,
        }

    def run(
        self,
        *,
        user_query: str,
        case_json_path: Optional[str] = None,
        case_payload: Optional[Dict[str, Any]] = None,
        task_mode: str = TASK_MODE_DIAGNOSIS,
        enable_search: Optional[bool] = None,
        search_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        resolved_task_mode = _validate_task_mode(task_mode)
        preserve_fields = self._preserved_case_fields_for_task_mode(resolved_task_mode)
        warnings: List[str] = []
        stage_errors: List[Dict[str, str]] = []
        if case_payload is None:
            if not case_json_path:
                raise ValueError("必须提供 case_json_path 或 case_payload。")
            payload = load_case_payload(
                Path(case_json_path),
                mode=self.config.case_input_mode,
                preserve_fields=preserve_fields,
            )
        else:
            payload = self._prepare_case_payload(case_payload, task_mode=resolved_task_mode)

        case_context = case_to_context(
            payload,
            max_field_chars=1400,
            lab_selection="sepsis_priority",
            task_mode=resolved_task_mode,
        )
        case_source = safe_path_text(Path(case_json_path)) if case_json_path else "(in-memory case payload)"
        sources: List[Dict[str, Any]] = [
            {
                "tag": "CASE1",
                "source": case_source,
                "snippet": _truncate_for_prompt(case_context, 280),
            }
        ]

        diagnosis_plan: Dict[str, Any] = {}
        assessment: Optional[Dict[str, Any]] = None
        treatment_result = self._build_empty_treatment_result()
        react_trace: List[Dict[str, Any]] = []
        tools_result = {
            "text": "",
            "sources": [],
            "sofa": {},
            "code": {},
            "static_code": {},
            "generated_code": {},
            "generated_code_goal": "",
        }
        imaging_result = {"text": "", "sources": [], "studies": []}
        rag_result = {"text": "", "sources": [], "items": []}
        reflection = {"passed": None, "issues": [], "raw_text": "", "revised_answer": "", "applied": False, "skipped": True, "error": ""}

        if resolved_task_mode in {TASK_MODE_DIAGNOSIS, TASK_MODE_DIAGNOSIS_AND_TREATMENT}:
            diagnosis_plan = self._build_sepsis3_plan()
            sofa_result: Dict[str, Any] = {}
            if self.config.enable_diagnosis_sofa_tool:
                sofa_result = self.sofa_tool.evaluate(payload)
            if self.config.enable_diagnosis_sofa_tool:
                tools_result = {
                    "text": clean_text(sofa_result.get("text", "")),
                    "sources": [],
                    "sofa": sofa_result,
                    "code": {},
                    "static_code": {},
                    "generated_code": {},
                    "generated_code_goal": "",
                }
            extraction_result = self._build_disabled_sepsis3_extraction_result()
            if self.config.enable_diagnosis_llm_extraction:
                extraction_result = self._run_sepsis3_extraction(
                    case_context=case_context,
                    sofa_result=sofa_result,
                )
            if len(extraction_result.get("attempts", [])) > 1 and extraction_result.get("ok", False):
                warnings.append("Sepsis-3 指标抽取首次返回不可解析内容，已使用重试结果。")
            assessment = self._assess_with_sepsis3(
                extraction_result=extraction_result,
                sofa_result=sofa_result,
                case_payload=payload,
                llm_extraction_enabled=self.config.enable_diagnosis_llm_extraction,
            )
            stage_errors.extend(assessment.get("stage_errors", []))

        if resolved_task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT}:
            try:
                imaging_result = self.imaging.summarize_case(payload)
            except Exception as exc:
                stage_errors.append(_make_stage_error("imaging", str(exc)))
                imaging_result = {"text": "", "sources": [], "studies": []}
            sources = self._deduplicate_sources(sources + imaging_result.get("sources", []))
            treatment_diagnosis_assessment = (
                assessment
                if resolved_task_mode == TASK_MODE_DIAGNOSIS_AND_TREATMENT
                else self._build_treatment_case_diagnosis_assessment(payload)
            )
            treatment_result = self._run_treatment_react(
                user_query=user_query,
                case_payload=payload,
                case_source=case_source,
                case_json_path=case_json_path,
                task_mode=resolved_task_mode,
                diagnosis_assessment=treatment_diagnosis_assessment,
                base_tools_result=tools_result if tools_result.get("sofa") else None,
                enable_search=enable_search,
                search_options=search_options,
                imaging_result=imaging_result,
            )
            react_trace = list(treatment_result.get("trace", []))
            if clean_text(treatment_result.get("react_error", "")):
                stage_errors.append(
                    _make_stage_error(
                        "treatment_react",
                        clean_text(treatment_result.get("react_error", "")),
                        raw_text=clean_text(treatment_result.get("raw", "")),
                    )
                )
            if clean_text(treatment_result.get("medical_generation_error", "")):
                stage_errors.append(
                    _make_stage_error(
                        "medical_treatment_generation",
                        clean_text(treatment_result.get("medical_generation_error", "")),
                        raw_text=clean_text(treatment_result.get("medical_generation_raw", "")),
                    )
                )
            tools_result = treatment_result.get("tools_result", tools_result)
            rag_result = treatment_result.get("rag_result", rag_result)
            sources = self._deduplicate_sources(sources + treatment_result.get("sources", []))

        final_answer = self._format_answer_v2(
            task_mode=resolved_task_mode,
            diagnosis_assessment=assessment if resolved_task_mode in {TASK_MODE_DIAGNOSIS, TASK_MODE_DIAGNOSIS_AND_TREATMENT} else None,
            treatment_result=treatment_result if resolved_task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT} else None,
        )
        if not final_answer and resolved_task_mode == TASK_MODE_TREATMENT:
            final_answer = clean_text(treatment_result.get("answer", "")) or clean_text(treatment_result.get("raw", ""))

        final_structured_summary = build_structured_summary(
            payload,
            tools_result,
            imaging_result,
            rag_result,
            task_mode=resolved_task_mode,
        )
        skip_reflection = bool(
            self.config.medical_treatment_backend is not None
            and resolved_task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT}
        )
        if skip_reflection:
            reflection = {
                "passed": None,
                "issues": [],
                "raw_text": "",
                "revised_answer": "",
                "applied": False,
                "error": "",
                "skipped": True,
                "skip_reason": "medical_treatment_backend_active",
            }
        else:
            reflection = self._reflect_if_needed(
                user_query=user_query,
                task_mode=resolved_task_mode,
                answer=final_answer,
                structured_summary=final_structured_summary,
            )
        if clean_text(reflection.get("error", "")):
            stage_errors.append(
                _make_stage_error(
                    "reflection",
                    clean_text(reflection.get("error", "")),
                    raw_text=clean_text(reflection.get("raw_text", "")),
                )
            )
        if clean_text(reflection.get("revised_answer", "")) and reflection.get("applied", False):
            final_answer = clean_text(reflection.get("revised_answer", ""))

        plan = {
            "task_mode": resolved_task_mode,
            "diagnosis": diagnosis_plan if resolved_task_mode in {TASK_MODE_DIAGNOSIS, TASK_MODE_DIAGNOSIS_AND_TREATMENT} else {},
            "treatment": treatment_result.get("plan", {}) if resolved_task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT} else {},
        }
        diagnosis_meta = assessment or {}
        deduped_sources = self._deduplicate_sources(sources)
        base_text_preset = identify_text_backend_preset_name(self.config.default_text_backend)
        medical_treatment_preset = identify_text_backend_preset_name(self.config.medical_treatment_backend)
        model_routes = {
            "reflection": self.config.resolve_text_backend("reflection").describe(),
            "code_writer": self.config.resolve_text_backend("code_writer").describe(),
            "diagnosis": self.config.resolve_text_backend("diagnosis").describe(),
            "treatment": self.config.resolve_text_backend("treatment").describe(),
            "imaging": self.config.resolve_vision_backend().describe(),
        }
        if self.config.medical_treatment_backend is not None:
            model_routes["medical_treatment"] = self.config.medical_treatment_backend.describe()

        return {
            "answer": final_answer,
            "case_context": case_context,
            "plan": plan,
            "tools": tools_result,
            "imaging": imaging_result,
            "rag": rag_result,
            "reflection": reflection,
            "treatment": treatment_result,
            "meta": {
                "profile_name": self.config.profile_name,
                "case_input_mode": self.config.case_input_mode,
                "task_mode": resolved_task_mode,
                "warnings": warnings,
                "stage_errors": stage_errors,
                "react_trace": react_trace,
                "structured_summary": final_structured_summary.to_dict(),
                "heuristic_assessment": {},
                "diagnosis_stage_raw": "",
                "diagnosis_stage_parse_mode": "",
                "treatment_stage_raw": clean_text(treatment_result.get("raw", "")),
                "treatment_stage_parse_mode": "react" if resolved_task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT} else "",
                "synthesis_stage_raw": "",
                "synthesis_stage_parse_mode": "",
                "format_reflection_raw": clean_text(reflection.get("raw_text", "")),
                "format_reflection_applied": bool(reflection.get("applied", False)),
                "source_refs": [f"[{item['tag']}] {item['source']}" for item in deduped_sources],
                "sepsis3_extraction_raw": clean_text(diagnosis_meta.get("sepsis3_extraction_raw", "")),
                "sepsis3_extraction": diagnosis_meta.get("sepsis3_extraction", {}),
                "sepsis3_extraction_attempts": diagnosis_meta.get("sepsis3_extraction_attempts", []),
                "sepsis3_extraction_parse_mode": clean_text(diagnosis_meta.get("sepsis3_extraction_parse_mode", "")),
                "sepsis3_judgement": diagnosis_meta.get("sepsis3_judgement", {}),
                "sepsis3_component_scores": diagnosis_meta.get("sepsis3_judgement", {}).get("component_scores", {}),
                "sepsis3_final_label": clean_text(diagnosis_meta.get("classification", "")),
                "sepsis3_failure_mode": clean_text(diagnosis_meta.get("classification", "")) or ("extraction_failed" if diagnosis_meta.get("stage_errors") else ""),
                "treatment_react_status": clean_text(treatment_result.get("react_status", "")) or clean_text(treatment_result.get("status", "")),
                "treatment_react_error": clean_text(treatment_result.get("react_error", "")) or clean_text(treatment_result.get("error", "")),
                "treatment_react_rounds": len(react_trace),
                "treatment_tool_call_count": len(treatment_result.get("tool_calls", [])),
                "treatment_search_enabled": bool(self.config.builtin_search_default if enable_search is None else enable_search),
                "treatment_generation_mode": clean_text(treatment_result.get("treatment_generation_mode", "")) or "fullstack_base",
                "base_text_preset": base_text_preset,
                "medical_treatment_preset": medical_treatment_preset,
                "model_routes": model_routes,
            },
        }

    def _merge_tool_results(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(left)
        merged["text"] = "\n\n".join(part for part in [clean_text(left.get("text", "")), clean_text(right.get("text", ""))] if part).strip()
        merged["sources"] = list(left.get("sources", [])) + list(right.get("sources", []))
        if right.get("sofa"):
            merged["sofa"] = right.get("sofa", {})
        if right.get("static_code"):
            merged["static_code"] = right.get("static_code", {})
            merged["code"] = right.get("static_code", {})
        if right.get("generated_code"):
            merged["generated_code"] = right.get("generated_code", {})
        if clean_text(right.get("generated_code_goal", "")):
            merged["generated_code_goal"] = clean_text(right.get("generated_code_goal", ""))
        return merged

    def _run_generated_code_tool(
        self,
        *,
        user_query: str,
        case_payload: Dict[str, Any],
        case_json_path: Optional[str],
        goal_hint: str,
    ) -> Dict[str, Any]:
        if not self.config.enable_generated_code:
            return {"text": "", "sources": [], "generated_code": {}, "generated_code_goal": clean_text(goal_hint)}

        sources: List[Dict[str, Any]] = []
        tool_lines: List[str] = []
        generated_result: Dict[str, Any] = {}
        if not case_json_path:
            return {
                "text": "Qwen 代码工具跳过：未提供 case_json_path。",
                "sources": [],
                "generated_code": {},
                "generated_code_goal": clean_text(goal_hint),
            }

        try:
            candidate_dirs = collect_candidate_study_paths(case_payload, self.config.case_image_root)
            generated_result = self._get_generated_code_tool().run_generated_probe(
                user_query=user_query,
                case_payload=case_payload,
                case_json_path=Path(case_json_path),
                candidate_dirs=candidate_dirs,
                timeout=int(self.config.http_timeout_seconds),
                goal_hint=goal_hint,
            )
            if clean_text(generated_result.get("script_path", "")):
                sources.append(
                    {
                        "tag": "TOOL2",
                        "source": clean_text(generated_result["script_path"]),
                        "snippet": clean_text(generated_result.get("stdout", ""))[:280],
                    }
                )
            sanitized_generated_text = _build_clinical_tool_text({"generated_code": generated_result, "static_code": {}, "sofa": {}})
            if sanitized_generated_text:
                tool_lines.append("Qwen 生成代码工具结果：")
                tool_lines.append(sanitized_generated_text)
        except Exception as exc:
            tool_lines.append(f"Qwen 生成代码工具失败：{exc}")

        return {
            "text": "\n".join(line for line in tool_lines if clean_text(line)).strip(),
            "sources": sources,
            "generated_code": generated_result,
            "generated_code_goal": clean_text(goal_hint),
        }

    def _format_answer(self, assessment: Dict[str, Any]) -> str:
        lines = [
            f"诊断结果：{clean_text(assessment.get('diagnosis_result', ''))}",
            _render_labeled_section("诊断依据", _normalize_string_list(assessment.get("diagnostic_basis"))),
        ]
        treatment_advice = _normalize_string_list(assessment.get("treatment_advice"))
        if treatment_advice:
            lines.append(_render_labeled_section("治疗建议", treatment_advice))
        return "\n".join(lines).strip()

    def _merge_reflection_revised_answer(
        self,
        *,
        task_mode: str,
        current_answer: str,
        candidate_answer: str,
    ) -> str:
        resolved_task_mode = _validate_task_mode(task_mode)
        current_text = clean_text(current_answer)
        candidate_text = clean_text(candidate_answer)
        if not candidate_text:
            return ""

        if resolved_task_mode == TASK_MODE_TREATMENT:
            current_parsed = _parse_treatment_stage_response(current_text)
            candidate_parsed = _parse_treatment_stage_response(candidate_text)
            merged_advice = (
                _normalize_string_list(candidate_parsed.get("treatment_advice"))
                or _normalize_string_list(current_parsed.get("treatment_advice"))
            )
            treatment_payload = {
                "treatment_advice": merged_advice,
                "answer": candidate_text or current_text,
                "raw": candidate_text or current_text,
            }
            return self._format_answer_v2(
                task_mode=resolved_task_mode,
                diagnosis_assessment=None,
                treatment_result=treatment_payload,
            )

        current_parsed = _parse_assessment_response(current_text)
        candidate_parsed = _parse_assessment_response(candidate_text)

        diagnosis_result = clean_text(candidate_parsed.get("diagnosis_result", "")) or clean_text(current_parsed.get("diagnosis_result", ""))
        diagnostic_basis = (
            _normalize_string_list(candidate_parsed.get("diagnostic_basis"))
            or _normalize_string_list(current_parsed.get("diagnostic_basis"))
        )
        treatment_advice = (
            _normalize_string_list(candidate_parsed.get("treatment_advice"))
            or _normalize_string_list(current_parsed.get("treatment_advice"))
        )

        diagnosis_payload = None
        if diagnosis_result or diagnostic_basis:
            diagnosis_payload = {
                "diagnosis_result": diagnosis_result,
                "diagnostic_basis": diagnostic_basis,
            }

        treatment_payload = None
        if resolved_task_mode == TASK_MODE_DIAGNOSIS_AND_TREATMENT and treatment_advice:
            treatment_payload = {
                "treatment_advice": treatment_advice,
                "answer": candidate_text or current_text,
                "raw": candidate_text or current_text,
            }

        return self._format_answer_v2(
            task_mode=resolved_task_mode,
            diagnosis_assessment=diagnosis_payload,
            treatment_result=treatment_payload,
        )

    def _reflect_if_needed(
        self,
        *,
        user_query: str,
        task_mode: str,
        answer: str,
        structured_summary: StructuredSummary,
    ) -> Dict[str, Any]:
        if not self.config.enable_reflection:
            return {"passed": None, "issues": [], "raw_text": "", "revised_answer": "", "applied": False, "error": "", "skipped": True}

        summary_prompt = render_summary_prompt(structured_summary)
        if task_mode == TASK_MODE_DIAGNOSIS:
            format_instruction = "若需修订，请输出完整的“诊断结果 / 诊断依据”两段。"
        elif task_mode == TASK_MODE_TREATMENT:
            format_instruction = "若需修订，请只输出完整的“治疗建议”一段。"
        else:
            format_instruction = "若需修订，请输出完整的“诊断结果 / 诊断依据 / 治疗建议”三段。"
        prompt = (
            f"用户问题：{user_query}\n\n"
            f"{summary_prompt}\n\n"
            f"当前答案：\n{answer}\n\n"
            "请检查答案是否与证据一致、是否存在遗漏或表述不当。"
            f"{format_instruction} 若需要修订，请在 revised_answer 中给出完整修订稿。"
        )
        raw_text = ""
        error = ""
        try:
            raw_text = self._generate_text(
                backend_config=self.config.resolve_text_backend("reflection"),
                system_prompt=REFLECTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_new_tokens=self.config.reflection_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
            )
        except Exception as exc:
            error = str(exc)
            raw_text = ""

        payload = extract_json_object(raw_text) or {}
        if not payload:
            if not error:
                error = "empty_or_non_json_response"
            return {
                "passed": None,
                "issues": [],
                "raw_text": clean_text(raw_text),
                "revised_answer": "",
                "applied": False,
                "error": clean_text(error),
                "skipped": False,
            }

        passed = _parse_bool(payload.get("pass"), True)
        issues = _normalize_string_list(payload.get("issues"))
        revised_answer = ""
        applied = False

        candidate_answer = clean_text(payload.get("revised_answer", ""))
        if candidate_answer and not passed:
            revised_answer = self._merge_reflection_revised_answer(
                task_mode=task_mode,
                current_answer=answer,
                candidate_answer=candidate_answer,
            )
            if revised_answer:
                applied = True

        return {
            "passed": passed,
            "issues": issues,
            "raw_text": clean_text(raw_text),
            "revised_answer": revised_answer,
            "applied": applied,
            "error": clean_text(error),
            "skipped": False,
        }

    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for source in sources:
            key = (source.get("tag"), source.get("source"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(source)
        return deduped

    def _complete_chat_with_tools(
        self,
        *,
        backend_config: Any,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        request_timeout_seconds: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        extra_body_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        backend = self._get_text_backend(backend_config)
        if not hasattr(backend, "complete_chat"):
            raise RuntimeError(f"backend does not support tool calling: {backend_config.describe()}")
        return backend.complete_chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            request_timeout_seconds=request_timeout_seconds,
            tools=tools,
            tool_choice=tool_choice,
            extra_body_override=extra_body_override,
        )

    def _build_empty_treatment_result(
        self,
        *,
        status: str = "skipped",
        error: str = "",
        raw_text: str = "",
        treatment_advice: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        advice = _normalize_string_list(treatment_advice)
        return {
            "triggered": False,
            "status": clean_text(status) or "skipped",
            "error": clean_text(error),
            "react_status": clean_text(status) or "skipped",
            "react_error": clean_text(error),
            "raw": clean_text(raw_text),
            "treatment_advice": advice,
            "answer": "",
            "treatment_generation_mode": "fullstack_base",
            "trace": [],
            "tool_calls": [],
            "sources": [],
            "tools_result": {
                "text": "",
                "sources": [],
                "sofa": {},
                "code": {},
                "static_code": {},
                "generated_code": {},
                "generated_code_goal": "",
            },
            "rag_result": {"text": "", "sources": [], "items": []},
            "structured_summary": {},
            "plan": {
                "task_category": "treatment_react",
                "intent": "治疗建议",
                "use_rag": bool(self.config.enable_local_rag),
                "use_data_tools": True,
                "use_generated_code": bool(self.config.enable_generated_code),
                "use_builtin_search": bool(self.config.builtin_search_default),
            },
        }

    def _build_case_sections(
        self,
        *,
        task_mode: str,
        case_payload: Dict[str, Any],
        case_context: str,
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        imaging_result: Dict[str, Any],
    ) -> Dict[str, str]:
        payload = normalize_case_payload(case_payload)
        sections = {
            "case_overview": case_to_context(
                payload,
                max_field_chars=700,
                lab_selection="sepsis_priority",
                task_mode=task_mode,
            ),
            "structured_summary": render_summary_prompt(structured_summary),
            "chief_complaint": normalize_text(payload.get("主诉", "")),
            "present_illness": normalize_text(payload.get("现病史", "")),
            "past_history": normalize_text(payload.get("既往史", "")),
            "personal_history": normalize_text(payload.get("个人史", "")),
            "allergy_history": normalize_text(payload.get("过敏史", "")),
            "physical_exam": normalize_text(payload.get("体格检查", "")),
            "specialist_exam": normalize_text(payload.get("专科检查", "")),
            "lab_and_auxiliary": get_case_auxiliary_exam_text(payload, max_lab_items=80, lab_selection="sepsis_priority"),
            "admission_status": normalize_text(payload.get("入院情况", "")),
            "clinical_course": normalize_text(payload.get("诊疗过程", "")),
            "imaging_summary": normalize_text(imaging_result.get("text", "")),
            "full_context": case_context,
        }
        if diagnosis_assessment and (
            clean_text(diagnosis_assessment.get("diagnosis_result", ""))
            or _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))
        ):
            sections["diagnosis_context"] = "\n".join(
                [
                    f"诊断结果：{clean_text(diagnosis_assessment.get('diagnosis_result', ''))}",
                    _render_labeled_section("诊断依据", _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))),
                ]
            ).strip()
        return {name: text for name, text in sections.items() if clean_text(text)}

    def _build_case_chunks(self, case_sections: Dict[str, str]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        for section_name, section_text in case_sections.items():
            for index, chunk in enumerate(split_text_to_chunks(section_text, chunk_size=900, overlap=120), start=1):
                chunks.append(
                    {
                        "id": f"{section_name}:{index}",
                        "section": section_name,
                        "text": chunk,
                    }
                )
        return chunks

    def _resolve_case_section_names(self, requested_names: List[str], case_sections: Dict[str, str]) -> List[str]:
        available = list(case_sections.keys())
        lowered = {name.lower(): name for name in available}
        resolved: List[str] = []
        for requested_name in requested_names:
            normalized = clean_text(requested_name).lower().replace("-", "_").replace(" ", "_")
            if not normalized:
                continue
            if normalized in lowered:
                actual = lowered[normalized]
            else:
                actual = next(
                    (
                        candidate
                        for candidate in available
                        if normalized in candidate.lower() or candidate.lower() in normalized
                    ),
                    "",
                )
            if actual and actual not in resolved:
                resolved.append(actual)
        return resolved

    def _search_case_chunks(self, *, query: str, case_chunks: List[Dict[str, Any]], max_chunks: int) -> List[Dict[str, Any]]:
        normalized_query = normalize_text(query)
        if not normalized_query:
            return []
        query_lower = normalized_query.lower()
        terms: List[str] = []
        for token in re.findall(r"[A-Za-z]{2,}|[\u4e00-\u9fff]{1,8}|\d+(?:\.\d+)?", query_lower):
            if token and token not in terms:
                terms.append(token)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for chunk in case_chunks:
            chunk_text = normalize_text(chunk.get("text", ""))
            chunk_lower = chunk_text.lower()
            score = 0.0
            if query_lower in chunk_lower:
                score += 6.0
            for term in terms:
                if term in chunk_lower:
                    score += 1.0 + min(3, chunk_lower.count(term)) * 0.25
            if score <= 0:
                continue
            scored.append((score, chunk))
        scored.sort(key=lambda item: (item[0], item[1].get("id", "")), reverse=True)

        results: List[Dict[str, Any]] = []
        for score, chunk in scored[: max(1, min(max_chunks, 6))]:
            snippet = clean_text(chunk.get("text", ""))[:420]
            results.append(
                {
                    "id": chunk.get("id", ""),
                    "section": chunk.get("section", ""),
                    "score": round(score, 2),
                    "snippet": f"{snippet} [CASE1]",
                }
            )
        return results

    def _merge_rag_results(self, current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []
        seen = set()
        for item in list(current.get("items", [])) + list(new.get("items", [])):
            if not isinstance(item, dict):
                continue
            key = (clean_text(item.get("path", "")), clean_text(item.get("page", "")), clean_text(item.get("text", "")))
            if key in seen:
                continue
            seen.add(key)
            items.append(item)

        text_lines: List[str] = []
        sources: List[Dict[str, Any]] = []
        for index, item in enumerate(items, start=1):
            tag = f"RAG{index}"
            path_text = clean_text(item.get("path", ""))
            page_text = clean_text(item.get("page", ""))
            source = f"{path_text}#page={page_text}" if path_text and page_text else path_text
            snippet = clean_text(item.get("text", ""))[:280]
            title = clean_text(item.get("title", "")) or Path(path_text).name if path_text else "knowledge"
            if snippet:
                page_label = f" 第{page_text}页" if page_text else ""
                text_lines.append(f"[{tag}] {title}{page_label}：{snippet}")
                sources.append({"tag": tag, "source": source, "snippet": snippet})
        return {
            "items": items,
            "text": "\n".join(text_lines).strip(),
            "sources": sources,
        }

    def _build_treatment_tool_specs(self, case_sections: Dict[str, str]) -> List[Dict[str, Any]]:
        available_sections = list(case_sections.keys())
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_case_digest",
                    "description": "获取病例高价值摘要、当前结构化摘要和可用原始病历字段名。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_case_sections",
                    "description": "按字段名读取原始病历段落，适合查看主诉、现病史、体格检查、化验等全文。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "section_names": {
                                "type": "array",
                                "items": {"type": "string", "enum": available_sections},
                                "minItems": 1,
                                "maxItems": 4,
                            }
                        },
                        "required": ["section_names"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_case_text",
                    "description": "在长病历切块索引中检索与查询最相关的病例片段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "max_chunks": {"type": "integer", "minimum": 1, "maximum": 6, "default": 3},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_sofa_assessment",
                    "description": "运行或读取当前病例的 SOFA/血流动力学工具结果，返回评分与治疗关键指标。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_local_guidelines",
                    "description": "检索本地 `Agent/rag` 文献库中的指南/综述片段。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 6, "default": 3},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if self.config.enable_generated_code:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "run_generated_probe",
                        "description": "调用只读生成代码工具补充提取病例中的结构化线索；无 case_json_path 时会优雅降级。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string"},
                            },
                            "required": ["goal"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        disabled_tools = self._get_disabled_treatment_tools()
        if disabled_tools:
            tools = [
                tool
                for tool in tools
                if clean_text(((tool.get("function") or {}) if isinstance(tool, dict) else {}).get("name", "")) not in disabled_tools
            ]
        return tools

    def _get_disabled_treatment_tools(self) -> set[str]:
        return {
            clean_text(tool_name)
            for tool_name in getattr(self.config, "disabled_treatment_tools", ())
            if clean_text(tool_name)
        }

    def _scorecard_add(self, memory: Dict[str, List[str]], field: str, value: str, *, limit: int = 8) -> None:
        text = _truncate_for_prompt(value, 360)
        if not text:
            return
        bucket = memory.setdefault(field, [])
        if text not in bucket:
            bucket.append(text)
        if len(bucket) > limit:
            del bucket[:-limit]

    def _extract_scorecard_lines(self, text: str, markers: Sequence[str], *, limit: int = 5) -> List[str]:
        normalized = normalize_text(text)
        if not normalized:
            return []
        parts = re.split(r"[\n。；;]+", normalized)
        matched: List[str] = []
        for part in parts:
            line = clean_text(part)
            if not line:
                continue
            if any(marker.lower() in line.lower() for marker in markers):
                matched.append(line)
            if len(matched) >= limit:
                break
        return matched

    def _build_initial_treatment_scorecard_memory(
        self,
        *,
        case_sections: Dict[str, str],
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        sources: Sequence[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        memory: Dict[str, List[str]] = {field: [] for field in TREATMENT_SCORECARD_FIELDS}
        self._scorecard_add(
            memory,
            "completeness_targets",
            "最终建议需覆盖：抗感染/病原学、感染源控制、液体复苏与循环支持、器官支持、监测复评、补充检查。",
        )
        self._scorecard_add(
            memory,
            "actionability_targets",
            "每条建议应写成可执行动作：用药覆盖范围、培养/药敏、操作措施、监测指标、复评节点、升级或降阶梯条件。",
        )
        self._scorecard_add(
            memory,
            "unsupported_or_uncertain_claims",
            "不得编造病例中没有的具体数值；证据不足时写“需补充/复查/完善”。",
        )
        self._update_treatment_scorecard_memory(
            memory,
            case_sections=case_sections,
            structured_summary=structured_summary,
            diagnosis_assessment=diagnosis_assessment,
            sources=sources,
            tools_result={},
            rag_result={"text": "", "sources": [], "items": []},
            tool_name="initial_case_context",
            tool_content="\n".join(
                [
                    case_sections.get("case_overview", ""),
                    case_sections.get("diagnosis_context", ""),
                    structured_summary.prompt_text,
                ]
            ),
        )
        return memory

    def _update_treatment_scorecard_memory(
        self,
        memory: Dict[str, List[str]],
        *,
        case_sections: Dict[str, str],
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        sources: Sequence[Dict[str, Any]],
        tools_result: Dict[str, Any],
        rag_result: Dict[str, Any],
        tool_name: str = "",
        tool_content: str = "",
    ) -> Dict[str, List[str]]:
        diagnosis_context = clean_text(case_sections.get("diagnosis_context", ""))
        if diagnosis_assessment:
            diagnosis_result = clean_text(diagnosis_assessment.get("diagnosis_result", ""))
            if diagnosis_result:
                self._scorecard_add(memory, "clinical_reasonableness_evidence", f"诊断背景：{diagnosis_result}")
            for item in _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))[:4]:
                self._scorecard_add(memory, "clinical_reasonableness_evidence", f"诊断依据：{item}")
        if diagnosis_context:
            self._scorecard_add(memory, "clinical_reasonableness_evidence", f"病例诊断上下文：{diagnosis_context}")

        evidence_pool = "\n".join(
            clean_text(item)
            for item in (
                case_sections.get("case_overview", ""),
                structured_summary.prompt_text,
                tool_content,
                clean_text(tools_result.get("text", "")) if isinstance(tools_result, dict) else "",
                clean_text(rag_result.get("text", "")) if isinstance(rag_result, dict) else "",
            )
            if clean_text(item)
        )
        clinical_markers = ("感染", "发热", "寒战", "休克", "脓毒症", "肺炎", "腹膜炎", "胆管炎", "脓肿", "穿孔", "培养")
        safety_markers = (
            "过敏",
            "肾",
            "肌酐",
            "肝",
            "胆红素",
            "凝血",
            "血小板",
            "出血",
            "血栓",
            "低氧",
            "休克",
            "乳酸",
            "糖尿病",
            "高血糖",
        )
        action_markers = (
            "抗感染",
            "抗生素",
            "引流",
            "手术",
            "清创",
            "补液",
            "复苏",
            "升压",
            "氧疗",
            "呼吸",
            "CRRT",
            "监测",
            "复查",
            "培养",
        )
        for line in self._extract_scorecard_lines(evidence_pool, clinical_markers, limit=5):
            self._scorecard_add(memory, "clinical_reasonableness_evidence", line)
        for line in self._extract_scorecard_lines(evidence_pool, safety_markers, limit=6):
            self._scorecard_add(memory, "safety_evidence", line)
        for line in self._extract_scorecard_lines(evidence_pool, action_markers, limit=6):
            self._scorecard_add(memory, "actionability_targets", line)

        sofa_payload = tools_result.get("sofa", {}) if isinstance(tools_result, dict) else {}
        if isinstance(sofa_payload, dict) and sofa_payload:
            sofa_text = clean_text(sofa_payload.get("text", ""))
            sofa_total = sofa_payload.get("computed_total") or sofa_payload.get("reported_total")
            self._scorecard_add(
                memory,
                "clinical_reasonableness_evidence",
                f"SOFA工具结果：{('SOFA=' + str(sofa_total) + '；') if sofa_total is not None else ''}{_truncate_for_prompt(sofa_text, 260)}",
            )
            key_indicators = sofa_payload.get("key_indicators", {})
            if isinstance(key_indicators, dict) and key_indicators:
                self._scorecard_add(memory, "safety_evidence", f"器官功能/安全相关指标：{_truncate_for_prompt(json.dumps(key_indicators, ensure_ascii=False), 320)}")

        rag_text = clean_text(rag_result.get("text", "")) if isinstance(rag_result, dict) else ""
        if rag_text:
            self._scorecard_add(memory, "evidence_grounding", f"本地指南证据：{_truncate_for_prompt(rag_text, 360)}")
        if tool_name:
            self._scorecard_add(memory, "evidence_grounding", f"工具证据来源：{tool_name}")
        for source in self._deduplicate_sources(list(sources))[:6]:
            tag = clean_text(source.get("tag", ""))
            snippet = clean_text(source.get("snippet", ""))
            if tag or snippet:
                self._scorecard_add(memory, "evidence_grounding", f"[{tag or 'SOURCE'}] {_truncate_for_prompt(snippet, 260)}")
        return memory

    def _render_treatment_scorecard_memory(self, memory: Dict[str, List[str]]) -> str:
        labels = {
            "clinical_reasonableness_evidence": "临床合理性证据",
            "safety_evidence": "安全性证据",
            "evidence_grounding": "证据依据",
            "unsupported_or_uncertain_claims": "不得写死/需补充的信息",
            "completeness_targets": "完整性覆盖目标",
            "actionability_targets": "可执行性目标",
        }
        lines: List[str] = []
        for field in TREATMENT_SCORECARD_FIELDS:
            items = _normalize_string_list(memory.get(field, []))
            if not items:
                continue
            lines.append(f"{labels.get(field, field)}：")
            lines.extend(f"- {item}" for item in items[:8])
        return "\n".join(lines).strip() or "暂无评分维度证据记忆。"

    def _treatment_scorecard_evidence_text(
        self,
        memory: Dict[str, List[str]],
        case_sections: Dict[str, str],
        sources: Sequence[Dict[str, Any]],
        structured_summary: Optional[StructuredSummary] = None,
    ) -> str:
        return "\n".join(
            [
                self._render_treatment_scorecard_memory(memory),
                case_sections.get("case_overview", ""),
                case_sections.get("diagnosis_context", ""),
                case_sections.get("lab_and_auxiliary", ""),
                case_sections.get("clinical_course", ""),
                structured_summary.prompt_text if structured_summary is not None else "",
                self._build_treatment_source_digest(sources, limit=8),
            ]
        )

    def _treatment_scorecard_ready(self, memory: Dict[str, List[str]], tool_calls: Sequence[Dict[str, Any]]) -> bool:
        if len(tool_calls) < 2:
            return False
        required_fields = (
            "clinical_reasonableness_evidence",
            "safety_evidence",
            "evidence_grounding",
            "completeness_targets",
            "actionability_targets",
        )
        return all(bool(_normalize_string_list(memory.get(field, []))) for field in required_fields)

    def _build_treatment_react_user_prompt(
        self,
        *,
        user_query: str,
        case_sections: Dict[str, str],
        structured_summary: StructuredSummary,
        max_rounds: int,
    ) -> str:
        available_sections = "、".join(case_sections.keys())
        diagnosis_context = clean_text(case_sections.get("diagnosis_context", ""))
        prompt_parts = [
            f"用户请求：{clean_text(user_query) or '请给出治疗建议。'}",
            "本阶段只做信息收集，不输出最终治疗建议正文。请使用官方 tool-calling 风格按需补齐评分所需证据。",
            f"最多允许 {max_rounds} 轮工具调用。",
            f"可直接读取的病例摘要：\n{_truncate_for_prompt(case_sections.get('case_overview', ''), 1200)}",
            f"当前结构化摘要：\n{_truncate_for_prompt(structured_summary.prompt_text, 1600)}",
            f"可按需读取的原始字段：{available_sections}",
        ]
        if diagnosis_context:
            prompt_parts.append(f"已提供的诊断上下文：\n{diagnosis_context}")
        prompt_parts.append(
            "请按以下缺口决定下一步工具调用：\n"
            "1. 临床合理性：感染诊断、疑似感染源、病原学证据、病情严重程度、当前治疗背景。\n"
            "2. 安全性：过敏史、肝肾功能、凝血/血小板、休克、低氧、出血或血栓风险。\n"
            "3. 证据依据性：病例事实、SOFA/器官功能证据、本地指南依据；没有证据的结论写入不确定。\n"
            "4. 完整性：抗感染、感染源控制、液体复苏/循环支持、器官支持、监测复评、补充检查。\n"
            "5. 可执行性：药物覆盖范围、培养/药敏、操作措施、监测指标、复评频率、升级或降阶梯条件。\n"
            "如果上述证据已经足够，请不要再调用工具，只输出一句“证据已足够，可进入最终定稿”。\n"
            "禁止在本阶段输出“治疗建议：”段落、病例关键信息汇总、工具调用计划或 SOFA 计算过程。"
        )
        return "\n\n".join(part for part in prompt_parts if clean_text(part)).strip()

    def _build_treatment_source_digest(self, sources: Sequence[Dict[str, Any]], *, limit: int = 6) -> str:
        lines: List[str] = []
        for source in self._deduplicate_sources(list(sources))[: max(1, limit)]:
            tag = clean_text(source.get("tag", ""))
            source_name = clean_text(source.get("source", ""))
            snippet = clean_text(source.get("snippet", ""))
            label = f"[{tag}] {source_name}" if tag else source_name
            if snippet:
                lines.append(f"{label}：{_truncate_for_prompt(snippet, 220)}")
            elif label:
                lines.append(label)
        return "\n".join(lines).strip() or "暂无额外工具证据。"

    def _build_treatment_finalization_prompt(
        self,
        *,
        user_query: str,
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        case_sections: Dict[str, str],
        sources: Sequence[Dict[str, Any]],
        candidate_answer: str,
        quality_flags: Sequence[str],
        treatment_scorecard_memory: Optional[Dict[str, List[str]]] = None,
        collection_summary: str = "",
    ) -> str:
        diagnosis_lines: List[str] = []
        if diagnosis_assessment:
            diagnosis_result = clean_text(diagnosis_assessment.get("diagnosis_result", ""))
            if diagnosis_result:
                diagnosis_lines.append("诊断背景：" + diagnosis_result)
            diagnosis_basis = _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))
            if diagnosis_basis:
                diagnosis_lines.append(_render_labeled_section("诊断依据", diagnosis_basis))
        diagnosis_text = "\n".join(line for line in diagnosis_lines if clean_text(line)).strip() or "未提供明确诊断背景。"
        issue_text = "、".join(_normalize_string_list(list(quality_flags))) or "需要整理为最终治疗建议"
        scorecard_text = self._render_treatment_scorecard_memory(treatment_scorecard_memory or {})
        collection_text = clean_text(collection_summary)
        collection_block = (
            f"\n\nReAct收集阶段摘要（只能作为证据索引，不得复述为正文）：\n{_truncate_for_prompt(collection_text, 1200)}"
            if collection_text
            else ""
        )
        return (
            f"用户请求：{clean_text(user_query) or '请给出治疗建议。'}\n\n"
            f"需要避免的问题：{issue_text}\n\n"
            "硬性要求：\n"
            "1. 只输出“治疗建议：”一节，使用编号分点；不要输出病例摘要、分析过程、JSON、工具名、字段名或 section 名称。\n"
            "2. 至少写 5 条可执行动作，每条都必须是治疗措施、监测复评措施或补充检查计划，不能只是病例事实。\n"
            "3. 必须覆盖抗感染/病原学、感染源控制、液体复苏/循环支持、器官支持、监测复评、补充计划；确无证据时写“需评估/需补充/需复查”。\n"
            "4. 只能使用下方 scorecard memory 和病例事实中已经确认的信息；不得编造 PCT、CRP、肌酐、血小板、乳酸等具体数值。\n"
            "5. 需要体现安全性：根据过敏、肝肾功能、凝血/血小板、休克、低氧、出血/血栓风险进行用药和支持治疗约束。\n\n"
            f"评分维度 evidence memory：\n{scorecard_text}\n\n"
            f"病例概览：\n{_truncate_for_prompt(case_sections.get('case_overview', ''), 1600)}\n\n"
            f"{diagnosis_text}\n\n"
            f"{render_summary_prompt(structured_summary)}\n\n"
            f"已收集证据来源摘要：\n{self._build_treatment_source_digest(sources, limit=8)}"
            f"{collection_block}\n\n"
            f"候选/中间文本（仅用于识别要修正的问题，不得照抄）：\n{_truncate_for_prompt(candidate_answer, 1200)}"
        )

    def _finalize_treatment_answer(
        self,
        *,
        treatment_backend: Any,
        user_query: str,
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        case_sections: Dict[str, str],
        sources: Sequence[Dict[str, Any]],
        candidate_answer: str,
        quality_flags: Sequence[str],
        treatment_scorecard_memory: Optional[Dict[str, List[str]]] = None,
        collection_summary: str = "",
    ) -> Dict[str, Any]:
        raw_text = ""
        error = ""
        max_new_tokens = max(
            self.config.generation_max_new_tokens,
            int(getattr(treatment_backend, "default_max_new_tokens", 0) or 0),
            512,
        )
        timeout_seconds = max(
            self.config.http_timeout_seconds,
            float(getattr(treatment_backend, "default_timeout_seconds", 0.0) or 0.0),
            60.0,
        )
        try:
            raw_text = self._generate_text(
                backend_config=treatment_backend,
                system_prompt=TREATMENT_FINALIZATION_SYSTEM_PROMPT,
                user_prompt=self._build_treatment_finalization_prompt(
                    user_query=user_query,
                    structured_summary=structured_summary,
                    diagnosis_assessment=diagnosis_assessment,
                    case_sections=case_sections,
                    sources=sources,
                    candidate_answer=candidate_answer,
                    quality_flags=quality_flags,
                    treatment_scorecard_memory=treatment_scorecard_memory,
                    collection_summary=collection_summary,
                ),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=self.config.generation_repetition_penalty,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = str(exc)

        parsed = _parse_treatment_stage_response(raw_text)
        treatment_advice = _normalize_string_list(parsed.get("treatment_advice"))
        answer = _render_treatment_answer(raw_text, treatment_advice)
        if not error and not answer:
            error = "empty_treatment_answer"
        evidence_text = self._treatment_scorecard_evidence_text(
            treatment_scorecard_memory or {},
            case_sections,
            sources,
            structured_summary=structured_summary,
        )
        return {
            "answer": answer,
            "raw_text": clean_text(raw_text),
            "treatment_advice": treatment_advice,
            "error": clean_text(error),
            "quality_flags": _treatment_quality_flags(raw_text, treatment_advice, evidence_text=evidence_text),
        }

    def _build_treatment_extra_body_override(
        self,
        *,
        enable_search: Optional[bool],
        search_options: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        enabled = self.config.builtin_search_default if enable_search is None else bool(enable_search)
        payload: Dict[str, Any] = {}
        if self.config.builtin_search_options:
            payload.update(dict(self.config.builtin_search_options))
        if search_options:
            payload.update(dict(search_options))
        if enable_search is not None or payload:
            payload["enable_search"] = enabled
        return payload or None

    def _execute_treatment_tool_call(
        self,
        *,
        tool_call: Dict[str, Any],
        user_query: str,
        case_payload: Dict[str, Any],
        case_json_path: Optional[str],
        case_sections: Dict[str, str],
        case_chunks: List[Dict[str, Any]],
        task_mode: str,
        diagnosis_assessment: Optional[Dict[str, Any]],
        structured_summary: StructuredSummary,
        tools_result: Dict[str, Any],
        rag_result: Dict[str, Any],
        imaging_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        tool_name = clean_text(tool_call.get("name", ""))
        disabled_tools = self._get_disabled_treatment_tools()
        raw_arguments = clean_text(tool_call.get("arguments", "")) or "{}"
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
        except Exception:
            arguments = {}

        response_payload: Dict[str, Any]
        next_tools_result = tools_result
        next_rag_result = rag_result
        source_updates: List[Dict[str, Any]] = []

        if tool_name in disabled_tools:
            response_payload = {
                "error": f"tool_disabled:{tool_name}",
                "message": f"Tool {tool_name} is disabled for this run.",
            }
        elif tool_name == "get_case_digest":
            response_payload = {
                "case_overview": case_sections.get("case_overview", ""),
                "structured_summary": structured_summary.prompt_text,
                "diagnosis_context": case_sections.get("diagnosis_context", ""),
                "available_sections": list(case_sections.keys()),
            }
        elif tool_name == "read_case_sections":
            requested = ensure_list(arguments.get("section_names"))
            resolved = self._resolve_case_section_names([clean_text(item) for item in requested], case_sections)
            response_payload = {
                "sections": {name: case_sections.get(name, "") for name in resolved},
                "resolved_section_names": resolved,
                "available_sections": list(case_sections.keys()),
            }
        elif tool_name == "search_case_text":
            query = clean_text(arguments.get("query", ""))
            max_chunks = int(arguments.get("max_chunks", 3) or 3)
            response_payload = {
                "query": query,
                "matches": self._search_case_chunks(query=query, case_chunks=case_chunks, max_chunks=max_chunks),
            }
        elif tool_name == "run_sofa_assessment":
            sofa_payload = next_tools_result.get("sofa", {})
            if not isinstance(sofa_payload, dict) or not sofa_payload:
                sofa_payload = self.sofa_tool.evaluate(case_payload)
                source_updates = [
                    {
                        "tag": "TOOL_SOFA",
                        "source": "SOFA tool",
                        "snippet": clean_text(sofa_payload.get("text", ""))[:280],
                    }
                ]
                next_tools_result = self._merge_tool_results(
                    next_tools_result,
                    {
                        "text": clean_text(sofa_payload.get("text", "")),
                        "sources": source_updates,
                        "sofa": sofa_payload,
                        "code": {},
                        "static_code": {},
                        "generated_code": {},
                        "generated_code_goal": "",
                    },
                )
            response_payload = {
                "reported_total": sofa_payload.get("reported_total"),
                "computed_total": sofa_payload.get("computed_total"),
                "known_component_count": sofa_payload.get("known_component_count"),
                "component_scores": sofa_payload.get("component_scores", {}),
                "key_indicators": sofa_payload.get("key_indicators", {}),
                "text": clean_text(sofa_payload.get("text", "")),
            }
        elif tool_name == "search_local_guidelines":
            query = clean_text(arguments.get("query", ""))
            top_k = int(arguments.get("top_k", 3) or 3)
            if not self.config.enable_local_rag:
                response_payload = {"query": query, "matches": [], "message": "本地 RAG 已禁用。"}
            else:
                single_rag_result = self.rag.search([query], top_k=max(1, min(top_k, self.config.rag_top_k)))
                next_rag_result = self._merge_rag_results(next_rag_result, single_rag_result)
                source_updates = list(next_rag_result.get("sources", []))
                response_payload = {
                    "query": query,
                    "matches": [
                        {
                            "title": clean_text(item.get("title", "")),
                            "path": clean_text(item.get("path", "")),
                            "page": item.get("page"),
                            "snippet": clean_text(item.get("text", ""))[:280],
                        }
                        for item in single_rag_result.get("items", [])
                    ],
                    "text": clean_text(single_rag_result.get("text", "")),
                }
        elif tool_name == "run_generated_probe":
            goal = clean_text(arguments.get("goal", ""))
            generated_result = self._run_generated_code_tool(
                user_query=user_query,
                case_payload=case_payload,
                case_json_path=case_json_path,
                goal_hint=goal,
            )
            next_tools_result = self._merge_tool_results(
                next_tools_result,
                {
                    "text": clean_text(generated_result.get("text", "")),
                    "sources": list(generated_result.get("sources", [])),
                    "sofa": {},
                    "code": {},
                    "static_code": {},
                    "generated_code": generated_result.get("generated_code", {}),
                    "generated_code_goal": clean_text(generated_result.get("generated_code_goal", "")),
                },
            )
            source_updates = list(generated_result.get("sources", []))
            response_payload = {
                "goal": goal,
                "stdout": clean_text(generated_result.get("generated_code", {}).get("stdout", "")),
                "stderr": clean_text(generated_result.get("generated_code", {}).get("stderr", "")),
                "success": generated_result.get("generated_code", {}).get("success"),
                "message": clean_text(generated_result.get("text", "")),
            }
        else:
            response_payload = {"error": f"未知工具：{tool_name}"}

        next_structured_summary = build_structured_summary(
            case_payload,
            next_tools_result,
            imaging_result,
            next_rag_result,
            task_mode=task_mode,
        )
        next_case_sections = self._build_case_sections(
            task_mode=task_mode,
            case_payload=case_payload,
            case_context=case_sections.get("full_context", case_sections.get("case_overview", "")),
            structured_summary=next_structured_summary,
            diagnosis_assessment=diagnosis_assessment,
            imaging_result=imaging_result,
        )
        next_case_chunks = self._build_case_chunks(next_case_sections)

        return {
            "tool_name": tool_name,
            "tool_args": arguments,
            "content": json.dumps(response_payload, ensure_ascii=False),
            "tools_result": next_tools_result,
            "rag_result": next_rag_result,
            "structured_summary": next_structured_summary,
            "case_sections": next_case_sections,
            "case_chunks": next_case_chunks,
            "source_updates": source_updates,
        }

    def _format_answer_v2(
        self,
        *,
        task_mode: str,
        diagnosis_assessment: Optional[Dict[str, Any]],
        treatment_result: Optional[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        if task_mode in {TASK_MODE_DIAGNOSIS, TASK_MODE_DIAGNOSIS_AND_TREATMENT} and diagnosis_assessment:
            lines.append(f"诊断结果：{clean_text(diagnosis_assessment.get('diagnosis_result', ''))}")
            lines.append(_render_labeled_section("诊断依据", _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))))

        if task_mode in {TASK_MODE_TREATMENT, TASK_MODE_DIAGNOSIS_AND_TREATMENT} and treatment_result:
            advice = _normalize_string_list(treatment_result.get("treatment_advice"))
            if not advice:
                parsed = _parse_treatment_stage_response(
                    clean_text(treatment_result.get("answer", "")) or clean_text(treatment_result.get("raw", ""))
                )
                advice = _normalize_string_list(parsed.get("treatment_advice"))
            if advice:
                lines.append(_render_labeled_section("治疗建议", advice))
            else:
                answer_text = clean_text(treatment_result.get("answer", "")) or clean_text(treatment_result.get("raw", ""))
                if answer_text:
                    lines.append(answer_text if answer_text.startswith("治疗建议") else f"治疗建议：{answer_text}")
        return "\n".join(line for line in lines if clean_text(line)).strip()

    def _build_medical_treatment_generation_prompt(
        self,
        *,
        user_query: str,
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        case_sections: Dict[str, str],
    ) -> str:
        summary_prompt = render_summary_prompt(structured_summary)
        diagnosis_lines: List[str] = []
        if diagnosis_assessment:
            diagnosis_result = clean_text(diagnosis_assessment.get("diagnosis_result", ""))
            if diagnosis_result:
                diagnosis_lines.append("\u8bca\u65ad\u80cc\u666f\uff1a" + diagnosis_result)
            diagnosis_basis = _normalize_string_list(diagnosis_assessment.get("diagnostic_basis"))
            if diagnosis_basis:
                diagnosis_lines.append(_render_labeled_section("\u8bca\u65ad\u4f9d\u636e", diagnosis_basis))
        diagnosis_text = "\n".join(line for line in diagnosis_lines if clean_text(line)).strip() or "\u672a\u63d0\u4f9b\u660e\u786e\u8bca\u65ad\u80cc\u666f\u3002"
        case_overview = clean_text(case_sections.get("case_overview", ""))
        return (
            "\u7528\u6237\u95ee\u9898\uff1a" + user_query + "\n\n"
            + "\u75c5\u4f8b\u6982\u89c8\uff1a\n"
            + _truncate_for_prompt(case_overview, 1800)
            + "\n\n"
            + diagnosis_text
            + "\n\n"
            + summary_prompt
            + "\n\n"
            + "\u8bf7\u57fa\u4e8e\u4ee5\u4e0a\u5df2\u7ecf\u6536\u96c6\u5230\u7684\u75c5\u4f8b\u4e8b\u5b9e\u3001SOFA\u5173\u952e\u6307\u6807\u3001\u5f71\u50cf\u548c\u6307\u5357\u8bc1\u636e\uff0c\u53ea\u8f93\u51fa\u201c\u6cbb\u7597\u5efa\u8bae\uff1a\u201d\u4e00\u8282\u3002\n"
            + "\u8981\u6c42\uff1a\n"
            + "1. \u4f7f\u7528\u7f16\u53f7\u5206\u70b9\uff0c\u4f18\u5148\u8986\u76d6\u6297\u611f\u67d3\u3001\u6db2\u4f53\u590d\u82cf\u3001\u5347\u538b\u836f/\u5faa\u73af\u652f\u6301\u3001\u5668\u5b98\u652f\u6301\u3001\u76d1\u6d4b\u4e0e\u590d\u8bc4\u3002\n"
            + "2. \u4e0d\u8981\u7f16\u9020\u68c0\u67e5\u3001\u7528\u836f\u6216\u6307\u5357\u7ed3\u8bba\uff1b\u82e5\u4fe1\u606f\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u6307\u51fa\u4ecd\u9700\u8865\u5145\u7684\u4fe1\u606f\u3002\n"
            + "3. \u4e0d\u8981\u590d\u8ff0\u5206\u6790\u8fc7\u7a0b\u6216\u5de5\u5177\u4f7f\u7528\u60c5\u51b5\u3002"
        )

    def _generate_medical_treatment_answer(
        self,
        *,
        user_query: str,
        structured_summary: StructuredSummary,
        diagnosis_assessment: Optional[Dict[str, Any]],
        case_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        medical_backend = self.config.medical_treatment_backend
        if medical_backend is None:
            return {
                "answer": "",
                "raw_text": "",
                "treatment_advice": [],
                "parse_mode": "skipped",
                "error": "medical_treatment_backend_not_configured",
            }

        raw_text = ""
        error = ""
        max_new_tokens = max(
            self.config.generation_max_new_tokens,
            int(getattr(medical_backend, "default_max_new_tokens", 0) or 0),
        )
        timeout_seconds = max(
            self.config.http_timeout_seconds,
            float(getattr(medical_backend, "default_timeout_seconds", 0.0) or 0.0),
        )
        try:
            raw_text = self._generate_text(
                backend_config=medical_backend,
                system_prompt=MEDICAL_TREATMENT_GENERATION_SYSTEM_PROMPT,
                user_prompt=self._build_medical_treatment_generation_prompt(
                    user_query=user_query,
                    structured_summary=structured_summary,
                    diagnosis_assessment=diagnosis_assessment,
                    case_sections=case_sections,
                ),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=self.config.generation_repetition_penalty,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            error = str(exc)

        parsed = _parse_treatment_stage_response(raw_text)
        treatment_advice = _normalize_string_list(parsed.get("treatment_advice"))
        answer = ""
        if treatment_advice:
            answer = _render_labeled_section("\u6cbb\u7597\u5efa\u8bae", treatment_advice)
        else:
            normalized_raw = clean_text(raw_text)
            if normalized_raw:
                answer = normalized_raw if normalized_raw.startswith("\u6cbb\u7597\u5efa\u8bae") else "\u6cbb\u7597\u5efa\u8bae\uff1a" + normalized_raw

        if not error and not answer:
            error = "empty_treatment_answer"

        return {
            "answer": answer,
            "raw_text": clean_text(raw_text),
            "treatment_advice": treatment_advice,
            "parse_mode": clean_text(parsed.get("parse_mode", "")),
            "error": clean_text(error),
        }

    def _should_use_baichuan_react_handoff(self) -> bool:
        return identify_text_backend_preset_name(self.config.medical_treatment_backend) == "baichuan"

    def _generate_medical_treatment_answer_from_react_context(
        self,
        *,
        round_messages: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        medical_backend = self.config.medical_treatment_backend
        if medical_backend is None:
            return {
                "answer": "",
                "raw_text": "",
                "treatment_advice": [],
                "parse_mode": "skipped",
                "error": "medical_treatment_backend_not_configured",
            }

        handoff_messages = copy.deepcopy(list(round_messages))
        handoff_messages.append(
            {
                "role": "user",
                "content": (
                    "现在不要调用工具，也不要复述上面的任务要求、分析过程或病例摘要。"
                    "请直接基于以上已经给出的相同信息，输出最终“治疗建议：”一节。"
                ),
            }
        )

        raw_text = ""
        error = ""
        max_new_tokens = max(
            self.config.generation_max_new_tokens,
            int(getattr(medical_backend, "default_max_new_tokens", 0) or 0),
        )
        timeout_seconds = max(
            self.config.http_timeout_seconds,
            float(getattr(medical_backend, "default_timeout_seconds", 0.0) or 0.0),
        )
        try:
            response = self._complete_chat_with_tools(
                backend_config=medical_backend,
                messages=handoff_messages,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=self.config.generation_repetition_penalty,
                request_timeout_seconds=timeout_seconds,
                tools=None,
                tool_choice=None,
            )
            raw_text = clean_text(response.get("text", ""))
            if response.get("tool_calls"):
                error = "unexpected_tool_calls_in_medical_handoff"
        except Exception as exc:
            error = str(exc)

        parsed = _parse_treatment_stage_response(raw_text)
        treatment_advice = _normalize_string_list(parsed.get("treatment_advice"))
        answer = ""
        if treatment_advice:
            answer = _render_labeled_section("治疗建议", treatment_advice)
        else:
            normalized_raw = clean_text(raw_text)
            if normalized_raw:
                answer = normalized_raw if normalized_raw.startswith("治疗建议") else "治疗建议：" + normalized_raw

        if not error and not answer:
            error = "empty_treatment_answer"

        return {
            "answer": answer,
            "raw_text": clean_text(raw_text),
            "treatment_advice": treatment_advice,
            "parse_mode": clean_text(parsed.get("parse_mode", "")),
            "error": clean_text(error),
        }

    def _run_treatment_react(
        self,
        *,
        user_query: str,
        case_payload: Dict[str, Any],
        case_source: str,
        case_json_path: Optional[str],
        task_mode: str,
        diagnosis_assessment: Optional[Dict[str, Any]],
        base_tools_result: Optional[Dict[str, Any]],
        enable_search: Optional[bool],
        search_options: Optional[Dict[str, Any]],
        imaging_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.config.enable_treatment_react:
            return self._build_empty_treatment_result(status="disabled", error="treatment_react_disabled")

        treatment_backend = self.config.resolve_text_backend("treatment")
        tools_result = self._merge_tool_results(self._build_empty_treatment_result()["tools_result"], base_tools_result or {})
        rag_result = {"text": "", "sources": [], "items": []}
        structured_summary = build_structured_summary(
            case_payload,
            tools_result,
            imaging_result,
            rag_result,
            task_mode=task_mode,
        )
        case_context = case_to_context(
            case_payload,
            max_field_chars=1400,
            lab_selection="sepsis_priority",
            task_mode=task_mode,
        )
        case_sections = self._build_case_sections(
            task_mode=task_mode,
            case_payload=case_payload,
            case_context=case_context,
            structured_summary=structured_summary,
            diagnosis_assessment=diagnosis_assessment,
            imaging_result=imaging_result,
        )
        case_chunks = self._build_case_chunks(case_sections)
        tools = self._build_treatment_tool_specs(case_sections)
        extra_body_override = self._build_treatment_extra_body_override(
            enable_search=enable_search,
            search_options=search_options,
        )

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": TREATMENT_REACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self._build_treatment_react_user_prompt(
                    user_query=user_query,
                    case_sections=case_sections,
                    structured_summary=structured_summary,
                    max_rounds=self.config.max_react_rounds,
                ),
            },
        ]

        trace: List[Dict[str, Any]] = []
        flattened_tool_calls: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = [
            {
                "tag": "CASE1",
                "source": case_source,
                "snippet": _truncate_for_prompt(case_context, 280),
            }
        ]
        treatment_scorecard_memory = self._build_initial_treatment_scorecard_memory(
            case_sections=case_sections,
            structured_summary=structured_summary,
            diagnosis_assessment=diagnosis_assessment,
            sources=sources,
        )
        collection_summary = ""
        final_text = ""
        final_error = ""
        handoff_medical_generation: Optional[Dict[str, Any]] = None

        for round_index in range(1, self.config.max_react_rounds + 1):
            round_messages = messages
            if self.config.enable_treatment_round_reminders:
                round_messages = messages + [
                    {
                        "role": "user",
                        "content": _build_treatment_round_reminder(round_index, self.config.max_react_rounds),
                    }
                ]
            try:
                response = self._complete_chat_with_tools(
                    backend_config=treatment_backend,
                    messages=round_messages,
                    max_new_tokens=self.config.generation_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=self.config.generation_repetition_penalty,
                    request_timeout_seconds=self.config.http_timeout_seconds,
                    tools=tools,
                    tool_choice="auto",
                    extra_body_override=extra_body_override,
                )
            except Exception as exc:
                final_error = str(exc)
                break

            assistant_text = clean_text(response.get("text", ""))
            tool_calls = list(response.get("tool_calls", []))
            trace_entry: Dict[str, Any] = {
                "round": round_index,
                "assistant_text": assistant_text,
                "tool_calls": tool_calls,
                "tool_outputs": [],
            }

            assistant_message: Dict[str, Any] = {"role": "assistant", "content": assistant_text}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": clean_text(tool_call.get("id", "")),
                        "type": clean_text(tool_call.get("type", "")) or "function",
                        "function": {
                            "name": clean_text(tool_call.get("name", "")),
                            "arguments": clean_text(tool_call.get("arguments", "")),
                        },
                    }
                    for tool_call in tool_calls
                ]
            messages.append(assistant_message)

            if not tool_calls:
                collection_summary = assistant_text or "证据已足够，可进入最终定稿。"
                final_text = collection_summary
                if self._should_use_baichuan_react_handoff():
                    handoff_medical_generation = self._generate_medical_treatment_answer_from_react_context(
                        round_messages=round_messages,
                    )
                    final_text = clean_text(handoff_medical_generation.get("raw_text", "")) or final_text
                trace.append(trace_entry)
                break

            flattened_tool_calls.extend(tool_calls)
            for tool_call in tool_calls:
                tool_result = self._execute_treatment_tool_call(
                    tool_call=tool_call,
                    user_query=user_query,
                    case_payload=case_payload,
                    case_json_path=case_json_path,
                    case_sections=case_sections,
                    case_chunks=case_chunks,
                    task_mode=task_mode,
                    diagnosis_assessment=diagnosis_assessment,
                    structured_summary=structured_summary,
                    tools_result=tools_result,
                    rag_result=rag_result,
                    imaging_result=imaging_result,
                )
                tools_result = tool_result["tools_result"]
                rag_result = tool_result["rag_result"]
                structured_summary = tool_result["structured_summary"]
                case_sections = tool_result["case_sections"]
                case_chunks = tool_result["case_chunks"]
                sources = self._deduplicate_sources(sources + tool_result.get("source_updates", []))
                treatment_scorecard_memory = self._update_treatment_scorecard_memory(
                    treatment_scorecard_memory,
                    case_sections=case_sections,
                    structured_summary=structured_summary,
                    diagnosis_assessment=diagnosis_assessment,
                    sources=sources,
                    tools_result=tools_result,
                    rag_result=rag_result,
                    tool_name=tool_result["tool_name"],
                    tool_content=tool_result["content"],
                )
                tool_call_id = clean_text(tool_call.get("id", "")) or f"toolcall_{round_index}_{len(trace_entry['tool_outputs']) + 1}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result["content"],
                    }
                )
                trace_entry["tool_outputs"].append(
                    {
                        "tool_name": tool_result["tool_name"],
                        "tool_args": tool_result["tool_args"],
                        "content_preview": clean_text(tool_result["content"])[:500],
                    }
                )
            trace.append(trace_entry)
            if self.config.medical_treatment_backend is None and self._treatment_scorecard_ready(
                treatment_scorecard_memory,
                flattened_tool_calls,
            ):
                collection_summary = self._render_treatment_scorecard_memory(treatment_scorecard_memory)
                final_text = collection_summary
                break
        else:
            final_error = final_error or "treatment_react_round_limit_reached"

        if not final_text:
            collection_summary = collection_summary or self._render_treatment_scorecard_memory(treatment_scorecard_memory)
            final_text = collection_summary

        parsed = _parse_treatment_stage_response(final_text)
        treatment_advice = _normalize_string_list(parsed.get("treatment_advice"))
        answer = _render_treatment_answer(final_text, treatment_advice)
        react_error = final_error
        scorecard_evidence_text = self._treatment_scorecard_evidence_text(
            treatment_scorecard_memory,
            case_sections,
            sources,
            structured_summary=structured_summary,
        )
        quality_flags = _treatment_quality_flags(final_text, treatment_advice, evidence_text=scorecard_evidence_text)
        finalization_metadata: Dict[str, Any] = {
            "applied": False,
            "reason": "",
            "error": "",
            "raw_text": "",
        }
        if self.config.medical_treatment_backend is None:
            if final_error == "treatment_react_round_limit_reached" and "round_limit" not in quality_flags:
                quality_flags.append("round_limit")
            finalized_result = self._finalize_treatment_answer(
                treatment_backend=treatment_backend,
                user_query=user_query,
                structured_summary=structured_summary,
                diagnosis_assessment=diagnosis_assessment,
                case_sections=case_sections,
                sources=sources,
                candidate_answer=collection_summary or final_text,
                quality_flags=quality_flags,
                treatment_scorecard_memory=treatment_scorecard_memory,
                collection_summary=collection_summary,
            )
            finalized_error = clean_text(finalized_result.get("error", ""))
            finalized_answer = clean_text(finalized_result.get("answer", ""))
            finalized_advice = _normalize_string_list(finalized_result.get("treatment_advice"))
            finalized_flags = _normalize_string_list(finalized_result.get("quality_flags"))
            hard_finalization_flags = {
                "empty_or_unparsed",
                "tool_artifact",
                "tool_plan",
                "section_reference",
                "meta_summary_as_answer",
            }
            accept_finalized = bool(finalized_answer) and not finalized_error and (
                not set(finalized_flags) & hard_finalization_flags
            )
            finalization_metadata = {
                "applied": accept_finalized,
                "reason": ",".join(_normalize_string_list(quality_flags)),
                "error": finalized_error or (",".join(finalized_flags) if not accept_finalized else ""),
                "raw_text": clean_text(finalized_result.get("raw_text", "")),
            }
            if accept_finalized:
                final_text = clean_text(finalized_result.get("raw_text", "")) or final_text
                treatment_advice = finalized_advice
                answer = finalized_answer
                quality_flags = finalized_flags
                if final_error == "treatment_react_round_limit_reached":
                    final_error = ""
            else:
                final_text = clean_text(finalized_result.get("raw_text", "")) or final_text
                answer = ""
                treatment_advice = []
                quality_flags = finalized_flags or quality_flags
                final_error = finalized_error or (
                    "treatment_finalization_quality_failed:" + ",".join(finalized_flags)
                    if finalized_flags
                    else "treatment_finalization_failed"
                )
        react_status = "success" if answer and not final_error else ("error" if final_error else "incomplete")
        status = react_status
        result_error = final_error
        generation_mode = "fullstack_base"
        medical_generation = {
            "answer": "",
            "raw_text": "",
            "treatment_advice": [],
            "parse_mode": "",
            "error": "",
        }
        if self.config.medical_treatment_backend is not None:
            generation_mode = "medical_backend"
            medical_generation = handoff_medical_generation or self._generate_medical_treatment_answer(
                user_query=user_query,
                structured_summary=structured_summary,
                diagnosis_assessment=diagnosis_assessment,
                case_sections=case_sections,
            )
            medical_error = clean_text(medical_generation.get("error", ""))
            if medical_error:
                status = "error"
                result_error = f"medical_treatment_generation_failed: {medical_error}"
                answer = ""
                treatment_advice = []
                final_text = clean_text(medical_generation.get("raw_text", "")) or final_text
            else:
                final_text = clean_text(medical_generation.get("raw_text", "")) or final_text
                treatment_advice = _normalize_string_list(medical_generation.get("treatment_advice"))
                answer = clean_text(medical_generation.get("answer", ""))
                status = "success" if answer else "incomplete"
                result_error = ""
        result = self._build_empty_treatment_result(
            status=status,
            error=result_error,
            raw_text=final_text,
            treatment_advice=treatment_advice,
        )
        result.update(
            {
                "triggered": True,
                "answer": answer,
                "react_status": react_status,
                "react_error": react_error,
                "treatment_generation_mode": generation_mode,
                "medical_generation_error": clean_text(medical_generation.get("error", "")),
                "medical_generation_raw": clean_text(medical_generation.get("raw_text", "")),
                "finalization_applied": bool(finalization_metadata.get("applied")),
                "finalization_reason": clean_text(finalization_metadata.get("reason", "")),
                "finalization_error": clean_text(finalization_metadata.get("error", "")),
                "finalization_raw": clean_text(finalization_metadata.get("raw_text", "")),
                "treatment_quality_flags": _normalize_string_list(quality_flags),
                "treatment_scorecard_memory": treatment_scorecard_memory,
                "collection_summary": clean_text(collection_summary),
                "trace": trace,
                "tool_calls": flattened_tool_calls,
                "sources": self._deduplicate_sources(sources),
                "tools_result": tools_result,
                "rag_result": rag_result,
                "structured_summary": structured_summary.to_dict(),
                "plan": {
                    "task_category": "treatment_react",
                    "intent": "治疗建议",
                    "use_rag": bool(self.config.enable_local_rag),
                    "use_data_tools": True,
                    "use_generated_code": bool(self.config.enable_generated_code),
                    "use_builtin_search": bool(self.config.builtin_search_default if enable_search is None else enable_search),
                    "treatment_generation_mode": generation_mode,
                    "medical_treatment_backend": (
                        self.config.medical_treatment_backend.describe()
                        if self.config.medical_treatment_backend is not None
                        else ""
                    ),
                },
            }
        )
        return result
