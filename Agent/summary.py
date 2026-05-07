from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Sequence

from .utils import (
    clean_text,
    extract_diagnosis_candidates,
    first_non_empty,
    get_case_auxiliary_exam_text,
    normalize_case_payload,
    normalize_text,
    truncate_text,
)


SOURCE_PRIORITY = {"case": 4, "tool": 3, "imaging": 2, "rag": 1}
CATEGORY_SALIENCE = {
    "patient_state": 3,
    "supporting_evidence": 4,
    "opposing_evidence": 4,
    "suspected_source": 5,
    "organ_dysfunction": 5,
    "hemodynamics": 5,
    "microbiology": 5,
    "treatment_constraint": 4,
    "knowledge_support": 2,
}

INFECTION_TOKENS = (
    "感染",
    "发热",
    "寒战",
    "脓毒症",
    "菌血症",
    "抗感染",
    "CRP",
    "PCT",
    "降钙素原",
    "白细胞",
    "中性粒",
)
ORGAN_DYSFUNCTION_TOKENS = (
    "SOFA",
    "器官功能障碍",
    "器官功能异常",
    "肌酐",
    "胆红素",
    "少尿",
    "肾功能",
    "肝功能",
    "呼吸衰竭",
    "意识障碍",
    "血小板",
    "D-二聚体",
    "凝血",
    "乳酸",
)
HEMODYNAMIC_TOKENS = (
    "血压",
    "BP",
    "MAP",
    "低血压",
    "休克",
    "去甲肾上腺素",
    "肾上腺素",
    "升压药",
    "晶体液",
    "复苏",
)
MICROBIOLOGY_TOKENS = (
    "血培养",
    "尿培养",
    "痰培养",
    "培养",
    "细菌",
    "真菌",
    "大肠埃希菌",
    "革兰",
    "阳性",
    "阴性",
    "病原",
)
TREATMENT_CONSTRAINT_TOKENS = (
    "过敏",
    "休克",
    "低血压",
    "少尿",
    "肾功能",
    "肝功能",
    "凝血",
    "血小板",
    "胆道感染",
    "引流",
    "ERCP",
)
OPPOSING_TOKENS = ("排除", "不支持", "不足以支持", "未见明确", "未发现", "阴性", "无明显")
EXPLICIT_SOURCE_KEYWORDS = {
    "胆道感染": ("胆道感染", "胆管炎", "胆囊炎", "胆总管炎", "胆道源性"),
    "肺部感染": ("肺部感染", "肺炎", "下呼吸道感染"),
    "尿路感染": ("尿路感染", "泌尿系感染", "肾盂肾炎"),
    "腹腔感染": ("腹腔感染", "腹膜炎", "腹腔脓肿"),
}
DIRECT_INFECTION_EVIDENCE_TOKENS = (
    "感染",
    "发热",
    "寒战",
    "肺炎",
    "胆管炎",
    "胆囊炎",
    "尿路感染",
    "腹膜炎",
    "脓肿",
    "菌血症",
    "PCT",
    "CRP",
    "降钙素原",
    "白细胞",
    "中性粒",
    "培养",
)
INFECTION_SYMPTOM_TOKENS = (
    "发热",
    "寒战",
    "寒颤",
    "咳痰",
    "黄痰",
    "脓痰",
    "腹痛",
    "右上腹痛",
    "尿痛",
    "尿频",
    "尿急",
)
INFECTION_LAB_TOKENS = (
    "PCT",
    "CRP",
    "降钙素原",
    "白细胞",
    "中性粒",
    "血清淀粉样蛋白A",
    "SAA",
)
TOOL_NOISE_TOKENS = (
    "top_level_keys",
    "case_path",
    "candidate_dir",
    "_exists",
    "_dcm_count",
    "script_path",
    "imaging_entry_count",
)
TOOL_CLINICAL_TOKENS = DIRECT_INFECTION_EVIDENCE_TOKENS + (
    "血压",
    "MAP",
    "乳酸",
    "肌酐",
    "血小板",
    "总胆红素",
    "GCS",
    "PaO2",
    "FiO2",
)


@dataclass
class EvidenceItem:
    category: str
    source_type: str
    source_tag: str
    source_ref: str
    claim_text: str
    polarity: str
    salience: int
    timestamp: str = ""
    raw_value: str = ""

    def dedupe_key(self) -> tuple[str, str, str]:
        normalized_claim = re.sub(r"\s+", "", normalize_text(self.claim_text)).lower()
        return self.category, self.polarity, normalized_claim

    def rank(self) -> tuple[int, int, int, str]:
        return (
            SOURCE_PRIORITY.get(self.source_type, 0),
            int(self.salience),
            1 if clean_text(self.timestamp) else 0,
            clean_text(self.timestamp),
        )

    def to_display_text(self) -> str:
        claim = clean_text(self.claim_text)
        if not claim:
            return ""
        if self.source_tag and f"[{self.source_tag}]" not in claim:
            claim = f"{claim} [{self.source_tag}]"
        return claim

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuredSummary:
    patient_state: List[str]
    supporting_evidence: List[str]
    opposing_evidence: List[str]
    suspected_sources: List[str]
    organ_dysfunction: List[str]
    treatment_constraints: List[str]
    knowledge_support: List[str]
    conflicts: List[str]
    missing_information: List[str]
    prompt_text: str
    evidence_items: List[EvidenceItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["evidence_items"] = [item.to_dict() for item in self.evidence_items]
        return payload


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _sanitize_for_category_match(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    sanitized = normalized.replace("抗血小板", "")
    sanitized = sanitized.replace("控制血糖", "")
    sanitized = re.sub(r"高血压病[0-9一二三四五六七八九十]*级?", "", sanitized)
    return sanitized


def _split_claim_text(text: Any, *, max_parts: int = 8) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts: List[str] = []
    for line in normalized.splitlines():
        line_text = clean_text(line)
        if not line_text:
            continue
        segments = re.split(r"[；;]", line_text)
        for segment in segments:
            candidate = clean_text(segment)
            if not candidate:
                continue
            if re.fullmatch(r".*[:：]\s*", candidate):
                continue
            parts.append(truncate_text(candidate, 220))
            if len(parts) >= max_parts:
                return parts
    return parts


def _detect_source_labels(text: str) -> List[str]:
    labels: List[str] = []
    for label, keywords in EXPLICIT_SOURCE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            labels.append(label)
    return labels


def _contains_direct_infection_evidence(text: str) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return _contains_any(normalized, DIRECT_INFECTION_EVIDENCE_TOKENS)


def _summarize_infection_evidence(items: Sequence[EvidenceItem]) -> Dict[str, int]:
    summary = {"explicit_source": 0, "microbiology": 0, "symptom": 0, "lab": 0}
    for item in items:
        if item.source_type not in {"case", "imaging"} or item.polarity != "supporting":
            continue
        text = normalize_text(item.claim_text)
        if item.category == "suspected_source":
            summary["explicit_source"] += 1
        elif item.category == "microbiology":
            summary["microbiology"] += 1
        elif item.category == "supporting_evidence":
            if _contains_any(text, INFECTION_SYMPTOM_TOKENS):
                summary["symptom"] += 1
            if _contains_any(text, INFECTION_LAB_TOKENS):
                summary["lab"] += 1
    return summary


def _infer_polarity(text: str, default: str = "supporting") -> str:
    normalized = normalize_text(text)
    if not normalized:
        return default
    if _contains_any(normalized, OPPOSING_TOKENS):
        return "opposing"
    return default


def _infer_category(text: str, *, default_category: str) -> str:
    normalized = _sanitize_for_category_match(text)
    if not normalized:
        return default_category
    if _detect_source_labels(normalized):
        return "suspected_source"
    if _contains_any(normalized, MICROBIOLOGY_TOKENS):
        return "microbiology"
    if _contains_any(normalized, HEMODYNAMIC_TOKENS):
        return "hemodynamics"
    if _contains_any(normalized, ORGAN_DYSFUNCTION_TOKENS):
        return "organ_dysfunction"
    if _contains_any(normalized, TREATMENT_CONSTRAINT_TOKENS):
        return "treatment_constraint"
    if _contains_any(normalized, INFECTION_TOKENS):
        return "supporting_evidence"
    return default_category


def _make_item(
    *,
    category: str,
    source_type: str,
    source_tag: str,
    source_ref: str,
    claim_text: str,
    polarity: str,
    salience: Optional[int] = None,
    timestamp: str = "",
    raw_value: str = "",
) -> EvidenceItem:
    resolved_salience = salience if salience is not None else CATEGORY_SALIENCE.get(category, 2)
    return EvidenceItem(
        category=category,
        source_type=source_type,
        source_tag=source_tag,
        source_ref=source_ref,
        claim_text=truncate_text(claim_text, 240),
        polarity=polarity,
        salience=resolved_salience,
        timestamp=clean_text(timestamp),
        raw_value=clean_text(raw_value),
    )


def _should_include_case_diagnosis_evidence(task_mode: Optional[str]) -> bool:
    return clean_text(task_mode).lower() == "treatment"


def _extract_case_items(case_payload: Dict[str, Any], task_mode: Optional[str] = None) -> List[EvidenceItem]:
    payload = normalize_case_payload(case_payload)
    items: List[EvidenceItem] = []
    timestamp = clean_text(first_non_empty(payload, "记录时间", "检查时间"))

    basic_parts = [part for part in [clean_text(payload.get("性别", "")), clean_text(first_non_empty(payload, "住院年龄", "年龄"))] if part]
    chief = normalize_text(payload.get("主诉", ""))
    if basic_parts or chief:
        claim = "；".join(part for part in [f"基本信息：{'/'.join(basic_parts)}" if basic_parts else "", f"主诉：{chief}" if chief else ""] if part)
        items.append(
            _make_item(
                category="patient_state",
                source_type="case",
                source_tag="CASE1",
                source_ref="case_payload",
                claim_text=claim,
                polarity="neutral",
                salience=4,
                timestamp=timestamp,
            )
        )

    for field in ("现病史", "体格检查", "专科检查"):
        text = normalize_text(payload.get(field, ""))
        if not text:
            continue
        items.append(
            _make_item(
                category="patient_state",
                source_type="case",
                source_tag="CASE1",
                source_ref="case_payload",
                claim_text=f"{field}：{truncate_text(text, 220)}",
                polarity="neutral",
                salience=3,
                timestamp=timestamp,
            )
        )
        for claim in _split_claim_text(text, max_parts=6):
            category = _infer_category(claim, default_category="patient_state")
            if category == "patient_state":
                continue
            items.append(
                _make_item(
                    category=category,
                    source_type="case",
                    source_tag="CASE1",
                    source_ref="case_payload",
                    claim_text=claim,
                    polarity=_infer_polarity(claim),
                    timestamp=timestamp,
                )
            )

    allergy_text = normalize_text(payload.get("过敏史", ""))
    if allergy_text:
        items.append(
            _make_item(
                category="treatment_constraint",
                source_type="case",
                source_tag="CASE1",
                source_ref="case_payload",
                claim_text=f"过敏史：{truncate_text(allergy_text, 180)}",
                polarity="supporting",
                salience=4,
                timestamp=timestamp,
            )
        )

    auxiliary_text = get_case_auxiliary_exam_text(payload, max_lab_items=40)
    for claim in _split_claim_text(auxiliary_text, max_parts=24):
        category = _infer_category(claim, default_category="supporting_evidence")
        polarity = _infer_polarity(claim)
        items.append(
            _make_item(
                category=category,
                source_type="case",
                source_tag="CASE1",
                source_ref="case_payload",
                claim_text=claim,
                polarity=polarity,
                timestamp=timestamp,
            )
        )

    if _should_include_case_diagnosis_evidence(task_mode):
        for candidate in extract_diagnosis_candidates(payload)[:4]:
            if not _contains_direct_infection_evidence(candidate):
                continue
            items.append(
                _make_item(
                    category="supporting_evidence",
                    source_type="case",
                    source_tag="CASE1",
                    source_ref="case_payload",
                    claim_text=f"病历诊断线索：{candidate}",
                    polarity=_infer_polarity(candidate),
                    salience=4,
                    timestamp=timestamp,
                )
            )
    return items


def _extract_tool_items(tools_result: Dict[str, Any]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    sofa = tools_result.get("sofa", {}) if isinstance(tools_result.get("sofa", {}), dict) else {}
    reported_total = sofa.get("reported_total")
    computed_total = sofa.get("computed_total")
    if reported_total is not None:
        items.append(
            _make_item(
                category="organ_dysfunction",
                source_type="tool",
                source_tag="TOOL_SOFA",
                source_ref="sofa.reported_total",
                claim_text=f"病历显式记录 SOFA 评分：{reported_total}",
                polarity="supporting" if float(reported_total) >= 2 else "opposing",
                salience=5,
                raw_value=str(reported_total),
            )
        )
    if computed_total is not None:
        items.append(
            _make_item(
                category="organ_dysfunction",
                source_type="tool",
                source_tag="TOOL_SOFA",
                source_ref="sofa.computed_total",
                claim_text=f"工具可计算 SOFA 评分：{computed_total}",
                polarity="supporting" if float(computed_total) >= 2 else "opposing",
                salience=5,
                raw_value=str(computed_total),
            )
        )

    component_scores = sofa.get("component_scores", {}) if isinstance(sofa.get("component_scores", {}), dict) else {}
    for component, payload in component_scores.items():
        if not isinstance(payload, dict):
            continue
        score = payload.get("score")
        if score is None:
            continue
        description = clean_text(payload.get("description", component))
        value = clean_text(payload.get("value", ""))
        items.append(
            _make_item(
                category="hemodynamics" if component == "cardiovascular" else "organ_dysfunction",
                source_type="tool",
                source_tag="TOOL_SOFA",
                source_ref=f"sofa.component.{component}",
                claim_text=f"{description}评分：{score}" + (f"，原始值：{value}" if value else ""),
                polarity="supporting" if int(score) >= 1 else "opposing",
                salience=4,
                raw_value=value or str(score),
            )
        )

    for key, tag in (("static_code", "TOOL1"), ("generated_code", "TOOL2")):
        payload = tools_result.get(key, {})
        if not isinstance(payload, dict):
            continue
        source_ref = clean_text(payload.get("script_path", "")) or key
        combined_text = "\n".join(
            part for part in [clean_text(payload.get("stdout", "")), clean_text(payload.get("stderr", ""))] if part
        )
        for claim in _split_claim_text(combined_text, max_parts=10):
            if _contains_any(claim, TOOL_NOISE_TOKENS):
                continue
            if not _contains_any(claim, TOOL_CLINICAL_TOKENS):
                continue
            category = _infer_category(claim, default_category="supporting_evidence")
            items.append(
                _make_item(
                    category=category,
                    source_type="tool",
                    source_tag=tag,
                    source_ref=source_ref,
                    claim_text=claim,
                    polarity=_infer_polarity(claim),
                )
            )
    return items


def _extract_imaging_items(imaging_result: Dict[str, Any]) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    studies = imaging_result.get("studies", []) if isinstance(imaging_result.get("studies", []), list) else []
    sources = imaging_result.get("sources", []) if isinstance(imaging_result.get("sources", []), list) else []
    for index, study in enumerate(studies, start=1):
        if not isinstance(study, dict):
            continue
        source_tag = f"IMG{index}"
        if index - 1 < len(sources) and isinstance(sources[index - 1], dict):
            source_tag = clean_text(sources[index - 1].get("tag", "")) or source_tag
            source_ref = clean_text(sources[index - 1].get("source", "")) or clean_text(study.get("study_id", ""))
        else:
            source_ref = clean_text(study.get("study_id", ""))
        for field_name in ("findings", "conclusion"):
            claim_text = normalize_text(study.get(field_name, ""))
            if not claim_text:
                continue
            for claim in _split_claim_text(claim_text, max_parts=6):
                category = _infer_category(claim, default_category="supporting_evidence")
                items.append(
                    _make_item(
                        category=category,
                        source_type="imaging",
                        source_tag=source_tag,
                        source_ref=source_ref,
                        claim_text=claim,
                        polarity=_infer_polarity(claim),
                        salience=4,
                    )
                )
    if not items:
        for claim in _split_claim_text(imaging_result.get("text", ""), max_parts=6):
            items.append(
                _make_item(
                    category=_infer_category(claim, default_category="supporting_evidence"),
                    source_type="imaging",
                    source_tag="IMG1",
                    source_ref="imaging.text",
                    claim_text=claim,
                    polarity=_infer_polarity(claim),
                    salience=3,
                )
            )
    return items


def _extract_knowledge_items(result: Dict[str, Any], *, source_type: str, tag_prefix: str) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    for index, item in enumerate(result.get("items", []) if isinstance(result.get("items", []), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        if source_type == "rag":
            claim = normalize_text(item.get("text", ""))
            source_ref = clean_text(item.get("path", ""))
        else:
            claim = normalize_text(item.get("snippet", ""))
            source_ref = clean_text(item.get("url", ""))
        if not claim:
            continue
        title = clean_text(item.get("title", ""))
        prefix = f"{title}：" if title else ""
        compact_claim = re.sub(r"\s+", " ", claim).strip()
        items.append(
            _make_item(
                category="knowledge_support",
                source_type=source_type,
                source_tag=f"{tag_prefix}{index}",
                source_ref=source_ref,
                claim_text=prefix + truncate_text(compact_claim, 96),
                polarity="supporting",
                salience=2,
            )
        )
    return items


def _dedupe_items(items: List[EvidenceItem]) -> List[EvidenceItem]:
    best_items: Dict[tuple[str, str, str], EvidenceItem] = {}
    for item in items:
        key = item.dedupe_key()
        current = best_items.get(key)
        if current is None or item.rank() > current.rank():
            best_items[key] = item
    deduped = list(best_items.values())
    deduped.sort(key=lambda item: item.rank(), reverse=True)
    return deduped


def _take_display_lines(items: List[EvidenceItem], *, max_items: int) -> List[str]:
    lines: List[str] = []
    seen = set()
    for item in items:
        line = item.to_display_text()
        if not line or line in seen:
            continue
        seen.add(line)
        lines.append(line)
        if len(lines) >= max_items:
            break
    return lines


def _build_treatment_constraints(
    case_payload: Dict[str, Any],
    items: List[EvidenceItem],
    suspected_sources: List[str],
    organ_lines: List[str],
) -> List[str]:
    constraints = _take_display_lines([item for item in items if item.category == "treatment_constraint"], max_items=4)
    if suspected_sources and any("胆道感染" in item for item in suspected_sources):
        constraints.append("疑似胆道来源，需尽快评估源控制/引流可行性 [CASE1]")
    if organ_lines and any(_contains_any(line, ("休克", "MAP", "低血压", "去甲肾上腺素")) for line in organ_lines):
        constraints.append("存在血流动力学不稳定线索，治疗需兼顾液体复苏和升压支持 [TOOL_SOFA]")
    allergy = normalize_text(normalize_case_payload(case_payload).get("过敏史", ""))
    if allergy and not any("过敏史" in item for item in constraints):
        constraints.append(f"过敏史需要纳入用药选择：{truncate_text(allergy, 120)} [CASE1]")
    deduped: List[str] = []
    for item in constraints:
        if item not in deduped:
            deduped.append(item)
    return deduped[:5]


def _build_conflicts(
    supporting_lines: List[str],
    opposing_lines: List[str],
    suspected_sources: List[str],
    organ_lines: List[str],
) -> List[str]:
    conflicts: List[str] = []
    if supporting_lines and opposing_lines:
        conflicts.append("存在同时支持与削弱诊断的线索，需要结合时间轴和复查结果综合判断。")
    if supporting_lines and not organ_lines:
        conflicts.append("存在感染相关线索，但器官功能障碍证据不足。")
    if len(suspected_sources) >= 2:
        conflicts.append("感染来源线索不集中：" + "；".join(suspected_sources[:3]))
    return conflicts[:4]


def _build_missing_information(
    items: List[EvidenceItem],
    suspected_sources: List[str],
    organ_lines: List[str],
    treatment_constraints: List[str],
    imaging_result: Dict[str, Any],
) -> List[str]:
    categories_present = {item.category for item in items}
    missing: List[str] = []
    infection_summary = _summarize_infection_evidence(items)
    if infection_summary["explicit_source"] + infection_summary["microbiology"] + infection_summary["symptom"] <= 0:
        missing.append("感染直接证据不足。")
    if not suspected_sources:
        missing.append("感染来源尚不明确。")
    if not organ_lines:
        missing.append("器官功能障碍证据不足或缺少系统评估。")
    if "hemodynamics" not in categories_present:
        missing.append("血流动力学信息不足。")
    if "microbiology" not in categories_present:
        missing.append("缺少微生物学/培养证据。")
    imaging_studies = imaging_result.get("studies", []) if isinstance(imaging_result.get("studies", []), list) else []
    has_imaging_evidence = bool(clean_text(imaging_result.get("text", ""))) or any(
        clean_text(study.get("findings", "")) or clean_text(study.get("conclusion", ""))
        for study in imaging_studies
        if isinstance(study, dict)
    )
    if not has_imaging_evidence:
        missing.append("缺少影像来源灶证据。")
    if not treatment_constraints:
        missing.append("治疗约束信息有限。")
    return missing[:6]


def render_summary_prompt(summary: StructuredSummary) -> str:
    sections = [
        ("患者状态", summary.patient_state or ["暂无明确高价值患者状态摘要。"]),
        ("高优先级支持证据", summary.supporting_evidence or ["暂无明确支持证据。"]),
        ("反对或不足证据", summary.opposing_evidence or ["暂无明确反对证据。"]),
        ("疑似感染来源", summary.suspected_sources or ["感染来源暂不明确。"]),
        ("器官功能障碍线索", summary.organ_dysfunction or ["暂无明确器官功能障碍线索。"]),
        ("治疗约束", summary.treatment_constraints or ["暂无明确治疗约束线索。"]),
        ("知识支持", summary.knowledge_support or ["暂无额外知识支持。"]),
        ("证据冲突", summary.conflicts or ["暂无显著证据冲突。"]),
        ("信息缺口", summary.missing_information or ["暂无显著信息缺口。"]),
    ]
    lines = ["综合工作摘要："]
    for title, values in sections:
        lines.append(f"{title}：")
        lines.extend(f"- {clean_text(value)}" for value in values if clean_text(value))
    return "\n".join(lines).strip()


def build_structured_summary(
    case_payload: Dict[str, Any],
    tools_result: Dict[str, Any],
    imaging_result: Dict[str, Any],
    rag_result: Dict[str, Any],
    task_mode: Optional[str] = None,
) -> StructuredSummary:
    items = []
    items.extend(_extract_case_items(case_payload, task_mode=task_mode))
    items.extend(_extract_tool_items(tools_result))
    items.extend(_extract_imaging_items(imaging_result))
    items.extend(_extract_knowledge_items(rag_result, source_type="rag", tag_prefix="RAG"))
    deduped_items = _dedupe_items(items)

    patient_state = _take_display_lines([item for item in deduped_items if item.category == "patient_state"], max_items=4)
    supporting_evidence = _take_display_lines(
        [
            item
            for item in deduped_items
            if item.polarity == "supporting"
            and item.category in {"supporting_evidence", "suspected_source", "organ_dysfunction", "hemodynamics", "microbiology"}
        ],
        max_items=6,
    )
    opposing_evidence = _take_display_lines(
        [item for item in deduped_items if item.polarity == "opposing" and item.category != "knowledge_support"],
        max_items=4,
    )

    source_lines = _take_display_lines([item for item in deduped_items if item.category == "suspected_source"], max_items=4)
    suspected_sources: List[str] = []
    seen_sources = set()
    for line in source_lines:
        labels = _detect_source_labels(line)
        if not labels:
            labels = [clean_text(line)]
        for label in labels:
            if label not in seen_sources:
                seen_sources.add(label)
                suspected_sources.append(label if label == line else f"{label} [{line.split('[')[-1].rstrip(']') if '[' in line else 'CASE1'}]")

    organ_dysfunction = _take_display_lines(
        [item for item in deduped_items if item.category in {"organ_dysfunction", "hemodynamics"} and item.polarity == "supporting"],
        max_items=5,
    )
    treatment_constraints = _build_treatment_constraints(case_payload, deduped_items, suspected_sources, organ_dysfunction)
    knowledge_support = _take_display_lines([item for item in deduped_items if item.category == "knowledge_support"], max_items=3)
    conflicts = _build_conflicts(supporting_evidence, opposing_evidence, suspected_sources, organ_dysfunction)
    missing_information = _build_missing_information(deduped_items, suspected_sources, organ_dysfunction, treatment_constraints, imaging_result)

    summary = StructuredSummary(
        patient_state=patient_state,
        supporting_evidence=supporting_evidence,
        opposing_evidence=opposing_evidence,
        suspected_sources=suspected_sources,
        organ_dysfunction=organ_dysfunction,
        treatment_constraints=treatment_constraints,
        knowledge_support=knowledge_support,
        conflicts=conflicts,
        missing_information=missing_information,
        prompt_text="",
        evidence_items=deduped_items,
    )
    summary.prompt_text = render_summary_prompt(summary)
    return summary
