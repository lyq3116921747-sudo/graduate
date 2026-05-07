from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .utils import case_to_context, clean_text, normalize_case_payload


def _extract_numeric(text: str, patterns: Sequence[str]) -> Optional[float]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        groups = [group for group in match.groups() if group is not None]
        if not groups:
            continue
        try:
            return float(groups[-1])
        except Exception:
            continue
    return None


def _extract_python_code(text: str) -> str:
    normalized = clean_text(text)
    if not normalized:
        return ""
    fenced_match = re.search(r"```(?:python)?\s*(.*?)```", normalized, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return clean_text(fenced_match.group(1))
    return normalized


SEPSIS3_DIRECT_INFECTION_TERMS = (
    "肺部感染",
    "肺炎",
    "胆管炎",
    "胆囊炎",
    "尿路感染",
    "泌尿系感染",
    "肾盂肾炎",
    "腹膜炎",
    "腹腔感染",
    "脓肿",
    "蜂窝织炎",
    "褥疮感染",
    "菌血症",
    "败血症",
    "感染性坏死",
    "化脓",
    "宫腔感染",
    "盆腔感染",
    "切口感染",
    "胆道感染",
)
SEPSIS3_GENERIC_INFECTION_TERMS = ("感染",)
SEPSIS3_INFECTION_EXCLUDE_TERMS = (
    "预防感染",
    "防治感染",
    "围手术期常规抗感染",
    "术后常规抗感染",
    "感染风险",
    "感染指标",
)
SEPSIS3_PATHOGEN_TERMS = (
    "克雷伯",
    "大肠埃希",
    "铜绿假单胞菌",
    "鲍曼不动杆菌",
    "金黄色葡萄球菌",
    "链球菌",
    "肠球菌",
    "白色念珠菌",
    "念珠菌",
    "真菌",
    "革兰阴性杆菌",
    "革兰阳性球菌",
)
SEPSIS3_POSITIVE_MICROBIOLOGY_TERMS = ("培养提示", "培养出", "检出", "阳性", "生长")
SEPSIS3_NEGATIVE_MICROBIOLOGY_TERMS = ("未检出", "阴性", "未发现")
SEPSIS3_UNCERTAIN_TERMS = (
    "未排除脓毒性休克",
    "脓毒性休克未排除",
    "考虑脓毒症",
    "脓毒症待排",
    "脓毒性休克待排",
    "不除外脓毒性休克",
    "疑似感染",
    "感染待排",
)
SEPSIS3_COMPETING_TERMS = (
    "心源性休克",
    "失血性休克",
    "出血性休克",
    "术后出血",
    "围手术期低血压",
    "急性冠脉综合征",
    "心肌梗死",
    "心电机械分离",
    "室性心动过速",
    "脑出血",
    "脑疝",
    "大面积脑梗死",
    "肺栓塞",
)
SEPSIS3_DOMINANT_COMPETING_TERMS = ("心源性休克", "急性冠脉综合征", "心肌梗死", "肺栓塞", "失血性休克", "出血性休克")
SEPSIS3_CHRONIC_ORGAN_TERMS = {
    "renal": ("慢性肾", "CKD", "尿毒症", "维持性血液透析", "慢性肾功能不全"),
    "liver": ("肝硬化", "慢性肝病", "慢性乙肝", "肝炎后肝硬化"),
    "cns": ("痴呆", "长期意识障碍", "脑卒中后遗症", "长期昏迷"),
    "coagulation": ("血小板减少症", "再生障碍性贫血", "骨髓增生异常"),
}
SEPSIS3_ACUTE_ORGAN_HINTS = {
    "respiration": ("呼吸衰竭", "氧合下降", "机械通气", "呼吸机", "高流量吸氧", "插管"),
    "coagulation": ("凝血异常", "血小板下降", "DIC", "纤维蛋白原减低", "凝血功能障碍"),
    "liver": ("肝功能恶化", "黄疸", "急性肝功能不全", "胆红素升高", "肝损伤"),
    "cardiovascular": ("低血压", "休克", "血压测不出", "血压下降", "需升压药维持", "升压药"),
    "cns": ("意识模糊", "意识改变", "嗜睡", "谵妄", "昏迷", "神志淡漠", "新发意识障碍"),
    "renal": ("少尿", "无尿", "肌酐进行性升高", "肾功能恶化", "急性肾损伤", "AKI", "CRRT"),
}
SEPSIS3_VASOPRESSOR_TERMS = ("去甲肾上腺素", "肾上腺素", "多巴胺", "多巴酚丁胺", "去甲肾")
SEPSIS3_PRESENT_STATES = {"present", "positive", "yes", "true", "supported", "有", "存在", "明确"}
SEPSIS3_ABSENT_STATES = {"absent", "negative", "no", "false", "none", "无", "不存在", "未见"}


def _normalize_items(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    normalized: List[str] = []
    for item in items:
        text = clean_text(item)
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            numeric_value = float(value)
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return None
            return numeric_value
        except Exception:
            return None
    text = clean_text(value).replace(",", "")
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = clean_text(value).lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "是", "有", "存在"}:
        return True
    if text in {"0", "false", "no", "n", "否", "无", "不存在"}:
        return False
    return None


def _contains_any(text: str, keywords: Sequence[str]) -> bool:
    normalized = clean_text(text)
    return bool(normalized) and any(keyword in normalized for keyword in keywords)


def _is_positive_microbiology(text: str) -> bool:
    normalized = clean_text(text)
    if not normalized or _contains_any(normalized, SEPSIS3_NEGATIVE_MICROBIOLOGY_TERMS):
        return False
    return _contains_any(normalized, SEPSIS3_POSITIVE_MICROBIOLOGY_TERMS) or _contains_any(normalized, SEPSIS3_PATHOGEN_TERMS)


def _is_infection_text(text: str) -> bool:
    normalized = clean_text(text)
    if not normalized:
        return False
    if _contains_any(normalized, SEPSIS3_DIRECT_INFECTION_TERMS):
        return True
    if _contains_any(normalized, SEPSIS3_GENERIC_INFECTION_TERMS):
        return not _contains_any(normalized, SEPSIS3_INFECTION_EXCLUDE_TERMS)
    return False


def _normalize_state(value: Any) -> str:
    text = clean_text(value).lower()
    if text in SEPSIS3_PRESENT_STATES:
        return "present"
    if text in SEPSIS3_ABSENT_STATES:
        return "absent"
    return "uncertain"


def _normalize_component_payload(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


class Sepsis3Judge:
    def normalize_extraction_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = payload if isinstance(payload, dict) else {}
        acute_flags = self._normalize_organ_flag_map(normalized.get("acute_change_flags"))
        return {
            "infection_status": _normalize_state(normalized.get("infection_status")),
            "infection_evidence": _normalize_items(normalized.get("infection_evidence")),
            "infection_source": _normalize_items(normalized.get("infection_source")),
            "reported_sofa": _safe_float(normalized.get("reported_sofa")),
            "respiration": _normalize_component_payload(normalized.get("respiration")),
            "coagulation": _normalize_component_payload(normalized.get("coagulation")),
            "liver": _normalize_component_payload(normalized.get("liver")),
            "cardiovascular": _normalize_component_payload(normalized.get("cardiovascular")),
            "cns": _normalize_component_payload(normalized.get("cns")),
            "renal": _normalize_component_payload(normalized.get("renal")),
            "lactate": _safe_float(normalized.get("lactate")),
            "acute_change_flags": acute_flags,
            "chronic_confounders": _normalize_items(normalized.get("chronic_confounders")),
            "competing_noninfectious_causes": _normalize_items(normalized.get("competing_noninfectious_causes")),
            "shock_terms": _normalize_items(normalized.get("shock_terms")),
            "uncertainty_notes": _normalize_items(normalized.get("uncertainty_notes")),
            "evidence_quotes": _normalize_items(normalized.get("evidence_quotes")),
        }

    def evaluate(
        self,
        extraction_payload: Dict[str, Any],
        sofa_result: Optional[Dict[str, Any]] = None,
        case_payload: Optional[Dict[str, Any]] = None,
        *,
        enable_case_backfill: bool = True,
        enable_sofa_backfill: bool = True,
    ) -> Dict[str, Any]:
        extraction = self.normalize_extraction_payload(extraction_payload)
        sofa_payload = sofa_result if isinstance(sofa_result, dict) else {}
        merged = dict(extraction)
        if enable_case_backfill:
            merged = self._merge_with_case_payload(merged, case_payload)
        if enable_sofa_backfill:
            merged = self._merge_with_sofa_tool(merged, sofa_payload)

        infection_status = self._finalize_infection_status(merged)
        competing_causes = self._collect_competing_causes(merged)
        dominant_competing = any(_contains_any(item, SEPSIS3_DOMINANT_COMPETING_TERMS) for item in competing_causes)
        uncertainty_notes = list(merged["uncertainty_notes"])
        if merged["infection_status"] == "uncertain":
            uncertainty_notes.append("感染状态存在不确定性")

        computed_total, component_scores, counted_positive_organs = self._compute_sofa(merged)
        reported_sofa = merged["reported_sofa"]
        vasopressor_used = bool(merged["cardiovascular"].get("vasopressor_used"))
        lactate = _safe_float(merged.get("lactate"))
        map_value = self._resolve_map(merged["cardiovascular"])
        shock_supported = (
            infection_status == "present"
            and vasopressor_used
            and lactate is not None
            and lactate > 2
            and (
                (map_value is not None and map_value < 65)
                or bool(merged["cardiovascular"].get("low_bp_text"))
                or bool(merged["cardiovascular"].get("shock_text"))
                or bool(merged["shock_terms"])
            )
        )

        basis = self._build_basis(
            infection_status=infection_status,
            merged=merged,
            reported_sofa=reported_sofa,
            computed_total=computed_total,
            component_scores=component_scores,
            shock_supported=shock_supported,
            competing_causes=competing_causes,
            uncertainty_notes=uncertainty_notes,
        )

        positive_sofa = (reported_sofa is not None and reported_sofa >= 2) or (computed_total is not None and computed_total >= 2)
        if infection_status == "absent":
            classification = "non_sepsis"
            diagnosis_result = "目前不符合 Sepsis-3 脓毒症"
            reason = "未见可靠感染证据。"
            binary_label: Optional[int] = 0
        elif infection_status != "present":
            classification = "insufficient_evidence"
            diagnosis_result = "当前证据不足以按 Sepsis-3 判定为脓毒症"
            reason = "感染证据不足或存在冲突。"
            binary_label = None
        elif dominant_competing:
            classification = "insufficient_evidence"
            diagnosis_result = "当前证据不足以按 Sepsis-3 判定为脓毒症"
            reason = "存在更强的非感染性竞争病因。"
            binary_label = None
        elif shock_supported:
            classification = "sepsis"
            diagnosis_result = "符合 Sepsis-3 脓毒症，并存在脓毒性休克方向证据"
            reason = "存在感染证据，并满足升压药、乳酸升高和低灌注/低血压组合条件。"
            binary_label = 1
        elif positive_sofa:
            classification = "sepsis"
            diagnosis_result = "符合 Sepsis-3 脓毒症"
            reason = "存在感染证据，且 SOFA 达到 2 分及以上。"
            binary_label = 1
        elif computed_total is None and reported_sofa is None:
            classification = "insufficient_evidence"
            diagnosis_result = "当前证据不足以按 Sepsis-3 判定为脓毒症"
            reason = "关键 SOFA 指标缺失或受慢性基线混杂。"
            binary_label = None
        else:
            classification = "non_sepsis"
            diagnosis_result = "目前不符合 Sepsis-3 脓毒症"
            reason = "虽有感染线索，但当前不满足 SOFA>=2，也无脓毒性休克证据。"
            binary_label = 0

        if counted_positive_organs == 0 and computed_total == 0 and infection_status == "present":
            reason = "虽有感染线索，但未提取到可计入 SOFA 的急性器官功能障碍证据。"

        return {
            "classification": classification,
            "binary_label": binary_label,
            "diagnosis_result": diagnosis_result,
            "diagnostic_basis": basis,
            "sepsis3_reason": reason,
            "infection_status": infection_status,
            "shock_supported": shock_supported,
            "reported_sofa": int(reported_sofa) if reported_sofa is not None else None,
            "computed_sofa": computed_total,
            "component_scores": component_scores,
            "competing_causes": competing_causes,
            "uncertainty_notes": uncertainty_notes,
        }

    def _normalize_organ_flag_map(self, value: Any) -> Dict[str, List[str]]:
        flags = {organ: [] for organ in ("respiration", "coagulation", "liver", "cardiovascular", "cns", "renal", "general")}
        if isinstance(value, dict):
            for key, raw in value.items():
                target = "general"
                key_text = clean_text(key).lower()
                for organ in flags:
                    if organ != "general" and organ in key_text:
                        target = organ
                        break
                for item in _normalize_items(raw):
                    if item not in flags[target]:
                        flags[target].append(item)
        else:
            for item in _normalize_items(value):
                flags["general"].append(item)

        for organ, keywords in SEPSIS3_ACUTE_ORGAN_HINTS.items():
            for bucket in ("general", organ):
                for text in list(flags[bucket]):
                    if _contains_any(text, keywords) and text not in flags[organ]:
                        flags[organ].append(text)
        return flags

    def _merge_with_sofa_tool(self, extraction: Dict[str, Any], sofa_payload: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(extraction)
        raw_values = sofa_payload.get("raw_values", {}) if isinstance(sofa_payload.get("raw_values", {}), dict) else {}

        merged["reported_sofa"] = extraction["reported_sofa"]
        if merged["reported_sofa"] is None:
            merged["reported_sofa"] = _safe_float(sofa_payload.get("reported_total"))

        respiration = dict(extraction["respiration"])
        coagulation = dict(extraction["coagulation"])
        liver = dict(extraction["liver"])
        cardiovascular = dict(extraction["cardiovascular"])
        cns = dict(extraction["cns"])
        renal = dict(extraction["renal"])

        self._fill_missing(respiration, "pf_ratio", raw_values.get("pf_ratio"))
        self._fill_missing(respiration, "pao2", raw_values.get("pao2"))
        self._fill_missing(respiration, "fio2", raw_values.get("fio2"))
        self._fill_missing(coagulation, "platelets", raw_values.get("platelets"))
        self._fill_missing(liver, "bilirubin", raw_values.get("bilirubin"))
        liver.setdefault("unit", clean_text(raw_values.get("bilirubin_unit", "")))
        self._fill_missing(cardiovascular, "sbp", raw_values.get("sbp"))
        self._fill_missing(cardiovascular, "dbp", raw_values.get("dbp"))
        self._fill_missing(cardiovascular, "map", raw_values.get("map"))
        if cardiovascular.get("vasopressor_used") is None and raw_values.get("vasopressor_used") is not None:
            cardiovascular["vasopressor_used"] = raw_values.get("vasopressor_used")
        self._fill_missing(cns, "gcs", raw_values.get("gcs"))
        self._fill_missing(renal, "creatinine", raw_values.get("creatinine"))
        renal.setdefault("unit", clean_text(raw_values.get("creatinine_unit", "")))
        if merged.get("lactate") is None:
            merged["lactate"] = _safe_float(raw_values.get("lactate"))

        cardiovascular["vasopressor_name"] = clean_text(cardiovascular.get("vasopressor_name", "")) or clean_text(raw_values.get("vasopressor_name", ""))
        cardiovascular["low_bp_text"] = bool(cardiovascular.get("low_bp_text")) or bool(raw_values.get("low_bp_text"))
        cardiovascular["shock_text"] = bool(cardiovascular.get("shock_text")) or bool(raw_values.get("shock_text"))

        merged["respiration"] = respiration
        merged["coagulation"] = coagulation
        merged["liver"] = liver
        merged["cardiovascular"] = cardiovascular
        merged["cns"] = cns
        merged["renal"] = renal
        return merged

    def _merge_with_case_payload(self, extraction: Dict[str, Any], case_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(case_payload, dict):
            return dict(extraction)

        merged = dict(extraction)
        respiration = dict(extraction["respiration"])
        coagulation = dict(extraction["coagulation"])
        liver = dict(extraction["liver"])
        cardiovascular = dict(extraction["cardiovascular"])
        cns = dict(extraction["cns"])
        renal = dict(extraction["renal"])

        lab_backfill = self._extract_structured_lab_backfill(case_payload)
        text_backfill = self._extract_text_backfill(case_payload)

        self._fill_missing(respiration, "pf_ratio", lab_backfill.get("pf_ratio"))
        self._fill_missing(respiration, "pf_ratio", text_backfill.get("pf_ratio"))
        self._fill_missing(respiration, "pao2", lab_backfill.get("pao2"))
        self._fill_missing(respiration, "pao2", text_backfill.get("pao2"))
        self._fill_missing(respiration, "fio2", lab_backfill.get("fio2"))
        self._fill_missing(respiration, "fio2", text_backfill.get("fio2"))

        self._fill_missing(coagulation, "platelets", lab_backfill.get("platelets"))
        self._fill_missing(liver, "bilirubin", lab_backfill.get("bilirubin"))
        if not clean_text(liver.get("unit", "")):
            liver["unit"] = clean_text(lab_backfill.get("bilirubin_unit", ""))

        self._fill_missing(cardiovascular, "sbp", text_backfill.get("sbp"))
        self._fill_missing(cardiovascular, "dbp", text_backfill.get("dbp"))
        self._fill_missing(cardiovascular, "map", text_backfill.get("map"))
        if cardiovascular.get("vasopressor_used") is None and text_backfill.get("vasopressor_used") is not None:
            cardiovascular["vasopressor_used"] = text_backfill.get("vasopressor_used")
        cardiovascular["vasopressor_name"] = clean_text(cardiovascular.get("vasopressor_name", "")) or clean_text(
            text_backfill.get("vasopressor_name", "")
        )
        cardiovascular["low_bp_text"] = bool(cardiovascular.get("low_bp_text")) or bool(text_backfill.get("low_bp_text"))
        cardiovascular["shock_text"] = bool(cardiovascular.get("shock_text")) or bool(text_backfill.get("shock_text"))

        self._fill_missing(cns, "gcs", text_backfill.get("gcs"))
        if cns.get("new_mental_status_change") is None and text_backfill.get("new_mental_status_change") is not None:
            cns["new_mental_status_change"] = text_backfill.get("new_mental_status_change")

        self._fill_missing(renal, "creatinine", lab_backfill.get("creatinine"))
        if not clean_text(renal.get("unit", "")):
            renal["unit"] = clean_text(lab_backfill.get("creatinine_unit", ""))
        if renal.get("oliguria") is None and text_backfill.get("oliguria") is not None:
            renal["oliguria"] = text_backfill.get("oliguria")

        if merged.get("lactate") is None:
            merged["lactate"] = _safe_float(lab_backfill.get("lactate"))
        if merged.get("lactate") is None:
            merged["lactate"] = _safe_float(text_backfill.get("lactate"))
        if merged.get("reported_sofa") is None:
            merged["reported_sofa"] = _safe_float(text_backfill.get("reported_sofa"))

        if _safe_float(cns.get("gcs")) is not None:
            merged["uncertainty_notes"] = [note for note in merged["uncertainty_notes"] if "GCS未明确记录" not in note]

        for item in _normalize_items(text_backfill.get("shock_terms")):
            if item not in merged["shock_terms"]:
                merged["shock_terms"].append(item)

        merged["respiration"] = respiration
        merged["coagulation"] = coagulation
        merged["liver"] = liver
        merged["cardiovascular"] = cardiovascular
        merged["cns"] = cns
        merged["renal"] = renal
        return merged

    def _extract_structured_lab_backfill(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "platelets": None,
            "bilirubin": None,
            "bilirubin_unit": "",
            "creatinine": None,
            "creatinine_unit": "",
            "lactate": None,
            "pao2": None,
            "fio2": None,
            "pf_ratio": None,
        }
        best_pf_ratio: Optional[float] = None
        respiration_by_time: Dict[str, Dict[str, Any]] = {}

        for entry in self._iter_lab_entries(case_payload):
            indicator_name = clean_text(entry.get("检验指标名", ""))
            value = _safe_float(entry.get("检验指标值"))
            unit = clean_text(entry.get("单位", ""))
            if not indicator_name or value is None:
                continue

            report_time = clean_text(entry.get("报告日期", "")) or "__missing__"

            if self._is_platelet_indicator(indicator_name):
                if result["platelets"] is None or value < result["platelets"]:
                    result["platelets"] = value
                continue

            if self._is_bilirubin_indicator(indicator_name):
                candidate = self._convert_bilirubin_to_mgdl(value, unit)
                current = self._convert_bilirubin_to_mgdl(_safe_float(result["bilirubin"]), clean_text(result["bilirubin_unit"]))
                if candidate is not None and (current is None or candidate > current):
                    result["bilirubin"] = value
                    result["bilirubin_unit"] = unit
                continue

            if self._is_creatinine_indicator(indicator_name):
                candidate = self._convert_creatinine_to_mgdl(value, unit)
                current = self._convert_creatinine_to_mgdl(_safe_float(result["creatinine"]), clean_text(result["creatinine_unit"]))
                if candidate is not None and (current is None or candidate > current):
                    result["creatinine"] = value
                    result["creatinine_unit"] = unit
                continue

            if self._is_lactate_indicator(indicator_name):
                if result["lactate"] is None or value > result["lactate"]:
                    result["lactate"] = value
                continue

            if self._is_pao2_indicator(indicator_name):
                pao2_value = value * 7.50062 if "kpa" in unit.lower() else value
                bucket = respiration_by_time.setdefault(report_time, {})
                bucket["pao2"] = pao2_value
                continue

            if self._is_fio2_indicator(indicator_name):
                fio2_value = value / 100.0 if value > 1.5 else value
                bucket = respiration_by_time.setdefault(report_time, {})
                bucket["fio2"] = fio2_value

        for values in respiration_by_time.values():
            pao2_value = _safe_float(values.get("pao2"))
            fio2_value = _safe_float(values.get("fio2"))
            if pao2_value is not None and result["pao2"] is None:
                result["pao2"] = pao2_value
            if fio2_value is not None and result["fio2"] is None:
                result["fio2"] = fio2_value
            if pao2_value is None or fio2_value in {None, 0}:
                continue
            pf_ratio = pao2_value / fio2_value
            if best_pf_ratio is None or pf_ratio < best_pf_ratio:
                best_pf_ratio = pf_ratio
                result["pf_ratio"] = pf_ratio
                result["pao2"] = pao2_value
                result["fio2"] = fio2_value

        return result

    def _iter_lab_entries(self, case_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_entries = case_payload.get("实验室检查", [])
        if isinstance(raw_entries, dict):
            raw_entries = [raw_entries]
        if not isinstance(raw_entries, list):
            return []
        return [entry for entry in raw_entries if isinstance(entry, dict)]

    def _extract_text_backfill(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "reported_sofa": None,
            "pf_ratio": None,
            "pao2": None,
            "fio2": None,
            "sbp": None,
            "dbp": None,
            "map": None,
            "vasopressor_used": None,
            "vasopressor_name": "",
            "low_bp_text": False,
            "shock_text": False,
            "gcs": None,
            "new_mental_status_change": None,
            "oliguria": None,
            "lactate": None,
            "shock_terms": [],
        }

        min_map: Optional[float] = None
        min_gcs: Optional[float] = None
        vasopressor_hits: List[str] = []
        shock_terms: List[str] = []

        for text in self._iter_case_texts(case_payload):
            reported_sofa = _extract_numeric(text, [r"SOFA评分[:：]?\s*([0-9]+(?:\.\d+)?)"])
            if result["reported_sofa"] is None and reported_sofa is not None:
                result["reported_sofa"] = reported_sofa

            pf_ratio = _extract_numeric(text, [r"氧合指数[^0-9]{0,10}([0-9]+(?:\.\d+)?)"])
            if pf_ratio is not None and (result["pf_ratio"] is None or pf_ratio < result["pf_ratio"]):
                result["pf_ratio"] = pf_ratio

            bp_match = re.search(
                r"(?:BP|血压)[:：]?\s*([0-9]{2,3})\s*/\s*([0-9]{2,3})\s*(?:mmHg|mmhg)?",
                text,
                flags=re.IGNORECASE,
            )
            if bp_match:
                sbp = float(bp_match.group(1))
                dbp = float(bp_match.group(2))
                map_value = (sbp + 2 * dbp) / 3.0
                if min_map is None or map_value < min_map:
                    min_map = map_value
                    result["sbp"] = sbp
                    result["dbp"] = dbp
                    result["map"] = map_value

            gcs_score: Optional[float] = None
            evm_match = re.search(r"GCS[^EV0-9]{0,10}E\s*([1-4])\s*V\s*([1-5])\s*M\s*([1-6])", text, flags=re.IGNORECASE)
            if evm_match:
                gcs_score = float(sum(int(part) for part in evm_match.groups()))
            else:
                for pattern in (
                    r"GCS积分\s*[:：=]\s*((?:1[0-5])|(?:[3-9]))\b",
                    r"GCS评分\s*[:：=]\s*((?:1[0-5])|(?:[3-9]))\b",
                    r"GCS\s*[:：=]\s*((?:1[0-5])|(?:[3-9]))\b",
                ):
                    gcs_match = re.search(pattern, text, flags=re.IGNORECASE)
                    if gcs_match:
                        gcs_score = _safe_float(gcs_match.group(1))
                        break
            if gcs_score is not None and (min_gcs is None or gcs_score < min_gcs):
                min_gcs = gcs_score
                result["gcs"] = gcs_score

            lactate = _extract_numeric(
                text,
                [
                    r"乳酸浓度[^0-9]{0,20}([0-9]+(?:\.\d+)?)",
                    r"乳酸(?!脱氢酶)[^0-9]{0,20}([0-9]+(?:\.\d+)?)",
                    r"\bLac\b[^0-9]{0,20}([0-9]+(?:\.\d+)?)",
                ],
            )
            if lactate is not None and (result["lactate"] is None or lactate > result["lactate"]):
                result["lactate"] = lactate

            for term in SEPSIS3_VASOPRESSOR_TERMS:
                if term in text and term not in vasopressor_hits:
                    vasopressor_hits.append(term)
            if any(term in text for term in ("低血压", "血压下降", "血压测不出")):
                result["low_bp_text"] = True
            if "休克" in text:
                result["shock_text"] = True
                if "休克" not in shock_terms:
                    shock_terms.append("休克")
            if any(term in text for term in ("意识模糊", "谵妄", "昏迷", "嗜睡", "神志淡漠", "新发意识障碍")):
                result["new_mental_status_change"] = True
            if any(term in text for term in ("少尿", "无尿", "尿量减少")):
                result["oliguria"] = True

        if vasopressor_hits:
            result["vasopressor_used"] = True
            result["vasopressor_name"] = "、".join(vasopressor_hits)
        result["shock_terms"] = shock_terms
        return result

    def _iter_case_texts(self, case_payload: Dict[str, Any]) -> List[str]:
        texts: List[str] = []
        top_level_keys = (
            "主诉",
            "现病史",
            "入院情况",
            "体格检查",
            "辅助检查结果",
            "西医入院诊断",
            "西医出院诊断",
            "出院情况",
            "出院医嘱",
            "入院诊断",
            "出院诊断",
            "出院记录",
        )
        for key in top_level_keys:
            text = clean_text(case_payload.get(key, ""))
            if text:
                texts.append(text)
        for entry in case_payload.get("病程记录", []) if isinstance(case_payload.get("病程记录", []), list) else []:
            if not isinstance(entry, dict):
                continue
            for key in ("单次记录", "记录名称"):
                text = clean_text(entry.get(key, ""))
                if text:
                    texts.append(text)
        return texts

    def _is_platelet_indicator(self, name: str) -> bool:
        normalized = clean_text(name).upper()
        if "PLT" == normalized:
            return True
        if "血小板计数" in normalized:
            return True
        return False

    def _is_bilirubin_indicator(self, name: str) -> bool:
        normalized = clean_text(name).upper()
        return "总胆红素" in normalized or normalized == "TBIL"

    def _is_creatinine_indicator(self, name: str) -> bool:
        normalized = clean_text(name).upper()
        if normalized in {"肌酐", "CREA", "CR"}:
            return True
        return normalized.endswith("肌酐") and "尿素氮/肌酐" not in normalized

    def _is_lactate_indicator(self, name: str) -> bool:
        normalized = clean_text(name)
        return normalized in {"乳酸", "乳酸浓度", "Lac", "LAC"}

    def _is_pao2_indicator(self, name: str) -> bool:
        normalized = clean_text(name)
        if normalized in {"氧分压", "PaO2", "PO2"}:
            return True
        return False

    def _is_fio2_indicator(self, name: str) -> bool:
        normalized = clean_text(name)
        return normalized in {"吸氧浓度", "FiO2", "氧浓度"}

    def _fill_missing(self, payload: Dict[str, Any], key: str, value: Any) -> None:
        if key not in payload or payload.get(key) in {"", None}:
            payload[key] = value

    def _finalize_infection_status(self, merged: Dict[str, Any]) -> str:
        status = merged["infection_status"]
        infection_text = "\n".join(
            merged["infection_evidence"] + merged["infection_source"] + merged["evidence_quotes"] + merged["uncertainty_notes"]
        )
        if status == "present":
            return "present"
        if status == "absent" and not (_is_infection_text(infection_text) or _is_positive_microbiology(infection_text)):
            return "absent"
        if _is_positive_microbiology(infection_text) or _is_infection_text(infection_text) or merged["infection_source"]:
            return "present"
        if _contains_any(infection_text, SEPSIS3_UNCERTAIN_TERMS):
            return "uncertain"
        return status

    def _collect_competing_causes(self, merged: Dict[str, Any]) -> List[str]:
        competing: List[str] = []
        for item in merged["competing_noninfectious_causes"] + merged["evidence_quotes"] + merged["uncertainty_notes"]:
            if _contains_any(item, SEPSIS3_COMPETING_TERMS) and item not in competing:
                competing.append(item)
        return competing

    def _compute_sofa(self, merged: Dict[str, Any]) -> Tuple[Optional[int], Dict[str, Dict[str, Any]], int]:
        component_scores: Dict[str, Dict[str, Any]] = {}
        total = 0
        positive_organs = 0
        any_known = False

        respiration_pf = self._resolve_pf_ratio(merged["respiration"])
        coagulation_value = _safe_float(merged["coagulation"].get("platelets"))
        liver_value = self._convert_bilirubin_to_mgdl(_safe_float(merged["liver"].get("bilirubin")), clean_text(merged["liver"].get("unit", "")))
        cardiovascular_map = self._resolve_map(merged["cardiovascular"])
        cardiovascular_vasopressor = self._resolve_vasopressor_used(merged["cardiovascular"])
        cns_gcs = _safe_float(merged["cns"].get("gcs"))
        renal_value = self._convert_creatinine_to_mgdl(_safe_float(merged["renal"].get("creatinine")), clean_text(merged["renal"].get("unit", "")))

        raw_components = {
            "respiration": (respiration_pf, SOFATool()._respiration_score(respiration_pf), "PaO2/FiO2"),
            "coagulation": (coagulation_value, SOFATool()._coagulation_score(coagulation_value), "血小板"),
            "liver": (liver_value, SOFATool()._liver_score(liver_value), "总胆红素(mg/dL)"),
            "cardiovascular": (
                cardiovascular_map if cardiovascular_map is not None else ("vasopressor" if cardiovascular_vasopressor else None),
                self._cardiovascular_score(cardiovascular_map, cardiovascular_vasopressor),
                "MAP/血管活性药",
            ),
            "cns": (cns_gcs, SOFATool()._cns_score(cns_gcs), "GCS"),
            "renal": (renal_value, SOFATool()._renal_score(renal_value), "肌酐(mg/dL)"),
        }

        for organ, (value, score, description) in raw_components.items():
            acute_supported = self._organ_has_acute_support(organ, merged)
            confounded = self._organ_is_confounded(organ, merged) and not acute_supported
            counted = score is not None and not confounded
            component_scores[organ] = {
                "description": description,
                "value": value,
                "score": score,
                "acute_supported": acute_supported,
                "confounded": confounded,
                "counted": counted,
            }
            if score is None:
                continue
            any_known = True
            if counted:
                total += score
                if score > 0:
                    positive_organs += 1

        return (total if any_known else None), component_scores, positive_organs

    def _resolve_pf_ratio(self, respiration: Dict[str, Any]) -> Optional[float]:
        pf_ratio = _safe_float(respiration.get("pf_ratio"))
        if pf_ratio is not None:
            return pf_ratio
        pao2 = _safe_float(respiration.get("pao2"))
        fio2 = _safe_float(respiration.get("fio2"))
        if fio2 is not None and fio2 > 1.5:
            fio2 = fio2 / 100.0
        if pao2 is None or fio2 in {None, 0}:
            return None
        return pao2 / fio2

    def _resolve_map(self, cardiovascular: Dict[str, Any]) -> Optional[float]:
        map_value = _safe_float(cardiovascular.get("map"))
        if map_value is not None:
            return map_value
        sbp = _safe_float(cardiovascular.get("sbp"))
        dbp = _safe_float(cardiovascular.get("dbp"))
        if sbp is None or dbp is None:
            return None
        return (sbp + 2 * dbp) / 3.0

    def _resolve_vasopressor_used(self, cardiovascular: Dict[str, Any]) -> bool:
        bool_value = _safe_bool(cardiovascular.get("vasopressor_used"))
        if bool_value is not None:
            return bool_value
        return _contains_any(clean_text(cardiovascular.get("vasopressor_name", "")), SEPSIS3_VASOPRESSOR_TERMS)

    def _cardiovascular_score(self, map_value: Optional[float], vasopressor_used: bool) -> Optional[int]:
        if vasopressor_used:
            return 2
        if map_value is None:
            return None
        return 1 if map_value < 70 else 0

    def _organ_has_acute_support(self, organ: str, merged: Dict[str, Any]) -> bool:
        acute_flags = merged["acute_change_flags"]
        if acute_flags.get(organ):
            return True
        if any(_contains_any(item, SEPSIS3_ACUTE_ORGAN_HINTS.get(organ, ())) for item in acute_flags.get("general", [])):
            return True
        if organ == "cardiovascular":
            return self._resolve_vasopressor_used(merged["cardiovascular"]) or bool(merged["shock_terms"])
        if organ == "cns" and _safe_bool(merged["cns"].get("new_mental_status_change")):
            return True
        if organ == "renal" and _safe_bool(merged["renal"].get("oliguria")):
            return True
        return False

    def _organ_is_confounded(self, organ: str, merged: Dict[str, Any]) -> bool:
        chronic_text = "\n".join(merged["chronic_confounders"] + merged["evidence_quotes"])
        return _contains_any(chronic_text, SEPSIS3_CHRONIC_ORGAN_TERMS.get(organ, ()))

    def _convert_bilirubin_to_mgdl(self, value: Optional[float], unit: str) -> Optional[float]:
        if value is None:
            return None
        unit_text = unit.lower()
        if "umol" in unit_text or "μmol" in unit_text or "µmol" in unit_text:
            return value / 17.1
        return value / 17.1 if value > 20 else value

    def _convert_creatinine_to_mgdl(self, value: Optional[float], unit: str) -> Optional[float]:
        if value is None:
            return None
        unit_text = unit.lower()
        if "umol" in unit_text or "μmol" in unit_text or "µmol" in unit_text:
            return value / 88.4
        return value / 88.4 if value > 20 else value

    def _build_basis(
        self,
        *,
        infection_status: str,
        merged: Dict[str, Any],
        reported_sofa: Optional[float],
        computed_total: Optional[int],
        component_scores: Dict[str, Dict[str, Any]],
        shock_supported: bool,
        competing_causes: List[str],
        uncertainty_notes: List[str],
    ) -> List[str]:
        basis: List[str] = []
        if infection_status == "present":
            infection_evidence = merged["infection_evidence"] + merged["infection_source"]
            evidence_text = "；".join(infection_evidence[:3]) or "已提取到感染相关线索"
            basis.append(f"感染证据：{evidence_text}")
        elif infection_status == "absent":
            basis.append("未提取到可靠感染证据。")
        else:
            notes = "；".join((merged["uncertainty_notes"] + merged["evidence_quotes"])[:2]) or "感染证据不足或存在冲突。"
            basis.append(f"感染判断不确定：{notes}")

        sofa_parts: List[str] = []
        if reported_sofa is not None:
            sofa_parts.append(f"病历显式 SOFA={int(reported_sofa)}")
        if computed_total is not None:
            positive_components = []
            for organ, item in component_scores.items():
                if item.get("score") is None or not item.get("counted"):
                    continue
                positive_components.append(f"{organ}={item['score']}")
            component_text = "，".join(positive_components) if positive_components else "无可计入的阳性分项"
            sofa_parts.append(f"可提取指标计算 SOFA={computed_total}（{component_text}）")
        if sofa_parts:
            basis.append("；".join(sofa_parts))

        if shock_supported:
            lactate = merged.get("lactate")
            map_value = self._resolve_map(merged["cardiovascular"])
            shock_text = f"升压药={self._resolve_vasopressor_used(merged['cardiovascular'])}，乳酸={lactate}，MAP={map_value}"
            basis.append(f"脓毒性休克方向证据：{shock_text}")

        if competing_causes:
            basis.append(f"存在竞争性非感染病因：{'；'.join(competing_causes[:2])}")
        elif uncertainty_notes:
            basis.append(f"信息缺口/冲突：{'；'.join(uncertainty_notes[:2])}")
        return basis


class SOFATool:
    def evaluate(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        case_text = case_to_context(case_payload, max_field_chars=2000)
        if not clean_text(case_text):
            case_text = json.dumps(normalize_case_payload(case_payload), ensure_ascii=False)
        explicit_sofa = _extract_numeric(case_text, [r"SOFA评分[:：]?\s*([0-9]+(?:\.\d+)?)"])

        platelets = _extract_numeric(case_text, [r"(?:血小板计数|PLT)[^0-9]{0,20}([0-9]+(?:\.\d+)?)"])
        bilirubin_umol = _extract_numeric(
            case_text,
            [
                r"总胆红素(?:\(TBIL\))?[^0-9]{0,20}([0-9]+(?:\.\d+)?)",
                r"TBIL[^0-9]{0,20}([0-9]+(?:\.\d+)?)",
            ],
        )
        creatinine_raw = _extract_numeric(case_text, [r"(?:肌酐|CREA|Cr)[^0-9]{0,20}([0-9]+(?:\.\d+)?)"])
        lactate = _extract_numeric(case_text, [r"(?:乳酸|Lac)[^0-9]{0,20}([0-9]+(?:\.\d+)?)"])
        gcs = _extract_numeric(case_text, [r"GCS[^0-9]{0,10}([0-9]+(?:\.\d+)?)"])
        pao2 = _extract_numeric(case_text, [r"(?:PaO2|PO2|氧分压)[^0-9]{0,20}([0-9]+(?:\.\d+)?)"])
        fio2 = _extract_numeric(case_text, [r"(?:FiO2|吸氧浓度|氧浓度)[^0-9]{0,20}([0-9]+(?:\.\d+)?)"])

        bp_match = re.search(r"BP[:：]?\s*([0-9]+)\s*/\s*([0-9]+)\s*mmHg", case_text, flags=re.IGNORECASE)
        systolic = float(bp_match.group(1)) if bp_match else None
        diastolic = float(bp_match.group(2)) if bp_match else None
        map_value = None
        if systolic is not None and diastolic is not None:
            map_value = (systolic + 2 * diastolic) / 3.0
        vasopressor_used = any(token in clean_text(case_text) for token in ("肾上腺素", "去甲肾上腺素", "多巴酚丁胺", "多巴胺"))
        low_bp_text = any(token in clean_text(case_text) for token in ("低血压", "血压下降", "血压测不出"))
        shock_text = "休克" in clean_text(case_text)
        vasopressor_name = "去甲肾上腺素等血管活性药" if vasopressor_used else ""

        bilirubin_mg = bilirubin_umol / 17.1 if bilirubin_umol is not None else None
        creatinine_mg = None
        if creatinine_raw is not None:
            creatinine_mg = creatinine_raw / 88.4 if creatinine_raw > 20 else creatinine_raw
        if fio2 is not None and fio2 > 1.5:
            fio2 = fio2 / 100.0
        pf_ratio = None
        if pao2 is not None and fio2 is not None and fio2 > 0:
            pf_ratio = pao2 / fio2

        component_scores: Dict[str, Dict[str, Any]] = {
            "respiration": {
                "value": pf_ratio,
                "score": self._respiration_score(pf_ratio),
                "description": "PaO2/FiO2",
            },
            "coagulation": {
                "value": platelets,
                "score": self._coagulation_score(platelets),
                "description": "血小板",
            },
            "liver": {
                "value": bilirubin_umol,
                "score": self._liver_score(bilirubin_mg),
                "description": "总胆红素",
            },
            "cardiovascular": {
                "value": map_value,
                "score": self._cardiovascular_score(map_value, case_text),
                "description": "平均动脉压/血管活性药",
            },
            "cns": {
                "value": gcs,
                "score": self._cns_score(gcs),
                "description": "GCS",
            },
            "renal": {
                "value": creatinine_raw,
                "score": self._renal_score(creatinine_mg),
                "description": "肌酐",
            },
        }

        computable_scores = [item["score"] for item in component_scores.values() if item["score"] is not None]
        computed_total = sum(computable_scores) if computable_scores else None
        known_component_count = sum(1 for item in component_scores.values() if item["score"] is not None)

        key_indicators = {
            "sbp": systolic,
            "dbp": diastolic,
            "map": map_value,
            "lactate": lactate,
            "vasopressor_used": vasopressor_used,
            "vasopressor_name": vasopressor_name,
            "platelets": platelets,
            "bilirubin": bilirubin_umol,
            "creatinine": creatinine_raw,
            "gcs": gcs,
            "pao2": pao2,
            "fio2": fio2,
            "pf_ratio": pf_ratio,
            "low_bp_text": low_bp_text,
            "shock_text": shock_text,
        }

        lines = ["SOFA/治疗关键指标工具："]
        if explicit_sofa is not None:
            lines.append(f"- 病历中显式记录 SOFA 评分：{int(explicit_sofa)} 分。")
        if computed_total is not None:
            lines.append(
                f"- 根据可抽取指标可计算部分/完整 SOFA：{computed_total} 分（已覆盖 {known_component_count}/6 个器官系统）。"
            )
        else:
            lines.append("- 未能从当前文本中抽取到足够指标完成 SOFA 计算。")

        hemodynamic_parts = []
        if systolic is not None and diastolic is not None:
            hemodynamic_parts.append(f"血压 {int(systolic)}/{int(diastolic)} mmHg")
        if map_value is not None:
            hemodynamic_parts.append(f"MAP {map_value:.1f} mmHg")
        if lactate is not None:
            hemodynamic_parts.append(f"乳酸 {lactate:.2f} mmol/L")
        if vasopressor_used:
            hemodynamic_parts.append(f"已使用升压药：{vasopressor_name}")
        if low_bp_text:
            hemodynamic_parts.append("文本提及低血压")
        if shock_text:
            hemodynamic_parts.append("文本提及休克")
        if hemodynamic_parts:
            lines.append(f"- 血流动力学/灌注：{'；'.join(hemodynamic_parts)}。")

        respiratory_parts = []
        if pao2 is not None:
            respiratory_parts.append(f"PaO2 {pao2:.2f}")
        if fio2 is not None:
            respiratory_parts.append(f"FiO2 {fio2:.2f}")
        if pf_ratio is not None:
            respiratory_parts.append(f"PaO2/FiO2 {pf_ratio:.2f}")
        if respiratory_parts:
            lines.append(f"- 氧合/呼吸：{'；'.join(respiratory_parts)}。")

        organ_parts = []
        if creatinine_raw is not None:
            organ_parts.append(f"肌酐 {creatinine_raw:.2f}")
        if platelets is not None:
            organ_parts.append(f"血小板 {platelets:.2f} x10^9/L")
        if bilirubin_umol is not None:
            organ_parts.append(f"总胆红素 {bilirubin_umol:.2f} umol/L")
        if gcs is not None:
            organ_parts.append(f"GCS {gcs:.2f}")
        if organ_parts:
            lines.append(f"- 其他关键指标：{'；'.join(organ_parts)}。")

        missing = []
        if systolic is None or diastolic is None:
            missing.append("血压")
        if lactate is None:
            missing.append("乳酸")
        if pao2 is None or fio2 is None:
            missing.append("氧合(PaO2/FiO2)")
        if creatinine_raw is None:
            missing.append("肌酐")
        if platelets is None:
            missing.append("血小板")
        if bilirubin_umol is None:
            missing.append("总胆红素")
        if gcs is None:
            missing.append("GCS")
        if missing:
            lines.append(f"- 关键缺失：{'、'.join(missing)}。")

        return {
            "reported_total": explicit_sofa,
            "computed_total": computed_total,
            "component_scores": component_scores,
            "known_component_count": known_component_count,
            "key_indicators": key_indicators,
            "raw_values": {
                "platelets": platelets,
                "bilirubin": bilirubin_umol,
                "bilirubin_unit": "umol/L" if bilirubin_umol is not None else "",
                "creatinine": creatinine_raw,
                "creatinine_unit": "umol/L" if creatinine_raw is not None and creatinine_raw > 20 else "",
                "lactate": lactate,
                "gcs": gcs,
                "pao2": pao2,
                "fio2": fio2,
                "pf_ratio": pf_ratio,
                "sbp": systolic,
                "dbp": diastolic,
                "map": map_value,
                "vasopressor_used": vasopressor_used,
                "vasopressor_name": vasopressor_name,
                "low_bp_text": low_bp_text,
                "shock_text": shock_text,
            },
            "text": "\n".join(lines),
        }

    def _respiration_score(self, pf_ratio: Optional[float]) -> Optional[int]:
        if pf_ratio is None:
            return None
        if pf_ratio < 100:
            return 4
        if pf_ratio < 200:
            return 3
        if pf_ratio < 300:
            return 2
        if pf_ratio < 400:
            return 1
        return 0

    def _coagulation_score(self, platelets: Optional[float]) -> Optional[int]:
        if platelets is None:
            return None
        if platelets < 20:
            return 4
        if platelets < 50:
            return 3
        if platelets < 100:
            return 2
        if platelets < 150:
            return 1
        return 0

    def _liver_score(self, bilirubin_mg: Optional[float]) -> Optional[int]:
        if bilirubin_mg is None:
            return None
        if bilirubin_mg >= 12:
            return 4
        if bilirubin_mg >= 6:
            return 3
        if bilirubin_mg >= 2:
            return 2
        if bilirubin_mg >= 1.2:
            return 1
        return 0

    def _cardiovascular_score(self, map_value: Optional[float], case_text: str) -> Optional[int]:
        text = clean_text(case_text)
        if any(token in text for token in ("肾上腺素", "去甲肾上腺素", "多巴酚丁胺", "多巴胺")):
            return 2
        if map_value is None:
            return None
        if map_value < 70:
            return 1
        return 0

    def _cns_score(self, gcs: Optional[float]) -> Optional[int]:
        if gcs is None:
            return None
        if gcs < 6:
            return 4
        if gcs < 10:
            return 3
        if gcs < 13:
            return 2
        if gcs < 15:
            return 1
        return 0

    def _renal_score(self, creatinine_mg: Optional[float]) -> Optional[int]:
        if creatinine_mg is None:
            return None
        if creatinine_mg >= 5:
            return 4
        if creatinine_mg >= 3.5:
            return 3
        if creatinine_mg >= 2:
            return 2
        if creatinine_mg >= 1.2:
            return 1
        return 0


class PythonCodeTool:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.runtime_root.mkdir(parents=True, exist_ok=True)

    def execute_python(self, script: str, timeout: int = 20) -> Dict[str, Any]:
        task_id = f"code_exec_{uuid.uuid4().hex[:8]}"
        task_dir = self.runtime_root / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        script_path = task_dir / "tool_exec.py"
        script_path.write_text(script, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "success": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": clean_text(completed.stdout),
            "stderr": clean_text(completed.stderr),
            "script_path": str(script_path.resolve()),
        }

class LLMGeneratedPythonTool(PythonCodeTool):
    def __init__(self, runtime_root: Path, text_model: Any, max_new_tokens: int) -> None:
        super().__init__(runtime_root)
        self.text_model = text_model
        self.max_new_tokens = max_new_tokens

    def run_generated_probe(
        self,
        *,
        user_query: str,
        case_payload: Dict[str, Any],
        case_json_path: Optional[Path],
        candidate_dirs: Sequence[Path],
        timeout: int = 20,
        goal_hint: str = "",
    ) -> Dict[str, Any]:
        case_context = case_to_context(case_payload, max_field_chars=1600)
        case_path_text = str(case_json_path) if case_json_path else ""
        candidate_dir_lines = "\n".join(f"- {path}" for path in candidate_dirs[:4]) or "- 无候选影像目录"
        prompt = textwrap.dedent(
            f"""
            你要生成一个只读 Python 3 脚本，用于辅助分析当前病例。

            硬性限制：
            1. 只能读取病例 JSON 和候选影像目录中的文件。
            2. 不要联网，不要安装依赖，不要修改或删除任何项目文件。
            3. 只输出可执行 Python 代码，不要 Markdown，不要解释。
            4. 若要读取 DICOM，可尝试使用 pydicom；若失败必须优雅降级。
            5. 脚本必须通过 print 输出结果，输出尽量简洁。

            当前目标：
            - 用户问题：{clean_text(user_query) or '未提供'}
            - 代码分析目标：{clean_text(goal_hint) or '补充提取病例中的结构化线索、影像目录统计和可用 DICOM 信息'}

            可访问输入：
            - case_json_path = {case_path_text or '""'}
            - candidate_dirs:
            {candidate_dir_lines}

            病例摘要：
            {case_context}

            脚本至少完成：
            - 输出病例 JSON 的关键顶层字段和 meta 信息
            - 输出候选影像目录是否存在及 DICOM 文件数量
            - 尽量汇总感染相关实验室、生命体征或时间信息
            - 若能读取 DICOM，则汇总 modality / series_description / study_description 的去重结果
            """
        ).strip()

        raw_generation = self.text_model.generate(
            system_prompt="你是医疗代码执行助手，只能输出 Python 代码。",
            user_prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
        )
        generated_code = _extract_python_code(raw_generation)
        if not generated_code:
            return {
                "success": False,
                "returncode": None,
                "stdout": "",
                "stderr": "LLM 未返回可执行 Python 代码。",
                "script_path": "",
                "generated_code": "",
                "raw_generation": clean_text(raw_generation),
            }

        result = self.execute_python(generated_code, timeout=timeout)
        result["generated_code"] = generated_code
        result["raw_generation"] = clean_text(raw_generation)
        return result
