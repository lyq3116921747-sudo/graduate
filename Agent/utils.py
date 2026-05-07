from __future__ import annotations

import ast
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


RAW_TEXT_FIELDS = {
    "group_id",
    "EMPI",
    "住院号/门诊号",
    "流水号",
    "关联号",
    "唯一号",
    "姓名",
    "年龄",
    "住院年龄",
    "性别",
    "记录时间",
    "检查时间",
    "文件目录",
}

FOCUS_CONDITION_MAP = {
    "胆道感染": {
        "keywords": ("胆管炎", "胆总管炎", "胆总管结石", "胆囊炎", "右上腹痛", "胆道"),
        "web_query": "sepsis cholangitis source control guideline",
    },
    "肺部感染": {
        "keywords": ("肺炎", "肺部感染", "咳嗽", "痰", "呼吸衰竭", "两肺"),
        "web_query": "sepsis pneumonia management guideline",
    },
    "尿路感染": {
        "keywords": ("尿路感染", "尿频", "尿急", "泌尿系", "尿白细胞", "尿培养"),
        "web_query": "sepsis urinary tract infection management guideline",
    },
    "腹腔感染": {
        "keywords": ("腹膜炎", "腹腔感染", "腹痛", "腹腔积液", "肠梗阻"),
        "web_query": "sepsis intra abdominal infection management guideline",
    },
    "休克": {
        "keywords": ("休克", "低血压", "乳酸", "去甲肾上腺素", "灌注"),
        "web_query": "septic shock vasopressor fluid resuscitation guideline",
    },
}
FOCUS_CONDITION_EXPLICIT_KEYWORDS = {
    "胆道感染": ("胆道感染", "胆管炎", "胆囊炎", "胆总管炎", "胆道源性"),
    "肺部感染": ("肺部感染", "肺炎", "下呼吸道感染"),
    "尿路感染": ("尿路感染", "泌尿系感染", "肾盂肾炎"),
    "腹腔感染": ("腹腔感染", "腹膜炎", "腹腔脓肿"),
    "休克": ("休克", "感染性休克", "脓毒性休克"),
}

FOCUS_CONDITION_RAG_QUERY_MAP = {
    "胆道感染": "cholangitis biliary infection",
    "肺部感染": "pneumonia pulmonary infection",
    "尿路感染": "urinary tract infection urosepsis",
    "腹腔感染": "intra abdominal infection peritonitis",
    "休克": "septic shock hypotension lactate vasopressor",
}

RAG_EVIDENCE_TERM_MAP = {
    "发热": ("发热", "高热", "寒战", "体温38", "体温39"),
    "低血压": ("低血压", "血压低", "收缩压", "SBP", "MAP"),
    "休克": ("休克", "感染性休克", "脓毒性休克"),
    "乳酸升高": ("乳酸", "Lac", "lactate"),
    "呼吸衰竭": ("呼吸衰竭", "呼吸困难", "氧合", "PaO2", "FiO2"),
    "肾功能损伤": ("肌酐", "尿量减少", "少尿", "肾功能"),
    "肝功能损伤": ("胆红素", "转氨酶", "肝功能"),
    "凝血异常": ("血小板", "INR", "凝血", "D-二聚体"),
    "意识障碍": ("意识模糊", "嗜睡", "昏迷", "意识障碍"),
    "炎症指标升高": ("PCT", "CRP", "降钙素原", "白细胞", "中性粒"),
    "菌血症": ("血培养", "菌血症", "革兰阴性杆菌", "革兰阳性球菌"),
}

RAG_EVIDENCE_QUERY_MAP = {
    "发热": "fever",
    "低血压": "hypotension low blood pressure",
    "休克": "shock vasopressor",
    "乳酸升高": "lactate hyperlactatemia",
    "呼吸衰竭": "respiratory failure hypoxemia",
    "肾功能损伤": "acute kidney injury creatinine oliguria",
    "肝功能损伤": "bilirubin liver dysfunction",
    "凝血异常": "coagulopathy thrombocytopenia INR platelets",
    "意识障碍": "altered mental status delirium",
    "炎症指标升高": "procalcitonin CRP leukocytosis",
    "菌血症": "bacteremia bloodstream infection blood culture",
}

SUPPORTED_CASE_INPUT_MODES = {"clinical_assist", "strict_eval"}
SUPPORTED_LAB_SELECTION_MODES = {"default", "sepsis_priority"}
STRICT_EVAL_MASK_FIELDS = {
    "西医初步诊断",
    "中医初步诊断",
    "西医确定诊断",
    "西医诊断依据",
    "中医辨病辩证依据",
    "西医入院诊断",
    "中医入院诊断",
    "西医出院诊断",
    "中医出院诊断",
    "入院诊断",
    "出院诊断",
    "出院记录",
    "出院医嘱",
    "出院情况",
    "治疗建议",
}
DIAGNOSIS_RELATED_FIELDS = {
    "西医初步诊断",
    "中医初步诊断",
    "西医确定诊断",
    "西医诊断依据",
    "中医辨病辨证依据",
    "西医入院诊断",
    "中医入院诊断",
    "西医出院诊断",
    "中医出院诊断",
    "入院诊断",
    "出院诊断",
}
ALWAYS_HIDDEN_CASE_CONTEXT_FIELDS = {
    "出院医嘱",
    "出院情况",
}

LAB_NORMAL_TEXT_TOKENS = ("阴性", "未发现", "未检出", "清晰", "正常")
LAB_ABNORMAL_TEXT_TOKENS = ("阳性", "检出", "发现", "+")
SEPSIS_PRIORITY_LAB_FAMILY_ORDER = (
    "wbc",
    "neut_abs",
    "neut_pct",
    "crp",
    "pct",
    "culture",
    "ngs",
    "platelets",
    "bilirubin",
    "creatinine",
    "lactate",
    "pao2",
    "fio2",
)
SEPSIS_PRIORITY_LAB_TITLE = "Sepsis-3 相关实验室检查"
NON_BLOOD_MATRIX_TERMS = ("尿", "粪", "便", "痰", "分泌物", "脑脊液", "腹水", "胸水", "引流液")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate_text(text: str, max_chars: int) -> str:
    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 1)].rstrip() + "…"


def replace_bare_nan(raw_text: str) -> str:
    return re.sub(r"(?<![\w\"'])nan(?![\w\"'])", "None", raw_text)


def parse_python_like_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    raw = clean_text(value)
    if not raw:
        return []
    raw = replace_bare_nan(raw)
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return []
    if isinstance(parsed, list):
        return parsed
    return []


def first_non_empty(mapping: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        text = normalize_text(mapping.get(key, ""))
        if text:
            return text
    return ""


def normalize_case_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            normalized[key] = normalize_case_payload(value)
        elif isinstance(value, list):
            normalized[key] = value
        elif key in RAW_TEXT_FIELDS:
            normalized[key] = clean_text(value)
        else:
            normalized[key] = normalize_text(value)

    if not clean_text(normalized.get("年龄", "")):
        normalized["年龄"] = clean_text(payload.get("住院年龄", ""))
    if not clean_text(normalized.get("实验室检查", "")):
        normalized["实验室检查"] = normalize_text(payload.get("辅助检查结果", ""))
    if not clean_text(normalized.get("辅助检查结果", "")):
        normalized["辅助检查结果"] = normalize_text(payload.get("实验室检查", ""))

    imaging_entries = normalized.get("影像学检查", [])
    if isinstance(imaging_entries, dict):
        imaging_entries = [imaging_entries]
    elif not isinstance(imaging_entries, list):
        imaging_entries = parse_python_like_list(imaging_entries)

    normalized_entries: List[Dict[str, Any]] = []
    for entry in imaging_entries:
        if not isinstance(entry, dict):
            continue
        normalized_entry: Dict[str, Any] = {}
        for key, value in entry.items():
            if key in RAW_TEXT_FIELDS:
                normalized_entry[key] = clean_text(value)
            else:
                normalized_entry[key] = normalize_text(value)
        normalized_entries.append(normalized_entry)
    normalized["影像学检查"] = normalized_entries
    return normalized


def validate_case_input_mode(mode: str) -> str:
    normalized_mode = clean_text(mode) or "strict_eval"
    if normalized_mode not in SUPPORTED_CASE_INPUT_MODES:
        raise ValueError(f"不支持的病例输入模式: {mode}")
    return normalized_mode


def apply_case_input_mode(
    case_payload: Dict[str, Any],
    mode: str = "strict_eval",
    preserve_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    normalized = normalize_case_payload(case_payload)
    resolved_mode = validate_case_input_mode(mode)
    preserved = {clean_text(field) for field in (preserve_fields or []) if clean_text(field)}
    if resolved_mode == "clinical_assist":
        meta = dict(normalized.get("meta", {}) if isinstance(normalized.get("meta", {}), dict) else {})
        meta["case_input_mode"] = resolved_mode
        if preserved:
            meta["preserved_fields"] = sorted(preserved)
        normalized["meta"] = meta
        return normalized

    masked = dict(normalized)
    for field in STRICT_EVAL_MASK_FIELDS:
        if field in preserved:
            continue
        if field in masked:
            masked[field] = ""

    meta = dict(masked.get("meta", {}) if isinstance(masked.get("meta", {}), dict) else {})
    if "is_spesis" in meta:
        meta["is_spesis"] = ""
    meta["case_input_mode"] = resolved_mode
    meta["masked_fields"] = sorted(field for field in STRICT_EVAL_MASK_FIELDS if field not in preserved)
    if preserved:
        meta["preserved_fields"] = sorted(field for field in preserved if field in STRICT_EVAL_MASK_FIELDS)
    masked["meta"] = meta
    return masked


def build_case_payload_from_csv_row(
    row: Dict[str, Any],
    mode: str = "strict_eval",
    preserve_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    normalized_row = normalize_case_payload(dict(row))
    imaging_entries: List[Dict[str, Any]] = []
    imaging_text = normalize_text(normalized_row.get("检查结果", ""))
    if imaging_text and any(token in imaging_text for token in ("CT", "MRI", "超声", "彩超", "胸片", "影像", "放射")):
        imaging_entries.append(
            {
                "检查项目": "辅助检查",
                "检查时间": clean_text(normalized_row.get("记录时间", "")),
                "影像所见": imaging_text,
                "检查结论": "",
                "文件目录": "",
            }
        )

    payload = {
        "关联号": clean_text(normalized_row.get("group_id", "")),
        "唯一号": clean_text(normalized_row.get("EMPI", "")),
        "住院号/门诊号": clean_text(normalized_row.get("门诊/住院号", "")),
        "姓名": clean_text(normalized_row.get("姓名", "")),
        "年龄": clean_text(normalized_row.get("住院年龄", "")) or clean_text(normalized_row.get("年龄", "")),
        "性别": clean_text(normalized_row.get("性别", "")),
        "记录时间": clean_text(normalized_row.get("记录时间", "")),
        "主诉": normalize_text(normalized_row.get("主诉", "")),
        "现病史": normalize_text(normalized_row.get("现病史", "")),
        "既往史": normalize_text(normalized_row.get("既往史", "")),
        "个人史": normalize_text(normalized_row.get("个人史", "")),
        "过敏史": normalize_text(normalized_row.get("过敏史", "")),
        "体格检查": normalize_text(normalized_row.get("体格检查", "")),
        "专科检查": normalize_text(normalized_row.get("专科检查", "")),
        "实验室检查": normalize_text(normalized_row.get("辅助检查结果", "")),
        "辅助检查结果": normalize_text(normalized_row.get("辅助检查结果", "")),
        "检查结果": imaging_text,
        "西医初步诊断": normalize_text(normalized_row.get("西医初步诊断", "")),
        "西医确定诊断": normalize_text(normalized_row.get("西医确定诊断", "")),
        "西医诊断依据": normalize_text(normalized_row.get("西医诊断依据", "")),
        "中医辨病辩证依据": normalize_text(normalized_row.get("中医辨病辩证依据", "")),
        "入院情况": normalize_text(normalized_row.get("入院情况", "")),
        "诊疗过程": normalize_text(normalized_row.get("诊疗过程", "")),
        "西医入院诊断": normalize_text(normalized_row.get("西医入院诊断", "")),
        "中医入院诊断": normalize_text(normalized_row.get("中医入院诊断", "")),
        "西医出院诊断": normalize_text(normalized_row.get("西医出院诊断", "")),
        "中医出院诊断": normalize_text(normalized_row.get("中医出院诊断", "")),
        "出院情况": normalize_text(normalized_row.get("出院情况", "")),
        "出院医嘱": normalize_text(normalized_row.get("出院医嘱", "")),
        "需要完善的检查": normalize_text(normalized_row.get("需要完善的检查", "")),
        "治疗建议": normalize_text(normalized_row.get("治疗建议", "")),
        "影像学检查": imaging_entries,
        "meta": {
            "task": clean_text(normalized_row.get("task", "")),
            "source_file": clean_text(normalized_row.get("source_file", "")),
            "source_path": clean_text(normalized_row.get("source_path", "")),
            "row_index": clean_text(normalized_row.get("row_index", "")),
            "is_spesis": clean_text(normalized_row.get("is_spesis", "")),
        },
    }
    return apply_case_input_mode(payload, mode=mode, preserve_fields=preserve_fields)


def load_case_payload(
    case_json_path: Path,
    mode: str = "strict_eval",
    preserve_fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    payload = json.loads(Path(case_json_path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("病例 JSON 顶层必须是对象。")
    return apply_case_input_mode(payload, mode=mode, preserve_fields=preserve_fields)


def summarize_imaging_entries(entries: Sequence[Dict[str, Any]], max_entries: int = 3, max_field_chars: int = 500) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(entries[:max_entries], start=1):
        project = clean_text(entry.get("检查项目", "")) or "未注明项目"
        date_text = clean_text(entry.get("检查时间", ""))
        findings = truncate_text(entry.get("影像所见", ""), max_field_chars)
        conclusion = truncate_text(entry.get("检查结论", ""), max_field_chars)
        has_dcm = "是" if clean_text(entry.get("文件目录", "")) else "否"
        lines.append(f"影像学检查{idx}：项目={project}；时间={date_text or '未提供'}；存在DICOM目录={has_dcm}")
        if findings:
            lines.append(f"影像所见{idx}：{findings}")
        if conclusion:
            lines.append(f"检查结论{idx}：{conclusion}")
    return "\n".join(lines).strip()


def _get_lab_entries(case_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_entries = case_payload.get("实验室检查", [])
    if isinstance(raw_entries, dict):
        raw_entries = [raw_entries]
    elif not isinstance(raw_entries, list):
        raw_entries = parse_python_like_list(raw_entries)
    return [entry for entry in raw_entries if isinstance(entry, dict)]


def _looks_like_serialized_lab_list(value: Any) -> bool:
    text = clean_text(value)
    return bool(text.startswith("[") and "检验指标名" in text and "检验指标值" in text)


def _normalize_lab_scalar(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace("＜", "<").replace("＞", ">")
    text = text.replace("≤", "<=").replace("≦", "<=")
    text = text.replace("≥", ">=").replace("≧", ">=")
    text = text.replace("－", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_lab_selection_mode(mode: str) -> str:
    normalized_mode = clean_text(mode) or "default"
    if normalized_mode not in SUPPORTED_LAB_SELECTION_MODES:
        raise ValueError(f"不支持的实验室摘要模式: {mode}")
    return normalized_mode


def _lab_text_key(value: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", clean_text(value).lower())


def _looks_like_non_blood_matrix(project: str, metric_name: str) -> bool:
    haystack = f"{project} {metric_name}"
    return any(term in haystack for term in NON_BLOOD_MATRIX_TERMS)


def _classify_priority_lab_family(project: str, metric_name: str, value_text: str = "") -> Optional[str]:
    metric_key = _lab_text_key(metric_name)
    project_key = _lab_text_key(project)
    value_key = _lab_text_key(value_text)
    combined = f"{project_key} {metric_key} {value_key}"

    if ("ngs" in combined or "宏基因组" in combined or "高通量" in combined) and ("检出" in combined or "阳性" in combined or "发现" in combined or "病原" in combined or "菌" in combined):
        return "ngs"
    if any(token in combined for token in ("培养", "病原", "念珠菌", "埃希菌", "克雷伯", "假单胞菌", "肠球菌", "葡萄球菌", "链球菌", "不动杆菌")):
        return "culture"
    if ("降钙素原" in metric_name or metric_key == "pct" or "降钙素原" in project):
        return "pct"
    if "c反应蛋白" in metric_name.lower() or metric_key == "crp" or "超敏crp" in combined:
        return "crp"
    if not _looks_like_non_blood_matrix(project, metric_name):
        if metric_key in {"白细胞计数", "白细胞", "wbc", "leukocytecount"}:
            return "wbc"
        if metric_key in {"中性粒细胞", "中性粒", "neut", "neu"}:
            return "neut_abs"
        if metric_key in {"中性粒细胞百分比", "中性粒百分比", "neutpct", "neu%"}:
            return "neut_pct"
    if "血小板计数" in metric_name or metric_key == "plt":
        return "platelets"
    if metric_key in {"总胆红素", "tbil"}:
        return "bilirubin"
    if metric_key == "胆红素" and "尿" not in project and "尿" not in metric_name:
        return "bilirubin"
    if metric_key in {"肌酐", "crea", "cr"} and "尿" not in project and "尿" not in metric_name:
        return "creatinine"
    if "乳酸脱氢酶" not in metric_name and (metric_key == "乳酸" or metric_key == "lac" or "lactate" in metric_key):
        return "lactate"
    if metric_key in {"氧分压", "pao2", "po2"}:
        return "pao2"
    if metric_key in {"吸氧浓度", "fio2", "吸入氧浓度", "氧浓度"}:
        return "fio2"
    return None


def _lab_family_key(project: str, metric_name: str, value_text: str = "") -> Tuple[str, str]:
    priority_family = _classify_priority_lab_family(project, metric_name, value_text)
    if priority_family:
        return ("priority", priority_family)

    metric_key = _lab_text_key(metric_name)
    project_key = _lab_text_key(project)
    if metric_key in {"白细胞", "红细胞", "胆红素", "蛋白质", "葡萄糖"} and project_key:
        return ("generic", f"{project_key}:{metric_key}")
    return ("generic", metric_key or project_key or "unknown")


def _lab_recency_key(row: Dict[str, Any]) -> Tuple[str, int]:
    return (clean_text(row.get("report_date", "")), int(row.get("index", -1)))


def _is_lab_row_abnormal(row: Dict[str, Any]) -> bool:
    return bool(clean_text(row.get("abnormal_flag", "")))


def _build_lab_rows(case_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, entry in enumerate(_get_lab_entries(case_payload)):
        metric_name = clean_text(entry.get("检验指标名", ""))
        value_text = clean_text(entry.get("检验指标值", ""))
        if not metric_name or not value_text:
            continue
        project = clean_text(entry.get("检验项目", ""))
        reference_text = clean_text(entry.get("参考范围", ""))
        priority_family = _classify_priority_lab_family(project, metric_name, value_text)
        family_scope, family_name = _lab_family_key(project, metric_name, value_text)
        rows.append(
            {
                "index": index,
                "report_date": clean_text(entry.get("报告日期", "")),
                "project": project,
                "metric_name": metric_name,
                "value_text": _normalize_lab_scalar(value_text),
                "unit": clean_text(entry.get("单位", "")),
                "reference_text": _compact_lab_reference_text(reference_text),
                "abnormal_flag": classify_lab_abnormality(value_text, reference_text),
                "priority_family": priority_family,
                "family_scope": family_scope,
                "family_name": family_name,
                "exact_key": (
                    clean_text(entry.get("报告日期", "")),
                    project,
                    metric_name,
                    _normalize_lab_scalar(value_text),
                ),
            }
        )
    return rows


def _format_lab_summary_rows(rows: Sequence[Dict[str, Any]]) -> str:
    flag_map = {
        "high": "↑",
        "low": "↓",
        "positive": "异常",
        "abnormal": "异常",
    }
    lines: List[str] = []
    for row in rows:
        prefix_parts = [part for part in (row["report_date"], row["project"]) if part]
        prefix = " / ".join(prefix_parts)
        value_with_unit = f"{row['value_text']} {row['unit']}".strip()
        abnormal_flag = clean_text(row.get("abnormal_flag", ""))
        reference_text = clean_text(row.get("reference_text", ""))
        suffix_parts: List[str] = []
        if reference_text:
            suffix_parts.append(f"参考 {reference_text}")
        if abnormal_flag:
            suffix_parts.append(flag_map.get(abnormal_flag, "异常"))
        suffix = f" ({'，'.join(suffix_parts)})" if suffix_parts else ""
        line = f"{row['metric_name']}: {value_with_unit}{suffix}".strip()
        if prefix:
            line = f"{prefix} / {line}"
        lines.append(line)
    return "\n".join(lines).strip()


def _parse_lab_number(text: Any) -> Optional[float]:
    normalized = _normalize_lab_scalar(text).replace(",", "")
    if not normalized:
        return None
    sci_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*[xX×*]\s*10\^?\s*([+-]?\d+)", normalized)
    if sci_match:
        try:
            return float(sci_match.group(1)) * (10 ** int(sci_match.group(2)))
        except Exception:
            return None
    number_match = re.search(r"([+-]?\d+(?:\.\d+)?)", normalized)
    if number_match:
        try:
            return float(number_match.group(1))
        except Exception:
            return None
    return None


def _parse_lab_range(reference_text: str) -> Optional[Tuple[float, float]]:
    first_line = _normalize_lab_scalar(reference_text).splitlines()[0] if _normalize_lab_scalar(reference_text) else ""
    if not first_line:
        return None
    range_match = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?(?:\s*[xX×*]\s*10\^?\s*[+-]?\d+)?)\s*-\s*([+-]?\d+(?:\.\d+)?(?:\s*[xX×*]\s*10\^?\s*[+-]?\d+)?)",
        first_line,
    )
    if not range_match:
        return None
    lower = _parse_lab_number(range_match.group(1))
    upper = _parse_lab_number(range_match.group(2))
    if lower is None or upper is None:
        return None
    if lower <= upper:
        return lower, upper
    return upper, lower


def _parse_lab_threshold(reference_text: str) -> Optional[Tuple[str, float]]:
    first_line = _normalize_lab_scalar(reference_text).splitlines()[0] if _normalize_lab_scalar(reference_text) else ""
    if not first_line:
        return None
    threshold_match = re.match(
        r"^\s*(<=|>=|<|>)\s*([+-]?\d+(?:\.\d+)?(?:\s*[xX×*]\s*10\^?\s*[+-]?\d+)?)",
        first_line,
    )
    if not threshold_match:
        return None
    threshold_value = _parse_lab_number(threshold_match.group(2))
    if threshold_value is None:
        return None
    return threshold_match.group(1), threshold_value


def _compact_lab_reference_text(reference_text: str, max_chars: int = 64) -> str:
    normalized = _normalize_lab_scalar(reference_text)
    if not normalized:
        return ""
    first_line = normalized.splitlines()[0].strip() or normalized
    if len(first_line) <= max_chars:
        return first_line
    return first_line[: max_chars - 1].rstrip() + "…"


def _matches_normal_qualitative_value(value_text: str, reference_text: str) -> Optional[bool]:
    value = _normalize_lab_scalar(value_text)
    reference = _normalize_lab_scalar(reference_text)
    if not value or not reference:
        return None

    normalized_value = re.sub(r"\s+", "", value)
    normalized_reference = re.sub(r"\s+", "", reference)
    if normalized_value == normalized_reference:
        return True
    if normalized_reference in normalized_value or normalized_value in normalized_reference:
        return True

    reference_has_normal = any(token in reference for token in LAB_NORMAL_TEXT_TOKENS)
    reference_has_abnormal = any(token in reference for token in LAB_ABNORMAL_TEXT_TOKENS if token != "+")
    if reference_has_normal and reference_has_abnormal:
        return None

    if reference_has_normal:
        return any(token in value for token in LAB_NORMAL_TEXT_TOKENS)
    if reference_has_abnormal:
        return any(token in value for token in LAB_ABNORMAL_TEXT_TOKENS if token != "+")

    numeric_reference = _parse_lab_number(reference)
    numeric_value = _parse_lab_number(value)
    if numeric_reference is not None and numeric_value is not None and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", normalized_reference):
        return math.isclose(numeric_reference, numeric_value, rel_tol=0.0, abs_tol=1e-9)
    return None


def classify_lab_abnormality(value_text: Any, reference_text: Any) -> Optional[str]:
    value = _normalize_lab_scalar(value_text)
    reference = _normalize_lab_scalar(reference_text)
    if not value or not reference:
        return None

    numeric_value = _parse_lab_number(value)
    numeric_range = _parse_lab_range(reference)
    if numeric_value is not None and numeric_range is not None:
        lower, upper = numeric_range
        if numeric_value < lower:
            return "low"
        if numeric_value > upper:
            return "high"
        return None

    threshold = _parse_lab_threshold(reference)
    if numeric_value is not None and threshold is not None:
        comparator, boundary = threshold
        threshold_is_positive = "阳性" in reference or "positive" in reference.lower()
        if comparator == "<":
            is_normal = numeric_value < boundary
            abnormal_flag = "high"
        elif comparator == "<=":
            is_normal = numeric_value <= boundary
            abnormal_flag = "high"
        elif comparator == ">":
            is_normal = numeric_value > boundary
            abnormal_flag = "low"
        else:
            is_normal = numeric_value >= boundary
            abnormal_flag = "low"

        if threshold_is_positive:
            return "positive" if is_normal else None
        return None if is_normal else abnormal_flag

    is_normal_qualitative = _matches_normal_qualitative_value(value, reference)
    if is_normal_qualitative is True:
        return None
    if is_normal_qualitative is False:
        return "positive" if "+" in value or "阳性" in value else "abnormal"
    return None


def summarize_abnormal_lab_results(
    case_payload: Dict[str, Any],
    *,
    max_items: int = 60,
) -> str:
    rows = [row for row in _build_lab_rows(case_payload) if _is_lab_row_abnormal(row)]
    rows.sort(key=_lab_recency_key, reverse=True)

    selected_rows: List[Dict[str, Any]] = []
    seen_keys = set()
    for row in rows:
        dedupe_key = (row["project"], row["metric_name"])
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        selected_rows.append(row)
        if len(selected_rows) >= max_items:
            break
    return _format_lab_summary_rows(selected_rows)


def summarize_sepsis_priority_lab_results(
    case_payload: Dict[str, Any],
    *,
    max_items: int = 40,
) -> str:
    rows = _build_lab_rows(case_payload)
    rows.sort(key=_lab_recency_key, reverse=True)

    selected_rows: List[Dict[str, Any]] = []
    selected_exact_keys = set()
    used_scalar_families = set()

    def add_row(row: Dict[str, Any]) -> bool:
        exact_key = row["exact_key"]
        if exact_key in selected_exact_keys:
            return False
        if row["priority_family"] not in {"culture", "ngs"}:
            family_key = (row["family_scope"], row["family_name"])
            if family_key in used_scalar_families:
                return False
            used_scalar_families.add(family_key)
        selected_exact_keys.add(exact_key)
        selected_rows.append(row)
        return True

    for family in SEPSIS_PRIORITY_LAB_FAMILY_ORDER:
        if family in {"culture", "ngs"}:
            family_rows = [row for row in rows if row["priority_family"] == family]
            seen_family_rows = set()
            for row in family_rows:
                if len(selected_rows) >= max_items:
                    break
                if row["exact_key"] in seen_family_rows:
                    continue
                seen_family_rows.add(row["exact_key"])
                add_row(row)
            continue

        family_rows = [row for row in rows if row["priority_family"] == family]
        if not family_rows:
            continue
        best_row = sorted(
            family_rows,
            key=lambda item: (_is_lab_row_abnormal(item), _lab_recency_key(item)),
            reverse=True,
        )[0]
        add_row(best_row)
        if len(selected_rows) >= max_items:
            break

    if len(selected_rows) < max_items:
        for row in rows:
            if len(selected_rows) >= max_items:
                break
            if row["priority_family"]:
                continue
            if not _is_lab_row_abnormal(row):
                continue
            add_row(row)

    if len(selected_rows) < max_items:
        for row in rows:
            if len(selected_rows) >= max_items:
                break
            if row["priority_family"]:
                continue
            add_row(row)

    return _format_lab_summary_rows(selected_rows)


def get_case_auxiliary_exam_text(
    case_payload: Dict[str, Any],
    *,
    max_lab_items: int = 60,
    lab_selection: str = "default",
) -> str:
    parts: List[str] = []
    selection_mode = validate_lab_selection_mode(lab_selection)
    if selection_mode == "sepsis_priority":
        lab_summary = summarize_sepsis_priority_lab_results(case_payload, max_items=max_lab_items)
        if lab_summary:
            parts.append(f"{SEPSIS_PRIORITY_LAB_TITLE}：\n" + lab_summary)
    else:
        abnormal_lab_summary = summarize_abnormal_lab_results(case_payload, max_items=max_lab_items)
        if abnormal_lab_summary:
            parts.append("异常实验室检查：\n" + abnormal_lab_summary)

    auxiliary_text = normalize_text(case_payload.get("辅助检查结果", ""))
    if auxiliary_text and not _looks_like_serialized_lab_list(auxiliary_text):
        parts.append(auxiliary_text)

    check_text = normalize_text(case_payload.get("检查结果", ""))
    if check_text:
        parts.append(check_text)
    return "\n".join(part for part in parts if part).strip()


def _case_context_excluded_fields(task_mode: Optional[str]) -> set[str]:
    normalized_task_mode = clean_text(task_mode).lower()
    if normalized_task_mode in {"diagnosis", "treatment", "diagnosis_and_treatment"}:
        return set(DIAGNOSIS_RELATED_FIELDS) | set(ALWAYS_HIDDEN_CASE_CONTEXT_FIELDS)
    return set()


def case_to_context(
    case_payload: Dict[str, Any],
    max_field_chars: int = 1200,
    lab_selection: str = "default",
    task_mode: Optional[str] = None,
) -> str:
    payload = normalize_case_payload(case_payload)
    excluded_fields = _case_context_excluded_fields(task_mode)
    field_order = [
        "性别",
        "年龄",
        "住院号/门诊号",
        "唯一号",
        "记录时间",
        "主诉",
        "现病史",
        "既往史",
        "个人史",
        "过敏史",
        "体格检查",
        "专科检查",
        "实验室检查",
        "辅助检查结果",
        "检查结果",
        "西医初步诊断",
        "西医确定诊断",
        "西医诊断依据",
        "西医入院诊断",
        "西医出院诊断",
        "入院情况",
        "诊疗过程",
        "出院情况",
        "出院医嘱",
        "治疗建议",
    ]
    lines: List[str] = []
    auxiliary_exam_text = get_case_auxiliary_exam_text(
        payload,
        max_lab_items=40,
        lab_selection=lab_selection,
    )
    for field in field_order:
        if field in excluded_fields:
            continue
        if field in {"实验室检查", "辅助检查结果", "检查结果"}:
            continue
        value = payload.get(field, "")
        text = clean_text(value) if field in RAW_TEXT_FIELDS else normalize_text(value)
        if not text:
            continue
        lines.append(f"{field}：{truncate_text(text, max_field_chars)}")

    if auxiliary_exam_text:
        lines.append(f"辅助检查结果：{truncate_text(auxiliary_exam_text, max_field_chars)}")

    imaging_summary = summarize_imaging_entries(payload.get("影像学检查", []), max_field_chars=max_field_chars)
    if imaging_summary:
        lines.append(imaging_summary)
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    normalized = clean_text(text)
    if not normalized:
        return None
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[a-zA-Z0-9_]*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()

    start = normalized.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for idx in range(start, len(normalized)):
        char = normalized[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None

    candidate = normalized[start : end + 1]
    for parser in (json.loads, ast.literal_eval):
        try:
            payload = parser(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def flatten_texts(items: Iterable[Any]) -> str:
    parts = []
    for item in items:
        text = normalize_text(item)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def extract_diagnosis_candidates(case_payload: Dict[str, Any]) -> List[str]:
    diagnosis_text = flatten_texts(
        [
            case_payload.get("西医确定诊断", ""),
            case_payload.get("西医出院诊断", ""),
            case_payload.get("西医入院诊断", ""),
            case_payload.get("西医初步诊断", ""),
            case_payload.get("出院诊断", ""),
            case_payload.get("入院诊断", ""),
        ]
    )
    if not diagnosis_text:
        return []
    parts = re.split(r"[\n；;]+", diagnosis_text)
    candidates: List[str] = []
    for part in parts:
        text = re.sub(r"^\d+[.、]\s*", "", normalize_text(part))
        if text and text not in candidates:
            candidates.append(text)
    return candidates


def extract_focus_conditions(case_payload: Dict[str, Any]) -> List[str]:
    payload = normalize_case_payload(case_payload)
    haystack = "\n".join(
        part
        for part in [
            normalize_text(payload.get("主诉", "")),
            normalize_text(payload.get("现病史", "")),
            normalize_text(payload.get("体格检查", "")),
            normalize_text(payload.get("专科检查", "")),
            get_case_auxiliary_exam_text(payload, max_lab_items=20),
        ]
        if clean_text(part)
    )
    matches: List[str] = []
    for label, keywords in FOCUS_CONDITION_EXPLICIT_KEYWORDS.items():
        if any(keyword in haystack for keyword in keywords):
            matches.append(label)
    return matches


def extract_case_evidence_terms(case_payload: Dict[str, Any], max_terms: int = 6) -> List[str]:
    haystack = case_to_context(case_payload, max_field_chars=1800)
    matches: List[str] = []
    for label, keywords in RAG_EVIDENCE_TERM_MAP.items():
        if any(keyword in haystack for keyword in keywords):
            matches.append(label)
        if len(matches) >= max_terms:
            break
    return matches


def safe_path_text(path: Path) -> str:
    return str(Path(path).expanduser().resolve())
