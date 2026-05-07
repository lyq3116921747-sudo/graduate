from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Agent.tools import Sepsis3Judge, SOFATool
from Agent.utils import build_case_payload_from_csv_row, clean_text, load_case_payload


DEFAULT_CSV_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_CSV_NON_SEPSIS = PROJECT_ROOT / "miniTest" / "csv" / "sepsis3_extract_non_sepsis_2000_sample_200_sample_40.csv"
DEFAULT_JSON_MANIFEST = PROJECT_ROOT / "miniTest" / "json_sample_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "miniTest" / "sofa_baseline"
DEFAULT_RESULTS_CSV = "csv_results.csv"
DEFAULT_RESULTS_JSON = "json_results.csv"
DEFAULT_SUMMARY_JSON = "summary.json"
DEFAULT_SUMMARY_CSV = "summary.csv"
DEFAULT_RUN_LOG = "run_log.jsonl"
MODEL_PROFILE = "sofa_baseline/rule_regex"

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

TOP_LEVEL_TEXT_KEYS = (
    "主诉",
    "现病史",
    "既往史",
    "体格检查",
    "专科检查",
    "实验室检查",
    "辅助检查结果",
    "入院情况",
    "诊疗过程",
    "西医入院诊断",
    "西医出院诊断",
    "出院情况",
    "出院医嘱",
    "出院记录",
)
SENTENCE_SPLIT_RE = re.compile(r"[\n。；;!?！？]+")
TEMPERATURE_RE = re.compile(r"(?:Tmax|体温|T)\s*[:：=]?\s*(3[8-9](?:\.\d+)?|4\d(?:\.\d+)?)", flags=re.IGNORECASE)
VALUE_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

DIRECT_INFECTION_TERMS = (
    "肺部感染",
    "肺炎",
    "腹腔感染",
    "腹膜炎",
    "脓肿",
    "胆道感染",
    "胆管炎",
    "胆囊炎",
    "泌尿系感染",
    "尿路感染",
    "肾盂肾炎",
    "菌血症",
    "血流感染",
    "败血症",
    "切口感染",
    "导管相关感染",
    "宫腔感染",
    "盆腔感染",
    "真菌感染",
)
GENERIC_INFECTION_TERMS = ("感染",)
INFECTION_EXCLUDE_TERMS = (
    "预防感染",
    "防治感染",
    "术后常规抗感染",
    "围手术期常规抗感染",
    "抗感染治疗后好转",
    "感染风险",
    "感染指标",
)
POSITIVE_CULTURE_TERMS = ("培养出", "培养提示", "检出", "阳性", "生长")
NEGATIVE_CULTURE_TERMS = ("未检出", "阴性", "无菌生长")
PATHOGEN_TERMS = (
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
ANTIMICROBIAL_TERMS = (
    "抗感染",
    "抗菌",
    "抗真菌",
    "升级抗生素",
    "美罗培南",
    "亚胺培南",
    "万古霉素",
    "哌拉西林",
    "他唑巴坦",
    "舒普深",
    "伏立康唑",
    "头孢",
    "邦达",
)
FEVER_TERMS = ("发热", "高热", "寒战", "体温升高", "热峰")
UNCERTAIN_INFECTION_TERMS = (
    "感染待排",
    "疑似感染",
    "考虑感染",
    "不除外感染",
    "感染可能",
    "感染源不明",
    "感染证据不足",
    "培养阴性",
    "无菌生长",
)
ABSENT_INFECTION_TERMS = (
    "排除感染",
    "不考虑感染",
    "无明确感染",
    "未见明确感染",
    "无感染征象",
    "感染依据不足",
)
SOURCE_KEYWORDS = (
    ("肺部", ("肺部", "肺炎", "呼吸道", "双肺", "胸部CT提示炎症")),
    ("腹腔", ("腹腔", "腹膜", "腹部感染", "腹膜炎", "腹腔脓肿")),
    ("泌尿系", ("泌尿系", "尿路", "肾盂肾炎")),
    ("胆道", ("胆道", "胆管", "胆囊")),
    ("血流", ("菌血症", "败血症", "血流感染")),
    ("导管/伤口", ("导管相关", "切口感染", "伤口感染", "褥疮感染")),
    ("盆腔", ("盆腔感染", "宫腔感染")),
)
SHOCK_TERMS = ("脓毒性休克", "休克")
TUMOR_MIXED_TERMS = ("肿瘤热", "肿瘤坏死", "化疗后", "放疗后", "肿瘤进展")
COMPETING_TERMS = (
    "心源性休克",
    "失血性休克",
    "出血性休克",
    "肺栓塞",
    "急性冠脉综合征",
    "心肌梗死",
    "肿瘤热",
    "术后应激",
    "药物热",
)
CHRONIC_CONFOUNDER_TERMS = (
    "慢性肾功能不全",
    "维持性血液透析",
    "肝硬化",
    "慢性肝病",
    "痴呆",
    "长期意识障碍",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic SOFA + rule-regex sepsis baseline on miniTest.")
    parser.add_argument("--csv-sepsis", default=str(DEFAULT_CSV_SEPSIS))
    parser.add_argument("--csv-non-sepsis", default=str(DEFAULT_CSV_NON_SEPSIS))
    parser.add_argument("--json-manifest", default=str(DEFAULT_JSON_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["clinical_assist", "strict_eval"], default="strict_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except Exception:
            return None
    match = VALUE_RE.search(clean_text(value).replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


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
        "model_profile": MODEL_PROFILE,
    }


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


def _iter_case_texts(case_payload: Dict[str, Any]) -> Iterable[str]:
    for key in TOP_LEVEL_TEXT_KEYS:
        value = case_payload.get(key, "")
        if isinstance(value, str):
            text = clean_text(value)
            if text:
                yield text
        elif isinstance(value, dict):
            yield json.dumps(value, ensure_ascii=False)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    text = clean_text(item)
                    if text:
                        yield text
                elif isinstance(item, dict):
                    yield json.dumps(item, ensure_ascii=False)

    raw_progress = case_payload.get("病程记录", [])
    if isinstance(raw_progress, list):
        for entry in raw_progress:
            if not isinstance(entry, dict):
                continue
            for field in ("单次记录", "记录名称"):
                text = clean_text(entry.get(field, ""))
                if text:
                    yield text


def _iter_case_sentences(case_payload: Dict[str, Any]) -> List[str]:
    sentences: List[str] = []
    for text in _iter_case_texts(case_payload):
        for fragment in SENTENCE_SPLIT_RE.split(text):
            sentence = clean_text(fragment)
            if sentence:
                sentences.append(sentence)
    return sentences


def _add_unique(items: List[str], value: str, *, limit: Optional[int] = None) -> None:
    text = clean_text(value)
    if not text or text in items:
        return
    if limit is not None and len(items) >= limit:
        return
    items.append(text)


def _sentence_has_fever(sentence: str) -> bool:
    normalized = clean_text(sentence)
    if re.search(r"(?:无|未见|未|否认).{0,3}(?:发热|高热|寒战)", normalized):
        return False
    if any(term in normalized for term in FEVER_TERMS):
        return True
    match = TEMPERATURE_RE.search(normalized)
    return bool(match and float(match.group(1)) >= 38.0)


def _sentence_has_antimicrobial(sentence: str) -> bool:
    normalized = clean_text(sentence)
    return any(term in normalized for term in ANTIMICROBIAL_TERMS)


def _sentence_has_positive_microbiology(sentence: str) -> bool:
    normalized = clean_text(sentence)
    if not normalized or any(term in normalized for term in NEGATIVE_CULTURE_TERMS):
        return False
    return ("培养" in normalized and any(term in normalized for term in POSITIVE_CULTURE_TERMS)) or any(
        pathogen in normalized for pathogen in PATHOGEN_TERMS
    )


def _sentence_has_absent_infection(sentence: str) -> bool:
    normalized = clean_text(sentence)
    return any(term in normalized for term in ABSENT_INFECTION_TERMS)


def _sentence_has_uncertain_infection(sentence: str) -> bool:
    normalized = clean_text(sentence)
    return any(term in normalized for term in UNCERTAIN_INFECTION_TERMS)


def _sentence_has_direct_infection(sentence: str) -> bool:
    normalized = clean_text(sentence)
    if not normalized or _sentence_has_absent_infection(normalized) or _sentence_has_uncertain_infection(normalized):
        return False
    if any(term in normalized for term in DIRECT_INFECTION_TERMS):
        return True
    if "感染" not in normalized:
        return False
    return not any(term in normalized for term in INFECTION_EXCLUDE_TERMS + ANTIMICROBIAL_TERMS)


def _sentence_has_inflammatory_signal(sentence: str) -> bool:
    normalized = clean_text(sentence)
    if not normalized:
        return False
    if "降钙素原" in normalized or re.search(r"\bPCT\b", normalized, flags=re.IGNORECASE):
        value = _safe_float(normalized)
        if "↑" in normalized or "升高" in normalized or (value is not None and value >= 0.5):
            return True
    if "C反应蛋白" in normalized or re.search(r"\bCRP\b", normalized, flags=re.IGNORECASE):
        value = _safe_float(normalized)
        if "↑" in normalized or "升高" in normalized or (value is not None and value >= 10):
            return True
    if "白细胞计数" in normalized or re.search(r"\bWBC\b", normalized, flags=re.IGNORECASE):
        value = _safe_float(normalized)
        if "↑" in normalized or "升高" in normalized:
            return True
        if value is not None and (value >= 12 or value <= 4):
            return True
    if "中性粒细胞" in normalized and ("↑" in normalized or "升高" in normalized):
        return True
    return False


def _extract_sources(sentence: str) -> List[str]:
    normalized = clean_text(sentence)
    sources: List[str] = []
    for label, keywords in SOURCE_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            _add_unique(sources, label)
    return sources


def _extract_competing_causes(sentence: str) -> List[str]:
    normalized = clean_text(sentence)
    values: List[str] = []
    for term in COMPETING_TERMS:
        if term in normalized:
            _add_unique(values, term)
    return values


def _extract_chronic_confounders(sentence: str) -> List[str]:
    normalized = clean_text(sentence)
    values: List[str] = []
    for term in CHRONIC_CONFOUNDER_TERMS:
        if term in normalized:
            _add_unique(values, term)
    return values


def _extract_shock_terms(sentence: str) -> List[str]:
    normalized = clean_text(sentence)
    values: List[str] = []
    for term in SHOCK_TERMS:
        if term in normalized:
            _add_unique(values, term)
    return values


def build_rule_regex_extraction_payload(case_payload: Dict[str, Any]) -> Dict[str, Any]:
    sentences = _iter_case_sentences(case_payload)
    positive_sentences: List[str] = []
    uncertain_sentences: List[str] = []
    absent_sentences: List[str] = []
    source_labels: List[str] = []
    competing_causes: List[str] = []
    chronic_confounders: List[str] = []
    shock_terms: List[str] = []

    has_fever = False
    has_antimicrobial = False
    has_inflammatory_signal = False
    has_non_uncertain_infection_hint = False
    saw_tumor_mixed_signal = False
    saw_negative_microbiology = False

    for sentence in sentences:
        has_fever = has_fever or _sentence_has_fever(sentence)
        has_antimicrobial = has_antimicrobial or _sentence_has_antimicrobial(sentence)
        has_inflammatory_signal = has_inflammatory_signal or _sentence_has_inflammatory_signal(sentence)
        has_non_uncertain_infection_hint = has_non_uncertain_infection_hint or (
            "感染" in sentence
            and not _sentence_has_uncertain_infection(sentence)
            and not _sentence_has_absent_infection(sentence)
            and not _sentence_has_antimicrobial(sentence)
        )
        saw_negative_microbiology = saw_negative_microbiology or any(term in sentence for term in NEGATIVE_CULTURE_TERMS)
        saw_tumor_mixed_signal = saw_tumor_mixed_signal or any(term in sentence for term in TUMOR_MIXED_TERMS)

        for source in _extract_sources(sentence):
            _add_unique(source_labels, source, limit=2)
        for item in _extract_competing_causes(sentence):
            _add_unique(competing_causes, item, limit=2)
        for item in _extract_chronic_confounders(sentence):
            _add_unique(chronic_confounders, item, limit=2)
        for item in _extract_shock_terms(sentence):
            _add_unique(shock_terms, item, limit=2)

        if _sentence_has_positive_microbiology(sentence) or _sentence_has_direct_infection(sentence):
            _add_unique(positive_sentences, sentence, limit=2)
        elif _sentence_has_uncertain_infection(sentence):
            _add_unique(uncertain_sentences, sentence, limit=2)
        elif _sentence_has_absent_infection(sentence):
            _add_unique(absent_sentences, sentence, limit=2)

    infection_evidence: List[str] = []
    if any(_sentence_has_positive_microbiology(sentence) for sentence in positive_sentences):
        _add_unique(infection_evidence, "病原学阳性", limit=2)
    if positive_sentences:
        if source_labels:
            _add_unique(infection_evidence, f"{source_labels[0]}感染线索", limit=2)
        else:
            _add_unique(infection_evidence, "明确感染线索", limit=2)
    if has_fever and has_antimicrobial:
        _add_unique(infection_evidence, "发热并抗感染治疗", limit=2)
    if has_inflammatory_signal:
        _add_unique(infection_evidence, "炎症指标升高", limit=2)
    if not infection_evidence and has_antimicrobial:
        _add_unique(infection_evidence, "已启动抗感染治疗", limit=2)

    uncertainty_notes: List[str] = []
    if saw_negative_microbiology:
        _add_unique(uncertainty_notes, "培养阴性或未检出", limit=2)
    if any("感染源不明" in sentence or "感染证据不足" in sentence for sentence in uncertain_sentences):
        _add_unique(uncertainty_notes, "感染源不明", limit=2)
    if saw_tumor_mixed_signal and (has_fever or has_inflammatory_signal):
        _add_unique(uncertainty_notes, "炎症/肿瘤混杂", limit=2)
    for sentence in uncertain_sentences:
        if "感染待排" in sentence or "疑似感染" in sentence or "考虑感染" in sentence:
            _add_unique(uncertainty_notes, "感染证据存在不确定性", limit=2)

    if positive_sentences:
        infection_status = "present"
    elif source_labels and ((has_fever and has_antimicrobial) or (has_inflammatory_signal and has_antimicrobial)):
        infection_status = "present"
    elif has_antimicrobial and has_fever and has_non_uncertain_infection_hint:
        infection_status = "present"
    elif absent_sentences and not (uncertain_sentences or has_fever or has_inflammatory_signal or has_antimicrobial or source_labels):
        infection_status = "absent"
    elif uncertain_sentences or has_fever or has_inflammatory_signal or has_antimicrobial or source_labels:
        infection_status = "uncertain"
    else:
        infection_status = "uncertain"

    evidence_quotes: List[str] = []
    if positive_sentences:
        _add_unique(evidence_quotes, positive_sentences[0], limit=1)
    elif uncertain_sentences:
        _add_unique(evidence_quotes, uncertain_sentences[0], limit=1)
    elif absent_sentences:
        _add_unique(evidence_quotes, absent_sentences[0], limit=1)

    return {
        "infection_status": infection_status,
        "infection_evidence": infection_evidence[:2],
        "infection_source": source_labels[:2],
        "reported_sofa": None,
        "respiration": {"pao2": None, "fio2": None, "pf_ratio": None},
        "coagulation": {"platelets": None},
        "liver": {"bilirubin": None, "unit": ""},
        "cardiovascular": {
            "sbp": None,
            "dbp": None,
            "map": None,
            "vasopressor_used": None,
            "vasopressor_name": "",
            "low_bp_text": False,
            "shock_text": False,
        },
        "cns": {"gcs": None, "new_mental_status_change": None},
        "renal": {"creatinine": None, "unit": "", "oliguria": None},
        "lactate": None,
        "acute_change_flags": {
            "respiration": [],
            "coagulation": [],
            "liver": [],
            "cardiovascular": [],
            "cns": [],
            "renal": [],
            "general": [],
        },
        "chronic_confounders": chronic_confounders[:2],
        "competing_noninfectious_causes": competing_causes[:2],
        "shock_terms": shock_terms[:2],
        "uncertainty_notes": uncertainty_notes[:2],
        "evidence_quotes": evidence_quotes[:1],
    }


def render_diagnosis_answer(assessment: Dict[str, Any]) -> str:
    lines = [f"诊断结果：{clean_text(assessment.get('diagnosis_result', '')) or '未完成判断'}", "诊断依据："]
    basis = assessment.get("diagnostic_basis", [])
    if isinstance(basis, list) and basis:
        for idx, item in enumerate(basis, start=1):
            lines.append(f"{idx}. {clean_text(item)}")
    else:
        lines.append(f"1. {clean_text(assessment.get('sepsis3_reason', '')) or '未提取到可用诊断依据'}")
    return "\n".join(lines)


def build_sofa_baseline_result(
    *,
    case_payload: Dict[str, Any],
    sofa_tool: SOFATool,
    judge: Sepsis3Judge,
) -> Dict[str, Any]:
    sofa_result = sofa_tool.evaluate(case_payload)
    extraction_payload = build_rule_regex_extraction_payload(case_payload)
    normalized_extraction = judge.normalize_extraction_payload(extraction_payload)
    assessment = judge.evaluate(
        normalized_extraction,
        sofa_result=sofa_result,
        case_payload=case_payload,
        enable_case_backfill=True,
        enable_sofa_backfill=True,
    )
    return {
        "answer": render_diagnosis_answer(assessment),
        "meta": {
            "profile_name": MODEL_PROFILE,
            "stage_errors": [],
            "sepsis3_extraction_raw": json.dumps(normalized_extraction, ensure_ascii=False),
            "sepsis3_extraction": normalized_extraction,
            "sepsis3_extraction_attempts": [],
            "sepsis3_extraction_parse_mode": "rule_regex",
            "sepsis3_judgement": assessment,
            "sepsis3_component_scores": assessment.get("component_scores", {}),
            "sepsis3_final_label": clean_text(assessment.get("classification", "")),
            "sepsis3_failure_mode": clean_text(assessment.get("classification", "")),
        },
    }


def run_csv_baseline_evaluation(
    *,
    sofa_tool: SOFATool,
    judge: Sepsis3Judge,
    csv_path: Path,
    ground_truth: str,
    sample_group: str,
    mode: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int],
) -> int:
    processed = 0
    for row_index, row in enumerate(_read_csv_rows(csv_path)):
        case_identifier = f"csv::{csv_path.name}::{row_index}"
        if case_identifier in resume_case_ids:
            continue
        if limit is not None and processed >= limit:
            break
        payload = build_case_payload_from_csv_row(dict(row), mode=mode)
        result = build_sofa_baseline_result(case_payload=payload, sofa_tool=sofa_tool, judge=judge)
        result_row = build_result_row(
            case_identifier=case_identifier,
            sample_type="csv",
            sample_group=sample_group,
            ground_truth=ground_truth,
            source_path=str(csv_path),
            row_index=str(row_index),
            bucket="",
            filename=clean_text(row.get("流水号", "")) or clean_text(row.get("EMPI", "")),
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
    sofa_tool: SOFATool,
    judge: Sepsis3Judge,
    manifest_path: Path,
    mode: str,
    results_path: Path,
    run_log_path: Path,
    resume_case_ids: Set[str],
    limit: Optional[int],
) -> int:
    processed = 0
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bucket = clean_text(row.get("bucket", ""))
            ground_truth = clean_text(row.get("label", ""))
            filename = clean_text(row.get("filename", ""))
            source_path = resolve_json_case_path_from_manifest_row(row)
            case_identifier = f"json::{bucket}::{ground_truth}::{filename}"
            if case_identifier in resume_case_ids:
                continue
            if limit is not None and processed >= limit:
                break
            payload = load_case_payload(Path(source_path), mode=mode)
            result = build_sofa_baseline_result(case_payload=payload, sofa_tool=sofa_tool, judge=judge)
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


def evaluate_sofa_baseline(
    *,
    csv_sepsis_path: Path = DEFAULT_CSV_SEPSIS,
    csv_non_sepsis_path: Path = DEFAULT_CSV_NON_SEPSIS,
    json_manifest_path: Path = DEFAULT_JSON_MANIFEST,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    mode: str = "strict_eval",
    limit: Optional[int] = None,
    resume: bool = True,
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

    sofa_tool = SOFATool()
    judge = Sepsis3Judge()
    run_csv_baseline_evaluation(
        sofa_tool=sofa_tool,
        judge=judge,
        csv_path=csv_sepsis_path,
        ground_truth="sepsis",
        sample_group="csv_sepsis",
        mode=mode,
        results_path=csv_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_csv_ids,
        limit=limit,
    )
    run_csv_baseline_evaluation(
        sofa_tool=sofa_tool,
        judge=judge,
        csv_path=csv_non_sepsis_path,
        ground_truth="non_sepsis",
        sample_group="csv_non_sepsis",
        mode=mode,
        results_path=csv_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_csv_ids,
        limit=limit,
    )
    run_json_baseline_evaluation(
        sofa_tool=sofa_tool,
        judge=judge,
        manifest_path=json_manifest_path,
        mode=mode,
        results_path=json_results_path,
        run_log_path=run_log_path,
        resume_case_ids=resume_json_ids,
        limit=limit,
    )

    csv_rows = _load_result_rows(csv_results_path)
    json_rows = _load_result_rows(json_results_path)
    summary = build_summary(csv_rows, json_rows)
    write_summary_files(output_dir, summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_sofa_baseline(
        csv_sepsis_path=Path(args.csv_sepsis).expanduser().resolve(),
        csv_non_sepsis_path=Path(args.csv_non_sepsis).expanduser().resolve(),
        json_manifest_path=Path(args.json_manifest).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        mode=args.mode,
        limit=args.limit,
        resume=args.resume,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
