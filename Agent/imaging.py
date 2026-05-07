from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pydicom
import torch
from PIL import Image
from pydicom.multival import MultiValue
from pydicom.pixel_data_handlers.util import apply_voi_lut

from .config import AgentConfig
from .modeling import LocalMedGemmaMultimodalModel, build_vision_backend
from .utils import clean_text, normalize_text, parse_python_like_list


IMAGE_TOKEN = "<image_soft_token>"
PROMPT_TEMPLATE = (
    "任务类型：影像报告生成。\n"
    "你将收到一次影像检查采样得到的多张 DICOM 图像。\n"
    "请根据图像内容生成规范、简洁的中文影像报告，输出字段包括“影像所见”和“检查结论”。"
)
ASCII_PROMPT_TEMPLATE = (
    "Task: Imaging report generation.\n"
    "You will receive multiple DICOM-derived images from one study.\n"
    "Please output concise Chinese results with two fields only:\n"
    "影像所见：...\n"
    "检查结论：..."
)
NON_IMAGING_PROJECT_TOKENS = ("分子检测", "基因", "PCR", "测序", "病理", "检验", "化验", "NGS")
IMAGING_PROJECT_TOKENS = (
    "CT",
    "MR",
    "MRI",
    "超声",
    "彩超",
    "B超",
    "X线",
    "DR",
    "CR",
    "PET",
    "SPECT",
    "CTA",
    "MRA",
    "胸片",
    "平片",
    "造影",
    "影像",
    "心脏超声",
)


def map_windows_path_to_image_root(raw_path: str, image_root: Path) -> Optional[Path]:
    normalized = clean_text(raw_path)
    if not normalized:
        return None
    normalized = normalized.replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].upper()
        rest = normalized[2:].lstrip("/")
        return (Path(image_root) / drive / rest).resolve()
    candidate = Path(normalized).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (Path(image_root) / normalized).resolve()


def collect_candidate_study_paths(case_payload: Dict[str, Any], image_root: Path) -> List[Path]:
    entries = case_payload.get("影像学检查", [])
    if not isinstance(entries, list):
        entries = parse_python_like_list(entries)
    paths: List[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        mapped = map_windows_path_to_image_root(entry.get("文件目录", ""), image_root)
        if mapped is None:
            continue
        if mapped not in paths:
            paths.append(mapped)
    return paths


def _is_imaging_entry(entry: Dict[str, Any]) -> bool:
    project = clean_text(entry.get("检查项目", ""))
    combined = normalize_text("\n".join([project, entry.get("影像所见", ""), entry.get("检查结论", "")]))
    if any(token.lower() in combined.lower() for token in NON_IMAGING_PROJECT_TOKENS):
        return any(token.lower() in project.lower() for token in IMAGING_PROJECT_TOKENS)
    return any(token.lower() in combined.lower() for token in IMAGING_PROJECT_TOKENS)


def _ascii_safe_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").strip()
    return re.sub(r"\n{3,}", "\n\n", ascii_text)


def natural_sort_key(path: Path) -> List[Any]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", str(path))]


def is_dicom_candidate(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in {".dcm", ".ima", ""} or path.name.lower().startswith("image")


def collect_series_groups(study_path: Path) -> List[Dict[str, Any]]:
    if not study_path.exists():
        return []
    if study_path.is_file():
        if not is_dicom_candidate(study_path):
            return []
        return [
            {
                "series_path": str(study_path.parent),
                "series_name": study_path.parent.name,
                "dcm_paths": [str(study_path)],
            }
        ]

    groups: List[Dict[str, Any]] = []
    for root, _, files in os.walk(study_path):
        root_path = Path(root)
        dcm_files = [root_path / name for name in files if is_dicom_candidate(root_path / name)]
        if not dcm_files:
            continue
        dcm_files.sort(key=natural_sort_key)
        groups.append(
            {
                "series_path": str(root_path),
                "series_name": root_path.name,
                "dcm_paths": [str(path) for path in dcm_files],
            }
        )
    groups.sort(key=lambda item: natural_sort_key(Path(item["series_path"])))
    return groups


def uniform_sample_paths(paths: Sequence[str], target_count: int) -> Tuple[List[str], int]:
    if target_count <= 0 or not paths:
        return [], 1
    if len(paths) <= target_count:
        return list(paths), 1
    step = int(math.ceil(len(paths) / target_count))
    sampled = list(paths[::step])
    if sampled and sampled[-1] != paths[-1]:
        sampled.append(paths[-1])
    return sampled[:target_count], step


def allocate_counts(group_sizes: Sequence[int], budget: int) -> List[int]:
    counts = [0 for _ in group_sizes]
    if budget <= 0:
        return counts
    indexed = [(idx, size) for idx, size in enumerate(group_sizes) if size > 0]
    if not indexed:
        return counts
    total = sum(size for _, size in indexed)
    if total <= budget:
        return list(group_sizes)
    if len(indexed) >= budget:
        for idx, _ in sorted(indexed, key=lambda item: (-item[1], item[0]))[:budget]:
            counts[idx] = 1
        return counts
    for idx, _ in indexed:
        counts[idx] = 1
    remaining = budget - sum(counts)
    capacities = [max(0, size - counts[idx]) for idx, size in enumerate(group_sizes)]
    cap_total = sum(capacities)
    if remaining <= 0 or cap_total <= 0:
        return counts
    fractional: List[Tuple[float, int]] = []
    for idx, capacity in enumerate(capacities):
        if capacity <= 0:
            continue
        share = remaining * capacity / cap_total
        base = int(math.floor(share))
        allocated = min(base, capacity)
        counts[idx] += allocated
        fractional.append((share - base, idx))
    leftover = budget - sum(counts)
    if leftover > 0:
        for _, idx in sorted(fractional, reverse=True):
            if leftover <= 0:
                break
            if counts[idx] < group_sizes[idx]:
                counts[idx] += 1
                leftover -= 1
    return counts


def first_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, MultiValue):
        if len(value) == 0:
            return None
        value = value[0]
    try:
        return float(value)
    except Exception:
        return None


def format_float_value(value: Optional[float], suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return ""
    if abs(value - round(value)) < 1e-6:
        text = str(int(round(value)))
    else:
        text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return f"{text}{suffix}"


def format_pixel_spacing(value: Any) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, MultiValue):
            parts = [first_float(item) for item in value[:2]]
        elif isinstance(value, (list, tuple)):
            parts = [first_float(item) for item in value[:2]]
        else:
            return format_float_value(first_float(value), suffix=" mm")
        parts = [part for part in parts if part is not None]
        if not parts:
            return ""
        if len(parts) == 1:
            return format_float_value(parts[0], suffix=" mm")
        return f"{format_float_value(parts[0])}x{format_float_value(parts[1])} mm"
    except Exception:
        return ""


def detect_plane(orientation: Any) -> str:
    try:
        if isinstance(orientation, MultiValue):
            values = [float(item) for item in orientation[:6]]
        elif isinstance(orientation, (list, tuple)):
            values = [float(item) for item in orientation[:6]]
        else:
            return ""
        if len(values) < 6:
            return ""
        row = np.array(values[:3], dtype=np.float32)
        col = np.array(values[3:6], dtype=np.float32)
        normal = np.cross(row, col)
        axis = int(np.argmax(np.abs(normal)))
        confidence = float(np.abs(normal[axis]))
        if confidence < 0.8:
            return "oblique"
        return {0: "sagittal", 1: "coronal", 2: "axial"}[axis]
    except Exception:
        return ""


def detect_contrast_used(ds: Any) -> str:
    for field in ("ContrastBolusAgent", "ContrastBolusRoute", "ContrastBolusVolume"):
        if clean_text(getattr(ds, field, "")):
            return "yes"
    return ""


def reduce_pixel_array(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 3:
        arr = arr[arr.shape[0] // 2]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3]
        if arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            return np.moveaxis(arr, 0, -1)[..., :3]
        arr = arr[arr.shape[0] // 2]
    return arr


def normalize_gray_image(arr: np.ndarray, ds: Any) -> np.ndarray:
    arr = arr.astype(np.float32)
    slope = first_float(getattr(ds, "RescaleSlope", None))
    intercept = first_float(getattr(ds, "RescaleIntercept", None))
    if slope is not None:
        arr = arr * slope
    if intercept is not None:
        arr = arr + intercept

    center = first_float(getattr(ds, "WindowCenter", None))
    width = first_float(getattr(ds, "WindowWidth", None))
    if center is not None and width is not None and width > 1:
        low = center - width / 2.0
        high = center + width / 2.0
    else:
        low = float(np.percentile(arr, 1))
        high = float(np.percentile(arr, 99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(arr))
        high = float(np.max(arr))
    if high <= low:
        high = low + 1e-6
    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low + 1e-6)
    arr = (arr * 255.0).astype(np.uint8)
    photometric = clean_text(getattr(ds, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        arr = 255 - arr
    return arr


def normalize_rgb_image(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    else:
        low = float(np.percentile(arr, 1))
        high = float(np.percentile(arr, 99))
        if high <= low:
            low = float(np.min(arr))
            high = float(np.max(arr))
        if high <= low:
            high = low + 1e-6
        arr = np.clip(arr, low, high)
        arr = (arr - low) / (high - low + 1e-6) * 255.0
    return arr.astype(np.uint8)


def dicom_to_image_and_meta(path: Path) -> Tuple[Image.Image, Dict[str, str]]:
    ds = pydicom.dcmread(str(path), force=True)
    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM does not contain pixel data")
    try:
        pixel_array = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        pixel_array = ds.pixel_array
    pixel_array = reduce_pixel_array(np.asarray(pixel_array))
    if pixel_array.ndim == 2:
        array_uint8 = normalize_gray_image(pixel_array, ds)
        image = Image.fromarray(array_uint8).convert("RGB")
    elif pixel_array.ndim == 3 and pixel_array.shape[-1] in (3, 4):
        array_uint8 = normalize_rgb_image(pixel_array[..., :3])
        image = Image.fromarray(array_uint8).convert("RGB")
    else:
        raise ValueError(f"unsupported pixel shape: {pixel_array.shape}")

    meta = {
        "modality": clean_text(getattr(ds, "Modality", "")),
        "series_description": clean_text(getattr(ds, "SeriesDescription", "")),
        "study_description": clean_text(getattr(ds, "StudyDescription", "")),
        "body_part": clean_text(getattr(ds, "BodyPartExamined", "")),
        "plane": detect_plane(getattr(ds, "ImageOrientationPatient", None)),
        "laterality": clean_text(getattr(ds, "Laterality", "")),
        "view_position": clean_text(getattr(ds, "ViewPosition", "")),
        "patient_position": clean_text(getattr(ds, "PatientPosition", "")),
        "slice_thickness": format_float_value(first_float(getattr(ds, "SliceThickness", None)), suffix=" mm"),
        "spacing_between_slices": format_float_value(first_float(getattr(ds, "SpacingBetweenSlices", None)), suffix=" mm"),
        "pixel_spacing": format_pixel_spacing(getattr(ds, "PixelSpacing", None)),
        "kvp": format_float_value(first_float(getattr(ds, "KVP", None)), suffix=" kV"),
        "convolution_kernel": clean_text(getattr(ds, "ConvolutionKernel", "")),
        "contrast_used": detect_contrast_used(ds),
    }
    meta = {key: value for key, value in meta.items() if value}
    return image, meta


def load_images_and_metas(dcm_paths: Sequence[str]) -> Tuple[List[Image.Image], List[Dict[str, str]], List[str]]:
    images: List[Image.Image] = []
    metas: List[Dict[str, str]] = []
    loaded_paths: List[str] = []
    for dcm_path in dcm_paths:
        try:
            image, meta = dicom_to_image_and_meta(Path(dcm_path))
        except Exception:
            continue
        images.append(image)
        metas.append(meta)
        loaded_paths.append(str(dcm_path))
    return images, metas, loaded_paths


def build_meta_text(record: Dict[str, Any], metas: Sequence[Dict[str, str]], sampled_count: int) -> str:
    lines = [
        f"检查项目：{clean_text(record.get('project', '')) or '未提供'}",
        f"影像序列数：{int(record.get('series_count', 0) or 0)}",
        f"DICOM总数：{int(record.get('total_dcm_count', 0) or 0)}",
        f"采样后图像数：{sampled_count}",
    ]
    modality_counter = Counter(meta.get("modality", "") for meta in metas if clean_text(meta.get("modality", "")))
    if modality_counter:
        lines.append("模态分布：" + "；".join(f"{key}={value}" for key, value in sorted(modality_counter.items())))

    def summarize_values(key: str, label: str, max_values: int = 3) -> None:
        values = []
        seen = set()
        for meta in metas:
            value = clean_text(meta.get(key, ""))
            if not value or value in seen:
                continue
            seen.add(value)
            values.append(value)
            if len(values) >= max_values:
                break
        if values:
            lines.append(f"{label}：" + "；".join(values))

    summarize_values("plane", "Plane")
    summarize_values("body_part", "BodyPart")
    summarize_values("view_position", "ViewPosition")
    summarize_values("laterality", "Laterality")
    summarize_values("patient_position", "PatientPosition")
    summarize_values("slice_thickness", "SliceThickness")
    summarize_values("spacing_between_slices", "SpacingBetweenSlices")
    summarize_values("pixel_spacing", "PixelSpacing")
    summarize_values("kvp", "KVP")
    summarize_values("convolution_kernel", "ConvolutionKernel")
    summarize_values("contrast_used", "ContrastUsed")
    summarize_values("study_description", "StudyDescription")
    summarize_values("series_description", "SeriesDescription")
    return "\n".join(lines)


def get_image_token_id(processor: Any) -> Optional[int]:
    try:
        image_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        if image_token_id == processor.tokenizer.unk_token_id:
            return None
        return int(image_token_id)
    except Exception:
        return None


def resample_to_target_count(paths: Sequence[str], target_count: int) -> List[str]:
    sampled, _ = uniform_sample_paths(paths, target_count)
    return sampled


def build_generation_inputs(
    processor: Any,
    record: Dict[str, Any],
    max_length: int,
    context_reserve_tokens: int,
) -> Tuple[Dict[str, torch.Tensor], str, List[str], int]:
    sampled_paths = list(record.get("sampled_dcm_paths", []))
    image_seq_length = max(1, int(getattr(processor, "image_seq_length", 256)))
    max_images_by_context = max(1, (max_length - context_reserve_tokens) // image_seq_length)
    if len(sampled_paths) > max_images_by_context:
        sampled_paths = resample_to_target_count(sampled_paths, max_images_by_context)

    image_token_id = get_image_token_id(processor)
    last_inputs: Optional[Dict[str, torch.Tensor]] = None
    last_meta_text = ""
    last_loaded_paths: List[str] = []
    last_loaded_count = 0
    max_iterations = max(8, len(sampled_paths) + 8)
    for _ in range(max_iterations):
        images, metas, loaded_paths = load_images_and_metas(sampled_paths)
        if not images:
            images = [Image.fromarray(np.zeros((256, 256), dtype=np.uint8)).convert("RGB")]
            metas = []
            loaded_paths = []
            loaded_count = 0
        else:
            loaded_count = len(images)

        meta_text = build_meta_text(record, metas, loaded_count)
        user_text = f"{PROMPT_TEMPLATE}\n\n{meta_text}"
        user_content = [{"type": "image"} for _ in range(len(images))]
        user_content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": user_content}]
        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).strip()

        inputs = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            truncation=False,
        )
        prompt_len = int(inputs["input_ids"].shape[1])
        last_inputs = inputs
        last_meta_text = meta_text
        last_loaded_paths = loaded_paths
        last_loaded_count = loaded_count

        if image_token_id is not None and "pixel_values" in inputs:
            prompt_token_count = int((inputs["input_ids"] == image_token_id).sum().item())
            prompt_feature_count = int(inputs["pixel_values"].shape[0])
            expected_token_count = prompt_feature_count * image_seq_length
            if prompt_token_count != expected_token_count and len(sampled_paths) > 1:
                target_count = max(1, prompt_feature_count)
                if target_count >= len(sampled_paths):
                    target_count = len(sampled_paths) - 1
                sampled_paths = resample_to_target_count(sampled_paths, target_count)
                continue

        if prompt_len <= max_length - max(64, context_reserve_tokens):
            return inputs, meta_text, loaded_paths, loaded_count

        if len(sampled_paths) > 1:
            sampled_paths = resample_to_target_count(sampled_paths, len(sampled_paths) - 1)
            continue

        truncated_inputs = processor(
            text=prompt_text,
            images=images,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        return truncated_inputs, meta_text, loaded_paths, loaded_count

    if last_inputs is None:
        raise RuntimeError("无法构建影像推理输入。")
    return last_inputs, last_meta_text, last_loaded_paths, last_loaded_count


def try_parse_report_json_fields(text: str) -> Tuple[str, str]:
    normalized = clean_text(text)
    if not normalized:
        return "", ""
    if normalized.startswith("```"):
        normalized = re.sub(r"^```[a-zA-Z0-9_]*\n", "", normalized)
        normalized = re.sub(r"\n```$", "", normalized).strip()
    start = normalized.find("{")
    end = normalized.rfind("}")
    if start < 0 or end <= start:
        return "", ""
    try:
        payload = __import__("json").loads(normalized[start : end + 1])
    except Exception:
        return "", ""
    findings = normalize_text(payload.get("影像所见") or payload.get("findings") or payload.get("所见") or "")
    conclusion = normalize_text(payload.get("检查结论") or payload.get("conclusion") or payload.get("结论") or "")
    return findings, conclusion


def normalize_report_response_text(text: str) -> str:
    normalized = normalize_text(text)
    if not normalized:
        return ""
    normalized = re.sub(r"\*+\s*影像所见\s*[:：]\s*\*+", "影像所见：", normalized)
    normalized = re.sub(r"\*+\s*检查结论\s*[:：]\s*\*+", "检查结论：", normalized)
    normalized = re.sub(r"^[>#\-\s]+(?=影像所见[:：])", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"^[>#\-\s]+(?=检查结论[:：])", "", normalized, flags=re.MULTILINE)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def parse_report_fields(text: str) -> Tuple[str, str, str]:
    normalized = normalize_report_response_text(text)
    if not normalized:
        return "", "", "empty"
    json_findings, json_conclusion = try_parse_report_json_fields(normalized)
    if json_findings or json_conclusion:
        return json_findings, json_conclusion, "json"
    findings_match = re.search(r"影像所见[:：]\s*(.*?)(?=\n\s*检查结论[:：]|\Z)", normalized, flags=re.DOTALL)
    conclusion_match = re.search(r"检查结论[:：]\s*(.*)\Z", normalized, flags=re.DOTALL)
    if findings_match or conclusion_match:
        findings = normalize_text(findings_match.group(1) if findings_match else "")
        conclusion = normalize_text(conclusion_match.group(1) if conclusion_match else "")
        return findings, conclusion, "labeled"
    return normalized, "", "fallback_full"


def extract_inline_imaging_text(case_payload: Dict[str, Any], max_entries: int = 2) -> str:
    entries = case_payload.get("影像学检查", [])
    if not isinstance(entries, list):
        entries = parse_python_like_list(entries)
    lines: List[str] = []
    kept_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if not _is_imaging_entry(entry):
            continue
        project = clean_text(entry.get("检查项目", "")) or "未注明项目"
        findings = normalize_text(entry.get("影像所见", ""))
        conclusion = normalize_text(entry.get("检查结论", ""))
        if not findings and not conclusion:
            continue
        kept_count += 1
        if kept_count > max_entries:
            break
        lines.append(f"已有影像文本{kept_count}：项目={project}")
        if findings:
            lines.append(f"影像所见：{findings}")
        if conclusion:
            lines.append(f"检查结论：{conclusion}")
    return "\n".join(lines).strip()


def build_case_study_records(case_payload: Dict[str, Any], image_root: Path, max_images_per_study: int) -> List[Dict[str, Any]]:
    entries = case_payload.get("影像学检查", [])
    if not isinstance(entries, list):
        entries = parse_python_like_list(entries)

    records: List[Dict[str, Any]] = []
    for study_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        mapped_path = map_windows_path_to_image_root(entry.get("文件目录", ""), image_root)
        if mapped_path is None or not mapped_path.exists():
            continue
        series_groups = collect_series_groups(mapped_path)
        if not series_groups:
            continue
        group_sizes = [len(group["dcm_paths"]) for group in series_groups]
        total_dcm_count = int(sum(group_sizes))
        allocated = allocate_counts(group_sizes, max(1, max_images_per_study))
        sampled_dcm_paths: List[str] = []
        for group, target_count in zip(series_groups, allocated):
            sampled_paths, _ = uniform_sample_paths(group["dcm_paths"], target_count)
            sampled_dcm_paths.extend(sampled_paths)
        sampled_dcm_paths = list(dict.fromkeys(sampled_dcm_paths))
        if not sampled_dcm_paths:
            continue
        records.append(
            {
                "study_id": f"study-{study_idx}",
                "project": clean_text(entry.get("检查项目", "")),
                "mapped_path": str(mapped_path),
                "series_count": len(series_groups),
                "total_dcm_count": total_dcm_count,
                "sampled_dcm_paths": sampled_dcm_paths,
            }
        )
    return records


class ImagingStudyReportGenerator:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._backend: Optional[Any] = None

    def load_backend(self) -> Any:
        if self._backend is None:
            backend_config = self.config.resolve_vision_backend()
            self._backend = build_vision_backend(
                backend_config=backend_config,
                hf_cache_root=self.config.hf_cache_root,
                attn_implementation=self.config.attn_implementation,
                local_files_only=self.config.local_files_only,
                max_length=self.config.generation_max_length,
                timeout_seconds=self.config.http_timeout_seconds,
            )
        return self._backend

    def summarize_case(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        inline_text = extract_inline_imaging_text(case_payload)
        if not self.config.enable_imaging:
            return {"text": inline_text, "sources": [], "studies": []}

        study_records = build_case_study_records(
            case_payload,
            image_root=self.config.case_image_root,
            max_images_per_study=self.config.max_images_per_study,
        )
        if not study_records:
            return {"text": inline_text, "sources": [], "studies": []}

        try:
            backend = self.load_backend()
        except Exception as exc:
            fallback_studies = [{"study_id": "imaging_backend", "error": str(exc)}]
            return {"text": inline_text, "sources": [], "studies": fallback_studies}
        text_sections: List[str] = [inline_text] if inline_text else []
        sources: List[Dict[str, Any]] = []
        studies: List[Dict[str, Any]] = []
        for idx, record in enumerate(study_records[: self.config.max_imaging_studies], start=1):
            try:
                generation = self._generate_record(backend, record)
            except Exception as exc:
                studies.append(
                    {
                        "study_id": record["study_id"],
                        "error": str(exc),
                    }
                )
                continue

            findings, conclusion, parse_mode = parse_report_fields(generation["raw_response"])
            study_lines = [f"影像模型检查{idx}：项目={record.get('project') or '未注明项目'}"]
            if findings:
                study_lines.append(f"影像所见：{findings}")
            if conclusion:
                study_lines.append(f"检查结论：{conclusion}")
            if not findings and not conclusion:
                study_lines.append(f"原始输出：{clean_text(generation['raw_response'])[:400]}")
            text_sections.append("\n".join(study_lines))

            source_path = generation["used_dcm_paths"][0] if generation["used_dcm_paths"] else record["mapped_path"]
            sources.append(
                {
                    "tag": f"IMG{idx}",
                    "source": source_path,
                    "snippet": clean_text(conclusion or findings or generation["raw_response"])[:280],
                }
            )
            studies.append(
                {
                    "study_id": record["study_id"],
                    "project": record.get("project", ""),
                    "raw_response": generation["raw_response"],
                    "findings": findings,
                    "conclusion": conclusion,
                    "parse_mode": parse_mode,
                    "used_dcm_paths": generation["used_dcm_paths"],
                    "used_dcm_count": generation["used_dcm_count"],
                }
            )
        return {"text": "\n\n".join([part for part in text_sections if part]).strip(), "sources": sources, "studies": studies}

    def _persist_png_images(self, record: Dict[str, Any], images: Sequence[Image.Image], loaded_paths: Sequence[str]) -> List[Path]:
        record_key = clean_text(record.get("mapped_path", "")) or clean_text(record.get("study_id", "study"))
        cache_id = hashlib.md5(record_key.encode("utf-8")).hexdigest()[:12]
        cache_dir = self.config.runtime_root / "imaging_cache" / cache_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        cached_paths: List[Path] = []
        for idx, image in enumerate(images, start=1):
            source_stem = Path(loaded_paths[idx - 1]).stem if idx - 1 < len(loaded_paths) else f"img_{idx:02d}"
            safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_stem).strip("_") or f"img_{idx:02d}"
            png_path = cache_dir / f"{idx:02d}_{safe_stem}.png"
            if not png_path.exists():
                image.save(png_path, format="PNG")
            cached_paths.append(png_path)
        return cached_paths

    def _generate_record(self, backend: Any, record: Dict[str, Any]) -> Dict[str, Any]:
        sampled_paths = list(record.get("sampled_dcm_paths", []))
        if isinstance(backend, LocalMedGemmaMultimodalModel):
            max_images_by_context = backend.estimate_max_images(
                max_length=self.config.generation_max_length,
                context_reserve_tokens=self.config.context_reserve_tokens,
            )
            if len(sampled_paths) > max_images_by_context:
                sampled_paths = resample_to_target_count(sampled_paths, max_images_by_context)

        images, metas, loaded_paths = load_images_and_metas(sampled_paths)
        if not images:
            raise RuntimeError("未能从采样的 DICOM 中解码出有效图像。")

        meta_text = build_meta_text(record, metas, len(images))
        prompt_text = f"{PROMPT_TEMPLATE}\n\n{meta_text}"
        ascii_prompt_text = f"{ASCII_PROMPT_TEMPLATE}\n\n{_ascii_safe_text(meta_text)}".strip()

        if isinstance(backend, LocalMedGemmaMultimodalModel):
            raw_response = backend.generate(
                prompt_text=prompt_text,
                images=images,
                max_new_tokens=self.config.imaging_max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                repetition_penalty=1.0,
                max_length=self.config.generation_max_length,
            )
            cached_image_paths: List[str] = []
        else:
            cached_paths = self._persist_png_images(record, images, loaded_paths)
            try:
                raw_response = backend.generate(
                    system_prompt="",
                    user_prompt=prompt_text,
                    image_paths=cached_paths,
                    max_new_tokens=self.config.imaging_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
            except Exception as exc:
                error_text = str(exc)
                if "ascii" not in error_text.lower():
                    raise
                raw_response = backend.generate(
                    system_prompt="",
                    user_prompt=ascii_prompt_text,
                    image_paths=cached_paths,
                    max_new_tokens=self.config.imaging_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
            cached_image_paths = [str(path) for path in cached_paths]

        return {
            "raw_response": raw_response,
            "prompt_meta_text": meta_text,
            "used_dcm_count": len(images),
            "used_dcm_paths": loaded_paths,
            "cached_image_paths": cached_image_paths,
        }
