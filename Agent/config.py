from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 指向 毕业设计 目录
DEFAULT_ORIGINAL_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_FULL_FT_MODEL = PROJECT_ROOT / "models" / "medgemma-4b-it-sft-single-stage"
DEFAULT_REPLAY_LORA_CHECKPOINT = PROJECT_ROOT / "models" / "medgemma-4b-it-sft-lora-replay" / "checkpoint-1120"
DEFAULT_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_TEXT_MODEL = "qwen-plus"
DEFAULT_QWEN_VISION_MODEL = "qwen-vl-max-latest"
DEFAULT_KNOWLEDGE_ROOT = PROJECT_ROOT / "Agent" / "rag"
# API keys are intentionally not hard-coded in source code.
# Set DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, BAICHUAN_API_KEY, or ANTANGEL_API_KEY in the runtime environment.

@dataclass(frozen=True)
class ModelBackendConfig:
    provider: str
    model: Union[str, Path]
    adapter_path: Optional[Union[str, Path]] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    default_max_new_tokens: Optional[int] = None
    default_timeout_seconds: Optional[float] = None
    sepsis3_extraction_max_new_tokens: Optional[int] = None
    sepsis3_extraction_timeout_seconds: Optional[float] = None

    def describe(self) -> str:
        model_text = str(self.model)
        adapter_text = f"+adapter={self.adapter_path}" if self.adapter_path else ""
        base_url_text = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{model_text}{adapter_text}{base_url_text}"


def default_qwen_backend(model: Union[str, Path] = DEFAULT_QWEN_TEXT_MODEL) -> ModelBackendConfig:
    return ModelBackendConfig(
        provider="openai_compatible",
        model=model,
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
    )


def default_dashscope_multimodal_backend(model: Union[str, Path] = DEFAULT_QWEN_VISION_MODEL) -> ModelBackendConfig:
    return ModelBackendConfig(
        provider="dashscope_multimodal",
        model=model,
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
    )


def default_local_text_backend(model: Union[str, Path], adapter_path: Optional[Union[str, Path]] = None) -> ModelBackendConfig:
    return ModelBackendConfig(
        provider="local_medgemma_text",
        model=model,
        adapter_path=adapter_path,
    )


def default_local_multimodal_backend(model: Union[str, Path], adapter_path: Optional[Union[str, Path]] = None) -> ModelBackendConfig:
    return ModelBackendConfig(
        provider="local_medgemma_multimodal",
        model=model,
        adapter_path=adapter_path,
    )


def default_text_backend() -> ModelBackendConfig:
    return default_qwen_backend()


def default_vision_backend() -> ModelBackendConfig:
    return default_dashscope_multimodal_backend()


def default_case_image_root() -> Path:
    candidates = (
        PROJECT_ROOT / "data" / "multimodel1",
        PROJECT_ROOT / "data" / "multimodel",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


@dataclass
class AgentConfig:
    project_root: Path = PROJECT_ROOT
    runtime_root: Path = PROJECT_ROOT / "Agent" / ".runtime"
    knowledge_root: Path = DEFAULT_KNOWLEDGE_ROOT
    case_image_root: Path = field(default_factory=default_case_image_root)
    hf_cache_root: Path = DEFAULT_HF_CACHE_ROOT

    original_model_id: str = DEFAULT_ORIGINAL_MODEL_ID
    model_path: Union[str, Path] = DEFAULT_FULL_FT_MODEL
    text_adapter_path: Optional[Union[str, Path]] = None
    imaging_model_path: Optional[Union[str, Path]] = DEFAULT_ORIGINAL_MODEL_ID
    imaging_adapter_path: Optional[Union[str, Path]] = None

    default_text_backend: ModelBackendConfig = field(default_factory=default_text_backend)
    default_vision_backend: ModelBackendConfig = field(default_factory=default_vision_backend)

    reflection_backend: Optional[ModelBackendConfig] = None
    code_writer_backend: Optional[ModelBackendConfig] = None
    diagnosis_backend: Optional[ModelBackendConfig] = None
    treatment_backend: Optional[ModelBackendConfig] = field(default_factory=default_qwen_backend)
    medical_treatment_backend: Optional[ModelBackendConfig] = None
    imaging_backend: Optional[ModelBackendConfig] = None

    attn_implementation: str = "sdpa"
    local_files_only: bool = False

    rag_top_k: int = 3
    max_images_per_study: int = 24
    max_imaging_studies: int = 2
    context_reserve_tokens: int = 512
    case_input_mode: str = "strict_eval"

    generation_max_length: int = 4096
    generation_max_new_tokens: int = 256
    sepsis3_extraction_max_new_tokens: int = 512
    reflection_max_new_tokens: int = 192
    code_writer_max_new_tokens: int = 768
    imaging_max_new_tokens: int = 256
    generation_temperature: float = 0.0
    generation_top_p: float = 1.0
    generation_repetition_penalty: float = 1.0

    enable_local_rag: bool = True
    enable_reflection: bool = True
    enable_imaging: bool = True
    enable_generated_code: bool = False
    enable_treatment_round_reminders: bool = False
    enable_treatment_react: bool = True
    enable_diagnosis_llm_extraction: bool = True
    enable_diagnosis_sofa_tool: bool = True
    enable_diagnosis_case_backfill: bool = True
    disabled_treatment_tools: Tuple[str, ...] = ()

    max_react_rounds: int = 16
    http_timeout_seconds: float = 20.0
    sepsis3_extraction_timeout_seconds: float = 60.0
    profile_name: str = "default"
    builtin_search_default: bool = False
    builtin_search_options: Optional[Dict[str, Any]] = None

    def resolve_text_backend(self, role: str) -> ModelBackendConfig:
        backend = getattr(self, f"{role}_backend", None)
        if backend is not None:
            return backend
        return self.default_text_backend

    def resolve_vision_backend(self) -> ModelBackendConfig:
        return self.imaging_backend or self.default_vision_backend

    def ensure_runtime_dirs(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "rag").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "code_exec").mkdir(parents=True, exist_ok=True)
        (self.runtime_root / "imaging_cache").mkdir(parents=True, exist_ok=True)
