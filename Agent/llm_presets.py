from __future__ import annotations

from typing import Dict, Iterable

from .config import DEFAULT_DASHSCOPE_BASE_URL, ModelBackendConfig
from .utils import clean_text


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
BAICHUAN_BASE_URL = "https://api.baichuan-ai.com/v1/"
ANTANGEL_BASE_URL = "https://api.tbox.cn/api/llm/v1/chat/completions"


_TEXT_BACKEND_PRESETS: Dict[str, ModelBackendConfig] = {
    "deepseek": ModelBackendConfig(
        provider="openai_compatible",
        model="deepseek-chat",
        base_url=DEEPSEEK_BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
    ),
    "qwen_plus": ModelBackendConfig(
        provider="openai_compatible",
        model="qwen-plus",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "qwen_max": ModelBackendConfig(
        provider="openai_compatible",
        model="qwen-max",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
    ),
    "deepseek_cot": ModelBackendConfig(
        provider="openai_compatible",
        model="deepseek-chat",
        base_url=DEEPSEEK_BASE_URL,
        api_key_env="DEEPSEEK_API_KEY",
        extra_body={"thinking": {"type": "enabled"}},
        default_max_new_tokens=1024,
        default_timeout_seconds=120.0,
        sepsis3_extraction_max_new_tokens=4096,
        sepsis3_extraction_timeout_seconds=120.0,
    ),
    "qwen_plus_cot": ModelBackendConfig(
        provider="openai_compatible",
        model="qwen-plus",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
        extra_body={"enable_thinking": True},
        default_max_new_tokens=1024,
        default_timeout_seconds=120.0,
        sepsis3_extraction_max_new_tokens=4096,
        sepsis3_extraction_timeout_seconds=120.0,
    ),
    "qwen_max_cot": ModelBackendConfig(
        provider="openai_compatible",
        model="qwen3-max",
        base_url=DEFAULT_DASHSCOPE_BASE_URL,
        api_key_env="DASHSCOPE_API_KEY",
        extra_body={"enable_thinking": True},
        default_max_new_tokens=2048,
        default_timeout_seconds=180.0,
        sepsis3_extraction_max_new_tokens=6144,
        sepsis3_extraction_timeout_seconds=180.0,
    ),
    "baichuan": ModelBackendConfig(
        provider="openai_compatible",
        model="Baichuan-M3",
        base_url=BAICHUAN_BASE_URL,
        api_key_env="BAICHUAN_API_KEY",
    ),
    "antangel": ModelBackendConfig(
        provider="antangel",
        model="AntAngelMed",
        base_url=ANTANGEL_BASE_URL,
        api_key_env="ANTANGEL_API_KEY",
    ),
}

_ALIASES = {
    "deepseek": "deepseek",
    "deepseek-chat": "deepseek",
    "deepseek-cot": "deepseek_cot",
    "deepseek_cot": "deepseek_cot",
    "qwen-plus": "qwen_plus",
    "qwen_plus": "qwen_plus",
    "qwenplus": "qwen_plus",
    "qwen-max": "qwen_max",
    "qwen_max": "qwen_max",
    "qwenmax": "qwen_max",
    "qwen-plus-cot": "qwen_plus_cot",
    "qwen_plus_cot": "qwen_plus_cot",
    "qwenpluscot": "qwen_plus_cot",
    "qwen-max-cot": "qwen_max_cot",
    "qwen_max_cot": "qwen_max_cot",
    "qwenmaxcot": "qwen_max_cot",
    "antangel": "antangel",
    "antangelmed": "antangel",
    "baichuan": "baichuan",
    "baichuan-m3": "baichuan",
}

BASE_TEXT_PRESET_NAMES = ("qwen_plus", "deepseek", "qwen_max")
MEDICAL_TREATMENT_PRESET_NAMES = ("baichuan", "antangel")


def normalize_text_backend_preset_name(name: str) -> str:
    normalized = clean_text(name).lower().replace(" ", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in _TEXT_BACKEND_PRESETS:
        available = ", ".join(sorted(_TEXT_BACKEND_PRESETS))
        raise ValueError(f"Unknown text backend preset: {name}; available: {available}")
    return normalized


def normalize_allowed_text_backend_preset_name(name: str, *, allowed_presets: Iterable[str]) -> str:
    normalized = normalize_text_backend_preset_name(name)
    allowed = {normalize_text_backend_preset_name(item) for item in allowed_presets}
    if normalized not in allowed:
        available = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported text backend preset: {name}; allowed: {available}")
    return normalized


def list_text_backend_presets() -> Dict[str, str]:
    return {
        "deepseek": "DeepSeek deepseek-chat",
        "qwen_plus": "DashScope qwen-plus",
        "qwen_max": "DashScope qwen-max",
        "deepseek_cot": "DeepSeek deepseek-chat with thinking enabled",
        "qwen_plus_cot": "DashScope qwen-plus with thinking enabled",
        "qwen_max_cot": "DashScope qwen3-max with thinking enabled",
        "baichuan": "Baichuan-M3",
        "antangel": "AntAngelMed",
    }


def resolve_text_backend_preset(name: str) -> ModelBackendConfig:
    preset_name = normalize_text_backend_preset_name(name)
    return _TEXT_BACKEND_PRESETS[preset_name]


def identify_text_backend_preset_name(backend_config: ModelBackendConfig | None) -> str:
    if backend_config is None:
        return ""
    for preset_name, preset_config in _TEXT_BACKEND_PRESETS.items():
        if backend_config == preset_config:
            return preset_name
    return ""


__all__ = [
    "BASE_TEXT_PRESET_NAMES",
    "ANTANGEL_BASE_URL",
    "BAICHUAN_BASE_URL",
    "DEEPSEEK_BASE_URL",
    "MEDICAL_TREATMENT_PRESET_NAMES",
    "identify_text_backend_preset_name",
    "list_text_backend_presets",
    "normalize_allowed_text_backend_preset_name",
    "normalize_text_backend_preset_name",
    "resolve_text_backend_preset",
]
