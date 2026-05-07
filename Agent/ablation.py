from __future__ import annotations

from dataclasses import replace
from typing import Dict

from .config import (
    AgentConfig,
    default_dashscope_multimodal_backend,
    default_local_multimodal_backend,
    default_local_text_backend,
    default_qwen_backend,
)
from .modeling import resolve_model_source


def list_ablation_profiles() -> Dict[str, str]:
    return {
        "original_all": "Use the original MedGemma end-to-end without fine-tuning or LoRA.",
        "ft_lora_all": "Use the single-stage fine-tuned MedGemma for all text stages.",
        "default": "Default runtime stack with qwen-plus text and qwen-vl-max-latest imaging.",
        "no_rag": "Default runtime stack with local PDF RAG disabled.",
        "with_code_probe": "Default runtime stack with generated code probe enabled for treatment.",
        "no_reflection": "Default runtime stack with the final reflection stage disabled.",
        "no_traditional": "Diagnosis runtime with SOFA-based traditional support disabled.",
        "traditional_only_no_llm": "Diagnosis runtime with LLM extraction disabled but SOFA and rule fusion kept.",
    }


def resolve_original_model_reference(config: AgentConfig) -> str:
    return resolve_model_source(config.original_model_id, config.hf_cache_root)


def build_ablation_config(profile: str, base_config: AgentConfig) -> AgentConfig:
    if profile not in list_ablation_profiles():
        raise ValueError(f"未知 profile: {profile}")

    original_ref = resolve_original_model_reference(base_config)
    default_config = replace(
        base_config,
        profile_name="default",
        default_text_backend=default_qwen_backend(),
        default_vision_backend=default_dashscope_multimodal_backend(),
        reflection_backend=None,
        code_writer_backend=None,
        diagnosis_backend=None,
        treatment_backend=default_qwen_backend(),
        imaging_backend=None,
        enable_local_rag=True,
        enable_reflection=True,
        enable_imaging=True,
        enable_generated_code=False,
        enable_diagnosis_llm_extraction=True,
        enable_diagnosis_sofa_tool=True,
        enable_diagnosis_case_backfill=True,
    )

    if profile == "original_all":
        return replace(
            default_config,
            profile_name=profile,
            default_text_backend=default_local_text_backend(original_ref),
            default_vision_backend=default_local_multimodal_backend(original_ref),
        )
    if profile == "ft_lora_all":
        return replace(
            default_config,
            profile_name=profile,
            default_text_backend=default_local_text_backend(default_config.model_path, default_config.text_adapter_path),
        )
    if profile == "default":
        return default_config
    if profile == "no_rag":
        return replace(default_config, profile_name=profile, enable_local_rag=False)
    if profile == "with_code_probe":
        return replace(default_config, profile_name=profile, enable_generated_code=True)
    if profile == "no_reflection":
        return replace(default_config, profile_name=profile, enable_reflection=False)
    if profile == "no_traditional":
        return replace(default_config, profile_name=profile, enable_diagnosis_sofa_tool=False)
    if profile == "traditional_only_no_llm":
        return replace(
            default_config,
            profile_name=profile,
            enable_diagnosis_llm_extraction=False,
            enable_reflection=False,
        )
    raise ValueError(f"未知 profile: {profile}")
