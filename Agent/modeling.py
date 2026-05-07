from __future__ import annotations

import base64
import gc
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Union

import httpx
import torch

from .config import ModelBackendConfig
from .utils import clean_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_ENV_ALIASES = {
    "QW_KEY": "DASHSCOPE_API_KEY",
    "AntAngel_KEY": "ANTANGEL_API_KEY",
    "BAI_CHUAN_KEY": "BAICHUAN_API_KEY",
    "DeepSeek_KEY": "DEEPSEEK_API_KEY",
    "OPEN_AI_KEY": "OPENAI_API_KEY",
}
RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
DEFAULT_REMOTE_RETRY_ATTEMPTS = 3
DEFAULT_REMOTE_RETRY_BACKOFF_SECONDS = 1.0
_LOADED_ENV_FILES: set[str] = set()


@dataclass
class ModelBundle:
    processor: Any
    model: Any
    base_model_source: str
    adapter_path: Optional[str] = None


def _parse_env_value(raw_value: str) -> str:
    value = raw_value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def load_project_env(
    *,
    project_root: Optional[Union[str, Path]] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    force: bool = False,
) -> None:
    env_mapping = environ if environ is not None else os.environ
    env_file = (Path(project_root or PROJECT_ROOT).expanduser().resolve() / ".env").resolve()
    cache_key = str(env_file)
    if env_mapping is os.environ and not force and cache_key in _LOADED_ENV_FILES:
        return
    if not env_file.exists():
        if env_mapping is os.environ:
            _LOADED_ENV_FILES.add(cache_key)
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
        if value and not clean_text(env_mapping.get(key, "")):
            env_mapping[key] = value

    for legacy_key, standard_key in LEGACY_ENV_ALIASES.items():
        if clean_text(env_mapping.get(standard_key, "")):
            continue
        alias_value = clean_text(env_mapping.get(legacy_key, "")) or clean_text(parsed_values.get(legacy_key, ""))
        if alias_value:
            env_mapping[standard_key] = alias_value

    if env_mapping is os.environ:
        _LOADED_ENV_FILES.add(cache_key)


def build_http_timeout(timeout_seconds: Optional[float]) -> httpx.Timeout:
    resolved_timeout = max(float(timeout_seconds or 0.0), 0.0)
    return httpx.Timeout(
        connect=10.0,
        read=max(60.0, resolved_timeout or 60.0),
        write=20.0,
        pool=max(60.0, resolved_timeout or 60.0),
    )


def build_http_client(timeout_seconds: float) -> httpx.Client:
    load_project_env()
    proxy = (
        os.getenv("DASHSCOPE_PROXY")
        or os.getenv("OPENAI_PROXY")
        or os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("https_proxy")
        or os.getenv("http_proxy")
    )
    if proxy:
        return httpx.Client(trust_env=False, proxy=proxy, timeout=build_http_timeout(timeout_seconds))
    return httpx.Client(trust_env=False, timeout=build_http_timeout(timeout_seconds))


def resolve_api_key(*, model_name: str, api_key_env: Optional[str], api_key: Optional[str]) -> str:
    load_project_env()
    resolved_api_key = clean_text(api_key)
    env_candidates: List[str] = []
    if clean_text(api_key_env):
        env_candidates.append(clean_text(api_key_env))
    if "OPENAI_API_KEY" not in env_candidates:
        env_candidates.append("OPENAI_API_KEY")

    if not resolved_api_key:
        for env_name in env_candidates:
            resolved_api_key = clean_text(os.getenv(env_name))
            if resolved_api_key:
                break
    if not resolved_api_key:
        raise RuntimeError(
            f"未找到 {model_name} 的 API key，请设置 {api_key_env or 'OPENAI_API_KEY'} 或直接在配置中提供 api_key。"
        )
    return resolved_api_key


def _collect_text_segments(content: Any, text_parts: List[str]) -> None:
    if content is None:
        return
    if isinstance(content, str):
        text = clean_text(content)
        if text:
            text_parts.append(text)
        return
    if isinstance(content, list):
        for item in content:
            _collect_text_segments(item, text_parts)
        return
    if isinstance(content, dict):
        content_type = clean_text(content.get("type", "")).lower()
        if content_type in {"text", "output_text", "input_text", "reasoning"}:
            text = (
                clean_text(content.get("text", ""))
                or clean_text(content.get("content", ""))
                or clean_text(content.get("value", ""))
                or clean_text(content.get("answer", ""))
            )
            if text:
                text_parts.append(text)
        for key in (
            "text",
            "content",
            "output_text",
            "message",
            "messages",
            "choices",
            "delta",
            "reasoning",
            "reasoning_content",
            "output",
            "result",
            "answer",
            "final_answer",
            "value",
            "arguments",
            "parts",
            "data",
            "response",
        ):
            if key in content:
                _collect_text_segments(content.get(key), text_parts)


def extract_text_response_content(content: Any) -> str:
    text_parts: List[str] = []
    _collect_text_segments(content, text_parts)
    return "\n".join(part for part in text_parts if part).strip()


def extract_message_text_payload(message: Any) -> str:
    for attr_name in ("content", "reasoning_content", "output_text", "text", "answer"):
        if hasattr(message, attr_name):
            text = extract_text_response_content(getattr(message, attr_name))
            if text:
                return text
    message_payload = message.model_dump() if hasattr(message, "model_dump") else message
    return extract_text_response_content(message_payload)


def summarize_response_shape(payload: Any) -> str:
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())[:8]
        parts = [f"keys={keys}"]
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                parts.append(f"choice_keys={sorted(str(key) for key in first_choice.keys())[:8]}")
                message = first_choice.get("message")
                if isinstance(message, dict):
                    parts.append(f"message_keys={sorted(str(key) for key in message.keys())[:8]}")
                    if "content" in message:
                        content_value = message.get("content")
                        parts.append(f"message_content_type={type(content_value).__name__}")
                        content_text = extract_text_response_content(content_value)
                        if content_text:
                            parts.append(f"message_content_text_len={len(content_text)}")
                    if "reasoning_content" in message:
                        reasoning_value = message.get("reasoning_content")
                        parts.append(f"message_reasoning_type={type(reasoning_value).__name__}")
                        reasoning_text = extract_text_response_content(reasoning_value)
                        if reasoning_text:
                            parts.append(f"message_reasoning_text_len={len(reasoning_text)}")
        return "; ".join(parts)
    if isinstance(payload, list):
        return f"list(len={len(payload)})"
    return type(payload).__name__


def _extract_status_code(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status
    return None


def format_backend_error(exc: Exception) -> str:
    message = clean_text(str(exc)) or exc.__class__.__name__
    if message.startswith(("timeout:", "connect_error:", "http_status_", "empty_200_response:")):
        return message

    class_name = exc.__class__.__name__.lower()
    if isinstance(exc, httpx.TimeoutException) or "timeout" in class_name:
        return f"timeout: {message or 'Request timed out.'}"
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.NetworkError, httpx.RemoteProtocolError)) or any(
        token in class_name for token in ("connect", "connection", "network", "readerror")
    ):
        return f"connect_error: {message}"
    status_code = _extract_status_code(exc)
    if status_code is not None:
        return f"http_status_{status_code}: {message}"
    return message


def is_retryable_backend_error(exc: Exception) -> bool:
    message = clean_text(str(exc))
    if message.startswith("empty_200_response:"):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.NetworkError, httpx.RemoteProtocolError)):
        return True
    class_name = exc.__class__.__name__.lower()
    if any(token in class_name for token in ("timeout", "connect", "connection", "network", "readerror")):
        return True
    status_code = _extract_status_code(exc)
    return status_code in RETRYABLE_STATUS_CODES if status_code is not None else False


def _sleep_before_retry(attempt_index: int) -> None:
    time.sleep(DEFAULT_REMOTE_RETRY_BACKOFF_SECONDS * (2**attempt_index))


load_project_env()


def merge_nested_dicts(base: Optional[Dict[str, Any]], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base or {})
    for key, value in (overlay or {}).items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = merge_nested_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_local_snapshot(model_id: str, hf_cache_root: Union[str, Path]) -> Optional[str]:
    model_path = Path(model_id).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    if "/" not in model_id:
        return None

    org, name = model_id.split("/", 1)
    cache_dir = Path(hf_cache_root).expanduser() / f"models--{org}--{name}"
    refs_main = cache_dir / "refs" / "main"
    if not refs_main.exists():
        return None

    snapshot_name = refs_main.read_text(encoding="utf-8").strip()
    if not snapshot_name:
        return None

    snapshot_dir = (cache_dir / "snapshots" / snapshot_name).resolve()
    required = ("config.json", "preprocessor_config.json", "tokenizer_config.json")
    if not snapshot_dir.exists() or not all((snapshot_dir / name).exists() for name in required):
        return None
    return str(snapshot_dir)


def resolve_model_source(model_id: Union[str, Path], hf_cache_root: Union[str, Path]) -> str:
    model_text = str(model_id)
    direct_path = Path(model_text).expanduser()
    if direct_path.exists():
        return str(direct_path.resolve())
    local_snapshot = resolve_local_snapshot(model_text, hf_cache_root)
    if local_snapshot:
        return local_snapshot
    return model_text


def safe_processor_from_pretrained(model_source: str, local_files_only: bool) -> Any:
    from transformers import AutoProcessor

    kwargs: Dict[str, Any] = {
        "use_fast": True,
        "local_files_only": local_files_only,
        "fix_mistral_regex": True,
    }
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    try:
        return AutoProcessor.from_pretrained(model_source, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        try:
            return AutoProcessor.from_pretrained(model_source, **kwargs)
        except TypeError:
            kwargs.pop("fix_mistral_regex", None)
            return AutoProcessor.from_pretrained(model_source, **kwargs)


def safe_model_from_pretrained(model_source: str, model_kwargs: Dict[str, Any]) -> Any:
    from transformers import AutoModelForImageTextToText

    kwargs = dict(model_kwargs)
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token

    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    try:
        return AutoModelForImageTextToText.from_pretrained(model_source, **kwargs)
    except TypeError:
        kwargs.pop("token", None)
        try:
            return AutoModelForImageTextToText.from_pretrained(model_source, **kwargs)
        except TypeError:
            if "torch_dtype" in kwargs:
                kwargs["dtype"] = kwargs.pop("torch_dtype")
            return AutoModelForImageTextToText.from_pretrained(model_source, **kwargs)


def get_inference_device(model: Any) -> torch.device:
    try:
        device = getattr(model, "device", None)
        if device is not None and str(device) != "meta":
            return torch.device(device)
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model_bundle(
    *,
    base_model: Union[str, Path],
    adapter_path: Optional[Union[str, Path]],
    hf_cache_root: Union[str, Path],
    attn_implementation: str,
    local_files_only: bool = False,
    processor_source: Optional[Union[str, Path]] = None,
) -> ModelBundle:
    from peft import PeftModel

    base_model_source = resolve_model_source(base_model, hf_cache_root)
    base_is_local = Path(base_model_source).expanduser().exists()
    use_local_files = local_files_only or base_is_local

    processor_candidates = []
    if processor_source is not None:
        processor_candidates.append(resolve_model_source(processor_source, hf_cache_root))
    if adapter_path is not None:
        processor_candidates.append(str(Path(adapter_path).expanduser().resolve()))
    processor_candidates.append(base_model_source)

    processor = None
    processor_error: Optional[Exception] = None
    for candidate in processor_candidates:
        try:
            processor = safe_processor_from_pretrained(
                candidate,
                local_files_only=use_local_files or Path(candidate).expanduser().exists(),
            )
            break
        except Exception as exc:
            processor_error = exc
    if processor is None:
        raise RuntimeError(f"无法加载 processor: {processor_error}")

    model_kwargs: Dict[str, Any] = {
        "dtype": resolve_dtype(),
        "local_files_only": use_local_files,
    }
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        model_kwargs["attn_implementation"] = attn_implementation

    model = safe_model_from_pretrained(base_model_source, model_kwargs)

    resolved_adapter_path = None
    if adapter_path is not None:
        resolved_adapter_path = str(Path(adapter_path).expanduser().resolve())
        model = PeftModel.from_pretrained(
            model,
            resolved_adapter_path,
            is_trainable=False,
            local_files_only=True,
        )

    model.eval()
    if not torch.cuda.is_available():
        model.to("cpu")

    try:
        processor.tokenizer.padding_side = "right"
    except Exception:
        pass

    return ModelBundle(
        processor=processor,
        model=model,
        base_model_source=base_model_source,
        adapter_path=resolved_adapter_path,
    )


class MedGemmaTextModel:
    def __init__(
        self,
        *,
        base_model: Union[str, Path],
        adapter_path: Optional[Union[str, Path]],
        hf_cache_root: Union[str, Path],
        attn_implementation: str,
        local_files_only: bool,
        max_length: int,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.hf_cache_root = hf_cache_root
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only
        self.max_length = max_length
        self._bundle: Optional[ModelBundle] = None
        self.provider = "local_medgemma_text"

    def describe(self) -> str:
        model_text = str(self.base_model)
        adapter_text = f"+adapter={self.adapter_path}" if self.adapter_path else ""
        return f"{self.provider}:{model_text}{adapter_text}"

    def load(self) -> ModelBundle:
        if self._bundle is None:
            self._bundle = load_model_bundle(
                base_model=self.base_model,
                adapter_path=self.adapter_path,
                hf_cache_root=self.hf_cache_root,
                attn_implementation=self.attn_implementation,
                local_files_only=self.local_files_only,
            )
        return self._bundle

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_length: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        del request_timeout_seconds
        del response_format
        bundle = self.load()
        prompt_text = clean_text(system_prompt)
        user_text = clean_text(user_prompt)
        if prompt_text:
            combined_text = f"{prompt_text}\n\n{user_text}".strip()
        else:
            combined_text = user_text

        messages = [{"role": "user", "content": [{"type": "text", "text": combined_text}]}]
        prompt = bundle.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = bundle.processor(
            text=prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length or self.max_length,
        )
        target_device = get_inference_device(bundle.model)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        pad_token_id = bundle.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = bundle.processor.tokenizer.eos_token_id

        do_sample = temperature is not None and temperature > 0
        with torch.inference_mode():
            outputs = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                eos_token_id=bundle.processor.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return bundle.processor.decode(generated_tokens, skip_special_tokens=True).strip()

    def close(self) -> None:
        if self._bundle is None:
            return
        del self._bundle.model
        del self._bundle.processor
        self._bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenAICompatibleTextModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: Optional[str],
        api_key_env: Optional[str],
        api_key: Optional[str],
        timeout_seconds: float,
        default_extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        from openai import OpenAI

        self.provider = "openai_compatible"
        self.model_name = clean_text(model_name)
        self.base_url = clean_text(base_url)
        self._default_extra_body = merge_nested_dicts(None, default_extra_body)
        self._rich_text_messages = "baichuan-ai.com" in self.base_url
        self._merge_system_prompt_into_user = self._rich_text_messages
        self._http_client = build_http_client(timeout_seconds)
        self._client = OpenAI(
            api_key=resolve_api_key(model_name=self.model_name, api_key_env=api_key_env, api_key=api_key),
            base_url=self.base_url or None,
            http_client=self._http_client,
        )

    def describe(self) -> str:
        base_url = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model_name}{base_url}"

    def _message_content(self, text: str) -> Any:
        normalized = clean_text(text)
        if self._rich_text_messages:
            return [{"type": "text", "text": normalized}]
        return normalized

    def _normalize_messages(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            normalized_message = dict(message)
            role = clean_text(normalized_message.get("role", ""))
            if "content" in normalized_message and isinstance(normalized_message["content"], str):
                if role in {"system", "user"}:
                    normalized_message["content"] = self._message_content(normalized_message["content"])
            normalized_messages.append(normalized_message)
        return normalized_messages

    def _normalize_tool_calls(self, message_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_calls = message_payload.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            return []
        normalized_calls: List[Dict[str, Any]] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function_payload = tool_call.get("function", {}) if isinstance(tool_call.get("function", {}), dict) else {}
            normalized_calls.append(
                {
                    "id": clean_text(tool_call.get("id", "")),
                    "type": clean_text(tool_call.get("type", "")) or "function",
                    "name": clean_text(function_payload.get("name", "")),
                    "arguments": clean_text(function_payload.get("arguments", "")),
                }
            )
        return normalized_calls

    def complete_chat(
        self,
        *,
        messages: Sequence[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        request_timeout_seconds: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": self._normalize_messages(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        runtime_extra_body: Dict[str, Any] = {}
        if repetition_penalty != 1.0:
            runtime_extra_body["repetition_penalty"] = repetition_penalty
        if self._rich_text_messages:
            runtime_extra_body["metadata"] = {"evidence_scope": "grounded"}
        extra_body = merge_nested_dicts(self._default_extra_body, runtime_extra_body)
        extra_body = merge_nested_dicts(extra_body, extra_body_override)
        if extra_body:
            kwargs["extra_body"] = extra_body

        request_timeout = build_http_timeout(request_timeout_seconds)
        last_error: Optional[Exception] = None
        for attempt_index in range(DEFAULT_REMOTE_RETRY_ATTEMPTS):
            try:
                completion = self._client.chat.completions.create(timeout=request_timeout, **kwargs)
                completion_payload = completion.model_dump() if hasattr(completion, "model_dump") else {}
                choices = getattr(completion, "choices", None) or []
                if not choices:
                    raise RuntimeError(f"empty_200_response: {summarize_response_shape(completion_payload)}")
                message = choices[0].message
                message_payload = message.model_dump(exclude_none=True) if hasattr(message, "model_dump") else {}
                text = extract_message_text_payload(message) or extract_text_response_content(message_payload)
                return {
                    "text": clean_text(text),
                    "message": message_payload,
                    "tool_calls": self._normalize_tool_calls(message_payload),
                    "raw": completion_payload,
                }
            except Exception as exc:
                last_error = exc
                if attempt_index < DEFAULT_REMOTE_RETRY_ATTEMPTS - 1 and is_retryable_backend_error(exc):
                    _sleep_before_retry(attempt_index)
                    continue
                raise RuntimeError(format_backend_error(exc)) from exc

        raise RuntimeError(format_backend_error(last_error or RuntimeError("empty_200_response: no completion body")))

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_length: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        del max_length

        messages: List[Dict[str, Any]] = []
        merged_user_prompt = clean_text(user_prompt)
        if clean_text(system_prompt):
            if self._merge_system_prompt_into_user:
                merged_user_prompt = f"{clean_text(system_prompt)}\n\n{merged_user_prompt}".strip()
            else:
                messages.append({"role": "system", "content": self._message_content(system_prompt)})
        messages.append({"role": "user", "content": self._message_content(merged_user_prompt)})

        response = self.complete_chat(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            request_timeout_seconds=request_timeout_seconds,
            response_format=response_format,
        )
        if response.get("tool_calls"):
            raise RuntimeError("empty_200_response: received tool calls for plain text generation")
        text = clean_text(response.get("text", ""))
        if text:
            return text
        raise RuntimeError("empty_200_response: no text content")

    def close(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass


class AntAngelTextModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: Optional[str],
        api_key_env: Optional[str],
        api_key: Optional[str],
        timeout_seconds: float,
    ) -> None:
        self.provider = "antangel"
        self.model_name = clean_text(model_name)
        self.base_url = clean_text(base_url)
        self._http_client = build_http_client(timeout_seconds)
        self._api_key = resolve_api_key(model_name=self.model_name, api_key_env=api_key_env, api_key=api_key)

    def describe(self) -> str:
        base_url = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model_name}{base_url}"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_length: Optional[int] = None,
        request_timeout_seconds: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        del repetition_penalty
        del max_length

        merged_user_prompt = clean_text(user_prompt)
        if clean_text(system_prompt):
            merged_user_prompt = f"{clean_text(system_prompt)}\n\n{merged_user_prompt}".strip()
        messages: List[Dict[str, str]] = [{"role": "user", "content": merged_user_prompt}]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        last_error: Optional[Exception] = None
        for attempt_index in range(DEFAULT_REMOTE_RETRY_ATTEMPTS):
            try:
                response = self._http_client.post(
                    self.base_url,
                    headers={
                        "Authorization": self._api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=build_http_timeout(request_timeout_seconds),
                )
                response.raise_for_status()
                body = response.json()
                text = extract_text_response_content(body)
                if text:
                    return text
                raise RuntimeError(f"empty_200_response: {summarize_response_shape(body)}")
            except Exception as exc:
                last_error = exc
                if attempt_index < DEFAULT_REMOTE_RETRY_ATTEMPTS - 1 and is_retryable_backend_error(exc):
                    _sleep_before_retry(attempt_index)
                    continue
                raise RuntimeError(format_backend_error(exc)) from exc

        raise RuntimeError(format_backend_error(last_error or RuntimeError("empty_200_response: no response body")))

    def close(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass


def image_path_to_data_url(image_path: Union[str, Path]) -> str:
    path = Path(image_path).expanduser().resolve()
    mime_type, _ = mimetypes.guess_type(str(path))
    resolved_mime_type = mime_type or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{resolved_mime_type};base64,{encoded}"


class LocalMedGemmaMultimodalModel:
    def __init__(
        self,
        *,
        base_model: Union[str, Path],
        adapter_path: Optional[Union[str, Path]],
        hf_cache_root: Union[str, Path],
        attn_implementation: str,
        local_files_only: bool,
        max_length: int,
    ) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.hf_cache_root = hf_cache_root
        self.attn_implementation = attn_implementation
        self.local_files_only = local_files_only
        self.max_length = max_length
        self._bundle: Optional[ModelBundle] = None
        self.provider = "local_medgemma_multimodal"

    def describe(self) -> str:
        model_text = str(self.base_model)
        adapter_text = f"+adapter={self.adapter_path}" if self.adapter_path else ""
        return f"{self.provider}:{model_text}{adapter_text}"

    def load(self) -> ModelBundle:
        if self._bundle is None:
            self._bundle = load_model_bundle(
                base_model=self.base_model,
                adapter_path=self.adapter_path,
                hf_cache_root=self.hf_cache_root,
                attn_implementation=self.attn_implementation,
                local_files_only=self.local_files_only,
                processor_source=self.adapter_path,
            )
        return self._bundle

    def estimate_max_images(self, *, max_length: int, context_reserve_tokens: int) -> int:
        bundle = self.load()
        image_seq_length = max(1, int(getattr(bundle.processor, "image_seq_length", 256)))
        return max(1, (max_length - context_reserve_tokens) // image_seq_length)

    def generate(
        self,
        *,
        prompt_text: str,
        images: Sequence[Any],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_length: Optional[int] = None,
    ) -> str:
        bundle = self.load()
        user_content = [{"type": "image"} for _ in images]
        user_content.append({"type": "text", "text": clean_text(prompt_text)})
        messages = [{"role": "user", "content": user_content}]
        prompt = bundle.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = bundle.processor(
            text=prompt,
            images=list(images),
            return_tensors="pt",
            truncation=True,
            max_length=max_length or self.max_length,
        )
        target_device = get_inference_device(bundle.model)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}

        pad_token_id = bundle.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = bundle.processor.tokenizer.eos_token_id

        do_sample = temperature is not None and temperature > 0
        with torch.inference_mode():
            outputs = bundle.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
                top_p=top_p if do_sample else None,
                repetition_penalty=repetition_penalty,
                eos_token_id=bundle.processor.tokenizer.eos_token_id,
                pad_token_id=pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        return bundle.processor.decode(generated_tokens, skip_special_tokens=True).strip()

    def close(self) -> None:
        if self._bundle is None:
            return
        del self._bundle.model
        del self._bundle.processor
        self._bundle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DashScopeMultimodalModel:
    def __init__(
        self,
        *,
        model_name: str,
        base_url: Optional[str],
        api_key_env: Optional[str],
        api_key: Optional[str],
        timeout_seconds: float,
    ) -> None:
        from openai import OpenAI

        self.provider = "dashscope_multimodal"
        self.model_name = clean_text(model_name)
        self.base_url = clean_text(base_url)
        self._http_client = build_http_client(timeout_seconds)
        self._client = OpenAI(
            api_key=resolve_api_key(model_name=self.model_name, api_key_env=api_key_env, api_key=api_key),
            base_url=self.base_url or None,
            http_client=self._http_client,
        )

    def describe(self) -> str:
        base_url = f"@{self.base_url}" if self.base_url else ""
        return f"{self.provider}:{self.model_name}{base_url}"

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: Sequence[Union[str, Path]],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> str:
        messages: List[Dict[str, Any]] = []
        if clean_text(system_prompt):
            messages.append({"role": "system", "content": clean_text(system_prompt)})

        user_content: List[Dict[str, Any]] = []
        for image_path in image_paths:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(image_path)},
                }
            )
        user_content.append({"type": "text", "text": clean_text(user_prompt)})
        messages.append({"role": "user", "content": user_content})

        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if repetition_penalty != 1.0:
            kwargs["extra_body"] = {"repetition_penalty": repetition_penalty}

        completion = self._client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        return extract_text_response_content(getattr(message, "content", ""))

    def close(self) -> None:
        try:
            self._http_client.close()
        except Exception:
            pass


def build_text_backend(
    *,
    backend_config: ModelBackendConfig,
    hf_cache_root: Union[str, Path],
    attn_implementation: str,
    local_files_only: bool,
    max_length: int,
    timeout_seconds: float,
) -> Union[MedGemmaTextModel, OpenAICompatibleTextModel, AntAngelTextModel]:
    provider = clean_text(backend_config.provider).lower()
    resolved_timeout_seconds = max(float(timeout_seconds or 0.0), float(backend_config.default_timeout_seconds or 0.0))
    if provider == "openai_compatible":
        return OpenAICompatibleTextModel(
            model_name=str(backend_config.model),
            base_url=backend_config.base_url,
            api_key_env=backend_config.api_key_env,
            api_key=backend_config.api_key,
            timeout_seconds=resolved_timeout_seconds,
            default_extra_body=backend_config.extra_body,
        )
    if provider == "antangel":
        return AntAngelTextModel(
            model_name=str(backend_config.model),
            base_url=backend_config.base_url,
            api_key_env=backend_config.api_key_env,
            api_key=backend_config.api_key,
            timeout_seconds=resolved_timeout_seconds,
        )
    if provider in {"local_medgemma_text", "local_text"}:
        return MedGemmaTextModel(
            base_model=backend_config.model,
            adapter_path=backend_config.adapter_path,
            hf_cache_root=hf_cache_root,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            max_length=max_length,
        )
    raise ValueError(f"不支持的文本后端 provider: {backend_config.provider}")


def build_vision_backend(
    *,
    backend_config: ModelBackendConfig,
    hf_cache_root: Union[str, Path],
    attn_implementation: str,
    local_files_only: bool,
    max_length: int,
    timeout_seconds: float,
) -> Union[LocalMedGemmaMultimodalModel, DashScopeMultimodalModel]:
    provider = clean_text(backend_config.provider).lower()
    if provider == "dashscope_multimodal":
        return DashScopeMultimodalModel(
            model_name=str(backend_config.model),
            base_url=backend_config.base_url,
            api_key_env=backend_config.api_key_env,
            api_key=backend_config.api_key,
            timeout_seconds=timeout_seconds,
        )
    if provider in {"local_medgemma_multimodal", "local_multimodal"}:
        return LocalMedGemmaMultimodalModel(
            base_model=backend_config.model,
            adapter_path=backend_config.adapter_path,
            hf_cache_root=hf_cache_root,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
            max_length=max_length,
        )
    raise ValueError(f"不支持的影像后端 provider: {backend_config.provider}")
