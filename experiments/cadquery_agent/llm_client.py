"""Real OpenAI-compatible LLM client with provider-aware accounting."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests


PROVIDERS = {
    "qwen": {
        "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "key_env": "QWEN_API_KEY",
    },
    "deepseek": {
        "api_url": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
    },
    "siliconflow": {
        "api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "ollama": {
        "api_url": "http://127.0.0.1:11434/api/chat",
        "key_env": None,
    },
}


@dataclass
class LLMResult:
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    finish_reason: str
    reasoning_tokens: int
    thinking_mode: str
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class LLMRequestError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict]):
        super().__init__(message)
        self.attempts = attempts

    def to_dict(self) -> dict:
        return {"success": False, "error": str(self), "attempts": list(self.attempts)}


class OpenAICompatibleClient:
    def __init__(
        self, *, provider: str, model: str, timeout: int = 240,
        max_attempts: int = 4, seed: int | None = None,
    ):
        if provider not in PROVIDERS:
            raise ValueError(f"unsupported provider: {provider}")
        config = PROVIDERS[provider]
        self.provider = provider
        self.model = model
        self.api_url = os.getenv(f"{provider.upper()}_API_URL", config["api_url"]).strip()
        key_env = config["key_env"]
        self.credential_env = key_env
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", "900")) if provider == "ollama" else timeout
        self.max_attempts = max_attempts
        if seed is not None and not 0 <= int(seed) <= 2**31 - 1:
            raise ValueError("seed_must_be_between_0_and_2^31_minus_1")
        self.seed = None if seed is None else int(seed)

    def cache_identity(self) -> dict:
        identity = {
            "provider": self.provider, "model": self.model,
            "api_url": self.api_url,
            "client_version": "openai_compatible_v5_multimodal_context_fingerprinted",
            "max_attempts": self.max_attempts, "timeout": self.timeout,
            "seed": self.seed,
        }
        if self.provider == "ollama":
            identity["ollama_generation"] = {
                "request_num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "16384")),
                "request_max_tokens": int(os.getenv("OLLAMA_MAX_TOKENS", "4096")),
                "thinking_enabled": os.getenv("OLLAMA_THINK", "false").strip().lower()
                in {"1", "true", "yes", "on"},
                "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
                "server_context_length": int(
                    os.getenv("OLLAMA_SERVER_CONTEXT_LENGTH", "0")
                ),
                "server_num_parallel": int(
                    os.getenv("OLLAMA_SERVER_NUM_PARALLEL", "0")
                ),
                "model_context_length": int(
                    os.getenv("OLLAMA_MODEL_CONTEXT_LENGTH", "0")
                ),
            }
        return identity

    def complete(
        self, system: str, user: str, max_tokens: int = 2600,
        timeout_seconds: float | None = None,
    ) -> LLMResult:
        if self.provider == "ollama":
            return self._complete_ollama(system, user, max_tokens, timeout_seconds)
        return self._complete_remote(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def complete_multimodal(
        self, system: str, user: str, image_paths: list[str | Path],
        max_tokens: int = 2600, timeout_seconds: float | None = None,
    ) -> LLMResult:
        """Call a remote vision-language endpoint with local images as data URIs."""
        if self.provider == "ollama":
            raise ValueError("multimodal_local_images_not_supported_for_ollama")
        if not image_paths:
            return self.complete(system, user, max_tokens, timeout_seconds)
        content: list[dict] = [{"type": "text", "text": user}]
        for raw_path in image_paths:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"multimodal_image_missing:{path}")
            mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
            if not mime_type.startswith("image/"):
                raise ValueError(f"multimodal_file_not_image:{path.name}")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            })
        return self._complete_remote(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )

    def _complete_remote(
        self, messages: list[dict], *, max_tokens: int,
        timeout_seconds: float | None,
    ) -> LLMResult:
        if self.credential_env and not os.environ.get(self.credential_env, "").strip():
            attempt = {
                "attempt": 1, "provider": self.provider, "model": self.model,
                "success": False, "latency_s": 0.0,
                "error_type": "MissingCredential",
                "error": f"{self.credential_env} is required; no key is stored in source code",
            }
            raise LLMRequestError(attempt["error"], [attempt])
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.provider in {"qwen", "siliconflow"}:
            payload["enable_thinking"] = False
            if self.seed is not None:
                payload["seed"] = self.seed
        last_error: Exception | None = None
        attempts: list[dict] = []
        deadline = None if timeout_seconds is None else time.perf_counter() + max(0.0, timeout_seconds)
        for attempt in range(self.max_attempts):
            remaining = self.timeout if deadline is None else min(self.timeout, deadline - time.perf_counter())
            if remaining <= 0:
                break
            started = time.perf_counter()
            try:
                headers = {"Content-Type": "application/json"}
                if self.credential_env:
                    headers["Authorization"] = "Bearer " + os.environ[self.credential_env]
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=max(0.05, remaining),
                )
                response.raise_for_status()
                body = response.json()
                choice = body["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                finish_reason = str(choice.get("finish_reason") or "unknown")
                if not content.strip():
                    raise RuntimeError(f"empty content (finish_reason={finish_reason})")
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                attempts.append({
                    "attempt": attempt + 1, "provider": self.provider, "model": self.model,
                    "success": True, "latency_s": time.perf_counter() - started,
                })
                return LLMResult(
                    content=content,
                    model=str(body.get("model") or self.model),
                    provider=self.provider,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    total_tokens=int(usage.get("total_tokens", 0)),
                    latency_s=time.perf_counter() - started,
                    finish_reason=finish_reason,
                    reasoning_tokens=int(details.get("reasoning_tokens", 0)),
                    thinking_mode="provider_default",
                    attempts=attempts,
                )
            except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError) as error:
                last_error = error
                attempts.append({
                    "attempt": attempt + 1, "provider": self.provider, "model": self.model,
                    "success": False, "latency_s": time.perf_counter() - started,
                    "error_type": type(error).__name__, "error": str(error),
                })
                if attempt + 1 < self.max_attempts:
                    delay = float(2**attempt)
                    if deadline is not None:
                        delay = min(delay, max(0.0, deadline - time.perf_counter()))
                    if delay > 0:
                        time.sleep(delay)
        raise LLMRequestError(f"{self.provider} request failed after retries: {last_error}", attempts)

    def _complete_ollama(
        self, system: str, user: str, max_tokens: int, timeout_seconds: float | None = None,
    ) -> LLMResult:
        num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
        output_cap = int(os.getenv("OLLAMA_MAX_TOKENS", "4096"))
        model_context = int(os.getenv("OLLAMA_MODEL_CONTEXT_LENGTH", str(num_ctx)))
        if num_ctx > model_context:
            raise LLMRequestError(
                f"ollama_num_ctx_exceeds_model_context:{num_ctx}>{model_context}",
                [{
                    "attempt": 0, "provider": self.provider, "model": self.model,
                    "success": False, "latency_s": 0.0,
                    "error_type": "ContextContractError",
                    "error": f"request_num_ctx={num_ctx}, model_context={model_context}",
                }],
            )
        thinking_enabled = os.getenv("OLLAMA_THINK", "false").strip().lower() in {
            "1", "true", "yes", "on",
        }
        options = {
            "temperature": 0,
            "num_ctx": num_ctx,
            "num_predict": min(max_tokens, output_cap),
        }
        if self.seed is not None:
            options["seed"] = self.seed
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": thinking_enabled,
            "options": options,
            "keep_alive": os.getenv("OLLAMA_KEEP_ALIVE", "15m"),
        }
        last_error: Exception | None = None
        attempts: list[dict] = []
        deadline = None if timeout_seconds is None else time.perf_counter() + max(0.0, timeout_seconds)
        for attempt in range(self.max_attempts):
            remaining = self.timeout if deadline is None else min(self.timeout, deadline - time.perf_counter())
            if remaining <= 0:
                break
            started = time.perf_counter()
            try:
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=max(0.05, remaining),
                )
                response.raise_for_status()
                body = response.json()
                message = body.get("message") or {}
                content = message.get("content") or ""
                finish_reason = str(body.get("done_reason") or "unknown")
                if not content.strip():
                    raise RuntimeError(f"empty content (finish_reason={finish_reason})")
                prompt_tokens = int(body.get("prompt_eval_count", 0))
                completion_tokens = int(body.get("eval_count", 0))
                attempts.append({
                    "attempt": attempt + 1, "provider": self.provider, "model": self.model,
                    "success": True, "latency_s": time.perf_counter() - started,
                })
                return LLMResult(
                    content=content,
                    model=str(body.get("model") or self.model),
                    provider=self.provider,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    latency_s=time.perf_counter() - started,
                    finish_reason=finish_reason,
                    reasoning_tokens=0,
                    thinking_mode="enabled" if thinking_enabled else "disabled",
                    attempts=attempts,
                )
            except (requests.RequestException, KeyError, TypeError, ValueError, RuntimeError) as error:
                last_error = error
                attempts.append({
                    "attempt": attempt + 1, "provider": self.provider, "model": self.model,
                    "success": False, "latency_s": time.perf_counter() - started,
                    "error_type": type(error).__name__, "error": str(error),
                })
                if attempt + 1 < self.max_attempts:
                    delay = float(2**attempt)
                    if deadline is not None:
                        delay = min(delay, max(0.0, deadline - time.perf_counter()))
                    if delay > 0:
                        time.sleep(delay)
        raise LLMRequestError(f"ollama request failed after retries: {last_error}", attempts)
