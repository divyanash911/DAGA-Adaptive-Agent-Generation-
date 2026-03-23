"""
DAGA — Backend Abstraction Layer
Provides a unified interface for:
  - Local models via vLLM or Ollama
  - Remote models via OpenAI-compatible APIs (Anthropic, OpenAI, Together, etc.)

Each backend returns (text, input_tokens, output_tokens, latency_s).
Energy is estimated post-hoc by the telemetry layer using per-model Joules/token
profiles (from empirical benchmarks such as TokenPowerBench and our own records).
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Optional imports — only required if the respective backend is used
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import openai
except ImportError:
    openai = None  # type: ignore


# ──────────────────────────────────────────────
# Energy profile per model tier (J/token, empirical)
# Sources: TokenPowerBench, Wilhelm et al. 2025, our estimates
# ──────────────────────────────────────────────

from daga.core.models import ModelTier

MODEL_ENERGY_PROFILE: Dict[ModelTier, Dict[str, float]] = {
    ModelTier.SLM_NANO:     {"j_per_input_token": 0.000018, "j_per_output_token": 0.000025},
    ModelTier.SLM_SMALL:    {"j_per_input_token": 0.00012,  "j_per_output_token": 0.00018},
    ModelTier.LLM_MEDIUM:   {"j_per_input_token": 0.00045,  "j_per_output_token": 0.00070},
    ModelTier.LLM_LARGE:    {"j_per_input_token": 0.00120,  "j_per_output_token": 0.00190},
    ModelTier.LLM_FRONTIER: {"j_per_input_token": 0.00400,  "j_per_output_token": 0.00600},
}


def estimate_energy(
    input_tokens: int,
    output_tokens: int,
    tier: ModelTier,
) -> float:
    """Estimate energy consumed by one inference call (Joules)."""
    profile = MODEL_ENERGY_PROFILE[tier]
    return (
        input_tokens  * profile["j_per_input_token"]
        + output_tokens * profile["j_per_output_token"]
    )


# ──────────────────────────────────────────────
# Response container
# ──────────────────────────────────────────────

@dataclass
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    energy_j: float          # estimated
    model_id: str
    tier: ModelTier
    raw: Optional[Dict[str, Any]] = None  # full API response for debugging


# ──────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────

class ModelBackend(ABC):
    """
    All backends must implement `complete()`.
    Messages follow the OpenAI chat format.
    """

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @property
    @abstractmethod
    def tier(self) -> ModelTier: ...

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> ModelResponse: ...

    def _wrap(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        raw: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        energy = estimate_energy(input_tokens, output_tokens, self.tier)
        return ModelResponse(
            text          = text,
            input_tokens  = input_tokens,
            output_tokens = output_tokens,
            latency_s     = latency_s,
            energy_j      = energy,
            model_id      = self.model_id,
            tier          = self.tier,
            raw           = raw,
        )


# ──────────────────────────────────────────────
# OpenAI-compatible API backend
# (works with OpenAI, Together, Fireworks, Groq,
#  local vLLM --api-server, LM Studio, etc.)
# ──────────────────────────────────────────────

class OpenAICompatibleBackend(ModelBackend):
    """
    Generic backend for any server that exposes POST /v1/chat/completions.
    Set base_url to the endpoint root (e.g. "http://localhost:8000/v1" for vLLM).
    """

    def __init__(
        self,
        model_id: str,
        tier: ModelTier,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self._model_id = model_id
        self._tier     = tier
        self._base_url = base_url.rstrip("/")
        self._api_key  = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._extra_headers = extra_headers or {}

        if httpx is None:
            raise ImportError("Install httpx: pip install httpx")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tier(self) -> ModelTier:
        return self._tier

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }
        payload: Dict[str, Any] = {
            "model":       self._model_id,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if stop:
            payload["stop"] = stop

        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            json    = payload,
            headers = headers,
            timeout = 120.0,
        )

        # On 400: log the response body so we know exactly why OpenRouter rejected
        # (most common causes: context window exceeded, model offline, bad max_tokens)
        if resp.status_code == 400:
            try:
                err_body = resp.json()
            except Exception:
                err_body = resp.text[:400]
            # Truncate message to fit in context window and retry once at half max_tokens
            if isinstance(err_body, dict) and "context_length_exceeded" in str(err_body):
                payload["max_tokens"] = max(256, max_tokens // 2)
                # Also trim oldest non-system messages to reduce context
                user_msgs = [m for m in messages if m["role"] != "system"]
                sys_msgs  = [m for m in messages if m["role"] == "system"]
                if len(user_msgs) > 4:
                    trimmed = sys_msgs + user_msgs[-4:]
                    payload["messages"] = trimmed
                resp = httpx.post(
                    f"{self._base_url}/chat/completions",
                    json=payload, headers=headers, timeout=120.0,
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter 400: {err_body!r}. "
                    f"Model={self._model_id} max_tokens={max_tokens} "
                    f"msg_count={len(messages)} "
                    f"approx_chars={sum(len(m.get('content','')) for m in messages)}"
                )

        resp.raise_for_status()
        latency = time.perf_counter() - t0

        data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        print(f"OpenAI-compatible backend '{self.model_id}' response: {choice}...")
        usage  = data.get("usage", {})
        in_tok  = usage.get("prompt_tokens",     0)
        out_tok = usage.get("completion_tokens", 0)

        return self._wrap(choice, in_tok, out_tok, latency, data)


# ──────────────────────────────────────────────
# Anthropic API backend
# ──────────────────────────────────────────────

class AnthropicBackend(ModelBackend):
    """
    Backend for Anthropic's Claude models via the official API.
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        tier: ModelTier = ModelTier.LLM_FRONTIER,
        api_key: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._tier     = tier
        self._api_key  = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

        if httpx is None:
            raise ImportError("Install httpx: pip install httpx")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tier(self) -> ModelTier:
        return self._tier

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> ModelResponse:
        # Anthropic uses "system" as a top-level field
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model":       self._model_id,
            "messages":    filtered,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop

        t0 = time.perf_counter()
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            json    = payload,
            headers = headers,
            timeout = 180.0,
        )
        resp.raise_for_status()
        latency = time.perf_counter() - t0

        data = resp.json()
        text = data["content"][0]["text"]
        usage   = data.get("usage", {})
        in_tok  = usage.get("input_tokens",  0)
        out_tok = usage.get("output_tokens", 0)

        return self._wrap(text, in_tok, out_tok, latency, data)


# ──────────────────────────────────────────────
# Ollama backend (local, free)
# ──────────────────────────────────────────────

class OllamaBackend(ModelBackend):
    """
    Backend for locally running Ollama models.
    Default endpoint: http://localhost:11435
    """

    def __init__(
        self,
        model_id: str,
        tier: ModelTier,
        base_url: str = "http://localhost:11435",
    ) -> None:
        self._model_id = model_id
        self._tier     = tier
        self._base_url = base_url.rstrip("/")

        if httpx is None:
            raise ImportError("Install httpx: pip install httpx")

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tier(self) -> ModelTier:
        return self._tier

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> ModelResponse:
        payload: Dict[str, Any] = {
            "model":    self._model_id,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if stop:
            payload["options"]["stop"] = stop

        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self._base_url}/api/chat",
            json    = payload,
            timeout = 300.0,
        )
        print(f"Ollama response status: {resp.status_code}")
        # print(f"Ollama response body: {resp.text}")
        # try:
        #     json_body = resp.text.json()
        #     resp_text = json_body["messages"]["content"]
        # except Exception:
        #     resp_text = resp.text
        # print(f"Ollama response content: {resp_text}...")
        # resp.raise_for_status()
        latency = time.perf_counter() - t0
        import html

        data = resp.json()

        text = data.get("message", {}).get("content", "")

        # Fix escaped XML like \u003c → <
        text = html.unescape(text)

        print(f"Ollama response content (parsed): {text}...")

        # data    = resp.json()
        # text    = data["message"]["content"]
        in_tok  = data.get("prompt_eval_count",  0)
        out_tok = data.get("eval_count",         0)

        return self._wrap(text, in_tok, out_tok, latency, data)


# ──────────────────────────────────────────────
# Mock backend (for unit tests / offline dev)
# ──────────────────────────────────────────────

class MockBackend(ModelBackend):
    """
    Returns canned responses with realistic token/latency estimates.
    Useful for testing the planner and router without an inference server.
    """

    def __init__(
        self,
        model_id: str = "mock-7b",
        tier: ModelTier = ModelTier.SLM_SMALL,
        canned_response: str = "MOCK RESPONSE: patch applied.",
        simulated_latency: float = 1.5,
    ) -> None:
        self._model_id          = model_id
        self._tier              = tier
        self._canned            = canned_response
        self._simulated_latency = simulated_latency

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def tier(self) -> ModelTier:
        return self._tier

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> ModelResponse:
        in_tok  = sum(len(m["content"].split()) for m in messages)
        out_tok = len(self._canned.split())
        time.sleep(self._simulated_latency * 0.05)   # fast in tests
        return self._wrap(
            self._canned,
            in_tok,
            out_tok,
            self._simulated_latency,
        )


# ──────────────────────────────────────────────
# Backend registry
# ──────────────────────────────────────────────

class BackendRegistry:
    """
    Central catalog of available model backends.
    The meta-agent queries this to resolve ModelTier → concrete backend.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, ModelBackend] = {}

    def register(self, backend: ModelBackend, aliases: Optional[List[str]] = None) -> None:
        self._backends[backend.model_id] = backend
        for alias in (aliases or []):
            self._backends[alias] = backend

    def get(self, model_id: str) -> ModelBackend:
        if model_id not in self._backends:
            raise KeyError(f"Backend '{model_id}' not registered. "
                           f"Available: {list(self._backends)}")
        return self._backends[model_id]

    def get_by_tier(self, tier: ModelTier) -> List[ModelBackend]:
        return [b for b in self._backends.values() if b.tier == tier]

    def list_all(self) -> List[str]:
        return list(self._backends)

    def cheapest_for_tier(self, tier: ModelTier) -> Optional[ModelBackend]:
        """Returns the first registered backend of the given tier (used as default)."""
        options = self.get_by_tier(tier)
        return options[0] if options else None


def build_default_registry(
    use_mock: bool = False,
    anthropic_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    vllm_url: Optional[str] = None,
    ollama_url: str = "http://localhost:11435",
    openrouter_api_key: Optional[str] = None,
) -> BackendRegistry:
    """
    Convenience factory. Priority:
      1. use_mock=True           -> all mock backends (no network)
      2. openrouter_api_key set  -> all tiers via OpenRouter (recommended)
      3. fallback                -> Ollama SLMs + direct API keys for LLMs
    """
    reg = BackendRegistry()

    if use_mock:
        reg.register(MockBackend("mock-nano",    ModelTier.SLM_NANO,    simulated_latency=0.3),  ["slm_nano"])
        reg.register(MockBackend("mock-small",   ModelTier.SLM_SMALL,   simulated_latency=1.0),  ["slm_small"])
        reg.register(MockBackend("mock-medium",  ModelTier.LLM_MEDIUM,  simulated_latency=3.0),  ["llm_medium"])
        reg.register(MockBackend("mock-large",   ModelTier.LLM_LARGE,   simulated_latency=8.0),  ["llm_large"])
        reg.register(MockBackend("mock-frontier",ModelTier.LLM_FRONTIER, simulated_latency=15.0),["llm_frontier"])
        return reg

    or_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        return build_openrouter_registry(or_key)

    # Ollama (free, local) -- SLM tiers
    try:
        reg.register(OllamaBackend("qwen2.5:7b", ModelTier.SLM_NANO,  base_url=ollama_url), aliases=["slm_nano"])
        reg.register(OllamaBackend("qwen2.5:32b",  ModelTier.LLM_MEDIUM, base_url=ollama_url), aliases=["llm_medium"])
    except Exception:
        pass

    if vllm_url:
        reg.register(OpenAICompatibleBackend("qwen2.5-32b-instruct", ModelTier.LLM_MEDIUM, base_url=vllm_url), aliases=["llm_medium"])

    key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    if key:
        reg.register(AnthropicBackend("claude-sonnet-4-6", ModelTier.LLM_FRONTIER, api_key=key), aliases=["llm_frontier"])

    okey = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if okey:
        reg.register(OpenAICompatibleBackend("gpt-4o-mini", ModelTier.LLM_LARGE, base_url="https://api.openai.com/v1", api_key=okey), aliases=["llm_large"])

    return reg


# ──────────────────────────────────────────────
# OpenRouter registry builder
# ──────────────────────────────────────────────

OPENROUTER_DEFAULT_MODELS: Dict[ModelTier, str] = {
    ModelTier.SLM_NANO:     "qwen/qwen-2.5-1.5b-instruct",
    ModelTier.SLM_SMALL:    "qwen/qwen-2.5-7b-instruct",
    ModelTier.LLM_MEDIUM:   "qwen/qwen-2.5-coder-32b-instruct",
    ModelTier.LLM_LARGE:    "meta-llama/llama-3.1-70b-instruct",
    ModelTier.LLM_FRONTIER: "anthropic/claude-sonnet-4-6",
}


def build_openrouter_registry(
    api_key: str,
    model_map: Optional[Dict[ModelTier, str]] = None,
    site_url: str = "https://github.com/your-org/daga",
    site_name: str = "DAGA",
) -> BackendRegistry:
    """
    Build a registry where every tier is served by OpenRouter.
    OpenRouter exposes an OpenAI-compatible endpoint, so we reuse
    OpenAICompatibleBackend with the extra headers OpenRouter requires.

    Args:
        api_key:   Your OpenRouter API key (sk-or-v1-...).
        model_map: Override the default tier->model mapping.
                   Keys are ModelTier enum values, values are OpenRouter
                   model strings (e.g. "anthropic/claude-sonnet-4-6").
        site_url:  Shown in your OpenRouter dashboard (HTTP-Referer).
        site_name: Shown in your OpenRouter dashboard (X-Title).

    Example::

        from daga.backends.registry import build_openrouter_registry
        from daga.core.models import ModelTier

        registry = build_openrouter_registry(
            api_key = "sk-or-v1-...",
            model_map = {
                ModelTier.SLM_NANO:     "google/gemma-2-9b-it:free",
                ModelTier.SLM_SMALL:    "mistralai/mistral-7b-instruct:free",
                ModelTier.LLM_MEDIUM:   "qwen/qwen-2.5-32b-instruct",
                ModelTier.LLM_LARGE:    "meta-llama/llama-3.3-70b-instruct",
                ModelTier.LLM_FRONTIER: "anthropic/claude-sonnet-4-6",
            },
        )
    """
    resolved = {**OPENROUTER_DEFAULT_MODELS, **(model_map or {})}
    reg      = BackendRegistry()

    extra_headers = {
        "HTTP-Referer": site_url,
        "X-Title":      site_name,
    }

    tier_aliases = {
        ModelTier.SLM_NANO:     "slm_nano",
        ModelTier.SLM_SMALL:    "slm_small",
        ModelTier.LLM_MEDIUM:   "llm_medium",
        ModelTier.LLM_LARGE:    "llm_large",
        ModelTier.LLM_FRONTIER: "llm_frontier",
    }

    for tier, model_id in resolved.items():
        backend = OpenAICompatibleBackend(
            model_id      = model_id,
            tier          = tier,
            base_url      = "https://openrouter.ai/api/v1",
            api_key       = api_key,
            extra_headers = extra_headers,
        )
        reg.register(backend, aliases=[tier_aliases[tier]])

    return reg