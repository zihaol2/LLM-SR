
from __future__ import annotations

import dataclasses
from typing import Type
import os

import sampler
import evaluator


@dataclasses.dataclass(frozen=True)
class ExperienceBufferConfig:
    functions_per_prompt: int = 3
    num_islands: int = 10
    reset_period: int = 4 * 60 * 60
    cluster_sampling_temperature_init: float = 0.1
    cluster_sampling_temperature_period: int = 30_000


@dataclasses.dataclass(frozen=True)
class Config:
    experience_buffer: ExperienceBufferConfig = dataclasses.field(default_factory=ExperienceBufferConfig)
    num_samplers: int = 1
    num_evaluators: int = 1
    samples_per_prompt: int = 4
    # Wall-clock budget for a single sandbox evaluation. Coupled systems
    # (e.g. Predator_Prey: 2 fields, MAX_NPARAMS=15) legitimately need 30-45 s,
    # so the original 30 s silently killed most of their candidates.
    evaluate_timeout_seconds: int = 60
    use_api: bool = False
    api_model: str = "gpt-3.5-turbo"
    # Reasoning models (e.g. deepseek-flash) spend part of the budget on the
    # chain of thought, so a small max_tokens yields a truncated/empty answer.
    max_tokens: int = 16384
    # Empty string keeps the provider default (DeepSeek: high).
    reasoning_effort: str = ""
    # Empty string keeps the provider default (DeepSeek: enabled).
    # Set to "disabled" to skip the chain of thought (much faster and cheaper).
    thinking: str = ""
    # Any OpenAI-compatible base URL, e.g. https://api.teamorouter.cn/v1 .
    # Empty string falls back to the built-in DeepSeek / Zhipu endpoints.
    api_base_url: str = ""
    # Name of the environment variable holding the API key.
    api_key_env: str = "API_KEY"
    # How many of the samples_per_prompt requests to issue in parallel.
    concurrent_requests: int = 4


@dataclasses.dataclass()
class ClassConfig:
    llm_class: Type[sampler.LLM]
    sandbox_class: Type[evaluator.Sandbox]

