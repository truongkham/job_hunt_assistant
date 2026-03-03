from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from crewai import LLM
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)
@dataclass
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2


def load_llm_config(path: str = "configs/llm.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing LLM config file: {path}")
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def get_llm(cfg: LLMConfig, config_path: str = "configs/llm.yaml") -> LLM:
    raw = load_llm_config(config_path)
    providers = raw.get("providers", {})
    provider_cfg = providers.get(cfg.provider)
    if not provider_cfg:
        raise ValueError(f"Unknown provider '{cfg.provider}'. Available: {list(providers.keys())}")

    api_key_env = provider_cfg.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key env var '{api_key_env}' for provider '{cfg.provider}'")

    base_url = None
    if cfg.provider == "openai_compatible":
        base_url_env = provider_cfg.get("base_url_env", "OPENAI_BASE_URL")
        base_url = os.getenv(base_url_env)
        if not base_url:
            raise ValueError(f"Missing base URL env var '{base_url_env}' for openai_compatible provider")

    return LLM(
        model=cfg.model,
        api_key=api_key,
        base_url=base_url,
        temperature=cfg.temperature,
    )


def get_default_llm(config_path: str = "configs/llm.yaml") -> LLM:
    raw = load_llm_config(config_path)
    d = raw.get("default", {}) or {}

    # Optional experiment overrides via environment variables
    override_model = os.getenv("LLM_MODEL")
    override_provider = os.getenv("LLM_PROVIDER")

    cfg = LLMConfig(
        provider=override_provider or d.get("provider", "openai_compatible"),
        model=override_model or d.get("model", "llama-3.1-8b-instant"),
        temperature=float(d.get("temperature", 0.2)),
    )
    return get_llm(cfg, config_path=config_path)
