"""Model routing helpers for mixed remote-agent/local-judge runs."""
from __future__ import annotations

import os
from contextlib import contextmanager

from inspect_ai.model import get_model


@contextmanager
def _temporary_ollama_env(model_name: str):
    old_base = os.environ.get("OPENAI_BASE_URL")
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.environ.get(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
    )
    # Ollama's OpenAI-compatible endpoint accepts any non-empty key.
    os.environ["OPENAI_API_KEY"] = "ollama"
    try:
        yield get_model("openai/" + model_name)
    finally:
        if old_base is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = old_base
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key


def get_routed_model(spec: str):
    """Create a model from a provider spec.

    ``ollama/<name>`` is routed through Ollama's local OpenAI-compatible API;
    all other specs retain the normal Inspect AI provider behavior.
    """
    if spec.startswith("ollama/"):
        name = spec[len("ollama/"):].strip()
        if not name:
            raise ValueError("ollama model spec must be ollama/<model-name>")
        with _temporary_ollama_env(name) as model:
            return model
    return get_model(spec)


def evaluation_model_spec(spec: str) -> str:
    """Return the provider spec passed to inspect_eval for the main Agent."""
    if spec.startswith("ollama/"):
        return "openai/" + spec[len("ollama/"):].strip()
    return spec


@contextmanager
def evaluation_model_env(spec: str):
    """Temporarily route inspect_eval's main model through local Ollama."""
    if not spec.startswith("ollama/"):
        yield
        return
    old_base = os.environ.get("OPENAI_BASE_URL")
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_BASE_URL"] = os.environ.get(
        "OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
    )
    os.environ["OPENAI_API_KEY"] = "ollama"
    try:
        yield
    finally:
        if old_base is None:
            os.environ.pop("OPENAI_BASE_URL", None)
        else:
            os.environ["OPENAI_BASE_URL"] = old_base
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key
