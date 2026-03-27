"""
model.py — Model router. Swap backends without touching the runner.
Backends: Ollama (local, preferred) | HuggingFace (local) | OpenAI-compatible API
"""

from __future__ import annotations
import os
import json
import time
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Single interface for all model backends.
    Set AGENT_MODEL env var to switch:
      ollama:mistral     → local Ollama
      ollama:llama3      → local Ollama
      hf:mistral-7b      → local HuggingFace
    """

    def __init__(self, model_str: Optional[str] = None):
        self.model_str = model_str or os.environ.get("AGENT_MODEL", "ollama:mistral")
        backend, _, name = self.model_str.partition(":")
        self.backend = backend.lower()
        self.model_name = name or "mistral"
        self._hf_pipeline = None
        logger.info(f"Model router: {self.backend}:{self.model_name}")

    def call(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Call the configured model. Returns response text.
        Emits model.call and model.response hooks.
        """
        from src.agent.hooks import emit, Event

        emit(Event.MODEL_CALL, {
            "backend": self.backend,
            "model": self.model_name,
            "n_messages": len(messages),
            "max_tokens": max_tokens,
        })

        t0 = time.time()
        try:
            if self.backend == "ollama":
                text = self._call_ollama(messages, max_tokens, temperature, stop)
            elif self.backend == "hf":
                text = self._call_hf(messages, max_tokens, temperature)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            emit(Event.MODEL_RESPONSE, {
                "backend": self.backend,
                "model": self.model_name,
                "response_len": len(text),
                "latency": round(time.time() - t0, 3),
            })
            return text

        except Exception as e:
            emit(Event.MODEL_ERROR, {"error": str(e), "backend": self.backend})
            raise

    # ── Ollama ──────────────────────────────────────────────────────────────

    def _call_ollama(self, messages, max_tokens, temperature, stop):
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                **({"stop": stop} if stop else {}),
            },
        }
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"]

    # ── HuggingFace local ────────────────────────────────────────────────────

    def _call_hf(self, messages, max_tokens, temperature):
        if self._hf_pipeline is None:
            from transformers import pipeline
            import torch
            self._hf_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device=0 if __import__("torch").cuda.is_available() else -1,
                torch_dtype=__import__("torch").float16,
            )
        # convert messages to prompt
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        ) + "\nASSISTANT:"
        out = self._hf_pipeline(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        return out[0]["generated_text"][len(prompt):]

