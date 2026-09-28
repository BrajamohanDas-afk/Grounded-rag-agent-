"""
DAY 2 — LLM wrapper
===================
One tiny interface: `complete(system, user) -> str`. Swapping providers
(Anthropic, OpenAI, local Ollama, Groq) is then a config change, not a rewrite.
"""
from __future__ import annotations

import json
import os
import re


class LLM:
    def complete(self, system: str, user: str, max_tokens: int = 1500) -> str:
        raise NotImplementedError

    def complete_json(self, system: str, user: str, max_tokens: int = 1500, retries: int = 2) -> dict:
        """Ask for JSON, strip code fences, and retry with the parse error if it fails."""
        last_err = ""
        for _ in range(retries + 1):
            prompt = user if not last_err else (
                f"{user}\n\nYour previous reply was not valid JSON ({last_err}). "
                "Reply with ONLY the JSON object, no prose, no code fences.")
            raw = self.complete(system, prompt, max_tokens)
            try:
                return parse_json(raw)
            except ValueError as e:
                last_err = str(e)
        raise ValueError(f"LLM did not return valid JSON: {last_err}")


def parse_json(raw: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(str(e)) from e


class AnthropicLLM(LLM):
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    def complete(self, system, user, max_tokens=1500):
        resp = self.client.messages.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")


class OpenAICompatLLM(LLM):
    """Works with OpenAI, Groq, Together, and local Ollama (set OPENAI_BASE_URL)."""

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def complete(self, system, user, max_tokens=1500):
        resp = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class MockLLM(LLM):
    """Scripted replies for unit tests — lets you test the pipeline with no API key."""

    def __init__(self, replies: list[str] | None = None):
        self.replies = list(replies or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system, user, max_tokens=1500):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else '{"action": "answer"}'


def get_llm(provider: str | None = None) -> LLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    return {"anthropic": AnthropicLLM, "openai": OpenAICompatLLM, "mock": MockLLM}[provider]()
