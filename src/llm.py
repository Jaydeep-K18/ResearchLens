"""
Shared LLM access - Gemini Flash primary, Groq as backup.

WHY THIS FILE EXISTS
--------------------
Three separate modules need to call an LLM: basic_rag.py (Phase 2), pipeline.py
(Phase 6) and evaluate.py (Phase 7). Without a shared client, each would repeat
its own key loading, model selection, retry logic and error handling - and they
would drift apart. One place, one behaviour.

WHY TWO PROVIDERS
-----------------
Both free tiers have rate limits. Phase 7 runs 20 questions through 2 systems and
then 40 judge calls - about 60 requests in a burst, which is exactly the shape of
traffic that trips a per-minute quota. When Gemini returns 429, we fall back to
Groq rather than losing the run. If neither is configured, we fail with an
actionable message instead of a stack trace.

A NOTE ON THE SDK
-----------------
PROJECT_CONTEXT.md specifies `google-generativeai`. Google deprecated that
package in favour of `google-genai`, which is what we use. The API shape differs:

    old:  genai.configure(api_key=...); GenerativeModel("...").generate_content(p)
    new:  genai.Client(api_key=...).models.generate_content(model="...", contents=p)

Same service, supported SDK.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import PROJECT_ROOT  # noqa: E402

# Model preference order. If the first is retired or unavailable to your key, the
# client falls through to the next rather than dying - free-tier model names DO
# get rotated, and a retired name returns 404 rather than anything friendlier.
#
# Discover what your own key can reach with:
#     python src/llm.py --list-models
#
# Note that listing is not proof of access: models/gemini-2.5-flash appears in
# the list response but returns
#     404 "no longer available to new users"
# when actually called. Only a real call tells you the truth, which is why this
# is a fallthrough list rather than a single pinned name.
GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
]
GROQ_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

# The SDK's default HTTP timeout is too aggressive for these models: every
# gemini-3.x flash call returned 504 DEADLINE_EXCEEDED at 30s, which looks
# exactly like "this model is broken" but is really "it had not finished
# thinking yet". At 150s the same calls succeed in 3-8s.
HTTP_TIMEOUT_MS = 150_000

# Gemini 3.x models reason before answering, and those reasoning tokens are
# drawn from max_output_tokens. A request with max_output_tokens=300 can spend
# the entire budget thinking and return an EMPTY string - no error, just nothing.
# This floor makes sure there is always room for an actual answer after thinking.
MIN_OUTPUT_TOKENS = 1024

_keys_loaded = False


class LLMUnavailable(RuntimeError):
    """Raised when no provider is configured or all providers failed."""


def load_api_keys() -> None:
    """
    Load .env from the project root, once per process.

    We pass an explicit path rather than relying on python-dotenv's find_dotenv():
    that helper walks up the CALLER'S stack frames, which breaks when the caller
    is a Streamlit script, a pytest fixture, or a `python -c` one-liner.
    """
    global _keys_loaded
    if _keys_loaded:
        return
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
    _keys_loaded = True


def _real_key(name: str) -> str | None:
    """Return the key only if it is actually filled in, not the template placeholder."""
    value = (os.getenv(name) or "").strip()
    if not value or value.lower() in {"your_key_here", "none", "changeme"}:
        return None
    return value


class LLMClient:
    """
    Uniform text-in/text-out interface over Gemini and Groq.

    Usage:
        llm = LLMClient()
        if llm.available:
            answer = llm.generate("Explain self-attention in one sentence.")
    """

    def __init__(self, prefer: str = "gemini", temperature: float = 0.2) -> None:
        load_api_keys()
        self.temperature = temperature
        self.prefer = prefer

        self.gemini_key = _real_key("GEMINI_API_KEY")
        self.groq_key = _real_key("GROQ_API_KEY")

        # Allow explicit override, otherwise discover from the candidate list.
        self.gemini_model = os.getenv("GEMINI_MODEL") or None
        self.groq_model = os.getenv("GROQ_MODEL") or None

        self._gemini_client = None
        self._groq_client = None
        self.last_provider: str | None = None

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.gemini_key or self.groq_key)

    def status(self) -> str:
        parts = [
            f"Gemini: {'configured' if self.gemini_key else 'NOT configured'}",
            f"Groq: {'configured' if self.groq_key else 'NOT configured'}",
        ]
        return " | ".join(parts)

    @staticmethod
    def setup_hint() -> str:
        return (
            "No LLM API key found.\n"
            "  1. Get a free key at https://aistudio.google.com/apikey\n"
            "  2. Open the .env file in the project root\n"
            "  3. Replace 'your_key_here' with your key on the GEMINI_API_KEY line\n"
            "Retrieval (Phases 1-5) works without a key; only answer generation needs one."
        )

    # -- providers ---------------------------------------------------------

    def _gemini(self):
        if self._gemini_client is None:
            from google import genai
            from google.genai import types
            self._gemini_client = genai.Client(
                api_key=self.gemini_key,
                http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS),
            )
        return self._gemini_client

    def _groq(self):
        if self._groq_client is None:
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_key)
        return self._groq_client

    def _generate_gemini(self, prompt: str, temperature: float, max_tokens: int) -> str:
        from google.genai import types

        client = self._gemini()
        candidates = [self.gemini_model] if self.gemini_model else GEMINI_MODEL_CANDIDATES

        last_error: Exception | None = None
        for model_name in candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max(max_tokens, MIN_OUTPUT_TOKENS),
                        # "low" rather than the default: this is a grounded
                        # extraction-and-citation task, not a reasoning puzzle -
                        # the evidence is already in the prompt. Low thinking
                        # more than halves latency (7.2s -> 3.1s per call), which
                        # matters when Phase 7 makes ~80 calls in a run.
                        thinking_config=types.ThinkingConfig(thinking_level="low"),
                    ),
                )
                # Remember the model that worked so later calls skip the probing.
                self.gemini_model = model_name
                text = (response.text or "").strip()
                if text:
                    return text
                raise RuntimeError(
                    "Gemini returned an empty response (thinking may have consumed "
                    "the whole output budget - raise max_tokens)"
                )
            except Exception as exc:  # noqa: BLE001 - we classify below
                message = str(exc).lower()
                last_error = exc
                # A missing/retired model is worth retrying with the next
                # candidate. A quota error is not - it will fail identically.
                if "not found" in message or "404" in message or "unsupported" in message:
                    continue
                raise

        raise last_error or RuntimeError("no Gemini model available")

    def _generate_groq(self, prompt: str, temperature: float, max_tokens: int) -> str:
        client = self._groq()
        candidates = [self.groq_model] if self.groq_model else GROQ_MODEL_CANDIDATES

        last_error: Exception | None = None
        for model_name in candidates:
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self.groq_model = model_name
                return (completion.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                last_error = exc
                if "not found" in message or "decommission" in message or "404" in message:
                    continue
                raise

        raise last_error or RuntimeError("no Groq model available")

    # -- public API --------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int = 2048,
        retries: int = 3,
    ) -> str:
        """
        Send a prompt, get text back.

        Tries the preferred provider, retrying transient failures with
        exponential backoff plus jitter, then falls back to the other provider.

        Why backoff with jitter: free tiers rate-limit per minute. Retrying
        immediately just burns another request against the same window; waiting
        2s, then 4s, then 8s usually lands in the next one. Jitter matters when
        Phase 7 fires many calls in a burst - without it, every retry collides
        again at exactly the same moment.
        """
        if not self.available:
            raise LLMUnavailable(self.setup_hint())

        temperature = self.temperature if temperature is None else temperature

        order = ["gemini", "groq"] if self.prefer == "gemini" else ["groq", "gemini"]
        order = [p for p in order if (p == "gemini" and self.gemini_key) or
                 (p == "groq" and self.groq_key)]

        errors: list[str] = []
        for provider in order:
            generator = self._generate_gemini if provider == "gemini" else self._generate_groq

            for attempt in range(retries):
                try:
                    text = generator(prompt, temperature, max_tokens)
                    self.last_provider = provider
                    return text
                except Exception as exc:  # noqa: BLE001
                    message = str(exc)
                    lowered = message.lower()
                    transient = any(
                        marker in lowered
                        for marker in ("429", "rate", "quota", "timeout", "503",
                                       "overloaded", "unavailable", "500",
                                       # 504 DEADLINE_EXCEEDED shows up regularly
                                       # on the flash models and clears on retry;
                                       # without these markers it was treated as a
                                       # hard failure and skipped straight to the
                                       # fallback provider.
                                       "504", "deadline", "empty response")
                    )
                    if transient and attempt < retries - 1:
                        delay = (2 ** attempt) + random.uniform(0, 1)
                        print(f"  [{provider}] {message[:90]} - retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                    errors.append(f"{provider}: {message[:200]}")
                    break  # move on to the fallback provider

        raise LLMUnavailable(
            "All LLM providers failed:\n  " + "\n  ".join(errors)
        )


_shared_client: LLMClient | None = None


def get_llm(**kwargs) -> LLMClient:
    """Return a process-wide shared client (avoids re-reading .env everywhere)."""
    global _shared_client
    if _shared_client is None or kwargs:
        _shared_client = LLMClient(**kwargs)
    return _shared_client


def list_gemini_models(client: LLMClient) -> list[str]:
    """Every model this key can call generateContent on."""
    from google import genai
    from google.genai import types

    raw = genai.Client(api_key=client.gemini_key,
                       http_options=types.HttpOptions(timeout=HTTP_TIMEOUT_MS))
    names = []
    for model in raw.models.list():
        if "generateContent" in (getattr(model, "supported_actions", None) or []):
            names.append(model.name)
    return names


if __name__ == "__main__":
    import time

    from src.utils import setup_console
    setup_console()

    llm = LLMClient()
    print(f"Provider status: {llm.status()}")

    if not llm.available:
        print("\n" + LLMClient.setup_hint())
        raise SystemExit(0)

    if "--list-models" in sys.argv:
        print("\nModels this key can call:")
        for name in list_gemini_models(llm):
            marker = "  <- in candidate list" if any(
                c in name for c in GEMINI_MODEL_CANDIDATES) else ""
            print(f"   {name}{marker}")
        print("\nNOTE: appearing here does not guarantee access - some listed "
              "models still return 404 on call.")
        raise SystemExit(0)

    print("\nSending a test prompt ...")
    started = time.time()
    reply = llm.generate("Reply with exactly: KG-RAG LLM connection OK", max_tokens=50)
    print(f"  provider used: {llm.last_provider}")
    print(f"  model:         {llm.gemini_model or llm.groq_model}")
    print(f"  latency:       {time.time() - started:.1f}s")
    print(f"  response:      {reply}")
