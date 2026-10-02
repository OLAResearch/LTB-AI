"""Thin provider wrappers for the translator probe.

Only two operations are needed: produce a translation, and judge one
verification rule. Both reduce to "given a system prompt and a user prompt,
return text", so each provider implements a single `complete()` method.

SDKs are imported lazily so that `--dry-run` (and `--help`) work with nothing
installed. Install what you actually use:

    pip install openai          # OpenAI models
    pip install google-genai    # Gemini models
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from typing import Optional

# Parameters some models reject outright (reasoning models in particular).
# On a 400 that names one of these, the call is retried without it.
DROPPABLE_PARAMS = ('temperature', 'response_format', 'response_mime_type',
                    'system_instruction', 'automatic_function_calling',
                    'top_p', 'max_tokens')

# Gemma served through the Gemini API does not accept a system instruction, so
# the system prompt is folded into the user turn for these models.
NO_SYSTEM_PREFIXES = ('gemma',)

# Substrings that mean "retrying will not help": bad key, no access, no quota.
FATAL_MARKERS = ('invalid_api_key', 'api key not valid', 'incorrect api key',
                 'permission_denied', 'unauthorized', 'insufficient_quota',
                 'billing', 'model_not_found', 'does not exist')


class ProviderError(RuntimeError):
    """Raised for errors that retrying cannot fix (auth, quota, unknown model)."""


class Backend:
    """Interface every provider wrapper implements."""

    name = 'base'

    def complete(self, model: str, system: str, user: str, *,
                 json_mode: bool = False, temperature: Optional[float] = None) -> str:
        raise NotImplementedError

    def list_models(self) -> set:
        """Model ids this account can call, for `--verify-models`."""
        raise NotImplementedError


class OpenAIBackend(Backend):
    """OpenAI models via the official `openai` SDK (chat completions)."""

    name = 'openai'
    env_var = 'OPENAI_API_KEY'

    def __init__(self, api_key: str, timeout: float = 120.0):
        try:
            from openai import OpenAI
        except ImportError as exc:                                 # pragma: no cover
            raise ProviderError('The `openai` package is required for OpenAI models: '
                                'pip install openai') from exc
        self._client = OpenAI(api_key=api_key, timeout=timeout)

    def complete(self, model, system, user, *, json_mode=False, temperature=None):
        kwargs = {
            'model': model,
            'messages': [{'role': 'system', 'content': system},
                         {'role': 'user', 'content': user}],
        }
        if json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        if temperature is not None:
            kwargs['temperature'] = temperature

        while True:
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as exc:
                dropped = _drop_unsupported(kwargs, exc)
                if dropped is None:
                    raise _classify(exc)
                continue
        return (response.choices[0].message.content or '').strip()

    def list_models(self) -> set:
        try:
            return {model.id for model in self._client.models.list()}
        except Exception as exc:
            raise _classify(exc)


class GoogleBackend(Backend):
    """Gemini models via the official `google-genai` SDK."""

    name = 'google'
    env_var = 'GOOGLE_API_KEY'

    def __init__(self, api_key: str, timeout: float = 120.0):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:                                 # pragma: no cover
            raise ProviderError('The `google-genai` package is required for Gemini '
                                'models: pip install google-genai (note: this is not '
                                'the older `google-generativeai` package)') from exc
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(timeout * 1000)))

        # This probe never passes tools, but the SDK turns automatic function
        # calling on by default and logs a warning on every direct
        # generate_content call. Declaring it off (with maximum_remote_calls
        # left unset) takes the SDK's disabled path, so no AFC machinery runs
        # and nothing is logged.
        afc_config = getattr(types, 'AutomaticFunctionCallingConfig', None)
        self._afc_off = afc_config(disable=True) if afc_config else None

    def complete(self, model, system, user, *, json_mode=False, temperature=None):
        # Gemma has no system-instruction slot: prepend it to the user turn.
        if model.lower().startswith(NO_SYSTEM_PREFIXES):
            user = f'{system}\n\n{user}'
            system = None

        settings = {'system_instruction': system, 'temperature': temperature,
                    'response_mime_type': 'application/json' if json_mode else None,
                    'automatic_function_calling': self._afc_off}
        while True:
            config = self._types.GenerateContentConfig(
                **{k: v for k, v in settings.items() if v is not None})
            try:
                response = self._client.models.generate_content(
                    model=model, contents=user, config=config)
                break
            except Exception as exc:
                if _drop_unsupported(settings, exc) is None:
                    raise _classify(exc)
        return (response.text or '').strip()

    def list_models(self) -> set:
        try:
            names = set()
            for model in self._client.models.list():
                name = getattr(model, 'name', '') or ''
                names.add(name.split('/')[-1] if name else '')
            return {name for name in names if name}
        except Exception as exc:
            raise _classify(exc)


class DryRunBackend(Backend):
    """Deterministic stand-in that makes no network calls.

    Lets the whole pipeline -- prompts, concurrency, resume, judging, aggregation
    -- be exercised without an API key and without spending anything.
    """

    name = 'dry-run'
    env_var = None

    def __init__(self, api_key=None, timeout: float = 0.0, latency: float = 0.0):
        self._latency = latency

    def complete(self, model, system, user, *, json_mode=False, temperature=None):
        if self._latency:
            time.sleep(self._latency)
        digest = hashlib.sha256(f'{model}|{user}'.encode('utf-8')).hexdigest()
        if json_mode:
            # Deterministic pseudo-verdict, ~2/3 pass, so summaries look plausible.
            verdict = 'fail' if int(digest[:2], 16) % 3 == 0 else 'pass'
            return json.dumps({'verdict': verdict, 'reason': f'dry-run verdict {digest[:8]}'})
        return f'[dry-run translation by {model} · {digest[:12]}]'

    def list_models(self) -> set:
        return set()


BACKENDS = {'openai': OpenAIBackend, 'google': GoogleBackend}


def build_backend(provider: str, *, dry_run: bool = False,
                  timeout: float = 120.0) -> Backend:
    """Instantiate a provider wrapper, reading its API key from the environment."""
    if dry_run:
        return DryRunBackend()
    if provider not in BACKENDS:
        raise ProviderError(f'Unknown provider {provider!r}; '
                            f'expected one of {sorted(BACKENDS)}')
    cls = BACKENDS[provider]
    api_key = os.environ.get(cls.env_var, '').strip()
    if not api_key:
        raise ProviderError(f'{cls.env_var} is not set, so {provider} models cannot '
                            f'run. Export it, or pass --dry-run to test the pipeline '
                            f'without any API calls.')
    return cls(api_key=api_key, timeout=timeout)


def call_with_retries(fn, *args, retries: int = 3, base_delay: float = 1.5, **kwargs):
    """Call `fn`, retrying transient failures with exponential backoff and jitter."""
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except ProviderError:
            raise                                  # auth/quota/model: retrying is futile
        except Exception as exc:                   # rate limits, timeouts, 5xx
            last = exc
            if attempt == retries:
                break
            time.sleep(base_delay * (2 ** attempt) * (0.7 + 0.6 * random.random()))
    raise RuntimeError(f'failed after {retries + 1} attempts: {last}') from last


def _drop_unsupported(kwargs: dict, exc: Exception):
    """If `exc` complains about a droppable parameter, remove it and report which."""
    message = str(exc).lower()
    if 'unsupported' not in message and 'not supported' not in message \
            and 'unrecognized' not in message and 'invalid' not in message:
        return None
    for param in DROPPABLE_PARAMS:
        if param in message and param in kwargs:
            kwargs.pop(param)
            return param
    return None


def _classify(exc: Exception) -> Exception:
    """Turn hopeless provider errors into ProviderError so retries stop early."""
    message = str(exc).lower()
    if any(marker in message for marker in FATAL_MARKERS):
        return ProviderError(str(exc))
    return exc
