#!/usr/bin/env python3
"""Smoke test for the OpenAI chat completions API.

Sends one tiny request so we can confirm that the API key, base URL, and
basic connectivity all work before we build the real LLM module on top of
them. Uses only the standard library so it can run anywhere, including in
environments that do not have the project dependencies installed.

Environment variables:
    TLDW_OPENAI_API_KEY   required API key
    TLDW_OPENAI_BASE_URL  base URL (default https://api.openai.com/v1)
    TLDW_OPENAI_MODEL     model name (default gpt-6.1-sol)

Exit codes:
    0  request succeeded (HTTP 2xx)
    1  API returned an HTTP error
    2  TLDW_OPENAI_API_KEY is not set
    3  network or timeout error
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6.1-sol"
REQUEST_TIMEOUT_SECONDS = 30


def main() -> int:
    """Send the probe request and return the process exit code."""
    api_key = os.environ.get("TLDW_OPENAI_API_KEY")
    if not api_key:
        print("TLDW_OPENAI_API_KEY is not set; export it and retry.")
        return 2

    base_url = os.environ.get("TLDW_OPENAI_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("TLDW_OPENAI_MODEL", DEFAULT_MODEL)

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "tldw-smoke/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL {exc.code} {exc.reason}")
        print(body[:500])
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}")
        return 3

    print(f"OK {status} {model} body={body[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
