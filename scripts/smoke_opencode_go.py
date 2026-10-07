#!/usr/bin/env python3
"""Smoke test for the opencode go API gateway.

The gateway speaks the Anthropic Messages API shape and requires a session
header (x-opencode-session) in addition to the usual bearer token. This
script sends one tiny request so we can confirm that auth headers, the
session header, and basic connectivity all work before we build the real
LLM module on top of them.

It uses only the standard library so it can run anywhere, including in
environments that do not have the project dependencies installed.

Environment variables:
    TLDW_OPENCODE_API_KEY   required API key
    TLDW_OPENCODE_BASE_URL  gateway base URL (default https://opencode.ai/zen/go)
    TLDW_OPENCODE_MODEL     model name (default qwen3.8-max)

Exit codes:
    0  request succeeded (HTTP 2xx)
    1  gateway returned an HTTP error
    2  TLDW_OPENCODE_API_KEY is not set
    3  network or timeout error
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://opencode.ai/zen/go"
DEFAULT_MODEL = "qwen3.8-max"
REQUEST_TIMEOUT_SECONDS = 30


def _session_id() -> str:
    """Build a stable, per-host session id required by the gateway."""
    try:
        host = socket.gethostname()
    except OSError:
        host = "unknown-host"
    digest = hashlib.sha256(b"smoke-" + host.encode("utf-8")).hexdigest()
    return f"ses_{digest[:32]}"


def main() -> int:
    """Send the probe request and return the process exit code."""
    api_key = os.environ.get("TLDW_OPENCODE_API_KEY")
    if not api_key:
        print("TLDW_OPENCODE_API_KEY is not set; export it and retry.")
        return 2

    base_url = os.environ.get("TLDW_OPENCODE_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("TLDW_OPENCODE_MODEL", DEFAULT_MODEL)
    session_id = _session_id()

    payload = json.dumps(
        {
            "model": model,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "ping"}],
        }
    ).encode("utf-8")

    # Both auth styles are sent on purpose: the bearer token is the belt and
    # the x-api-key header is the braces, matching what the Anthropic SDK
    # sends. Some gateways check only one of them.
    request = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "x-opencode-session": session_id,
            "User-Agent": "tldw-smoke/1.0",
            "anthropic-version": "2023-06-01",
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

    print(f"OK {status} {model} session={session_id} body={body[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
