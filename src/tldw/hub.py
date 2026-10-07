"""Client helpers for the PubSubHubbub subscribe handshake.

YouTube pushes new-video notifications through a PubSubHubbub (WebSub) hub. To
start receiving them a subscriber POSTs a form to the hub with the channel's
topic URL, a callback URL the hub can reach, and the verification and lease
settings. This module builds that form and performs the POST. It keeps the hub
URL, the topic shape, and the default lease in one place so callers never have
to remember them.
"""

from __future__ import annotations

import httpx2

# The public hub YouTube documents for subscriptions. Kept as a module constant
# so tests and callers can point at a fake hub without editing this module.
HUB_URL = "https://pubsubhubbub.appspot.com/subscribe"

# The hub verifies asynchronously by default: it calls the callback with a
# challenge instead of answering the subscribe POST inline.
DEFAULT_VERIFY_MODE = "async"

# The hub grants roughly five days in practice. The subscriber must renew before
# this expires to avoid a gap in notifications.
DEFAULT_LEASE_SECONDS = 432000


def topic_url(channel_id: str) -> str:
    """Return the PubSubHubbub topic URL for a YouTube channel.

    The URL contains the /xml/ segment. The human-facing feed URL omits it, but
    the hub rejects subscriptions whose topic is missing /xml/, so this segment
    is load-bearing. The channel_id is inserted verbatim because YouTube channel
    ids are URL-safe.
    """
    return (
        "https://www.youtube.com/xml/feeds/videos.xml"
        f"?channel_id={channel_id}"
    )


def build_subscribe_form(
    channel_id: str,
    callback_url: str,
    *,
    mode: str = "subscribe",
    verify: str = DEFAULT_VERIFY_MODE,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    secret: str | None = None,
) -> dict[str, str]:
    """Build the form body for a hub subscribe or unsubscribe request.

    Values are returned raw and httpx percent-encodes them exactly once when the
    form is POSTed. Pre-encoding the topic here would make httpx encode the
    percent signs a second time and the hub would reject the topic.

    Fields:
        hub.mode: "subscribe" or "unsubscribe".
        hub.topic: the full topic URL from topic_url for the channel.
        hub.callback: the URL the hub calls to deliver notifications and to run
            the verification challenge.
        hub.verify: the verification style, "async" by default.
        hub.lease_seconds: the requested lease length as a string.
        hub.secret: an HMAC secret for signing deliveries. Included only when a
            secret is provided so the hub leaves the body unsigned otherwise.
    """
    form: dict[str, str] = {
        "hub.mode": mode,
        "hub.topic": topic_url(channel_id),
        "hub.callback": callback_url,
        "hub.verify": verify,
        "hub.lease_seconds": str(lease_seconds),
    }
    if secret is not None:
        form["hub.secret"] = secret
    return form


async def subscribe(
    client: httpx2.AsyncClient,
    channel_id: str,
    callback_url: str,
    *,
    hub_url: str = HUB_URL,
    verify: str = DEFAULT_VERIFY_MODE,
    lease_seconds: int = DEFAULT_LEASE_SECONDS,
    secret: str | None = None,
) -> int:
    """Send a subscribe request to the hub and return the status code.

    Builds the form, POSTs it to hub_url as application/x-www-form-urlencoded,
    and raises httpx2.HTTPStatusError when the hub answers with a non-2xx
    status. On success the response status code is returned so the caller can
    record which code the hub chose.
    """
    form = build_subscribe_form(
        channel_id,
        callback_url,
        verify=verify,
        lease_seconds=lease_seconds,
        secret=secret,
    )
    response = await client.post(hub_url, data=form)
    response.raise_for_status()
    return response.status_code
