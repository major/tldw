"""Command line entry point for the tldw PubSubHubbub subscriber.

The CLI's whole job is small: load settings from the environment, build the
FastAPI application, and hand it to uvicorn. Startup and shutdown behavior
lives in the app's lifespan, which subscribes every resolved channel and runs
the renewal loop, so the CLI only has to get uvicorn running.
"""

from __future__ import annotations

import logging

import uvicorn

from tldw import _version
from tldw.app import create_app
from tldw.config import Settings

__all__ = ["main"]

logger = logging.getLogger(__name__)

# Default port uvicorn listens on when nothing overrides it.
_DEFAULT_PORT = 8000


def main() -> None:
    """Load settings, build the app, and serve it with uvicorn.

    When TLDW_CALLBACK_URL is unset the app still starts, but no subscriptions
    are issued, so a warning names the variable an operator must set.
    """
    # Emit the build identity before anything else so a slow startup (such as
    # waiting on the hub) does not push the banner past other log lines.
    _version.log_banner(logger)
    settings = Settings()
    if settings.callback_url is None:
        logger.warning(
            "TLDW_CALLBACK_URL is not set; the app will start but no "
            "subscriptions will be issued"
        )
    app = create_app(settings)
    uvicorn.run(app, host="0.0.0.0", port=_DEFAULT_PORT)
