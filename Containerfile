FROM registry.access.redhat.com/hi/python:3.14-builder@sha256:a260cb9e1e713ff590b7340984ec1e8ea2cff0db4b453677e36c05a82f9fbe71 AS builder

ENV PATH="/tmp/.local/bin:${PATH}"

USER 0:0
WORKDIR /opt/app-root/src

COPY --chown=0:0 pyproject.toml uv.lock .python-version README.md ./
COPY --chown=0:0 src ./src
COPY --chown=0:0 channels.json ./

RUN python3 -m pip install --no-cache-dir uv==0.12.18 \
    && uv sync --locked --no-dev --no-editable --python python3.14

# yt-dlp needs a JavaScript runtime (Deno) to solve YouTube's player JS and
# extract formats. Without it the YouTube extractor logs:
#   "No supported JavaScript runtime could be found"
# We install the official static binary here in the builder stage and copy it
# into the runtime stage below. The builder image does not ship unzip, so the
# zip is extracted with `python3 -m zipfile` instead.
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
        -o /tmp/deno.zip \
    && python3 -m zipfile -e /tmp/deno.zip /usr/local/bin \
    && chmod +x /usr/local/bin/deno \
    && rm /tmp/deno.zip \
    && /usr/local/bin/deno --version

FROM registry.access.redhat.com/hi/python:3.14@sha256:9e5c94e0f676b2be9bf623fc1292b601358af588d4653eb547b4c7ae4295bae8

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_TIME}"

# Expose the build identity to the running process so the startup banner and
# the /version endpoint can report which commit and build is live. Operators
# pass these with --build-arg from their CI/CD system.
ENV TLDW_GIT_SHA=${GIT_SHA} \
    TLDW_BUILD_TIME=${BUILD_TIME}

# Persistent storage defaults. The tldw-data volume mounts at /data, so the
# queue database and the downloaded subtitle files live there by default.
# Operators override these only when they want a different layout.
ENV TLDW_QUEUE_FILE=/data/queue.sqlite3 \
    TLDW_TRANSCRIPT_DIR=/data/transcripts

WORKDIR /opt/app-root/src

COPY --from=builder --chown=65532:0 /opt/app-root/src/.venv /opt/app-root/src/.venv
COPY --from=builder --chown=65532:0 /opt/app-root/src/channels.json /opt/app-root/src/channels.json

# Deno JavaScript runtime for yt-dlp (see the builder stage comment). Placed in
# /usr/local/bin so it is on PATH for the non-root runtime user.
COPY --from=builder --chown=65532:0 /usr/local/bin/deno /usr/local/bin/deno

ENV PATH="/opt/app-root/src/.venv/bin:${PATH}"

EXPOSE 8000

USER 65532:0
ENTRYPOINT ["/opt/app-root/src/.venv/bin/tldw"]
