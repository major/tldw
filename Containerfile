# Official Python 3.14 image on Docker Hub, the full Debian-based variant
# (not `python:3.14-slim`). The image is layered on top of
# `buildpack-deps:bookworm`, so it ships with pip, the Python development
# headers, and the toolchain (gcc, make, etc.) needed to compile the project's
# wheels. That makes it suitable for both the builder stage below and the
# runtime stage at the bottom of this file.
FROM python:3.14@sha256:c23ebccb22bca6335521be462d1d4a3449a623c5ca8ce513d163c79c636dae79 AS builder

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

# Same official image for the runtime stage. Using the full Debian variant
# keeps the system libraries yt-dlp reaches into (libsqlite3, libssl, etc.)
# present, and gives us apt-get for installing ffmpeg below.
FROM python:3.14@sha256:c23ebccb22bca6335521be462d1d4a3449a623c5ca8ce513d163c79c636dae79

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

# ffmpeg is required by yt-dlp for format postprocessing (merging separate
# video/audio streams, transcoding, extracting audio, etc.). Install it from
# the Debian Bookworm repository and clean up the apt cache so the extra
# metadata does not bloat the final image. --no-install-recommends keeps the
# install to ffmpeg and its hard dependencies only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ffmpeg -version | head -n 1

WORKDIR /opt/app-root/src

# The official Python image ships a non-root `python` user (uid:gid 1000:1000).
# Copy the build artefacts with that ownership so the runtime user can read and
# execute them without ever needing root.
COPY --from=builder --chown=1000:1000 /opt/app-root/src/.venv /opt/app-root/src/.venv
COPY --from=builder --chown=1000:1000 /opt/app-root/src/channels.json /opt/app-root/src/channels.json

# Deno JavaScript runtime for yt-dlp (see the builder stage comment). Placed in
# /usr/local/bin so it is on PATH for the non-root runtime user.
COPY --from=builder --chown=1000:1000 /usr/local/bin/deno /usr/local/bin/deno

ENV PATH="/opt/app-root/src/.venv/bin:${PATH}"

EXPOSE 8000

USER 1000:1000
ENTRYPOINT ["/opt/app-root/src/.venv/bin/tldw"]
