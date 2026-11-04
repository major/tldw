# tldw

**T**oo **L**ong; **D**idn't **W**atch. A PubSubHubbub subscriber for YouTube channels that queues new videos, transcribes audio, and delivers summaries to Discord. :bell: :tv:

## Why :thinking:

YouTube still exposes a WebSub-style push endpoint for channel uploads. Polling the RSS feed works, but it is wasteful at idle and laggy when busy. Subscribing is one POST per channel, then notifications stream in. `tldw` does the subscribe, the verify handshake, and the renewal so you do not have to read the spec. :sparkles:

## Install :package:

```bash
git clone <repo-url> tldw
cd tldw
uv sync --all-extras --dev
```

`uv` reads `uv.lock` so the install is deterministic. Python 3.15 is required. :lock:

## Configure :gear:

`channels.yaml` in the project root is the default list:

```yaml
channels:
  - channel_id: UC_ywfvIR2JrnMuZt33y7QYQ
    name: Example channel
  - channel_id: UCvJZEG5x-DVYZKTz--pS39w
```

Each entry requires a string `channel_id`; `name` is optional metadata.

A channel id is the `UC...` value you find on the channel's YouTube page or in the URL of any of its videos. Two ways to override:

| Method | Behavior |
| --- | --- |
| Edit `channels.yaml` | Used when no env override is set |
| `TLDW_CHANNEL_IDS="UC_xxx,UC_zzz"` (CSV) | Replaces the file when non-empty |

Environment variables (all read from the `TLDW_` namespace):

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `TLDW_CALLBACK_URL` | Yes (for subscribes) | unset | Public URL the hub will POST notifications to |
| `TLDW_CHANNELS_FILE` | No | `channels.yaml` | Path to the YAML channel list |
| `TLDW_CHANNEL_IDS` | No | unset | CSV override for the channel list |
| `TLDW_TOPICS_FILE` | No | unset | Opt in to topics YAML; conflicts with a nonempty channel CSV |
| `TLDW_HUB_SECRET` | No | unset | HMAC secret for signing deliveries. When set, every incoming notification must carry a matching `X-Hub-Signature: sha1=...` header |
| `TLDW_DISCORD_WEBHOOK_URL` | No | unset | Legacy Discord destination and default topic destination in the example; without a usable destination the worker does not run |
| `TLDW_DISCORD_ALERT_USER_ID` | No | unset | Discord user snowflake (17 to 20 digits) to mention for video pipeline failures |
| `TLDW_QUEUE_FILE` | No | `queue.sqlite3` | Path to the SQLite queue database. Persist this directory in container deployments |
| `TLDW_TRANSCRIPT_DIR` | No | `transcripts` | Directory where the audio pipeline caches its `.txt` transcripts |
| `TLDW_TRANSCRIPT_LINES` | No | `10` | How many transcript lines to include in each Discord message |
| `TLDW_POLL_BASE_SECONDS` | No | `600` | First retry delay in seconds |
| `TLDW_POLL_CAP_SECONDS` | No | `3600` | Maximum retry delay in seconds |
| `TLDW_GIVEUP_SECONDS` | No | `172800` | Stop retrying a video after this many seconds |
| `TLDW_YTDLP_COOKIES_FILE` | No | unset | Optional path to a Netscape-format cookies file. Improves reliability when YouTube applies bot checks |
| `TLDW_OPENAI_API_KEY` | Yes (for worker) | unset | API key required for audio transcription; the worker exits without it |
| `TLDW_OPENAI_BASE_URL` | No | `https://api.openai.com/v1` | Base URL for the OpenAI-compatible endpoint |
| `TLDW_OPENAI_MODEL` | No | `gpt-6.1-sol` | Model name to request from the endpoint |
| `TLDW_LLM_TIMEOUT_SECONDS` | No | `180.0` | Per-call timeout for a takeaway request |
| `TLDW_LLM_MAX_OUTPUT_TOKENS` | No | `2048` | Maximum tokens the takeaway model may generate |
| `TLDW_LLM_MAX_INPUT_CHARS` | No | `300000` | Hard cap on transcript characters sent to the model. Longer transcripts are truncated with a warning |
| `TLDW_TAKEAWAY_MAX_BULLETS` | No | `5` | Maximum bullets kept per takeaway |
| `TLDW_AUDIO_DOWNLOAD_DELAY_SECONDS` | No | `300` | Delay before the first audio download. Debounces notifications and gives YouTube's pipeline time to finish producing the video |
| `TLDW_AUDIO_DIR` | No | `audio` | Directory for raw audio downloads and compressed artifacts. May be ephemeral: the worker re-downloads on crash before the transcript is cached |
| `TLDW_AUDIO_FORMAT` | No | `webm` | Output container for ffmpeg. Must be in OpenAI's accepted set: `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, or `webm`. Use `webm` for the smallest files (Opus codec) |
| `TLDW_AUDIO_BITRATE` | No | `32k` | Target bitrate for ffmpeg Opus encoding. `32k` is enough for speech and keeps a 1-hour video under 15 MB, well under OpenAI's 25 MB upload limit |
| `TLDW_FFMPEG_TIMEOUT_SECONDS` | No | `900` | Per-call ffmpeg timeout, in seconds |
| `TLDW_TRANSCRIBE_MODEL` | No | `gpt-transcribe` | OpenAI speech-to-text model. `gpt-transcribe` is the current recommended model; the `gpt-4o-transcribe` family is deprecated and shuts down 2027-02-26 |
| `TLDW_TRANSCRIBE_LANGS` | No | `["en"]` | JSON list of ISO-639-1 language hints to pass to the transcription API |
| `TLDW_TRANSCRIBE_TIMEOUT_SECONDS` | No | `600` | Per-call transcription timeout, in seconds |
| `TLDW_INCLUDE_SHORTS` | No | `false` | When `true`, YouTube Shorts are delivered alongside full-length videos. Shorts URLs (`/shorts/<id>`) are filtered out by default so the digest focuses on long-form uploads |
| `TLDW_INCLUDE_LIVE` | No | `false` | By default, upcoming, current, recently ended, and replayed live events are skipped using metadata before audio download, not filtered from the feed. Set `true` to restore the original behavior for new pending videos. Terminal records are not revived, and older cached or overdue records are not retroactively filtered |

For private channels, use a separate file via `TLDW_CHANNELS_FILE` or use the `TLDW_CHANNEL_IDS` CSV override. Do not commit private channel lists. :file_folder:

**Migration:** `TLDW_CHANNELS_FILE` now expects YAML with `channels` entries as shown above, rather than the old `channel_ids` list. Convert custom files to the new schema. The CSV override still works. Rebuild container images to include the updated default list.

For opt-in topic routing and a safe migration, see [Topics YAML](docs/topics.md).

### What happens when no webhook is configured

In legacy mode, when `TLDW_DISCORD_WEBHOOK_URL` is unset, the lifespan does not start the transcript worker. Videos are still enqueued, but nothing is downloaded or sent. Configure the destination and restart to drain the persistent queue. In topic mode, enabled topics require destinations at startup. If no configured topic has a usable destination, the worker is skipped and existing queued work remains stored. When all topics are disabled, new notifications are dropped rather than queued.

Set `TLDW_DISCORD_ALERT_USER_ID` to opt in to operational failure alerts in the same channel as summaries, using the same webhook. The SQLite queue persists the first failure per video and stage, including across restarts; existing terminal videos are not backfilled. Deduplication is not global, so a shared quota incident can alert for each affected video. Queued alerts show the failure time and the video's current status. Delivery is at least once, so a crash after Discord accepts an alert can rarely cause a duplicate. A Discord outage cannot be reported to Discord until it recovers. Startup configuration failures, including a missing API key, are not monitored by these alerts. Unset the ID to disable alert creation and delivery without disabling summaries.

### Persistent storage

`TLDW_QUEUE_FILE` (and its parent directory) and `TLDW_TRANSCRIPT_DIR` must live on persistent storage: a named volume in compose, a PersistentVolumeClaim in Kubernetes. An `emptyDir` or a container-local path loses the queue when the pod is rescheduled. Because the PubSubHubbub hub does not redeliver a notification after a 200 response, a lost queue means those videos are silently dropped.

### Cookies for bot-checked egress IPs :cookie:

If `yt-dlp` logs `ERROR: Did not get any data blocks` over and over on a single video, or the worker output shows `Sign in to confirm you're not a bot`, YouTube has most likely flagged your cluster's egress IP. Recent yt-dlp releases (we pin `>=2026.8.19,<2027`) and the audio-format chain in `build_audio_ydl_opts` make this less frequent, but they do not eliminate it.

The fix is to authenticate the request with cookies from a browser session that is already logged into YouTube. Export them in Netscape format with an extension such as "Get cookies.txt LOCALLY" (Firefox) or "cookies.txt" (Chrome), then save the file somewhere the worker can read, for example `/data/youtube-cookies.txt`, and point `TLDW_YTDLP_COOKIES_FILE` at it.

The shipped `compose.yml` already mounts the `tldw-data` volume at `/data`. Drop the cookies file at `/data/youtube-cookies.txt` and set `TLDW_YTDLP_COOKIES_FILE=/data/youtube-cookies.txt` in the environment, and the worker picks it up on the next start.

In Kubernetes, mount the file from a `Secret` (or a `ConfigMap` if you do not mind the file being readable in etcd):

```yaml
volumes:
  - name: youtube-cookies
    secret:
      secretName: youtube-cookies
containers:
  - name: tldw
    env:
      - name: TLDW_YTDLP_COOKIES_FILE
        value: /etc/secrets/youtube-cookies/cookies.txt
    volumeMounts:
      - name: youtube-cookies
        mountPath: /etc/secrets/youtube-cookies
        readOnly: true
```

YouTube session cookies expire, typically after a few weeks of inactivity. When they go stale the original symptom comes back, so refresh the file from the browser and restart the pod.

### LLM video takeaways

The API key is required for audio transcription; the worker exits without it. After transcription, the worker analyzes the transcript and posts a briefing header plus three takeaway embeds instead of the plain digest. Each takeaway has a short title, a summary, and bullet points that may link back to supported timestamps in the video.

If the model call fails, times out, or returns an invalid shape, the worker logs a warning and sends the plain digest instead. A bad LLM call never costs a retry against YouTube: it is not a download, so it does not consume the request budget. :robot:

#### Output token diagnostics :mag:

Each returned model response emits a structured `event=llm_response` log before output validation, including responses that trigger retries or fail the analysis. Fields include `video_id`, `model`, `attempt` (starting at 1 per analysis), `input_tokens`, `output_tokens`, `reasoning_tokens`, `max_output_tokens`, `output_budget_fraction`, `finish_reason`, and `provider_finish_reason`. Reasoning tokens are included in output tokens, not added to them; unavailable reasoning counts and finish reasons are `null`. Prompts, transcript text, output content, and unrelated provider metadata are not included.

- **INFO, `LLM response`:** normal output usage.
- **WARNING, `LLM output token usage is near the limit`:** output usage is at least 90% of the configured budget, without an explicit truncation reason. Even 100% usage alone does not prove truncation.
- **WARNING, `LLM output token limit reached`:** the normalized finish reason is `length` or the provider reason is `max_output_tokens`. This is an explicit signal that the output budget was exhausted, even if validation later fails.

Use these warnings to decide whether to raise `TLDW_LLM_MAX_OUTPUT_TOKENS`. The diagnostics do not change the budget, retries, or fallback behavior. Requests that raise before returning a model response still use the existing failure log and have no response-usage record. SDK-internal transport retries are not counted as separate attempts. For a nonpositive configured budget, `output_budget_fraction` is `null` and usage-based warnings are disabled.

## Run :rocket:

```bash
uv run tldw                 # starts the server on 0.0.0.0:8000
# or:
uv run python -m tldw
```

On startup `tldw` prints one log line per channel it subscribes to. On shutdown the lifespan cancels the renewal task and closes the shared HTTP client. :arrows_counterclockwise:

### Human-readable logs :mag:

Application and Uvicorn logs use one JSON object per line on stdout, with
`time` (UTC), `level`, `msg`, and `logger` fields. Log-call extras appear as
top-level fields, and tracebacks stay inside the JSON record. Use `hl -F`
to format the logs for reading:

```bash
uv run tldw | hl -F
tail -f tldw.log | hl -F
```

## Run with container :whale:

A multi-stage `Containerfile` and `compose.yml` ship with the repo. They pin
the official `python:3.15` (full Debian-based image, not the `-slim`
variant), install the locked runtime dependencies plus `ffmpeg` for
`yt-dlp` postprocessing, and run as a non-root user. The default
`channels.yaml` is baked into the image; override the channel list with
`TLDW_CHANNEL_IDS` at runtime. :package:

```bash
# Build and start the container in the background
make container-build
make container-up

# Tail human-readable logs
podman logs -f tldw | hl -F

# Stop the container
make container-down
```

The container publishes port `8000`. `TLDW_CALLBACK_URL` is still required for
the app to issue subscriptions. Put it (and any other `TLDW_*` overrides) in a
local `.env` file and compose will load it for you. :lock:

## Deployment

Persistent storage is mandatory. `TLDW_QUEUE_FILE` (and its parent directory) and `TLDW_TRANSCRIPT_DIR` must live on storage that survives a reschedule: a named volume in compose, a PersistentVolumeClaim in Kubernetes. An `emptyDir` or a container-local path loses the queue, and because the PubSubHubbub hub does not redeliver after a 200 response, that means those videos are silently dropped.

Keep the replica count at exactly one. The worker drains the queue serially, and that serial design is the rate-limit defense for YouTube: a second pod would double-download the same videos. SQLite over a network filesystem is also unsafe. Scale CPU and memory, not replicas.

Supply `TLDW_DISCORD_WEBHOOK_URL` from a Secret in production, not a plain environment variable, so the webhook URL is not exposed in the pod spec or container logs.

The deployment artifacts:

| File | Purpose |
| --- | --- |
| `.env.example` | Complete list of `TLDW_*` variables with comments. Copy to `.env`. |
| `compose.yml` | Single service plus the `tldw-data` named volume mounted at `/data`. |
| `k8s/deployment.yaml` | Deployment with the transcript env vars and the `/data` volume mount. |
| `k8s/pvc.yaml` | PersistentVolumeClaim named `tldw-data` for the queue and transcripts. |

## Tunnel for local development :globe_with_meridians:

The Google hub needs a public URL it can reach. Localhost is not enough. Pick one:

```bash
# Quick: free URL changes on every restart
ngrok http 8000
# Stable: free with a named tunnel, persistent hostname
cloudflared tunnel --url http://localhost:8000
```

Copy the public URL, including the path, and set:

```bash
export TLDW_CALLBACK_URL="https://your-tunnel.example/pubsub/callback"
uv run tldw
```

Restart `tldw` after the tunnel URL changes. The subscription is bound to the callback URL, so a new URL orphans the previous subscription and the renewal task will quietly subscribe again with the new one. :warning:

## Verify a subscription :mag:

YouTube's hub exposes a debug page that shows whether a subscription is verified and when its lease ends:

```
https://pubsubhubbub.appspot.com/subscription-details?hub.callback=<callback-url>&hub.topic=https://www.youtube.com/xml/feeds/videos.xml?channel_id=<channel-id>
```

URL-encode the values. A verified subscription shows `State: verified` and a `Lease seconds: ~432000` line. :white_check_mark:

## How it works :electric_plug:

```
+-------------+      POST /subscribe       +----------------------+
| tldw startup| -------------------------> | pubsubhubbub.appspot |
+-------------+                            +----------------------+
                                                |
                                                | async verification
                                                v
                                       +----------------------+
                                       | GET /pubsub/callback |
                                       | echo hub.challenge   |
                                       +----------------------+
                                                |
                                                | later, on new upload
                                                v
+-------------+   POST /pubsub/callback    +----------------------+
| tldw prints | <------------------------ | hub delivers Atom XML |
| one line    |    X-Hub-Signature: sha1= | (sometimes) |
+-------------+                            +----------------------+

The renewal loop re-POSTs /subscribe every lease * 0.8 seconds (default 4 days)
so the lease never lapses.
```

Five-second version:

1. Lifespan startup resolves the channel list and POSTs a `subscribe` form to `pubsubhubbub.appspot.com/subscribe` for each id.
2. The hub GETs `/pubsub/callback?hub.mode=subscribe&hub.challenge=...`. We echo the challenge as `text/plain`. If we return anything else (404, 410), the hub refuses the subscription.
3. When a creator publishes a video the hub POSTs the Atom feed to the same URL. If `TLDW_HUB_SECRET` is set we verify the `X-Hub-Signature` HMAC; on mismatch we return 403 and log a warning. Otherwise we parse the feed with stdlib `xml.etree.ElementTree` and `print()` one `[YouTube] <channel>: <title> (<url>)` line per entry.
4. Malformed XML is logged at WARNING and answered 200 so the hub does not retry forever.
5. A background renewal task re-subscribes every `lease_seconds * 0.8` seconds (about 4 days for the default lease) so notifications keep flowing. :repeat:

## Development :wrench:

```bash
make check         # ty + pytest with branch coverage (the CI gate)
make test          # pytest only
make typecheck     # ty
make install       # uv sync --locked --all-extras --dev
```

All test work happens through `make check`. The suite covers feed parsing, the renderer, the hub client, config resolution, the verify/notify handlers, the lifespan, and the renewal loop. Tests use `httpx2.MockTransport` for the outbound POST and `fastapi.testclient.TestClient` for the inbound HTTP surface, so they run with `--disable-socket` and never touch the network. :test_tube:

Project layout:

```
src/tldw/
  __init__.py       # re-exports tldw.cli.main
  __main__.py       # python -m tldw entry
  app.py            # FastAPI factory, GET verify, POST notify, lifespan, renewal loop
  audio.py          # yt-dlp audio-only download + ffmpeg compress
  cli.py            # CLI entry point that hands the app to uvicorn
  config.py         # Settings (pydantic-settings) + resolve_channel_ids()
  feed.py           # parse_atom(body) -> list[VideoEntry]
  hub.py            # subscribe(), build_subscribe_form(), topic_url()
  renderer.py       # format_video_line(entry) -> str
  transcribe.py     # OpenAI speech-to-text client

tests/
  conftest.py             # shared fixtures (captured Atom payload)
  test_app.py             # GET verify, POST notify + HMAC matrix, lifespan + renewal
  test_cli.py             # CLI + python -m tldw (well, lives in test_tldw.py)
  test_config.py          # Settings + resolve_channel_ids env/file/validation matrix
  test_feed.py            # parse_atom matrix
  test_harness.py         # guards the --allow-unix-socket pytest setting
  test_hub.py             # topic_url, build_subscribe_form, subscribe via MockTransport
  test_renderer.py        # format_video_line matrix
  test_tldw.py            # CLI + module re-exports

channels.yaml       # default channel list
```

## Out of scope :no_entry_sign:

What `tldw` does not do (yet):

- Persist lease state across restarts (in-memory only).
- A polling fallback for missed pushes. The Google hub can occasionally drop deliveries; if that bites you, poll the RSS feed in a separate process.
- Production hardening: no metrics, no health endpoint, no secrets manager, no deploy story.
- Docker image, systemd unit, or any other packaging. Run `uv run tldw` under your favorite supervisor.

Add these when the missing pieces start to matter, not before. :seedling:
