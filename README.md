# tldw

**T**oo **L**ong; **D**idn't **W**atch. A tiny PubSubHubbub subscriber for YouTube channels. When one of your followed creators publishes a new video, the YouTube hub calls a local callback and `tldw` prints a one-line summary to your terminal. That is all `tldw` does today. Whatever you want to do with the notification, you add later. :bell: :tv:

## Why :thinking:

YouTube still exposes a WebSub-style push endpoint for channel uploads. Polling the RSS feed works, but it is wasteful at idle and laggy when busy. Subscribing is one POST per channel, then notifications stream in. `tldw` does the subscribe, the verify handshake, and the renewal so you do not have to read the spec. :sparkles:

## Install :package:

```bash
git clone <repo-url> tldw
cd tldw
uv sync --all-extras --dev
```

`uv` reads `uv.lock` so the install is deterministic. Python 3.14 is required. :lock:

## Configure :gear:

`channels.json` in the project root is the default list:

```json
{
  "channel_ids": [
    "UC_ywfvIR2JrnMuZt33y7QYQ",
    "UCvJZEG5x-DVYZKTz--pS39w"
  ]
}
```

A channel id is the `UC...` value you find on the channel's YouTube page or in the URL of any of its videos. Two ways to override:

| Method | Behavior |
| --- | --- |
| Edit `channels.json` | Used when no env override is set |
| `TLDW_CHANNEL_IDS="UC_xxx,UC_zzz"` (CSV) | Replaces the file when non-empty |

Environment variables (all read from the `TLDW_` namespace):

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `TLDW_CALLBACK_URL` | Yes (for subscribes) | unset | Public URL the hub will POST notifications to |
| `TLDW_CHANNELS_FILE` | No | `channels.json` | Path to the JSON channel list |
| `TLDW_CHANNEL_IDS` | No | unset | CSV override for the channel list |
| `TLDW_HUB_SECRET` | No | unset | HMAC secret for signing deliveries. When set, every incoming notification must carry a matching `X-Hub-Signature: sha1=...` header |
| `TLDW_DISCORD_WEBHOOK_URL` | No | unset | Webhook URL for the transcript-to-Discord delivery. When unset, transcripts are still queued and downloaded but not sent |
| `TLDW_QUEUE_FILE` | No | `queue.sqlite3` | Path to the SQLite queue database. Persist this directory in container deployments |
| `TLDW_TRANSCRIPT_DIR` | No | `transcripts` | Directory where downloaded `.vtt` files are kept |
| `TLDW_TRANSCRIPT_LINES` | No | `10` | How many transcript lines to include in each Discord message |
| `TLDW_POLL_BASE_SECONDS` | No | `600` | First retry delay in seconds |
| `TLDW_POLL_CAP_SECONDS` | No | `3600` | Maximum retry delay in seconds |
| `TLDW_GIVEUP_SECONDS` | No | `172800` | Stop retrying a video after this many seconds |
| `TLDW_YTDLP_COOKIES_FILE` | No | unset | Optional path to a Netscape-format cookies file. Improves reliability when YouTube applies bot checks |
| `TLDW_TRANSCRIPT_LANGS` | No | `["en", "en-orig"]` | Language codes to request from yt-dlp. Use exact codes only; a regex like `en.*` triggers 429s |

The file is gitignored-by-convention. Do not commit it if you have private channels. Keep `channels.json` for the default list, or commit an example and let operators override with `TLDW_CHANNEL_IDS`. :file_folder:

### What happens when no webhook is configured

When `TLDW_DISCORD_WEBHOOK_URL` is unset the worker still enqueues videos and downloads their transcripts to `TLDW_TRANSCRIPT_DIR`. Only the Discord send is skipped. Nothing is lost: the queue rows stay in the database, so an operator can set the webhook URL later and the pending transcripts are delivered on the next drain.

### Persistent storage

`TLDW_QUEUE_FILE` (and its parent directory) and `TLDW_TRANSCRIPT_DIR` must live on persistent storage: a named volume in compose, a PersistentVolumeClaim in Kubernetes. An `emptyDir` or a container-local path loses the queue when the pod is rescheduled. Because the PubSubHubbub hub does not redeliver a notification after a 200 response, a lost queue means those videos are silently dropped.

## Run :rocket:

```bash
uv run tldw                 # starts the server on 0.0.0.0:8000
# or:
uv run python -m tldw
```

On startup `tldw` prints one log line per channel it subscribes to. On shutdown the lifespan cancels the renewal task and closes the shared HTTP client. :arrows_counterclockwise:

## Run with container :whale:

A multi-stage `Containerfile` and `compose.yml` ship with the repo. They pin the
same UBI 9 Python 3.14 digest used by `stocknews`, install the locked runtime
dependencies only, and run as a non-root user. The default `channels.json` is
baked into the image; override the channel list with `TLDW_CHANNEL_IDS` at
runtime. :package:

```bash
# Build and start the container in the background
make container-build
make container-up

# Tail logs
podman logs -f tldw

# Stop the container
make container-down
```

The container publishes port `8000`. `TLDW_CALLBACK_URL` is still required for
the app to issue subscriptions. Put it (and any other `TLDW_*` overrides) in a
local `.env` file and compose will load it for you. :lock:

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
make check         # ty + pyright + pytest with branch coverage (the CI gate)
make test          # pytest only
make typecheck     # ty + pyright
make install       # uv sync --locked --all-extras --dev
```

All test work happens through `make check`. The suite covers feed parsing, the renderer, the hub client, config resolution, the verify/notify handlers, the lifespan, and the renewal loop. Tests use `httpx2.MockTransport` for the outbound POST and `fastapi.testclient.TestClient` for the inbound HTTP surface, so they run with `--disable-socket` and never touch the network. :test_tube:

Project layout:

```
src/tldw/
  __init__.py       # re-exports tldw.cli.main
  __main__.py       # python -m tldw entry
  app.py            # FastAPI factory, GET verify, POST notify, lifespan, renewal loop
  cli.py            # CLI entry point that hands the app to uvicorn
  config.py         # Settings (pydantic-settings) + resolve_channel_ids()
  feed.py           # parse_atom(body) -> list[VideoEntry]
  hub.py            # subscribe(), build_subscribe_form(), topic_url()
  renderer.py       # format_video_line(entry) -> str

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

channels.json       # default channel list
```

## Out of scope :no_entry_sign:

What `tldw` does not do (yet):

- Persist lease state across restarts (in-memory only).
- Act on notifications beyond printing them.
- A polling fallback for missed pushes. The Google hub can occasionally drop deliveries; if that bites you, poll the RSS feed in a separate process.
- Production hardening: no metrics, no health endpoint, no secrets manager, no deploy story.
- Docker image, systemd unit, or any other packaging. Run `uv run tldw` under your favorite supervisor.

Add these when the missing pieces start to matter, not before. :seedling: