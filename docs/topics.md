# Topic configuration

Topics are opt-in. Without `TLDW_TOPICS_FILE`, the existing legacy configuration
and deployment defaults remain in effect. No `topics.yaml` is loaded
automatically. See [topics.example.yaml](../topics.example.yaml) for a version 1
document with the existing nine default sources and a disabled legal starter.
The [legal prompt](../prompts/legal.md) is not validated on real legal videos;
synthetic fixtures validate plumbing only. Evaluate it before using it.

## Configuration and behavior

The document has `version: 1` and a `topics` mapping with a required `default`
topic. Each topic has `enabled`, `channels` (objects with `channel_id` and
optional `name`), a `prompt` with exactly one of `builtin: trading` or `file`,
and `discord.webhook_env`, the name of an environment variable containing a
webhook. Never put webhook values in YAML. Enabled topics need sources and a
nonempty destination. Disabled topics may have no sources or secret, but their
prompt file must still be readable. YAML is strict: unknown fields, duplicate
keys, invalid IDs, and duplicate active sources are rejected. Prompt file paths
are relative to the topics file. Changes are loaded as a startup snapshot;
restart after editing the file, prompts, or environment.

Topic YAML supplies the active sources instead of `channels.yaml` or CSV. A
nonempty `TLDW_CHANNEL_IDS` conflicts with topics at runtime and must be
removed. In single-destination mode, that CSV overrides the configured channel
YAML file; compare against the effective source set, not necessarily the
repository's `channels.yaml`.
Channel names in YAML are metadata and do not change feed metadata. An active
source belongs to only one topic; there is no fanout. In YAML mode, notifications
from unknown or disabled sources are acknowledged and dropped, not assigned to
the default topic.

Enabled controls new subscriptions and routing. A disabled topic can still be
drainable if its destination exists. Queued work saves topic IDs, not a prompt
or destination snapshot: processing uses the current prompt and destination for
that ID. Keep IDs stable and do not rename or reuse them while any backlog
references them. Every topic uses the same structured output and audio
transcription pipeline. The API key is required for transcription and the
worker exits without it. An analysis failure after transcription can fall back
to a plain digest; an empty transcript also produces a plain notification.

## Preflight and migration

1. First deploy the topic-aware release in legacy mode and confirm it works.
   Preserve the existing persistent queue and transcript directories, hub
   subscriptions, callback URL, and worker pacing. Run only one instance.
2. Prepare a copy of the example matching the **effective** legacy source set,
   including any CSV override. Set its default prompt to the built-in trading
   prompt and its destination reference to `TLDW_DISCORD_WEBHOOK_URL`; leave
   other topics disabled. Set the default webhook and the API key as before.
   The comparison checks the exact source set, prompt, and destination
   reference. Enabled topic destinations are required for validation.
3. Run the following offline preflight with the same environment as the running
   service, including its real legacy CSV if present. `Settings` does not load
   `.env` itself: export the same values or use `uv run --env-file .env`.
   The validator does not contact the network or start the service.

   ```bash
   TLDW_TOPICS_FILE=topics.example.yaml uv run --env-file .env tldw-validate-config --compare-legacy
   ```

   The comparison temporarily permits CSV and YAML together. Remove
   `TLDW_CHANNEL_IDS` from the runtime environment before starting with YAML.
   Leave `TLDW_TOPICS_FILE` unset, not blank, for legacy mode.
4. Gracefully stop the single instance. Back up the **actual persisted queue**
   consistently using the SQLite backup API, or stop and checkpoint it before
   making a backup. Do not copy a live database without its WAL. Keep the
   backup safe along with the persistent transcript data. Inspect pending and
   terminal jobs and unsent terminal alerts, and record counts for comparison.
5. Validate the stable, checkpointed backup, not the live database. The queue
   validator opens it immutable and read-only and refuses **any** WAL file.
   With the same exported environment and no runtime CSV, run:

   ```bash
   TLDW_TOPICS_FILE=topics.example.yaml uv run --env-file .env tldw-validate-config --queue-file /path/to/checkpointed-backup.sqlite3
   ```

   Remove the CSV from `.env` before this runtime-style validation. Check the
   reported backlog references and retain the backup. This check covers pending
   jobs and unsent alerts, not all terminal records; compare the recorded queue
   counts separately.
6. Start the topic-aware release with only the default topic enabled. Confirm
   the same persistent queue opened, compare pending and terminal jobs and
   unsent terminal alerts, and verify summaries and alerts reach the expected
   destination. Restart and inspect logs if something is wrong. No zero-downtime
   or exactly-once delivery is promised. The existing 48-hour wall-clock expiry
   continues to run while work is held, so long pauses can expire jobs.

Rollback by unsetting the topics file is suitable only while work and unsent
alerts are default-only. If any nondefault work is pending or has unsent alerts,
never use an older binary: retain the topic-aware release and definitions and
destinations until the backlog drains. The current binary blocks unknown routes.
Restoring a backup loses jobs received after that backup and is not a casual
rollback procedure.

To add legal later, add valid legal sources and its webhook secret, enable the
topic, validate, and restart. No code change is needed. Validate against a
checkpointed backup again if backlog exists. Keep existing topic IDs and
destinations available for their backlog.

## Containers

The image includes the example and prompt, but does not activate them. You may
explicitly set `TLDW_TOPICS_FILE=/opt/app-root/src/topics.example.yaml` to use
the baked example; external mounts are recommended for editable configuration.
For the optional compose overlay, run from the repository root:

```bash
podman compose -f compose.yml -f compose.topics.yml up -d
```

The overlay mounts the example and prompts read-only and fixes the container
path at `/config/topics.yaml`. Do not set a local host path as the container
variable. The base compose file still loads webhook variables from `.env`.
Remove `TLDW_CHANNEL_IDS` before startup, including from `.env`. Use both
`-f` options for other compose operations too. Merging these files into the
repository does not enable the overlay.

## Kubernetes

The patch is a strategic merge patch for a Deployment named `tldw`; it does not
change the base manifest. Keep one replica and the `Recreate` strategy. Create
the ConfigMap from the files after adapting them to your environment:

```bash
kubectl create configmap tldw-topics --from-file=topics.yaml=topics.example.yaml --from-file=legal.md=prompts/legal.md
kubectl create secret generic tldw-discord --from-file=default-webhook=/secure/trading-webhook
```

Use a protected file to avoid placing the secret value in command arguments or
shell history. The optional `legal-webhook` key is not needed while legal is
disabled; provision it securely before enabling legal. The patch does not
configure the runtime API key: provide it through your existing Secret-backed
deployment configuration. Remove any runtime `TLDW_CHANNEL_IDS` override.

Apply the patch **locally to your rendered deployment**, not the raw template
with placeholders. Inspect the result, including the existing API key Secret,
storage, one replica, and `Recreate`, before applying during the migration:

```bash
kubectl patch --local --type=strategic -f rendered-deployment.yaml --patch-file k8s/topics-patch.yaml -o yaml > deployment-with-topics.yaml
kubectl apply -f deployment-with-topics.yaml
```

The ConfigMap mounts the prompt at `prompts/legal.md` relative to the mounted
topics file. Restart after updating the ConfigMap or Secret so the new snapshot
is loaded. Creating or merging the patch alone does not enable topics.
